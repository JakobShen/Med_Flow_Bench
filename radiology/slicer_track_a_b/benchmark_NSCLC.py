#!/usr/bin/env python3
"""Benchmark: NSCLC Radiogenomics multimodal CT/PET QA.

Questions (fixed multiple choice; final answer must be compact like 1C2D3B4A5B):
  1) Tumor location
  2) Pathological T stage
  3) Pathological N stage
  4) Histology
  5) Histopathological Grade

Rule-based evaluation only (no LLM-as-judge).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import sys

SCRIPT_SUPPORT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_SUPPORT_ROOT))

from _script_support import (
    build_context_bundle,
    build_model_runtime,
    configure_logging,
    default_model_from_env,
    ensure_repo_imports,
    ensure_slicer_runtime,
    resolve_run_output_dir,
    resolve_bridge_dir,
)

ensure_repo_imports()

from medflow_prompts import prompt_metadata_bundle, render_prompt
from slicer_agent_engine.agents.auto_agent import run_auto_task
from slicer_agent_engine.app.bootstrap import restart_slicer_runtime
from slicer_agent_engine.benchmarking.check_segment import (
    SegmentGroundTruth,
    evaluate_segment_prediction,
    prepare_segment_ground_truth,
)
from slicer_agent_engine.benchmarking.judge import extract_numbered_answers
from slicer_agent_engine.benchmarking.paper_table import build_nsclc_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.toolpack_cli import (
    TOOLPACK_PROMPT_REF,
    add_toolpack_cli_arguments,
    resolve_toolpack_cli_selection,
)
from slicer_agent_engine.toolpacks import TOOL_PACK_ACTIVATION_MODE_AUTO, ToolPackSelection
from slicer_agent_engine.benchmarking.nsclc import build_task_prompt, iter_cases
from slicer_agent_engine.load_helper import format_catalog_for_prompt
from slicer_agent_engine.slicer_client import SlicerClient


DEFAULT_DATA_ROOT = "."
QUESTION_IDS = [1, 2, 3, 4, 5]
QUESTION_NAMES = {
    1: "Tumor location",
    2: "Pathological T stage",
    3: "Pathological N stage",
    4: "Histology",
    5: "Histopathological Grade",
}
EVIDENCE_VIEWS = ["axial", "sagittal", "coronal"]
SLICER_DISCONNECT_MARKERS = (
    "Connection reset by peer",
    "Connection refused",
    "Read timed out",
    "Failed to establish a new connection",
)
PROMPT_REF = "3d_slicer/track_a_b/nsclc.json"


def _split_numbered_answers(compact: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    s = str(compact or "").upper()
    i = 0
    while i + 1 < len(s):
        if s[i].isdigit() and s[i + 1].isalpha():
            out[int(s[i])] = s[i + 1]
            i += 2
            continue
        i += 1
    return out


def _compact_from_map(answer_map: Dict[int, Optional[str]]) -> Optional[str]:
    parts: List[str] = []
    for qid in QUESTION_IDS:
        a = answer_map.get(qid)
        if a is None:
            continue
        parts.append(f"{qid}{a}")
    return "".join(parts) if parts else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NSCLC Radiogenomics benchmark (CT/PET DICOM).")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Manifest root containing NSCLC_Radiogenomics and metadata.csv")
    p.add_argument("--metadata_csv", type=str, default="metadata.csv", help="TCIA metadata CSV path")
    p.add_argument("--labels_csv", type=str, default="NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv", help="Labels CSV path")
    p.add_argument("--provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), help="LLM provider name. Use local_vllm for a local OpenAI-compatible vLLM server; auto infers hosted providers from --model.")
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model name")
    p.add_argument("--limit", type=int, default=100, help="Run only the first N cases (default: 100, 0 = all)")
    p.add_argument("--case_ids", type=str, default="", help="Comma-separated case IDs to run, e.g. R01-035,R01-036. Applied before --limit.")
    p.add_argument("--no_skip_amc", action="store_true", help="Include AMC cases. By default AMC cases are skipped.")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_nsclc", help="Directory for logs and outputs")
    p.add_argument("--max_rounds", type=int, default=20, help="Max tool-calling rounds")
    p.add_argument("--max_tools", type=int, default=20, help="Max executed tool calls per case")
    p.add_argument("--case_timeout_sec", type=int, default=0, help="Fail and continue if a single case exceeds this timeout in seconds (0 = disabled)")
    p.add_argument("--restart_slicer_every", type=int, default=0, help="Restart Slicer before every N cases (0 = disabled; --vista defaults to every case)")
    p.add_argument("--enable_code_interpreter", action="store_true", default=os.environ.get("OPENAI_ENABLE_CODE_INTERPRETER", "").strip().lower() in {"1", "true", "yes", "on"}, help="Expose a provider-native code execution tool during the run when supported (OpenAI Code Interpreter, Gemini code execution, or Qwen code_interpreter).")
    p.add_argument("--reasoning_effort", type=str, choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None, help="Optional provider reasoning-effort hint.")
    p.add_argument("--base_url", type=str, default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"), help="Slicer WebServer base URL")
    p.add_argument("--slicer_executable", type=str, default=os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"), help="Path to Slicer executable")
    p.add_argument("--enable_gemini", action="store_true", help="Enable Gemini video tools (requires GEMINI_API_KEY)")
    p.add_argument("--vista", action="store_true", help="Enable NSCLC VISTA3D mode: provide and auto-activate MONAI Label / VISTA3D tools with a light prompt hint.")
    add_toolpack_cli_arguments(p)
    p.add_argument("--balance_diagnosis_subtypes", action="store_true", help="No-op for NSCLC; kept for CLI symmetry with UCSF-PDGM.")
    return p


def _apply_vista_mode(toolpack_selection: ToolPackSelection, *, enabled: bool) -> ToolPackSelection:
    if not enabled:
        return toolpack_selection
    provided = toolpack_selection.provided_tool_packs
    if provided is not None:
        provided = tuple(sorted({*provided, "monai_label"}))
    return ToolPackSelection(
        provided_tool_packs=provided,
        activation_mode=TOOL_PACK_ACTIVATION_MODE_AUTO,
        prompt_hints=(
            *toolpack_selection.prompt_hints,
            render_prompt(TOOLPACK_PROMPT_REF, "templates.benchmark_hints.vista3d"),
        ),
    )


def build_instructions(*, enable_code_execution_hint: bool = False, toolpack_selection: ToolPackSelection) -> str:
    case_series_variant = (
        "disabled"
        if not toolpack_selection.provides_pack("case_series")
        else ("manual" if toolpack_selection.requires_manual_activation("case_series") else "auto")
    )
    base = render_prompt(
        PROMPT_REF,
        "templates.system",
        case_series_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.case_series_system.{case_series_variant}",
        ),
        tool_guidance=toolpack_selection.pick_text(
            none_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.none"),
            manual_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.manual"),
            auto_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.auto"),
        ),
        code_execution_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.code_execution.{'enabled' if enable_code_execution_hint else 'disabled'}",
        ),
    )
    return toolpack_selection.append_prompt_hints(base)


def _collect_tool_usage(runlog_path: Optional[Path]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    if runlog_path and runlog_path.exists():
        with runlog_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("kind")) != "run_event":
                    continue
                if str(rec.get("event_type")) != "tool_call_start":
                    continue
                tool_name = str(rec.get("tool_name") or "").strip()
                if not tool_name:
                    continue
                counts[tool_name] = counts.get(tool_name, 0) + 1
    return {
        "n_tool_calls": int(sum(counts.values())),
        "n_unique_tools": int(len(counts)),
        "tools_called": sorted(counts.keys()),
        "tool_call_counts": {k: counts[k] for k in sorted(counts.keys())},
    }


class CaseTimeoutError(TimeoutError):
    pass


@contextmanager
def _case_timeout(seconds: int, *, label: str):
    seconds = max(0, int(seconds or 0))
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):  # type: ignore[unused-argument]
        raise CaseTimeoutError(f"{label} timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _merge_tool_counts(dest: Dict[str, int], src: Dict[str, int]) -> None:
    for name, count in src.items():
        dest[str(name)] = int(dest.get(str(name), 0)) + int(count)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _score_questions(
    *,
    gt_map: Dict[int, Optional[str]],
    pred_compact: Optional[str],
    q_scored: Dict[int, int],
    q_correct: Dict[int, int],
) -> Tuple[Dict[str, Any], int, int]:
    pred_map = _split_numbered_answers(pred_compact)
    per_question: Dict[str, Any] = {}
    n_questions_scored = 0
    n_questions_correct = 0
    for qid in QUESTION_IDS:
        gt_q = gt_map.get(qid)
        pred_q = pred_map.get(qid)
        scored_q = gt_q is not None
        correct_q = bool(scored_q and pred_q == gt_q)
        if scored_q:
            q_scored[qid] += 1
            n_questions_scored += 1
            if correct_q:
                q_correct[qid] += 1
                n_questions_correct += 1
        per_question[str(qid)] = {
            "name": QUESTION_NAMES[qid],
            "gt": gt_q,
            "pred": pred_q,
            "scored": scored_q,
            "correct": correct_q,
        }
    return per_question, n_questions_scored, n_questions_correct


def _safe_prepare_segment_gt(*, base_url: str, bridge_dir: Path, out_dir: Path, case_path: Path) -> Tuple[Optional[SegmentGroundTruth], Optional[str]]:
    try:
        hidden_bundle = build_context_bundle(
            base_url=base_url,
            out_dir=out_dir,
            bridge_dir=bridge_dir,
            enable_gemini_video=False,
        )
        gt = prepare_segment_ground_truth(hidden_bundle.ctx, case_path=case_path, out_dir=out_dir, clear_scene_after=True)
        return gt, None
    except Exception as e:
        return None, str(e)


def _is_slicer_disconnect_error(text: object) -> bool:
    value = str(text or "")
    return any(marker in value for marker in SLICER_DISCONNECT_MARKERS)


def _first_slicer_disconnect_from_runlog(runlog_path: Path) -> Optional[str]:
    if not runlog_path.exists():
        return None
    try:
        for line in runlog_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not _is_slicer_disconnect_error(line):
                continue
            try:
                payload = json.loads(line)
                tool = payload.get("tool_name") or payload.get("event_type") or "<unknown>"
                err = payload.get("error") or payload.get("result_summary") or line
                return f"{tool}: {err}"
            except Exception:
                return line
    except Exception as e:
        return f"failed to inspect runlog for Slicer disconnects: {e}"
    return None


def _restart_slicer_runtime_for_case(*, base_url: str, bridge_dir: Path, slicer_executable: Path, case_log: logging.Logger, reason: str) -> None:
    restart_slicer_runtime(
        base_url=base_url,
        bridge_dir=bridge_dir,
        slicer_executable=slicer_executable,
        logger=case_log,
        reason=reason,
        require_exec=True,
        require_slice_render=True,
    )


def _ensure_slicer_runtime_for_case(*, base_url: str, bridge_dir: Path, slicer_executable: Path, case_log: logging.Logger, force_restart: bool, reason: str) -> None:
    if force_restart:
        _restart_slicer_runtime_for_case(
            base_url=base_url,
            bridge_dir=bridge_dir,
            slicer_executable=slicer_executable,
            case_log=case_log,
            reason=reason,
        )
        return
    try:
        client = SlicerClient(base_url=base_url, timeout_s=5.0)
        client.get_system_version()
        client.assert_exec_enabled()
        client.assert_slice_widgets_available()
    except Exception as e:
        _restart_slicer_runtime_for_case(
            base_url=base_url,
            bridge_dir=bridge_dir,
            slicer_executable=slicer_executable,
            case_log=case_log,
            reason=f"health check failed: {e}",
        )


def _segment_gt_summary(gt: Optional[SegmentGroundTruth]) -> Optional[Dict[str, Any]]:
    if gt is None:
        return None
    return {
        "has_segment": bool(gt.has_segment),
        "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
        "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
        "segmentation_name": gt.segmentation_name,
        "segment_name": gt.segment_name,
        "source_volume_name": gt.source_volume_name,
        "source_modality": gt.source_modality,
        "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in EVIDENCE_VIEWS},
        "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
        "voxel_count": gt.voxel_count,
        "segment_total_voxel_count": gt.segment_total_voxel_count,
        "component_count": gt.component_count,
        "notes": list(gt.notes),
    }

def _emit_segment_debug(case_log: logging.Logger, *, case_id: str, segment_check: Dict[str, Any]) -> None:
    if not isinstance(segment_check, dict):
        return
    for line in segment_check.get("debug_lines") or []:
        msg = f"[segment-check][{case_id}] {line}"
        case_log.info(msg)


def _update_segment_metrics(
    *,
    gt: Optional[SegmentGroundTruth],
    segment_check: Dict[str, Any],
    key_slice_scored: Dict[str, int],
    key_slice_correct: Dict[str, int],
    rsa_metrics: Dict[str, int],
    provenance_metrics: Dict[str, int],
    evidence_case_metrics: Dict[str, int],
) -> None:
    if gt is None or not gt.has_segment:
        return
    evidence_case_metrics["n_cases_with_segment_gt"] += 1
    if bool(segment_check.get("geometry_exact_correct", False)):
        evidence_case_metrics["n_cases_with_geometry_exact_correct"] += 1
    if bool(segment_check.get("evidence_exact_correct", False)):
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] += 1

    key_slice_payload = segment_check.get("key_slice") or {}
    for view in EVIDENCE_VIEWS:
        key_slice_scored[view] += 1
        if bool((key_slice_payload.get(view) or {}).get("correct", False)):
            key_slice_correct[view] += 1

    rsa_metrics["n_scored"] += 1
    if bool((segment_check.get("rsa") or {}).get("correct", False)):
        rsa_metrics["n_correct"] += 1

    provenance_metrics["n_scored"] += 1
    if bool((segment_check.get("provenance") or {}).get("ok", False)):
        provenance_metrics["n_correct"] += 1


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    metadata_csv = Path(args.metadata_csv).expanduser()
    if not metadata_csv.is_absolute():
        metadata_csv = (data_root / metadata_csv).resolve()
    labels_csv = Path(args.labels_csv).expanduser()
    if not labels_csv.is_absolute():
        labels_csv = (data_root / labels_csv).resolve()

    log_dir = resolve_run_output_dir(args.log_dir, model=args.model)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark.log")
    log = logging.getLogger("benchmark_nsclc")

    bridge_dir = resolve_bridge_dir()
    slicer_log_dir = log_dir / "slicer_logs"
    slicer_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SLICER_LAUNCH_LOG_DIR", str(slicer_log_dir))
    os.environ["SLICER_EXECUTABLE"] = str(Path(args.slicer_executable).expanduser())
    ensure_slicer_runtime(
        base_url=args.base_url,
        bridge_dir=bridge_dir,
        slicer_executable=Path(args.slicer_executable).expanduser(),
        require_exec=True,
        require_slice_render=True,
    )

    model_runtime = build_model_runtime(
        provider=args.provider,
        model=args.model,
        enable_code_interpreter=bool(args.enable_code_interpreter),
    )
    resolved_provider = model_runtime.resolved_provider
    model_client = model_runtime.model_client
    extra_model_tools = model_runtime.extra_model_tools
    toolpack_selection = _apply_vista_mode(resolve_toolpack_cli_selection(args), enabled=bool(args.vista))
    instructions = build_instructions(enable_code_execution_hint=bool(extra_model_tools), toolpack_selection=toolpack_selection)

    requested_limit = args.limit if args.limit and args.limit > 0 else None
    skip_amc = not bool(args.no_skip_amc)
    cases = list(iter_cases(data_root=data_root, metadata_csv=metadata_csv, labels_csv=labels_csv, limit=None))
    skipped_amc = 0
    if skip_amc:
        before_count = len(cases)
        cases = [case for case in cases if not str(case.case_id).upper().startswith("AMC-")]
        skipped_amc = before_count - len(cases)
    requested_case_ids = [item.strip() for item in str(args.case_ids or "").split(",") if item.strip()]
    if requested_case_ids:
        requested_case_id_set = set(requested_case_ids)
        cases = [case for case in cases if str(case.case_id) in requested_case_id_set]
        missing_case_ids = [case_id for case_id in requested_case_ids if case_id not in {str(case.case_id) for case in cases}]
        if missing_case_ids:
            log.warning("Requested case IDs were not found after filtering: %s", ", ".join(missing_case_ids))
    if requested_limit is not None:
        cases = cases[:requested_limit]
    if skipped_amc:
        log.info("Skipping %d AMC cases by default. Use --no_skip_amc to include them.", skipped_amc)
    log.info("Found %d NSCLC cases under %s", len(cases), data_root)

    results: List[Dict[str, Any]] = []
    n_case_scored = 0
    n_case_exact_correct = 0
    n_case_exact_with_evidence_correct = 0
    case_with_evidence_partial_score_total = 0.0
    q_scored = {qid: 0 for qid in QUESTION_IDS}
    q_correct = {qid: 0 for qid in QUESTION_IDS}
    n_failed = 0
    aggregate_tool_call_counts: Dict[str, int] = {}
    total_tool_calls = 0

    key_slice_scored = {view: 0 for view in EVIDENCE_VIEWS}
    key_slice_correct = {view: 0 for view in EVIDENCE_VIEWS}
    rsa_metrics = {"n_scored": 0, "n_correct": 0}
    provenance_metrics = {"n_scored": 0, "n_correct": 0}
    evidence_case_metrics = {
        "n_cases_with_segment_gt": 0,
        "n_cases_with_geometry_exact_correct": 0,
        "n_cases_with_evidence_exact_correct": 0,
    }

    if args.balance_diagnosis_subtypes:
        log.info("Ignoring --balance_diagnosis_subtypes for NSCLC benchmark (not applicable).")

    slicer_executable_path = Path(args.slicer_executable).expanduser()
    restart_slicer_every = max(0, int(args.restart_slicer_every or 0))
    case_timeout_sec = max(0, int(args.case_timeout_sec or 0))
    if args.vista and restart_slicer_every == 0:
        restart_slicer_every = 1
    if restart_slicer_every:
        log.info("Restarting Slicer before every %d case(s).", restart_slicer_every)

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)

        session = None
        segment_gt = None
        segment_gt_error = None
        default_alias = "diagnostic_ct"
        preload_paths: List[Path] = []
        gt_map = {
            1: case.gt_location,
            2: case.gt_t_stage,
            3: case.gt_n_stage,
            4: case.gt_histology,
            5: case.gt_grade,
        }
        gt = _compact_from_map(gt_map)
        try:
            with _case_timeout(case_timeout_sec, label=f"Case {case.case_id}"):
                _ensure_slicer_runtime_for_case(
                    base_url=args.base_url,
                    bridge_dir=bridge_dir,
                    slicer_executable=slicer_executable_path,
                    case_log=case_log,
                    force_restart=bool(restart_slicer_every and (idx - 1) % restart_slicer_every == 0),
                    reason=f"restart_slicer_every={restart_slicer_every}",
                )

                segment_gt_dir = case_out / "segment_gt"
                segment_gt, segment_gt_error = _safe_prepare_segment_gt(
                    base_url=args.base_url,
                    bridge_dir=bridge_dir,
                    out_dir=segment_gt_dir,
                    case_path=case.case_root,
                )
                if segment_gt_error and _is_slicer_disconnect_error(segment_gt_error):
                    case_log.warning("Segment GT discovery hit a Slicer disconnect; restarting and retrying once: %s", segment_gt_error)
                    _restart_slicer_runtime_for_case(
                        base_url=args.base_url,
                        bridge_dir=bridge_dir,
                        slicer_executable=slicer_executable_path,
                        case_log=case_log,
                        reason="segment GT discovery disconnected",
                    )
                    segment_gt, segment_gt_error = _safe_prepare_segment_gt(
                        base_url=args.base_url,
                        bridge_dir=bridge_dir,
                        out_dir=segment_gt_dir,
                        case_path=case.case_root,
                    )
                if segment_gt_error:
                    case_log.warning("Segment GT discovery failed for %s: %s", case.case_id, segment_gt_error)

                context_bundle = build_context_bundle(
                    base_url=args.base_url,
                    out_dir=case_out,
                    bridge_dir=bridge_dir,
                    enable_gemini_video=bool(args.enable_gemini),
                )
                session = context_bundle.session
                session.log_run_event(
                    "case_script_start",
                    script="benchmark_NSCLC.py",
                    case_id=case.case_id,
                    model=args.model,
                    max_rounds=int(args.max_rounds),
                    max_tools=int(args.max_tools),
                    vista=bool(args.vista),
                    toolpacks=toolpack_selection.to_summary_dict(),
                )
                ctx = context_bundle.ctx

                catalog_summary = case.catalog.to_summary()
                (case_out / "catalog.json").write_text(json.dumps(catalog_summary, indent=2), encoding="utf-8")
                if segment_gt is not None and segment_gt.gt_json_path and segment_gt.gt_json_path.exists():
                    try:
                        payload = json.loads(segment_gt.gt_json_path.read_text(encoding="utf-8"))
                        (case_out / "segment_gt_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                default_alias = case.catalog.default_aliases[0] if case.catalog.default_aliases else "diagnostic_ct"
                preload_paths = case.catalog.all_series_paths(prefer_latest=True)
                preload_display_names = case.catalog.all_series_display_names(prefer_latest=True)
                active_prefer = list(case.catalog.alias_active_prefer(default_alias))
                if segment_gt is not None and segment_gt.has_segment and segment_gt.source_volume_name:
                    active_prefer = [str(segment_gt.source_volume_name), *active_prefer]
                open_case_kwargs = {
                    "preset": "ct_pet_chest",
                    "layout": "four_up",
                    "clear_scene_first": True,
                    "dicom_series_dirs": [str(p) for p in preload_paths],
                    "series_display_names": preload_display_names,
                    "active_prefer": active_prefer,
                    "window_preset": case.catalog.alias_window_preset(default_alias),
                    "baseline": True,
                    "size": 512,
                }

                evidence_source_note = ""
                if segment_gt is not None and segment_gt.has_segment:
                    source_name = str(segment_gt.source_volume_name or "").strip()
                    source_modality = str(segment_gt.source_modality or "").strip()
                    source_label = (source_name or "<unknown>") + (
                        f" ({source_modality})" if source_modality else ""
                    )
                    evidence_source_note = render_prompt(
                        PROMPT_REF,
                        "templates.fragments.case_brief_evidence_source",
                        source_label=source_label,
                    )
                case_series_variant = (
                    "disabled"
                    if not toolpack_selection.provides_pack("case_series")
                    else ("manual" if toolpack_selection.requires_manual_activation("case_series") else "auto")
                )
                case_brief = render_prompt(
                    PROMPT_REF,
                    "templates.case_brief",
                    case_id=case.case_id,
                    default_alias=default_alias,
                    evidence_source_note=evidence_source_note,
                    case_series_note=render_prompt(
                        PROMPT_REF,
                        f"templates.fragments.case_brief_series_note.{case_series_variant}",
                    ),
                    catalog=format_catalog_for_prompt(case.catalog),
                )

                user_msg = build_task_prompt(
                    case,
                    toolpack_selection=toolpack_selection,
                    segment_evidence=segment_gt.to_prompt_spec() if segment_gt is not None and segment_gt.has_segment else None,
                )

                agent_res = run_auto_task(
                    provider=resolved_provider,
                    model_client=model_client,
                    model=args.model,
                    ctx=ctx,
                    case_path=case.case_root,
                    instructions=instructions,
                    user_message=user_msg,
                    max_rounds=int(args.max_rounds),
                    max_tools=int(args.max_tools),
                    open_case_kwargs=open_case_kwargs,
                    case_brief_text=case_brief,
                    case_series_catalog=case.catalog,
                    extra_model_tools=extra_model_tools,
                    reasoning_effort=args.reasoning_effort,
                    prompt_metadata=prompt_metadata_bundle(PROMPT_REF, TOOLPACK_PROMPT_REF),
                    provided_tool_packs=toolpack_selection.provided_tool_packs,
                    tool_pack_activation_mode=toolpack_selection.activation_mode,
                    model_request_timeout_sec=120.0,
                    model_request_retry_on_timeout=1,
                )
            raw_text = agent_res.text
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")
            slicer_disconnect = _first_slicer_disconnect_from_runlog(session.runlog_path)
            if slicer_disconnect:
                raise RuntimeError(
                    "Slicer WebServer disconnected during case; marking case failed instead of scoring model text. "
                    f"First disconnect: {slicer_disconnect}"
                )

            pred = extract_numbered_answers(raw_text, question_ids=QUESTION_IDS)
            per_question, n_questions_scored, n_questions_correct = _score_questions(
                gt_map=gt_map,
                pred_compact=pred,
                q_scored=q_scored,
                q_correct=q_correct,
            )
            scored = n_questions_scored > 0
            correct = bool(scored and n_questions_correct == n_questions_scored)
            if scored:
                n_case_scored += 1
                n_case_exact_correct += int(correct)

            segment_check = evaluate_segment_prediction(prediction=raw_text, gt=segment_gt, trace_path=session.trace_path) if segment_gt is not None else {
                "available": False,
                "has_segment_gt": False,
                "segment_gt_error": segment_gt_error,
            }
            _emit_segment_debug(case_log, case_id=case.case_id, segment_check=segment_check)
            _update_segment_metrics(
                gt=segment_gt,
                segment_check=segment_check,
                key_slice_scored=key_slice_scored,
                key_slice_correct=key_slice_correct,
                rsa_metrics=rsa_metrics,
                provenance_metrics=provenance_metrics,
                evidence_case_metrics=evidence_case_metrics,
            )
            correct_with_evidence = bool(correct and (segment_check.get("evidence_exact_correct", False) if segment_gt is not None and segment_gt.has_segment else True))
            case_with_evidence_partial_score = (
                float(n_questions_correct) / float(len(QUESTION_IDS))
                if scored and segment_gt is not None and segment_gt.has_segment and bool(segment_check.get("evidence_exact_correct", False))
                else 0.0
            )
            if scored:
                n_case_exact_with_evidence_correct += int(correct_with_evidence)
                case_with_evidence_partial_score_total += case_with_evidence_partial_score

            tool_usage = _collect_tool_usage(session.runlog_path if session is not None else None)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "gt": gt,
                "pred": pred,
                "runlog_path": str(session.runlog_path) if session is not None else None,
                "trace_path": str(session.trace_path) if session is not None else None,
                "agent_output_path": str(case_out / "agent_output.txt"),
                "correct": correct,
                "correct_with_evidence": correct_with_evidence,
                "case_with_evidence_partial_score": case_with_evidence_partial_score,
                "scored": scored,
                "per_question": per_question,
                "n_questions_scored": n_questions_scored,
                "n_questions_correct": n_questions_correct,
                "has_segment_gt": bool(segment_gt is not None and segment_gt.has_segment),
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
                "seconds": round(time.time() - t0, 3),
                "default_alias": default_alias,
                "default_paths": [str(p) for p in preload_paths],
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script="benchmark_NSCLC.py",
                case_id=case.case_id,
                ok=True,
                prediction=pred,
                ground_truth=gt,
                correct=correct,
                correct_with_evidence=correct_with_evidence,
                case_with_evidence_partial_score=case_with_evidence_partial_score,
                scored=scored,
                has_segment_gt=bool(segment_gt is not None and segment_gt.has_segment),
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                elapsed_sec=rec["seconds"],
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.info(
                "Completed case %d/%d: %s | pred=%s | gt=%s | case_exact=%s | case_exact_with_evidence=%s | partial_with_evidence=%.3f | q_correct=%d/%d | tool_calls=%d",
                idx,
                len(cases),
                case.case_id,
                pred,
                gt,
                correct,
                correct_with_evidence,
                case_with_evidence_partial_score,
                n_questions_correct,
                n_questions_scored,
                tool_usage["n_tool_calls"],
            )
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | pred={pred} | gt={gt} | evidence_exact={segment_check.get('evidence_exact_correct')} | partial_with_evidence={case_with_evidence_partial_score:.3f} | tool_calls={tool_usage['n_tool_calls']}"
            )
        except Exception as e:
            n_failed += 1
            per_question, n_questions_scored, _ = _score_questions(
                gt_map=gt_map,
                pred_compact=None,
                q_scored=q_scored,
                q_correct=q_correct,
            )
            scored = bool(case.gt_answer)
            if scored:
                n_case_scored += 1
            segment_check = evaluate_segment_prediction(
                prediction="",
                gt=segment_gt,
                trace_path=(session.trace_path if session is not None else None),
            ) if segment_gt is not None else {
                "available": False,
                "has_segment_gt": False,
                "segment_gt_error": segment_gt_error,
            }
            _emit_segment_debug(case_log, case_id=case.case_id, segment_check=segment_check)
            _update_segment_metrics(
                gt=segment_gt,
                segment_check=segment_check,
                key_slice_scored=key_slice_scored,
                key_slice_correct=key_slice_correct,
                rsa_metrics=rsa_metrics,
                provenance_metrics=provenance_metrics,
                evidence_case_metrics=evidence_case_metrics,
            )
            correct_with_evidence = False
            case_with_evidence_partial_score = 0.0
            if scored:
                n_case_exact_with_evidence_correct += 0
                case_with_evidence_partial_score_total += 0.0

            tool_usage = _collect_tool_usage(session.runlog_path if session is not None else None)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "gt": gt,
                "pred": None,
                "runlog_path": str(session.runlog_path) if session is not None else None,
                "trace_path": str(session.trace_path) if session is not None else None,
                "correct": False,
                "correct_with_evidence": False,
                "case_with_evidence_partial_score": 0.0,
                "scored": scored,
                "per_question": per_question,
                "n_questions_scored": n_questions_scored,
                "n_questions_correct": 0,
                "has_segment_gt": bool(segment_gt is not None and segment_gt.has_segment),
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
                "error": str(e),
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            if session is not None:
                session.log_run_event(
                    "case_script_finished",
                    script="benchmark_NSCLC.py",
                    case_id=case.case_id,
                    ok=False,
                    error=str(e),
                    elapsed_sec=rec["seconds"],
                    case_with_evidence_partial_score=0.0,
                    scored=scored,
                    has_segment_gt=bool(segment_gt is not None and segment_gt.has_segment),
                    evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                    n_tool_calls=tool_usage["n_tool_calls"],
                    tools_called=tool_usage["tools_called"],
                    tool_call_counts=tool_usage["tool_call_counts"],
                )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(f"Completed case {idx}/{len(cases)}: {case.case_id} | pred=<error> | gt={gt} | tool_calls={tool_usage['n_tool_calls']}")

    question_metrics: Dict[str, Any] = {}
    n_questions_scored_total = sum(q_scored.values())
    n_questions_correct_total = sum(q_correct.values())
    question_accuracy = (n_questions_correct_total / n_questions_scored_total) if n_questions_scored_total else 0.0
    for qid in QUESTION_IDS:
        nq = q_scored[qid]
        cq = q_correct[qid]
        question_metrics[str(qid)] = {
            "name": QUESTION_NAMES[qid],
            "n_scored": nq,
            "n_correct": cq,
            "accuracy": (cq / nq) if nq else 0.0,
        }

    case_exact_accuracy = (n_case_exact_correct / n_case_scored) if n_case_scored else 0.0
    case_exact_with_evidence_accuracy = (n_case_exact_with_evidence_correct / n_case_scored) if n_case_scored else 0.0
    case_with_evidence_partial_accuracy = (case_with_evidence_partial_score_total / n_case_scored) if n_case_scored else 0.0

    average_tool_calls_per_case = (total_tool_calls / len(results)) if results else 0.0

    key_slice_metrics = {
        view: {
            "n_scored": key_slice_scored[view],
            "n_correct": key_slice_correct[view],
            "accuracy": (key_slice_correct[view] / key_slice_scored[view]) if key_slice_scored[view] else 0.0,
        }
        for view in EVIDENCE_VIEWS
    }
    n_key_slice_items_scored = sum(key_slice_scored.values())
    n_key_slice_items_correct = sum(key_slice_correct.values())
    avg_evidence_correct = (n_key_slice_items_correct / n_key_slice_items_scored) if n_key_slice_items_scored else 0.0
    localization_acc = (rsa_metrics["n_correct"] / rsa_metrics["n_scored"]) if rsa_metrics["n_scored"] else 0.0
    n_evidence_items_scored = sum(key_slice_scored.values()) + rsa_metrics["n_scored"]
    n_evidence_items_correct = sum(key_slice_correct.values()) + rsa_metrics["n_correct"]
    evidence_item_accuracy = (n_evidence_items_correct / n_evidence_items_scored) if n_evidence_items_scored else 0.0
    geometry_case_accuracy = (
        evidence_case_metrics["n_cases_with_geometry_exact_correct"] / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
    evidence_case_accuracy = (
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
    provenance_accuracy = (
        provenance_metrics["n_correct"] / provenance_metrics["n_scored"]
        if provenance_metrics["n_scored"]
        else 0.0
    )

    summary = {
        "model": args.model,
        "data_root": str(data_root),
        "metadata_csv": str(metadata_csv),
        "labels_csv": str(labels_csv),
        "skip_amc": skip_amc,
        "n_amc_skipped": skipped_amc,
        "case_ids": requested_case_ids,
        "max_rounds": int(args.max_rounds),
        "max_tools": int(args.max_tools),
        "restart_slicer_every": restart_slicer_every,
        "n_cases": len(cases),
        "n_case_scored": n_case_scored,
        "n_case_exact_correct": n_case_exact_correct,
        "case_exact_accuracy": case_exact_accuracy,
        "n_case_exact_with_evidence_correct": n_case_exact_with_evidence_correct,
        "case_exact_with_evidence_accuracy": case_exact_with_evidence_accuracy,
        "case_with_evidence_partial_accuracy": case_with_evidence_partial_accuracy,
        "n_questions_scored": n_questions_scored_total,
        "n_questions_correct": n_questions_correct_total,
        "question_accuracy": question_accuracy,
        "avg_evidence_correct": avg_evidence_correct,
        "Localization_acc": localization_acc,
        "question_metrics": question_metrics,
        "segment_metrics": {
            "n_cases_with_segment_gt": evidence_case_metrics["n_cases_with_segment_gt"],
            "n_cases_with_geometry_exact_correct": evidence_case_metrics["n_cases_with_geometry_exact_correct"],
            "geometry_case_accuracy": geometry_case_accuracy,
            "n_cases_with_evidence_exact_correct": evidence_case_metrics["n_cases_with_evidence_exact_correct"],
            "evidence_case_accuracy": evidence_case_accuracy,
            "n_evidence_items_scored": n_evidence_items_scored,
            "n_evidence_items_correct": n_evidence_items_correct,
            "evidence_item_accuracy": evidence_item_accuracy,
            "key_slice_metrics": key_slice_metrics,
            "rsa_metrics": {
                "n_scored": rsa_metrics["n_scored"],
                "n_correct": rsa_metrics["n_correct"],
                "accuracy": localization_acc,
            },
            "provenance_metrics": {
                "n_scored": provenance_metrics["n_scored"],
                "n_correct": provenance_metrics["n_correct"],
                "accuracy": provenance_accuracy,
            },
        },
        # Backward-compatible aliases (case-exact metric).
        "n_scored": n_case_scored,
        "n_correct": n_case_exact_correct,
        "accuracy": case_exact_accuracy,
        "n_failed": n_failed,
        "total_tool_calls": total_tool_calls,
        "average_tool_calls_per_case": average_tool_calls_per_case,
        "tool_call_counts": {k: aggregate_tool_call_counts[k] for k in sorted(aggregate_tool_call_counts.keys())},
        "tools_called": sorted(aggregate_tool_call_counts.keys()),
        "n_unique_tools_used": len(aggregate_tool_call_counts),
        "toolpacks": toolpack_selection.to_summary_dict(),
        "vista": bool(args.vista),
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_nsclc_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    log.info(
        "DONE. question_accuracy=%.4f (%d/%d) case_exact_accuracy=%.4f (%d/%d) case_exact_with_evidence_accuracy=%.4f (%d/%d) case_with_evidence_partial_accuracy=%.4f failed=%d total_tool_calls=%d avg_tool_calls_per_case=%.2f tools=%s",
        question_accuracy,
        n_questions_correct_total,
        n_questions_scored_total,
        case_exact_accuracy,
        n_case_exact_correct,
        n_case_scored,
        case_exact_with_evidence_accuracy,
        n_case_exact_with_evidence_correct,
        n_case_scored,
        case_with_evidence_partial_accuracy,
        n_failed,
        total_tool_calls,
        average_tool_calls_per_case,
        sorted(aggregate_tool_call_counts.keys()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
