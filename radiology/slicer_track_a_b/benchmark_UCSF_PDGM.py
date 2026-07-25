#!/usr/bin/env python3
"""Benchmark: UCSF-PDGM 3-way classification with optional segment-grounded evidence.

Task (3-way multiple choice):
  A) Glioblastoma
  B) Oligodendroglioma / Astrocytoma
  C) No tumor

Ground truth is read from UCSF-PDGM-metadata_v5.csv.

When a hidden NIfTI tumor segmentation is available for the case, the benchmark also
requires grounded evidence output:
  - KEY_SLICE_AXIAL_MM
  - KEY_SLICE_SAGITTAL_MM
  - KEY_SLICE_CORONAL_MM
  - RSA: [R, A, S]

Evaluation remains deterministic and rule-based (no LLM-as-judge).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.benchmarking.check_segment import (
    SegmentGroundTruth,
    evaluate_segment_prediction,
    prepare_nifti_segment_ground_truth,
)
from slicer_agent_engine.benchmarking.judge import extract_choice
from slicer_agent_engine.benchmarking.paper_table import build_ucsf_pdgm_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.toolpack_cli import (
    TOOLPACK_PROMPT_REF,
    add_toolpack_cli_arguments,
    resolve_toolpack_cli_selection,
)
from slicer_agent_engine.toolpacks import ToolPackSelection
from slicer_agent_engine.benchmarking.ucsf_pdgm import diagnosis_subtype, iter_cases, summarize_subtype_metrics


EVIDENCE_VIEWS = ["axial", "sagittal", "coronal"]


def _ucsf_pdgm_evidence_exact_correct(segment_check: Dict[str, Any]) -> bool:
    key_slice_payload = segment_check.get("key_slice") or {}
    return bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UCSF-PDGM benchmark (NIfTI).")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing UCSF-PDGM-*_nifti folders")
    p.add_argument("--csv_path", type=str, default="UCSF-PDGM-metadata_v5.csv", help="Metadata CSV path")
    p.add_argument("--provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), help="LLM provider name. Use local_vllm for a local OpenAI-compatible vLLM server; auto infers hosted providers from --model.")
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model name")
    p.add_argument("--limit", type=int, default=100, help="Run only the first N cases (default: 100, 0 = all)")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_ucsf_pdgm", help="Directory for logs and outputs")
    p.add_argument("--max_rounds", type=int, default=20, help="Max tool-calling rounds")
    p.add_argument("--max_tools", type=int, default=0, help="Max executed tool calls per case/question (0 = unlimited)")
    p.add_argument("--case_timeout_sec", type=int, default=0, help="Fail and continue if a single case exceeds this timeout in seconds (0 = disabled)")
    p.add_argument("--restart_slicer_every", type=int, default=0, help="Restart Slicer before every N cases/questions (0 = disabled)")
    p.add_argument("--enable_code_interpreter", action="store_true", default=os.environ.get("OPENAI_ENABLE_CODE_INTERPRETER", "").strip().lower() in {"1", "true", "yes", "on"}, help="Expose a provider-native code execution tool during the run when supported (OpenAI Code Interpreter, Gemini code execution, or Qwen code_interpreter).")
    p.add_argument("--reasoning_effort", type=str, choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None, help="Optional provider reasoning-effort hint.")
    p.add_argument("--base_url", type=str, default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"), help="Slicer WebServer base URL")
    p.add_argument(
        "--slicer_executable",
        type=str,
        default=os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"),
        help="Path to Slicer executable",
    )
    p.add_argument("--include_extra", action="store_true", help="Load all non-seg NIfTI volumes (not just T1/T1c/T2/FLAIR)")
    p.add_argument("--include_segmentations", action="store_true", help="Ignored by this benchmark to avoid leaking hidden GT segmentations into the agent scene.")
    p.add_argument("--no_prefer_bias", action="store_true", help="Do not prefer *_bias.nii.gz when selecting modalities")
    p.add_argument("--enable_gemini", action="store_true", help="Enable Gemini video analysis tools (requires GEMINI_API_KEY)")
    p.add_argument("--force_gemini", action="store_true", help="Force prompt to require both keyframes and Gemini video analysis when using scroll")
    p.add_argument(
        "--disturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add distractor options D/E/F to the multiple-choice prompt.",
    )
    add_toolpack_cli_arguments(p)
    p.add_argument(
        "--balance_diagnosis_subtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance UCSF-PDGM sampling across Glioblastoma, Oligodendroglioma, and Astrocytoma as evenly as possible.",
    )
    return p


PROMPT_REF = "3d_slicer/track_a_b/ucsf_pdgm.json"


def build_instructions(
    disturb: bool = False,
    force_gemini: bool = False,
    *,
    enable_code_execution_hint: bool = False,
    toolpack_selection: ToolPackSelection,
) -> str:
    answer_set = "A|B|C|D|E|F" if disturb else "A|B|C"
    base = render_prompt(
        PROMPT_REF,
        "templates.system",
        answer_set=answer_set,
        tool_guidance=toolpack_selection.pick_text(
            none_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.none"),
            manual_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.manual"),
            auto_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.auto"),
        ),
        force_gemini_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.force_gemini.{'enabled' if force_gemini else 'disabled'}",
        ),
        code_execution_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.code_execution.{'enabled' if enable_code_execution_hint else 'disabled'}",
        ),
    )
    return toolpack_selection.append_prompt_hints(base)


def _format_segment_requirement(segment_evidence: Dict[str, Any]) -> str:
    source_name = str(segment_evidence.get("source_volume_name") or "").strip()
    source_line = (
        render_prompt(
            PROMPT_REF,
            "templates.fragments.segment_source_line",
            source_name=source_name,
        )
        if source_name
        else ""
    )
    return render_prompt(PROMPT_REF, "templates.segment_requirement", source_line=source_line)


def build_user_message(
    case_id: str,
    case_path: Path,
    disturb: bool = False,
    *,
    segment_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    case_path = case_path.expanduser().resolve()
    extra_options = ""
    answer_set = "A|B|C"
    if disturb:
        extra_options = render_prompt(PROMPT_REF, "templates.fragments.disturb_options.enabled")
        answer_set = "A|B|C|D|E|F"

    evidence_block = ""
    if segment_evidence and bool(segment_evidence.get("has_segment")):
        evidence_block = _format_segment_requirement(segment_evidence) + "\n"

    return render_prompt(
        PROMPT_REF,
        "templates.user",
        case_id=case_id,
        case_path=case_path,
        extra_options=extra_options,
        evidence_block=evidence_block,
        answer_set=answer_set,
    )


def extract_choice_with_disturb(text: str, *, disturb: bool = False) -> Optional[str]:
    pred = extract_choice(text)
    if pred is not None or not disturb:
        return pred

    m = re.search(r"\b(?:ANSWER|CHOICE|OPTION)\s*[:\-]?\s*([DEF])\b", text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper()

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines[-10:]):
        if re.fullmatch(r"[DEF]", ln, re.IGNORECASE):
            return ln.upper()
    return None


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


def _safe_prepare_segment_gt(
    *,
    base_url: str,
    bridge_dir: Path,
    out_dir: Path,
    case_path: Path,
    segment_path: Optional[Path],
    reference_volume_paths: Optional[Sequence[Path]],
) -> Tuple[Optional[SegmentGroundTruth], Optional[str]]:
    if segment_path is None:
        return None, None
    try:
        hidden_bundle = build_context_bundle(
            base_url=base_url,
            out_dir=out_dir,
            bridge_dir=bridge_dir,
            enable_gemini_video=False,
        )
        gt = prepare_nifti_segment_ground_truth(
            hidden_bundle.ctx,
            case_path=case_path,
            out_dir=out_dir,
            segment_path=segment_path,
            reference_volume_paths=reference_volume_paths,
            source_modality="MR",
            provenance_policy="not_applicable",
            clear_scene_after=True,
        )
        return gt, None
    except Exception as e:
        return None, str(e)


def _restart_slicer_runtime_for_case(*, base_url: str, bridge_dir: Path, slicer_executable: Path, case_log: logging.Logger) -> None:
    restart_slicer_runtime(
        base_url=base_url,
        bridge_dir=bridge_dir,
        slicer_executable=slicer_executable,
        logger=case_log,
        reason="segmentation toolpack or periodic restart is active",
        require_exec=True,
        require_slice_render=True,
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
        "provenance_policy": gt.provenance_policy,
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
        case_log.info("[segment-check][%s] %s", case_id, line)


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
    if _ucsf_pdgm_evidence_exact_correct(segment_check):
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] += 1

    key_slice_payload = segment_check.get("key_slice") or {}
    for view in EVIDENCE_VIEWS:
        key_slice_scored[view] += 1
        if bool((key_slice_payload.get(view) or {}).get("correct", False)):
            key_slice_correct[view] += 1

    rsa_metrics["n_scored"] += 1
    if bool((segment_check.get("rsa") or {}).get("correct", False)):
        rsa_metrics["n_correct"] += 1

    provenance_payload = segment_check.get("provenance") or {}
    if bool(provenance_payload.get("scored", False)):
        provenance_metrics["n_scored"] += 1
        if bool(provenance_payload.get("ok", False)):
            provenance_metrics["n_correct"] += 1


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    os.environ["SLICER_CASE_ROOT"] = str(data_root)
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = (data_root / csv_path).resolve()

    log_dir = resolve_run_output_dir(args.log_dir, model=args.model)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark.log")
    log = logging.getLogger("benchmark_ucsf_pdgm")

    bridge_dir = resolve_bridge_dir()
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

    if args.include_segmentations:
        log.warning("Ignoring --include_segmentations to avoid leaking hidden GT segmentations into the agent scene.")

    limit = args.limit if args.limit and args.limit > 0 else None
    cases = list(
        iter_cases(
            data_root,
            csv_path,
            limit=limit,
            include_extra=bool(args.include_extra),
            include_segmentations=False,
            prefer_bias=not bool(args.no_prefer_bias),
            balance_diagnosis_subtypes=bool(args.balance_diagnosis_subtypes),
        )
    )
    log.info("Found %d cases under %s", len(cases), data_root)

    toolpack_selection = resolve_toolpack_cli_selection(args)
    segmentation_toolpack_active = toolpack_selection.provides_pack("segmentation")
    instructions = build_instructions(
        bool(args.disturb),
        bool(args.force_gemini),
        enable_code_execution_hint=bool(extra_model_tools),
        toolpack_selection=toolpack_selection,
    )

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_correct_with_evidence = 0
    n_localization_correct = 0
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

    slicer_executable_path = Path(args.slicer_executable).expanduser()
    restart_slicer_every = max(0, int(args.restart_slicer_every or 0))
    case_timeout_sec = max(0, int(args.case_timeout_sec or 0))

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)
        segment_gt: Optional[SegmentGroundTruth] = None
        segment_gt_error: Optional[str] = None
        session = None

        try:
            with _case_timeout(case_timeout_sec, label=f"Case {case.case_id}"):
                if segmentation_toolpack_active or (restart_slicer_every and (idx - 1) % restart_slicer_every == 0):
                    _restart_slicer_runtime_for_case(
                        base_url=args.base_url,
                        bridge_dir=bridge_dir,
                        slicer_executable=slicer_executable_path,
                        case_log=case_log,
                    )

                segment_gt_dir = case_out / "segment_gt"
                segment_gt, segment_gt_error = _safe_prepare_segment_gt(
                    base_url=args.base_url,
                    bridge_dir=bridge_dir,
                    out_dir=segment_gt_dir,
                    case_path=case.case_dir,
                    segment_path=case.segmentation_file,
                    reference_volume_paths=case.nifti_files,
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
                    script="benchmark_UCSF_PDGM.py",
                    case_id=case.case_id,
                    model=args.model,
                    max_rounds=int(args.max_rounds),
                    toolpacks=toolpack_selection.to_summary_dict(),
                )
                ctx = context_bundle.ctx

                if segment_gt is not None and segment_gt.gt_json_path and segment_gt.gt_json_path.exists():
                    try:
                        payload = json.loads(segment_gt.gt_json_path.read_text(encoding="utf-8"))
                        (case_out / "segment_gt_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                user_msg = build_user_message(
                    case.case_id,
                    case.case_dir,
                    bool(args.disturb),
                    segment_evidence=segment_gt.to_prompt_spec() if segment_gt is not None and segment_gt.has_segment else None,
                )

                open_case_kwargs: Dict[str, Any] = {
                    "preset": "mri_brain",
                    "layout": "four_up",
                    "clear_scene_first": True,
                    "nifti_files": [str(p) for p in case.nifti_files],
                    "include_segmentations": False,
                }
                active_prefer: List[str] = []
                if segment_gt is not None and segment_gt.has_segment and segment_gt.source_volume_name:
                    active_prefer.append(str(segment_gt.source_volume_name))
                if case.nifti_files:
                    active_prefer.append(case.nifti_files[0].stem)
                    active_prefer.append(case.nifti_files[0].name)
                if active_prefer:
                    open_case_kwargs["active_prefer"] = active_prefer

                core_sequences = ", ".join([p.name for p in case.nifti_files]) if case.nifti_files else "none"
                evidence_variant = "disabled"
                evidence_variables: Dict[str, Any] = {}
                if segment_gt is not None and segment_gt.has_segment:
                    evidence_variant = (
                        "named_reference"
                        if segment_gt.source_volume_name
                        else "shared_space"
                    )
                    evidence_variables["source_name"] = str(segment_gt.source_volume_name or "")

                case_brief = render_prompt(
                    PROMPT_REF,
                    "templates.case_brief",
                    case_id=case.case_id,
                    csv_id=case.csv_id,
                    core_sequences=core_sequences,
                    evidence_note=render_prompt(
                        PROMPT_REF,
                        f"templates.fragments.case_brief_evidence.{evidence_variant}",
                        **evidence_variables,
                    ),
                )

                agent_res = run_auto_task(
                    provider=resolved_provider,
                    model_client=model_client,
                    model=args.model,
                    ctx=ctx,
                    case_path=case.case_dir,
                    instructions=instructions,
                    user_message=user_msg,
                    max_rounds=int(args.max_rounds),
                    max_tools=(int(args.max_tools) if int(args.max_tools) > 0 else None),
                    extra_model_tools=extra_model_tools,
                    reasoning_effort=args.reasoning_effort,
                    prompt_metadata=prompt_metadata_bundle(PROMPT_REF, TOOLPACK_PROMPT_REF),
                    provided_tool_packs=toolpack_selection.provided_tool_packs,
                    tool_pack_activation_mode=toolpack_selection.activation_mode,
                    model_request_timeout_sec=120.0,
                    model_request_retry_on_timeout=1,
                    open_case_kwargs=open_case_kwargs,
                    case_brief_text=case_brief,
                )
            raw_text = agent_res.text
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")

            pred = extract_choice_with_disturb(raw_text, disturb=bool(args.disturb))
            gt = case.label

            scored = gt in {"A", "B", "C"}
            correct = bool(scored and pred == gt)
            if scored:
                n_scored += 1
                n_correct += int(correct)

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
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_payload = segment_check.get("key_slice") or {}
            key_slice_all_correct = (
                bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))
                if has_segment_gt
                else None
            )
            if has_segment_gt:
                segment_check["evidence_exact_correct"] = key_slice_all_correct
            localization_correct = (
                bool((segment_check.get("rsa") or {}).get("correct", False))
                if has_segment_gt
                else None
            )
            correct_with_evidence = (
                bool(correct and key_slice_all_correct)
                if has_segment_gt
                else None
            )
            if bool(correct_with_evidence):
                n_correct_with_evidence += 1
            if bool(localization_correct):
                n_localization_correct += 1

            tool_usage = _collect_tool_usage(session.runlog_path if session is not None else None)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "csv_id": case.csv_id,
                "diagnosis": case.diagnosis,
                "diagnosis_subtype": diagnosis_subtype(case.diagnosis),
                "gt": gt,
                "pred": pred,
                "runlog_path": str(session.runlog_path) if session is not None else None,
                "trace_path": str(session.trace_path) if session is not None else None,
                "agent_output_path": str(case_out / "agent_output.txt"),
                "correct": correct,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "scored": scored,
                "nifti_files": [p.name for p in case.nifti_files],
                "segmentation_file": str(case.segmentation_file) if case.segmentation_file else None,
                "has_segment_gt": has_segment_gt,
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script="benchmark_UCSF_PDGM.py",
                case_id=case.case_id,
                ok=True,
                prediction=pred,
                ground_truth=gt,
                correct=correct,
                correct_with_evidence=correct_with_evidence,
                key_slice_all_correct=key_slice_all_correct,
                localization_correct=localization_correct,
                scored=scored,
                has_segment_gt=has_segment_gt,
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                elapsed_sec=rec["seconds"],
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

            case_log.info(
                "Completed case %d/%d: %s | pred=%s | gt=%s | correct=%s | correct_with_evidence=%s | localization_correct=%s | tool_calls=%d",
                idx,
                len(cases),
                case.case_id,
                pred,
                gt,
                correct,
                correct_with_evidence,
                localization_correct,
                tool_usage["n_tool_calls"],
            )
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | pred={pred} | gt={gt} | "
                f"evidence_correct={correct_with_evidence} | localization_correct={localization_correct} | "
                f"tool_calls={tool_usage['n_tool_calls']}"
            )
        except Exception as e:
            n_failed += 1
            gt = case.label
            scored = gt in {"A", "B", "C"}
            if scored:
                n_scored += 1

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
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_payload = segment_check.get("key_slice") or {}
            key_slice_all_correct = (
                bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))
                if has_segment_gt
                else None
            )
            if has_segment_gt:
                segment_check["evidence_exact_correct"] = key_slice_all_correct
            localization_correct = (
                bool((segment_check.get("rsa") or {}).get("correct", False))
                if has_segment_gt
                else None
            )
            correct_with_evidence = False if has_segment_gt else None
            if bool(localization_correct):
                n_localization_correct += 1

            tool_usage = _collect_tool_usage(session.runlog_path)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "csv_id": case.csv_id,
                "diagnosis": case.diagnosis,
                "diagnosis_subtype": diagnosis_subtype(case.diagnosis),
                "gt": gt,
                "pred": None,
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
                "correct": False,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "scored": scored,
                "nifti_files": [p.name for p in case.nifti_files],
                "segmentation_file": str(case.segmentation_file) if case.segmentation_file else None,
                "has_segment_gt": has_segment_gt,
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
                    script="benchmark_UCSF_PDGM.py",
                    case_id=case.case_id,
                    ok=False,
                    error=str(e),
                    elapsed_sec=rec["seconds"],
                    scored=scored,
                    has_segment_gt=has_segment_gt,
                    correct_with_evidence=correct_with_evidence,
                    key_slice_all_correct=key_slice_all_correct,
                    localization_correct=localization_correct,
                    evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                    n_tool_calls=tool_usage["n_tool_calls"],
                    tools_called=tool_usage["tools_called"],
                    tool_call_counts=tool_usage["tool_call_counts"],
                )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | pred=<error> | gt={gt} | "
                f"evidence_correct={correct_with_evidence} | localization_correct={localization_correct} | "
                f"tool_calls={tool_usage['n_tool_calls']}"
            )

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    accuracy_with_evidence = (
        n_correct_with_evidence / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
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
    per_subtype = summarize_subtype_metrics(results)

    summary = {
        "model": args.model,
        "max_rounds": int(args.max_rounds),
        "max_tools": (int(args.max_tools) if int(args.max_tools) > 0 else None),
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "disturb": bool(args.disturb),
        "n_cases": len(cases),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "avg_evidence_correct": avg_evidence_correct,
        "Localization_acc": localization_acc,
        "n_localization_correct": n_localization_correct,
        "n_correct_with_evidence": n_correct_with_evidence,
        "accuracy_with_evidence": accuracy_with_evidence,
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
        "n_failed": n_failed,
        "total_tool_calls": total_tool_calls,
        "average_tool_calls_per_case": average_tool_calls_per_case,
        "tool_call_counts": {k: aggregate_tool_call_counts[k] for k in sorted(aggregate_tool_call_counts.keys())},
        "tools_called": sorted(aggregate_tool_call_counts.keys()),
        "n_unique_tools_used": len(aggregate_tool_call_counts),
        "toolpacks": toolpack_selection.to_summary_dict(),
        "balance_diagnosis_subtypes": bool(args.balance_diagnosis_subtypes),
        "per_subtype": per_subtype,
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_ucsf_pdgm_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info(
        "DONE. accuracy=%.4f (%d/%d) accuracy_with_evidence=%.4f (%d/%d segment-gt) localization_acc=%.4f (%d/%d segment-gt) failed=%d total_tool_calls=%d avg_tool_calls_per_case=%.2f tools=%s",
        accuracy,
        n_correct,
        n_scored,
        accuracy_with_evidence,
        n_correct_with_evidence,
        evidence_case_metrics["n_cases_with_segment_gt"],
        localization_acc,
        n_localization_correct,
        rsa_metrics["n_scored"],
        n_failed,
        total_tool_calls,
        average_tool_calls_per_case,
        sorted(aggregate_tool_call_counts.keys()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
