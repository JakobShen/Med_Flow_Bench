#!/usr/bin/env python3
"""Benchmark: LUMIERE pairwise longitudinal RANO response classification.

Task (4-way multiple choice):
  A) PD
  B) SD
  C) PR
  D) CR

Each benchmark item contains two MRI studies from the same patient:
  - OLD: reference study
  - CURRENT: study to classify

The benchmark uses DeepBraTumIA atlas-space skull-stripped MRI volumes by
default, with an optional native/unregistered image-space mode:
  CT1, T1, T2, FLAIR.

Evaluation is deterministic and rule-based (no LLM-as-judge).
In addition to the multiple-choice answer, the agent is asked to emit a small
set of YES/NO evidence fields describing the CURRENT study. Evidence scoring is
also deterministic and skips fields whose ground truth is unknown from the CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import time
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
from slicer_agent_engine.benchmarking.judge import extract_choice_from_set
from slicer_agent_engine.benchmarking.paper_table import build_lumiere_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.toolpack_cli import (
    TOOLPACK_PROMPT_REF,
    add_toolpack_cli_arguments,
    resolve_toolpack_cli_selection,
)
from slicer_agent_engine.toolpacks import ToolPackSelection
from slicer_agent_engine.benchmarking.lumiere import (
    CHOICE_TO_RANO_LABEL,
    LUMIERE_IMAGE_SPACE_REGISTERED,
    LUMIERE_IMAGE_SPACE_UNREGISTERED,
    LUMIERE_IMAGE_SPACES,
    VALID_RANO_LABELS,
    _balance_questions_by_label,
    LUMIEREQuestion,
    build_question_bank,
    format_loaded_volume_guide,
    stage_question_inputs,
    staged_volume_stem,
)


ANSWER_CHOICES = tuple(CHOICE_TO_RANO_LABEL.keys())
EVIDENCE_FIELDS: Tuple[str, ...] = (
    "NON_MEASURABLE_LESION",
    "T2_PROGRESSION",
    "CURRENT_TARGET_LESION",
    "NEW_TARGET_LESION",
    "CURRENT_NON_TARGET_LESION",
    "NEW_NON_TARGET_LESION",
)
REPORT_TABLE_EVIDENCE_FIELDS: Tuple[str, ...] = tuple(field for field in EVIDENCE_FIELDS if field != "NON_MEASURABLE_LESION")

# Accept plain lines like `T2_PROGRESSION: YES` and tolerate bullets/quotes.
_EVIDENCE_LINE_RE = re.compile(
    r"^[\s>*\-\u2022]*([A-Z][A-Z0-9_]+)\s*:\s*(YES|NO)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Evidence GT extraction is intentionally conservative.
# - NON_MEASURABLE_LESION comes directly from the CSV flag (always scored).
# - The remaining fields are scored only when the rationale contains an explicit cue.
_T2_PROGRESSION_YES_RE = re.compile(r"\b(?:discrete\s+)?t2(?:w)?\s*-?\s*progr(?:ession)?\.?", re.IGNORECASE)
_NEW_TARGET_YES_RE = re.compile(
    r"\b(?:new\s+(?:measurable\s+)?target\s+(?:l\.?|lesion)|new\s+tl)\b",
    re.IGNORECASE,
)
_CURRENT_TARGET_YES_PATTERNS: Tuple[re.Pattern[str], ...] = (
    _NEW_TARGET_YES_RE,
    re.compile(r"\bprogressing\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    re.compile(r"\bregressing\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    re.compile(r"\bnon-?measurable\s+(?:l\.?|lesion)\s+to\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    # Plain measurable target lesion with count/measurement after `Target L`.
    re.compile(r"(?<!non-)(?<!non )\btarget\s+l\.?\s*:?\s*\d", re.IGNORECASE),
)
_CURRENT_NON_TARGET_NO_RE = re.compile(r"\bno\s+non-?target\s+(?:l\.?|lesion(?:s)?)\b", re.IGNORECASE)
_CURRENT_NON_TARGET_YES_RE = re.compile(
    r"\b(?:new\s+|regressing\s+|progressing\s+)?(?:non-?measurable\s+)?non-?target\s+(?:l\.?|lesion(?:s)?)\b",
    re.IGNORECASE,
)
_NEW_NON_TARGET_YES_RE = re.compile(
    r"\b(?:\d+\s+)?new\s+(?:non-?measurable\s+)?non-?target\s+(?:l\.?|lesion(?:s)?)\b",
    re.IGNORECASE,
)


def _filter_labels_with_min_count(
    questions: Sequence[LUMIEREQuestion],
    *,
    min_count: int,
) -> Tuple[List[LUMIEREQuestion], int, List[str], Dict[str, int]]:
    counts = {label: 0 for label in VALID_RANO_LABELS}
    for question in questions:
        label = str(question.gt_rano_label or "").strip()
        if label in counts:
            counts[label] += 1
    observed_labels = [label for label in VALID_RANO_LABELS if counts[label] > 0]
    if min_count <= 0:
        return list(questions), 0, observed_labels, counts
    selected_labels = [label for label in VALID_RANO_LABELS if counts[label] >= min_count]
    if not selected_labels:
        return list(questions), 0, observed_labels, counts
    kept = [
        question
        for question in questions
        if str(question.gt_rano_label or "").strip() in set(selected_labels)
    ]
    return kept, len(questions) - len(kept), selected_labels, counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LUMIERE longitudinal RANO benchmark.")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing the LUMIERE Imaging folder and CSV files")
    p.add_argument("--ratings_csv", type=str, default="LUMIERE-ExpertRating-v202211.csv", help="Expert RANO ratings CSV path")
    p.add_argument("--datacompleteness_csv", type=str, default="LUMIERE-datacompleteness.csv", help="Data completeness CSV path")
    p.add_argument(
        "--lumiere_image_space",
        choices=LUMIERE_IMAGE_SPACES,
        default=LUMIERE_IMAGE_SPACE_REGISTERED,
        help="Use registered DeepBraTumIA atlas-space images or native unregistered top-level LUMIERE images (default: registered).",
    )
    p.add_argument("--provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), help="LLM provider name. Use local_vllm for a local OpenAI-compatible vLLM server; auto infers hosted providers from --model.")
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model name")
    limit_group = p.add_mutually_exclusive_group()
    limit_group.add_argument("--limit_patient", type=int, default=0, help="Run all eligible questions from the first N patients (0 = all)")
    limit_group.add_argument("--limit_question", type=int, default=0, help="Run only the first N eligible questions globally (0 = all)")
    limit_group.add_argument("--limit", dest="limit_question", type=int, default=0, help="Alias for --limit_question")
    p.add_argument("--question_id", type=str, default="", help="Run one exact LUMIERE question ID, e.g. Patient-022__week-045__to__week-053.")
    p.add_argument(
        "--balance_rano_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance LUMIERE sampling across PD, SD, PR, and CR as evenly as possible (default: enabled).",
    )
    p.add_argument(
        "--min_scorable_evidence_fields",
        type=int,
        default=2,
        help="Keep only questions with at least this many scorable ReportTable evidence fields (default: 2). Set 0 to disable filtering.",
    )
    p.add_argument(
        "--min_label_count_before_balance",
        type=int,
        default=10,
        help="Before balancing/limiting, keep only RANO labels with at least this many eligible questions after evidence filtering (default: 10). Set 0 to disable.",
    )
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_lumiere", help="Directory for logs and outputs")
    p.add_argument("--max_rounds", type=int, default=12, help="Max tool-calling rounds")
    p.add_argument("--max_tools", type=int, default=0, help="Compatibility option; current LUMIERE runner uses max_rounds as the tool budget.")
    p.add_argument("--enable_code_interpreter", action="store_true", default=os.environ.get("OPENAI_ENABLE_CODE_INTERPRETER", "").strip().lower() in {"1", "true", "yes", "on"}, help="Expose a provider-native code execution tool during the run when supported (OpenAI Code Interpreter, Gemini code execution, or Qwen code_interpreter).")
    p.add_argument("--reasoning_effort", type=str, choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None, help="Optional provider reasoning-effort hint.")
    p.add_argument("--base_url", type=str, default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"), help="Slicer WebServer base URL")
    p.add_argument(
        "--slicer_executable",
        type=str,
        default=os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"),
        help="Path to Slicer executable",
    )
    p.add_argument(
        "--restart_slicer_every_question",
        action="store_true",
        help="Force a fresh Slicer WebServer at the start of every LUMIERE question.",
    )
    p.add_argument(
        "--fail_on_slicer_disconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail a question if logs show the Slicer WebServer disconnected instead of scoring a placeholder answer.",
    )
    p.add_argument("--enable_gemini", action="store_true", help="Enable Gemini video analysis tools (requires GEMINI_API_KEY)")
    add_toolpack_cli_arguments(p)
    return p


def _slicer_port(base_url: str) -> Optional[int]:
    try:
        return urlparse(base_url).port
    except Exception:
        return None


def _terminate_slicer_on_port(base_url: str, *, log: logging.Logger) -> None:
    port = _slicer_port(base_url)
    if not port:
        return
    try:
        proc = subprocess.run(["lsof", "-ti", f"TCP:{port}"], check=False, capture_output=True, text=True)
    except Exception as e:
        log.warning("Could not inspect Slicer listener on port %s: %s", port, e)
        return
    pids = [pid.strip() for pid in proc.stdout.splitlines() if pid.strip().isdigit()]
    if not pids:
        return
    log.info("Terminating stale Slicer listener(s) on port %s: %s", port, ", ".join(pids))
    subprocess.run(["kill", "-TERM", *pids], check=False)
    time.sleep(2.0)
    still_alive = [pid for pid in pids if Path("/proc", pid).exists()]
    if still_alive:
        log.warning("Force killing stale Slicer listener(s) on port %s: %s", port, ", ".join(still_alive))
        subprocess.run(["kill", "-KILL", *still_alive], check=False)
        time.sleep(1.0)


def _ensure_or_restart_slicer(
    *,
    base_url: str,
    bridge_dir: Path,
    slicer_executable: Path,
    restart: bool,
    log: logging.Logger,
) -> None:
    if restart:
        _terminate_slicer_on_port(base_url, log=log)
    ensure_slicer_runtime(
        base_url=base_url,
        bridge_dir=bridge_dir,
        slicer_executable=slicer_executable,
        require_exec=True,
        require_slice_render=True,
    )


def _contains_slicer_disconnect_text(text: str) -> bool:
    needles = (
        "Connection refused",
        "Failed to call Slicer WebServer",
        "Max retries exceeded with url",
        "Slicer WebServer is currently unreachable",
        "Slicer WebServer being unreachable",
    )
    return any(needle in text for needle in needles)


def _case_has_slicer_disconnect(case_out: Path, session: Any = None) -> bool:
    paths = [case_out / "trace.jsonl", case_out / "runlog.jsonl", case_out / "agent_output.txt"]
    if session is not None:
        for attr in ("trace_path", "runlog_path"):
            p = getattr(session, attr, None)
            if p is not None:
                paths.append(Path(p))
    for path in paths:
        if not path.exists():
            continue
        try:
            if _contains_slicer_disconnect_text(path.read_text(encoding="utf-8", errors="replace")):
                return True
        except Exception:
            continue
    return False


PROMPT_REF = "3d_slicer/track_a_b/lumiere.json"


def build_instructions(
    *,
    enable_code_execution_hint: bool = False,
    toolpack_selection: ToolPackSelection,
    image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED,
) -> str:
    image_space_variant = (
        "unregistered"
        if image_space == LUMIERE_IMAGE_SPACE_UNREGISTERED
        else "registered"
    )
    text = render_prompt(
        PROMPT_REF,
        "templates.system",
        tool_guidance=toolpack_selection.pick_text(
            none_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.none"),
            manual_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.manual"),
            auto_text=render_prompt(PROMPT_REF, "templates.fragments.tool_guidance.auto"),
        ),
        registration_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.registration_guidance.{image_space_variant}",
        ),
        code_execution_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.code_execution.{'enabled' if enable_code_execution_hint else 'disabled'}",
        ),
    )
    return toolpack_selection.append_prompt_hints(text)


def build_case_brief(question: LUMIEREQuestion, *, image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED) -> str:
    image_space_variant = (
        "unregistered"
        if image_space == LUMIERE_IMAGE_SPACE_UNREGISTERED
        else "registered"
    )
    return render_prompt(
        PROMPT_REF,
        "templates.case_brief",
        question_id=question.question_id,
        image_space_text=render_prompt(
            PROMPT_REF,
            f"templates.fragments.image_space.{image_space_variant}",
        ),
        patient_id=question.patient_id,
        old_timepoint=question.old_visit.timepoint,
        current_timepoint=question.current_visit.timepoint,
        loaded_volume_guide=format_loaded_volume_guide(question),
    )


def _output_template_text() -> str:
    return render_prompt(PROMPT_REF, "templates.output")


def build_user_message(
    question: LUMIEREQuestion,
    *,
    staged_dir: Path,
    staged_files: Sequence[Path],
    image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED,
) -> str:
    file_lines = "\n".join(f"  - {str(path)}" for path in staged_files)
    image_space_variant = (
        "unregistered"
        if image_space == LUMIERE_IMAGE_SPACE_UNREGISTERED
        else "registered"
    )
    return render_prompt(
        PROMPT_REF,
        "templates.user",
        question_id=question.question_id,
        patient_id=question.patient_id,
        old_timepoint=question.old_visit.timepoint,
        current_timepoint=question.current_visit.timepoint,
        staged_dir=staged_dir,
        file_lines=file_lines,
        registration_hint=render_prompt(
            PROMPT_REF,
            f"templates.fragments.registration_hint.{image_space_variant}",
        ),
        output_template=_output_template_text(),
    )


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


def _merge_tool_counts(dest: Dict[str, int], src: Dict[str, int]) -> None:
    for name, count in src.items():
        dest[str(name)] = int(dest.get(str(name), 0)) + int(count)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _resolve_csv_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_root / path).resolve()


def _per_class_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {label: 0 for label in VALID_RANO_LABELS}
    correct = {label: 0 for label in VALID_RANO_LABELS}
    for rec in results:
        gt_label = str(rec.get("gt_rano_label") or "").strip()
        if gt_label not in counts:
            continue
        counts[gt_label] += 1
        if bool(rec.get("correct", False)):
            correct[gt_label] += 1
    accuracy = {
        label: ((correct[label] / counts[label]) if counts[label] else 0.0)
        for label in VALID_RANO_LABELS
    }
    observed = [accuracy[label] for label in VALID_RANO_LABELS if counts[label] > 0]
    balanced_accuracy = (sum(observed) / len(observed)) if observed else 0.0
    return {
        "counts": counts,
        "correct": correct,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def _normalize_rationale_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_evidence_predictions(text: str) -> Dict[str, Optional[bool]]:
    found: Dict[str, bool] = {}
    if text:
        for match in _EVIDENCE_LINE_RE.finditer(text):
            raw_name = str(match.group(1) or "").strip().upper()
            if raw_name.startswith("EVIDENCE_"):
                raw_name = raw_name[len("EVIDENCE_") :]
            if raw_name not in EVIDENCE_FIELDS:
                continue
            found[raw_name] = str(match.group(2) or "").strip().upper() == "YES"
    return {field: found.get(field) for field in EVIDENCE_FIELDS}


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _infer_evidence_ground_truth(question: LUMIEREQuestion) -> Tuple[Dict[str, Optional[bool]], Dict[str, Optional[str]]]:
    rationale = _normalize_rationale_text(question.current_visit.rationale)
    gt: Dict[str, Optional[bool]] = {field: None for field in EVIDENCE_FIELDS}
    source: Dict[str, Optional[str]] = {field: None for field in EVIDENCE_FIELDS}

    # Explicit CSV field: always score this.
    gt["NON_MEASURABLE_LESION"] = bool(question.current_visit.non_measurable_lesions)
    source["NON_MEASURABLE_LESION"] = "csv.NonMeasurableLesions"

    if not rationale:
        return gt, source

    if _T2_PROGRESSION_YES_RE.search(rationale):
        gt["T2_PROGRESSION"] = True
        source["T2_PROGRESSION"] = "rationale:T2-Progr"

    if _matches_any(rationale, _CURRENT_TARGET_YES_PATTERNS):
        gt["CURRENT_TARGET_LESION"] = True
        source["CURRENT_TARGET_LESION"] = "rationale:target lesion mention"

    if _NEW_TARGET_YES_RE.search(rationale):
        gt["NEW_TARGET_LESION"] = True
        source["NEW_TARGET_LESION"] = "rationale:new target lesion"

    if _CURRENT_NON_TARGET_NO_RE.search(rationale):
        gt["CURRENT_NON_TARGET_LESION"] = False
        source["CURRENT_NON_TARGET_LESION"] = "rationale:no non-target lesion"
    elif _CURRENT_NON_TARGET_YES_RE.search(rationale):
        gt["CURRENT_NON_TARGET_LESION"] = True
        source["CURRENT_NON_TARGET_LESION"] = "rationale:non-target lesion mention"

    if _NEW_NON_TARGET_YES_RE.search(rationale):
        gt["NEW_NON_TARGET_LESION"] = True
        source["NEW_NON_TARGET_LESION"] = "rationale:new non-target lesion"

    return gt, source


def _count_reporttable_evidence_fields(question: LUMIEREQuestion) -> int:
    gt, _ = _infer_evidence_ground_truth(question)
    return sum(1 for field in REPORT_TABLE_EVIDENCE_FIELDS if gt.get(field) is not None)


def _format_bool_label(value: Optional[bool], *, unknown_label: str = "UNKNOWN") -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return unknown_label


def _score_evidence_fields(
    *,
    gt: Dict[str, Optional[bool]],
    pred: Dict[str, Optional[bool]],
    source: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    details: Dict[str, Dict[str, Any]] = {}
    n_scored = 0
    n_correct = 0
    for field in EVIDENCE_FIELDS:
        gt_val = gt.get(field)
        pred_val = pred.get(field)
        scored = gt_val is not None
        correct = bool(scored and pred_val is not None and pred_val == gt_val)
        if scored:
            n_scored += 1
            n_correct += int(correct)
        details[field] = {
            "gt": gt_val,
            "gt_label": _format_bool_label(gt_val),
            "pred": pred_val,
            "pred_label": _format_bool_label(pred_val, unknown_label="MISSING"),
            "source": source.get(field),
            "scored": scored,
            "correct": correct,
        }
    return details, n_scored, n_correct


def _aggregate_evidence_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_field: Dict[str, Dict[str, Any]] = {}
    total_scored = 0
    total_correct = 0
    n_questions_with_any_evidence_scored = 0
    n_questions_all_scored_evidence_correct = 0
    n_questions_correct_and_all_scored_evidence_correct = 0

    for rec in results:
        evidence = rec.get("evidence") or {}
        case_has_scored_evidence = any(bool((evidence.get(field) or {}).get("scored", False)) for field in EVIDENCE_FIELDS)
        if case_has_scored_evidence:
            n_questions_with_any_evidence_scored += 1
            if int(rec.get("n_evidence_fields_scored") or 0) == int(rec.get("n_evidence_fields_correct") or 0):
                n_questions_all_scored_evidence_correct += 1
                if bool(rec.get("correct")):
                    n_questions_correct_and_all_scored_evidence_correct += 1

    for field in EVIDENCE_FIELDS:
        n_scored = 0
        n_correct = 0
        n_gt_yes = 0
        n_gt_no = 0
        n_pred_missing = 0
        for rec in results:
            item = (rec.get("evidence") or {}).get(field) or {}
            if not bool(item.get("scored", False)):
                continue
            n_scored += 1
            gt_val = item.get("gt")
            pred_val = item.get("pred")
            if gt_val is True:
                n_gt_yes += 1
            elif gt_val is False:
                n_gt_no += 1
            if pred_val is None:
                n_pred_missing += 1
            if bool(item.get("correct", False)):
                n_correct += 1
        accuracy = (n_correct / n_scored) if n_scored else 0.0
        per_field[field] = {
            "n_scored": n_scored,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "n_gt_yes": n_gt_yes,
            "n_gt_no": n_gt_no,
            "n_pred_missing": n_pred_missing,
        }
        total_scored += n_scored
        total_correct += n_correct

    micro_accuracy = (total_correct / total_scored) if total_scored else 0.0
    all_scored_evidence_correct_accuracy = (
        n_questions_all_scored_evidence_correct / n_questions_with_any_evidence_scored
        if n_questions_with_any_evidence_scored
        else 0.0
    )
    correct_and_all_scored_evidence_correct_accuracy = (
        n_questions_correct_and_all_scored_evidence_correct / n_questions_with_any_evidence_scored
        if n_questions_with_any_evidence_scored
        else 0.0
    )
    return {
        "fields": per_field,
        "total_scored": total_scored,
        "total_correct": total_correct,
        "micro_accuracy": micro_accuracy,
        "n_questions_with_any_evidence_scored": n_questions_with_any_evidence_scored,
        "n_questions_all_scored_evidence_correct": n_questions_all_scored_evidence_correct,
        "all_scored_evidence_correct_accuracy": all_scored_evidence_correct_accuracy,
        "n_questions_correct_and_all_scored_evidence_correct": n_questions_correct_and_all_scored_evidence_correct,
        "correct_and_all_scored_evidence_correct_accuracy": correct_and_all_scored_evidence_correct_accuracy,
    }


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    os.environ["SLICER_CASE_ROOT"] = str(data_root)
    ratings_csv = _resolve_csv_path(data_root, args.ratings_csv)
    datacompleteness_csv = _resolve_csv_path(data_root, args.datacompleteness_csv)

    log_dir = resolve_run_output_dir(args.log_dir, model=args.model)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark.log")
    log = logging.getLogger("benchmark_lumiere")

    bridge_dir = resolve_bridge_dir()
    os.environ["SLICER_EXECUTABLE"] = str(Path(args.slicer_executable).expanduser())
    _ensure_or_restart_slicer(
        base_url=args.base_url,
        bridge_dir=bridge_dir,
        slicer_executable=Path(args.slicer_executable).expanduser(),
        restart=False,
        log=log,
    )

    model_runtime = build_model_runtime(
        provider=args.provider,
        model=args.model,
        enable_code_interpreter=bool(args.enable_code_interpreter),
    )
    resolved_provider = model_runtime.resolved_provider
    model_client = model_runtime.model_client
    extra_model_tools = model_runtime.extra_model_tools

    question_id_filter = str(args.question_id or "").strip()
    question_bank = build_question_bank(
        data_root=data_root,
        ratings_csv=ratings_csv,
        datacompleteness_csv=datacompleteness_csv,
        limit_patient=None if question_id_filter else (args.limit_patient if args.limit_patient and args.limit_patient > 0 else None),
        limit_question=None,
        balance_rano_labels=False,
        image_space=args.lumiere_image_space,
    )
    min_scorable_evidence_fields = max(0, int(args.min_scorable_evidence_fields or 0))
    questions = list(question_bank.questions)
    if question_id_filter:
        questions = [question for question in questions if str(question.question_id) == question_id_filter]
        if not questions:
            raise SystemExit(f"No LUMIERE question matched --question_id {question_id_filter!r}")
        if args.limit_question and args.limit_question > 0:
            questions = questions[: int(args.limit_question)]
        log.info("Filtered to %d LUMIERE question(s) matching --question_id=%s", len(questions), question_id_filter)
    questions_before_evidence_filter = list(questions)
    n_filtered_insufficient_scorable_evidence = 0
    if min_scorable_evidence_fields > 0:
        questions = [
            question
            for question in questions_before_evidence_filter
            if _count_reporttable_evidence_fields(question) >= min_scorable_evidence_fields
        ]
        n_filtered_insufficient_scorable_evidence = len(questions_before_evidence_filter) - len(questions)
    min_label_count_before_balance = max(0, int(args.min_label_count_before_balance or 0))
    questions_before_label_count_filter = list(questions)
    questions, n_filtered_low_count_labels, selected_balance_labels, per_label_counts_after_evidence = _filter_labels_with_min_count(
        questions,
        min_count=min_label_count_before_balance,
    )
    questions_before_balance_filter = list(questions)
    if bool(args.balance_rano_labels):
        questions = _balance_questions_by_label(questions)
    questions_before_final_limit = list(questions)
    if args.limit_question and args.limit_question > 0:
        questions = questions[: int(args.limit_question)]
    bank_stats = dict(question_bank.stats)
    bank_stats["question_id_filter"] = question_id_filter or None
    bank_stats["min_scorable_evidence_fields"] = min_scorable_evidence_fields
    bank_stats["n_questions_before_evidence_filter"] = len(questions_before_evidence_filter)
    bank_stats["n_questions_after_evidence_filter"] = len(questions_before_balance_filter)
    bank_stats["n_filtered_insufficient_scorable_evidence"] = n_filtered_insufficient_scorable_evidence
    bank_stats["n_patients_after_evidence_filter"] = len({q.patient_id for q in questions_before_balance_filter})
    bank_stats["min_label_count_before_balance"] = min_label_count_before_balance
    bank_stats["n_questions_before_label_count_filter"] = len(questions_before_label_count_filter)
    bank_stats["n_questions_after_label_count_filter"] = len(questions_before_balance_filter)
    bank_stats["n_filtered_low_count_labels"] = n_filtered_low_count_labels
    bank_stats["selected_labels_before_balance"] = list(selected_balance_labels)
    bank_stats["per_label_counts_after_evidence_filter"] = {
        label: int(per_label_counts_after_evidence.get(label, 0)) for label in VALID_RANO_LABELS
    }
    per_label_counts_after_label_count_filter = {label: 0 for label in VALID_RANO_LABELS}
    for question in questions_before_balance_filter:
        per_label_counts_after_label_count_filter[question.gt_rano_label] += 1
    bank_stats["per_label_counts_after_label_count_filter"] = per_label_counts_after_label_count_filter
    bank_stats["n_questions_before_balance_filter"] = len(questions_before_balance_filter)
    bank_stats["n_questions_after_balance_filter"] = len(questions_before_final_limit)
    bank_stats["n_filtered_excess_balanced_subtypes"] = 0
    bank_stats["n_patients_after_balance_filter"] = len({q.patient_id for q in questions_before_final_limit})
    bank_stats["limit_question"] = int(args.limit_question or 0)
    bank_stats["n_questions_before_final_limit"] = len(questions_before_final_limit)
    bank_stats["n_questions_after_limit"] = len(questions)
    bank_stats["n_patients_after_limit"] = len({q.patient_id for q in questions})
    per_label_counts_after_limit = {label: 0 for label in VALID_RANO_LABELS}
    for question in questions:
        per_label_counts_after_limit[question.gt_rano_label] += 1
    bank_stats["per_label_counts_after_limit"] = per_label_counts_after_limit
    if min_scorable_evidence_fields > 0:
        log.info(
            "Filtering LUMIERE questions to those with at least %d scorable ReportTable evidence fields: kept %d / %d.",
            min_scorable_evidence_fields,
            len(questions_before_balance_filter),
            len(questions_before_evidence_filter),
        )
    if min_label_count_before_balance > 0:
        log.info(
            "Filtering LUMIERE labels to those with at least %d eligible questions before balancing: kept labels=%s, kept %d / %d questions.",
            min_label_count_before_balance,
            ",".join(selected_balance_labels) if selected_balance_labels else "<none>",
            len(questions_before_balance_filter),
            len(questions_before_label_count_filter),
        )
    if bool(args.balance_rano_labels):
        log.info(
            "Balancing LUMIERE questions by label ordering before final limit: prepared %d questions.",
            len(questions_before_final_limit),
        )
    if args.limit_question and args.limit_question > 0 and len(questions_before_final_limit) != len(questions):
        log.info(
            "Applying final LUMIERE question limit after evidence/label-count/balance filtering: kept %d / %d.",
            len(questions),
            len(questions_before_final_limit),
        )
    log.info("Prepared %d LUMIERE questions from %s", len(questions), data_root)

    toolpack_selection = resolve_toolpack_cli_selection(args)
    instructions = build_instructions(
        enable_code_execution_hint=bool(extra_model_tools),
        toolpack_selection=toolpack_selection,
        image_space=args.lumiere_image_space,
    )

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_failed = 0
    aggregate_tool_call_counts: Dict[str, int] = {}
    total_tool_calls = 0

    for idx, question in enumerate(questions, start=1):
        t0 = time.time()
        case_out = log_dir / question.question_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.{question.question_id}")
        _print_progress(f"Processing question {idx}/{len(questions)}: {question.question_id}")
        case_log.info("Processing question %d/%d: %s", idx, len(questions), question.question_id)

        staged = None
        session = None
        gt_evidence, gt_evidence_source = _infer_evidence_ground_truth(question)
        try:
            _ensure_or_restart_slicer(
                base_url=args.base_url,
                bridge_dir=bridge_dir,
                slicer_executable=Path(args.slicer_executable).expanduser(),
                restart=bool(args.restart_slicer_every_question),
                log=case_log,
            )
            staged = stage_question_inputs(question, case_out / "staged_inputs", image_space=args.lumiere_image_space)
            staging_manifest = {
                "question_id": question.question_id,
                "patient_id": question.patient_id,
                "old_timepoint": question.old_visit.timepoint,
                "current_timepoint": question.current_visit.timepoint,
                "image_space": args.lumiere_image_space,
                "all_files": [str(path) for path in staged.all_files],
                "old_files": {k: str(v) for k, v in staged.old_files.items()},
                "current_files": {k: str(v) for k, v in staged.current_files.items()},
            }
            (case_out / "question_manifest.json").write_text(json.dumps(staging_manifest, indent=2), encoding="utf-8")

            context_bundle = build_context_bundle(
                base_url=args.base_url,
                out_dir=case_out,
                bridge_dir=bridge_dir,
                enable_gemini_video=bool(args.enable_gemini),
            )
            session = context_bundle.session
            ctx = context_bundle.ctx
            session.log_run_event(
                "case_script_start",
                script="benchmark_LUMIERE.py",
                case_id=question.question_id,
                patient_id=question.patient_id,
                model=args.model,
                max_rounds=int(args.max_rounds),
                toolpacks=toolpack_selection.to_summary_dict(),
            )

            current_ct1_stem = staged_volume_stem(side="CURRENT", timepoint=question.current_visit.timepoint, modality="CT1")
            open_case_kwargs: Dict[str, Any] = {
                "preset": "mri_brain",
                "layout": "four_up",
                "clear_scene_first": True,
                "nifti_files": [str(path) for path in staged.all_files],
                "include_segmentations": False,
                "active_prefer": [current_ct1_stem, "CURRENT", "CT1"],
            }

            user_msg = build_user_message(
                question,
                staged_dir=staged.stage_dir,
                staged_files=staged.all_files,
                image_space=args.lumiere_image_space,
            )
            case_brief = build_case_brief(question, image_space=args.lumiere_image_space)

            agent_res = run_auto_task(
                provider=resolved_provider,
                model_client=model_client,
                model=args.model,
                ctx=ctx,
                case_path=staged.stage_dir,
                instructions=instructions,
                user_message=user_msg,
                max_rounds=int(args.max_rounds),
                open_case_kwargs=open_case_kwargs,
                case_brief_text=case_brief,
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
            if bool(args.fail_on_slicer_disconnect) and _case_has_slicer_disconnect(case_out, session):
                raise RuntimeError("Slicer WebServer disconnected during this question; refusing to score a placeholder answer.")

            gt_choice = question.gt_choice
            gt_rano_label = question.gt_rano_label
            pred = extract_choice_from_set(raw_text, valid_choices=ANSWER_CHOICES)
            scored = gt_choice in ANSWER_CHOICES
            correct = bool(scored and pred == gt_choice)
            if scored:
                n_scored += 1
                n_correct += int(correct)

            pred_evidence = _extract_evidence_predictions(raw_text)
            evidence_details, n_evidence_scored, n_evidence_correct = _score_evidence_fields(
                gt=gt_evidence,
                pred=pred_evidence,
                source=gt_evidence_source,
            )

            tool_usage = _collect_tool_usage(session.runlog_path)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])

            rec = {
                "question_id": question.question_id,
                "patient_id": question.patient_id,
                "question_index_for_patient": question.question_index_for_patient,
                "old_timepoint": question.old_visit.timepoint,
                "current_timepoint": question.current_visit.timepoint,
                "gt": gt_choice,
                "gt_rano_label": gt_rano_label,
                "pred": pred,
                "pred_rano_label": CHOICE_TO_RANO_LABEL.get(pred) if pred else None,
                "correct": correct,
                "scored": scored,
                "current_rationale": question.current_visit.rationale,
                "evidence": evidence_details,
                "n_evidence_fields_scored": n_evidence_scored,
                "n_evidence_fields_correct": n_evidence_correct,
                "evidence_micro_accuracy_for_case": ((n_evidence_correct / n_evidence_scored) if n_evidence_scored else None),
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
                "agent_output_path": str(case_out / "agent_output.txt"),
                "staged_case_path": str(staged.stage_dir),
                "old_files": {k: str(v) for k, v in staged.old_files.items()},
                "current_files": {k: str(v) for k, v in staged.current_files.items()},
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script="benchmark_LUMIERE.py",
                case_id=question.question_id,
                patient_id=question.patient_id,
                ok=True,
                prediction=pred,
                ground_truth=gt_choice,
                ground_truth_rano_label=gt_rano_label,
                correct=correct,
                scored=scored,
                elapsed_sec=rec["seconds"],
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
                n_evidence_fields_scored=n_evidence_scored,
                n_evidence_fields_correct=n_evidence_correct,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.info(
                "Completed question %d/%d: %s | pred=%s | gt=%s (%s) | correct=%s | evidence=%d/%d | tool_calls=%d",
                idx,
                len(questions),
                question.question_id,
                pred,
                gt_choice,
                gt_rano_label,
                correct,
                n_evidence_correct,
                n_evidence_scored,
                tool_usage["n_tool_calls"],
            )
            case_log.info("Model answer for %s:\n%s", question.question_id, raw_text)
            _print_progress(
                f"Completed question {idx}/{len(questions)}: {question.question_id} | pred={pred} | gt={gt_choice} ({gt_rano_label}) | evidence={n_evidence_correct}/{n_evidence_scored} | tool_calls={tool_usage['n_tool_calls']}"
            )
        except Exception as e:
            n_failed += 1
            gt_choice = question.gt_choice
            gt_rano_label = question.gt_rano_label
            n_scored += 1
            tool_usage = _collect_tool_usage(session.runlog_path if session is not None else None)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            evidence_details, n_evidence_scored, n_evidence_correct = _score_evidence_fields(
                gt=gt_evidence,
                pred={field: None for field in EVIDENCE_FIELDS},
                source=gt_evidence_source,
            )
            rec = {
                "question_id": question.question_id,
                "patient_id": question.patient_id,
                "question_index_for_patient": question.question_index_for_patient,
                "old_timepoint": question.old_visit.timepoint,
                "current_timepoint": question.current_visit.timepoint,
                "gt": gt_choice,
                "gt_rano_label": gt_rano_label,
                "pred": None,
                "pred_rano_label": None,
                "correct": False,
                "scored": True,
                "current_rationale": question.current_visit.rationale,
                "evidence": evidence_details,
                "n_evidence_fields_scored": n_evidence_scored,
                "n_evidence_fields_correct": n_evidence_correct,
                "evidence_micro_accuracy_for_case": ((n_evidence_correct / n_evidence_scored) if n_evidence_scored else None),
                "staged_case_path": str(staged.stage_dir) if staged is not None else None,
                "runlog_path": str(session.runlog_path) if session is not None else None,
                "trace_path": str(session.trace_path) if session is not None else None,
                "error": str(e),
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Question failed: %s", e)
            _print_progress(
                f"Completed question {idx}/{len(questions)}: {question.question_id} | pred=<error> | gt={gt_choice} ({gt_rano_label})"
            )

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    average_tool_calls_per_question = (total_tool_calls / len(results)) if results else 0.0
    class_metrics = _per_class_metrics(results)
    evidence_metrics = _aggregate_evidence_metrics(results)

    summary = {
        "model": args.model,
        "provider": resolved_provider,
        "data_root": str(data_root),
        "ratings_csv": str(ratings_csv),
        "datacompleteness_csv": str(datacompleteness_csv),
        "image_space": args.lumiere_image_space,
        "answer_mapping": {choice: label for choice, label in CHOICE_TO_RANO_LABEL.items()},
        "evidence_fields": list(EVIDENCE_FIELDS),
        "n_questions": len(questions),
        "n_patients_in_run": len({q.patient_id for q in questions}),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "avg_evidence_correct": evidence_metrics["micro_accuracy"],
        "strict_avg_evidence_correct": evidence_metrics["correct_and_all_scored_evidence_correct_accuracy"],
        "per_class": class_metrics,
        "evidence": evidence_metrics,
        "n_failed": n_failed,
        "total_tool_calls": total_tool_calls,
        "average_tool_calls_per_question": average_tool_calls_per_question,
        "tool_call_counts": {k: aggregate_tool_call_counts[k] for k in sorted(aggregate_tool_call_counts.keys())},
        "tools_called": sorted(aggregate_tool_call_counts.keys()),
        "n_unique_tools_used": len(aggregate_tool_call_counts),
        "limit_patient": int(args.limit_patient or 0),
        "limit_question": int(args.limit_question or 0),
        "balance_rano_labels": bool(args.balance_rano_labels),
        "min_scorable_evidence_fields": min_scorable_evidence_fields,
        "toolpacks": toolpack_selection.to_summary_dict(),
        "question_bank": bank_stats,
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_lumiere_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info(
        "DONE. accuracy=%.4f (%d/%d) failed=%d total_tool_calls=%d avg_tool_calls_per_question=%.2f tools=%s evidence_micro=%.4f",
        accuracy,
        n_correct,
        n_scored,
        n_failed,
        total_tool_calls,
        average_tool_calls_per_question,
        sorted(aggregate_tool_call_counts.keys()),
        evidence_metrics["micro_accuracy"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
