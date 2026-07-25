#!/usr/bin/env python3
"""Track C / Open-Method benchmark runner for LUMIERE.

This runner reuses the question-bank construction and deterministic evidence
scoring while allowing Track C methods to operate on raw atlas-space NIfTI
volumes or derived image representations.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys

SCRIPT_SUPPORT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_SUPPORT_ROOT))

from _script_support import configure_logging, default_model_from_env, ensure_repo_imports, resolve_run_output_dir

ensure_repo_imports()

from medflow_prompts import prompt_metadata, render_prompt
from slicer_agent_engine.benchmarking.judge import extract_choice_from_set
from slicer_agent_engine.benchmarking.paper_table import build_lumiere_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.lumiere import (
    ANSWER_CHOICES,
    ATLAS_MODALITIES,
    CHOICE_TO_RANO_LABEL,
    EVIDENCE_FIELDS,
    LUMIEREQuestion,
    VALID_RANO_LABELS,
    _balance_questions_by_label,
    _CURRENT_NON_TARGET_NO_RE,
    _CURRENT_NON_TARGET_YES_RE,
    _CURRENT_TARGET_YES_PATTERNS,
    _NEW_NON_TARGET_YES_RE,
    _NEW_TARGET_YES_RE,
    _T2_PROGRESSION_YES_RE,
    _aggregate_evidence_metrics,
    _count_reporttable_evidence_fields,
    _extract_evidence_predictions,
    _infer_evidence_ground_truth,
    _per_class_metrics,
    _score_evidence_fields,
    build_question_bank,
    format_loaded_volume_guide,
    stage_question_inputs,
)
from slicer_agent_engine.session import SessionManager
from radiology_track_c import (
    ExternalWorkerError,
    aggregate_sequence_outputs,
    build_compressed_figures,
    build_table3_artifacts,
    build_volume_summary,
    load_case_volumes,
    method_display_name,
    normalize_method_name,
    parse_csv_list,
    run_external_worker,
    run_multimodal_prompt,
    safe_token,
)


SCRIPT_NAME = "benchmark_LUMIERE_track_c.py"
PROMPT_REF = "track_c/lumiere.json"


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
    selected_set = set(selected_labels)
    kept = [
        question
        for question in questions
        if str(question.gt_rano_label or "").strip() in selected_set
    ]
    return kept, len(questions) - len(kept), selected_labels, counts


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _zero_tool_usage() -> Dict[str, Any]:
    return {
        "n_tool_calls": 0,
        "n_unique_tools": 0,
        "tools_called": [],
        "tool_call_counts": {},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LUMIERE Track C benchmark.")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing the LUMIERE Imaging folder and CSV files")
    p.add_argument("--ratings_csv", type=str, default="LUMIERE-ExpertRating-v202211.csv", help="Expert RANO ratings CSV path")
    p.add_argument("--datacompleteness_csv", type=str, default="LUMIERE-datacompleteness.csv", help="Data completeness CSV path")
    limit_group = p.add_mutually_exclusive_group()
    limit_group.add_argument("--limit_patient", type=int, default=0, help="Run all eligible questions from the first N patients (0 = all)")
    limit_group.add_argument("--limit_question", type=int, default=0, help="Run only the first N eligible questions globally (0 = all)")
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
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_lumiere_track_c", help="Directory for outputs")
    p.add_argument(
        "--method",
        type=str,
        default="table3_api",
        help="Track C method: compressed_figures | table3_api | m3d | radfm",
    )
    p.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "auto"),
        help="LLM provider for API-backed Track C methods; local_vllm connects to a local OpenAI-compatible vLLM server.",
    )
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model for API-backed Track C methods")
    p.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None,
        help="Optional provider reasoning-effort hint for API-backed Track C methods.",
    )
    p.add_argument("--api_timeout_sec", type=float, default=300.0, help="Timeout for API-backed model requests")
    p.add_argument("--worker_timeout_sec", type=float, default=1800.0, help="Timeout for local worker calls (M3D/RadFM)")

    p.add_argument("--figure_planes", type=str, default="axial,sagittal,coronal", help="Planes for compressed_figures")
    p.add_argument("--slices_per_figure", type=int, default=42, help="Slices per figure for compressed_figures")
    p.add_argument("--max_figures", type=int, default=9, help="Maximum number of figures for compressed_figures")

    p.add_argument("--table3_planes", type=str, default="axial", help="Planes for the Table-3-style sampled-slices baseline")
    p.add_argument("--table3_sample_slices", type=int, default=10, help="Number of uniformly sampled slices per plane")
    p.add_argument(
        "--table3_delivery",
        type=str,
        choices=["montage", "images"],
        default="montage",
        help="Whether to send sampled slices as per-volume montages or as separate images.",
    )

    p.add_argument(
        "--lumiere_modalities",
        type=str,
        default=",".join(ATLAS_MODALITIES),
        help="Comma-separated modality subset drawn from CT1,T1,T2,FLAIR. API methods use the full staged set; worker methods aggregate across these paired modalities.",
    )
    p.add_argument(
        "--pair_gap_slices",
        type=int,
        default=4,
        help="Number of blank separator slices inserted between OLD and CURRENT when building synthetic worker pair volumes.",
    )

    p.add_argument("--worker_python", type=str, default=None, help="Default Python executable for external Track C workers")

    p.add_argument(
        "--m3d_python",
        type=str,
        default=None,
        help="Python executable for the M3D worker. Defaults to --worker_python or the current interpreter.",
    )
    p.add_argument(
        "--m3d_model_name_or_path",
        type=str,
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        help="HF model path or local model path for M3D",
    )
    p.add_argument("--m3d_device", type=str, default="cuda", help="Device for M3D worker")
    p.add_argument("--m3d_dtype", type=str, default="bfloat16", help="torch dtype for M3D worker")
    p.add_argument("--m3d_max_new_tokens", type=int, default=256, help="Generation length for M3D worker")

    p.add_argument(
        "--radfm_python",
        type=str,
        default=None,
        help="Python executable for the RadFM worker. Defaults to --worker_python or the current interpreter.",
    )
    p.add_argument("--radfm_repo_root", type=str, default=None, help="Path to a cloned RadFM repository")
    p.add_argument("--radfm_checkpoint", type=str, default=None, help="Path to RadFM pytorch_model.bin checkpoint")
    p.add_argument("--radfm_language_dir", type=str, default=None, help="Optional explicit Path to RadFM Language_files")
    p.add_argument("--radfm_device", type=str, default="cuda", help="Device for RadFM worker")
    p.add_argument("--radfm_target_depth", type=int, default=32, help="Target depth for RadFM 3D volume preprocessing")
    p.add_argument("--radfm_max_new_tokens", type=int, default=200, help="Generation length for RadFM worker")
    return p


def _resolve_csv_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_root / path).resolve()


def _selected_modalities(raw: str | Sequence[str]) -> List[str]:
    requested = [str(item).strip().upper() for item in parse_csv_list(raw, default=ATLAS_MODALITIES)]
    allowed = {str(item).upper() for item in ATLAS_MODALITIES}
    out: List[str] = []
    for item in requested:
        if item and item in allowed and item not in out:
            out.append(item)
    if not out:
        out = list(ATLAS_MODALITIES)
    return out


def _build_api_artifacts_and_summary(
    *,
    method: str,
    question_id: str,
    case_out: Path,
    volumes: Sequence[Any],
    args: argparse.Namespace,
) -> Tuple[List[Any], List[str], Dict[str, Any]]:
    artifacts_dir = case_out / "artifacts"
    if method == "compressed_figures":
        planes = parse_csv_list(args.figure_planes, default=("axial", "sagittal", "coronal"))
        bundle = build_compressed_figures(
            volumes,
            out_dir=artifacts_dir,
            case_id=question_id,
            planes=planes,
            slices_per_figure=int(args.slices_per_figure),
            max_figures=int(args.max_figures),
        )
        summary_lines = [
            f"compressed_figures with planes={planes}",
            f"{len(bundle.artifacts)} figure(s) created, {int(args.slices_per_figure)} slices per figure, max_figures={int(args.max_figures)}",
        ]
    elif method == "table3_api":
        planes = parse_csv_list(args.table3_planes, default=("axial",))
        bundle = build_table3_artifacts(
            volumes,
            out_dir=artifacts_dir,
            case_id=question_id,
            planes=planes,
            sample_slices=int(args.table3_sample_slices),
            delivery=str(args.table3_delivery),
        )
        summary_lines = [
            f"table3_api with planes={planes}",
            f"{len(bundle.artifacts)} artifact(s) created using {int(args.table3_sample_slices)} uniformly sampled slices per plane",
            f"delivery={args.table3_delivery}",
        ]
    else:
        raise ValueError(f"Artifact builder called for unsupported method: {method}")

    metadata = {
        "manifest_path": str(bundle.manifest_path),
        "manifest": bundle.manifest,
    }
    return bundle.artifacts, summary_lines, metadata


def _output_template_text() -> str:
    return render_prompt(PROMPT_REF, "templates.output")


def _build_system_instructions(*, method: str, uses_attached_images: bool) -> str:
    return render_prompt(
        PROMPT_REF,
        "templates.system",
        method_name=method_display_name(method),
        delivery_guidance=render_prompt(
            PROMPT_REF,
            "templates.fragments.delivery.attached_images"
            if uses_attached_images
            else "templates.fragments.delivery.external_worker",
        ),
    )


def _build_case_message(
    *,
    question: LUMIEREQuestion,
    method: str,
    artifact_summary_lines: Sequence[str],
    staged_files: Sequence[Path],
) -> str:
    series_lines = "\n".join(f"- {str(path.name)}" for path in staged_files) if staged_files else "- <none>"
    artifact_block = "\n".join(f"- {line}" for line in artifact_summary_lines) if artifact_summary_lines else "- No artifact summary available."
    return render_prompt(
        PROMPT_REF,
        "templates.user",
        question_id=question.question_id,
        patient_id=question.patient_id,
        old_timepoint=question.old_visit.timepoint,
        current_timepoint=question.current_visit.timepoint,
        method_name=method_display_name(method),
        series_lines=series_lines,
        artifact_block=artifact_block,
        loaded_volume_guide=format_loaded_volume_guide(question),
        output_template=_output_template_text(),
    )


def _build_worker_question(*, question: LUMIEREQuestion, modality: str, gap_slices: int) -> str:
    return render_prompt(
        PROMPT_REF,
        "templates.worker",
        modality=modality,
        old_timepoint=question.old_visit.timepoint,
        gap_slices=int(gap_slices),
        current_timepoint=question.current_visit.timepoint,
        output_template=_output_template_text(),
    )


def _infer_worker_evidence_from_text(text: str) -> Dict[str, Optional[bool]]:
    normalized = str(text or "")
    if not normalized.strip():
        return {field: None for field in EVIDENCE_FIELDS}

    out = _extract_evidence_predictions(normalized)
    if out.get("T2_PROGRESSION") is None and _T2_PROGRESSION_YES_RE.search(normalized):
        out["T2_PROGRESSION"] = True
    if out.get("NEW_TARGET_LESION") is None and _NEW_TARGET_YES_RE.search(normalized):
        out["NEW_TARGET_LESION"] = True
    if out.get("CURRENT_TARGET_LESION") is None and any(p.search(normalized) for p in _CURRENT_TARGET_YES_PATTERNS):
        out["CURRENT_TARGET_LESION"] = True
    if out.get("CURRENT_NON_TARGET_LESION") is None:
        if _CURRENT_NON_TARGET_NO_RE.search(normalized):
            out["CURRENT_NON_TARGET_LESION"] = False
        elif _CURRENT_NON_TARGET_YES_RE.search(normalized):
            out["CURRENT_NON_TARGET_LESION"] = True
    if out.get("NEW_NON_TARGET_LESION") is None and _NEW_NON_TARGET_YES_RE.search(normalized):
        out["NEW_NON_TARGET_LESION"] = True
    return out


def _merge_worker_evidence(texts: Sequence[str]) -> Dict[str, Optional[bool]]:
    votes: Dict[str, List[bool]] = {field: [] for field in EVIDENCE_FIELDS}
    for text in texts:
        inferred = _infer_worker_evidence_from_text(text)
        for field, value in inferred.items():
            if value is not None:
                votes[field].append(bool(value))

    merged: Dict[str, Optional[bool]] = {}
    for field in EVIDENCE_FIELDS:
        field_votes = votes[field]
        if not field_votes:
            merged[field] = None
            continue
        counts = collections.Counter(field_votes)
        merged[field] = counts[True] >= counts[False]
    return merged


def _pick_worker_rationale(texts: Sequence[str]) -> str:
    for text in texts:
        stripped = str(text or "").strip()
        if not stripped:
            continue
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        kept: List[str] = []
        for line in lines:
            upper = line.upper()
            if upper.startswith("ANSWER:"):
                continue
            if upper.startswith("SEQUENCE_VOTE_SUMMARY:"):
                break
            if any(upper.startswith(f"{field}:") for field in EVIDENCE_FIELDS):
                continue
            kept.append(line)
        if kept:
            return " ".join(kept)
    return "Most likely answer based on the paired longitudinal volume comparison."


def _normalize_m3d_output_text(*, raw_text: str, fallback_choice: Optional[str], worker_outputs: Sequence[Dict[str, Any]]) -> str:
    worker_texts = [str(item.get("text") or "") for item in worker_outputs]
    choice = extract_choice_from_set(raw_text, valid_choices=ANSWER_CHOICES) or fallback_choice
    if choice is None:
        choice = "B"

    merged_evidence = _merge_worker_evidence([raw_text, *worker_texts])
    rationale = _pick_worker_rationale([raw_text, *worker_texts])
    lines = [rationale, ""]
    for field in EVIDENCE_FIELDS:
        value = merged_evidence.get(field)
        token = "YES" if value is True else "NO"
        lines.append(f"{field}: {token}")
    lines.append(f"ANSWER: {choice}")
    return "\n".join(lines).strip()


def _build_pair_volume(
    *,
    old_path: Path,
    current_path: Path,
    out_path: Path,
    gap_slices: int,
) -> Path:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Building LUMIERE worker pair volumes requires nibabel.") from e

    old_img = nib.load(str(old_path))
    cur_img = nib.load(str(current_path))
    old_arr = np.asarray(old_img.get_fdata(dtype=np.float32))
    cur_arr = np.asarray(cur_img.get_fdata(dtype=np.float32))
    old_arr = np.squeeze(old_arr)
    cur_arr = np.squeeze(cur_arr)
    if old_arr.ndim != 3 or cur_arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI pair volumes, got old={list(old_arr.shape)} current={list(cur_arr.shape)}")
    if old_arr.shape != cur_arr.shape:
        raise ValueError(
            f"OLD and CURRENT shapes differ for pair volume creation: old={list(old_arr.shape)} current={list(cur_arr.shape)}"
        )

    gap = np.zeros((old_arr.shape[0], old_arr.shape[1], max(0, int(gap_slices))), dtype=np.float32)
    pair_arr = np.concatenate([old_arr, gap, cur_arr], axis=2).astype(np.float32, copy=False)

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(pair_arr, np.asarray(old_img.affine, dtype=np.float64))
    nib.save(img, str(out_path))
    return out_path


def _build_pair_inputs(
    *,
    question: LUMIEREQuestion,
    staged,
    modalities: Sequence[str],
    out_dir: Path,
    gap_slices: int,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for modality in modalities:
        old_path = staged.old_files.get(modality)
        current_path = staged.current_files.get(modality)
        if old_path is None or current_path is None:
            continue
        token = safe_token(f"{question.question_id}_{modality}_pair")
        pair_path = _build_pair_volume(
            old_path=Path(old_path),
            current_path=Path(current_path),
            out_path=out_dir / f"{token}.nii.gz",
            gap_slices=gap_slices,
        )
        pairs.append(
            {
                "modality": modality,
                "old_path": str(old_path),
                "current_path": str(current_path),
                "pair_path": str(pair_path),
            }
        )
    if not pairs:
        raise RuntimeError(f"No paired modalities available for question {question.question_id}")
    return pairs


def _run_m3d_method(
    *,
    question: LUMIEREQuestion,
    pair_inputs: Sequence[Dict[str, Any]],
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
) -> Dict[str, Any]:
    worker_script = Path(__file__).resolve().parent / "workers" / "m3d_worker.py"
    outputs: List[Dict[str, Any]] = []
    for item in pair_inputs:
        modality = str(item["modality"])
        pair_path = Path(str(item["pair_path"])).expanduser().resolve()
        work_dir = case_out / "workers" / "m3d" / safe_token(modality)
        request = {
            "nifti_path": str(pair_path),
            "question": _build_worker_question(question=question, modality=modality, gap_slices=int(args.pair_gap_slices)),
            "model_name_or_path": args.m3d_model_name_or_path,
            "device": args.m3d_device,
            "dtype": args.m3d_dtype,
            "max_new_tokens": int(args.m3d_max_new_tokens),
        }
        payload = run_external_worker(
            worker_script=worker_script,
            request_payload=request,
            out_dir=work_dir,
            python_executable=args.m3d_python or args.worker_python,
            timeout_sec=float(args.worker_timeout_sec),
            session=session,
            event_prefix="track_c_m3d_worker",
            prompt_metadata=prompt_metadata(PROMPT_REF),
        )
        if not bool(payload.get("ok", False)):
            raise ExternalWorkerError(str(payload.get("error") or "M3D worker returned ok=false"))
        payload["label"] = modality
        payload["volume_name"] = modality
        outputs.append(payload)

    aggregate = aggregate_sequence_outputs(outputs, valid_choices=ANSWER_CHOICES)
    return {
        "text": str(aggregate.get("text") or ""),
        "choice": aggregate.get("choice"),
        "worker_outputs": aggregate.get("outputs") or outputs,
        "aggregation": aggregate,
    }


def _run_radfm_method(
    *,
    question: LUMIEREQuestion,
    pair_inputs: Sequence[Dict[str, Any]],
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
) -> Dict[str, Any]:
    if not args.radfm_repo_root:
        raise ValueError("--radfm_repo_root is required for --method radfm")
    if not args.radfm_checkpoint:
        raise ValueError("--radfm_checkpoint is required for --method radfm")

    worker_script = Path(__file__).resolve().parent / "workers" / "radfm_worker.py"
    outputs: List[Dict[str, Any]] = []
    for item in pair_inputs:
        modality = str(item["modality"])
        pair_path = Path(str(item["pair_path"])).expanduser().resolve()
        work_dir = case_out / "workers" / "radfm" / safe_token(modality)
        request = {
            "repo_root": args.radfm_repo_root,
            "checkpoint_path": args.radfm_checkpoint,
            "language_dir": args.radfm_language_dir,
            "nifti_path": str(pair_path),
            "question": _build_worker_question(question=question, modality=modality, gap_slices=int(args.pair_gap_slices)),
            "valid_choices": list(ANSWER_CHOICES),
            "device": args.radfm_device,
            "target_depth": int(args.radfm_target_depth),
            "max_new_tokens": int(args.radfm_max_new_tokens),
        }
        payload = run_external_worker(
            worker_script=worker_script,
            request_payload=request,
            out_dir=work_dir,
            python_executable=args.radfm_python or args.worker_python,
            timeout_sec=float(args.worker_timeout_sec),
            session=session,
            event_prefix="track_c_radfm_worker",
            prompt_metadata=prompt_metadata(PROMPT_REF),
        )
        if not bool(payload.get("ok", False)):
            raise ExternalWorkerError(str(payload.get("error") or "RadFM worker returned ok=false"))
        payload["label"] = modality
        payload["volume_name"] = modality
        outputs.append(payload)

    aggregate = aggregate_sequence_outputs(outputs, valid_choices=ANSWER_CHOICES)
    return {
        "text": str(aggregate.get("text") or ""),
        "choice": aggregate.get("choice"),
        "worker_outputs": aggregate.get("outputs") or outputs,
        "aggregation": aggregate,
    }


def _run_api_method(
    *,
    method: str,
    question: LUMIEREQuestion,
    case_out: Path,
    staged_files: Sequence[Path],
    volumes: Sequence[Any],
    args: argparse.Namespace,
    session: SessionManager,
) -> Dict[str, Any]:
    artifacts, artifact_summary_lines, artifact_meta = _build_api_artifacts_and_summary(
        method=method,
        question_id=question.question_id,
        case_out=case_out,
        volumes=volumes,
        args=args,
    )
    for artifact in artifacts:
        session.log_run_event(
            "track_c_artifact_generated",
            method=method,
            kind=artifact.kind,
            path=str(artifact.path),
            description=artifact.description,
        )

    message = _build_case_message(
        question=question,
        method=method,
        artifact_summary_lines=artifact_summary_lines,
        staged_files=staged_files,
    )
    instructions = _build_system_instructions(method=method, uses_attached_images=True)
    model_result = run_multimodal_prompt(
        provider=args.provider,
        model=args.model,
        instructions=instructions,
        user_message=message,
        artifacts=artifacts,
        reasoning_effort=args.reasoning_effort,
        timeout_sec=float(args.api_timeout_sec),
        retry_on_timeout=1,
        session=session,
        prompt_metadata=prompt_metadata(PROMPT_REF),
    )
    return {
        "text": str(model_result.get("text") or ""),
        "choice": extract_choice_from_set(str(model_result.get("text") or ""), valid_choices=ANSWER_CHOICES),
        "provider": model_result.get("provider"),
        "response_id": model_result.get("response_id"),
        "artifact_summary": artifact_summary_lines,
        "artifact_meta": artifact_meta,
        "unexpected_tool_calls": model_result.get("unexpected_tool_calls") or [],
    }


def _method_config_summary(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {"method": method, "modalities": _selected_modalities(args.lumiere_modalities), "pair_gap_slices": int(args.pair_gap_slices)}
    if method == "compressed_figures":
        base.update(
            {
                "figure_planes": parse_csv_list(args.figure_planes),
                "slices_per_figure": int(args.slices_per_figure),
                "max_figures": int(args.max_figures),
                "provider": args.provider,
                "model": args.model,
            }
        )
        return base
    if method == "table3_api":
        base.update(
            {
                "table3_planes": parse_csv_list(args.table3_planes),
                "table3_sample_slices": int(args.table3_sample_slices),
                "table3_delivery": str(args.table3_delivery),
                "provider": args.provider,
                "model": args.model,
            }
        )
        return base
    if method == "m3d":
        base.update(
            {
                "m3d_model_name_or_path": args.m3d_model_name_or_path,
                "m3d_device": args.m3d_device,
                "m3d_dtype": args.m3d_dtype,
                "m3d_max_new_tokens": int(args.m3d_max_new_tokens),
                "m3d_python": args.m3d_python or args.worker_python,
            }
        )
        return base
    if method == "radfm":
        base.update(
            {
                "radfm_repo_root": args.radfm_repo_root,
                "radfm_checkpoint": args.radfm_checkpoint,
                "radfm_language_dir": args.radfm_language_dir,
                "radfm_device": args.radfm_device,
                "radfm_target_depth": int(args.radfm_target_depth),
                "radfm_max_new_tokens": int(args.radfm_max_new_tokens),
                "radfm_python": args.radfm_python or args.worker_python,
            }
        )
        return base
    return base


def main() -> None:
    args = build_parser().parse_args()
    method = normalize_method_name(args.method)

    data_root = Path(args.data_root).expanduser().resolve()
    ratings_csv = _resolve_csv_path(data_root, args.ratings_csv)
    datacompleteness_csv = _resolve_csv_path(data_root, args.datacompleteness_csv)
    modalities = _selected_modalities(args.lumiere_modalities)

    run_name = args.model if method in {"compressed_figures", "table3_api"} else method
    log_dir = resolve_run_output_dir(args.log_dir, model=run_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark_track_c.log")
    log = logging.getLogger("benchmark.lumiere.track_c")

    question_bank = build_question_bank(
        data_root=data_root,
        ratings_csv=ratings_csv,
        datacompleteness_csv=datacompleteness_csv,
        limit_patient=args.limit_patient if args.limit_patient and args.limit_patient > 0 else None,
        limit_question=None,
        balance_rano_labels=False,
    )
    min_scorable_evidence_fields = max(0, int(args.min_scorable_evidence_fields or 0))
    questions_before_evidence_filter = list(question_bank.questions)
    questions = questions_before_evidence_filter
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
    questions, n_filtered_low_count_labels, selected_balance_labels, per_label_counts_after_evidence = (
        _filter_labels_with_min_count(
            questions,
            min_count=min_label_count_before_balance,
        )
    )
    questions_before_balance_filter = list(questions)
    if bool(args.balance_rano_labels):
        questions = _balance_questions_by_label(questions)
    questions_before_final_limit = list(questions)
    if args.limit_question and args.limit_question > 0:
        questions = questions[: int(args.limit_question)]
    bank_stats = dict(question_bank.stats)
    bank_stats["min_scorable_evidence_fields"] = min_scorable_evidence_fields
    bank_stats["n_questions_before_evidence_filter"] = len(questions_before_evidence_filter)
    bank_stats["n_questions_after_evidence_filter"] = len(questions_before_label_count_filter)
    bank_stats["n_filtered_insufficient_scorable_evidence"] = n_filtered_insufficient_scorable_evidence
    bank_stats["n_patients_after_evidence_filter"] = len({q.patient_id for q in questions_before_label_count_filter})
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
    log.info("Track C method=%s (%s)", method, method_display_name(method))
    if min_scorable_evidence_fields > 0:
        log.info(
            "Filtering LUMIERE questions to those with at least %d scorable ReportTable evidence fields: kept %d / %d.",
            min_scorable_evidence_fields,
            len(questions_before_label_count_filter),
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
            "Applying final LUMIERE question limit after evidence/balance filtering: kept %d / %d.",
            len(questions),
            len(questions_before_final_limit),
        )
    log.info("Prepared %d LUMIERE questions from %s", len(questions), data_root)

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_failed = 0

    for idx, question in enumerate(questions, start=1):
        t0 = time.time()
        case_out = log_dir / question.question_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.lumiere.track_c.{question.question_id}")
        _print_progress(f"Processing question {idx}/{len(questions)}: {question.question_id}")
        case_log.info("Processing question %d/%d: %s", idx, len(questions), question.question_id)

        staged = None
        gt_evidence, gt_evidence_source = _infer_evidence_ground_truth(question)
        session = SessionManager(out_dir=case_out)
        session.update_state(case_preset="track_c_lumiere", loaded_series_display_names=[], loaded_series_dirs=[], active_tool_packs=[])
        session.log_run_event(
            "case_script_start",
            script=SCRIPT_NAME,
            case_id=question.question_id,
            patient_id=question.patient_id,
            method=method,
            method_display_name=method_display_name(method),
            method_config=_method_config_summary(args, method),
        )

        try:
            staged = stage_question_inputs(question, case_out / "staged_inputs")
            staging_manifest = {
                "question_id": question.question_id,
                "patient_id": question.patient_id,
                "old_timepoint": question.old_visit.timepoint,
                "current_timepoint": question.current_visit.timepoint,
                "all_files": [str(path) for path in staged.all_files],
                "old_files": {k: str(v) for k, v in staged.old_files.items()},
                "current_files": {k: str(v) for k, v in staged.current_files.items()},
                "selected_modalities": modalities,
            }
            (case_out / "question_manifest.json").write_text(json.dumps(staging_manifest, indent=2), encoding="utf-8")
            session.update_state(
                loaded_series_display_names=[path.name for path in staged.all_files],
                loaded_series_dirs=[str(path) for path in staged.all_files],
            )

            if method in {"compressed_figures", "table3_api"}:
                volumes = load_case_volumes(staged.all_files)
                planes_for_summary = parse_csv_list(args.figure_planes) if method == "compressed_figures" else parse_csv_list(args.table3_planes)
                volume_summary = build_volume_summary(volumes, planes=planes_for_summary)
                (case_out / "volume_summary.json").write_text(json.dumps(volume_summary, indent=2), encoding="utf-8")
                session.log_run_event("track_c_volumes_loaded", n_volumes=len(volumes), volume_summary=volume_summary)
                run_result = _run_api_method(
                    method=method,
                    question=question,
                    case_out=case_out,
                    staged_files=staged.all_files,
                    volumes=volumes,
                    args=args,
                    session=session,
                )
            elif method == "m3d":
                pair_inputs = _build_pair_inputs(
                    question=question,
                    staged=staged,
                    modalities=modalities,
                    out_dir=case_out / "worker_pairs",
                    gap_slices=int(args.pair_gap_slices),
                )
                (case_out / "pair_inputs.json").write_text(json.dumps(pair_inputs, indent=2), encoding="utf-8")
                session.log_run_event("track_c_pair_inputs_prepared", pair_inputs=pair_inputs)
                run_result = _run_m3d_method(
                    question=question,
                    pair_inputs=pair_inputs,
                    case_out=case_out,
                    args=args,
                    session=session,
                )
            elif method == "radfm":
                pair_inputs = _build_pair_inputs(
                    question=question,
                    staged=staged,
                    modalities=modalities,
                    out_dir=case_out / "worker_pairs",
                    gap_slices=int(args.pair_gap_slices),
                )
                (case_out / "pair_inputs.json").write_text(json.dumps(pair_inputs, indent=2), encoding="utf-8")
                session.log_run_event("track_c_pair_inputs_prepared", pair_inputs=pair_inputs)
                run_result = _run_radfm_method(
                    question=question,
                    pair_inputs=pair_inputs,
                    case_out=case_out,
                    args=args,
                    session=session,
                )
            else:  # pragma: no cover - normalize_method_name already guards this.
                raise ValueError(f"Unsupported Track C method: {method}")

            raw_text = str(run_result.get("text") or "")
            if method in {"m3d", "radfm"}:
                raw_text = _normalize_m3d_output_text(
                    raw_text=raw_text,
                    fallback_choice=run_result.get("choice"),
                    worker_outputs=run_result.get("worker_outputs") or [],
                )
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")

            gt_choice = question.gt_choice
            gt_rano_label = question.gt_rano_label
            pred = extract_choice_from_set(raw_text, valid_choices=ANSWER_CHOICES) or run_result.get("choice")
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

            tool_usage = _zero_tool_usage()
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
                "method": method,
                "method_display_name": method_display_name(method),
                "method_config": _method_config_summary(args, method),
                "method_result": run_result,
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script=SCRIPT_NAME,
                case_id=question.question_id,
                patient_id=question.patient_id,
                ok=True,
                prediction=pred,
                ground_truth=gt_choice,
                ground_truth_rano_label=gt_rano_label,
                correct=correct,
                scored=scored,
                elapsed_sec=rec["seconds"],
                n_evidence_fields_scored=n_evidence_scored,
                n_evidence_fields_correct=n_evidence_correct,
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.info(
                "Completed question %d/%d: %s | method=%s | pred=%s | gt=%s (%s) | correct=%s | evidence=%d/%d",
                idx,
                len(questions),
                question.question_id,
                method,
                pred,
                gt_choice,
                gt_rano_label,
                correct,
                n_evidence_correct,
                n_evidence_scored,
            )
            case_log.info("Model answer for %s:\n%s", question.question_id, raw_text)
            _print_progress(
                f"Completed question {idx}/{len(questions)}: {question.question_id} | method={method} | pred={pred} | gt={gt_choice} ({gt_rano_label}) | evidence={n_evidence_correct}/{n_evidence_scored}"
            )
        except Exception as e:
            n_failed += 1
            gt_choice = question.gt_choice
            gt_rano_label = question.gt_rano_label
            n_scored += 1
            tool_usage = _zero_tool_usage()
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
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
                "method": method,
                "method_display_name": method_display_name(method),
                "method_config": _method_config_summary(args, method),
                "error": str(e),
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script=SCRIPT_NAME,
                case_id=question.question_id,
                patient_id=question.patient_id,
                ok=False,
                error=str(e),
                elapsed_sec=rec["seconds"],
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Question failed: %s", e)
            _print_progress(
                f"Completed question {idx}/{len(questions)}: {question.question_id} | method={method} | pred=<error> | gt={gt_choice} ({gt_rano_label})"
            )

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    class_metrics = _per_class_metrics(results)
    evidence_metrics = _aggregate_evidence_metrics(results)

    summary = {
        "script": SCRIPT_NAME,
        "method": method,
        "method_display_name": method_display_name(method),
        "method_config": _method_config_summary(args, method),
        "data_root": str(data_root),
        "ratings_csv": str(ratings_csv),
        "datacompleteness_csv": str(datacompleteness_csv),
        "image_space": "deepbratumia_atlas",
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
        "total_tool_calls": 0,
        "average_tool_calls_per_question": 0.0,
        "tool_call_counts": {},
        "tools_called": [],
        "n_unique_tools_used": 0,
        "limit_patient": int(args.limit_patient or 0),
        "limit_question": int(args.limit_question or 0),
        "balance_rano_labels": bool(args.balance_rano_labels),
        "min_scorable_evidence_fields": min_scorable_evidence_fields,
        "min_label_count_before_balance": min_label_count_before_balance,
        "question_bank": bank_stats,
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_lumiere_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info(
        "DONE. method=%s accuracy=%.4f (%d/%d) failed=%d evidence_micro=%.4f",
        method,
        accuracy,
        n_correct,
        n_scored,
        n_failed,
        evidence_metrics["micro_accuracy"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
