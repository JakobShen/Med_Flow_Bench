#!/usr/bin/env python3
"""Track3 / Open-Method benchmark runner for UCSF-PDGM.

This runner reuses the benchmark dataloader, answer extraction, and
deterministic judging code while allowing Track C methods to operate on raw
NIfTI volumes or derived image representations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from _script_support import configure_logging, default_model_from_env, ensure_repo_imports, resolve_run_output_dir

ensure_repo_imports()

from slicer_agent_engine.benchmarking.check_segment import SegmentGroundTruth, evaluate_segment_prediction
from slicer_agent_engine.benchmarking.judge import extract_choice_from_set
from slicer_agent_engine.benchmarking.paper_table import build_ucsf_pdgm_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.ucsf_pdgm import diagnosis_subtype, iter_cases, summarize_subtype_metrics
from slicer_agent_engine.session import SessionManager
from slicer_agent_engine.track3 import (
    ExternalWorkerError,
    aggregate_sequence_outputs,
    build_case_message,
    build_compressed_figures,
    build_system_instructions,
    build_table3_artifacts,
    build_volume_summary,
    load_case_volumes,
    method_display_name,
    normalize_method_name,
    parse_csv_list,
    prepare_nifti_segment_ground_truth_py,
    run_external_worker,
    run_multimodal_prompt,
    safe_token,
)


EVIDENCE_VIEWS = ["axial", "sagittal", "coronal"]
SCRIPT_NAME = "benchmark_UCSF_PDGM_track3.py"


def _ucsf_pdgm_evidence_exact_correct(segment_check: Dict[str, Any]) -> bool:
    key_slice_payload = segment_check.get("key_slice") or {}
    return bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UCSF-PDGM Track3 benchmark.")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing UCSF-PDGM-*_nifti folders")
    p.add_argument("--csv_path", type=str, default="UCSF-PDGM-metadata_v5.csv", help="Metadata CSV path")
    p.add_argument("--limit", type=int, default=100, help="Run only the first N cases (default: 100, 0 = all)")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_ucsf_pdgm_track3", help="Directory for outputs")
    p.add_argument("--method", type=str, default="table3_api", help="Track3 method: compressed_figures | table3_api | m3d | radfm")
    p.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "auto"),
        help="LLM provider for API-backed Track3 methods (compressed_figures/table3_api)",
    )
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model for API-backed Track3 methods")
    p.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None,
        help="Optional provider reasoning-effort hint for API-backed Track3 methods.",
    )
    p.add_argument("--include_extra", action="store_true", help="Load all non-seg NIfTI volumes (not just T1/T1c/T2/FLAIR)")
    p.add_argument("--no_prefer_bias", action="store_true", help="Do not prefer *_bias.nii.gz when selecting modalities")
    p.add_argument(
        "--balance_diagnosis_subtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance UCSF-PDGM sampling across Glioblastoma, Oligodendroglioma, and Astrocytoma when possible.",
    )
    p.add_argument(
        "--evaluate_segment_evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a hidden segmentation exists, ask the model for the EVIDENCE block and score it deterministically.",
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
        help="Whether to send sampled slices as per-sequence montages or as separate images.",
    )

    p.add_argument("--worker_python", type=str, default=None, help="Default Python executable for external Track3 workers")

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
        "--m3d_sequence_strategy",
        type=str,
        choices=["primary", "all_vote"],
        default="primary",
        help="How to handle multi-sequence UCSF cases with M3D.",
    )

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
    p.add_argument(
        "--radfm_sequence_strategy",
        type=str,
        choices=["primary", "all_vote"],
        default="primary",
        help="How to handle multi-sequence UCSF cases with RadFM.",
    )
    return p


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _zero_tool_usage() -> Dict[str, Any]:
    return {
        "n_tool_calls": 0,
        "n_unique_tools": 0,
        "tools_called": [],
        "tool_call_counts": {},
    }


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


def _prepare_segment_gt(
    *,
    out_dir: Path,
    case_path: Path,
    segment_path: Optional[Path],
    reference_volume_paths: Sequence[Path],
) -> Tuple[Optional[SegmentGroundTruth], Optional[str]]:
    if segment_path is None:
        return None, None
    try:
        gt = prepare_nifti_segment_ground_truth_py(
            case_path=case_path,
            out_dir=out_dir,
            segment_path=segment_path,
            reference_volume_paths=reference_volume_paths,
            source_modality="MR",
            provenance_policy="not_applicable",
        )
        return gt, None
    except Exception as e:
        return None, str(e)


def _build_api_artifacts_and_summary(
    *,
    method: str,
    case_id: str,
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
            case_id=case_id,
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
            case_id=case_id,
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


def _select_sequence_paths(paths: Sequence[Path], strategy: str) -> List[Path]:
    strategy_l = str(strategy).strip().lower()
    if strategy_l == "primary":
        return [paths[0]] if paths else []
    if strategy_l == "all_vote":
        return list(paths)
    raise ValueError(f"Unsupported sequence strategy: {strategy!r}")


def _build_worker_question(*, sequence_name: str, include_evidence: bool) -> str:
    evidence_block = ""
    if include_evidence:
        evidence_block = (
            "If possible, include the following block exactly once before the final answer line:\n"
            "EVIDENCE:\n"
            "KEY_SLICE_AXIAL_MM: <float>\n"
            "KEY_SLICE_SAGITTAL_MM: <float>\n"
            "KEY_SLICE_CORONAL_MM: <float>\n"
            "RSA: [<R>, <A>, <S>]\n"
        )
    return (
        f"Sequence under review: {sequence_name}\n"
        "Brain MRI diagnosis task. Choose exactly one answer:\n"
        "A) Glioblastoma\n"
        "B) Oligodendroglioma / Astrocytoma\n"
        "C) No tumor\n\n"
        + evidence_block
        + "Write a brief rationale and end with a final line exactly formatted as: ANSWER: <A|B|C>."
    )


def _run_m3d_method(
    *,
    case,
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
    include_evidence: bool,
) -> Dict[str, Any]:
    sequence_paths = _select_sequence_paths(case.nifti_files, args.m3d_sequence_strategy)
    worker_script = Path(__file__).resolve().parent / "track3_workers" / "m3d_worker.py"
    outputs: List[Dict[str, Any]] = []
    for path in sequence_paths:
        work_dir = case_out / "workers" / "m3d" / safe_token(path.name)
        request = {
            "nifti_path": str(path),
            "question": _build_worker_question(sequence_name=path.name, include_evidence=include_evidence),
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
            event_prefix="track3_m3d_worker",
        )
        if not bool(payload.get("ok", False)):
            raise ExternalWorkerError(str(payload.get("error") or "M3D worker returned ok=false"))
        payload["label"] = path.name
        payload["volume_name"] = path.name
        outputs.append(payload)

    aggregate = aggregate_sequence_outputs(outputs, valid_choices=("A", "B", "C"))
    return {
        "text": str(aggregate.get("text") or ""),
        "choice": aggregate.get("choice"),
        "worker_outputs": aggregate.get("outputs") or outputs,
        "aggregation": aggregate,
    }


def _run_radfm_method(
    *,
    case,
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
    include_evidence: bool,
) -> Dict[str, Any]:
    if not args.radfm_repo_root:
        raise ValueError("--radfm_repo_root is required for --method radfm")
    if not args.radfm_checkpoint:
        raise ValueError("--radfm_checkpoint is required for --method radfm")

    sequence_paths = _select_sequence_paths(case.nifti_files, args.radfm_sequence_strategy)
    worker_script = Path(__file__).resolve().parent / "track3_workers" / "radfm_worker.py"
    outputs: List[Dict[str, Any]] = []
    for path in sequence_paths:
        work_dir = case_out / "workers" / "radfm" / safe_token(path.name)
        request = {
            "repo_root": args.radfm_repo_root,
            "checkpoint_path": args.radfm_checkpoint,
            "language_dir": args.radfm_language_dir,
            "nifti_path": str(path),
            "question": _build_worker_question(sequence_name=path.name, include_evidence=include_evidence),
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
            event_prefix="track3_radfm_worker",
        )
        if not bool(payload.get("ok", False)):
            raise ExternalWorkerError(str(payload.get("error") or "RadFM worker returned ok=false"))
        payload["label"] = path.name
        payload["volume_name"] = path.name
        outputs.append(payload)

    aggregate = aggregate_sequence_outputs(outputs, valid_choices=("A", "B", "C"))
    return {
        "text": str(aggregate.get("text") or ""),
        "choice": aggregate.get("choice"),
        "worker_outputs": aggregate.get("outputs") or outputs,
        "aggregation": aggregate,
    }


def _run_api_method(
    *,
    method: str,
    case,
    case_out: Path,
    volumes: Sequence[Any],
    segment_gt: Optional[SegmentGroundTruth],
    args: argparse.Namespace,
    session: SessionManager,
) -> Dict[str, Any]:
    artifacts, artifact_summary_lines, artifact_meta = _build_api_artifacts_and_summary(
        method=method,
        case_id=case.case_id,
        case_out=case_out,
        volumes=volumes,
        args=args,
    )
    for artifact in artifacts:
        session.log_run_event(
            "track3_artifact_generated",
            method=method,
            kind=artifact.kind,
            path=str(artifact.path),
            description=artifact.description,
        )

    message = build_case_message(
        case_id=case.case_id,
        case_path=case.case_dir,
        nifti_files=case.nifti_files,
        method=method,
        artifact_summary_lines=artifact_summary_lines,
        segment_evidence=(segment_gt.to_prompt_spec() if segment_gt is not None and segment_gt.has_segment and bool(args.evaluate_segment_evidence) else None),
    )
    instructions = build_system_instructions(
        method=method,
        uses_attached_images=True,
        include_evidence=bool(segment_gt is not None and segment_gt.has_segment and bool(args.evaluate_segment_evidence)),
    )
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
    )
    return {
        "text": str(model_result.get("text") or ""),
        "choice": extract_choice_from_set(str(model_result.get("text") or ""), valid_choices=("A", "B", "C")),
        "provider": model_result.get("provider"),
        "response_id": model_result.get("response_id"),
        "artifact_summary": artifact_summary_lines,
        "artifact_meta": artifact_meta,
        "unexpected_tool_calls": model_result.get("unexpected_tool_calls") or [],
    }


def _method_config_summary(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    if method == "compressed_figures":
        return {
            "method": method,
            "figure_planes": parse_csv_list(args.figure_planes),
            "slices_per_figure": int(args.slices_per_figure),
            "max_figures": int(args.max_figures),
            "provider": args.provider,
            "model": args.model,
        }
    if method == "table3_api":
        return {
            "method": method,
            "table3_planes": parse_csv_list(args.table3_planes),
            "table3_sample_slices": int(args.table3_sample_slices),
            "table3_delivery": str(args.table3_delivery),
            "provider": args.provider,
            "model": args.model,
        }
    if method == "m3d":
        return {
            "method": method,
            "m3d_model_name_or_path": args.m3d_model_name_or_path,
            "m3d_device": args.m3d_device,
            "m3d_dtype": args.m3d_dtype,
            "m3d_max_new_tokens": int(args.m3d_max_new_tokens),
            "m3d_sequence_strategy": args.m3d_sequence_strategy,
            "m3d_python": args.m3d_python or args.worker_python,
        }
    if method == "radfm":
        return {
            "method": method,
            "radfm_repo_root": args.radfm_repo_root,
            "radfm_checkpoint": args.radfm_checkpoint,
            "radfm_language_dir": args.radfm_language_dir,
            "radfm_device": args.radfm_device,
            "radfm_target_depth": int(args.radfm_target_depth),
            "radfm_max_new_tokens": int(args.radfm_max_new_tokens),
            "radfm_sequence_strategy": args.radfm_sequence_strategy,
            "radfm_python": args.radfm_python or args.worker_python,
        }
    return {"method": method}


def main() -> None:
    args = build_parser().parse_args()
    method = normalize_method_name(args.method)

    data_root = Path(args.data_root).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = (data_root / csv_path).resolve()

    run_name = args.model if method in {"compressed_figures", "table3_api"} else method
    log_dir = resolve_run_output_dir(args.log_dir, model=run_name)
    configure_logging(log_dir, log_name="benchmark_track3.log")
    log = logging.getLogger("benchmark.track3")

    cases = list(
        iter_cases(
            data_root,
            csv_path,
            limit=(args.limit if args.limit > 0 else None),
            include_extra=bool(args.include_extra),
            include_segmentations=False,
            prefer_bias=not bool(args.no_prefer_bias),
            balance_diagnosis_subtypes=bool(args.balance_diagnosis_subtypes),
        )
    )
    log.info("Track3 method=%s (%s)", method, method_display_name(method))
    log.info("Found %d cases under %s", len(cases), data_root)

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_correct_with_evidence = 0
    n_localization_correct = 0
    n_failed = 0

    key_slice_scored = {view: 0 for view in EVIDENCE_VIEWS}
    key_slice_correct = {view: 0 for view in EVIDENCE_VIEWS}
    rsa_metrics = {"n_scored": 0, "n_correct": 0}
    provenance_metrics = {"n_scored": 0, "n_correct": 0}
    evidence_case_metrics = {
        "n_cases_with_segment_gt": 0,
        "n_cases_with_geometry_exact_correct": 0,
        "n_cases_with_evidence_exact_correct": 0,
    }

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.track3.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)

        session = SessionManager(out_dir=case_out)
        session.update_state(
            case_preset="track3_ucsf_pdgm",
            loaded_series_display_names=[p.name for p in case.nifti_files],
            loaded_series_dirs=[str(p.parent) for p in case.nifti_files],
            active_tool_packs=[],
        )
        session.log_run_event(
            "case_script_start",
            script=SCRIPT_NAME,
            case_id=case.case_id,
            method=method,
            method_display_name=method_display_name(method),
            method_config=_method_config_summary(args, method),
        )

        segment_gt_dir = case_out / "segment_gt"
        segment_gt, segment_gt_error = _prepare_segment_gt(
            out_dir=segment_gt_dir,
            case_path=case.case_dir,
            segment_path=case.segmentation_file,
            reference_volume_paths=case.nifti_files,
        )
        if segment_gt_error:
            case_log.warning("Segment GT discovery failed for %s: %s", case.case_id, segment_gt_error)

        try:
            volumes = load_case_volumes(case.nifti_files) if method in {"compressed_figures", "table3_api"} else []
            if volumes:
                volume_summary = build_volume_summary(
                    volumes,
                    planes=(parse_csv_list(args.figure_planes) if method == "compressed_figures" else parse_csv_list(args.table3_planes)),
                )
                (case_out / "volume_summary.json").write_text(json.dumps(volume_summary, indent=2), encoding="utf-8")
                session.log_run_event("track3_volumes_loaded", n_volumes=len(volumes), volume_summary=volume_summary)

            if method in {"compressed_figures", "table3_api"}:
                run_result = _run_api_method(
                    method=method,
                    case=case,
                    case_out=case_out,
                    volumes=volumes,
                    segment_gt=segment_gt,
                    args=args,
                    session=session,
                )
            elif method == "m3d":
                run_result = _run_m3d_method(
                    case=case,
                    case_out=case_out,
                    args=args,
                    session=session,
                    include_evidence=bool(segment_gt is not None and segment_gt.has_segment and bool(args.evaluate_segment_evidence)),
                )
            elif method == "radfm":
                run_result = _run_radfm_method(
                    case=case,
                    case_out=case_out,
                    args=args,
                    session=session,
                    include_evidence=bool(segment_gt is not None and segment_gt.has_segment and bool(args.evaluate_segment_evidence)),
                )
            else:  # pragma: no cover - normalize_method_name already guards this.
                raise ValueError(f"Unsupported Track3 method: {method}")

            raw_text = str(run_result.get("text") or "")
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")

            pred = extract_choice_from_set(raw_text, valid_choices=("A", "B", "C"))
            gt = case.label
            scored = gt in {"A", "B", "C"}
            correct = bool(scored and pred == gt)
            if scored:
                n_scored += 1
                n_correct += int(correct)

            segment_check = (
                evaluate_segment_prediction(prediction=raw_text, gt=segment_gt, trace_path=session.trace_path)
                if segment_gt is not None
                else {"available": False, "has_segment_gt": False, "segment_gt_error": segment_gt_error}
            )
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
            localization_correct = bool((segment_check.get("rsa") or {}).get("correct", False)) if has_segment_gt else None
            correct_with_evidence = bool(correct and key_slice_all_correct) if has_segment_gt else None
            if bool(correct_with_evidence):
                n_correct_with_evidence += 1
            if bool(localization_correct):
                n_localization_correct += 1

            tool_usage = _zero_tool_usage()
            rec = {
                "case_id": case.case_id,
                "csv_id": case.csv_id,
                "diagnosis": case.diagnosis,
                "diagnosis_subtype": diagnosis_subtype(case.diagnosis),
                "gt": gt,
                "pred": pred,
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
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
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

            case_log.info(
                "Completed case %d/%d: %s | method=%s | pred=%s | gt=%s | correct=%s | correct_with_evidence=%s | localization_correct=%s",
                idx,
                len(cases),
                case.case_id,
                method,
                pred,
                gt,
                correct,
                correct_with_evidence,
                localization_correct,
            )
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | method={method} | pred={pred} | gt={gt} | "
                f"evidence_correct={correct_with_evidence} | localization_correct={localization_correct}"
            )
        except Exception as e:
            n_failed += 1
            gt = case.label
            scored = gt in {"A", "B", "C"}
            if scored:
                n_scored += 1

            segment_check = (
                evaluate_segment_prediction(prediction="", gt=segment_gt, trace_path=session.trace_path)
                if segment_gt is not None
                else {"available": False, "has_segment_gt": False, "segment_gt_error": segment_gt_error}
            )
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
            localization_correct = bool((segment_check.get("rsa") or {}).get("correct", False)) if has_segment_gt else None
            correct_with_evidence = False if has_segment_gt else None
            if bool(localization_correct):
                n_localization_correct += 1

            tool_usage = _zero_tool_usage()
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
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(f"Completed case {idx}/{len(cases)}: {case.case_id} | method={method} | pred=<error> | gt={gt}")

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    accuracy_with_evidence = (
        n_correct_with_evidence / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
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
        "script": SCRIPT_NAME,
        "method": method,
        "method_display_name": method_display_name(method),
        "method_config": _method_config_summary(args, method),
        "data_root": str(data_root),
        "csv_path": str(csv_path),
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
        "total_tool_calls": 0,
        "average_tool_calls_per_case": 0.0,
        "tool_call_counts": {},
        "tools_called": [],
        "n_unique_tools_used": 0,
        "balance_diagnosis_subtypes": bool(args.balance_diagnosis_subtypes),
        "per_subtype": per_subtype,
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_ucsf_pdgm_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info(
        "DONE. method=%s accuracy=%.4f (%d/%d) accuracy_with_evidence=%.4f (%d/%d segment-gt) localization_acc=%.4f (%d/%d segment-gt) failed=%d",
        method,
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
