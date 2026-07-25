#!/usr/bin/env python3
"""Track C / Open-Method benchmark runner for NSCLC Radiogenomics.

This runner reuses the NSCLC case iterator and deterministic question scoring
while allowing Track C methods to operate on raw DICOM series or derived image
representations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys

SCRIPT_SUPPORT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_SUPPORT_ROOT))

from _script_support import (
    configure_logging,
    default_model_from_env,
    ensure_repo_imports,
    resolve_run_output_dir,
)

ensure_repo_imports()

from medflow_prompts import prompt_metadata, render_prompt
from slicer_agent_engine.benchmarking.check_segment import (
    SegmentGroundTruth,
    discover_nifti_segmentation_file,
    evaluate_segment_prediction,
)
from slicer_agent_engine.benchmarking.judge import extract_choice_from_set, extract_numbered_answers
from slicer_agent_engine.benchmarking.paper_table import build_nsclc_paper_table_data, write_paper_table_data
from slicer_agent_engine.benchmarking.nsclc import (
    EVIDENCE_VIEWS,
    GRADE_OPTIONS,
    HISTOLOGY_OPTIONS,
    LOCATION_OPTIONS,
    N_STAGE_OPTIONS,
    QUESTION_IDS,
    QUESTION_NAMES,
    T_STAGE_OPTIONS,
    _compact_from_map,
    _emit_segment_debug,
    _score_questions,
    _segment_gt_summary,
    _split_numbered_answers,
    _update_segment_metrics,
    iter_cases,
)
from slicer_agent_engine.load_helper import SeriesRecord
from slicer_agent_engine.session import SessionManager
from radiology_track_c import (
    ExternalWorkerError,
    TrackCConfigError,
    build_compressed_figures,
    build_table3_artifacts,
    build_volume_summary,
    method_display_name,
    normalize_method_name,
    parse_csv_list,
    prepare_dicom_segment_ground_truth_py,
    run_external_worker,
    run_multimodal_prompt,
    safe_token,
    prepare_nifti_segment_ground_truth_py,
)
from radiology_track_c.volumes import LoadedVolume


SCRIPT_NAME = "benchmark_NSCLC_track_c.py"
PROMPT_REF = "track_c/nsclc.json"
ANSWER_FORMAT = "1A2B3C4D5E"
QUESTION_OPTION_MAPS: Dict[int, Dict[str, str]] = {
    1: LOCATION_OPTIONS,
    2: T_STAGE_OPTIONS,
    3: N_STAGE_OPTIONS,
    4: HISTOLOGY_OPTIONS,
    5: GRADE_OPTIONS,
}


def _nsclc_track_c_evidence_exact_correct(segment_check: Dict[str, Any]) -> bool:
    key_slice_payload = segment_check.get("key_slice") or {}
    return bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))


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
    p = argparse.ArgumentParser(description="Run NSCLC Track C benchmark with hidden-seg evidence scoring.")
    p.add_argument("--data_root", type=str, default=".", help="Manifest root containing NSCLC_Radiogenomics and metadata.csv")
    p.add_argument("--metadata_csv", type=str, default="metadata.csv", help="TCIA metadata CSV path")
    p.add_argument(
        "--labels_csv",
        type=str,
        default="NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv",
        help="Labels CSV path",
    )
    p.add_argument("--limit", type=int, default=100, help="Run only the first N cases (default: 100, 0 = all)")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_nsclc_track_c", help="Directory for outputs")
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
    p.add_argument(
        "--series_alias",
        type=str,
        default="core",
        help="Which NSCLC curated alias group to use: default | all | core | diagnostic_ct | pet_primary | pet_ct_core | ct_only | pet_only | pet_nac | pet_ac | ctac | fusion_ct",
    )
    p.add_argument("--case_ids", type=str, default="", help="Optional comma-separated case IDs to run, e.g. AMC-001,R01-001")
    p.add_argument("--no_skip_amc", action="store_true", help="Include AMC cases. By default AMC cases are skipped.")
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
        help="Whether to send sampled slices as per-series montages or as separate images.",
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
        "--m3d_sequence_strategy",
        type=str,
        choices=["primary", "all_vote"],
        default="all_vote",
        help="How to handle multiple selected NSCLC series with M3D.",
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
        default="all_vote",
        help="How to handle multiple selected NSCLC series with RadFM.",
    )
    return p


def _resolve_csv_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (data_root / path).resolve()


def _normalize_uint8(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if not np.isfinite(lo):
        lo = float(finite.min())
    if not np.isfinite(hi):
        hi = float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


def _series_token(record: SeriesRecord) -> str:
    role = str(record.role or "series").replace("_", " ").strip()
    modality = str(record.modality or "IMG").strip().upper() or "IMG"
    date = str(record.study_date or "").strip()
    return " ".join(part for part in [modality, role, date] if part).strip() or record.abs_path.name


def _load_dicom_series_volume(series_path: Path, *, label: Optional[str] = None) -> LoadedVolume:
    try:
        import pydicom  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "NSCLC Track C DICOM loading requires pydicom. Install it in the Track C environment, e.g. `pip install pydicom`."
        ) from e

    path = Path(series_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DICOM series path does not exist: {path}")

    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted([p for p in path.rglob("*") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No files found under DICOM series path: {path}")

    frames: List[Tuple[float, int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, str]] = []
    fallback_arrays: List[Tuple[int, np.ndarray]] = []
    common_shape: Optional[Tuple[int, int]] = None

    for candidate in candidates:
        try:
            ds = pydicom.dcmread(str(candidate), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        try:
            arr = ds.pixel_array
        except Exception:
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2:
            continue
        if common_shape is None:
            common_shape = tuple(int(v) for v in arr.shape)
        if tuple(int(v) for v in arr.shape) != common_shape:
            continue
        arr = arr.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        instance_number = int(getattr(ds, "InstanceNumber", len(frames) + len(fallback_arrays) + 1) or len(frames) + len(fallback_arrays) + 1)
        iop_raw = getattr(ds, "ImageOrientationPatient", None)
        ipp_raw = getattr(ds, "ImagePositionPatient", None)
        spacing_raw = getattr(ds, "PixelSpacing", None)
        series_label = str(getattr(ds, "SeriesDescription", "") or label or path.name)
        if iop_raw is not None and ipp_raw is not None and spacing_raw is not None:
            try:
                iop = np.asarray([float(v) for v in iop_raw], dtype=np.float64)
                ipp = np.asarray([float(v) for v in ipp_raw], dtype=np.float64)
                spacing = np.asarray([float(v) for v in spacing_raw], dtype=np.float64)
                if iop.shape == (6,) and ipp.shape == (3,) and spacing.shape == (2,):
                    row_dir = iop[:3]
                    col_dir = iop[3:]
                    normal = np.cross(row_dir, col_dir)
                    offset = float(np.dot(normal, ipp))
                    frames.append((offset, instance_number, arr, row_dir, col_dir, ipp, series_label))
                    continue
            except Exception:
                pass
        fallback_arrays.append((instance_number, arr))

    if not frames and not fallback_arrays:
        raise RuntimeError(f"No readable 2D pixel data found under DICOM series path: {path}")

    if frames:
        frames.sort(key=lambda item: (item[0], item[1]))
        sorted_arrays = [item[2] for item in frames]
        row_dir = np.asarray(frames[0][3], dtype=np.float64)
        col_dir = np.asarray(frames[0][4], dtype=np.float64)
        first_ipp = np.asarray(frames[0][5], dtype=np.float64)
        series_label = str(label or frames[0][6] or path.name)
        pixel_spacing = np.asarray(
            [
                float(getattr(pydicom.dcmread(str(candidates[0]), stop_before_pixels=True, force=True), "PixelSpacing", [1.0, 1.0])[0]),
                float(getattr(pydicom.dcmread(str(candidates[0]), stop_before_pixels=True, force=True), "PixelSpacing", [1.0, 1.0])[1]),
            ],
            dtype=np.float64,
        )
        row_spacing = float(pixel_spacing[0])
        col_spacing = float(pixel_spacing[1])
        if len(frames) >= 2:
            diffs = [np.asarray(frames[i + 1][5], dtype=np.float64) - np.asarray(frames[i][5], dtype=np.float64) for i in range(len(frames) - 1)]
            slice_step = np.mean(np.stack(diffs, axis=0), axis=0)
            if float(np.linalg.norm(slice_step)) <= 1e-6:
                normal = np.cross(row_dir, col_dir)
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm <= 1e-6:
                    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    normal = normal / normal_norm
                slice_step = normal
        else:
            normal = np.cross(row_dir, col_dir)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-6:
                normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                normal = normal / normal_norm
            ds0 = pydicom.dcmread(str(candidates[0]), stop_before_pixels=True, force=True)
            slice_spacing = float(getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 1.0)) or 1.0)
            slice_step = normal * slice_spacing

        vol_kji = np.stack(sorted_arrays, axis=0)
        data_ijk = np.transpose(vol_kji, (2, 1, 0)).astype(np.float32, copy=False)
        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[:3, 0] = row_dir * col_spacing
        affine_lps[:3, 1] = col_dir * row_spacing
        affine_lps[:3, 2] = slice_step
        affine_lps[:3, 3] = first_ipp
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine_ras = lps_to_ras @ affine_lps
    else:
        fallback_arrays.sort(key=lambda item: item[0])
        vol_kji = np.stack([item[1] for item in fallback_arrays], axis=0)
        data_ijk = np.transpose(vol_kji, (2, 1, 0)).astype(np.float32, copy=False)
        affine_ras = np.eye(4, dtype=np.float64)
        series_label = str(label or path.name)

    stem = str(label or path.name)
    return LoadedVolume(
        path=path,
        name=str(series_label),
        stem=stem,
        data_ijk=data_ijk,
        affine_ijk_to_ras=np.asarray(affine_ras, dtype=np.float64),
        display_uint8=_normalize_uint8(data_ijk),
    )


def _load_case_series(records: Sequence[SeriesRecord]) -> List[LoadedVolume]:
    volumes: List[LoadedVolume] = []
    for record in records:
        volumes.append(_load_dicom_series_volume(record.abs_path, label=_series_token(record)))
    return volumes


def _save_loaded_volume_as_nifti(volume: LoadedVolume, out_path: Path) -> Path:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Saving Track C worker inputs requires nibabel.") from e

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(volume.data_ijk, dtype=np.float32), np.asarray(volume.affine_ijk_to_ras, dtype=np.float64))
    nib.save(img, str(out_path))
    return out_path


def _resolve_selected_records(case, alias: str) -> Tuple[str, List[SeriesRecord]]:
    alias_raw = str(alias or "default").strip()
    alias_l = alias_raw.lower()
    if alias_l in {"", "default", "auto"}:
        if case.catalog.default_aliases:
            resolved = str(case.catalog.default_aliases[0])
            records = list(case.catalog.alias_map.get(resolved, []))
            if records:
                return resolved, records
        return "all", list(case.catalog.all_series_records(prefer_latest=True))
    if alias_l == "all":
        return "all", list(case.catalog.all_series_records(prefer_latest=True))
    if alias_raw in case.catalog.alias_map:
        return alias_raw, list(case.catalog.alias_map[alias_raw])
    if alias_l in case.catalog.alias_map:
        return alias_l, list(case.catalog.alias_map[alias_l])
    raise TrackCConfigError(
        f"Unsupported --series_alias {alias_raw!r} for case {case.case_id}. Available aliases: {sorted(case.catalog.alias_map.keys())} plus 'default' and 'all'."
    )


def _parse_case_ids(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _filter_cases_by_ids(cases: Sequence[Any], raw_case_ids: str) -> List[Any]:
    case_ids = _parse_case_ids(raw_case_ids)
    if not case_ids:
        return list(cases)
    by_id = {str(case.case_id): case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise TrackCConfigError(f"Unknown --case_ids values: {missing}. Available examples: {sorted(list(by_id))[:10]}")
    return [by_id[case_id] for case_id in case_ids]


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalize_modality(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"PET", "PT"}:
        return "PT"
    if text == "CT":
        return "CT"
    return text


def _score_source_record(record: SeriesRecord, *, source_name: str, source_modality: str) -> int:
    score = 0
    record_modality = _normalize_modality(getattr(record, "modality", ""))
    desired_modality = _normalize_modality(source_modality)
    if desired_modality and record_modality == desired_modality:
        score += 80

    source_norm = _normalize_match_text(source_name)
    if not source_norm:
        return score

    candidates = [
        str(getattr(record, "display_name", "") or ""),
        str(getattr(record, "series_description", "") or ""),
        str(getattr(record, "abs_path", Path("."))).split("/")[-1],
        str(getattr(record, "abs_path", Path("."))).split("/")[-2] if len(str(getattr(record, "abs_path", Path("."))).split("/")) >= 2 else "",
        str(getattr(record, "raw_file_location", "") or ""),
    ]
    for candidate in candidates:
        cand_norm = _normalize_match_text(candidate)
        if not cand_norm:
            continue
        if cand_norm == source_norm:
            score += 300
            continue
        if source_norm in cand_norm or cand_norm in source_norm:
            score += 180
            continue
        source_tokens = set(source_norm.split())
        cand_tokens = set(cand_norm.split())
        overlap = len(source_tokens & cand_tokens)
        if overlap:
            score += 25 * overlap
    return score


def _pick_first_record(records: Sequence[SeriesRecord], *, modality: str) -> Optional[SeriesRecord]:
    desired = _normalize_modality(modality)
    for record in records:
        if _normalize_modality(getattr(record, "modality", "")) == desired:
            return record
    return None


def _resolve_evidence_record(
    *,
    case,
    selected_records: Sequence[SeriesRecord],
    segment_gt: Optional[SegmentGroundTruth],
) -> Tuple[SeriesRecord, Dict[str, Any]]:
    all_records = list(case.catalog.all_series_records(prefer_latest=True))
    if not all_records:
        raise RuntimeError(f"No series records available for evidence selection in case {case.case_id}")

    source_name = str(segment_gt.source_volume_name or "").strip() if segment_gt is not None and segment_gt.has_segment else ""
    source_modality = str(segment_gt.source_modality or "").strip() if segment_gt is not None and segment_gt.has_segment else ""
    if source_name or source_modality:
        ranked = sorted(
            (
                (
                    _score_source_record(
                        record,
                        source_name=source_name,
                        source_modality=source_modality,
                    ),
                    record,
                )
                for record in all_records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked and ranked[0][0] >= 120:
            best_score, best_record = ranked[0]
            return best_record, {
                "strategy": "segment_source_match",
                "source_volume_name": source_name,
                "source_modality": source_modality,
                "match_score": int(best_score),
            }

    desired_modality = _normalize_modality(source_modality) if source_modality else "CT"
    if desired_modality == "PT":
        alias_candidates = case.catalog.alias_map.get("pet_primary") or case.catalog.alias_map.get("pet_only") or []
    else:
        alias_candidates = case.catalog.alias_map.get("diagnostic_ct") or case.catalog.alias_map.get("ct_only") or []
    if alias_candidates:
        return alias_candidates[0], {
            "strategy": "default_alias_fallback",
            "source_volume_name": source_name,
            "source_modality": source_modality,
            "selected_alias": "pet_primary" if desired_modality == "PT" else "diagnostic_ct",
        }

    by_modality = _pick_first_record(all_records, modality=desired_modality)
    if by_modality is not None:
        return by_modality, {
            "strategy": "modality_fallback",
            "source_volume_name": source_name,
            "source_modality": source_modality,
            "selected_modality": desired_modality,
        }

    if selected_records:
        return selected_records[0], {
            "strategy": "selected_series_fallback",
            "source_volume_name": source_name,
            "source_modality": source_modality,
        }

    return all_records[0], {
        "strategy": "first_series_fallback",
        "source_volume_name": source_name,
        "source_modality": source_modality,
    }


def _find_loaded_volume_for_record(volumes: Sequence[LoadedVolume], record: SeriesRecord) -> Optional[LoadedVolume]:
    record_path = str(Path(record.abs_path).expanduser().resolve())
    for volume in volumes:
        if str(Path(volume.path).expanduser().resolve()) == record_path:
            return volume
    return None


def _build_nsclc_segmentation_question(record: SeriesRecord) -> str:
    modality = _normalize_modality(getattr(record, "modality", ""))
    if modality == "PT":
        return "What is hypermetabolic primary lung tumor in this image? Please output the segmentation mask."
    return "What is primary lung tumor in this image? Please output the segmentation mask."


def _format_worker_evidence_block(payload: Dict[str, Any]) -> Optional[str]:
    evidence = payload.get("evidence") or {}
    key_slice_mm = evidence.get("key_slice_mm") or {}
    point_ras = evidence.get("point_ras") or []
    if len(point_ras) != 3 or not key_slice_mm:
        return None
    return "\n".join(
        [
            "EVIDENCE:",
            f"KEY_SLICE_AXIAL_MM: {float(key_slice_mm.get('axial', 0.0)):.4f}",
            f"KEY_SLICE_SAGITTAL_MM: {float(key_slice_mm.get('sagittal', 0.0)):.4f}",
            f"KEY_SLICE_CORONAL_MM: {float(key_slice_mm.get('coronal', 0.0)):.4f}",
            f"RSA: [{float(point_ras[0]):.4f}, {float(point_ras[1]):.4f}, {float(point_ras[2]):.4f}]",
        ]
    )


def _build_api_artifacts_and_summary(
    *,
    method: str,
    case_id: str,
    case_out: Path,
    volumes: Sequence[LoadedVolume],
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


def _format_options_block(title: str, mapping: Dict[str, str]) -> str:
    lines = [f"{key}) {value}" for key, value in mapping.items()]
    return f"{title}:\n" + "\n".join(lines)


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
    case,
    method: str,
    selected_alias: str,
    selected_records: Sequence[SeriesRecord],
    artifact_summary_lines: Sequence[str],
) -> str:
    series_lines = [
        f"- {record.display_name} | role={record.role} | modality={record.modality}"
        for record in selected_records
    ]
    artifact_block = "\n".join(f"- {line}" for line in artifact_summary_lines) if artifact_summary_lines else "- No artifact summary available."
    return render_prompt(
        PROMPT_REF,
        "templates.user",
        case_id=case.case_id,
        method_name=method_display_name(method),
        selected_alias=selected_alias,
        series_lines="\n".join(series_lines) if series_lines else "- <none>",
        artifact_block=artifact_block,
        location_options=_format_options_block("1) Tumor location", LOCATION_OPTIONS),
        t_stage_options=_format_options_block("2) Pathological T stage", T_STAGE_OPTIONS),
        n_stage_options=_format_options_block("3) Pathological N stage", N_STAGE_OPTIONS),
        histology_options=_format_options_block("4) Histology", HISTOLOGY_OPTIONS),
        grade_options=_format_options_block("5) Histopathological Grade", GRADE_OPTIONS),
    )


def _build_worker_question(*, series_name: str) -> str:
    return render_prompt(
        PROMPT_REF,
        "templates.worker_all_questions",
        series_name=series_name,
        location_options=_format_options_block("1) Tumor location", LOCATION_OPTIONS),
        t_stage_options=_format_options_block("2) Pathological T stage", T_STAGE_OPTIONS),
        n_stage_options=_format_options_block("3) Pathological N stage", N_STAGE_OPTIONS),
        histology_options=_format_options_block("4) Histology", HISTOLOGY_OPTIONS),
        grade_options=_format_options_block("5) Histopathological Grade", GRADE_OPTIONS),
    )


def _build_worker_single_question(*, series_name: str, question_id: int) -> str:
    question_name = QUESTION_NAMES[int(question_id)]
    options = QUESTION_OPTION_MAPS[int(question_id)]
    valid = ", ".join(options.keys())
    return render_prompt(
        PROMPT_REF,
        "templates.worker_single_question",
        series_name=series_name,
        question_id=int(question_id),
        question_name=question_name,
        question_options=_format_options_block(f"Question {int(question_id)}", options),
        valid_options=valid,
    )


def _build_worker_evidence_question(*, series_name: str, segmentation_question: str) -> str:
    return render_prompt(
        PROMPT_REF,
        "templates.worker_evidence",
        series_name=series_name,
        segmentation_question=(
            segmentation_question.strip()
            if segmentation_question.strip()
            else render_prompt(
                PROMPT_REF,
                "templates.fragments.default_segmentation_question",
            )
        ),
    )


def _normalize_question_choice(*, question_id: int, value: Any) -> Optional[str]:
    token = str(value or "").strip().upper()
    if not token:
        return None
    return token if token in QUESTION_OPTION_MAPS[int(question_id)] else None


def _normalize_pred_map(raw_map: Any) -> Dict[str, str]:
    if not isinstance(raw_map, dict):
        return {}
    normalized: Dict[str, str] = {}
    for qid in QUESTION_IDS:
        raw_value = raw_map.get(str(qid), raw_map.get(qid))
        token = _normalize_question_choice(question_id=int(qid), value=raw_value)
        if token:
            normalized[str(qid)] = token
    return normalized


def _select_worker_volumes(volumes: Sequence[LoadedVolume], strategy: str) -> List[LoadedVolume]:
    strategy_l = str(strategy).strip().lower()
    if strategy_l == "primary":
        return [volumes[0]] if volumes else []
    if strategy_l == "all_vote":
        return list(volumes)
    raise ValueError(f"Unsupported sequence strategy: {strategy!r}")


def _aggregate_worker_outputs(outputs: Sequence[Dict[str, Any]], *, evidence_block: Optional[str] = None) -> Dict[str, Any]:
    normalized_outputs: List[Dict[str, Any]] = []
    vote_counts: Dict[str, Dict[str, int]] = {str(qid): {} for qid in QUESTION_IDS}
    final_map: Dict[int, Optional[str]] = {qid: None for qid in QUESTION_IDS}

    for idx, raw in enumerate(outputs, start=1):
        text = str(raw.get("text") or "")
        pred_map = _normalize_pred_map(raw.get("pred_map"))
        if pred_map:
            compact = _compact_from_map({qid: pred_map.get(str(qid)) for qid in QUESTION_IDS})
        else:
            compact = extract_numbered_answers(text, question_ids=QUESTION_IDS)
            pred_map = {
                str(qid): token
                for qid, token in _split_numbered_answers(compact).items()
                if _normalize_question_choice(question_id=int(qid), value=token)
            }
        enriched = dict(raw)
        enriched["pred_compact"] = compact
        enriched["pred_map"] = pred_map
        enriched["label"] = str(raw.get("label") or raw.get("series_name") or raw.get("volume_name") or f"run_{idx}")
        normalized_outputs.append(enriched)

    for qid in QUESTION_IDS:
        counts: Counter[str] = Counter()
        first_seen: Dict[str, int] = {}
        for idx, item in enumerate(normalized_outputs):
            ans = (item.get("pred_map") or {}).get(str(qid))
            if not ans:
                continue
            ans = str(ans).strip().upper()
            counts[ans] += 1
            first_seen.setdefault(ans, idx)
        vote_counts[str(qid)] = {k: int(v) for k, v in sorted(counts.items())}
        if not counts:
            continue
        best = max(counts.values())
        winners = [ans for ans, n in counts.items() if n == best]
        final_map[qid] = sorted(winners, key=lambda ans: first_seen.get(ans, 10**9))[0]

    final_compact = _compact_from_map(final_map)
    winning_output: Optional[Dict[str, Any]] = None
    if final_compact:
        for item in normalized_outputs:
            if item.get("pred_compact") == final_compact:
                winning_output = item
                break
    if winning_output is None and normalized_outputs:
        winning_output = normalized_outputs[0]

    base_text = str((winning_output or {}).get("text") or "")
    summary_lines = ["SERIES_VOTE_SUMMARY:"]
    for item in normalized_outputs:
        summary_lines.append(f"- {item['label']}: {item.get('pred_compact') or '<none>'}")
    summary_lines.append(f"- FINAL_VOTE: {final_compact or '<none>'}")
    summary_lines.append(f"- PER_QUESTION_COUNTS: {vote_counts}")

    final_text = base_text.strip()
    if evidence_block:
        if final_text:
            final_text += "\n\n"
        final_text += evidence_block
    if final_text:
        final_text += "\n\n"
    if final_text:
        final_text += "\n".join(summary_lines)
    else:
        final_text = "\n".join(summary_lines)
    if final_compact:
        final_text += f"\nANSWER: {final_compact}"

    return {
        "text": final_text,
        "pred_compact": final_compact,
        "vote_counts": vote_counts,
        "worker_outputs": normalized_outputs,
        "winning_output": winning_output,
    }


def _run_m3d_single_question(
    *,
    worker_script: Path,
    nifti_path: Path,
    out_dir: Path,
    volume_name: str,
    question_id: int,
    args: argparse.Namespace,
    session: SessionManager,
) -> Tuple[Dict[str, Any], Optional[str]]:
    request = {
        "nifti_path": str(nifti_path),
        "question": _build_worker_single_question(series_name=volume_name, question_id=int(question_id)),
        "model_name_or_path": args.m3d_model_name_or_path,
        "device": args.m3d_device,
        "dtype": args.m3d_dtype,
        "max_new_tokens": int(args.m3d_max_new_tokens),
        "include_evidence": False,
    }
    payload = run_external_worker(
        worker_script=worker_script,
        request_payload=request,
        out_dir=out_dir,
        python_executable=args.m3d_python or args.worker_python,
        timeout_sec=float(args.worker_timeout_sec),
        session=session,
        event_prefix=f"track_c_m3d_worker_q{int(question_id)}",
        prompt_metadata=prompt_metadata(PROMPT_REF),
    )
    if not bool(payload.get("ok", False)):
        raise ExternalWorkerError(str(payload.get("error") or "M3D worker returned ok=false"))
    choice = extract_choice_from_set(
        str(payload.get("text") or ""),
        valid_choices=tuple(QUESTION_OPTION_MAPS[int(question_id)].keys()),
    )
    payload["question_id"] = int(question_id)
    payload["choice"] = choice
    return payload, choice


def _run_radfm_single_question(
    *,
    worker_script: Path,
    nifti_path: Path,
    out_dir: Path,
    volume_name: str,
    question_id: int,
    args: argparse.Namespace,
    session: SessionManager,
) -> Tuple[Dict[str, Any], Optional[str]]:
    valid_choices = tuple(QUESTION_OPTION_MAPS[int(question_id)].keys())
    request = {
        "repo_root": args.radfm_repo_root,
        "checkpoint_path": args.radfm_checkpoint,
        "language_dir": args.radfm_language_dir,
        "nifti_path": str(nifti_path),
        "question": _build_worker_single_question(series_name=volume_name, question_id=int(question_id)),
        "question_id": int(question_id),
        "valid_choices": list(valid_choices),
        "include_evidence": False,
        "device": args.radfm_device,
        "target_depth": int(args.radfm_target_depth),
        "max_new_tokens": int(args.radfm_max_new_tokens),
    }
    payload = run_external_worker(
        worker_script=worker_script,
        request_payload=request,
        out_dir=out_dir,
        python_executable=args.radfm_python or args.worker_python,
        timeout_sec=float(args.worker_timeout_sec),
        session=session,
        event_prefix=f"track_c_radfm_worker_q{int(question_id)}",
        prompt_metadata=prompt_metadata(PROMPT_REF),
    )
    if not bool(payload.get("ok", False)):
        raise ExternalWorkerError(str(payload.get("error") or "RadFM worker returned ok=false"))
    choice = extract_choice_from_set(
        str(payload.get("text") or ""),
        valid_choices=valid_choices,
    )
    payload["question_id"] = int(question_id)
    payload["choice"] = choice
    return payload, choice


def _run_m3d_method(
    *,
    case,
    volumes: Sequence[LoadedVolume],
    selected_records: Sequence[SeriesRecord],
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
    segment_gt: Optional[SegmentGroundTruth],
) -> Dict[str, Any]:
    selected_volumes = _select_worker_volumes(volumes, args.m3d_sequence_strategy)
    if not selected_volumes:
        raise RuntimeError(f"No loaded volumes available for M3D classification in case {case.case_id}")
    worker_script = Path(__file__).resolve().parent / "workers" / "m3d_worker.py"
    outputs: List[Dict[str, Any]] = []
    for volume in selected_volumes:
        work_token = safe_token(volume.stem or volume.name)
        work_dir = case_out / "workers" / "m3d" / work_token
        nifti_path = _save_loaded_volume_as_nifti(volume, work_dir / f"{work_token}.nii.gz")
        pred_map: Dict[str, str] = {}
        question_outputs: Dict[str, Any] = {}
        summary_lines = [f"QUESTION_SUMMARY: {volume.name}"]
        for qid in QUESTION_IDS:
            payload, choice = _run_m3d_single_question(
                worker_script=worker_script,
                nifti_path=nifti_path,
                out_dir=work_dir / f"q{int(qid)}",
                volume_name=volume.name,
                question_id=int(qid),
                args=args,
                session=session,
            )
            question_outputs[str(qid)] = payload
            if choice:
                pred_map[str(qid)] = choice
            summary_lines.append(f"- Q{int(qid)} {QUESTION_NAMES[int(qid)]}: {choice or '<none>'}")
        outputs.append(
            {
                "label": volume.name,
                "series_name": volume.name,
                "pred_map": pred_map,
                "pred_compact": _compact_from_map({qid: pred_map.get(str(qid)) for qid in QUESTION_IDS}),
                "question_outputs": question_outputs,
                "text": "\n".join(summary_lines),
            }
        )

    evidence_record, evidence_selection = _resolve_evidence_record(
        case=case,
        selected_records=selected_records,
        segment_gt=segment_gt,
    )
    evidence_volume = _find_loaded_volume_for_record(volumes, evidence_record)
    if evidence_volume is None:
        evidence_volume = _load_dicom_series_volume(evidence_record.abs_path, label=_series_token(evidence_record))
    evidence_token = safe_token(f"evidence_{evidence_volume.stem or evidence_volume.name}")
    evidence_dir = case_out / "workers" / "m3d_evidence" / evidence_token
    evidence_nifti_path = _save_loaded_volume_as_nifti(evidence_volume, evidence_dir / f"{evidence_token}.nii.gz")
    evidence_request = {
        "nifti_path": str(evidence_nifti_path),
        "question": _build_worker_question(series_name=evidence_volume.name),
        "model_name_or_path": args.m3d_model_name_or_path,
        "device": args.m3d_device,
        "dtype": args.m3d_dtype,
        "max_new_tokens": int(args.m3d_max_new_tokens),
        "include_evidence": True,
        "segmentation_question": _build_nsclc_segmentation_question(evidence_record),
    }
    evidence_payload = run_external_worker(
        worker_script=worker_script,
        request_payload=evidence_request,
        out_dir=evidence_dir,
        python_executable=args.m3d_python or args.worker_python,
        timeout_sec=float(args.worker_timeout_sec),
        session=session,
        event_prefix="track_c_m3d_evidence_worker",
        prompt_metadata=prompt_metadata(PROMPT_REF),
    )
    if not bool(evidence_payload.get("ok", False)):
        raise ExternalWorkerError(str(evidence_payload.get("error") or "M3D evidence worker returned ok=false"))
    evidence_block = _format_worker_evidence_block(evidence_payload)
    if not evidence_block:
        raise ExternalWorkerError(
            f"M3D evidence worker did not return a usable EVIDENCE block for case {case.case_id}: {evidence_payload.get('evidence_error') or evidence_payload.get('error') or 'missing evidence'}"
        )

    aggregate = _aggregate_worker_outputs(outputs, evidence_block=evidence_block)
    return {
        "text": str(aggregate.get("text") or ""),
        "pred_compact": aggregate.get("pred_compact"),
        "worker_outputs": aggregate.get("worker_outputs") or outputs,
        "aggregation": aggregate,
        "evidence_worker_output": evidence_payload,
        "evidence_selection": evidence_selection,
        "evidence_record": evidence_record.as_dict(),
    }


def _run_radfm_method(
    *,
    case,
    volumes: Sequence[LoadedVolume],
    selected_records: Sequence[SeriesRecord],
    case_out: Path,
    args: argparse.Namespace,
    session: SessionManager,
    segment_gt: Optional[SegmentGroundTruth],
) -> Dict[str, Any]:
    if not args.radfm_repo_root:
        raise ValueError("--radfm_repo_root is required for --method radfm")
    if not args.radfm_checkpoint:
        raise ValueError("--radfm_checkpoint is required for --method radfm")

    selected_volumes = _select_worker_volumes(volumes, args.radfm_sequence_strategy)
    if not selected_volumes:
        raise RuntimeError(f"No loaded volumes available for RadFM classification in case {case.case_id}")
    worker_script = Path(__file__).resolve().parent / "workers" / "radfm_worker.py"
    outputs: List[Dict[str, Any]] = []
    for volume in selected_volumes:
        work_token = safe_token(volume.stem or volume.name)
        work_dir = case_out / "workers" / "radfm" / work_token
        nifti_path = _save_loaded_volume_as_nifti(volume, work_dir / f"{work_token}.nii.gz")
        pred_map: Dict[str, str] = {}
        question_outputs: Dict[str, Any] = {}
        summary_lines = [f"QUESTION_SUMMARY: {volume.name}"]
        for qid in QUESTION_IDS:
            payload, choice = _run_radfm_single_question(
                worker_script=worker_script,
                nifti_path=nifti_path,
                out_dir=work_dir / f"q{int(qid)}",
                volume_name=volume.name,
                question_id=int(qid),
                args=args,
                session=session,
            )
            question_outputs[str(qid)] = payload
            if choice:
                pred_map[str(qid)] = choice
            summary_lines.append(f"- Q{int(qid)} {QUESTION_NAMES[int(qid)]}: {choice or '<none>'}")
        outputs.append(
            {
                "label": volume.name,
                "series_name": volume.name,
                "pred_map": pred_map,
                "pred_compact": _compact_from_map({qid: pred_map.get(str(qid)) for qid in QUESTION_IDS}),
                "question_outputs": question_outputs,
                "text": "\n".join(summary_lines),
            }
        )

    evidence_record, evidence_selection = _resolve_evidence_record(
        case=case,
        selected_records=selected_records,
        segment_gt=segment_gt,
    )
    evidence_volume = _find_loaded_volume_for_record(volumes, evidence_record)
    if evidence_volume is None:
        evidence_volume = _load_dicom_series_volume(evidence_record.abs_path, label=_series_token(evidence_record))
    evidence_token = safe_token(f"evidence_{evidence_volume.stem or evidence_volume.name}")
    evidence_dir = case_out / "workers" / "radfm_evidence" / evidence_token
    evidence_nifti_path = _save_loaded_volume_as_nifti(evidence_volume, evidence_dir / f"{evidence_token}.nii.gz")
    evidence_request = {
        "repo_root": args.radfm_repo_root,
        "checkpoint_path": args.radfm_checkpoint,
        "language_dir": args.radfm_language_dir,
        "nifti_path": str(evidence_nifti_path),
        "question": _build_worker_evidence_question(
            series_name=evidence_volume.name,
            segmentation_question=_build_nsclc_segmentation_question(evidence_record),
        ),
        "include_evidence": True,
        "segmentation_question": _build_nsclc_segmentation_question(evidence_record),
        "device": args.radfm_device,
        "target_depth": int(args.radfm_target_depth),
        "max_new_tokens": int(args.radfm_max_new_tokens),
    }
    evidence_payload = run_external_worker(
        worker_script=worker_script,
        request_payload=evidence_request,
        out_dir=evidence_dir,
        python_executable=args.radfm_python or args.worker_python,
        timeout_sec=float(args.worker_timeout_sec),
        session=session,
        event_prefix="track_c_radfm_evidence_worker",
        prompt_metadata=prompt_metadata(PROMPT_REF),
    )
    if not bool(evidence_payload.get("ok", False)):
        raise ExternalWorkerError(str(evidence_payload.get("error") or "RadFM evidence worker returned ok=false"))
    evidence_block = _format_worker_evidence_block(evidence_payload)
    if not evidence_block:
        raise ExternalWorkerError(
            f"RadFM evidence worker did not return a usable EVIDENCE block for case {case.case_id}: {evidence_payload.get('error') or 'missing evidence'}"
        )

    aggregate = _aggregate_worker_outputs(outputs, evidence_block=evidence_block)
    return {
        "text": str(aggregate.get("text") or ""),
        "pred_compact": aggregate.get("pred_compact"),
        "worker_outputs": aggregate.get("worker_outputs") or outputs,
        "aggregation": aggregate,
        "evidence_worker_output": evidence_payload,
        "evidence_selection": evidence_selection,
        "evidence_record": evidence_record.as_dict(),
    }


def _run_api_method(
    *,
    method: str,
    case,
    case_out: Path,
    selected_alias: str,
    selected_records: Sequence[SeriesRecord],
    volumes: Sequence[LoadedVolume],
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
            "track_c_artifact_generated",
            method=method,
            kind=artifact.kind,
            path=str(artifact.path),
            description=artifact.description,
        )

    message = _build_case_message(
        case=case,
        method=method,
        selected_alias=selected_alias,
        selected_records=selected_records,
        artifact_summary_lines=artifact_summary_lines,
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
    text = str(model_result.get("text") or "")
    return {
        "text": text,
        "pred_compact": extract_numbered_answers(text, question_ids=QUESTION_IDS),
        "provider": model_result.get("provider"),
        "response_id": model_result.get("response_id"),
        "artifact_summary": artifact_summary_lines,
        "artifact_meta": artifact_meta,
        "unexpected_tool_calls": model_result.get("unexpected_tool_calls") or [],
    }


def _method_config_summary(args: argparse.Namespace, method: str, *, selected_alias: Optional[str] = None) -> Dict[str, Any]:
    base: Dict[str, Any] = {"method": method, "series_alias": selected_alias or args.series_alias}
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
                "m3d_sequence_strategy": args.m3d_sequence_strategy,
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
                "radfm_sequence_strategy": args.radfm_sequence_strategy,
                "radfm_python": args.radfm_python or args.worker_python,
            }
        )
        return base
    return base


def main() -> None:
    args = build_parser().parse_args()
    method = normalize_method_name(args.method)

    data_root = Path(args.data_root).expanduser().resolve()
    metadata_csv = _resolve_csv_path(data_root, args.metadata_csv)
    labels_csv = _resolve_csv_path(data_root, args.labels_csv)

    run_name = args.model if method in {"compressed_figures", "table3_api"} else method
    log_dir = resolve_run_output_dir(args.log_dir, model=run_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark_track_c.log")
    log = logging.getLogger("benchmark.nsclc.track_c")

    requested_limit = None if args.case_ids else (args.limit if args.limit and args.limit > 0 else None)
    skip_amc = not bool(args.no_skip_amc)
    cases = list(iter_cases(data_root=data_root, metadata_csv=metadata_csv, labels_csv=labels_csv, limit=None))
    cases = _filter_cases_by_ids(cases, args.case_ids)
    skipped_amc = 0
    if skip_amc:
        before_count = len(cases)
        cases = [case for case in cases if not str(case.case_id).upper().startswith("AMC-")]
        skipped_amc = before_count - len(cases)
    if requested_limit is not None:
        cases = cases[:requested_limit]
    log.info("Track C method=%s (%s)", method, method_display_name(method))
    if skipped_amc:
        log.info("Skipping %d AMC cases by default. Use --no_skip_amc to include them.", skipped_amc)
    log.info("Found %d NSCLC cases under %s", len(cases), data_root)

    results: List[Dict[str, Any]] = []
    n_case_scored = 0
    n_case_exact_correct = 0
    n_case_exact_with_evidence_correct = 0
    n_correct_with_evidence = 0
    n_localization_correct = 0
    case_with_evidence_partial_score_total = 0.0
    q_scored = {qid: 0 for qid in QUESTION_IDS}
    q_correct = {qid: 0 for qid in QUESTION_IDS}
    n_failed = 0
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

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.nsclc.track_c.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)

        selected_alias, selected_records = _resolve_selected_records(case, args.series_alias)
        if not selected_records:
            raise RuntimeError(f"No imaging series resolved for case {case.case_id} with alias {selected_alias!r}")

        session = SessionManager(out_dir=case_out)
        session.update_state(
            case_preset="track_c_nsclc",
            loaded_series_display_names=[record.display_name for record in selected_records],
            loaded_series_dirs=[str(record.abs_path) for record in selected_records],
            active_tool_packs=[],
        )
        session.log_run_event(
            "case_script_start",
            script=SCRIPT_NAME,
            case_id=case.case_id,
            method=method,
            method_display_name=method_display_name(method),
            method_config=_method_config_summary(args, method, selected_alias=selected_alias),
        )

        catalog_summary = case.catalog.to_summary()
        (case_out / "catalog.json").write_text(json.dumps(catalog_summary, indent=2), encoding="utf-8")
        selected_manifest = {
            "selected_alias": selected_alias,
            "selected_records": [record.as_dict() for record in selected_records],
        }
        (case_out / "selected_series.json").write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")

        gt_map = {
            1: case.gt_location,
            2: case.gt_t_stage,
            3: case.gt_n_stage,
            4: case.gt_histology,
            5: case.gt_grade,
        }
        gt = _compact_from_map(gt_map)
        segment_gt: Optional[SegmentGroundTruth] = None
        segment_gt_error: Optional[str] = None

        try:
            volumes = _load_case_series(selected_records)
            segment_gt_dir = case_out / "segment_gt"
            try:
                refs_dir = segment_gt_dir / "references"
                prepared_dir = segment_gt_dir / "prepared_gt"
                refs_dir.mkdir(parents=True, exist_ok=True)
                reference_volume_paths: List[Path] = []
                for volume in volumes:
                    volume_token = safe_token(volume.stem or volume.name) or "volume"
                    reference_volume_paths.append(_save_loaded_volume_as_nifti(volume, refs_dir / f"{volume_token}.nii.gz"))
                nifti_segment_path = discover_nifti_segmentation_file(case.case_root)
                if nifti_segment_path is not None:
                    segment_gt = prepare_nifti_segment_ground_truth_py(
                        case_path=case.case_root,
                        out_dir=prepared_dir,
                        segment_path=nifti_segment_path,
                        reference_volume_paths=reference_volume_paths,
                        source_modality=None,
                        provenance_policy="not_applicable",
                    )
                else:
                    segment_gt = prepare_dicom_segment_ground_truth_py(
                        case_path=case.case_root,
                        out_dir=prepared_dir,
                        segment_series_paths=case.segment_series_paths,
                        source_modality="CT",
                        provenance_policy="not_applicable",
                    )
                if not segment_gt.has_segment:
                    segment_gt_error = (
                        "No usable NIfTI or public DICOM SEG tumor mask found for Track C scoring."
                    )
            except Exception as e:
                segment_gt_error = str(e)
            if segment_gt_error:
                case_log.warning("Segment GT discovery failed for %s: %s", case.case_id, segment_gt_error)
            if segment_gt is not None and segment_gt.gt_json_path and segment_gt.gt_json_path.exists():
                try:
                    payload = json.loads(segment_gt.gt_json_path.read_text(encoding="utf-8"))
                    (case_out / "segment_gt_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if segment_gt is None or not segment_gt.has_segment:
                case_log.info("No hidden segment GT available for %s; evidence metrics will be skipped.", case.case_id)

            planes_for_summary = parse_csv_list(args.figure_planes) if method == "compressed_figures" else parse_csv_list(args.table3_planes)
            volume_summary = build_volume_summary(volumes, planes=planes_for_summary)
            (case_out / "volume_summary.json").write_text(json.dumps(volume_summary, indent=2), encoding="utf-8")
            session.log_run_event("track_c_volumes_loaded", n_volumes=len(volumes), volume_summary=volume_summary)

            if method in {"compressed_figures", "table3_api"}:
                run_result = _run_api_method(
                    method=method,
                    case=case,
                    case_out=case_out,
                    selected_alias=selected_alias,
                    selected_records=selected_records,
                    volumes=volumes,
                    args=args,
                    session=session,
                )
            elif method == "m3d":
                run_result = _run_m3d_method(
                    case=case,
                    volumes=volumes,
                    selected_records=selected_records,
                    case_out=case_out,
                    args=args,
                    session=session,
                    segment_gt=segment_gt,
                )
            elif method == "radfm":
                run_result = _run_radfm_method(
                    case=case,
                    volumes=volumes,
                    selected_records=selected_records,
                    case_out=case_out,
                    args=args,
                    session=session,
                    segment_gt=segment_gt,
                )
            else:  # pragma: no cover - normalize_method_name already guards this.
                raise ValueError(f"Unsupported Track C method: {method}")

            raw_text = str(run_result.get("text") or "")
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")

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

            segment_check = (
                evaluate_segment_prediction(prediction=raw_text, gt=segment_gt, trace_path=session.trace_path)
                if segment_gt is not None
                else {
                    "available": False,
                    "has_segment_gt": False,
                    "segment_gt_error": segment_gt_error,
                }
            )
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_all_correct = (
                _nsclc_track_c_evidence_exact_correct(segment_check)
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
            correct_with_evidence = (
                bool(correct and key_slice_all_correct)
                if has_segment_gt
                else None
            )
            case_with_evidence_partial_score = (
                float(n_questions_correct) / float(len(QUESTION_IDS))
                if scored and has_segment_gt and bool(key_slice_all_correct)
                else 0.0
            )
            if bool(correct_with_evidence):
                n_case_exact_with_evidence_correct += 1
                n_correct_with_evidence += 1
            if bool(localization_correct):
                n_localization_correct += 1
            if scored and has_segment_gt:
                case_with_evidence_partial_score_total += case_with_evidence_partial_score

            tool_usage = _zero_tool_usage()
            rec = {
                "case_id": case.case_id,
                "gt": gt,
                "pred": pred,
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
                "agent_output_path": str(case_out / "agent_output.txt"),
                "correct": correct,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "case_with_evidence_partial_score": case_with_evidence_partial_score,
                "scored": scored,
                "per_question": per_question,
                "n_questions_scored": n_questions_scored,
                "n_questions_correct": n_questions_correct,
                "has_segment_gt": bool(segment_gt is not None and segment_gt.has_segment),
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
                "selected_alias": selected_alias,
                "selected_series": [record.as_dict() for record in selected_records],
                "method": method,
                "method_display_name": method_display_name(method),
                "method_config": _method_config_summary(args, method, selected_alias=selected_alias),
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
                case_with_evidence_partial_score=case_with_evidence_partial_score,
                scored=scored,
                has_segment_gt=bool(segment_gt is not None and segment_gt.has_segment),
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                elapsed_sec=rec["seconds"],
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

            case_log.info(
                "Completed case %d/%d: %s | method=%s | pred=%s | gt=%s | case_exact=%s | case_exact_with_evidence=%s | localization_correct=%s | partial_with_evidence=%.3f | q_correct=%d/%d",
                idx,
                len(cases),
                case.case_id,
                method,
                pred,
                gt,
                correct,
                correct_with_evidence,
                localization_correct,
                case_with_evidence_partial_score,
                n_questions_correct,
                n_questions_scored,
            )
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | method={method} | pred={pred} | gt={gt} | evidence_exact={segment_check.get('evidence_exact_correct')} | localization_correct={localization_correct} | partial_with_evidence={case_with_evidence_partial_score:.3f}"
            )
        except Exception as e:
            n_failed += 1
            per_question, n_questions_scored, _ = _score_questions(
                gt_map=gt_map,
                pred_compact=None,
                q_scored=q_scored,
                q_correct=q_correct,
            )
            scored = bool(gt)
            if scored:
                n_case_scored += 1

            segment_check = (
                evaluate_segment_prediction(prediction="", gt=segment_gt, trace_path=session.trace_path)
                if segment_gt is not None
                else {
                    "available": False,
                    "has_segment_gt": False,
                    "segment_gt_error": segment_gt_error,
                }
            )
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_all_correct = (
                _nsclc_track_c_evidence_exact_correct(segment_check)
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
            correct_with_evidence = False if has_segment_gt else None
            case_with_evidence_partial_score = 0.0
            if bool(localization_correct):
                n_localization_correct += 1

            tool_usage = _zero_tool_usage()
            rec = {
                "case_id": case.case_id,
                "gt": gt,
                "pred": None,
                "runlog_path": str(session.runlog_path),
                "trace_path": str(session.trace_path),
                "correct": False,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "case_with_evidence_partial_score": 0.0,
                "scored": scored,
                "per_question": per_question,
                "n_questions_scored": n_questions_scored,
                "n_questions_correct": 0,
                "has_segment_gt": bool(segment_gt is not None and segment_gt.has_segment),
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
                "selected_alias": selected_alias,
                "selected_series": [record.as_dict() for record in selected_records],
                "method": method,
                "method_display_name": method_display_name(method),
                "method_config": _method_config_summary(args, method, selected_alias=selected_alias),
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
                case_with_evidence_partial_score=0.0,
                scored=scored,
                has_segment_gt=bool(segment_gt is not None and segment_gt.has_segment),
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                method=method,
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(f"Completed case {idx}/{len(cases)}: {case.case_id} | method={method} | pred=<error> | gt={gt}")

    case_accuracy = (n_case_exact_correct / n_case_scored) if n_case_scored else 0.0
    per_question_metrics = {
        str(qid): {
            "name": QUESTION_NAMES[qid],
            "n_scored": q_scored[qid],
            "n_correct": q_correct[qid],
            "accuracy": (q_correct[qid] / q_scored[qid]) if q_scored[qid] else 0.0,
        }
        for qid in QUESTION_IDS
    }
    n_questions_scored_total = sum(q_scored.values())
    n_questions_correct_total = sum(q_correct.values())
    question_micro_accuracy = (n_questions_correct_total / n_questions_scored_total) if n_questions_scored_total else 0.0
    n_cases_with_segment_gt = evidence_case_metrics["n_cases_with_segment_gt"]
    case_exact_with_evidence_accuracy = (n_case_exact_with_evidence_correct / n_cases_with_segment_gt) if n_cases_with_segment_gt else 0.0
    case_with_evidence_partial_accuracy = (case_with_evidence_partial_score_total / n_cases_with_segment_gt) if n_cases_with_segment_gt else 0.0
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
        "script": SCRIPT_NAME,
        "method": method,
        "method_display_name": method_display_name(method),
        "method_config": _method_config_summary(args, method, selected_alias=args.series_alias),
        "data_root": str(data_root),
        "metadata_csv": str(metadata_csv),
        "labels_csv": str(labels_csv),
        "skip_amc": skip_amc,
        "n_amc_skipped": skipped_amc,
        "n_cases": len(cases),
        "n_case_scored": n_case_scored,
        "n_case_exact_correct": n_case_exact_correct,
        "case_exact_accuracy": case_accuracy,
        "n_case_exact_with_evidence_correct": n_case_exact_with_evidence_correct,
        "n_correct_with_evidence": n_correct_with_evidence,
        "case_exact_with_evidence_accuracy": case_exact_with_evidence_accuracy,
        "case_with_evidence_partial_accuracy": case_with_evidence_partial_accuracy,
        "n_questions_scored": n_questions_scored_total,
        "n_questions_correct": n_questions_correct_total,
        "question_accuracy": question_micro_accuracy,
        "question_micro_accuracy": question_micro_accuracy,
        "per_question": per_question_metrics,
        "avg_evidence_correct": avg_evidence_correct,
        "n_localization_correct": n_localization_correct,
        "Localization_acc": localization_acc,
        "localization_acc": localization_acc,
        "accuracy_with_evidence": case_exact_with_evidence_accuracy,
        "segment_metrics": {
            "available": True,
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
        "n_scored": n_case_scored,
        "n_correct": n_case_exact_correct,
        "accuracy": case_accuracy,
        "n_failed": n_failed,
        "total_tool_calls": total_tool_calls,
        "average_tool_calls_per_case": 0.0,
        "tool_call_counts": {},
        "tools_called": [],
        "n_unique_tools_used": 0,
        "series_alias": args.series_alias,
        "case_ids": _parse_case_ids(args.case_ids),
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_table_data(log_dir, build_nsclc_paper_table_data(summary))
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info(
        "DONE. method=%s question_accuracy=%.4f (%d/%d) case_exact_accuracy=%.4f (%d/%d) case_exact_with_evidence_accuracy=%.4f (%d/%d) case_with_evidence_partial_accuracy=%.4f failed=%d",
        method,
        question_micro_accuracy,
        n_questions_correct_total,
        n_questions_scored_total,
        case_accuracy,
        n_case_exact_correct,
        n_case_scored,
        case_exact_with_evidence_accuracy,
        n_case_exact_with_evidence_correct,
        n_case_scored,
        case_with_evidence_partial_accuracy,
        n_failed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
