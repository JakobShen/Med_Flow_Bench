from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from slicer_agent_engine.benchmarking.check_segment import (
    SegmentGroundTruth,
    discover_nifti_segmentation_file,
    discover_reference_nifti_files,
    load_segment_ground_truth,
)


def _load_nifti_array_and_affine(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Track3 NIfTI GT preparation requires nibabel. Install it in the Track3 environment, e.g. `pip install nibabel`."
        ) from e

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI segmentation at {path}, got shape {list(data.shape)}")
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"Unexpected affine shape for {path}: {affine.shape}")
    return data, affine


def _normalize_name(name: Optional[str]) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _reference_priority(node_name: str) -> int:
    norm = _normalize_name(node_name)
    ordered = (
        ("t1c", 0),
        ("t1 ce", 0),
        ("t1 gad", 0),
        ("post contrast", 0),
        ("contrast", 0),
        ("flair", 1),
        ("t2", 2),
        ("t1", 3),
        ("dwi", 4),
        ("adc", 4),
        ("swi", 5),
        ("asl", 6),
    )
    for token, score in ordered:
        if token in norm:
            return score
    return 99


def _geometry_matches(seg_shape: Tuple[int, int, int], seg_affine: np.ndarray, ref_path: Path) -> bool:
    try:
        ref_data, ref_affine = _load_nifti_array_and_affine(ref_path)
    except Exception:
        return False
    return tuple(int(v) for v in ref_data.shape) == tuple(int(v) for v in seg_shape) and bool(
        np.allclose(seg_affine, ref_affine, atol=1e-4, rtol=0.0)
    )


def _apply_matrix(matrix: np.ndarray, point_ijk: Sequence[float]) -> List[float]:
    vec = np.asarray([float(point_ijk[0]), float(point_ijk[1]), float(point_ijk[2]), 1.0], dtype=float)
    out = matrix @ vec
    return [float(out[0]), float(out[1]), float(out[2])]


def _normal_axis_label(normal: Sequence[float]) -> str:
    x, y, z = [float(v) for v in normal]
    abs_vals = [abs(x), abs(y), abs(z)]
    axis = int(max(range(3), key=lambda idx: abs_vals[idx]))
    if axis == 0:
        return "R" if x >= 0 else "L"
    if axis == 1:
        return "A" if y >= 0 else "P"
    return "S" if z >= 0 else "I"


def _slice_normal(view: str) -> List[float]:
    view_l = str(view).strip().lower()
    fallback = {
        "axial": [0.0, 0.0, 1.0],
        "sagittal": [-1.0, 0.0, 0.0],
        "coronal": [0.0, 1.0, 0.0],
    }
    if view_l not in fallback:
        raise ValueError(f"Unsupported view: {view!r}")
    return list(fallback[view_l])


def _projection_half_extent(axis_vectors: Dict[str, np.ndarray], normal_vec: Sequence[float]) -> float:
    n = np.asarray(normal_vec, dtype=float)
    return 0.5 * (
        abs(float(np.dot(axis_vectors["i"], n)))
        + abs(float(np.dot(axis_vectors["j"], n)))
        + abs(float(np.dot(axis_vectors["k"], n)))
    )


def _largest_connected_component(mask_ijk: np.ndarray) -> Tuple[np.ndarray, int, int]:
    mask_bool = np.asarray(mask_ijk > 0, dtype=bool)
    if not bool(mask_bool.any()):
        return np.zeros_like(mask_bool, dtype=np.uint8), 0, 0

    try:
        from scipy import ndimage as ndi  # type: ignore

        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labels, component_count = ndi.label(mask_bool.astype(np.uint8), structure=structure)
        if int(component_count) <= 0:
            return np.zeros_like(mask_bool, dtype=np.uint8), 0, 0
        counts = np.bincount(labels.ravel())
        if counts.size <= 1:
            return np.zeros_like(mask_bool, dtype=np.uint8), int(component_count), 0
        counts[0] = 0
        best_label = int(np.argmax(counts))
        best_count = int(counts[best_label])
        out = (labels == best_label).astype(np.uint8)
        return out, int(component_count), best_count
    except Exception:
        coords = np.argwhere(mask_bool)
        remaining = {(int(i), int(j), int(k)) for i, j, k in coords.tolist()}
        if not remaining:
            return np.zeros_like(mask_bool, dtype=np.uint8), 0, 0
        neighbors = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbors.append((di, dj, dk))
        component_count = 0
        best_component: List[Tuple[int, int, int]] = []
        while remaining:
            seed = remaining.pop()
            queue = deque([seed])
            component = [seed]
            while queue:
                i, j, k = queue.pop()
                for di, dj, dk in neighbors:
                    nb = (i + di, j + dj, k + dk)
                    if nb in remaining:
                        remaining.remove(nb)
                        queue.append(nb)
                        component.append(nb)
            component_count += 1
            if len(component) > len(best_component):
                best_component = component
        out = np.zeros_like(mask_bool, dtype=np.uint8)
        for i, j, k in best_component:
            out[int(i), int(j), int(k)] = 1
        return out, int(component_count), int(len(best_component))


def prepare_nifti_segment_ground_truth_py(
    *,
    case_path: Union[str, Path],
    out_dir: Union[str, Path],
    segment_path: Optional[Union[str, Path]] = None,
    reference_volume_paths: Optional[Sequence[Union[str, Path]]] = None,
    source_modality: Optional[str] = "MR",
    provenance_policy: str = "not_applicable",
) -> SegmentGroundTruth:
    case_path_p = Path(case_path).expanduser().resolve()
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    segment_path_p = (
        Path(segment_path).expanduser().resolve()
        if segment_path is not None
        else discover_nifti_segmentation_file(case_path_p)
    )
    if segment_path_p is None or not segment_path_p.exists():
        payload = {
            "ok": True,
            "case_path": str(case_path_p),
            "has_segment": False,
            "provenance_policy": provenance_policy,
            "notes": [f"No NIfTI segmentation file found under {case_path_p}."],
        }
        gt_json_path = out_dir_p / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        return load_segment_ground_truth(payload)

    if reference_volume_paths:
        ref_paths = []
        for path in reference_volume_paths:
            path_p = Path(path).expanduser().resolve()
            if path_p.exists() and path_p.is_file() and path_p != segment_path_p:
                ref_paths.append(path_p)
    else:
        ref_paths = [p for p in discover_reference_nifti_files(case_path_p) if p != segment_path_p]

    notes: List[str] = []
    seg_data_ijk, ijk_to_ras = _load_nifti_array_and_affine(segment_path_p)
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    positive_values = sorted(int(round(float(v))) for v in np.unique(seg_data_ijk[seg_data_ijk > 0]))
    if positive_values:
        notes.append(f"positive_labels={positive_values}")

    tumor_union_ijk = (seg_data_ijk > 0).astype(np.uint8, copy=False)
    voxel_count_total = int(np.count_nonzero(tumor_union_ijk))
    if voxel_count_total <= 0:
        payload = {
            "ok": True,
            "case_path": str(case_path_p),
            "has_segment": False,
            "provenance_policy": provenance_policy,
            "notes": notes + ["Segmentation volume is empty after positive-label union."],
        }
        gt_json_path = out_dir_p / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        return load_segment_ground_truth(payload)

    mask_ijk, component_count, largest_component_voxel_count = _largest_connected_component(tumor_union_ijk)
    if int(largest_component_voxel_count) <= 0:
        payload = {
            "ok": True,
            "case_path": str(case_path_p),
            "has_segment": False,
            "provenance_policy": provenance_policy,
            "notes": notes + ["Largest connected component extraction produced an empty mask."],
        }
        gt_json_path = out_dir_p / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        return load_segment_ground_truth(payload)

    geometry_matches = [p for p in ref_paths if _geometry_matches(mask_ijk.shape, ijk_to_ras, p)]
    geometry_matches = sorted(geometry_matches, key=lambda p: (_reference_priority(p.name), _normalize_name(p.name)))
    ref_volume = geometry_matches[0] if geometry_matches else None
    if geometry_matches:
        match_names = [str(path.name) for path in geometry_matches]
        if len(match_names) == 1:
            notes.append(f"Segmentation geometry uniquely matches reference volume: {match_names[0]}")
        else:
            notes.append(f"Segmentation geometry matches multiple reference volumes in a shared co-registered MRI space: {match_names}")
    else:
        notes.append("No geometry-matched reference MRI volume was found among the provided candidate volumes; GT uses the segmentation affine directly.")
    if str(provenance_policy).lower() in {"not_applicable", "none", "off"}:
        notes.append("Source-series provenance is not enforced for this NIfTI benchmark because the MRI volumes share a co-registered space.")

    nz_ijk = np.argwhere(mask_ijk > 0)
    imin, jmin, kmin = [int(v) for v in nz_ijk.min(axis=0)]
    imax, jmax, kmax = [int(v) for v in nz_ijk.max(axis=0)]

    ijk_points = np.column_stack(
        (
            nz_ijk[:, 0].astype(float),
            nz_ijk[:, 1].astype(float),
            nz_ijk[:, 2].astype(float),
            np.ones((nz_ijk.shape[0],), dtype=float),
        )
    )
    ras_points = (ijk_to_ras @ ijk_points.T).T[:, :3]
    rs = ras_points[:, 0]
    aas = ras_points[:, 1]
    ss = ras_points[:, 2]

    axis_vectors = {
        "i": np.asarray(ijk_to_ras[:3, 0], dtype=float),
        "j": np.asarray(ijk_to_ras[:3, 1], dtype=float),
        "k": np.asarray(ijk_to_ras[:3, 2], dtype=float),
    }
    slice_normals = {view: _slice_normal(view) for view in ("axial", "sagittal", "coronal")}
    key_slice_axis_labels = {view: _normal_axis_label(slice_normals[view]) for view in slice_normals}

    key_slice_ranges_mm: Dict[str, List[float]] = {}
    for view in ("axial", "sagittal", "coronal"):
        normal = np.asarray(slice_normals[view], dtype=float)
        offsets = ras_points @ normal
        delta = _projection_half_extent(axis_vectors, normal)
        key_slice_ranges_mm[view] = [float(offsets.min() - delta), float(offsets.max() + delta)]

    ras_axis_ranges_mm: Dict[str, List[float]] = {}
    for view, axis_vec, coords in (
        ("sagittal", np.asarray([1.0, 0.0, 0.0], dtype=float), rs),
        ("coronal", np.asarray([0.0, 1.0, 0.0], dtype=float), aas),
        ("axial", np.asarray([0.0, 0.0, 1.0], dtype=float), ss),
    ):
        delta = _projection_half_extent(axis_vectors, axis_vec)
        ras_axis_ranges_mm[view] = [float(coords.min() - delta), float(coords.max() + delta)]

    centroid = nz_ijk.mean(axis=0)
    nearest_idx = int(np.argmin(((nz_ijk - centroid) ** 2).sum(axis=1)))
    i0, j0, k0 = [int(v) for v in nz_ijk[nearest_idx]]
    representative_point_ras = _apply_matrix(ijk_to_ras, (i0, j0, k0))
    center_ras = _apply_matrix(ijk_to_ras, (float(centroid[0]), float(centroid[1]), float(centroid[2])))

    if int(component_count) > 1:
        notes.append(
            f"The segmentation contains {int(component_count)} disconnected components; benchmark GT uses only the largest component (voxel_count={int(largest_component_voxel_count)})."
        )

    mask_kji = np.transpose(mask_ijk.astype(np.uint8, copy=False), (2, 1, 0))
    mask_npz_path = out_dir_p / "segment_mask.npz"
    np.savez_compressed(str(mask_npz_path), mask=mask_kji)

    payload = {
        "ok": True,
        "case_path": str(case_path_p),
        "has_segment": True,
        "segmentation_name": str(segment_path_p.name),
        "segment_name": "tumor_union",
        "source_volume_name": str(ref_volume.name) if ref_volume is not None else None,
        "source_modality": str(source_modality or "") or None,
        "provenance_policy": provenance_policy,
        "voxel_count": int(largest_component_voxel_count),
        "segment_total_voxel_count": int(voxel_count_total),
        "component_count": int(component_count),
        "mask_shape_kji": [int(v) for v in mask_kji.shape],
        "mask_npz_path": str(mask_npz_path),
        "ras_to_ijk": ras_to_ijk.tolist(),
        "ijk_to_ras": ijk_to_ras.tolist(),
        "representative_point_ras": [float(v) for v in representative_point_ras],
        "center_ras": [float(v) for v in center_ras],
        "key_slice_axis_labels": key_slice_axis_labels,
        "key_slice_ranges_mm": key_slice_ranges_mm,
        "ras_axis_ranges_mm": ras_axis_ranges_mm,
        "ijk_bbox": {
            "i": [int(imin), int(imax)],
            "j": [int(jmin), int(jmax)],
            "k": [int(kmin), int(kmax)],
        },
        "notes": notes,
    }
    gt_json_path = out_dir_p / "segment_gt.json"
    gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["gt_json_path"] = str(gt_json_path)
    return load_segment_ground_truth(payload)


__all__ = ["prepare_nifti_segment_ground_truth_py"]
