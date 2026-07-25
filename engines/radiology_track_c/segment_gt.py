from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
            "Track C NIfTI GT preparation requires nibabel. Install it in the Track C environment, e.g. `pip install nibabel`."
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


def _functional_group_item(ds: Any, frame: Any, sequence_name: str) -> Optional[Any]:
    shared_groups = getattr(ds, "SharedFunctionalGroupsSequence", None)
    shared = shared_groups[0] if shared_groups else None
    for container in (frame, shared):
        if container is None:
            continue
        sequence = getattr(container, sequence_name, None)
        if sequence:
            return sequence[0]
    return None


def _dicom_segment_candidates(segment_path: Path) -> List[Dict[str, Any]]:
    try:
        import pydicom  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Track C DICOM SEG preparation requires pydicom. Install the track_c dependency profile."
        ) from e

    ds = pydicom.dcmread(str(segment_path), force=True)
    if str(getattr(ds, "Modality", "")).strip().upper() != "SEG":
        return []

    pixel_array = np.asarray(ds.pixel_array)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, :, :]
    if pixel_array.ndim != 3:
        raise ValueError(
            f"Expected DICOM SEG frames with shape [frame,row,column], got {list(pixel_array.shape)}"
        )

    per_frame = list(getattr(ds, "PerFrameFunctionalGroupsSequence", []) or [])
    if len(per_frame) != int(pixel_array.shape[0]):
        raise ValueError(
            f"DICOM SEG frame metadata count {len(per_frame)} does not match pixel frames {pixel_array.shape[0]}"
        )

    segment_items = {
        int(getattr(item, "SegmentNumber")): item
        for item in (getattr(ds, "SegmentSequence", []) or [])
        if getattr(item, "SegmentNumber", None) is not None
    }
    default_segment_number = next(iter(segment_items), 1)
    frame_groups: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    pixel_spacing_by_segment: Dict[int, Tuple[float, float]] = {}
    slice_spacing_by_segment: Dict[int, float] = {}

    for frame_index, frame in enumerate(per_frame):
        orientation_item = _functional_group_item(ds, frame, "PlaneOrientationSequence")
        position_item = _functional_group_item(ds, frame, "PlanePositionSequence")
        measures_item = _functional_group_item(ds, frame, "PixelMeasuresSequence")
        segment_item = _functional_group_item(ds, frame, "SegmentIdentificationSequence")
        if orientation_item is None or position_item is None or measures_item is None:
            raise ValueError("DICOM SEG is missing plane orientation, position, or pixel measures")

        orientation = np.asarray(
            [float(value) for value in orientation_item.ImageOrientationPatient],
            dtype=np.float64,
        )
        position = np.asarray(
            [float(value) for value in position_item.ImagePositionPatient],
            dtype=np.float64,
        )
        spacing = tuple(float(value) for value in measures_item.PixelSpacing)
        if orientation.shape != (6,) or position.shape != (3,) or len(spacing) != 2:
            raise ValueError("DICOM SEG contains invalid image-plane geometry")

        segment_number = int(
            getattr(segment_item, "ReferencedSegmentNumber", default_segment_number)
        )
        row_direction = orientation[:3]
        column_direction = orientation[3:]
        normal = np.cross(row_direction, column_direction)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            raise ValueError("DICOM SEG contains a degenerate image orientation")
        frame_groups.setdefault(segment_number, []).append(
            (
                np.asarray(pixel_array[frame_index] > 0, dtype=np.uint8),
                position,
                row_direction,
                column_direction,
            )
        )
        pixel_spacing_by_segment[segment_number] = (float(spacing[0]), float(spacing[1]))
        declared_slice_spacing = getattr(
            measures_item,
            "SpacingBetweenSlices",
            getattr(measures_item, "SliceThickness", 1.0),
        )
        slice_spacing_by_segment[segment_number] = abs(float(declared_slice_spacing or 1.0))

    candidates: List[Dict[str, Any]] = []
    for segment_number, frames in frame_groups.items():
        if not frames:
            continue
        row_direction = frames[0][2]
        column_direction = frames[0][3]
        normal = np.cross(row_direction, column_direction)
        normal = normal / float(np.linalg.norm(normal))
        for _, _, candidate_row, candidate_column in frames[1:]:
            if not np.allclose(candidate_row, row_direction, atol=1e-5, rtol=0.0):
                raise ValueError("DICOM SEG frame row orientations are inconsistent")
            if not np.allclose(candidate_column, column_direction, atol=1e-5, rtol=0.0):
                raise ValueError("DICOM SEG frame column orientations are inconsistent")

        ordered_frames = sorted(frames, key=lambda item: float(np.dot(normal, item[1])))
        offsets = np.asarray(
            [float(np.dot(normal, item[1])) for item in ordered_frames],
            dtype=np.float64,
        )
        positive_differences = np.diff(np.unique(np.round(offsets, decimals=6)))
        positive_differences = positive_differences[positive_differences > 1e-5]
        if positive_differences.size:
            slice_spacing = float(np.min(positive_differences))
        else:
            slice_spacing = float(slice_spacing_by_segment.get(segment_number, 1.0))
        if slice_spacing <= 0:
            raise ValueError("DICOM SEG contains a non-positive slice spacing")

        origin = np.asarray(ordered_frames[0][1], dtype=np.float64)
        origin_offset = float(offsets[0])
        k_indices = np.rint((offsets - origin_offset) / slice_spacing).astype(int)
        residuals = np.abs(offsets - (origin_offset + k_indices * slice_spacing))
        tolerance = max(0.1, 0.05 * slice_spacing)
        if residuals.size and float(residuals.max()) > tolerance:
            raise ValueError(
                f"DICOM SEG frame positions are not on a regular grid (max residual={float(residuals.max()):.4f} mm)"
            )

        rows, columns = [int(value) for value in ordered_frames[0][0].shape]
        mask_kji = np.zeros((int(k_indices.max()) + 1, rows, columns), dtype=np.uint8)
        for k_index, (frame_mask, _, _, _) in zip(k_indices.tolist(), ordered_frames):
            mask_kji[int(k_index)] |= np.asarray(frame_mask, dtype=np.uint8)
        mask_ijk = np.transpose(mask_kji, (2, 1, 0))
        voxel_count = int(np.count_nonzero(mask_ijk))
        if voxel_count <= 0:
            continue

        row_spacing, column_spacing = pixel_spacing_by_segment[segment_number]
        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[:3, 0] = row_direction * column_spacing
        affine_lps[:3, 1] = column_direction * row_spacing
        affine_lps[:3, 2] = normal * slice_spacing
        affine_lps[:3, 3] = origin
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine_ras = lps_to_ras @ affine_lps

        dicom_segment = segment_items.get(segment_number)
        label = str(getattr(dicom_segment, "SegmentLabel", "") or f"segment_{segment_number}")
        property_item = None
        if dicom_segment is not None:
            property_sequence = getattr(
                dicom_segment,
                "SegmentedPropertyTypeCodeSequence",
                None,
            )
            if property_sequence:
                property_item = property_sequence[0]
        property_meaning = str(getattr(property_item, "CodeMeaning", "") or "")
        normalized_description = _normalize_name(f"{label} {property_meaning}")
        score = voxel_count
        if any(
            token in normalized_description
            for token in ("tumor", "tumour", "lesion", "mass", "nodule", "primary", "gtv")
        ):
            score += 5_000_000
        if "upper arm" in normalized_description:
            score -= 10_000_000

        candidates.append(
            {
                "mask_ijk": mask_ijk,
                "affine_ras": affine_ras,
                "voxel_count": voxel_count,
                "score": score,
                "segment_number": segment_number,
                "segment_label": label,
                "property_meaning": property_meaning,
                "series_description": str(
                    getattr(ds, "SeriesDescription", "") or segment_path.parent.name
                ),
            }
        )
    return candidates


def prepare_dicom_segment_ground_truth_py(
    *,
    case_path: Union[str, Path],
    out_dir: Union[str, Path],
    segment_series_paths: Sequence[Union[str, Path]],
    source_modality: Optional[str] = "CT",
    provenance_policy: str = "not_applicable",
) -> SegmentGroundTruth:
    case_path_p = Path(case_path).expanduser().resolve()
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    dicom_files: List[Path] = []
    for raw_path in segment_series_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            dicom_files.append(path)
        elif path.is_dir():
            dicom_files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    dicom_files = sorted(set(dicom_files))

    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []
    for dicom_file in dicom_files:
        try:
            candidates.extend(_dicom_segment_candidates(dicom_file))
        except Exception as exc:
            errors.append(f"{dicom_file.name}: {exc}")

    if not candidates:
        payload = {
            "ok": True,
            "case_path": str(case_path_p),
            "has_segment": False,
            "provenance_policy": provenance_policy,
            "notes": ["No usable DICOM SEG tumor mask was found."] + errors,
        }
        gt_json_path = out_dir_p / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        return load_segment_ground_truth(payload)

    best = max(
        candidates,
        key=lambda candidate: (int(candidate["score"]), int(candidate["voxel_count"])),
    )
    generated_nifti = out_dir_p / "public_dicom_seg.nii.gz"
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Track C DICOM SEG preparation requires nibabel. Install the track_c dependency profile."
        ) from e
    nib.save(
        nib.Nifti1Image(
            np.asarray(best["mask_ijk"], dtype=np.uint8),
            np.asarray(best["affine_ras"], dtype=np.float64),
        ),
        str(generated_nifti),
    )

    result = prepare_nifti_segment_ground_truth_py(
        case_path=case_path_p,
        out_dir=out_dir_p,
        segment_path=generated_nifti,
        reference_volume_paths=None,
        source_modality=source_modality,
        provenance_policy=provenance_policy,
    )
    if result.gt_json_path is None:
        return result

    payload = json.loads(result.gt_json_path.read_text(encoding="utf-8"))
    label = str(best["segment_label"])
    property_meaning = str(best["property_meaning"])
    payload.update(
        {
            "segmentation_name": str(best["series_description"]),
            "segment_name": "public primary tumor segmentation",
            "source_volume_name": None,
            "source_modality": str(source_modality or "") or None,
            "dicom_segment_label": label,
            "dicom_segment_property": property_meaning,
            "dicom_segment_candidate_count": len(candidates),
        }
    )
    payload["notes"] = [
        (
            "Loaded public NSCLC-Radiogenomics DICOM SEG as the primary tumor mask; "
            "the collection contains known generic or incorrect SegmentLabel values, "
            "so selection does not depend on the label text."
        ),
        f"dicom_segment_label={label!r}",
        f"dicom_segment_property={property_meaning!r}",
        f"dicom_segment_candidates={len(candidates)}",
    ] + list(payload.get("notes") or []) + errors
    payload["gt_json_path"] = str(result.gt_json_path)
    result.gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return load_segment_ground_truth(payload)


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


__all__ = [
    "prepare_dicom_segment_ground_truth_py",
    "prepare_nifti_segment_ground_truth_py",
]
