from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


_SEGMENT_NAME_POSITIVE_HINTS = (
    "tumor",
    "tumour",
    "lesion",
    "mass",
    "nodule",
    "primary",
    "gtv",
    "target",
)

_SEGMENT_NAME_NEGATIVE_HINTS = (
    "lung",
    "lobe",
    "body",
    "background",
    "airway",
    "vessel",
    "bronch",
    "whole",
)

_NIFTI_SEGMENTATION_NAME_HINTS = (
    "tumor_segmentation",
    "tumour_segmentation",
    "segmentation",
    "tumor_mask",
    "tumour_mask",
    "mask",
    "label",
    "seg",
)

_CANONICAL_VIEWS = ("axial", "sagittal", "coronal")
_VIEW_TO_RAS_LABEL = {
    "axial": "S",
    "sagittal": "R",
    "coronal": "A",
}

_KEY_SLICE_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "axial": [
        re.compile(r"KEY[\s_-]*SLICE[\s_-]*AXIAL(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"AXIAL[\s_-]*KEY[\s_-]*SLICE(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    ],
    "sagittal": [
        re.compile(r"KEY[\s_-]*SLICE[\s_-]*SAGITTAL(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"SAGITTAL[\s_-]*KEY[\s_-]*SLICE(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    ],
    "coronal": [
        re.compile(r"KEY[\s_-]*SLICE[\s_-]*CORONAL(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"CORONAL[\s_-]*KEY[\s_-]*SLICE(?:[\s_-]*MM)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    ],
}

_POINT_LINE_PATTERNS = [
    re.compile(r"\bRSA\b\s*[:=]\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bRAS\b\s*[:=]\s*(.+)$", re.IGNORECASE),
]

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class SegmentGroundTruth:
    case_path: Optional[Path]
    has_segment: bool
    gt_json_path: Optional[Path]
    mask_npz_path: Optional[Path]
    segmentation_name: Optional[str]
    segment_name: Optional[str]
    source_volume_name: Optional[str]
    source_modality: Optional[str]
    provenance_policy: str = "exact_source"
    key_slice_ranges_mm: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    key_slice_axis_labels: Dict[str, str] = field(default_factory=dict)
    ras_axis_ranges_mm: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    representative_point_ras: Optional[Tuple[float, float, float]] = None
    center_ras: Optional[Tuple[float, float, float]] = None
    ras_to_ijk: Optional[Tuple[Tuple[float, float, float, float], ...]] = None
    ijk_to_ras: Optional[Tuple[Tuple[float, float, float, float], ...]] = None
    mask_shape_kji: Optional[Tuple[int, int, int]] = None
    voxel_count: Optional[int] = None
    segment_total_voxel_count: Optional[int] = None
    component_count: Optional[int] = None
    notes: Tuple[str, ...] = ()

    def range_for(self, view: str) -> Optional[Tuple[float, float]]:
        return self.key_slice_ranges_mm.get(str(view).lower())

    def ras_axis_range_for(self, view: str) -> Optional[Tuple[float, float]]:
        return self.ras_axis_ranges_mm.get(str(view).lower())

    def axis_label_for(self, view: str) -> Optional[str]:
        label = self.key_slice_axis_labels.get(str(view).lower())
        return str(label) if label else None

    def to_prompt_spec(self) -> Dict[str, Any]:
        return {
            "has_segment": bool(self.has_segment),
            "source_volume_name": self.source_volume_name,
            "source_modality": self.source_modality,
            "segmentation_name": self.segmentation_name,
            "segment_name": self.segment_name,
            "provenance_policy": self.provenance_policy,
            "largest_component_policy": bool(self.component_count and int(self.component_count) > 1),
            "key_slice_convention": "canonical_ras_axis_mm",
        }


@dataclass(frozen=True)
class SegmentEvidencePrediction:
    key_slice_mm: Dict[str, Optional[float]]
    point_ras: Optional[Tuple[float, float, float]]
    raw_text: str


def _normalize_volume_name(name: Optional[str]) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def infer_modality_from_name(name: Optional[str]) -> Optional[str]:
    norm = _normalize_volume_name(name)
    if not norm:
        return None
    if any(tok in norm for tok in ("pet", "suv", "uptake", "ramla", "wb")):
        return "PT"
    if any(tok in norm for tok in ("ct", "thorax", "lung", "mediast")):
        return "CT"
    if any(tok in norm for tok in ("mri", "mr", "t1", "t1c", "t1ce", "flair", "t2", "dwi", "swi", "asl", "adc", "diffusion", "perfusion")):
        return "MR"
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_triplet(value: Sequence[Any]) -> Optional[Tuple[float, float, float]]:
    if len(value) != 3:
        return None
    nums = [_parse_float(v) for v in value]
    if any(v is None for v in nums):
        return None
    return (float(nums[0]), float(nums[1]), float(nums[2]))


def _parse_range_map(payload: Mapping[str, Any], field_name: str) -> Dict[str, Tuple[float, float]]:
    raw_map = payload.get(field_name) or {}
    out: Dict[str, Tuple[float, float]] = {}
    for view in _CANONICAL_VIEWS:
        raw = raw_map.get(view)
        if not isinstance(raw, Sequence) or len(raw) != 2:
            continue
        lo = _parse_float(raw[0])
        hi = _parse_float(raw[1])
        if lo is None or hi is None:
            continue
        lo_f, hi_f = sorted((float(lo), float(hi)))
        out[view] = (lo_f, hi_f)
    return out


def load_segment_ground_truth(payload_or_path: Union[Mapping[str, Any], str, Path]) -> SegmentGroundTruth:
    if isinstance(payload_or_path, Mapping):
        payload = dict(payload_or_path)
        gt_json_path = Path(str(payload.get("gt_json_path"))).expanduser().resolve() if payload.get("gt_json_path") else None
    else:
        gt_json_path = Path(payload_or_path).expanduser().resolve()
        with gt_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    case_path = Path(str(payload.get("case_path"))).expanduser().resolve() if payload.get("case_path") else None
    mask_npz_path = Path(str(payload.get("mask_npz_path"))).expanduser().resolve() if payload.get("mask_npz_path") else None

    key_slice_ranges_mm = _parse_range_map(payload, "key_slice_ranges_mm")
    ras_axis_ranges_mm = _parse_range_map(payload, "ras_axis_ranges_mm")

    key_slice_axis_labels_raw = payload.get("key_slice_axis_labels") or {}
    key_slice_axis_labels = {
        view: str(key_slice_axis_labels_raw.get(view)).strip().upper()
        for view in _CANONICAL_VIEWS
        if str(key_slice_axis_labels_raw.get(view) or "").strip()
    }

    rep = None
    center = None
    if isinstance(payload.get("representative_point_ras"), Sequence):
        rep = _as_triplet(payload["representative_point_ras"])
    if isinstance(payload.get("center_ras"), Sequence):
        center = _as_triplet(payload["center_ras"])

    ras_to_ijk = None
    ijk_to_ras = None
    if isinstance(payload.get("ras_to_ijk"), Sequence) and len(payload["ras_to_ijk"]) == 4:
        ras_to_ijk = tuple(tuple(float(v) for v in row) for row in payload["ras_to_ijk"])
    if isinstance(payload.get("ijk_to_ras"), Sequence) and len(payload["ijk_to_ras"]) == 4:
        ijk_to_ras = tuple(tuple(float(v) for v in row) for row in payload["ijk_to_ras"])

    shape = None
    if isinstance(payload.get("mask_shape_kji"), Sequence) and len(payload["mask_shape_kji"]) == 3:
        try:
            shape = tuple(int(v) for v in payload["mask_shape_kji"])
        except Exception:
            shape = None

    notes_raw = payload.get("notes") or []
    notes = tuple(str(x) for x in notes_raw) if isinstance(notes_raw, Sequence) and not isinstance(notes_raw, (str, bytes)) else ()

    return SegmentGroundTruth(
        case_path=case_path,
        has_segment=bool(payload.get("has_segment", False)),
        gt_json_path=gt_json_path,
        mask_npz_path=mask_npz_path,
        segmentation_name=str(payload.get("segmentation_name") or "") or None,
        segment_name=str(payload.get("segment_name") or "") or None,
        source_volume_name=str(payload.get("source_volume_name") or "") or None,
        source_modality=str(payload.get("source_modality") or "") or None,
        provenance_policy=str(payload.get("provenance_policy") or "exact_source"),
        key_slice_ranges_mm=key_slice_ranges_mm,
        key_slice_axis_labels=key_slice_axis_labels,
        ras_axis_ranges_mm=ras_axis_ranges_mm,
        representative_point_ras=rep,
        center_ras=center,
        ras_to_ijk=ras_to_ijk,
        ijk_to_ras=ijk_to_ras,
        mask_shape_kji=shape,
        voxel_count=int(payload["voxel_count"]) if payload.get("voxel_count") is not None else None,
        segment_total_voxel_count=int(payload["segment_total_voxel_count"]) if payload.get("segment_total_voxel_count") is not None else None,
        component_count=int(payload["component_count"]) if payload.get("component_count") is not None else None,
        notes=notes,
    )


def _segment_name_bonus(name: str) -> int:
    name_l = str(name or "").lower()
    bonus = 0
    if any(tok in name_l for tok in _SEGMENT_NAME_POSITIVE_HINTS):
        bonus += 5_000_000
    if any(tok in name_l for tok in _SEGMENT_NAME_NEGATIVE_HINTS):
        bonus -= 3_000_000
    return bonus


def _is_nifti_path(path: Union[str, Path]) -> bool:
    name = str(path).lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _looks_like_nifti_segmentation_name(path: Union[str, Path]) -> bool:
    name = Path(str(path)).name.lower()
    if not _is_nifti_path(name):
        return False
    return any(token in name for token in _NIFTI_SEGMENTATION_NAME_HINTS)


def discover_nifti_segmentation_file(case_path: Union[str, Path]) -> Optional[Path]:
    case_path_p = Path(case_path).expanduser().resolve()
    if case_path_p.is_file():
        return case_path_p if _looks_like_nifti_segmentation_name(case_path_p) else None
    if not case_path_p.exists() or not case_path_p.is_dir():
        return None

    candidates = [p for p in sorted(case_path_p.iterdir()) if p.is_file() and _looks_like_nifti_segmentation_name(p)]
    if not candidates:
        return None

    def _score(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        score = 0
        if "tumor_segmentation" in name or "tumour_segmentation" in name:
            score += 100
        if "segmentation" in name:
            score += 50
        if "tumor" in name or "tumour" in name:
            score += 20
        if "mask" in name:
            score += 10
        if "label" in name:
            score += 5
        return (-score, name)

    candidates.sort(key=_score)
    return candidates[0]


def discover_reference_nifti_files(case_path: Union[str, Path]) -> List[Path]:
    case_path_p = Path(case_path).expanduser().resolve()
    if case_path_p.is_file():
        return []
    if not case_path_p.exists() or not case_path_p.is_dir():
        return []
    return [
        p
        for p in sorted(case_path_p.iterdir())
        if p.is_file() and _is_nifti_path(p) and not _looks_like_nifti_segmentation_name(p)
    ]


def _build_slicer_segment_gt_script(*, case_path: Path, out_dir: Path) -> str:
    case_path_json = json.dumps(str(case_path), ensure_ascii=False)
    out_dir_json = json.dumps(str(out_dir), ensure_ascii=False)
    positive_hints = json.dumps(list(_SEGMENT_NAME_POSITIVE_HINTS), ensure_ascii=False)
    negative_hints = json.dumps(list(_SEGMENT_NAME_NEGATIVE_HINTS), ensure_ascii=False)
    return rf"""
import json
import math
import pathlib
from collections import deque

import numpy as np
import slicer

case_path = pathlib.Path({case_path_json})
out_dir = pathlib.Path({out_dir_json})
out_dir.mkdir(parents=True, exist_ok=True)
POSITIVE_HINTS = tuple({positive_hints})
NEGATIVE_HINTS = tuple({negative_hints})


def _normalize_name(name):
    import re
    text = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower())
    return re.sub(r"\\s+", " ", text).strip()


def _infer_modality(node):
    if node is None:
        return None
    for key in ("DICOM.Modality", "Modality"):
        try:
            value = node.GetAttribute(key)
        except Exception:
            value = None
        if value:
            return str(value).upper()
    name = _normalize_name(node.GetName() if node is not None else None)
    if any(tok in name for tok in ("pet", "suv", "uptake", "ramla", "wb")):
        return "PT"
    if any(tok in name for tok in ("ct", "thorax", "lung", "mediast")):
        return "CT"
    return None


def _segment_ids(seg_node):
    seg = seg_node.GetSegmentation()
    out = []
    for i in range(int(seg.GetNumberOfSegments())):
        sid = seg.GetNthSegmentID(i)
        if sid:
            out.append(str(sid))
    return out


def _name_bonus(name):
    name_l = str(name or "").lower()
    bonus = 0
    if any(tok in name_l for tok in POSITIVE_HINTS):
        bonus += 5000000
    if any(tok in name_l for tok in NEGATIVE_HINTS):
        bonus -= 3000000
    return bonus


def _scalar_volumes():
    try:
        return list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
    except Exception:
        return []


def _candidate_reference_volumes(seg_node):
    out = []
    role = None
    try:
        role = seg_node.GetReferenceImageGeometryReferenceRole()
    except Exception:
        role = "referenceImageGeometryRef"
    ref = None
    if role:
        try:
            ref = seg_node.GetNodeReference(role)
        except Exception:
            ref = None
    if ref is not None and hasattr(ref, "IsA") and ref.IsA("vtkMRMLScalarVolumeNode"):
        out.append(ref)
    for node in _scalar_volumes():
        if ref is not None and node.GetID() == ref.GetID():
            continue
        out.append(node)
    return out


def _mat_to_list(node, which):
    import vtk
    mat = vtk.vtkMatrix4x4()
    if which == "ijk_to_ras":
        node.GetIJKToRASMatrix(mat)
    else:
        node.GetRASToIJKMatrix(mat)
    return [[float(mat.GetElement(r, c)) for c in range(4)] for r in range(4)]


def _apply_matrix(matrix, point_xyz):
    vals = [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0]
    out = []
    for r in range(4):
        total = 0.0
        for c in range(4):
            total += float(matrix[r][c]) * vals[c]
        out.append(total)
    return [float(out[0]), float(out[1]), float(out[2])]


def _slice_node(view):
    lm = slicer.app.layoutManager()
    if lm is None:
        raise RuntimeError("No layout manager available")
    name = {{"axial": "Red", "sagittal": "Yellow", "coronal": "Green"}}[str(view)]
    widget = lm.sliceWidget(name)
    if widget is None:
        raise RuntimeError(f"Slice widget unavailable for {{view}}")
    node = widget.mrmlSliceNode()
    try:
        if str(view) == "axial" and hasattr(node, "SetOrientationToAxial"):
            node.SetOrientationToAxial()
        elif str(view) == "sagittal" and hasattr(node, "SetOrientationToSagittal"):
            node.SetOrientationToSagittal()
        elif str(view) == "coronal" and hasattr(node, "SetOrientationToCoronal"):
            node.SetOrientationToCoronal()
        elif hasattr(node, "SetOrientation"):
            node.SetOrientation(str(view).capitalize())
        slicer.app.processEvents()
    except Exception:
        pass
    return node


def _activate_reference_volume(volume_node):
    if volume_node is None:
        return
    try:
        app_logic = slicer.app.applicationLogic()
        sel = app_logic.GetSelectionNode() if app_logic is not None else None
        if sel is not None and hasattr(sel, "SetReferenceActiveVolumeID"):
            sel.SetReferenceActiveVolumeID(volume_node.GetID())
        if app_logic is not None:
            app_logic.PropagateVolumeSelection(0)
    except Exception:
        pass
    try:
        slicer.util.setSliceViewerLayers(background=volume_node, foreground=None, label=None, fit=False)
    except Exception:
        pass
    lm = slicer.app.layoutManager()
    if lm is not None:
        for view, name in (("axial", "Red"), ("sagittal", "Yellow"), ("coronal", "Green")):
            try:
                widget = lm.sliceWidget(name)
                if widget is None:
                    continue
                node = widget.mrmlSliceNode()
                if str(view) == "axial" and hasattr(node, "SetOrientationToAxial"):
                    node.SetOrientationToAxial()
                elif str(view) == "sagittal" and hasattr(node, "SetOrientationToSagittal"):
                    node.SetOrientationToSagittal()
                elif str(view) == "coronal" and hasattr(node, "SetOrientationToCoronal"):
                    node.SetOrientationToCoronal()
                logic = widget.sliceLogic() if hasattr(widget, "sliceLogic") else None
                if logic is not None and hasattr(logic, "FitSliceToAll"):
                    logic.FitSliceToAll()
            except Exception:
                pass
    try:
        slicer.app.processEvents()
    except Exception:
        pass


def _slice_normal(view):
    sn = _slice_node(view)
    try:
        m = sn.GetSliceToRAS()
        n = [float(m.GetElement(0, 2)), float(m.GetElement(1, 2)), float(m.GetElement(2, 2))]
        mag = math.sqrt(sum(v * v for v in n))
        if mag > 0:
            return [float(v / mag) for v in n]
    except Exception:
        pass
    fallback = {{
        "axial": [0.0, 0.0, 1.0],
        "sagittal": [-1.0, 0.0, 0.0],
        "coronal": [0.0, 1.0, 0.0],
    }}
    return list(fallback[str(view)])


def _normal_axis_label(normal):
    x, y, z = [float(v) for v in normal]
    abs_vals = [abs(x), abs(y), abs(z)]
    axis = int(max(range(3), key=lambda idx: abs_vals[idx]))
    if axis == 0:
        return "R" if x >= 0 else "L"
    if axis == 1:
        return "A" if y >= 0 else "P"
    return "S" if z >= 0 else "I"


def _project_point(point_ras, normal):
    return float(point_ras[0]) * float(normal[0]) + float(point_ras[1]) * float(normal[1]) + float(point_ras[2]) * float(normal[2])


def _largest_connected_component(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return np.zeros_like(mask, dtype=np.uint8), 0, 0
    remaining = {{(int(k), int(j), int(i)) for k, j, i in coords.tolist()}}
    neighbors = []
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dk == 0 and dj == 0 and di == 0:
                    continue
                neighbors.append((dk, dj, di))
    largest = []
    component_count = 0
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        comp = [seed]
        while queue:
            k, j, i = queue.pop()
            for dk, dj, di in neighbors:
                nb = (k + dk, j + dj, i + di)
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)
                    comp.append(nb)
        component_count += 1
        if len(comp) > len(largest):
            largest = comp
    out = np.zeros_like(mask, dtype=np.uint8)
    for k, j, i in largest:
        out[int(k), int(j), int(i)] = 1
    return out, int(component_count), int(len(largest))


best = None
notes = []
for seg_node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
    for sid in _segment_ids(seg_node):
        segment = seg_node.GetSegmentation().GetSegment(sid)
        segment_name = segment.GetName() if segment is not None else sid
        segment_best = None
        for idx, ref_volume in enumerate(_candidate_reference_volumes(seg_node)):
            try:
                mask = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, sid, ref_volume)
            except Exception as exc:
                notes.append(f"arrayFromSegmentBinaryLabelmap failed for {{segment_name}} on {{ref_volume.GetName() if ref_volume is not None else 'None'}}: {{exc}}")
                continue
            if mask is None:
                continue
            mask = np.asarray(mask)
            voxel_count_total = int(np.count_nonzero(mask))
            if voxel_count_total <= 0:
                continue
            largest_mask, component_count, largest_component_voxel_count = _largest_connected_component(mask)
            if largest_component_voxel_count <= 0:
                continue
            score = int(largest_component_voxel_count) + _name_bonus(segment_name) + (1000000 if idx == 0 else 0)
            candidate = {{
                "segmentation_node": seg_node,
                "segment_id": str(sid),
                "segment_name": str(segment_name),
                "segment_total_voxel_count": voxel_count_total,
                "largest_component_voxel_count": int(largest_component_voxel_count),
                "component_count": int(component_count),
                "mask": largest_mask.astype(np.uint8, copy=False),
                "reference_volume": ref_volume,
                "score": score,
            }}
            if segment_best is None or candidate["score"] > segment_best["score"]:
                segment_best = candidate
        if segment_best is None:
            continue
        if best is None or segment_best["score"] > best["score"]:
            best = segment_best

if best is None:
    payload = {{
        "ok": True,
        "case_path": str(case_path),
        "has_segment": False,
        "notes": notes,
    }}
    gt_json_path = out_dir / "segment_gt.json"
    gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["gt_json_path"] = str(gt_json_path)
    __execResult = payload
else:
    mask = best["mask"]
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        payload = {{
            "ok": True,
            "case_path": str(case_path),
            "has_segment": False,
            "notes": notes + ["Best segment mask was empty after extraction."],
        }}
        gt_json_path = out_dir / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        __execResult = payload
    else:
        ref_volume = best["reference_volume"]
        _activate_reference_volume(ref_volume)
        ijk_to_ras = _mat_to_list(ref_volume, "ijk_to_ras")
        ras_to_ijk = _mat_to_list(ref_volume, "ras_to_ijk")
        ijk_to_ras_arr = np.asarray(ijk_to_ras, dtype=float)

        kmin, jmin, imin = [int(v) for v in nz.min(axis=0)]
        kmax, jmax, imax = [int(v) for v in nz.max(axis=0)]

        ijk_points = np.column_stack((nz[:, 2].astype(float), nz[:, 1].astype(float), nz[:, 0].astype(float), np.ones((nz.shape[0],), dtype=float)))
        ras_points = (ijk_to_ras_arr @ ijk_points.T).T[:, :3]
        rs = ras_points[:, 0]
        aas = ras_points[:, 1]
        ss = ras_points[:, 2]

        axis_vectors = {{
            "i": ijk_to_ras_arr[:3, 0],
            "j": ijk_to_ras_arr[:3, 1],
            "k": ijk_to_ras_arr[:3, 2],
        }}

        def _projection_half_extent(normal_vec):
            n = np.asarray(normal_vec, dtype=float)
            return 0.5 * (
                abs(float(np.dot(axis_vectors["i"], n)))
                + abs(float(np.dot(axis_vectors["j"], n)))
                + abs(float(np.dot(axis_vectors["k"], n)))
            )

        slice_normals = {{view: _slice_normal(view) for view in ("axial", "sagittal", "coronal")}}
        key_slice_axis_labels = {{view: _normal_axis_label(slice_normals[view]) for view in slice_normals}}
        key_slice_ranges_mm = {{}}
        for view in ("axial", "sagittal", "coronal"):
            normal = np.asarray(slice_normals[view], dtype=float)
            offsets = ras_points @ normal
            delta = _projection_half_extent(normal)
            key_slice_ranges_mm[view] = [float(offsets.min() - delta), float(offsets.max() + delta)]

        ras_axis_ranges_mm = {{}}
        for view, axis_vec in (("sagittal", np.asarray([1.0, 0.0, 0.0], dtype=float)), ("coronal", np.asarray([0.0, 1.0, 0.0], dtype=float)), ("axial", np.asarray([0.0, 0.0, 1.0], dtype=float))):
            delta = _projection_half_extent(axis_vec)
            if view == "sagittal":
                coords = rs
            elif view == "coronal":
                coords = aas
            else:
                coords = ss
            ras_axis_ranges_mm[view] = [float(coords.min() - delta), float(coords.max() + delta)]

        centroid = nz.mean(axis=0)
        nearest_idx = int(np.argmin(((nz - centroid) ** 2).sum(axis=1)))
        k0, j0, i0 = [int(v) for v in nz[nearest_idx]]
        representative_point_ras = _apply_matrix(ijk_to_ras, (i0, j0, k0))
        center_ras = _apply_matrix(ijk_to_ras, (float(centroid[2]), float(centroid[1]), float(centroid[0])))

        if int(best["component_count"]) > 1:
            notes.append(
                f"Selected segment '{{best['segment_name']}}' contains {{int(best['component_count'])}} disconnected components; benchmark GT uses only the largest component (voxel_count={{int(best['largest_component_voxel_count'])}})."
            )

        mask_npz_path = out_dir / "segment_mask.npz"
        np.savez_compressed(str(mask_npz_path), mask=mask.astype(np.uint8, copy=False))

        payload = {{
            "ok": True,
            "case_path": str(case_path),
            "has_segment": True,
            "segmentation_name": str(best["segmentation_node"].GetName()),
            "segment_name": str(best["segment_name"]),
            "source_volume_name": str(ref_volume.GetName()),
            "source_modality": _infer_modality(ref_volume),
            "voxel_count": int(best["largest_component_voxel_count"]),
            "segment_total_voxel_count": int(best["segment_total_voxel_count"]),
            "component_count": int(best["component_count"]),
            "mask_shape_kji": [int(v) for v in mask.shape],
            "mask_npz_path": str(mask_npz_path),
            "ras_to_ijk": ras_to_ijk,
            "ijk_to_ras": ijk_to_ras,
            "representative_point_ras": [float(v) for v in representative_point_ras],
            "center_ras": [float(v) for v in center_ras],
            "key_slice_axis_labels": key_slice_axis_labels,
            "key_slice_ranges_mm": key_slice_ranges_mm,
            "ras_axis_ranges_mm": ras_axis_ranges_mm,
            "ijk_bbox": {{
                "i": [int(imin), int(imax)],
                "j": [int(jmin), int(jmax)],
                "k": [int(kmin), int(kmax)],
            }},
            "notes": notes,
        }}
        gt_json_path = out_dir / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        __execResult = payload
"""


def prepare_segment_ground_truth(
    ctx: Any,
    *,
    case_path: Union[str, Path],
    out_dir: Union[str, Path],
    clear_scene_after: bool = True,
) -> SegmentGroundTruth:
    case_path_p = Path(case_path).expanduser().resolve()
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Hidden GT loading: import the full case root so any DICOM SEG series can be discovered.
    ctx.load_dicom(case_path_p, clear_scene_first=True, active_prefer=["ct", "thorax", "lung", "pet", "suv"])
    script = _build_slicer_segment_gt_script(case_path=case_path_p, out_dir=out_dir_p)
    result = ctx.client.exec_python(script, timeout_s=120.0)
    if clear_scene_after:
        try:
            ctx.clear_scene()
        except Exception:
            pass
    return load_segment_ground_truth(result)




def _build_slicer_nifti_segment_gt_script(
    *,
    case_path: Path,
    out_dir: Path,
    segment_path: Path,
    source_modality: Optional[str] = None,
    provenance_policy: str = "not_applicable",
) -> str:
    case_path_json = json.dumps(str(case_path), ensure_ascii=False)
    out_dir_json = json.dumps(str(out_dir), ensure_ascii=False)
    segment_path_json = json.dumps(str(segment_path), ensure_ascii=False)
    source_modality_json = json.dumps(str(source_modality or "MR"), ensure_ascii=False)
    provenance_policy_json = json.dumps(str(provenance_policy or "not_applicable"), ensure_ascii=False)
    positive_hints = json.dumps(list(_SEGMENT_NAME_POSITIVE_HINTS), ensure_ascii=False)
    return rf"""
import json
import math
import pathlib
from collections import deque

import numpy as np
import slicer

case_path = pathlib.Path({case_path_json})
out_dir = pathlib.Path({out_dir_json})
segment_path = pathlib.Path({segment_path_json})
out_dir.mkdir(parents=True, exist_ok=True)
SOURCE_MODALITY = {source_modality_json}
PROVENANCE_POLICY = {provenance_policy_json}
POSITIVE_HINTS = tuple({positive_hints})


def _normalize_name(name):
    import re
    text = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _strip_nii_suffix(name):
    text = str(name or "")
    lower = text.lower()
    if lower.endswith('.nii.gz'):
        return text[:-7]
    if lower.endswith('.nii'):
        return text[:-4]
    return text


SEGMENT_HINT = _normalize_name(_strip_nii_suffix(segment_path.name))


def _scalar_or_label_volumes():
    out = []
    seen = set()
    for class_name in ("vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"):
        try:
            nodes = list(slicer.util.getNodesByClass(class_name))
        except Exception:
            nodes = []
        for node in nodes:
            node_id = node.GetID() if node is not None and hasattr(node, 'GetID') else None
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            out.append(node)
    return out


def _volume_name_score(node_name):
    norm = _normalize_name(node_name)
    score = 0
    if SEGMENT_HINT and SEGMENT_HINT in norm:
        score += 1000
    if "tumor segmentation" in norm or "tumour segmentation" in norm:
        score += 500
    if "segmentation" in norm:
        score += 300
    if "tumor" in norm or "tumour" in norm:
        score += 80
    if "mask" in norm:
        score += 40
    if "label" in norm:
        score += 20
    if any(tok in norm for tok in POSITIVE_HINTS):
        score += 10
    if any(tok in norm for tok in ("t1", "t1c", "t1ce", "flair", "t2", "dwi", "swi", "asl", "adc", "diffusion", "perfusion")):
        score -= 200
    return score


def _mat_to_list(node, which):
    import vtk
    mat = vtk.vtkMatrix4x4()
    if which == "ijk_to_ras":
        node.GetIJKToRASMatrix(mat)
    else:
        node.GetRASToIJKMatrix(mat)
    return [[float(mat.GetElement(r, c)) for c in range(4)] for r in range(4)]


def _apply_matrix(matrix, point_xyz):
    vals = [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0]
    out = []
    for r in range(4):
        total = 0.0
        for c in range(4):
            total += float(matrix[r][c]) * vals[c]
        out.append(total)
    return [float(out[0]), float(out[1]), float(out[2])]


def _slice_node(view):
    lm = slicer.app.layoutManager()
    if lm is None:
        raise RuntimeError("No layout manager available")
    name = {{"axial": "Red", "sagittal": "Yellow", "coronal": "Green"}}[str(view)]
    widget = lm.sliceWidget(name)
    if widget is None:
        raise RuntimeError(f"Slice widget unavailable for {{view}}")
    node = widget.mrmlSliceNode()
    try:
        if str(view) == "axial" and hasattr(node, "SetOrientationToAxial"):
            node.SetOrientationToAxial()
        elif str(view) == "sagittal" and hasattr(node, "SetOrientationToSagittal"):
            node.SetOrientationToSagittal()
        elif str(view) == "coronal" and hasattr(node, "SetOrientationToCoronal"):
            node.SetOrientationToCoronal()
        elif hasattr(node, "SetOrientation"):
            node.SetOrientation(str(view).capitalize())
        slicer.app.processEvents()
    except Exception:
        pass
    return node


def _activate_reference_volume(volume_node):
    if volume_node is None:
        return
    try:
        app_logic = slicer.app.applicationLogic()
        sel = app_logic.GetSelectionNode() if app_logic is not None else None
        if sel is not None and hasattr(sel, "SetReferenceActiveVolumeID"):
            sel.SetReferenceActiveVolumeID(volume_node.GetID())
        if app_logic is not None:
            app_logic.PropagateVolumeSelection(0)
    except Exception:
        pass
    try:
        slicer.util.setSliceViewerLayers(background=volume_node, foreground=None, label=None, fit=False)
    except Exception:
        pass
    lm = slicer.app.layoutManager()
    if lm is not None:
        for view, name in (("axial", "Red"), ("sagittal", "Yellow"), ("coronal", "Green")):
            try:
                widget = lm.sliceWidget(name)
                if widget is None:
                    continue
                node = widget.mrmlSliceNode()
                if str(view) == "axial" and hasattr(node, "SetOrientationToAxial"):
                    node.SetOrientationToAxial()
                elif str(view) == "sagittal" and hasattr(node, "SetOrientationToSagittal"):
                    node.SetOrientationToSagittal()
                elif str(view) == "coronal" and hasattr(node, "SetOrientationToCoronal"):
                    node.SetOrientationToCoronal()
                logic = widget.sliceLogic() if hasattr(widget, "sliceLogic") else None
                if logic is not None and hasattr(logic, "FitSliceToAll"):
                    logic.FitSliceToAll()
            except Exception:
                pass
    try:
        slicer.app.processEvents()
    except Exception:
        pass


def _slice_normal(view):
    sn = _slice_node(view)
    try:
        m = sn.GetSliceToRAS()
        n = [float(m.GetElement(0, 2)), float(m.GetElement(1, 2)), float(m.GetElement(2, 2))]
        mag = math.sqrt(sum(v * v for v in n))
        if mag > 0:
            return [float(v / mag) for v in n]
    except Exception:
        pass
    fallback = {{
        "axial": [0.0, 0.0, 1.0],
        "sagittal": [-1.0, 0.0, 0.0],
        "coronal": [0.0, 1.0, 0.0],
    }}
    return list(fallback[str(view)])


def _normal_axis_label(normal):
    x, y, z = [float(v) for v in normal]
    abs_vals = [abs(x), abs(y), abs(z)]
    axis = int(max(range(3), key=lambda idx: abs_vals[idx]))
    if axis == 0:
        return "R" if x >= 0 else "L"
    if axis == 1:
        return "A" if y >= 0 else "P"
    return "S" if z >= 0 else "I"


def _largest_connected_component(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return np.zeros_like(mask, dtype=np.uint8), 0, 0
    remaining = {{(int(k), int(j), int(i)) for k, j, i in coords.tolist()}}
    neighbors = []
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dk == 0 and dj == 0 and di == 0:
                    continue
                neighbors.append((dk, dj, di))
    largest = []
    component_count = 0
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        comp = [seed]
        while queue:
            k, j, i = queue.pop()
            for dk, dj, di in neighbors:
                nb = (k + dk, j + dj, i + di)
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)
                    comp.append(nb)
        component_count += 1
        if len(comp) > len(largest):
            largest = comp
    out = np.zeros_like(mask, dtype=np.uint8)
    for k, j, i in largest:
        out[int(k), int(j), int(i)] = 1
    return out, int(component_count), int(len(largest))


def _projection_half_extent(axis_vectors, normal_vec):
    n = np.asarray(normal_vec, dtype=float)
    return 0.5 * (
        abs(float(np.dot(axis_vectors["i"], n)))
        + abs(float(np.dot(axis_vectors["j"], n)))
        + abs(float(np.dot(axis_vectors["k"], n)))
    )


def _geometry_matches(seg_node, ref_node):
    seg_arr = np.asarray(slicer.util.arrayFromVolume(seg_node))
    ref_arr = np.asarray(slicer.util.arrayFromVolume(ref_node))
    seg_arr = np.squeeze(seg_arr)
    ref_arr = np.squeeze(ref_arr)
    if seg_arr.ndim != 3 or ref_arr.ndim != 3:
        return False
    if tuple(int(v) for v in seg_arr.shape) != tuple(int(v) for v in ref_arr.shape):
        return False
    seg_mat = np.asarray(_mat_to_list(seg_node, "ijk_to_ras"), dtype=float)
    ref_mat = np.asarray(_mat_to_list(ref_node, "ijk_to_ras"), dtype=float)
    return bool(np.allclose(seg_mat, ref_mat, atol=1e-4, rtol=0.0))


def _reference_priority(node_name):
    norm = _normalize_name(node_name)
    ordered = (
        ("t1c", 0),
        ("t1 ce", 0),
        ("t1 gad", 0),
        ("post contrast", 0),
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


notes = []
volumes = _scalar_or_label_volumes()
seg_node = None
best_score = None
for node in volumes:
    score = _volume_name_score(node.GetName() if node is not None else None)
    if best_score is None or score > best_score:
        best_score = score
        seg_node = node

if seg_node is None:
    payload = {{
        "ok": True,
        "case_path": str(case_path),
        "has_segment": False,
        "provenance_policy": PROVENANCE_POLICY,
        "notes": [f"No NIfTI segmentation volume was loaded for {{segment_path.name}}."],
    }}
    gt_json_path = out_dir / "segment_gt.json"
    gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["gt_json_path"] = str(gt_json_path)
    __execResult = payload
else:
    seg_array = np.asarray(slicer.util.arrayFromVolume(seg_node))
    seg_array = np.squeeze(seg_array)
    if seg_array.ndim != 3:
        payload = {{
            "ok": True,
            "case_path": str(case_path),
            "has_segment": False,
            "provenance_policy": PROVENANCE_POLICY,
            "notes": [f"Loaded segmentation volume {{seg_node.GetName()}} is not 3D after squeeze: shape={{list(seg_array.shape)}}."],
        }}
        gt_json_path = out_dir / "segment_gt.json"
        gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["gt_json_path"] = str(gt_json_path)
        __execResult = payload
    else:
        positive_values = sorted(int(round(float(v))) for v in np.unique(seg_array[seg_array > 0]))
        if positive_values:
            notes.append(f"positive_labels={{positive_values}}")
        tumor_mask = (seg_array > 0).astype(np.uint8, copy=False)
        voxel_count_total = int(np.count_nonzero(tumor_mask))
        if voxel_count_total <= 0:
            payload = {{
                "ok": True,
                "case_path": str(case_path),
                "has_segment": False,
                "provenance_policy": PROVENANCE_POLICY,
                "notes": notes + ["Segmentation volume is empty after positive-label union."],
            }}
            gt_json_path = out_dir / "segment_gt.json"
            gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["gt_json_path"] = str(gt_json_path)
            __execResult = payload
        else:
            mask, component_count, largest_component_voxel_count = _largest_connected_component(tumor_mask)
            if largest_component_voxel_count <= 0:
                payload = {{
                    "ok": True,
                    "case_path": str(case_path),
                    "has_segment": False,
                    "provenance_policy": PROVENANCE_POLICY,
                    "notes": notes + ["Largest connected component extraction produced an empty mask."],
                }}
                gt_json_path = out_dir / "segment_gt.json"
                gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                payload["gt_json_path"] = str(gt_json_path)
                __execResult = payload
            else:
                geometry_matches = []
                for node in volumes:
                    if node is None or node.GetID() == seg_node.GetID():
                        continue
                    try:
                        if _geometry_matches(seg_node, node):
                            geometry_matches.append(node)
                    except Exception as exc:
                        notes.append(f"geometry comparison failed for {{node.GetName()}}: {{exc}}")
                geometry_matches = sorted(geometry_matches, key=lambda node: (_reference_priority(node.GetName()), _normalize_name(node.GetName())))
                ref_volume = geometry_matches[0] if geometry_matches else None
                if geometry_matches:
                    match_names = [str(node.GetName()) for node in geometry_matches]
                    if len(match_names) == 1:
                        notes.append(f"Segmentation geometry uniquely matches loaded reference volume: {{match_names[0]}}")
                    else:
                        notes.append(f"Segmentation geometry matches multiple loaded reference volumes in a shared co-registered MRI space: {{match_names}}")
                else:
                    notes.append("No geometry-matched reference MRI volume was found among the loaded candidate volumes; GT uses the segmentation affine directly.")
                if str(PROVENANCE_POLICY).lower() in {"not_applicable", "none", "off"}:
                    notes.append("Source-series provenance is not enforced for this NIfTI benchmark because the loaded MRI volumes share a co-registered space.")

                view_volume = ref_volume if ref_volume is not None else seg_node
                _activate_reference_volume(view_volume)
                ijk_to_ras = _mat_to_list(seg_node, "ijk_to_ras")
                ras_to_ijk = _mat_to_list(seg_node, "ras_to_ijk")
                ijk_to_ras_arr = np.asarray(ijk_to_ras, dtype=float)

                nz = np.argwhere(mask > 0)
                kmin, jmin, imin = [int(v) for v in nz.min(axis=0)]
                kmax, jmax, imax = [int(v) for v in nz.max(axis=0)]

                ijk_points = np.column_stack((nz[:, 2].astype(float), nz[:, 1].astype(float), nz[:, 0].astype(float), np.ones((nz.shape[0],), dtype=float)))
                ras_points = (ijk_to_ras_arr @ ijk_points.T).T[:, :3]
                rs = ras_points[:, 0]
                aas = ras_points[:, 1]
                ss = ras_points[:, 2]

                axis_vectors = {{
                    "i": ijk_to_ras_arr[:3, 0],
                    "j": ijk_to_ras_arr[:3, 1],
                    "k": ijk_to_ras_arr[:3, 2],
                }}
                slice_normals = {{view: _slice_normal(view) for view in ("axial", "sagittal", "coronal")}}
                key_slice_axis_labels = {{view: _normal_axis_label(slice_normals[view]) for view in slice_normals}}
                key_slice_ranges_mm = {{}}
                for view in ("axial", "sagittal", "coronal"):
                    normal = np.asarray(slice_normals[view], dtype=float)
                    offsets = ras_points @ normal
                    delta = _projection_half_extent(axis_vectors, normal)
                    key_slice_ranges_mm[view] = [float(offsets.min() - delta), float(offsets.max() + delta)]

                ras_axis_ranges_mm = {{}}
                for view, axis_vec in (("sagittal", np.asarray([1.0, 0.0, 0.0], dtype=float)), ("coronal", np.asarray([0.0, 1.0, 0.0], dtype=float)), ("axial", np.asarray([0.0, 0.0, 1.0], dtype=float))):
                    delta = _projection_half_extent(axis_vectors, axis_vec)
                    if view == "sagittal":
                        coords = rs
                    elif view == "coronal":
                        coords = aas
                    else:
                        coords = ss
                    ras_axis_ranges_mm[view] = [float(coords.min() - delta), float(coords.max() + delta)]

                centroid = nz.mean(axis=0)
                nearest_idx = int(np.argmin(((nz - centroid) ** 2).sum(axis=1)))
                k0, j0, i0 = [int(v) for v in nz[nearest_idx]]
                representative_point_ras = _apply_matrix(ijk_to_ras, (i0, j0, k0))
                center_ras = _apply_matrix(ijk_to_ras, (float(centroid[2]), float(centroid[1]), float(centroid[0])))

                if int(component_count) > 1:
                    notes.append(
                        f"The segmentation contains {{int(component_count)}} disconnected components; benchmark GT uses only the largest component (voxel_count={{int(largest_component_voxel_count)}})."
                    )

                mask_npz_path = out_dir / "segment_mask.npz"
                np.savez_compressed(str(mask_npz_path), mask=mask.astype(np.uint8, copy=False))

                payload = {{
                    "ok": True,
                    "case_path": str(case_path),
                    "has_segment": True,
                    "segmentation_name": str(seg_node.GetName()),
                    "segment_name": "tumor_union",
                    "source_volume_name": str(ref_volume.GetName()) if ref_volume is not None else None,
                    "source_modality": str(SOURCE_MODALITY or "") or None,
                    "provenance_policy": PROVENANCE_POLICY,
                    "voxel_count": int(largest_component_voxel_count),
                    "segment_total_voxel_count": int(voxel_count_total),
                    "component_count": int(component_count),
                    "mask_shape_kji": [int(v) for v in mask.shape],
                    "mask_npz_path": str(mask_npz_path),
                    "ras_to_ijk": ras_to_ijk,
                    "ijk_to_ras": ijk_to_ras,
                    "representative_point_ras": [float(v) for v in representative_point_ras],
                    "center_ras": [float(v) for v in center_ras],
                    "key_slice_axis_labels": key_slice_axis_labels,
                    "key_slice_ranges_mm": key_slice_ranges_mm,
                    "ras_axis_ranges_mm": ras_axis_ranges_mm,
                    "ijk_bbox": {{
                        "i": [int(imin), int(imax)],
                        "j": [int(jmin), int(jmax)],
                        "k": [int(kmin), int(kmax)],
                    }},
                    "notes": notes,
                }}
                gt_json_path = out_dir / "segment_gt.json"
                gt_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                payload["gt_json_path"] = str(gt_json_path)
                __execResult = payload
"""


def prepare_nifti_segment_ground_truth(
    ctx: Any,
    *,
    case_path: Union[str, Path],
    out_dir: Union[str, Path],
    segment_path: Optional[Union[str, Path]] = None,
    reference_volume_paths: Optional[Sequence[Union[str, Path]]] = None,
    source_modality: Optional[str] = "MR",
    provenance_policy: str = "not_applicable",
    clear_scene_after: bool = True,
) -> SegmentGroundTruth:
    case_path_p = Path(case_path).expanduser().resolve()
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    segment_path_p = Path(segment_path).expanduser().resolve() if segment_path is not None else discover_nifti_segmentation_file(case_path_p)
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

    ref_paths: List[Path] = []
    if reference_volume_paths:
        for path in reference_volume_paths:
            path_p = Path(path).expanduser().resolve()
            if path_p.exists() and path_p.is_file() and _is_nifti_path(path_p) and path_p != segment_path_p:
                ref_paths.append(path_p)
    else:
        ref_paths = [p for p in discover_reference_nifti_files(case_path_p) if p != segment_path_p]

    load_paths: List[Path] = [segment_path_p]
    seen = {str(segment_path_p)}
    for path_p in ref_paths:
        if str(path_p) in seen:
            continue
        load_paths.append(path_p)
        seen.add(str(path_p))

    active_prefer = [segment_path_p.stem, segment_path_p.name]
    ctx.load_nifti(load_paths, clear_scene_first=True, active_prefer=active_prefer)
    script = _build_slicer_nifti_segment_gt_script(
        case_path=case_path_p,
        out_dir=out_dir_p,
        segment_path=segment_path_p,
        source_modality=source_modality,
        provenance_policy=provenance_policy,
    )
    result = ctx.client.exec_python(script, timeout_s=120.0)
    if clear_scene_after:
        try:
            ctx.clear_scene()
        except Exception:
            pass
    return load_segment_ground_truth(result)


def parse_agent_evidence(text: str) -> SegmentEvidencePrediction:
    raw_text = text or ""
    key_slice_mm: Dict[str, Optional[float]] = {view: None for view in _CANONICAL_VIEWS}
    point_ras: Optional[Tuple[float, float, float]] = None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    for view in _CANONICAL_VIEWS:
        for pat in _KEY_SLICE_PATTERNS[view]:
            match = pat.search(raw_text)
            if match:
                key_slice_mm[view] = float(match.group(1))
                break

    for line in lines:
        for pat in _POINT_LINE_PATTERNS:
            match = pat.search(line)
            if not match:
                continue
            nums = _FLOAT_RE.findall(match.group(1))
            if len(nums) >= 3:
                point_ras = (float(nums[0]), float(nums[1]), float(nums[2]))
                break
        if point_ras is not None:
            break

    if point_ras is None:
        whole_text_match = re.search(r"\b(?:RSA|RAS)\b\s*[:=]\s*([\[\(]?[^\n\r]+)", raw_text, flags=re.IGNORECASE)
        if whole_text_match:
            nums = _FLOAT_RE.findall(whole_text_match.group(1))
            if len(nums) >= 3:
                point_ras = (float(nums[0]), float(nums[1]), float(nums[2]))

    return SegmentEvidencePrediction(key_slice_mm=key_slice_mm, point_ras=point_ras, raw_text=raw_text)


def _load_mask(mask_npz_path: Path) -> np.ndarray:
    payload = np.load(mask_npz_path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "mask" not in payload:
            raise KeyError(f"mask_npz file does not contain 'mask': {mask_npz_path}")
        return np.asarray(payload["mask"])
    return np.asarray(payload)


def _match_volume_name(expected: Optional[str], observed: Optional[str]) -> bool:
    exp = _normalize_volume_name(expected)
    obs = _normalize_volume_name(observed)
    if not exp or not obs:
        return False
    return exp == obs or exp in obs or obs in exp


def _format_mm_pair(pair: Optional[Sequence[float]]) -> str:
    if not pair or len(pair) != 2:
        return "<none>"
    try:
        lo, hi = sorted((float(pair[0]), float(pair[1])))
    except Exception:
        return str(pair)
    return f"[{lo:.4f}, {hi:.4f}]"


def compare_key_slice(view: str, pred_mm: Optional[float], gt: SegmentGroundTruth) -> Dict[str, Any]:
    canonical_view = str(view).lower()
    range_mm = gt.ras_axis_range_for(canonical_view)
    axis_label = _VIEW_TO_RAS_LABEL.get(canonical_view)
    scored = range_mm is not None and gt.has_segment
    lo = hi = None
    correct = False
    if range_mm is not None:
        lo, hi = sorted((float(range_mm[0]), float(range_mm[1])))
        correct = bool(pred_mm is not None and lo <= float(pred_mm) <= hi)
    debug_message = (
        f"{canonical_view}/{axis_label or '?'}: gt_range_mm={_format_mm_pair([lo, hi] if lo is not None and hi is not None else None)} "
        f"predicted={float(pred_mm):.4f} correct={correct}"
        if pred_mm is not None
        else f"{canonical_view}/{axis_label or '?'}: gt_range_mm={_format_mm_pair([lo, hi] if lo is not None and hi is not None else None)} predicted=<missing> correct={correct}"
    )
    return {
        "view": canonical_view,
        "space": "canonical_ras_axis_mm",
        "axis_label": axis_label,
        "scored": scored,
        "pred_mm": float(pred_mm) if pred_mm is not None else None,
        "range_mm": [lo, hi] if lo is not None and hi is not None else None,
        "ras_axis_range_mm": [lo, hi] if lo is not None and hi is not None else None,
        "correct": correct,
        "debug_message": debug_message,
    }


def check_ras_point(pred_ras: Optional[Sequence[float]], gt: SegmentGroundTruth) -> Dict[str, Any]:
    scored = bool(gt.has_segment and gt.mask_npz_path and gt.ras_to_ijk)
    if not scored:
        return {
            "scored": False,
            "pred_ras": list(pred_ras) if pred_ras is not None else None,
            "continuous_ijk": None,
            "index_ijk": None,
            "mask_value": None,
            "correct": False,
        }

    if pred_ras is None or len(pred_ras) != 3:
        return {
            "scored": True,
            "pred_ras": list(pred_ras) if pred_ras is not None else None,
            "continuous_ijk": None,
            "index_ijk": None,
            "mask_value": None,
            "correct": False,
        }

    mask = _load_mask(gt.mask_npz_path)  # shape [k, j, i]
    ras_to_ijk = np.asarray(gt.ras_to_ijk, dtype=float)
    ras_vec = np.asarray([float(pred_ras[0]), float(pred_ras[1]), float(pred_ras[2]), 1.0], dtype=float)
    ijk_cont = ras_to_ijk @ ras_vec
    i_cont, j_cont, k_cont = float(ijk_cont[0]), float(ijk_cont[1]), float(ijk_cont[2])
    i_idx = int(math.floor(i_cont + 0.5))
    j_idx = int(math.floor(j_cont + 0.5))
    k_idx = int(math.floor(k_cont + 0.5))

    in_bounds = (
        0 <= k_idx < int(mask.shape[0])
        and 0 <= j_idx < int(mask.shape[1])
        and 0 <= i_idx < int(mask.shape[2])
    )
    mask_value = int(mask[k_idx, j_idx, i_idx]) if in_bounds else 0
    correct = bool(in_bounds and mask_value > 0)
    return {
        "scored": True,
        "pred_ras": [float(pred_ras[0]), float(pred_ras[1]), float(pred_ras[2])],
        "continuous_ijk": [i_cont, j_cont, k_cont],
        "index_ijk": [i_idx, j_idx, k_idx],
        "mask_value": mask_value,
        "correct": correct,
    }


def _format_key_slice_debug_summary(key_slice_results: Mapping[str, Mapping[str, Any]]) -> str:
    def _range_for(view: str) -> str:
        payload = key_slice_results.get(view) or {}
        return _format_mm_pair(payload.get("range_mm"))

    def _pred_for(view: str) -> str:
        payload = key_slice_results.get(view) or {}
        pred = payload.get("pred_mm")
        if pred is None:
            return "<missing>"
        try:
            return f"{float(pred):.4f}"
        except Exception:
            return str(pred)

    def _correct_for(view: str) -> str:
        payload = key_slice_results.get(view) or {}
        return str(bool(payload.get("correct", False)))

    axial_label = _VIEW_TO_RAS_LABEL.get("axial", "S")
    sagittal_label = _VIEW_TO_RAS_LABEL.get("sagittal", "R")
    coronal_label = _VIEW_TO_RAS_LABEL.get("coronal", "A")
    return (
        f"key_slices: range ({axial_label}:{_range_for('axial')}) ({sagittal_label}:{_range_for('sagittal')}) ({coronal_label}:{_range_for('coronal')}) "
        f"pred ({axial_label}:{_pred_for('axial')}) ({sagittal_label}:{_pred_for('sagittal')}) ({coronal_label}:{_pred_for('coronal')}) "
        f"correct ({axial_label}:{_correct_for('axial')}) ({sagittal_label}:{_correct_for('sagittal')}) ({coronal_label}:{_correct_for('coronal')})"
    )


def analyze_series_provenance(
    *,
    trace_path: Optional[Union[str, Path]],
    pred_ras: Optional[Sequence[float]],
    expected_source_volume_name: Optional[str],
    expected_source_modality: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scored": True,
        "ok": False,
        "reason": None,
        "expected_source_volume_name": expected_source_volume_name,
        "expected_source_modality": expected_source_modality,
        "n_image_coords_to_ras_events": 0,
        "matched_event": None,
    }
    if trace_path is None:
        out["reason"] = "trace_path_missing"
        return out

    trace_path_p = Path(trace_path).expanduser().resolve()
    if not trace_path_p.exists():
        out["reason"] = "trace_file_missing"
        return out

    events: List[Dict[str, Any]] = []
    with trace_path_p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("kind") != "action_event":
                continue
            if str(rec.get("tool_name") or "") != "image_coords_to_ras":
                continue
            if not rec.get("ok", False):
                continue
            result = rec.get("result") or {}
            point = result.get("point_ras") or result.get("center_ras")
            point_triplet = _as_triplet(point) if isinstance(point, Sequence) else None
            state = rec.get("state") or {}
            active_name = state.get("active_volume_name")
            active_modality = infer_modality_from_name(active_name)
            event = {
                "line_no": line_no,
                "active_volume_name": active_name,
                "active_volume_id": state.get("active_volume_id"),
                "active_volume_modality": active_modality,
                "point_ras": list(point_triplet) if point_triplet is not None else None,
                "distance_to_pred_ras_mm": None,
            }
            if pred_ras is not None and point_triplet is not None and len(pred_ras) == 3:
                dist = math.dist([float(pred_ras[0]), float(pred_ras[1]), float(pred_ras[2])], list(point_triplet))
                event["distance_to_pred_ras_mm"] = float(dist)
            events.append(event)

    out["n_image_coords_to_ras_events"] = len(events)
    if not events:
        out["reason"] = "no_image_coords_to_ras_call"
        return out

    expected_name = expected_source_volume_name or None
    matching_events = [e for e in events if _match_volume_name(expected_name, e.get("active_volume_name"))] if expected_name else []
    if expected_name and matching_events:
        if pred_ras is not None:
            matching_events.sort(key=lambda e: (float(e.get("distance_to_pred_ras_mm") or float("inf")), -int(e["line_no"])))
        else:
            matching_events.sort(key=lambda e: int(e["line_no"]), reverse=True)
        chosen = matching_events[0]
        out["ok"] = True
        out["reason"] = "source_volume_name_match"
        out["matched_event"] = chosen
        return out

    if not expected_name and expected_source_modality:
        modality_matches = [e for e in events if str(e.get("active_volume_modality") or "").upper() == str(expected_source_modality).upper()]
        if modality_matches:
            if pred_ras is not None:
                modality_matches.sort(key=lambda e: (float(e.get("distance_to_pred_ras_mm") or float("inf")), -int(e["line_no"])))
            else:
                modality_matches.sort(key=lambda e: int(e["line_no"]), reverse=True)
            chosen = modality_matches[0]
            out["ok"] = True
            out["reason"] = "source_modality_match"
            out["matched_event"] = chosen
            return out

    if pred_ras is not None:
        events.sort(key=lambda e: (float(e.get("distance_to_pred_ras_mm") or float("inf")), -int(e["line_no"])))
    else:
        events.sort(key=lambda e: int(e["line_no"]), reverse=True)
    out["matched_event"] = events[0]
    out["reason"] = "source_volume_name_mismatch" if expected_name else "source_modality_mismatch"
    return out


def evaluate_segment_prediction(
    *,
    prediction: Union[str, SegmentEvidencePrediction],
    gt: SegmentGroundTruth,
    trace_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if isinstance(prediction, SegmentEvidencePrediction):
        pred = prediction
    else:
        pred = parse_agent_evidence(str(prediction or ""))

    if not gt.has_segment:
        return {
            "available": False,
            "has_segment_gt": False,
            "prediction": {
                "key_slice_mm": {view: pred.key_slice_mm.get(view) for view in _CANONICAL_VIEWS},
                "point_ras": list(pred.point_ras) if pred.point_ras is not None else None,
            },
        }

    key_slice_results = {
        view: compare_key_slice(view, pred.key_slice_mm.get(view), gt)
        for view in _CANONICAL_VIEWS
    }
    rsa_result = check_ras_point(pred.point_ras, gt)
    geometry_exact_correct = bool(all(key_slice_results[view]["correct"] for view in _CANONICAL_VIEWS) and rsa_result["correct"])

    provenance_policy = str(getattr(gt, "provenance_policy", "exact_source") or "exact_source").strip().lower()
    if provenance_policy in {"", "exact_source", "strict", "required"}:
        provenance = analyze_series_provenance(
            trace_path=trace_path,
            pred_ras=pred.point_ras,
            expected_source_volume_name=gt.source_volume_name,
            expected_source_modality=gt.source_modality,
        )
    elif provenance_policy in {"modality_only", "modality"}:
        provenance = analyze_series_provenance(
            trace_path=trace_path,
            pred_ras=pred.point_ras,
            expected_source_volume_name=None,
            expected_source_modality=gt.source_modality,
        )
    elif provenance_policy in {"not_applicable", "n/a", "none", "off", "disabled", "no_check"}:
        provenance = {
            "scored": False,
            "ok": True,
            "reason": "not_required",
            "expected_source_volume_name": gt.source_volume_name,
            "expected_source_modality": gt.source_modality,
            "n_image_coords_to_ras_events": 0,
            "matched_event": None,
        }
    else:
        provenance = {
            "scored": False,
            "ok": True,
            "reason": f"unrecognized_policy:{provenance_policy}",
            "expected_source_volume_name": gt.source_volume_name,
            "expected_source_modality": gt.source_modality,
            "n_image_coords_to_ras_events": 0,
            "matched_event": None,
        }
    evidence_exact_correct = bool(geometry_exact_correct and (provenance.get("ok", False) if provenance.get("scored", True) else True))

    debug_lines: List[str] = []
    debug_lines.append(
        f"source_volume_name={gt.source_volume_name!r} source_modality={gt.source_modality!r} segment_name={gt.segment_name!r}"
    )
    if gt.component_count is not None:
        debug_lines.append(
            f"components={int(gt.component_count)} largest_component_voxel_count={int(gt.voxel_count or 0)} segment_total_voxel_count={int(gt.segment_total_voxel_count or gt.voxel_count or 0)}"
        )
    debug_lines.append(_format_key_slice_debug_summary(key_slice_results))
    debug_lines.append(
        f"rsa: pred={list(pred.point_ras) if pred.point_ras is not None else None} index_ijk={(rsa_result.get('index_ijk'))} mask_value={rsa_result.get('mask_value')} correct={rsa_result.get('correct')}"
    )
    matched_event = provenance.get("matched_event") or {}
    debug_lines.append(
        f"provenance: ok={provenance.get('ok')} reason={provenance.get('reason')} matched_active_volume={matched_event.get('active_volume_name')} matched_modality={matched_event.get('active_volume_modality')} distance_to_pred_ras_mm={matched_event.get('distance_to_pred_ras_mm')}"
    )

    return {
        "available": True,
        "has_segment_gt": True,
        "gt": {
            "segmentation_name": gt.segmentation_name,
            "segment_name": gt.segment_name,
            "source_volume_name": gt.source_volume_name,
            "source_modality": gt.source_modality,
            "provenance_policy": gt.provenance_policy,
            "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in _CANONICAL_VIEWS},
            "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in _CANONICAL_VIEWS},
            "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in _CANONICAL_VIEWS},
            "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
            "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
            "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
            "voxel_count": gt.voxel_count,
            "segment_total_voxel_count": gt.segment_total_voxel_count,
            "component_count": gt.component_count,
            "notes": list(gt.notes),
        },
        "prediction": {
            "key_slice_mm": {view: pred.key_slice_mm.get(view) for view in _CANONICAL_VIEWS},
            "point_ras": list(pred.point_ras) if pred.point_ras is not None else None,
        },
        "key_slice": key_slice_results,
        "rsa": rsa_result,
        "geometry_exact_correct": geometry_exact_correct,
        "provenance": provenance,
        "evidence_exact_correct": evidence_exact_correct,
        "debug_lines": debug_lines,
    }


def build_default_positive_prediction(gt: SegmentGroundTruth) -> SegmentEvidencePrediction:
    if not gt.has_segment:
        raise ValueError("Ground truth does not contain a segment.")
    key_slice_mm: Dict[str, Optional[float]] = {}
    for view in _CANONICAL_VIEWS:
        rng = gt.ras_axis_range_for(view)
        if rng is None:
            key_slice_mm[view] = None
        else:
            key_slice_mm[view] = float((float(rng[0]) + float(rng[1])) / 2.0)
    return SegmentEvidencePrediction(
        key_slice_mm=key_slice_mm,
        point_ras=gt.representative_point_ras,
        raw_text="<generated positive prediction>",
    )


def build_default_negative_prediction(gt: SegmentGroundTruth, *, margin_mm: float = 50.0) -> SegmentEvidencePrediction:
    if not gt.has_segment:
        raise ValueError("Ground truth does not contain a segment.")
    key_slice_mm: Dict[str, Optional[float]] = {}
    for view in _CANONICAL_VIEWS:
        rng = gt.ras_axis_range_for(view)
        if rng is None:
            key_slice_mm[view] = None
        else:
            lo, hi = sorted((float(rng[0]), float(rng[1])))
            key_slice_mm[view] = float(hi + float(margin_mm))
    point_ras = None
    if gt.representative_point_ras is not None:
        point_ras = (
            float(gt.representative_point_ras[0] + margin_mm),
            float(gt.representative_point_ras[1] + margin_mm),
            float(gt.representative_point_ras[2] + margin_mm),
        )
    return SegmentEvidencePrediction(
        key_slice_mm=key_slice_mm,
        point_ras=point_ras,
        raw_text="<generated negative prediction>",
    )


__all__ = [
    "SegmentEvidencePrediction",
    "SegmentGroundTruth",
    "analyze_series_provenance",
    "build_default_negative_prediction",
    "build_default_positive_prediction",
    "check_ras_point",
    "compare_key_slice",
    "evaluate_segment_prediction",
    "infer_modality_from_name",
    "load_segment_ground_truth",
    "parse_agent_evidence",
    "prepare_nifti_segment_ground_truth",
    "prepare_segment_ground_truth",
]
