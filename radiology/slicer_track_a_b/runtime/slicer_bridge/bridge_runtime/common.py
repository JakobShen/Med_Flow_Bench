
"""Shared helpers for the Slicer-side bridge runtime."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import slicer

BRIDGE_VERSION = "0.5.0"



_SLICE_VIEW_MAP = {
    "axial": "Red",
    "red": "Red",
    "r": "Red",
    "sagittal": "Yellow",
    "yellow": "Yellow",
    "y": "Yellow",
    "coronal": "Green",
    "green": "Green",
    "g": "Green",
}
_PUBLIC_VIEW_MAP = {
    "axial": "axial",
    "red": "axial",
    "r": "axial",
    "sagittal": "sagittal",
    "yellow": "sagittal",
    "y": "sagittal",
    "coronal": "coronal",
    "green": "coronal",
    "g": "coronal",
}
_SLICE_PUBLIC_VIEWS = (
    ("red", "axial"),
    ("yellow", "sagittal"),
    ("green", "coronal"),
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _first_scalar_volume_node(*, exclude_id: Optional[str] = None):
    try:
        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    except Exception:
        return None
    for volume in volumes or []:
        try:
            if exclude_id and str(volume.GetID()) == str(exclude_id):
                continue
            if hasattr(volume, "GetImageData") and volume.GetImageData() is None:
                continue
            return volume
        except Exception:
            continue
    return None


def _set_active_volume(volume_node) -> None:
    if volume_node is None:
        raise RuntimeError("No volume node provided for activation")

    app_logic = slicer.app.applicationLogic()
    sel = app_logic.GetSelectionNode() if app_logic is not None else None
    if sel is not None and hasattr(sel, "SetReferenceActiveVolumeID"):
        sel.SetReferenceActiveVolumeID(volume_node.GetID())
    if app_logic is not None:
        app_logic.PropagateVolumeSelection(0)

    # Keep slice viewer layers synchronized with the selected node. This avoids WebServer
    # /slicer/slice failures after switching volumes when background layers lag behind the
    # selection node state.
    try:
        if hasattr(volume_node, "IsA") and volume_node.IsA("vtkMRMLScalarVolumeNode"):
            _volume_display_node(volume_node)
            if hasattr(volume_node, "GetImageData") and volume_node.GetImageData() is None:
                raise RuntimeError(f"Selected scalar volume has no image data: {volume_node.GetName()}")
            slicer.util.setSliceViewerLayers(background=volume_node, foreground=None, label=None, fit=False)
            if sel is not None and hasattr(sel, "SetReferenceActiveLabelVolumeID"):
                sel.SetReferenceActiveLabelVolumeID("")
        elif hasattr(volume_node, "IsA") and volume_node.IsA("vtkMRMLLabelMapVolumeNode"):
            background = _first_scalar_volume_node(exclude_id=volume_node.GetID())
            slicer.util.setSliceViewerLayers(background=background, foreground=None, label=volume_node, fit=False)
            if sel is not None and hasattr(sel, "SetReferenceActiveLabelVolumeID"):
                sel.SetReferenceActiveLabelVolumeID(volume_node.GetID())
        slicer.app.processEvents()
    except Exception:
        # Fall back to selection-node-only behavior if a specific Slicer build does not
        # support viewer layer synchronization as expected.
        slicer.app.processEvents()


def _get_node_by_id_or_name(node_id: Optional[str] = None, node_name: Optional[str] = None):
    if node_id:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node:
            return node
    if node_name:
        return slicer.util.getNode(node_name)
    raise ValueError("Must provide node_id or node_name")


def _get_active_volume_node():
    app_logic = slicer.app.applicationLogic()
    sel = app_logic.GetSelectionNode()
    vid = sel.GetActiveVolumeID()
    if vid:
        n = slicer.mrmlScene.GetNodeByID(vid)
        if n:
            return n
    vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    return vols[0] if vols else None


def _require_layout_manager():
    lm = slicer.app.layoutManager()
    if lm is None:
        raise RuntimeError(
            "Slicer layoutManager is not available. "
            "This typically means Slicer was started without a main window. "
            "Slice rendering tools require a main window (or Xvfb on headless Linux)."
        )
    return lm


def _norm_view(view: str) -> str:
    v = (view or "").strip().lower()
    if v in _SLICE_VIEW_MAP:
        return _SLICE_VIEW_MAP[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _public_view(view: str) -> str:
    v = (view or "").strip().lower()
    if v in _PUBLIC_VIEW_MAP:
        return _PUBLIC_VIEW_MAP[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _slice_widget(view: str):
    lm = _require_layout_manager()
    w = lm.sliceWidget(_norm_view(view))
    if w is None:
        raise RuntimeError(f"Slice widget not found for view={view!r}")
    return w


def _slice_node(view: str):
    return _slice_widget(view).mrmlSliceNode()


def _slice_logic(view: str):
    return _slice_widget(view).sliceLogic()


def _slice_composite_node(view: str):
    return _slice_widget(view).mrmlSliceCompositeNode()


def _node_by_id(node_id: Optional[str]):
    if not node_id:
        return None
    try:
        return slicer.mrmlScene.GetNodeByID(str(node_id))
    except Exception:
        return None


def _slice_volume_nodes(view: str):
    comp = _slice_composite_node(view)
    bg_id = str(comp.GetBackgroundVolumeID()) if comp and hasattr(comp, "GetBackgroundVolumeID") else None
    fg_id = str(comp.GetForegroundVolumeID()) if comp and hasattr(comp, "GetForegroundVolumeID") else None
    label_id = str(comp.GetLabelVolumeID()) if comp and hasattr(comp, "GetLabelVolumeID") else None
    return _node_by_id(bg_id), _node_by_id(fg_id), _node_by_id(label_id)


def _best_slice_volume_node(view: str):
    bg_node, fg_node, label_node = _slice_volume_nodes(view)
    for node in (bg_node, fg_node, label_node, _get_active_volume_node()):
        if node is not None and hasattr(node, "IsA") and (node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode")):
            return node
    return None


def _volume_ras_bounds(volume_node) -> List[float]:
    bounds = [0.0] * 6
    if volume_node is None:
        raise RuntimeError("No volume node for bounds computation")
    if hasattr(volume_node, "GetRASBounds"):
        volume_node.GetRASBounds(bounds)
        return [float(x) for x in bounds]
    if hasattr(volume_node, "GetBounds"):
        volume_node.GetBounds(bounds)
        return [float(x) for x in bounds]
    raise RuntimeError("Volume node does not expose RAS bounds")


def _slice_normal_ras(sn) -> List[float]:
    try:
        m = sn.GetSliceToRAS()
        n = [float(m.GetElement(0, 2)), float(m.GetElement(1, 2)), float(m.GetElement(2, 2))]
        mag = math.sqrt(sum(v * v for v in n))
        if mag > 0:
            return [v / mag for v in n]
    except Exception:
        pass
    orientation = str(sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else "").lower()
    if "sag" in orientation:
        return [1.0, 0.0, 0.0]
    if "cor" in orientation:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _viewer_axis_label_from_normal(normal: Sequence[float]) -> str:
    vals = [float(v) for v in normal]
    axis = int(max(range(3), key=lambda idx: abs(vals[idx])))
    comp = vals[axis]
    if axis == 0:
        return "R" if comp >= 0 else "L"
    if axis == 1:
        return "A" if comp >= 0 else "P"
    return "S" if comp >= 0 else "I"


def _canonical_axis_label_from_normal(normal: Sequence[float]) -> str:
    vals = [float(v) for v in normal]
    axis = int(max(range(3), key=lambda idx: abs(vals[idx])))
    return ("R", "A", "S")[axis]


def _sign_to_canonical_axis_from_normal(normal: Sequence[float]) -> float:
    vals = [float(v) for v in normal]
    axis = int(max(range(3), key=lambda idx: abs(vals[idx])))
    return 1.0 if float(vals[axis]) >= 0.0 else -1.0


def _canonical_slice_offset_info(sn, *, slice_offset: Optional[float] = None) -> Dict[str, Any]:
    normal = _slice_normal_ras(sn)
    viewer_axis_label = _viewer_axis_label_from_normal(normal)
    canonical_axis_label = _canonical_axis_label_from_normal(normal)
    sign_to_canonical = _sign_to_canonical_axis_from_normal(normal)
    try:
        raw_offset = float(sn.GetSliceOffset()) if slice_offset is None else float(slice_offset)
    except Exception:
        raw_offset = None
    canonical_slice_offset = float(sign_to_canonical * raw_offset) if raw_offset is not None else None
    return {
        "viewer_axis_label": viewer_axis_label,
        "canonical_axis_label": canonical_axis_label,
        "canonical_sign": float(sign_to_canonical),
        "canonical_slice_offset": canonical_slice_offset,
    }


def _canonicalize_slice_offset_range(sn, *, min_offset: Optional[float], max_offset: Optional[float]) -> Dict[str, Any]:
    info = _canonical_slice_offset_info(sn, slice_offset=None)
    out = {
        "viewer_axis_label": info.get("viewer_axis_label"),
        "canonical_axis_label": info.get("canonical_axis_label"),
        "canonical_sign": info.get("canonical_sign"),
        "canonical_min_offset": None,
        "canonical_max_offset": None,
    }
    try:
        lo = float(min_offset)
        hi = float(max_offset)
    except Exception:
        return out
    sign = float(info.get("canonical_sign") or 1.0)
    clo = sign * lo
    chi = sign * hi
    if chi < clo:
        clo, chi = chi, clo
    out["canonical_min_offset"] = float(clo)
    out["canonical_max_offset"] = float(chi)
    return out


def _project_bounds_onto_normal(bounds: Sequence[float], normal: Sequence[float]) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    nx, ny, nz = [float(v) for v in normal]
    projections: List[float] = []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            for z in (zmin, zmax):
                projections.append(x * nx + y * ny + z * nz)
    return float(min(projections)), float(max(projections))


def _fallback_slice_offset_range(*, view: str) -> Dict[str, Any]:
    sn = _slice_node(view)
    volume_node = _best_slice_volume_node(view)
    if volume_node is None:
        raise RuntimeError("No visible volume available to estimate slice offset range.")
    bounds = _volume_ras_bounds(volume_node)
    lo, hi = _project_bounds_onto_normal(bounds, _slice_normal_ras(sn))
    if hi < lo:
        lo, hi = hi, lo
    canonical_range = _canonicalize_slice_offset_range(sn, min_offset=lo, max_offset=hi)
    return {
        "view": _public_view(view),
        "min_offset": float(lo),
        "max_offset": float(hi),
        "derived_from": "volume_ras_bounds",
        "source_volume_id": volume_node.GetID(),
        "source_volume_name": volume_node.GetName(),
        **canonical_range,
    }


def _reset_interaction_state() -> Dict[str, Any]:
    details: Dict[str, Any] = {"ok": True}
    try:
        app_logic = slicer.app.applicationLogic()
        interaction_node = app_logic.GetInteractionNode() if app_logic is not None else None
        if interaction_node is not None:
            if hasattr(interaction_node, "SetCurrentInteractionMode") and hasattr(slicer.vtkMRMLInteractionNode, "ViewTransform"):
                interaction_node.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
            if hasattr(interaction_node, "SetPlaceModePersistence"):
                interaction_node.SetPlaceModePersistence(0)
            details["interaction_mode"] = int(interaction_node.GetCurrentInteractionMode()) if hasattr(interaction_node, "GetCurrentInteractionMode") else None
    except Exception as e:
        details["interaction_warning"] = str(e)
    slicer.app.processEvents()
    return details


def _capture_slice_view_png(*, view: str, output_path: str, include_controller: bool = False) -> Dict[str, Any]:
    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sw = _slice_widget(view)
    widget = sw if include_controller else sw.sliceView()
    capture_method = "qt_slice_view_grab"
    capture_warning = None

    # Give Qt a chance to repaint overlays (segmentations, crosshairs, etc.) before capture.
    try:
        if hasattr(widget, "forceRender"):
            widget.forceRender()
        if hasattr(widget, "repaint"):
            widget.repaint()
    except Exception:
        pass
    slicer.app.processEvents()

    ok = False
    pixmap = widget.grab()
    try:
        ok = bool(pixmap.save(output_path, "PNG"))
    except TypeError:
        ok = bool(pixmap.save(output_path))

    # Some Qt/Slicer builds report success from QPixmap.save but the file is not yet materialized.
    # Fall back to saving the underlying image explicitly and verify on disk before returning.
    if (not ok or not os.path.exists(output_path)) and hasattr(pixmap, "toImage"):
        try:
            image = pixmap.toImage()
            try:
                ok = bool(image.save(output_path, "PNG"))
            except TypeError:
                ok = bool(image.save(output_path))
        except Exception as exc:
            capture_warning = str(exc)

    slicer.app.processEvents()
    if not ok or not os.path.exists(output_path):
        raise RuntimeError(f"Failed to save slice view capture for {view!r} to {output_path}")
    try:
        sn = _slice_node(view)
        orientation = sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation())
        slice_offset = float(sn.GetSliceOffset())
        canonical_info = _canonical_slice_offset_info(sn, slice_offset=slice_offset)
    except Exception:
        orientation = None
        slice_offset = None
        canonical_info = {}
    return {
        "ok": True,
        "view": _public_view(view),
        "output_path": output_path,
        "include_controller": bool(include_controller),
        "capture_method": capture_method,
        "capture_warning": capture_warning,
        "orientation": orientation,
        "slice_offset": slice_offset,
        **canonical_info,
    }


def _jump_slices_to_location(x: float, y: float, z: float, centered: bool = True) -> None:
    if hasattr(slicer.util, "jumpSlicesToLocation"):
        try:
            slicer.util.jumpSlicesToLocation(float(x), float(y), float(z), bool(centered))
            slicer.app.processEvents()
            return
        except Exception:
            pass
    markups_logic_factory = getattr(getattr(slicer.modules, "markups", None), "logic", None)
    if callable(markups_logic_factory):
        logic = markups_logic_factory()
        if logic is not None and hasattr(logic, "JumpSlicesToLocation"):
            logic.JumpSlicesToLocation(float(x), float(y), float(z), bool(centered))
            slicer.app.processEvents()
            return
    raise RuntimeError("No supported JumpSlicesToLocation API is available in this Slicer build.")


def _safe_slab_type_as_string(sn) -> Optional[str]:
    """Best-effort slab reconstruction type string across Slicer/VTK wrapper variants.

    Some Slicer builds expose `GetSlabReconstructionTypeAsString(int)` (utility/static method),
    while others expose `GetSlabReconstructionTypeAsString()` (no-arg instance method).

    We avoid raising here because slab state is often informational.
    """

    if not hasattr(sn, "GetSlabReconstructionTypeAsString"):
        return None

    fn = getattr(sn, "GetSlabReconstructionTypeAsString")

    # Variant B (most common in recent wrappers): requires the type argument.
    try:
        t = int(sn.GetSlabReconstructionType()) if hasattr(sn, "GetSlabReconstructionType") else None
        if t is not None:
            return str(fn(t))
    except Exception:
        pass

    # Variant A: no-arg
    try:
        return str(fn())
    except Exception:
        return None


def _slab_type_code_for(sn, target: str) -> Optional[int]:
    """Return the integer slab reconstruction type code for a desired target.

    Slicer/VTK wrapper variants differ across versions:
      - Some expose SetSlabReconstructionTypeToMax/Min/Mean
      - Some only expose SetSlabReconstructionType(int)
      - The integer-to-mode mapping is not guaranteed stable

    We therefore discover the mapping dynamically using
    GetSlabReconstructionTypeAsString(int) when available.
    """

    if not hasattr(sn, "GetSlabReconstructionTypeAsString"):
        return None

    fn = getattr(sn, "GetSlabReconstructionTypeAsString")

    # Only works for the variant that takes the type argument.
    try:
        _ = fn(0)
        takes_arg = True
    except TypeError:
        takes_arg = False
    except Exception:
        takes_arg = True
    if not takes_arg:
        return None

    t = (target or "").strip().lower()
    if t in {"mip", "max", "maximum"}:
        keywords = {"max", "maximum", "mip"}
    elif t in {"min", "minimum"}:
        keywords = {"min", "minimum"}
    elif t in {"mean", "avg", "average"}:
        keywords = {"mean", "avg", "average"}
    else:
        return None

    # Scan a small range of likely enum values.
    for code in range(0, 8):
        try:
            s = str(fn(int(code))).lower()
        except Exception:
            continue
        if any(k in s for k in keywords):
            return int(code)
    return None


def _volume_display_node(volume_node):
    if volume_node is None:
        return None
    dn = volume_node.GetDisplayNode()
    if dn is None:
        # Create default display nodes if missing
        try:
            slicer.modules.volumes.logic().CreateDefaultVolumeDisplayNodes(volume_node)
        except Exception:
            pass
        dn = volume_node.GetDisplayNode()
    return dn


def _is_nifti_path(p: str) -> bool:
    pl = (p or "").lower()
    return pl.endswith(".nii") or pl.endswith(".nii.gz")


def _load_volume_file(path: str, *, name: Optional[str] = None):
    """Load a scalar volume from disk (NIfTI, NRRD, etc.).

    We primarily use this for NIfTI (*.nii, *.nii.gz).
    The exact ``slicer.util.loadVolume`` signature can vary across versions,
    so we implement a small compatibility shim.
    """

    props = {}
    if name:
        props["name"] = str(name)

    # Newer Slicer: loadVolume(path, properties=..., returnNode=True) -> (success, node)
    try:
        success, node = slicer.util.loadVolume(str(path), properties=props or None, returnNode=True)
        return bool(success), node
    except TypeError:
        pass
    except Exception:
        # Fall through to older call patterns
        pass

    # Older Slicer: loadVolume(path, properties=...) -> node (or raises)
    try:
        node = slicer.util.loadVolume(str(path), properties=props or None)
        return bool(node is not None), node
    except Exception:
        return False, None


def _select_active_by_preference(loaded: List[Dict[str, Any]], *, prefer: Optional[Sequence[str]] = None):
    """Pick an active volume from loaded list using filename/name heuristics."""

    if not loaded:
        return None
    if not prefer:
        return loaded[0]

    prefer_l = [str(x).lower() for x in prefer]
    # Score each loaded item by earliest preference substring match.
    best = None
    best_score = 10**9
    for item in loaded:
        key = f"{item.get('name','')} {item.get('file','')}".lower()
        score = 10**9
        for i, sub in enumerate(prefer_l):
            if sub and sub in key:
                score = i
                break
        if score < best_score:
            best_score = score
            best = item
    return best or loaded[0]

# Export internal helpers explicitly so sibling bridge modules can use star-import
# without losing underscore-prefixed utility functions during refactors.
__all__ = [name for name in globals() if not name.startswith("__")]
