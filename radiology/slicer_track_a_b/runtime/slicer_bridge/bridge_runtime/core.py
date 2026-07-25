
"""Scene, viewer, quantification, and export operations for the Slicer bridge."""
from __future__ import annotations

from .common import *  # noqa: F401,F403

# -----------------------------------------------------------------------------
# L0: Viewer actuation (deterministic)
# -----------------------------------------------------------------------------


def _get_viewer_state() -> Dict[str, Any]:
    vol = _get_active_volume_node()
    display = _volume_display_node(vol)

    state: Dict[str, Any] = {
        "active_volume_id": vol.GetID() if vol else None,
        "active_volume_name": vol.GetName() if vol else None,
        "volume_window": float(display.GetWindow()) if display else None,
        "volume_level": float(display.GetLevel()) if display else None,
        "volume_auto_window_level": bool(display.GetAutoWindowLevel()) if display else None,
        "volume_interpolate": bool(display.GetInterpolate()) if display and hasattr(display, "GetInterpolate") else None,
        "views": {},
    }

    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        try:
            sn = _slice_node(slicer_view)
            comp = _slice_composite_node(slicer_view)
            bg_id = str(comp.GetBackgroundVolumeID()) if comp else None
            fg_id = str(comp.GetForegroundVolumeID()) if comp else None
            label_id = str(comp.GetLabelVolumeID()) if comp and hasattr(comp, "GetLabelVolumeID") else None
            bg_node = slicer.mrmlScene.GetNodeByID(bg_id) if bg_id else None
            fg_node = slicer.mrmlScene.GetNodeByID(fg_id) if fg_id else None
            label_node = slicer.mrmlScene.GetNodeByID(label_id) if label_id else None
            view_state: Dict[str, Any] = {
                "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation()),
                "slice_offset": float(sn.GetSliceOffset()),
                "field_of_view": list(sn.GetFieldOfView()) if hasattr(sn, "GetFieldOfView") else None,
                "xyz_origin": list(sn.GetXYZOrigin()) if hasattr(sn, "GetXYZOrigin") else None,
                "background_volume_id": bg_id,
                "background_volume_name": bg_node.GetName() if bg_node is not None else None,
                "foreground_volume_id": fg_id,
                "foreground_volume_name": fg_node.GetName() if fg_node is not None else None,
                "foreground_opacity": float(comp.GetForegroundOpacity()) if comp and hasattr(comp, "GetForegroundOpacity") else None,
                "label_volume_id": label_id,
                "label_volume_name": label_node.GetName() if label_node is not None else None,
                "label_opacity": float(comp.GetLabelOpacity()) if comp and hasattr(comp, "GetLabelOpacity") else None,
                "linked_control": int(comp.GetLinkedControl()) if comp and hasattr(comp, "GetLinkedControl") else None,
            }
            view_state.update(_canonical_slice_offset_info(sn, slice_offset=view_state.get("slice_offset")))

            try:
                offset_range = _get_slice_offset_range(view=public_view)
                view_state["min_offset"] = offset_range.get("min_offset")
                view_state["max_offset"] = offset_range.get("max_offset")
                if offset_range.get("canonical_axis_label") is not None:
                    view_state["canonical_axis_label"] = offset_range.get("canonical_axis_label")
                if offset_range.get("canonical_min_offset") is not None:
                    view_state["canonical_min_offset"] = offset_range.get("canonical_min_offset")
                if offset_range.get("canonical_max_offset") is not None:
                    view_state["canonical_max_offset"] = offset_range.get("canonical_max_offset")
                if offset_range.get("viewer_axis_label") is not None:
                    view_state["viewer_axis_label"] = offset_range.get("viewer_axis_label")
            except Exception:
                view_state["min_offset"] = None
                view_state["max_offset"] = None
                view_state["canonical_min_offset"] = None
                view_state["canonical_max_offset"] = None

            # Thick slab (if available in this Slicer build)
            if hasattr(sn, "GetSlabReconstructionEnabled"):
                view_state["slab_enabled"] = bool(sn.GetSlabReconstructionEnabled())
            if hasattr(sn, "GetSlabReconstructionThickness"):
                view_state["slab_thickness_mm"] = float(sn.GetSlabReconstructionThickness())
            slab_type_str = _safe_slab_type_as_string(sn)
            if slab_type_str is not None:
                view_state["slab_type"] = slab_type_str
            elif hasattr(sn, "GetSlabReconstructionType"):
                view_state["slab_type"] = int(sn.GetSlabReconstructionType())

            state["views"][public_view] = view_state
        except Exception as e:
            state["views"][public_view] = {"error": str(e)}

    return state


def _set_window_level(
    *,
    window: Optional[float] = None,
    level: Optional[float] = None,
    auto: Optional[bool] = None,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    vol = None
    if volume_id or volume_name:
        vol = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    else:
        vol = _get_active_volume_node()

    if vol is None:
        raise RuntimeError("No active volume to set window/level on.")

    dn = _volume_display_node(vol)
    if dn is None:
        raise RuntimeError("Volume display node is not available for window/level setting.")

    if auto is not None:
        dn.SetAutoWindowLevel(bool(auto))

    if window is not None and level is not None:
        dn.SetAutoWindowLevel(False)
        dn.SetWindowLevel(float(window), float(level))

    slicer.app.processEvents()

    return {
        "volume_id": vol.GetID(),
        "volume_name": vol.GetName(),
        "auto_window_level": bool(dn.GetAutoWindowLevel()),
        "window": float(dn.GetWindow()),
        "level": float(dn.GetLevel()),
    }


def _set_interpolation(
    *,
    interpolate: bool,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    vol = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if vol is None:
        raise RuntimeError("No active volume to set interpolation on.")

    dn = _volume_display_node(vol)
    if dn is None or not hasattr(dn, "SetInterpolate"):
        raise RuntimeError("Volume display node does not support interpolation toggling.")

    dn.SetInterpolate(1 if interpolate else 0)
    slicer.app.processEvents()
    return {"volume_id": vol.GetID(), "volume_name": vol.GetName(), "interpolate": bool(dn.GetInterpolate())}


def _set_slice_orientation(*, view: str, orientation: str) -> Dict[str, Any]:
    sn = _slice_node(view)
    o = (orientation or "").strip().lower()

    if o in {"axial", "transverse"}:
        if hasattr(sn, "SetOrientationToAxial"):
            sn.SetOrientationToAxial()
        else:
            sn.SetOrientation("Axial")
    elif o in {"sagittal", "sag"}:
        if hasattr(sn, "SetOrientationToSagittal"):
            sn.SetOrientationToSagittal()
        else:
            sn.SetOrientation("Sagittal")
    elif o in {"coronal", "cor"}:
        if hasattr(sn, "SetOrientationToCoronal"):
            sn.SetOrientationToCoronal()
        else:
            sn.SetOrientation("Coronal")
    else:
        # Allow custom orientation strings (Slicer supports additional named orientations)
        sn.SetOrientation(str(orientation))

    slicer.app.processEvents()
    return {"view": _public_view(view), "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation())}


def _get_slice_offset_range(*, view: str) -> Dict[str, Any]:
    logic = _slice_logic(view)
    lo = None
    hi = None
    if hasattr(logic, "GetLowestVolumeSliceOffset") and hasattr(logic, "GetHighestVolumeSliceOffset"):
        try:
            lo = float(logic.GetLowestVolumeSliceOffset())
            hi = float(logic.GetHighestVolumeSliceOffset())
        except Exception:
            lo = None
            hi = None
    elif hasattr(logic, "GetSliceOffsetRange"):
        try:
            r = logic.GetSliceOffsetRange()
            lo = float(r[0])
            hi = float(r[1])
        except Exception:
            lo = None
            hi = None

    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return _fallback_slice_offset_range(view=view)

    if hi < lo:
        lo, hi = hi, lo
    sn = _slice_node(view)
    canonical_range = _canonicalize_slice_offset_range(sn, min_offset=lo, max_offset=hi)
    return {
        "view": _public_view(view),
        "min_offset": float(lo),
        "max_offset": float(hi),
        "derived_from": "slice_logic",
        **canonical_range,
    }


def _set_slice_offset(*, view: str, offset: float) -> Dict[str, Any]:
    sn = _slice_node(view)
    sn.SetSliceOffset(float(offset))
    slicer.app.processEvents()
    result = {"view": _public_view(view), "slice_offset": float(sn.GetSliceOffset())}
    result.update(_canonical_slice_offset_info(sn, slice_offset=result.get("slice_offset")))
    return result


def _set_slice_scroll_to(*, view: str, scroll_to: float) -> Dict[str, Any]:
    r = _get_slice_offset_range(view=view)
    lo = float(r["min_offset"])
    hi = float(r["max_offset"])
    t = float(scroll_to)
    t = max(0.0, min(1.0, t))
    off = lo + t * (hi - lo)
    return _set_slice_offset(view=view, offset=off) | {"scroll_to": t, "min_offset": lo, "max_offset": hi}


def _fit_slice(*, view: str) -> Dict[str, Any]:
    logic = _slice_logic(view)
    logic.FitSliceToAll()
    slicer.app.processEvents()
    sn = _slice_node(view)
    result = {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView()), "slice_offset": float(sn.GetSliceOffset())}
    result.update(_canonical_slice_offset_info(sn, slice_offset=result.get("slice_offset")))
    return result


def _zoom_slice_relative(*, view: str, factor: float) -> Dict[str, Any]:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    sn = _slice_node(view)
    if not hasattr(sn, "GetFieldOfView") or not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support field-of-view control in this Slicer build.")
    fov = sn.GetFieldOfView()
    sn.SetFieldOfView(float(fov[0]) / float(factor), float(fov[1]) / float(factor), float(fov[2]))
    slicer.app.processEvents()
    return {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView()), "factor": float(factor)}




def _set_field_of_view(*, view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support SetFieldOfView in this Slicer build.")
    if len(field_of_view) != 3:
        raise ValueError("field_of_view must be length-3 [x,y,z]")
    fx, fy, fz = [float(v) for v in field_of_view]
    sn.SetFieldOfView(fx, fy, fz)
    slicer.app.processEvents()
    return {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView())}


def _set_xyz_origin(*, view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetXYZOrigin"):
        raise RuntimeError("SliceNode does not support SetXYZOrigin in this Slicer build.")
    if len(xyz_origin) != 3:
        raise ValueError("xyz_origin must be length-3 [x,y,z]")
    x, y, z = [float(v) for v in xyz_origin]
    sn.SetXYZOrigin(x, y, z)
    slicer.app.processEvents()
    return {"view": _public_view(view), "xyz_origin": list(sn.GetXYZOrigin())}
def _jump_to_ras(*, ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
    if len(ras) != 3:
        raise ValueError("ras must be length-3 [x,y,z] in RAS (mm)")
    x, y, z = [float(v) for v in ras]
    _jump_slices_to_location(x, y, z, centered)
    slicer.app.processEvents()
    return {"ras": [x, y, z], "centered": bool(centered)}


def _volume_center_ras(volume_node) -> List[float]:
    bounds = _volume_ras_bounds(volume_node)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return [float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0), float((zmin + zmax) / 2.0)]


def _recover_standard_views(*, volume_id: Optional[str] = None, volume_name: Optional[str] = None, centered: bool = True, fit: bool = True) -> Dict[str, Any]:
    volume_node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if volume_node is None:
        raise RuntimeError("No volume available to recover standard views.")
    if hasattr(volume_node, "IsA") and (volume_node.IsA("vtkMRMLScalarVolumeNode") or volume_node.IsA("vtkMRMLLabelMapVolumeNode")):
        _set_active_volume(volume_node)

    orientation_results: Dict[str, Any] = {}
    for view_name, orientation in (("axial", "axial"), ("sagittal", "sagittal"), ("coronal", "coronal")):
        orientation_results[view_name] = _set_slice_orientation(view=view_name, orientation=orientation)

    center_ras = _volume_center_ras(volume_node)
    _jump_slices_to_location(center_ras[0], center_ras[1], center_ras[2], centered)

    fit_results: Dict[str, Any] = {}
    if fit:
        for view_name in ("axial", "sagittal", "coronal"):
            fit_results[view_name] = _fit_slice(view=view_name)

    return {
        "volume_id": volume_node.GetID(),
        "volume_name": volume_node.GetName(),
        "center_ras": center_ras,
        "centered": bool(centered),
        "fit": bool(fit),
        "orientations": orientation_results,
        "fit_results": fit_results,
        "viewer_state": _get_viewer_state(),
    }


def _set_linked_slices(*, enabled: bool) -> Dict[str, Any]:
    results: Dict[str, Any] = {"enabled": bool(enabled), "views": {}}
    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        try:
            comp = _slice_composite_node(slicer_view)
            if comp is None:
                continue
            if hasattr(comp, "SetLinkedControl"):
                comp.SetLinkedControl(1 if enabled else 0)
            if hasattr(comp, "SetHotLinkedControl"):
                comp.SetHotLinkedControl(1 if enabled else 0)
            results["views"][public_view] = {
                "linked_control": int(comp.GetLinkedControl()) if hasattr(comp, "GetLinkedControl") else None
            }
        except Exception as e:
            results["views"][public_view] = {"error": str(e)}
    slicer.app.processEvents()
    return results


def _set_layout(*, layout: str) -> Dict[str, Any]:
    """Set a standard Slicer layout.

    Note: not all Slicer builds expose the same layout constants. We resolve
    constants via ``getattr`` with safe fallbacks so that missing optional
    constants do not break the tool.
    """

    _require_layout_manager()
    ln = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

    def _layout_const(primary: str, fallback: str) -> int:
        # vtkMRMLLayoutNode layout constants are integers.
        if hasattr(slicer.vtkMRMLLayoutNode, primary):
            return int(getattr(slicer.vtkMRMLLayoutNode, primary))
        if hasattr(slicer.vtkMRMLLayoutNode, fallback):
            return int(getattr(slicer.vtkMRMLLayoutNode, fallback))
        raise AttributeError(
            f"Neither layout constant {primary!r} nor fallback {fallback!r} exists on vtkMRMLLayoutNode"
        )

    name = (layout or "").strip().lower()
    # Small, deterministic subset of layouts.
    layout_map = {
        "four_up": _layout_const("SlicerLayoutFourUpView", "SlicerLayoutConventionalView"),
        "fourup": _layout_const("SlicerLayoutFourUpView", "SlicerLayoutConventionalView"),
        "conventional": _layout_const("SlicerLayoutConventionalView", "SlicerLayoutFourUpView"),
        # Some builds no longer expose a dedicated 3-slice-only layout.
        # Fall back to FourUp (3 slices + 3D) which is always safe.
        "three_up": _layout_const("SlicerLayoutThreeUpView", "SlicerLayoutFourUpView"),
        "one_up_axial": _layout_const("SlicerLayoutOneUpRedSliceView", "SlicerLayoutFourUpView"),
        "one_up_sagittal": _layout_const("SlicerLayoutOneUpYellowSliceView", "SlicerLayoutFourUpView"),
        "one_up_coronal": _layout_const("SlicerLayoutOneUpGreenSliceView", "SlicerLayoutFourUpView"),
        "one_up_red": _layout_const("SlicerLayoutOneUpRedSliceView", "SlicerLayoutFourUpView"),
        "one_up_yellow": _layout_const("SlicerLayoutOneUpYellowSliceView", "SlicerLayoutFourUpView"),
        "one_up_green": _layout_const("SlicerLayoutOneUpGreenSliceView", "SlicerLayoutFourUpView"),
    }
    if name not in layout_map:
        raise ValueError(f"Unsupported layout: {layout!r}. Supported: {sorted(layout_map.keys())}")
    ln.SetViewArrangement(int(layout_map[name]))
    slicer.app.processEvents()
    return {"layout": name, "layout_id": int(layout_map[name])}


# -----------------------------------------------------------------------------
# L1: Clinical deterministic quantification
# -----------------------------------------------------------------------------


def _set_thick_slab(*, view: str, enabled: bool, thickness_mm: float = 0.0, mode: str = "mip") -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetSlabReconstructionEnabled"):
        raise RuntimeError("This Slicer build does not support thick slab reconstruction on SliceNode.")

    sn.SetSlabReconstructionEnabled(1 if enabled else 0)

    if hasattr(sn, "SetSlabReconstructionThickness"):
        sn.SetSlabReconstructionThickness(float(thickness_mm))

    # Best-effort mapping across Slicer builds.
    mode_l = (mode or "").strip().lower()

    if mode_l in {"mip", "max", "maximum"}:
        target = "max"
        keywords = {"max", "maximum", "mip"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMax",
            "SetSlabReconstructionTypeToMaximum",
            "SetSlabReconstructionTypeToMIP",
        )
    elif mode_l in {"min", "minimum"}:
        target = "min"
        keywords = {"min", "minimum"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMin",
            "SetSlabReconstructionTypeToMinimum",
        )
    elif mode_l in {"mean", "avg", "average"}:
        target = "mean"
        keywords = {"mean", "avg", "average"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMean",
            "SetSlabReconstructionTypeToAverage",
        )
    else:
        raise ValueError("mode must be one of: mip/max, min, mean")

    # 1) Try dedicated convenience setters if available.
    applied = False
    for fn in preferred_setters:
        if hasattr(sn, fn):
            try:
                getattr(sn, fn)()
                applied = True
                break
            except Exception:
                pass

    # 2) Resolve enum mapping dynamically via GetSlabReconstructionTypeAsString(int)
    #    (mapping differs across VTK wrapper builds).
    if not applied and hasattr(sn, "SetSlabReconstructionType"):
        code = _slab_type_code_for(sn, target)
        if code is not None:
            try:
                sn.SetSlabReconstructionType(int(code))
                applied = True
            except Exception:
                applied = False

    # 3) Last resort: brute-force a small enum range and keep the first match.
    if not applied and hasattr(sn, "SetSlabReconstructionType"):
        for code in range(0, 8):
            try:
                sn.SetSlabReconstructionType(int(code))
                s = (_safe_slab_type_as_string(sn) or "").lower()
                if any(k in s for k in keywords):
                    applied = True
                    break
            except Exception:
                continue

    if not applied:
        raise RuntimeError(f"Unable to set slab reconstruction type for mode={mode_l!r} on this Slicer build")

    slicer.app.processEvents()

    out: Dict[str, Any] = {
        "view": _public_view(view),
        "enabled": bool(sn.GetSlabReconstructionEnabled()) if hasattr(sn, "GetSlabReconstructionEnabled") else bool(enabled),
        "thickness_mm": float(sn.GetSlabReconstructionThickness()) if hasattr(sn, "GetSlabReconstructionThickness") else float(thickness_mm),
    }
    slab_mode_str = _safe_slab_type_as_string(sn)
    if slab_mode_str is not None:
        out["mode"] = slab_mode_str
    elif hasattr(sn, "GetSlabReconstructionType"):
        out["mode"] = int(sn.GetSlabReconstructionType())
    else:
        out["mode"] = mode_l
    return out


def _set_fusion(*, background_volume_id: str, foreground_volume_id: str, opacity: float = 0.5) -> Dict[str, Any]:
    bg = slicer.mrmlScene.GetNodeByID(background_volume_id)
    fg = slicer.mrmlScene.GetNodeByID(foreground_volume_id)
    if bg is None:
        raise ValueError(f"Background volume not found: {background_volume_id}")
    if fg is None:
        raise ValueError(f"Foreground volume not found: {foreground_volume_id}")
    slicer.util.setSliceViewerLayers(background=bg, foreground=fg, foregroundOpacity=float(opacity))
    slicer.app.processEvents()
    return {"background_volume_id": bg.GetID(), "foreground_volume_id": fg.GetID(), "opacity": float(opacity)}


def _clear_fusion() -> Dict[str, Any]:
    results: Dict[str, Any] = {"views": {}}
    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        comp = _slice_composite_node(slicer_view)
        if comp is None:
            results["views"][public_view] = {"error": "slice composite node missing"}
            continue
        try:
            if hasattr(comp, "SetForegroundVolumeID"):
                try:
                    comp.SetForegroundVolumeID(None)
                except Exception:
                    comp.SetForegroundVolumeID("")
            if hasattr(comp, "SetForegroundOpacity"):
                comp.SetForegroundOpacity(0.0)
            results["views"][public_view] = {
                "background_volume_id": str(comp.GetBackgroundVolumeID()) if hasattr(comp, "GetBackgroundVolumeID") else None,
                "foreground_volume_id": str(comp.GetForegroundVolumeID()) if hasattr(comp, "GetForegroundVolumeID") else None,
                "foreground_opacity": float(comp.GetForegroundOpacity()) if hasattr(comp, "GetForegroundOpacity") else None,
            }
        except Exception as e:
            results["views"][public_view] = {"error": str(e)}
    slicer.app.processEvents()
    return results


def _compute_subtraction(*, volume_a_id: str, volume_b_id: str, output_name: str = "Subtraction") -> Dict[str, Any]:
    va = slicer.mrmlScene.GetNodeByID(volume_a_id)
    vb = slicer.mrmlScene.GetNodeByID(volume_b_id)
    if va is None or vb is None:
        raise ValueError("volume_a_id and volume_b_id must exist in the scene.")
    if not va.IsA("vtkMRMLScalarVolumeNode") or not vb.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("Subtraction currently supports scalar volumes only.")

    import numpy as np

    a = slicer.util.arrayFromVolume(va).astype(np.float32, copy=False)
    b = slicer.util.arrayFromVolume(vb).astype(np.float32, copy=False)
    if a.shape != b.shape:
        raise ValueError(f"Volume shapes differ: {a.shape} vs {b.shape} (must be resampled first).")

    diff = a - b

    # Clone geometry from A for determinism.
    out = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, va, str(output_name))
    slicer.util.updateVolumeFromArray(out, diff)

    dn = _volume_display_node(out)
    if dn is not None:
        dn.SetAutoWindowLevel(True)

    _set_active_volume(out)
    slicer.app.processEvents()
    return {"output_volume_id": out.GetID(), "output_volume_name": out.GetName()}


def _resample_volume_isotropic(**kwargs) -> Dict[str, Any]:
    """Lazy proxy to avoid importing bridge_runtime.segmentation at module import time.

    The segmentation bridge imports helpers from this module, so importing its resample helper
    directly at the top level would create a circular import. Resolve the implementation only
    when the tool is actually executed.
    """

    from .segmentation import _resample_volume_isotropic as _impl

    return _impl(**kwargs)


def _ras_to_ijk_float(volume_node, ras: Sequence[float]) -> Tuple[float, float, float]:
    if len(ras) != 3:
        raise ValueError("ras must be length-3 [x,y,z]")
    import vtk

    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    x, y, z = [float(v) for v in ras]
    out = [0.0, 0.0, 0.0, 0.0]
    m.MultiplyPoint([x, y, z, 1.0], out)
    return float(out[0]), float(out[1]), float(out[2])


def _ras_to_volume_ijk(
    *,
    ras: Sequence[float],
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume is available for RAS to IJK conversion.")
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a volume: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
    ijk_float = _ras_to_ijk_float(node, ras)
    return {
        "volume_id": node.GetID(),
        "volume_name": node.GetName(),
        "ras": [float(v) for v in ras],
        "ijk_float": [float(v) for v in ijk_float],
        "ijk": [int(round(float(v))) for v in ijk_float],
    }


def _roi_stats_ijk(
    volume_id: Optional[str],
    volume_name: Optional[str],
    ijk_min: List[int],
    ijk_max: List[int],
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a scalar volume: {node.GetClassName()}")

    if len(ijk_min) != 3 or len(ijk_max) != 3:
        raise ValueError("ijk_min and ijk_max must be length-3 arrays [i,j,k]")

    i0, j0, k0 = [int(x) for x in ijk_min]
    i1, j1, k1 = [int(x) for x in ijk_max]

    dims = node.GetImageData().GetDimensions()  # (i, j, k)
    i0 = max(0, min(i0, dims[0]))
    i1 = max(0, min(i1, dims[0]))
    j0 = max(0, min(j0, dims[1]))
    j1 = max(0, min(j1, dims[1]))
    k0 = max(0, min(k0, dims[2]))
    k1 = max(0, min(k1, dims[2]))

    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        raise ValueError(f"Empty ROI after clamping. ijk_min={ijk_min} ijk_max={ijk_max} dims={dims}")

    import numpy as np

    arr = slicer.util.arrayFromVolume(node)  # shape: (k, j, i)
    roi = arr[k0:k1, j0:j1, i0:i1]
    roi_f = roi.astype(np.float64, copy=False)

    return {
        "volume_id": node.GetID(),
        "volume_name": node.GetName(),
        "dims_ijk": list(dims),
        "ijk_min": [i0, j0, k0],
        "ijk_max": [i1, j1, k1],
        "n_voxels": int(roi_f.size),
        "min": float(np.min(roi_f)),
        "max": float(np.max(roi_f)),
        "mean": float(np.mean(roi_f)),
        "std": float(np.std(roi_f)),
    }


def _roi_stats_ras_box(
    *,
    volume_id: Optional[str],
    volume_name: Optional[str],
    ras_min: Sequence[float],
    ras_max: Sequence[float],
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume for ROI stats.")
    if not node.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("roi_stats_ras_box currently supports scalar volumes only.")

    if len(ras_min) != 3 or len(ras_max) != 3:
        raise ValueError("ras_min and ras_max must be length-3 [x,y,z]")

    xmin, ymin, zmin = [float(v) for v in ras_min]
    xmax, ymax, zmax = [float(v) for v in ras_max]
    # 8 corners in RAS
    corners = [
        (xmin, ymin, zmin),
        (xmin, ymin, zmax),
        (xmin, ymax, zmin),
        (xmin, ymax, zmax),
        (xmax, ymin, zmin),
        (xmax, ymin, zmax),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
    ]
    ijk = [_ras_to_ijk_float(node, c) for c in corners]
    is_ = [p[0] for p in ijk]
    js_ = [p[1] for p in ijk]
    ks_ = [p[2] for p in ijk]

    i0 = int(math.floor(min(is_)))
    j0 = int(math.floor(min(js_)))
    k0 = int(math.floor(min(ks_)))
    i1 = int(math.floor(max(is_))) + 1
    j1 = int(math.floor(max(js_))) + 1
    k1 = int(math.floor(max(ks_))) + 1

    return _roi_stats_ijk(volume_id=node.GetID(), volume_name=None, ijk_min=[i0, j0, k0], ijk_max=[i1, j1, k1]) | {
        "ras_min": [xmin, ymin, zmin],
        "ras_max": [xmax, ymax, zmax],
        "note": "ROI is computed as the axis-aligned IJK bounding box of the provided RAS box corners.",
    }


def _sample_intensity_ras(
    *,
    volume_id: Optional[str],
    volume_name: Optional[str],
    ras: Sequence[float],
    method: str = "nearest",
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume for sampling.")
    if not node.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("sample_intensity_ras currently supports scalar volumes only.")

    import numpy as np

    i_f, j_f, k_f = _ras_to_ijk_float(node, ras)
    arr = slicer.util.arrayFromVolume(node)  # (k,j,i)
    kz, jy, ix = arr.shape  # k, j, i

    m = (method or "").strip().lower()

    def _in_bounds(i: int, j: int, k: int) -> bool:
        return (0 <= i < ix) and (0 <= j < jy) and (0 <= k < kz)

    if m == "nearest":
        i = int(round(i_f))
        j = int(round(j_f))
        k = int(round(k_f))
        if not _in_bounds(i, j, k):
            return {"in_bounds": False, "value": None, "ijk": [i, j, k], "ras": [float(x) for x in ras]}
        return {"in_bounds": True, "value": float(arr[k, j, i]), "ijk": [i, j, k], "ras": [float(x) for x in ras]}

    if m in {"trilinear", "linear"}:
        i0 = int(math.floor(i_f))
        j0 = int(math.floor(j_f))
        k0 = int(math.floor(k_f))
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        if not (_in_bounds(i0, j0, k0) and _in_bounds(i1, j1, k1)):
            return {"in_bounds": False, "value": None, "ijk_float": [i_f, j_f, k_f], "ras": [float(x) for x in ras]}

        di = i_f - i0
        dj = j_f - j0
        dk = k_f - k0

        # 8-corner trilinear interpolation
        v000 = float(arr[k0, j0, i0])
        v100 = float(arr[k0, j0, i1])
        v010 = float(arr[k0, j1, i0])
        v110 = float(arr[k0, j1, i1])
        v001 = float(arr[k1, j0, i0])
        v101 = float(arr[k1, j0, i1])
        v011 = float(arr[k1, j1, i0])
        v111 = float(arr[k1, j1, i1])

        v00 = v000 * (1 - di) + v100 * di
        v01 = v001 * (1 - di) + v101 * di
        v10 = v010 * (1 - di) + v110 * di
        v11 = v011 * (1 - di) + v111 * di

        v0 = v00 * (1 - dj) + v10 * dj
        v1 = v01 * (1 - dj) + v11 * dj

        v = v0 * (1 - dk) + v1 * dk

        return {"in_bounds": True, "value": float(v), "ijk_float": [i_f, j_f, k_f], "ras": [float(x) for x in ras]}

    raise ValueError("method must be one of: nearest, trilinear")


def _measure_distance_ras(*, p1: Sequence[float], p2: Sequence[float]) -> Dict[str, Any]:
    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("p1 and p2 must be length-3 [x,y,z] in RAS (mm)")
    x1, y1, z1 = [float(v) for v in p1]
    x2, y2, z2 = [float(v) for v in p2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return {"p1": [x1, y1, z1], "p2": [x2, y2, z2], "distance_mm": float(d)}


def _measure_angle_ras(*, p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> Dict[str, Any]:
    if len(p1) != 3 or len(p2) != 3 or len(p3) != 3:
        raise ValueError("p1,p2,p3 must be length-3 [x,y,z] in RAS (mm)")
    import numpy as np

    a = np.array([float(v) for v in p1], dtype=np.float64)
    b = np.array([float(v) for v in p2], dtype=np.float64)
    c = np.array([float(v) for v in p3], dtype=np.float64)

    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("Degenerate angle: two points coincide with vertex.")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    return {"p1": a.tolist(), "p2": b.tolist(), "p3": c.tolist(), "angle_deg": float(ang)}


def _measure_area_polygon_ras(*, points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for polygon area.")
    import numpy as np

    pts = np.array([[float(v) for v in p] for p in points], dtype=np.float64)
    p0 = pts[0]
    # Define plane from first 3 points
    v1 = pts[1] - p0
    v2 = pts[2] - p0
    n = np.cross(v1, v2)
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        raise ValueError("Degenerate polygon: first three points are collinear.")
    n = n / nn
    xaxis = v1 / float(np.linalg.norm(v1))
    yaxis = np.cross(n, xaxis)

    # Project to 2D
    xy = np.stack([np.dot(pts - p0, xaxis), np.dot(pts - p0, yaxis)], axis=1)
    x = xy[:, 0]
    y = xy[:, 1]
    # Shoelace formula
    area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return {"n_points": int(len(points)), "area_mm2": float(area)}


# -----------------------------------------------------------------------------
# Existing tools (scene/DICOM/export)
# -----------------------------------------------------------------------------


def _clear_scene() -> Dict[str, Any]:
    reset = _reset_interaction_state()
    slicer.mrmlScene.Clear()
    slicer.app.processEvents()
    return {"cleared": True, "reset_interaction": reset}


def _load_dicom(
    dicom_dir: str,
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    dicom_dir = os.path.expanduser(dicom_dir)
    dicom_dir = os.path.abspath(dicom_dir)
    if not os.path.isdir(dicom_dir):
        raise ValueError(f"dicom_dir is not a directory: {dicom_dir}")

    return _load_dicom_series(
        dicom_dirs=[dicom_dir],
        clear_scene_first=clear_scene_first,
        active_prefer=active_prefer,
    )


def _load_dicom_series(
    *,
    dicom_dirs: Sequence[str],
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    series_dirs: List[str] = []
    for d in dicom_dirs or []:
        dp = os.path.abspath(os.path.expanduser(str(d)))
        if os.path.isdir(dp):
            series_dirs.append(dp)

    if not series_dirs:
        raise ValueError("No valid dicom_dirs provided")

    if clear_scene_first:
        _reset_interaction_state()
        slicer.mrmlScene.Clear()

    from DICOMLib import DICOMUtils

    loaded_node_ids: List[str] = []
    patient_uids: List[str] = []
    imported_dirs: List[str] = []
    skipped_dirs: List[Dict[str, Any]] = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        for d in series_dirs:
            try:
                DICOMUtils.importDicom(d, db)
                imported_dirs.append(d)
            except Exception as e:
                skipped_dirs.append({"dir": d, "reason": f"import_failed: {e}"})

        patient_uids = list(db.patients())
        for patient_uid in patient_uids:
            try:
                loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))
            except Exception as e:
                skipped_dirs.append({"dir": str(patient_uid), "reason": f"load_patient_failed: {e}"})

    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    active_volume_id = None
    active_volume_name = None
    if volumes:
        loaded_items: List[Dict[str, Any]] = []
        loaded_set = set(loaded_node_ids)
        for v in volumes:
            if v.GetID() in loaded_set:
                loaded_items.append({"id": v.GetID(), "name": v.GetName(), "file": v.GetName()})

        pick = _select_active_by_preference(loaded_items if loaded_items else [{"id": v.GetID(), "name": v.GetName(), "file": v.GetName()} for v in volumes], prefer=active_prefer)
        if pick is not None:
            volume_node = slicer.mrmlScene.GetNodeByID(str(pick.get("id")))
            if volume_node is not None:
                _set_active_volume(volume_node)
                active_volume_id = volume_node.GetID()
                active_volume_name = volume_node.GetName()

    slicer.app.processEvents()

    return {
        "loaded_node_ids": loaded_node_ids,
        "patient_uids": patient_uids,
        "active_volume_id": active_volume_id,
        "active_volume_name": active_volume_name,
        "n_scalar_volumes": len(volumes),
        "imported_dirs": imported_dirs,
        "skipped_dirs": skipped_dirs,
    }


def _load_nifti(
    *,
    nifti_dir: Optional[str] = None,
    nifti_files: Optional[Sequence[str]] = None,
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load NIfTI volumes from a directory or an explicit file list.

    This tool is intentionally conservative:
      - Only loads paths ending with .nii or .nii.gz
      - Ignores missing paths (records them as skipped)

    Selection of the active volume can be guided by `active_prefer`, a list of
    substrings (earlier = higher priority) matched against filename/name.
    """

    files: List[str] = []
    if nifti_files:
        files = [str(x) for x in nifti_files]
    elif nifti_dir:
        d = os.path.abspath(os.path.expanduser(str(nifti_dir)))
        if not os.path.isdir(d):
            raise ValueError(f"nifti_dir is not a directory: {d}")
        # Deterministic ordering.
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and _is_nifti_path(p):
                files.append(p)
    else:
        raise ValueError("Must provide nifti_dir or nifti_files")

    if clear_scene_first:
        _reset_interaction_state()
        slicer.mrmlScene.Clear()

    loaded: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for f in files:
        fp = os.path.abspath(os.path.expanduser(str(f)))
        if not os.path.isfile(fp):
            skipped.append({"file": fp, "reason": "not_found"})
            continue
        if not _is_nifti_path(fp):
            skipped.append({"file": fp, "reason": "not_nifti"})
            continue

        name = os.path.splitext(os.path.basename(fp))[0]
        if name.endswith(".nii"):
            name = name[:-4]

        success, node = _load_volume_file(fp, name=name)
        if not success or node is None:
            skipped.append({"file": fp, "reason": "load_failed"})
            continue

        loaded.append({"file": fp, "id": node.GetID(), "name": node.GetName()})

    active_volume_id = None
    active_volume_name = None
    pick = _select_active_by_preference(loaded, prefer=active_prefer)
    if pick is not None:
        node = slicer.mrmlScene.GetNodeByID(str(pick.get("id")))
        if node is not None:
            _set_active_volume(node)
            active_volume_id = node.GetID()
            active_volume_name = node.GetName()

    slicer.app.processEvents()

    return {
        "loaded": loaded,
        "skipped": skipped,
        "active_volume_id": active_volume_id,
        "active_volume_name": active_volume_name,
        "n_loaded": len(loaded),
    }


def _select_volume(volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a volume: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
    _set_active_volume(node)
    slicer.app.processEvents()
    return {"active_volume_id": node.GetID(), "active_volume_name": node.GetName()}


def _export_volume_nifti(
    *,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
    output_path: str,
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume to export.")
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a volume: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")

    out_path = os.path.abspath(os.path.expanduser(str(output_path)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = bool(slicer.util.saveNode(node, out_path))
    if not ok or not os.path.exists(out_path):
        raise RuntimeError(f"Failed to export volume to NIfTI: {out_path}")
    return {
        "path": out_path,
        "volume_id": node.GetID(),
        "volume_name": node.GetName(),
    }


def _load_labelmap_as_segmentation(
    *,
    labelmap_path: str,
    reference_volume_id: Optional[str] = None,
    reference_volume_name: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    remove_labelmap: bool = True,
    show_labelmap_in_viewer: bool = False,
    visible_label_values: Optional[Sequence[int]] = None,
    label_opacity: float = 0.3,
) -> Dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(str(labelmap_path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labelmap path does not exist: {path}")

    reference = (
        _get_node_by_id_or_name(node_id=reference_volume_id, node_name=reference_volume_name)
        if (reference_volume_id or reference_volume_name)
        else _get_active_volume_node()
    )
    if reference is None:
        raise RuntimeError("No reference volume available for labelmap import.")

    loaded = slicer.util.loadLabelVolume(path, returnNode=True)
    if isinstance(loaded, tuple):
        success, label_node = loaded
    else:
        success, label_node = True, loaded
    if not success or label_node is None:
        raise RuntimeError(f"Failed to load labelmap: {path}")

    visible_values = [int(v) for v in (visible_label_values or [])]
    if visible_values:
        try:
            import numpy as np

            arr = slicer.util.arrayFromVolume(label_node)
            arr[~np.isin(arr, visible_values)] = 0
            slicer.util.arrayFromVolumeModified(label_node)
        except Exception:
            pass

    name = segmentation_name or f"{os.path.splitext(os.path.basename(path))[0]} segmentation"
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference)
    logic = slicer.modules.segmentations.logic()
    logic.ImportLabelmapToSegmentationNode(label_node, segmentation_node)
    segmentation = segmentation_node.GetSegmentation()
    segment_ids: List[str] = []
    segment_names: List[str] = []
    for i in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(i)
        segment = segmentation.GetSegment(segment_id)
        segment_ids.append(str(segment_id))
        segment_names.append(str(segment.GetName()) if segment is not None else str(segment_id))

    try:
        segmentation_node.GetDisplayNode().SetVisibility2DFill(True)
        segmentation_node.GetDisplayNode().SetVisibility2DOutline(True)
    except Exception:
        pass
    _set_active_volume(reference)

    labelmap_overlay_id = None
    labelmap_overlay_name = None
    if show_labelmap_in_viewer:
        _set_labelmap_overlay(
            labelmap_volume_id=label_node.GetID(),
            reference_volume_id=reference.GetID(),
            visible=True,
            opacity=float(label_opacity),
            visible_label_values=visible_values or None,
        )
        labelmap_overlay_id = label_node.GetID()
        labelmap_overlay_name = label_node.GetName()

    if remove_labelmap and not show_labelmap_in_viewer:
        slicer.mrmlScene.RemoveNode(label_node)
    slicer.app.processEvents()

    return {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segment_ids": segment_ids,
        "segment_names": segment_names,
        "segment_id": segment_ids[0] if segment_ids else None,
        "segment_name": segment_names[0] if segment_names else None,
        "source_volume_id": reference.GetID(),
        "source_volume_name": reference.GetName(),
        "labelmap_path": path,
        "labelmap_volume_id": labelmap_overlay_id,
        "labelmap_volume_name": labelmap_overlay_name,
        "labelmap_overlay_visible": bool(show_labelmap_in_viewer),
    }


def _set_labelmap_overlay(
    *,
    labelmap_volume_id: Optional[str] = None,
    labelmap_volume_name: Optional[str] = None,
    reference_volume_id: Optional[str] = None,
    reference_volume_name: Optional[str] = None,
    visible: bool = True,
    opacity: float = 0.3,
    visible_label_values: Optional[Sequence[int]] = None,
    color: Optional[Sequence[float]] = None,
    color_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    reference = (
        _get_node_by_id_or_name(node_id=reference_volume_id, node_name=reference_volume_name)
        if (reference_volume_id or reference_volume_name)
        else _get_active_volume_node()
    )
    if reference is None:
        raise RuntimeError("No reference volume available for labelmap overlay.")

    label_node = None
    if visible:
        label_node = _get_node_by_id_or_name(node_id=labelmap_volume_id, node_name=labelmap_volume_name)
        if label_node is None or not label_node.IsA("vtkMRMLLabelMapVolumeNode"):
            raise RuntimeError("Labelmap overlay node is not a vtkMRMLLabelMapVolumeNode.")
        display_node = label_node.GetDisplayNode()
        if display_node is None:
            label_node.CreateDefaultDisplayNodes()
            display_node = label_node.GetDisplayNode()
        values = [int(v) for v in (visible_label_values or [])]
        if not values:
            try:
                import numpy as np

                arr = slicer.util.arrayFromVolume(label_node)
                values = [int(v) for v in sorted(np.unique(arr).tolist()) if int(v) != 0]
            except Exception:
                values = [1]
        try:
            alpha = max(0.0, min(1.0, float(opacity)))
            table_alpha = alpha if color_alpha is None else max(0.0, min(1.0, float(color_alpha)))
            color_values = [float(v) for v in (color or (0.0, 1.0, 0.0))]
            while len(color_values) < 3:
                color_values.append(0.0)
            r, g, b = [max(0.0, min(1.0, float(v))) for v in color_values[:3]]
            max_value = max([1] + values)
            color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", f"{label_node.GetName()} green overlay")
            color_node.SetTypeToUser()
            color_node.SetNumberOfColors(int(max_value) + 1)
            for idx in range(int(max_value) + 1):
                color_node.SetColor(int(idx), str(idx), 0.0, 0.0, 0.0, 0.0)
            for value in values:
                if value > 0:
                    color_node.SetColor(int(value), str(value), r, g, b, table_alpha)
            if display_node is not None and hasattr(display_node, "SetAndObserveColorNodeID"):
                display_node.SetAndObserveColorNodeID(color_node.GetID())
        except Exception:
            pass

    slicer.util.setSliceViewerLayers(background=reference, foreground=None, label=(label_node if visible else None), fit=False)
    app_logic = slicer.app.applicationLogic()
    sel = app_logic.GetSelectionNode() if app_logic is not None else None
    if sel is not None and hasattr(sel, "SetReferenceActiveLabelVolumeID"):
        sel.SetReferenceActiveLabelVolumeID(label_node.GetID() if visible and label_node is not None else "")
    for slicer_view, _public_view in _SLICE_PUBLIC_VIEWS:
        try:
            comp = _slice_composite_node(slicer_view)
            if comp is not None and hasattr(comp, "SetLabelOpacity"):
                comp.SetLabelOpacity(max(0.0, min(1.0, float(opacity))) if visible else 0.0)
        except Exception:
            pass
    slicer.app.processEvents()
    return {
        "ok": True,
        "visible": bool(visible),
        "labelmap_volume_id": label_node.GetID() if label_node is not None else None,
        "labelmap_volume_name": label_node.GetName() if label_node is not None else None,
        "reference_volume_id": reference.GetID(),
        "reference_volume_name": reference.GetName(),
        "opacity": max(0.0, min(1.0, float(opacity))),
    }



def _ping_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bridge_version": BRIDGE_VERSION,
        "slicer_version": slicer.app.applicationVersion,
        "slicer_revision": slicer.app.revision,
    }


CORE_BRIDGE_HANDLERS: Dict[str, Any] = {
    # Scene / data
    "ping": _ping_handler,
    "clear_scene": lambda args: _clear_scene(),
    "load_dicom": lambda args: _load_dicom(
        dicom_dir=str(args["dicom_dir"]),
        clear_scene_first=bool(args.get("clear_scene_first", True)),
        active_prefer=args.get("active_prefer"),
    ),
    "load_dicom_series": lambda args: _load_dicom_series(
        dicom_dirs=args.get("dicom_dirs") or [],
        clear_scene_first=bool(args.get("clear_scene_first", True)),
        active_prefer=args.get("active_prefer"),
    ),
    "load_nifti": lambda args: _load_nifti(
        nifti_dir=args.get("nifti_dir"),
        nifti_files=args.get("nifti_files"),
        clear_scene_first=bool(args.get("clear_scene_first", True)),
        active_prefer=args.get("active_prefer"),
    ),
    "select_volume": lambda args: _select_volume(volume_id=args.get("volume_id"), volume_name=args.get("volume_name")),
    "export_volume_nifti": lambda args: _export_volume_nifti(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        output_path=str(args["output_path"]),
    ),
    "ras_to_volume_ijk": lambda args: _ras_to_volume_ijk(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        ras=args["ras"],
    ),
    "load_labelmap_as_segmentation": lambda args: _load_labelmap_as_segmentation(
        labelmap_path=str(args["labelmap_path"]),
        reference_volume_id=args.get("reference_volume_id"),
        reference_volume_name=args.get("reference_volume_name"),
        segmentation_name=args.get("segmentation_name"),
        remove_labelmap=bool(args.get("remove_labelmap", True)),
        show_labelmap_in_viewer=bool(args.get("show_labelmap_in_viewer", False)),
        visible_label_values=args.get("visible_label_values"),
        label_opacity=float(args.get("label_opacity", 0.3)),
    ),
    "set_labelmap_overlay": lambda args: _set_labelmap_overlay(
        labelmap_volume_id=args.get("labelmap_volume_id"),
        labelmap_volume_name=args.get("labelmap_volume_name"),
        reference_volume_id=args.get("reference_volume_id"),
        reference_volume_name=args.get("reference_volume_name"),
        visible=bool(args.get("visible", True)),
        opacity=float(args.get("opacity", 0.3)),
        visible_label_values=args.get("visible_label_values"),
        color=args.get("color"),
        color_alpha=args.get("color_alpha"),
    ),
    # Viewer actuation
    "get_viewer_state": lambda args: _get_viewer_state(),
    "set_window_level": lambda args: _set_window_level(
        window=args.get("window"),
        level=args.get("level"),
        auto=args.get("auto"),
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
    ),
    "set_interpolation": lambda args: _set_interpolation(
        interpolate=bool(args["interpolate"]),
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
    ),
    "set_slice_orientation": lambda args: _set_slice_orientation(view=str(args["view"]), orientation=str(args["orientation"])),
    "get_slice_offset_range": lambda args: _get_slice_offset_range(view=str(args["view"])),
    "set_slice_offset": lambda args: _set_slice_offset(view=str(args["view"]), offset=float(args["offset"])),
    "set_slice_scroll_to": lambda args: _set_slice_scroll_to(view=str(args["view"]), scroll_to=float(args["scroll_to"])),
    "fit_slice": lambda args: _fit_slice(view=str(args["view"])),
    "zoom_slice_relative": lambda args: _zoom_slice_relative(view=str(args["view"]), factor=float(args["factor"])),
    "set_field_of_view": lambda args: _set_field_of_view(view=str(args["view"]), field_of_view=args["field_of_view"]),
    "set_xyz_origin": lambda args: _set_xyz_origin(view=str(args["view"]), xyz_origin=args["xyz_origin"]),
    "jump_to_ras": lambda args: _jump_to_ras(ras=args["ras"], centered=bool(args.get("centered", True))),
    "capture_slice_view_png": lambda args: _capture_slice_view_png(
        view=str(args.get("view", "axial")),
        output_path=str(args["output_path"]),
        include_controller=bool(args.get("include_controller", False)),
    ),
    "recover_standard_views": lambda args: _recover_standard_views(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        centered=bool(args.get("centered", True)),
        fit=bool(args.get("fit", True)),
    ),
    "set_linked_slices": lambda args: _set_linked_slices(enabled=bool(args["enabled"])),
    "set_layout": lambda args: _set_layout(layout=str(args["layout"])),
    # Deterministic quantification / derived volumes
    "roi_stats_ijk": lambda args: _roi_stats_ijk(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        ijk_min=args.get("ijk_min"),
        ijk_max=args.get("ijk_max"),
    ),
    "roi_stats_ras_box": lambda args: _roi_stats_ras_box(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        ras_min=args.get("ras_min"),
        ras_max=args.get("ras_max"),
    ),
    "sample_intensity_ras": lambda args: _sample_intensity_ras(
        volume_id=args.get("volume_id"),
        volume_name=args.get("volume_name"),
        ras=args.get("ras"),
        method=str(args.get("method", "nearest")),
    ),
    "measure_distance_ras": lambda args: _measure_distance_ras(p1=args.get("p1"), p2=args.get("p2")),
    "measure_angle_ras": lambda args: _measure_angle_ras(p1=args.get("p1"), p2=args.get("p2"), p3=args.get("p3")),
    "measure_area_polygon_ras": lambda args: _measure_area_polygon_ras(points=args.get("points")),
    "set_thick_slab": lambda args: _set_thick_slab(
        view=str(args["view"]),
        enabled=bool(args.get("enabled", True)),
        thickness_mm=float(args.get("thickness_mm", 0.0)),
        mode=str(args.get("mode", "mip")),
    ),
    "set_fusion": lambda args: _set_fusion(
        background_volume_id=str(args["background_volume_id"]),
        foreground_volume_id=str(args["foreground_volume_id"]),
        opacity=float(args.get("opacity", 0.5)),
    ),
    "clear_fusion": lambda args: _clear_fusion(),
    "compute_subtraction": lambda args: _compute_subtraction(
        volume_a_id=str(args["volume_a_id"]),
        volume_b_id=str(args["volume_b_id"]),
        output_name=str(args.get("output_name", "Subtraction")),
    ),
    "resample_volume_isotropic": lambda args: _resample_volume_isotropic(
        output_spacing_mm=float(args.get("output_spacing_mm", 1.0)),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        output_name=args.get("output_name"),
        interpolation_type=str(args.get("interpolation_type", "linear")),
        make_active=bool(args.get("make_active", True)),
    ),
}
