
"""Segmentation operations for the Slicer bridge."""
from __future__ import annotations

import time

from .common import *  # noqa: F401,F403
from .core import _fit_slice, _ras_to_ijk_float, _set_labelmap_overlay, _set_slice_offset, _set_slice_orientation, _set_slice_scroll_to

# -----------------------------------------------------------------------------
# Segmentation helpers / curated Segment Editor wrappers
# -----------------------------------------------------------------------------


def _list_segmentation_nodes() -> List[Any]:
    try:
        return list(slicer.util.getNodesByClass("vtkMRMLSegmentationNode"))
    except Exception:
        return []


def _get_segmentation_node(*, segmentation_id: Optional[str] = None, segmentation_name: Optional[str] = None):
    if segmentation_id or segmentation_name:
        node = _get_node_by_id_or_name(node_id=segmentation_id, node_name=segmentation_name)
        if not node.IsA("vtkMRMLSegmentationNode"):
            raise ValueError(f"Node is not a segmentation: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
        return node

    nodes = _list_segmentation_nodes()
    if len(nodes) == 1:
        return nodes[0]
    if not nodes:
        raise ValueError("No segmentation nodes are loaded.")
    raise ValueError("Multiple segmentations are loaded; specify segmentation_id or segmentation_name.")


def _iter_segment_ids(segmentation_node) -> List[str]:
    segmentation = segmentation_node.GetSegmentation()
    ids: List[str] = []
    for i in range(int(segmentation.GetNumberOfSegments())):
        try:
            segment_id = segmentation.GetNthSegmentID(i)
        except Exception:
            segment_id = None
        if segment_id:
            ids.append(str(segment_id))
    return ids


def _find_segment_id(segmentation_node, *, segment_id: Optional[str] = None, segment_name: Optional[str] = None) -> Optional[str]:
    segmentation = segmentation_node.GetSegmentation()
    if segment_id:
        try:
            if segmentation.GetSegment(str(segment_id)) is not None:
                return str(segment_id)
        except Exception:
            pass
    if segment_name:
        try:
            sid = segmentation.GetSegmentIdBySegmentName(str(segment_name))
            if sid:
                return str(sid)
        except Exception:
            pass
        want = str(segment_name).strip().lower()
        for sid in _iter_segment_ids(segmentation_node):
            segment = segmentation.GetSegment(sid)
            if segment is None:
                continue
            if want == str(segment.GetName()).strip().lower():
                return str(sid)
        for sid in _iter_segment_ids(segmentation_node):
            segment = segmentation.GetSegment(sid)
            if segment is None:
                continue
            if want in str(segment.GetName()).strip().lower():
                return str(sid)
    return None


def _resolve_source_volume_node(*, volume_id: Optional[str] = None, volume_name: Optional[str] = None):
    node = None
    if volume_id or volume_name:
        node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    else:
        node = _get_active_volume_node()
    if node is None:
        raise ValueError("No source volume is available.")
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Source node is not a volume: {node.GetClassName()}")
    return node


def _ensure_segmentation_node(
    *,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    source_volume_node=None,
    create_if_missing: bool = False,
):
    try:
        node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
        created = False
    except Exception:
        if not create_if_missing:
            raise
        name = str(segmentation_name or "Segmentation")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        node.CreateDefaultDisplayNodes()
        created = True

    if source_volume_node is not None and hasattr(node, "SetReferenceImageGeometryParameterFromVolumeNode"):
        node.SetReferenceImageGeometryParameterFromVolumeNode(source_volume_node)

    display_node = node.GetDisplayNode()
    if display_node is not None:
        if hasattr(display_node, "SetVisibility2D"):
            display_node.SetVisibility2D(True)
        if hasattr(display_node, "SetVisibility3D"):
            display_node.SetVisibility3D(True)
        if hasattr(display_node, "SetOpacity"):
            display_node.SetOpacity(1.0)

    return node, created


def _ensure_segment(segmentation_node, *, segment_id: Optional[str] = None, segment_name: Optional[str] = None, create_if_missing: bool = False):
    segmentation = segmentation_node.GetSegmentation()
    sid = _find_segment_id(segmentation_node, segment_id=segment_id, segment_name=segment_name)
    if sid:
        return sid, False
    if not create_if_missing:
        raise ValueError(f"Segment not found in segmentation {segmentation_node.GetName()}: id={segment_id!r} name={segment_name!r}")

    name = str(segment_name or "Segment")
    new_id = None
    if hasattr(segmentation, "AddEmptySegment"):
        try:
            new_id = segmentation.AddEmptySegment("", name)
        except Exception:
            try:
                new_id = segmentation.AddEmptySegment(name)
            except Exception:
                new_id = None
    if not new_id:
        try:
            import vtkSegmentationCorePython as vtkSegmentationCore  # type: ignore

            segment = vtkSegmentationCore.vtkSegment()
            segment.SetName(name)
            segmentation.AddSegment(segment)
            new_id = _find_segment_id(segmentation_node, segment_name=name)
        except Exception as e:
            raise RuntimeError(f"Unable to create new segment {name!r}: {e}")
    return str(new_id), True


def _segment_summary(segmentation_node, segment_id: str) -> Dict[str, Any]:
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment is None:
        raise ValueError(f"Segment not found: {segment_id}")
    display_node = segmentation_node.GetDisplayNode()
    visible = None
    try:
        if display_node is not None and hasattr(display_node, "GetSegmentVisibility"):
            visible = bool(display_node.GetSegmentVisibility(segment_id))
    except Exception:
        visible = None
    return {
        "segment_id": str(segment_id),
        "segment_name": str(segment.GetName()),
        "color": list(segment.GetColor()) if hasattr(segment, "GetColor") else None,
        "visible": visible,
    }


def _segmentation_summary(segmentation_node) -> Dict[str, Any]:
    seg_ids = _iter_segment_ids(segmentation_node)
    return {
        "id": segmentation_node.GetID(),
        "name": segmentation_node.GetName(),
        "n_segments": len(seg_ids),
        "segments": [_segment_summary(segmentation_node, sid) for sid in seg_ids[:50]],
    }


def _list_segmentations() -> Dict[str, Any]:
    nodes = _list_segmentation_nodes()
    segs = [_segmentation_summary(node) for node in nodes]
    active_seg = segs[0] if len(segs) == 1 else None
    return {
        "segmentations": segs,
        "active_segmentation": active_seg,
        "n_segmentations": len(segs),
    }


def _list_segments(*, segmentation_id: Optional[str] = None, segmentation_name: Optional[str] = None) -> Dict[str, Any]:
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    seg_ids = _iter_segment_ids(segmentation_node)
    return {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segments": [_segment_summary(segmentation_node, sid) for sid in seg_ids],
        "n_segments": len(seg_ids),
    }


def _segment_bounds_and_center_ras(segmentation_node, segment_id: str) -> Tuple[List[Optional[float]], List[float], str]:
    bounds: List[Optional[float]] = [None, None, None, None, None, None]
    source = "segment_center"
    try:
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        if segment is not None and hasattr(segment, "GetBounds"):
            raw = [0.0] * 6
            segment.GetBounds(raw)
            if all(math.isfinite(float(v)) for v in raw) and raw[0] <= raw[1] and raw[2] <= raw[3] and raw[4] <= raw[5]:
                bounds = [float(v) for v in raw]
                source = "segment_bounds"
    except Exception:
        pass

    if bounds[0] is None:
        try:
            raw = [0.0] * 6
            if hasattr(segmentation_node, "GetRASBounds"):
                segmentation_node.GetRASBounds(raw)
            else:
                segmentation_node.GetBounds(raw)
            if all(math.isfinite(float(v)) for v in raw) and raw[0] <= raw[1] and raw[2] <= raw[3] and raw[4] <= raw[5]:
                bounds = [float(v) for v in raw]
                source = "segmentation_bounds"
        except Exception:
            pass

    try:
        center_raw = segmentation_node.GetSegmentCenterRAS(segment_id)
        center = [float(center_raw[0]), float(center_raw[1]), float(center_raw[2])]
    except Exception:
        if bounds[0] is None:
            raise
        center = [
            float((bounds[0] + bounds[1]) / 2.0),
            float((bounds[2] + bounds[3]) / 2.0),
            float((bounds[4] + bounds[5]) / 2.0),
        ]
        source = f"{source}_center"
    return bounds, center, source


def _show_segmentation(
    *,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
    visible: bool = True,
    fill: bool = True,
    outline: bool = True,
    opacity: float = 0.6,
    color: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    display_node = segmentation_node.GetDisplayNode()
    if display_node is None:
        segmentation_node.CreateDefaultDisplayNodes()
        display_node = segmentation_node.GetDisplayNode()

    target_segment_id = _find_segment_id(segmentation_node, segment_id=segment_id, segment_name=segment_name)
    has_target = target_segment_id is not None
    target_ids = [target_segment_id] if has_target else _iter_segment_ids(segmentation_node)
    is_visible = bool(visible)
    if display_node is not None:
        node_visible = True if has_target else is_visible
        for method_name, value in (
            ("SetVisibility", node_visible),
            ("SetVisibility2D", node_visible),
            ("SetVisibility3D", node_visible),
            ("SetVisibility2DFill", bool(fill)),
            ("SetVisibility2DOutline", bool(outline)),
        ):
            try:
                method = getattr(display_node, method_name, None)
                if callable(method):
                    method(value)
            except Exception:
                pass
        try:
            if hasattr(display_node, "SetOpacity"):
                display_node.SetOpacity(float(opacity))
        except Exception:
            pass
        for sid in target_ids:
            try:
                if hasattr(display_node, "SetSegmentVisibility"):
                    display_node.SetSegmentVisibility(str(sid), is_visible)
            except Exception:
                pass

    if color is not None:
        rgb = [float(v) for v in color[:3]]
        if len(rgb) == 3:
            for sid in target_ids:
                try:
                    segment = segmentation_node.GetSegmentation().GetSegment(str(sid))
                    if segment is not None and hasattr(segment, "SetColor"):
                        segment.SetColor(rgb[0], rgb[1], rgb[2])
                except Exception:
                    pass

    labelmap_overlay = None
    if not is_visible:
        labelmap_overlay = _clear_labelmap_overlay()

    slicer.app.processEvents()
    result = _segmentation_summary(segmentation_node)
    target_segment = segmentation_node.GetSegmentation().GetSegment(target_segment_id) if target_segment_id else None
    result.update(
        {
            "segmentation_id": segmentation_node.GetID(),
            "segmentation_name": segmentation_node.GetName(),
            "segment_id": str(target_segment_id) if target_segment_id else None,
            "segment_name": target_segment.GetName() if target_segment is not None else None,
            "visible": is_visible,
            "fill": bool(fill),
            "outline": bool(outline),
            "opacity": float(opacity),
        }
    )
    if labelmap_overlay is not None:
        result["labelmap_overlay"] = labelmap_overlay
    return result


def _clear_labelmap_overlay() -> Dict[str, Any]:
    reference = _get_active_volume_node() or _first_scalar_volume_node()
    if reference is None:
        return {"ok": False, "visible": False, "error": "No reference volume available for labelmap overlay."}
    return _set_labelmap_overlay(reference_volume_id=reference.GetID(), visible=False)


def _hide_segmentation_overlay(
    *,
    target: str = "all",
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    target_norm = str(target or "all").strip().lower()
    hide_all = target_norm in {"all", "*"}
    if hide_all:
        nodes = _list_segmentation_nodes()
    else:
        nodes = [_get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)]

    hidden: List[Dict[str, Any]] = []
    for node in nodes:
        display_node = node.GetDisplayNode()
        if display_node is None:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        target_segment_id = _find_segment_id(node, segment_id=segment_id, segment_name=segment_name)
        target_ids = [target_segment_id] if target_segment_id is not None else _iter_segment_ids(node)
        if display_node is not None:
            if target_segment_id is None:
                for method_name, value in (
                    ("SetVisibility", False),
                    ("SetVisibility2D", False),
                    ("SetVisibility3D", False),
                    ("SetVisibility2DFill", False),
                    ("SetVisibility2DOutline", False),
                ):
                    try:
                        method = getattr(display_node, method_name, None)
                        if callable(method):
                            method(value)
                    except Exception:
                        pass
            for sid in target_ids:
                for method_name, args in (
                    ("SetSegmentVisibility", (str(sid), False)),
                    ("SetSegmentVisibility2DFill", (str(sid), False)),
                    ("SetSegmentVisibility2DOutline", (str(sid), False)),
                ):
                    try:
                        method = getattr(display_node, method_name, None)
                        if callable(method):
                            method(*args)
                    except Exception:
                        pass
            try:
                display_node.Modified()
            except Exception:
                pass
        try:
            node.Modified()
        except Exception:
            pass
        summary = _segmentation_summary(node)
        summary.update({"segmentation_id": node.GetID(), "segmentation_name": node.GetName(), "visible": False})
        if target_segment_id is not None:
            summary["segment_id"] = str(target_segment_id)
        hidden.append(summary)

    labelmap_overlay = _clear_labelmap_overlay()
    slicer.app.processEvents()
    return {
        "ok": bool(labelmap_overlay.get("ok", True)),
        "target": "all" if hide_all else "segmentation_id",
        "visible": False,
        "hidden_segmentations": hidden,
        "labelmap_overlay": labelmap_overlay,
        "segmentation_id": None if hide_all or not hidden else hidden[0].get("segmentation_id"),
        "segmentation_name": None if hide_all or not hidden else hidden[0].get("segmentation_name"),
    }


def _show_segment_overlay_in_views(segmentation_node, segment_id: str, *, source_volume_node=None, opacity: float = 0.1) -> Dict[str, Any]:
    """Make Slicer's native 2D segmentation overlay visible for screenshot capture."""
    if source_volume_node is not None:
        try:
            _set_active_volume(source_volume_node)
        except Exception:
            pass
    result = _show_segmentation(
        segmentation_id=segmentation_node.GetID(),
        segment_id=str(segment_id),
        visible=True,
        fill=True,
        outline=True,
        opacity=float(opacity),
        color=[0.0, 1.0, 0.0],
    )
    display_node = segmentation_node.GetDisplayNode()
    if display_node is not None:
        for method_name, args in (
            ("SetOpacity2DFill", (float(opacity),)),
            ("SetOpacity2DOutline", (1.0,)),
            ("SetSegmentOpacity2DFill", (str(segment_id), float(opacity))),
            ("SetSegmentOpacity2DOutline", (str(segment_id), 1.0)),
            ("SetSegmentVisibility", (str(segment_id), True)),
            ("SetSegmentVisibility2DFill", (str(segment_id), True)),
            ("SetSegmentVisibility2DOutline", (str(segment_id), True)),
        ):
            try:
                method = getattr(display_node, method_name, None)
                if callable(method):
                    method(*args)
            except Exception:
                pass
        try:
            display_node.Modified()
        except Exception:
            pass
    try:
        segment = segmentation_node.GetSegmentation().GetSegment(str(segment_id))
        if segment is not None:
            segment.SetColor(0.0, 1.0, 0.0)
            segment.Modified()
    except Exception:
        pass
    try:
        segmentation_node.Modified()
    except Exception:
        pass
    labelmap_result = None
    if source_volume_node is not None:
        try:
            labelmap_result = _export_segment_labelmap_overlay(
                segmentation_node=segmentation_node,
                segment_id=str(segment_id),
                source_volume_node=source_volume_node,
                opacity=float(opacity),
            )
        except Exception as exc:
            labelmap_result = {"ok": False, "error": str(exc)}
    try:
        if hasattr(slicer.util, "forceRenderAllViews"):
            slicer.util.forceRenderAllViews()
    except Exception:
        pass
    for view in ("axial", "sagittal", "coronal"):
        try:
            slice_view = _slice_widget(view).sliceView()
            if hasattr(slice_view, "forceRender"):
                slice_view.forceRender()
            if hasattr(slice_view, "repaint"):
                slice_view.repaint()
        except Exception:
            pass
    for _ in range(5):
        slicer.app.processEvents()
        time.sleep(0.05)
    if labelmap_result is not None:
        result["labelmap_overlay"] = labelmap_result
    return result


def _export_segment_labelmap_overlay(*, segmentation_node, segment_id: str, source_volume_node, opacity: float = 0.1) -> Dict[str, Any]:
    import vtk

    labelmap_name = f"{segmentation_node.GetName()}_{segment_id}_overlay_label"
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", labelmap_name)
    labelmap_node.CreateDefaultDisplayNodes()

    segment_ids = vtk.vtkStringArray()
    segment_ids.InsertNextValue(str(segment_id))
    logic = slicer.modules.segmentations.logic()
    ok = logic.ExportSegmentsToLabelmapNode(segmentation_node, segment_ids, labelmap_node, source_volume_node)
    if ok is False:
        raise RuntimeError("ExportSegmentsToLabelmapNode returned false")

    result = _set_labelmap_overlay(
        labelmap_volume_id=labelmap_node.GetID(),
        reference_volume_id=source_volume_node.GetID(),
        visible=True,
        opacity=float(opacity),
        color=(0.0, 1.0, 0.0),
        color_alpha=1.0,
    )
    result.update({"labelmap_volume_id": labelmap_node.GetID(), "labelmap_volume_name": labelmap_node.GetName()})
    return result


def _create_segment_editor(segmentation_node, source_volume_node=None):
    try:
        import SegmentEditorEffects  # noqa: F401
    except Exception:
        pass
    widget = slicer.qMRMLSegmentEditorWidget()
    widget.setMRMLScene(slicer.mrmlScene)
    editor_node = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(editor_node)
    if hasattr(editor_node, "SetOverwriteMode") and hasattr(slicer.vtkMRMLSegmentEditorNode, "OverwriteNone"):
        editor_node.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
    widget.setMRMLSegmentEditorNode(editor_node)
    widget.setSegmentationNode(segmentation_node)
    if source_volume_node is not None:
        if hasattr(widget, "setSourceVolumeNode"):
            widget.setSourceVolumeNode(source_volume_node)
        elif hasattr(widget, "setMasterVolumeNode"):
            widget.setMasterVolumeNode(source_volume_node)
    return widget, editor_node


def _dispose_qt_widget(widget) -> None:
    if widget is None:
        return
    try:
        if hasattr(widget, "hide"):
            widget.hide()
    except Exception:
        pass
    try:
        widget.setParent(None)
    except Exception:
        pass
    try:
        widget.deleteLater()
    except Exception:
        pass
    try:
        slicer.app.processEvents()
    except Exception:
        pass


def _cleanup_segment_editor(widget, editor_node) -> None:
    try:
        if hasattr(widget, "setActiveEffectByName"):
            widget.setActiveEffectByName("")
    except Exception:
        pass
    try:
        if hasattr(widget, "setMRMLSegmentEditorNode"):
            widget.setMRMLSegmentEditorNode(None)
    except Exception:
        pass
    try:
        if hasattr(widget, "setSegmentationNode"):
            widget.setSegmentationNode(None)
    except Exception:
        pass
    try:
        if hasattr(widget, "setSourceVolumeNode"):
            widget.setSourceVolumeNode(None)
        elif hasattr(widget, "setMasterVolumeNode"):
            widget.setMasterVolumeNode(None)
    except Exception:
        pass
    try:
        if editor_node is not None:
            slicer.mrmlScene.RemoveNode(editor_node)
    except Exception:
        pass
    _dispose_qt_widget(widget)


def _set_selected_segment(widget, editor_node, segment_id: str) -> None:
    if hasattr(editor_node, "SetSelectedSegmentID"):
        editor_node.SetSelectedSegmentID(str(segment_id))
    if hasattr(widget, "setCurrentSegmentID"):
        try:
            widget.setCurrentSegmentID(str(segment_id))
        except Exception:
            pass
    slicer.app.processEvents()


def _apply_effect(widget, effect_name: str, params: Dict[str, Any]) -> None:
    widget.setActiveEffectByName(effect_name)
    effect = widget.activeEffect()
    if effect is None:
        raise RuntimeError(f"Segment Editor effect not available: {effect_name}")
    for key, value in params.items():
        if value is None:
            continue
        effect.setParameter(str(key), str(value))

    effect_impl = effect.self() if hasattr(effect, "self") else effect
    for fn_name in ("onApply", "apply"):
        fn = getattr(effect_impl, fn_name, None)
        if callable(fn):
            fn()
            slicer.app.processEvents()
            return
    raise RuntimeError(f"Could not apply Segment Editor effect {effect_name!r}: no onApply/apply method")


def _segment_basic_stats(segmentation_node, segment_id: str, source_volume_node=None) -> Dict[str, Any]:
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment is None:
        raise ValueError(f"Segment not found: {segment_id}")

    stats: Dict[str, Any] = {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segment_id": str(segment_id),
        "segment_name": segment.GetName(),
    }

    try:
        center_ras = segmentation_node.GetSegmentCenterRAS(segment_id)
        stats["center_ras"] = [float(x) for x in center_ras]
    except Exception:
        stats["center_ras"] = None

    if source_volume_node is None:
        return stats

    try:
        import numpy as np

        segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id, source_volume_node)
        voxel_count = int(np.count_nonzero(segment_array))
        stats["voxel_count"] = voxel_count
        spacing = [float(x) for x in source_volume_node.GetSpacing()]
        voxel_volume_mm3 = abs(spacing[0] * spacing[1] * spacing[2])
        stats["volume_mm3"] = float(voxel_count * voxel_volume_mm3)
        stats["volume_cm3"] = float(stats["volume_mm3"] / 1000.0)

        if voxel_count > 0:
            volume_array = slicer.util.arrayFromVolume(source_volume_node)
            voxels = volume_array[segment_array > 0]
            stats["min"] = float(voxels.min())
            stats["max"] = float(voxels.max())
            stats["mean"] = float(voxels.mean())
            stats["median"] = float(np.median(voxels))
            stats["std"] = float(voxels.std())
    except Exception as e:
        stats["statistics_warning"] = str(e)

    stats["source_volume_id"] = source_volume_node.GetID()
    stats["source_volume_name"] = source_volume_node.GetName()
    return stats


def _segment_from_threshold(
    *,
    minimum_threshold: float,
    maximum_threshold: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(
            widget,
            "Threshold",
            {
                "MinimumThreshold": float(minimum_threshold),
                "MaximumThreshold": float(maximum_threshold),
            },
        )
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)

    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "minimum_threshold": float(minimum_threshold),
        "maximum_threshold": float(maximum_threshold),
    }


def _slice_ras_from_normalized(*, view: str, x_norm: float, y_norm: float, orientation: Optional[str] = None, offset: Optional[float] = None, scroll_to: Optional[float] = None) -> List[float]:
    public_view = _public_view(view)
    if orientation is not None:
        _set_slice_orientation(view=public_view, orientation=str(orientation))
    if offset is not None:
        _set_slice_offset(view=public_view, offset=float(offset))
    elif scroll_to is not None:
        _set_slice_scroll_to(view=public_view, scroll_to=float(scroll_to))
    sn = _slice_node(public_view)
    dims = None
    if hasattr(sn, "GetDimensions"):
        try:
            dims = sn.GetDimensions()
        except Exception:
            dims = None
    if not dims or len(dims) < 2 or dims[0] <= 1 or dims[1] <= 1:
        try:
            slice_view = _slice_widget(view).sliceView()
            width_attr = getattr(slice_view, "width", None)
            height_attr = getattr(slice_view, "height", None)
            width = int(width_attr() if callable(width_attr) else width_attr)
            height = int(height_attr() if callable(height_attr) else height_attr)
            dims = (width, height, 1)
        except Exception:
            dims = (512, 512, 1)
    width = max(float(dims[0]), 1.0)
    height = max(float(dims[1]), 1.0)
    x = max(0.0, min(1.0, float(x_norm))) * width
    # Observed PNGs and normal image coordinates use top-left origin; VTK slice XY uses bottom-left.
    y = (1.0 - max(0.0, min(1.0, float(y_norm)))) * height
    m = sn.GetXYToRAS()
    out = [0.0, 0.0, 0.0, 0.0]
    m.MultiplyPoint([x, y, 0.0, 1.0], out)
    return [float(out[0]), float(out[1]), float(out[2])]




def _image_coords_to_ras(
    *,
    view: str,
    point_norm: Optional[Sequence[float]] = None,
    bbox_norm: Optional[Sequence[float]] = None,
    orientation: Optional[str] = None,
    offset: Optional[float] = None,
    scroll_to: Optional[float] = None,
) -> Dict[str, Any]:
    public_view = _public_view(view)
    if point_norm is None and bbox_norm is None:
        raise ValueError("Provide exactly one of point_norm or bbox_norm. Never send both. If you want the center of a bbox, send only bbox_norm.")

    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    if point_norm is not None:
        if len(point_norm) != 2:
            raise ValueError("point_norm must be [x_norm, y_norm]")
        x_norm = _clamp01(float(point_norm[0]))
        y_norm = _clamp01(float(point_norm[1]))
        point_ras = _slice_ras_from_normalized(view=public_view, x_norm=x_norm, y_norm=y_norm, orientation=orientation, offset=offset, scroll_to=scroll_to)
        return {
            "ok": True,
            "view": public_view,
            "point_norm": [x_norm, y_norm],
            "orientation": str(orientation) if orientation is not None else None,
            "offset": float(offset) if offset is not None else None,
            "scroll_to": float(scroll_to) if scroll_to is not None else None,
            "point_ras": point_ras,
            "center_ras": point_ras,
        }

    if bbox_norm is None or len(bbox_norm) != 4:
        raise ValueError("bbox_norm must be [x_min_norm, y_min_norm, x_max_norm, y_max_norm]")
    x1, y1, x2, y2 = [_clamp01(float(v)) for v in bbox_norm]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    corners = {
        "top_left": _slice_ras_from_normalized(view=public_view, x_norm=x1, y_norm=y1, orientation=orientation, offset=offset, scroll_to=scroll_to),
        "top_right": _slice_ras_from_normalized(view=public_view, x_norm=x2, y_norm=y1, orientation=orientation, offset=offset, scroll_to=scroll_to),
        "bottom_right": _slice_ras_from_normalized(view=public_view, x_norm=x2, y_norm=y2, orientation=orientation, offset=offset, scroll_to=scroll_to),
        "bottom_left": _slice_ras_from_normalized(view=public_view, x_norm=x1, y_norm=y2, orientation=orientation, offset=offset, scroll_to=scroll_to),
    }
    center_ras = _slice_ras_from_normalized(view=public_view, x_norm=cx, y_norm=cy, orientation=orientation, offset=offset, scroll_to=scroll_to)
    return {
        "ok": True,
        "view": public_view,
        "bbox_norm": [x1, y1, x2, y2],
        "bbox_corners_ras": corners,
        "center_norm": [cx, cy],
        "orientation": str(orientation) if orientation is not None else None,
        "offset": float(offset) if offset is not None else None,
        "scroll_to": float(scroll_to) if scroll_to is not None else None,
        "center_ras": center_ras,
        "point_ras": center_ras,
    }



def _resolve_segment_seed_ras(
    *,
    seed_view: Optional[str] = None,
    seed_x_norm: Optional[float] = None,
    seed_y_norm: Optional[float] = None,
    ras_seed: Optional[Sequence[float]] = None,
    seed_segment_id: Optional[str] = None,
    seed_segment_name: Optional[str] = None,
    segmentation_node=None,
    seed_orientation: Optional[str] = None,
    seed_offset_mm: Optional[float] = None,
    seed_scroll_to: Optional[float] = None,
) -> Tuple[List[float], str]:
    if ras_seed is not None:
        if len(ras_seed) != 3:
            raise ValueError("ras_seed must be length-3 [x,y,z] in RAS mm")
        return [float(v) for v in ras_seed], "ras_seed"

    if seed_segment_id or seed_segment_name:
        if segmentation_node is None:
            raise ValueError("seed_segment_id/seed_segment_name requires a segmentation node")
        sid = _find_segment_id(segmentation_node, segment_id=seed_segment_id, segment_name=seed_segment_name)
        if not sid:
            raise ValueError(f"Seed segment not found: id={seed_segment_id!r} name={seed_segment_name!r}")
        center_ras = segmentation_node.GetSegmentCenterRAS(sid)
        return [float(v) for v in center_ras], "segment_center"

    if seed_view is not None and seed_x_norm is not None and seed_y_norm is not None:
        return _slice_ras_from_normalized(view=str(seed_view), x_norm=float(seed_x_norm), y_norm=float(seed_y_norm), orientation=seed_orientation, offset=seed_offset_mm, scroll_to=seed_scroll_to), "view_normalized"

    raise ValueError(
        "Provide one seed source: ras_seed=[x,y,z], or seed_view + seed_x_norm + seed_y_norm, or seed_segment_id/seed_segment_name."
    )


def _segment_local_threshold(
    *,
    minimum_threshold: float,
    maximum_threshold: float,
    minimum_diameter_mm: float = 3.0,
    feature_size_mm: float = 3.0,
    segmentation_algorithm: str = "GrowCut",
    seed_view: Optional[str] = None,
    seed_x_norm: Optional[float] = None,
    seed_y_norm: Optional[float] = None,
    ras_seed: Optional[Sequence[float]] = None,
    seed_segment_id: Optional[str] = None,
    seed_segment_name: Optional[str] = None,
    seed_orientation: Optional[str] = None,
    seed_offset_mm: Optional[float] = None,
    seed_scroll_to: Optional[float] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    import vtk

    allowed_algorithms = {"Masking", "GrowCut", "WaterShed"}
    algorithm = str(segmentation_algorithm or "GrowCut").strip()
    if algorithm not in allowed_algorithms:
        raise ValueError(f"Unsupported segmentation_algorithm: {segmentation_algorithm!r}. Allowed: {sorted(allowed_algorithms)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    seed_ras, seed_source = _resolve_segment_seed_ras(
        seed_view=seed_view,
        seed_x_norm=seed_x_norm,
        seed_y_norm=seed_y_norm,
        ras_seed=ras_seed,
        seed_segment_id=seed_segment_id,
        seed_segment_name=seed_segment_name,
        segmentation_node=segmentation_node,
        seed_orientation=seed_orientation,
        seed_offset_mm=seed_offset_mm,
        seed_scroll_to=seed_scroll_to,
    )
    seed_ijk = _ras_to_ijk_float(source_volume_node, seed_ras)
    ijk_points = vtk.vtkPoints()
    ijk_points.InsertNextPoint(float(seed_ijk[0]), float(seed_ijk[1]), float(seed_ijk[2]))

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        widget.setActiveEffectByName("Local Threshold")
        effect = widget.activeEffect()
        if effect is None:
            raise RuntimeError(
                "Segment Editor effect not available: Local Threshold. Install SegmentEditorExtraEffects and restart Slicer."
            )
        effect_name_attr = getattr(effect, "name", None)
        effect_name = effect_name_attr() if callable(effect_name_attr) else effect_name_attr
        if effect_name and "local threshold" not in str(effect_name).lower():
            raise RuntimeError(
                f"Active Segment Editor effect is {effect_name!r}, not 'Local Threshold'. Install SegmentEditorExtraEffects and restart Slicer."
            )
        effect.setParameter("MinimumThreshold", float(minimum_threshold))
        effect.setParameter("MaximumThreshold", float(maximum_threshold))
        effect.setParameter("MinimumDiameterMm", float(minimum_diameter_mm))
        effect.setParameter("FeatureSizeMm", float(feature_size_mm))
        effect.setParameter("SegmentationAlgorithm", algorithm)
        effect_impl = effect.self() if hasattr(effect, "self") else effect
        apply_fn = getattr(effect_impl, "apply", None)
        if not callable(apply_fn):
            raise RuntimeError("Local Threshold effect does not expose apply(ijkPoints) in this Slicer build.")
        apply_fn(ijk_points)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)

    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "minimum_threshold": float(minimum_threshold),
        "maximum_threshold": float(maximum_threshold),
        "minimum_diameter_mm": float(minimum_diameter_mm),
        "feature_size_mm": float(feature_size_mm),
        "segmentation_algorithm": algorithm,
        "seed_ras": [float(v) for v in seed_ras],
        "seed_ijk": [float(v) for v in seed_ijk],
        "seed_source": seed_source,
        "seed_view": str(seed_view) if seed_view is not None else None,
        "seed_orientation": str(seed_orientation) if seed_orientation is not None else None,
        "seed_offset_mm": float(seed_offset_mm) if seed_offset_mm is not None else None,
        "seed_scroll_to": float(seed_scroll_to) if seed_scroll_to is not None else None,
        "seed_x_norm": float(seed_x_norm) if seed_x_norm is not None else None,
        "seed_y_norm": float(seed_y_norm) if seed_y_norm is not None else None,
    }


def _segment_margin(
    *,
    margin_size_mm: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )
    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Margin", {"MarginSizeMm": float(margin_size_mm)})
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "margin_size_mm": float(margin_size_mm)}


def _segment_smoothing(
    *,
    smoothing_method: str = "MEDIAN",
    kernel_size_mm: float = 3.0,
    gaussian_std_mm: Optional[float] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"MEDIAN", "GAUSSIAN", "MORPHOLOGICAL_CLOSING"}
    method = str(smoothing_method or "MEDIAN").strip().upper()
    if method not in allowed:
        raise ValueError(f"Unsupported smoothing_method: {smoothing_method!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )
    params: Dict[str, Any] = {"SmoothingMethod": method, "KernelSizeMm": float(kernel_size_mm)}
    if method == "GAUSSIAN":
        params["GaussianStandardDeviationMm"] = float(gaussian_std_mm if gaussian_std_mm is not None else kernel_size_mm)

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Smoothing", params)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "smoothing_method": method, "kernel_size_mm": float(kernel_size_mm)}


def _segment_islands(
    *,
    operation: str = "KEEP_LARGEST_ISLAND",
    minimum_size: int = 1000,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"KEEP_LARGEST_ISLAND", "REMOVE_SMALL_ISLANDS"}
    op = str(operation or "KEEP_LARGEST_ISLAND").strip().upper()
    if op not in allowed:
        raise ValueError(f"Unsupported islands operation: {operation!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Islands", {"Operation": op, "MinimumSize": int(minimum_size)})
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "operation": op, "minimum_size": int(minimum_size)}


def _segment_logical(
    *,
    operation: str,
    modifier_segment_id: Optional[str] = None,
    modifier_segment_name: Optional[str] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"COPY", "UNION", "INTERSECT", "SUBTRACT", "INVERT", "CLEAR", "FILL"}
    op = str(operation or "").strip().upper()
    if op not in allowed:
        raise ValueError(f"Unsupported logical operation: {operation!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=(op in {"COPY", "FILL", "CLEAR", "INVERT"}),
    )
    params: Dict[str, Any] = {"Operation": op, "BypassMasking": 1}
    if op in {"COPY", "UNION", "INTERSECT", "SUBTRACT"}:
        modifier_id = _find_segment_id(segmentation_node, segment_id=modifier_segment_id, segment_name=modifier_segment_name)
        if not modifier_id:
            raise ValueError(f"Modifier segment not found: id={modifier_segment_id!r} name={modifier_segment_name!r}")
        params["ModifierSegmentID"] = modifier_id

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Logical operators", params)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "operation": op, "modifier_segment_id": params.get("ModifierSegmentID")}


def _segment_edit_sphere(
    *,
    ras_center: Sequence[float],
    radius_mm: float,
    action: str = "add",
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    if len(ras_center) != 3:
        raise ValueError("ras_center must be length-3 [x,y,z] in RAS mm")
    op = str(action or "add").strip().lower()
    if op not in {"add", "erase"}:
        raise ValueError("action must be 'add' or 'erase'")

    import vtk

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(float(ras_center[0]), float(ras_center[1]), float(ras_center[2]))
    sphere.SetRadius(float(radius_mm))
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    sphere.Update()

    temp_name = f"__tmp_sphere_{os.getpid()}"
    temp_segment_id = segmentation_node.AddSegmentFromClosedSurfaceRepresentation(sphere.GetOutput(), temp_name, [1.0, 0.0, 0.0])
    if not temp_segment_id:
        raise RuntimeError("Failed to create temporary sphere segment")

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(
            widget,
            "Logical operators",
            {
                "Operation": "UNION" if op == "add" else "SUBTRACT",
                "ModifierSegmentID": str(temp_segment_id),
                "BypassMasking": 1,
            },
        )
    finally:
        try:
            segmentation_node.GetSegmentation().RemoveSegment(str(temp_segment_id))
        except Exception:
            pass
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    _show_segment_overlay_in_views(segmentation_node, target_segment_id, source_volume_node=source_volume_node)
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "action": op,
        "ras_center": [float(x) for x in ras_center],
        "radius_mm": float(radius_mm),
    }


def _segment_statistics(
    *,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    target_id = _find_segment_id(segmentation_node, segment_id=segment_id, segment_name=segment_name)
    segment_ids = [target_id] if target_id else _iter_segment_ids(segmentation_node)
    stats_list = [_segment_basic_stats(segmentation_node, sid, source_volume_node) for sid in segment_ids]
    result: Dict[str, Any] = {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "source_volume_id": source_volume_node.GetID(),
        "source_volume_name": source_volume_node.GetName(),
        "segments": stats_list,
        "n_segments": len(stats_list),
    }
    if target_id and stats_list:
        result.update(stats_list[0])
    return result


def _center_on_segment(
    *,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    target_segment_id, _ = _ensure_segment(segmentation_node, segment_id=segment_id, segment_name=segment_name, create_if_missing=False)
    bounds_ras, center_ras, center_source = _segment_bounds_and_center_ras(segmentation_node, target_segment_id)
    view_results: Dict[str, Any] = {}
    for view, orientation in (("axial", "axial"), ("sagittal", "sagittal"), ("coronal", "coronal")):
        view_payload: Dict[str, Any] = {}
        try:
            view_payload["orientation"] = _set_slice_orientation(view=view, orientation=orientation)
        except Exception as exc:
            view_payload["orientation_error"] = str(exc)
        try:
            view_payload["fit"] = _fit_slice(view=view)
        except Exception as exc:
            view_payload["fit_error"] = str(exc)
        view_results[view] = view_payload
    _jump_slices_to_location(float(center_ras[0]), float(center_ras[1]), float(center_ras[2]), True)
    slicer.app.processEvents()
    for view in ("axial", "sagittal", "coronal"):
        try:
            sn = _slice_node(view)
            final_offset = float(sn.GetSliceOffset())
            final_state = {"slice_offset": final_offset}
            final_state.update(_canonical_slice_offset_info(sn, slice_offset=final_offset))
            view_results.setdefault(view, {})["final"] = final_state
        except Exception as exc:
            view_results.setdefault(view, {})["final_error"] = str(exc)
    segment = segmentation_node.GetSegmentation().GetSegment(target_segment_id)
    return {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segment_id": str(target_segment_id),
        "segment_name": segment.GetName() if segment is not None else None,
        "center_ras": [float(x) for x in center_ras],
        "bounds_ras": bounds_ras,
        "center_source": center_source,
        "views": view_results,
    }


def _resample_volume_isotropic(
    *,
    output_spacing_mm: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    output_name: Optional[str] = None,
    interpolation_type: str = "linear",
    make_active: bool = True,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    spacing = float(output_spacing_mm)
    if spacing <= 0:
        raise ValueError("output_spacing_mm must be > 0")

    interpolation_map = {
        "linear": "linear",
        "nearest": "nn",
        "nearestneighbor": "nn",
        "nn": "nn",
        "bspline": "bspline",
        "cubic": "bspline",
        "windowedsinc": "ws",
        "ws": "ws",
    }
    interp_key = str(interpolation_type or "linear").strip().lower()
    cli_interp = interpolation_map.get(interp_key)
    if cli_interp is None:
        raise ValueError(f"Unsupported interpolation_type: {interpolation_type!r}")

    out_name = str(output_name or f"{source_volume_node.GetName()}_iso_{spacing:g}mm")
    out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name)
    params = {
        "InputVolume": source_volume_node.GetID(),
        "OutputVolume": out_node.GetID(),
        "outputPixelSpacing": f"{spacing},{spacing},{spacing}",
        "interpolationType": cli_interp,
    }
    cli_module = getattr(slicer.modules, "resamplescalarvolume", None)
    if cli_module is None:
        raise RuntimeError("Resample Scalar Volume module is unavailable in this Slicer build.")
    cli_node = slicer.cli.runSync(cli_module, None, params)
    status_string = cli_node.GetStatusString() if cli_node is not None and hasattr(cli_node, "GetStatusString") else None
    if cli_node is not None and hasattr(cli_node, "GetStatus") and hasattr(cli_node, "ErrorsMask"):
        if int(cli_node.GetStatus()) & int(cli_node.ErrorsMask):
            err_text = cli_node.GetErrorText() if hasattr(cli_node, "GetErrorText") else status_string
            try:
                slicer.mrmlScene.RemoveNode(out_node)
            except Exception:
                pass
            raise RuntimeError(f"Resample Scalar Volume failed: {err_text}")
    if make_active:
        _set_active_volume(out_node)
    out_spacing = [float(x) for x in out_node.GetSpacing()] if hasattr(out_node, "GetSpacing") else None
    slicer.app.processEvents()
    return {
        "source_volume_id": source_volume_node.GetID(),
        "source_volume_name": source_volume_node.GetName(),
        "output_volume_id": out_node.GetID(),
        "output_volume_name": out_node.GetName(),
        "output_spacing": out_spacing,
        "interpolation_type": interp_key,
        "status": status_string,
        "made_active": bool(make_active),
    }


def _export_segmentation_dicom(segmentation_id: str, reference_volume_id: str, output_folder: str) -> Dict[str, Any]:
    output_folder = os.path.expanduser(output_folder)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    segmentation_node = slicer.mrmlScene.GetNodeByID(segmentation_id)
    if segmentation_node is None:
        raise ValueError(f"Segmentation node not found: {segmentation_id}")
    if not segmentation_node.IsA("vtkMRMLSegmentationNode"):
        raise ValueError(f"Node is not a segmentation: {segmentation_node.GetClassName()}")

    reference_volume_node = slicer.mrmlScene.GetNodeByID(reference_volume_id)
    if reference_volume_node is None:
        raise ValueError(f"Reference volume node not found: {reference_volume_id}")

    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    reference_item = sh_node.GetItemByDataNode(reference_volume_node)
    if reference_item == 0:
        raise RuntimeError("Reference volume is not in subject hierarchy (unexpected).")

    study_item = sh_node.GetItemParent(reference_item)
    if study_item == 0:
        study_item = sh_node.GetSceneItemID()

    segmentation_item = sh_node.GetItemByDataNode(segmentation_node)
    if segmentation_item == 0:
        raise RuntimeError("Segmentation is not in subject hierarchy (unexpected).")

    sh_node.SetItemParent(segmentation_item, study_item)

    plugin_module = None
    try:
        import DICOMSegmentationPlugin as plugin_module  # type: ignore
    except Exception:
        plugin_module = None

    if plugin_module is None:
        dicom_plugins = getattr(slicer.modules, "dicomPlugins", None)
        if isinstance(dicom_plugins, dict):
            plugin_module = dicom_plugins.get("DICOMSegmentationPlugin")
        elif hasattr(dicom_plugins, "get"):
            try:
                plugin_module = dicom_plugins.get("DICOMSegmentationPlugin")
            except Exception:
                plugin_module = None
        elif dicom_plugins is not None and hasattr(dicom_plugins, "DICOMSegmentationPlugin"):
            plugin_module = getattr(dicom_plugins, "DICOMSegmentationPlugin")

    if plugin_module is None or not hasattr(plugin_module, "DICOMSegmentationPluginClass"):
        raise RuntimeError(
            "DICOMSegmentationPlugin is unavailable. Install Quantitative Reporting and restart Slicer before exporting DICOM SEG."
        )

    exporter = plugin_module.DICOMSegmentationPluginClass()
    exportables = exporter.examineForExport(segmentation_item)
    if not exportables:
        raise RuntimeError("No exportables found for segmentation (is it empty?)")

    for exp in exportables:
        exp.directory = output_folder

    exporter.export(exportables)

    written = []
    for fn in os.listdir(output_folder):
        if fn.lower().endswith(".dcm"):
            written.append(os.path.join(output_folder, fn))
    written.sort()

    return {"output_folder": output_folder, "files": written, "n_files": len(written)}



SEGMENTATION_BRIDGE_HANDLERS: Dict[str, Any] = {
    "list_segmentations": lambda args: _list_segmentations(),
    "list_segments": lambda args: _list_segments(
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
    ),
    "show_segmentation": lambda args: _show_segmentation(
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
        visible=bool(args.get("visible", True)),
        fill=bool(args.get("fill", True)),
        outline=bool(args.get("outline", True)),
        opacity=float(args.get("opacity", 0.6)),
        color=args.get("color"),
    ),
    "hide_segmentation": lambda args: _hide_segmentation_overlay(
        target=str(args.get("target", "all")),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "image_coords_to_ras": lambda args: _image_coords_to_ras(
        view=args["view"],
        point_norm=args.get("point_norm"),
        bbox_norm=args.get("bbox_norm"),
        orientation=args.get("orientation"),
        offset=(float(args["offset"]) if args.get("offset") is not None else None),
        scroll_to=(float(args["scroll_to"]) if args.get("scroll_to") is not None else None),
    ),
    "segment_from_threshold": lambda args: _segment_from_threshold(
        minimum_threshold=float(args["minimum_threshold"]),
        maximum_threshold=float(args["maximum_threshold"]),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_local_threshold": lambda args: _segment_local_threshold(
        minimum_threshold=float(args["minimum_threshold"]),
        maximum_threshold=float(args["maximum_threshold"]),
        minimum_diameter_mm=float(args.get("minimum_diameter_mm", 3.0)),
        feature_size_mm=float(args.get("feature_size_mm", 3.0)),
        segmentation_algorithm=str(args.get("segmentation_algorithm", "GrowCut")),
        seed_view=args.get("seed_view"),
        seed_x_norm=(float(args["seed_x_norm"]) if args.get("seed_x_norm") is not None else None),
        seed_y_norm=(float(args["seed_y_norm"]) if args.get("seed_y_norm") is not None else None),
        ras_seed=args.get("ras_seed"),
        seed_segment_id=args.get("seed_segment_id"),
        seed_segment_name=args.get("seed_segment_name"),
        seed_orientation=args.get("seed_orientation"),
        seed_offset_mm=args.get("seed_offset_mm"),
        seed_scroll_to=args.get("seed_scroll_to"),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_edit_sphere": lambda args: _segment_edit_sphere(
        ras_center=args.get("ras_center"),
        radius_mm=float(args["radius_mm"]),
        action=str(args.get("action", "add")),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_margin": lambda args: _segment_margin(
        margin_size_mm=float(args["margin_size_mm"]),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_smoothing": lambda args: _segment_smoothing(
        smoothing_method=str(args.get("smoothing_method", "MEDIAN")),
        kernel_size_mm=float(args.get("kernel_size_mm", 3.0)),
        gaussian_std_mm=(float(args["gaussian_std_mm"]) if args.get("gaussian_std_mm") is not None else None),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_islands": lambda args: _segment_islands(
        operation=str(args.get("operation", "KEEP_LARGEST_ISLAND")),
        minimum_size=int(args.get("minimum_size", 1000)),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_logical": lambda args: _segment_logical(
        operation=str(args["operation"]),
        modifier_segment_id=args.get("modifier_segment_id"),
        modifier_segment_name=args.get("modifier_segment_name"),
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "segment_statistics": lambda args: _segment_statistics(
        source_volume_id=args.get("source_volume_id"),
        source_volume_name=args.get("source_volume_name"),
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "center_on_segment": lambda args: _center_on_segment(
        segmentation_id=args.get("segmentation_id"),
        segmentation_name=args.get("segmentation_name"),
        segment_id=args.get("segment_id"),
        segment_name=args.get("segment_name"),
    ),
    "export_segmentation_dicom": lambda args: _export_segmentation_dicom(
        segmentation_id=str(args["segmentation_id"]),
        reference_volume_id=str(args["reference_volume_id"]),
        output_folder=str(args["output_folder"]),
    ),
}
