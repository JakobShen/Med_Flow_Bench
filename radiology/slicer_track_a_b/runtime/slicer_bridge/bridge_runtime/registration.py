"""Registration and transform operations for the Slicer bridge."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import vtk

from .common import *  # noqa: F401,F403
from .core import _get_active_volume_node, _get_node_by_id_or_name, _set_active_volume


# -----------------------------------------------------------------------------
# Generic node helpers
# -----------------------------------------------------------------------------


_NULL_LIKE_REFS = {"none", "null", "nil", "(none)", "<none>"}


def _normalize_optional_ref(ref: Optional[str]) -> Optional[str]:
    if ref is None:
        return None
    text = str(ref).strip()
    if not text:
        return None
    if text.lower() in _NULL_LIKE_REFS:
        return None
    return text


def _normalize_string_list(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in str(value).split(",")]
        return [part.strip("\"'") for part in parts if _normalize_optional_ref(part) is not None]
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            text = _normalize_optional_ref(item)
            if text is not None:
                items.append(text.strip("\"'"))
        return items
    text = _normalize_optional_ref(value)
    return [text] if text is not None else []


_BRAINS_PHASE_ALIASES = {
    "rigid": "Rigid",
    "userigid": "Rigid",
    "linear": "Rigid",
    "scaleversor3d": "ScaleVersor3D",
    "scale_versor_3d": "ScaleVersor3D",
    "scale-skew-versor-3d": "ScaleSkewVersor3D",
    "scaleskewversor3d": "ScaleSkewVersor3D",
    "scale_skew_versor_3d": "ScaleSkewVersor3D",
    "affine": "Affine",
    "useaffine": "Affine",
    "bspline": "BSpline",
    "b-spline": "BSpline",
    "b_spline": "BSpline",
    "syn": "SyN",
    "composite": "Composite",
}


def _canonicalize_brains_phase(value: str) -> str:
    key = str(value).strip().strip("\"'")
    lookup = key.replace(" ", "").replace("-", "").replace("_", "").lower()
    return _BRAINS_PHASE_ALIASES.get(lookup, _BRAINS_PHASE_ALIASES.get(key.lower(), key))


def _looks_like_path(value: Optional[Any]) -> bool:
    text = _normalize_optional_ref(value)
    if text is None:
        return False
    return os.path.isabs(text) or os.sep in text or (os.altsep is not None and os.altsep in text) or "." in os.path.basename(text)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_phase_parameter_length(
    values: Optional[Sequence[Any]],
    *,
    target_len: int,
    cast_type=float,
) -> Tuple[Optional[List[Any]], bool]:
    normalized = _coerce_number_list(values, cast_type=cast_type)
    if normalized is None:
        return None, False
    if target_len <= 0 or len(normalized) in {0, 1, target_len}:
        return normalized, False
    if len(normalized) > target_len:
        return normalized[:target_len], True
    padded = list(normalized)
    padded.extend([normalized[-1]] * (target_len - len(normalized)))
    return padded, True


def _effective_brains_phase_names(args: Dict[str, Any]) -> Tuple[List[str], bool, bool]:
    explicit_phase_pairs = [
        ("useRigid", "Rigid"),
        ("useScaleVersor3D", "ScaleVersor3D"),
        ("useScaleSkewVersor3D", "ScaleSkewVersor3D"),
        ("useAffine", "Affine"),
        ("useBSpline", "BSpline"),
        ("useSyN", "SyN"),
        ("useComposite", "Composite"),
    ]
    explicit_phases = [phase_name for key, phase_name in explicit_phase_pairs if _coerce_bool(args.get(key))]
    if explicit_phases:
        return explicit_phases, True, False

    transform_type_values = [_canonicalize_brains_phase(value) for value in _normalize_string_list(args.get("transformType"))]
    if transform_type_values:
        return transform_type_values, False, False

    return ["Rigid"], False, True


def _iter_scene_nodes() -> List[Any]:
    scene = slicer.mrmlScene
    return [scene.GetNthNode(i) for i in range(int(scene.GetNumberOfNodes()))]


def _list_visible_transform_nodes() -> List[Any]:
    nodes: List[Any] = []
    for node in _iter_scene_nodes():
        try:
            if node is None or not node.IsA("vtkMRMLTransformNode"):
                continue
            if hasattr(node, "GetHideFromEditors") and node.GetHideFromEditors():
                continue
            nodes.append(node)
        except Exception:
            continue
    return nodes


def _find_node_by_ref(ref: Optional[str], *, expected_classes: Optional[Sequence[str]] = None):
    text = _normalize_optional_ref(ref)
    if text is None:
        return None
    try:
        node = slicer.mrmlScene.GetNodeByID(text)
    except Exception:
        node = None
    if node is None:
        try:
            node = slicer.util.getNode(text)
        except Exception:
            node = None
    if node is None:
        return None
    if expected_classes and not any(node.IsA(class_name) for class_name in expected_classes):
        return None
    return node


def _resolve_transform_node(*, transform: Optional[str] = None, transform_id: Optional[str] = None, transform_name: Optional[str] = None):
    for ref in (transform_id, transform, transform_name):
        node = _find_node_by_ref(ref, expected_classes=["vtkMRMLTransformNode"])
        if node is not None:
            return node
    nodes = _list_visible_transform_nodes()
    if len(nodes) == 1:
        return nodes[0]
    if not nodes:
        raise ValueError("No transform nodes are loaded.")
    raise ValueError("Multiple transform nodes are loaded; specify transform or transform_id or transform_name.")


def _resolve_volume_node_ref(*, ref: Optional[str] = None, fallback_active: bool = False):
    node = _find_node_by_ref(
        ref,
        expected_classes=[
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLVectorVolumeNode",
            "vtkMRMLLabelMapVolumeNode",
        ],
    )
    if node is not None:
        return node
    if fallback_active:
        node = _get_active_volume_node()
        if node is not None:
            return node
    return None


def _resolve_scene_node_list(
    *,
    node_refs: Optional[Sequence[str]] = None,
    node_ids: Optional[Sequence[str]] = None,
    node_names: Optional[Sequence[str]] = None,
    node_ref: Optional[str] = None,
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
) -> List[Any]:
    seen: set[str] = set()
    nodes: List[Any] = []
    candidates: List[str] = []
    for values in (node_refs or [], node_ids or [], node_names or []):
        for value in values:
            text = _normalize_optional_ref(value)
            if text is not None:
                candidates.append(text)
    for single in (node_ref, node_id, node_name):
        text = _normalize_optional_ref(single)
        if text is not None:
            candidates.append(text)
    for ref in candidates:
        node = _find_node_by_ref(ref)
        if node is None:
            raise ValueError(f"Scene node not found: {ref}")
        node_id_value = str(node.GetID())
        if node_id_value in seen:
            continue
        seen.add(node_id_value)
        nodes.append(node)
    return nodes


# -----------------------------------------------------------------------------
# Matrix / transform helpers
# -----------------------------------------------------------------------------


def _identity_matrix() -> vtk.vtkMatrix4x4:
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    return matrix


def _copy_matrix(src: vtk.vtkMatrix4x4) -> vtk.vtkMatrix4x4:
    out = vtk.vtkMatrix4x4()
    out.DeepCopy(src)
    return out


def _matrix_to_rows(matrix: Optional[vtk.vtkMatrix4x4]) -> Optional[List[List[float]]]:
    if matrix is None:
        return None
    return [[float(matrix.GetElement(r, c)) for c in range(4)] for r in range(4)]


def _matrix_from_rows(rows: Sequence[Sequence[float]]) -> vtk.vtkMatrix4x4:
    if len(rows) != 4:
        raise ValueError("matrix_4x4 must have four rows")
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for r in range(4):
        row = list(rows[r])
        if len(row) != 4:
            raise ValueError("matrix_4x4 must be 4x4")
        for c in range(4):
            matrix.SetElement(r, c, float(row[c]))
    return matrix


def _matrix_multiply(a: vtk.vtkMatrix4x4, b: vtk.vtkMatrix4x4) -> vtk.vtkMatrix4x4:
    out = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(a, b, out)
    return out


def _matrix_point_multiply(matrix: vtk.vtkMatrix4x4, point_xyz: Sequence[float]) -> List[float]:
    point = [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0]
    out = [0.0, 0.0, 0.0, 0.0]
    matrix.MultiplyPoint(point, out)
    w = out[3] if abs(float(out[3])) > 1e-12 else 1.0
    return [float(out[0] / w), float(out[1] / w), float(out[2] / w)]


def _matrix_inverse(matrix: vtk.vtkMatrix4x4) -> vtk.vtkMatrix4x4:
    out = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(matrix, out)
    return out


def _linear_matrix_to_parent(transform_node) -> vtk.vtkMatrix4x4:
    matrix = vtk.vtkMatrix4x4()
    try:
        ok = bool(transform_node.GetMatrixTransformToParent(matrix))
    except Exception:
        ok = False
    if ok:
        return matrix
    transform_to_parent = None
    try:
        transform_to_parent = transform_node.GetTransformToParent()
    except Exception:
        transform_to_parent = None
    if transform_to_parent is None:
        return _identity_matrix()
    raise ValueError("Transform is not a simple linear transform and cannot be edited by matrix.")


def _set_linear_matrix_to_parent(transform_node, matrix: vtk.vtkMatrix4x4) -> None:
    try:
        ok = bool(transform_node.SetMatrixTransformToParent(matrix))
    except Exception:
        ok = False
    if not ok:
        raise ValueError("Transform is not a simple linear transform and cannot be updated by matrix.")
    slicer.app.processEvents()


def _parent_to_world_matrix(transform_node) -> vtk.vtkMatrix4x4:
    parent_node = transform_node.GetParentTransformNode() if hasattr(transform_node, "GetParentTransformNode") else None
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    if parent_node is None:
        return matrix
    ok = bool(slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(parent_node, None, matrix))
    if not ok:
        raise ValueError("Parent transform chain is not linear; this edit is only supported for linear parent transforms.")
    return matrix


def _center_local(transform_node) -> List[float]:
    try:
        values = transform_node.GetCenterOfTransformation()
        return [float(values[0]), float(values[1]), float(values[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def _local_to_world_point(transform_node, point_local: Sequence[float]) -> List[float]:
    gt = vtk.vtkGeneralTransform()
    transform_node.GetTransformToWorld(gt)
    out = [0.0, 0.0, 0.0]
    gt.TransformPoint([float(point_local[0]), float(point_local[1]), float(point_local[2])], out)
    return [float(out[0]), float(out[1]), float(out[2])]


def _world_to_local_point(transform_node, point_world: Sequence[float]) -> List[float]:
    gt = vtk.vtkGeneralTransform()
    transform_node.GetTransformFromWorld(gt)
    out = [0.0, 0.0, 0.0]
    gt.TransformPoint([float(point_world[0]), float(point_world[1]), float(point_world[2])], out)
    return [float(out[0]), float(out[1]), float(out[2])]


def _make_translation_matrix(translation_xyz: Sequence[float]) -> vtk.vtkMatrix4x4:
    tx, ty, tz = [float(v) for v in translation_xyz]
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0, 3, tx)
    matrix.SetElement(1, 3, ty)
    matrix.SetElement(2, 3, tz)
    return matrix


def _make_centered_world_delta(
    *,
    center_world: Sequence[float],
    rotation_degrees: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    shear: Optional[Sequence[float]] = None,
    translation_parent: Optional[Sequence[float]] = None,
) -> vtk.vtkMatrix4x4:
    cx, cy, cz = [float(v) for v in center_world]
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Identity()
    if translation_parent is not None:
        tx, ty, tz = [float(v) for v in translation_parent]
        transform.Translate(tx, ty, tz)
    transform.Translate(cx, cy, cz)
    if rotation_degrees is not None:
        rx, ry, rz = [float(v) for v in rotation_degrees]
        transform.RotateX(rx)
        transform.RotateY(ry)
        transform.RotateZ(rz)
    if scale is not None:
        sx, sy, sz = [float(v) for v in scale]
        transform.Scale(sx, sy, sz)
    if shear is not None:
        values = [float(v) for v in shear]
        if len(values) != 6:
            raise ValueError("shear must have six values ordered as [xy, xz, yx, yz, zx, zy]")
        shear_matrix = vtk.vtkMatrix4x4()
        shear_matrix.Identity()
        shear_matrix.SetElement(0, 1, values[0])
        shear_matrix.SetElement(0, 2, values[1])
        shear_matrix.SetElement(1, 0, values[2])
        shear_matrix.SetElement(1, 2, values[3])
        shear_matrix.SetElement(2, 0, values[4])
        shear_matrix.SetElement(2, 1, values[5])
        transform.Concatenate(shear_matrix)
    transform.Translate(-cx, -cy, -cz)
    return _copy_matrix(transform.GetMatrix())


def _make_local_translation_matrix(translation_xyz: Sequence[float]) -> vtk.vtkMatrix4x4:
    return _make_translation_matrix(translation_xyz)


def _node_ras_bounds(node) -> List[float]:
    bounds = [0.0] * 6
    if hasattr(node, "GetRASBounds"):
        node.GetRASBounds(bounds)
        return [float(v) for v in bounds]
    if hasattr(node, "GetBounds"):
        node.GetBounds(bounds)
        return [float(v) for v in bounds]
    raise ValueError(f"Node does not provide bounds: {node.GetName()}")


def _bounds_center(bounds: Sequence[float]) -> List[float]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    return [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]


def _union_bounds(bounds_list: Sequence[Sequence[float]]) -> List[float]:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for bounds in bounds_list:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
        mins[0] = min(mins[0], xmin)
        maxs[0] = max(maxs[0], xmax)
        mins[1] = min(mins[1], ymin)
        maxs[1] = max(maxs[1], ymax)
        mins[2] = min(mins[2], zmin)
        maxs[2] = max(maxs[2], zmax)
    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]


def _transform_kind(transform_node) -> str:
    if bool(transform_node.IsComposite()):
        return "composite"
    if bool(transform_node.IsLinear()):
        return "linear"
    try:
        transform = transform_node.GetTransformToParent() or transform_node.GetTransformFromParent()
    except Exception:
        transform = None
    if transform is None:
        return "identity"
    class_name = str(transform.GetClassName())
    if "BSpline" in class_name:
        return "bspline"
    if "Grid" in class_name or "Displacement" in class_name:
        return "grid"
    if "ThinPlate" in class_name:
        return "thin_plate"
    return class_name


def _transform_direction(transform_node) -> Optional[str]:
    try:
        to_parent = transform_node.GetTransformToParent()
    except Exception:
        to_parent = None
    try:
        from_parent = transform_node.GetTransformFromParent()
    except Exception:
        from_parent = None
    if to_parent is not None and from_parent is None:
        return "to_parent"
    if from_parent is not None and to_parent is None:
        return "from_parent"
    return None


def _transformed_nodes_for_transform(transform_node) -> List[Any]:
    transform_id = str(transform_node.GetID())
    nodes: List[Any] = []
    for node in _iter_scene_nodes():
        try:
            if node is None or not hasattr(node, "GetTransformNodeID"):
                continue
            node_transform_id = node.GetTransformNodeID()
            if node_transform_id and str(node_transform_id) == transform_id:
                nodes.append(node)
        except Exception:
            continue
    return nodes


def _node_brief(node) -> Dict[str, Any]:
    return {
        "id": str(node.GetID()),
        "name": str(node.GetName()),
        "class_name": str(node.GetClassName()),
    }


def _transform_summary(transform_node, *, include_transformed_nodes: bool = True) -> Dict[str, Any]:
    matrix_rows = None
    if bool(transform_node.IsLinear()):
        try:
            matrix_rows = _matrix_to_rows(_linear_matrix_to_parent(transform_node))
        except Exception:
            matrix_rows = None
    center_local = _center_local(transform_node)
    try:
        center_world = _local_to_world_point(transform_node, center_local)
    except Exception:
        center_world = None
    parent_node = transform_node.GetParentTransformNode() if hasattr(transform_node, "GetParentTransformNode") else None
    payload: Dict[str, Any] = {
        "transform_id": str(transform_node.GetID()),
        "transform_name": str(transform_node.GetName()),
        "class_name": str(transform_node.GetClassName()),
        "transform_kind": _transform_kind(transform_node),
        "transform_direction": _transform_direction(transform_node),
        "is_linear": bool(transform_node.IsLinear()),
        "is_composite": bool(transform_node.IsComposite()),
        "parent_transform_id": str(parent_node.GetID()) if parent_node is not None else None,
        "parent_transform_name": str(parent_node.GetName()) if parent_node is not None else None,
        "center_local": center_local,
        "center_world": center_world,
    }
    if matrix_rows is not None:
        payload["matrix_4x4"] = matrix_rows
    if include_transformed_nodes:
        transformed_nodes = [_node_brief(node) for node in _transformed_nodes_for_transform(transform_node)]
        payload["transformed_nodes"] = transformed_nodes
        payload["n_transformed_nodes"] = len(transformed_nodes)
    return payload


# -----------------------------------------------------------------------------
# Transforms logic helpers
# -----------------------------------------------------------------------------


def _transforms_logic():
    logic_factory = getattr(getattr(slicer.modules, "transforms", None), "logic", None)
    if not callable(logic_factory):
        raise RuntimeError("Transforms module logic is not available.")
    return logic_factory()


def _create_node_by_class(class_name: str, name: str):
    unique_name = slicer.mrmlScene.GenerateUniqueName(str(name))
    node = slicer.mrmlScene.AddNewNodeByClass(class_name, unique_name)
    if hasattr(node, "CreateDefaultDisplayNodes"):
        try:
            node.CreateDefaultDisplayNodes()
        except Exception:
            pass
    return node


def _resolve_or_create_transform_output(ref_or_name: Optional[str], *, default_name: str):
    ref_or_name = _normalize_optional_ref(ref_or_name)
    node = _find_node_by_ref(ref_or_name, expected_classes=["vtkMRMLTransformNode"]) if ref_or_name else None
    if node is not None:
        return node
    node_name = str(ref_or_name).strip() if ref_or_name else default_name
    return _create_node_by_class("vtkMRMLTransformNode", node_name)


def _resolve_or_create_volume_output(ref_or_name: Optional[str], *, class_name: str, default_name: str):
    ref_or_name = _normalize_optional_ref(ref_or_name)
    node = _find_node_by_ref(
        ref_or_name,
        expected_classes=["vtkMRMLScalarVolumeNode", "vtkMRMLVectorVolumeNode", "vtkMRMLLabelMapVolumeNode"],
    ) if ref_or_name else None
    if node is not None:
        return node
    node_name = str(ref_or_name).strip() if ref_or_name else default_name
    return _create_node_by_class(class_name, node_name)


# -----------------------------------------------------------------------------
# Manual transform handlers
# -----------------------------------------------------------------------------


def _list_transforms(_: Dict[str, Any]) -> Dict[str, Any]:
    nodes = _list_visible_transform_nodes()
    payload = {
        "n_transforms": len(nodes),
        "transforms": [_transform_summary(node) for node in nodes],
    }
    if len(nodes) == 1:
        payload.update(_transform_summary(nodes[0]))
    return payload


def _create_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name") or "Transform")
    kind = str(args.get("kind") or "general").strip().lower()
    class_name = "vtkMRMLTransformNode"
    if kind == "linear":
        # vtkMRMLLinearTransformNode still exists in Slicer for compatibility.
        class_name = "vtkMRMLLinearTransformNode"
    elif kind != "general":
        raise ValueError("create_transform kind must be 'general' or 'linear'")
    node = _create_node_by_class(class_name, name)
    if kind == "linear":
        _set_linear_matrix_to_parent(node, _identity_matrix())
    return {**_transform_summary(node), "made_active": bool(args.get("make_active", True))}


def _get_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    return _transform_summary(node)


def _edit_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    matrix_rows = args.get("matrix_4x4")
    translation = args.get("translation")
    rotation = args.get("rotation_degrees")
    scale = args.get("scale")
    shear = args.get("shear")
    reset_to_identity = bool(args.get("reset_to_identity", False))
    translation_reference_frame = str(args.get("translation_reference_frame") or "parent").strip().lower()
    if translation_reference_frame not in {"parent", "local"}:
        raise ValueError("translation_reference_frame must be 'parent' or 'local'")

    has_component_edit = any(value is not None for value in (translation, rotation, scale, shear))
    if not reset_to_identity and matrix_rows is None and not has_component_edit:
        raise ValueError("Provide matrix_4x4 or one of translation, rotation_degrees, scale, shear, or set reset_to_identity=true.")

    current_matrix = _identity_matrix() if reset_to_identity else _linear_matrix_to_parent(transform_node)

    if matrix_rows is not None:
        target_matrix = _matrix_from_rows(matrix_rows)
        _set_linear_matrix_to_parent(transform_node, target_matrix)
        return _transform_summary(transform_node)

    center_local = _center_local(transform_node)
    parent_to_world = _parent_to_world_matrix(transform_node)
    local_to_world = _matrix_multiply(parent_to_world, current_matrix)
    center_world = _matrix_point_multiply(local_to_world, center_local)

    world_delta = _make_centered_world_delta(
        center_world=center_world,
        rotation_degrees=rotation,
        scale=scale,
        shear=shear,
        translation_parent=translation if translation_reference_frame == "parent" else None,
    )
    new_local_to_world = _matrix_multiply(world_delta, local_to_world)
    new_local_to_parent = _matrix_multiply(_matrix_inverse(parent_to_world), new_local_to_world)

    if translation is not None and translation_reference_frame == "local":
        local_translation = _make_local_translation_matrix(translation)
        new_local_to_parent = _matrix_multiply(new_local_to_parent, local_translation)

    _set_linear_matrix_to_parent(transform_node, new_local_to_parent)
    return _transform_summary(transform_node)


def _set_transform_center(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    mode = str(args.get("mode") or "world_ras").strip().lower()
    if mode == "world_ras":
        ras = args.get("ras")
        if ras is None:
            raise ValueError("ras is required when mode=world_ras")
        local_point = _world_to_local_point(transform_node, ras)
    elif mode == "local":
        local = args.get("local")
        if local is None:
            raise ValueError("local is required when mode=local")
        local_point = [float(v) for v in local]
    elif mode == "origin":
        local_point = [0.0, 0.0, 0.0]
    elif mode == "transformed_nodes":
        nodes = _transformed_nodes_for_transform(transform_node)
        if not nodes:
            raise ValueError("No transformed nodes are associated with this transform.")
        world_center = _bounds_center(_union_bounds([_node_ras_bounds(node) for node in nodes]))
        local_point = _world_to_local_point(transform_node, world_center)
    elif mode == "node_bounds_center":
        node = None
        for ref in (args.get("node_id"), args.get("node_ref"), args.get("node_name")):
            node = _find_node_by_ref(ref)
            if node is not None:
                break
        if node is None:
            raise ValueError("node_ref or node_id or node_name is required when mode=node_bounds_center")
        world_center = _bounds_center(_node_ras_bounds(node))
        local_point = _world_to_local_point(transform_node, world_center)
    else:
        raise ValueError("Unsupported mode for set_transform_center")

    transform_node.SetCenterOfTransformation(local_point)
    slicer.app.processEvents()
    return _transform_summary(transform_node)


def _apply_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    nodes = _resolve_scene_node_list(
        node_refs=args.get("node_refs"),
        node_ids=args.get("node_ids"),
        node_names=args.get("node_names"),
        node_ref=args.get("node_ref"),
        node_id=args.get("node_id"),
        node_name=args.get("node_name"),
    )
    if not nodes:
        active_volume = _get_active_volume_node()
        if active_volume is not None:
            nodes = [active_volume]
    if not nodes:
        raise ValueError("No scene nodes were provided to apply_transform.")

    applied: List[Dict[str, Any]] = []
    for node in nodes:
        if not hasattr(node, "SetAndObserveTransformNodeID"):
            raise ValueError(f"Node does not support transforms: {node.GetName()}")
        node.SetAndObserveTransformNodeID(transform_node.GetID())
        applied.append(_node_brief(node))
    slicer.app.processEvents()
    return {**_transform_summary(transform_node), "applied_nodes": applied, "n_applied_nodes": len(applied)}


def _remove_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    nodes = _resolve_scene_node_list(
        node_refs=args.get("node_refs"),
        node_ids=args.get("node_ids"),
        node_names=args.get("node_names"),
        node_ref=args.get("node_ref"),
        node_id=args.get("node_id"),
        node_name=args.get("node_name"),
    )
    if not nodes:
        nodes = _transformed_nodes_for_transform(transform_node)
    removed: List[Dict[str, Any]] = []
    transform_id = str(transform_node.GetID())
    for node in nodes:
        if not hasattr(node, "GetTransformNodeID") or not hasattr(node, "SetAndObserveTransformNodeID"):
            continue
        direct_transform_id = node.GetTransformNodeID()
        if direct_transform_id and str(direct_transform_id) == transform_id:
            node.SetAndObserveTransformNodeID(None)
            removed.append(_node_brief(node))
    slicer.app.processEvents()
    return {**_transform_summary(transform_node), "removed_nodes": removed, "n_removed_nodes": len(removed)}


def _harden_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    nodes = _resolve_scene_node_list(
        node_refs=args.get("node_refs"),
        node_ids=args.get("node_ids"),
        node_names=args.get("node_names"),
        node_ref=args.get("node_ref"),
        node_id=args.get("node_id"),
        node_name=args.get("node_name"),
    )
    if not nodes:
        nodes = _transformed_nodes_for_transform(transform_node)
    hardened: List[Dict[str, Any]] = []
    for node in nodes:
        if not hasattr(node, "HardenTransform"):
            continue
        node.HardenTransform()
        hardened.append(_node_brief(node))
    slicer.app.processEvents()
    return {**_transform_summary(transform_node), "hardened_nodes": hardened, "n_hardened_nodes": len(hardened)}


def _invert_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    transform_node.Inverse()
    if hasattr(transform_node, "InverseName"):
        try:
            transform_node.InverseName()
        except Exception:
            pass
    slicer.app.processEvents()
    return _transform_summary(transform_node)


def _compose_transforms(args: Dict[str, Any]) -> Dict[str, Any]:
    child = _resolve_transform_node(
        transform=args.get("child_transform"),
        transform_id=args.get("child_transform_id"),
        transform_name=args.get("child_transform_name"),
    )
    parent = _resolve_transform_node(
        transform=args.get("parent_transform"),
        transform_id=args.get("parent_transform_id"),
        transform_name=args.get("parent_transform_name"),
    )
    if child.GetID() == parent.GetID():
        raise ValueError("Child and parent transforms must be different nodes.")
    if hasattr(child, "IsTransformNodeMyChild") and bool(child.IsTransformNodeMyChild(parent)):
        raise ValueError("Cannot create a cyclic transform hierarchy.")
    child.SetAndObserveTransformNodeID(parent.GetID())
    slicer.app.processEvents()
    payload = _transform_summary(child)
    payload["parent_transform_id"] = str(parent.GetID())
    payload["parent_transform_name"] = str(parent.GetName())
    return payload


def _split_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    before_ids = {str(node.GetID()) for node in _list_visible_transform_nodes()}
    ok = bool(transform_node.Split())
    if not ok:
        raise ValueError("Split failed. The transform may not be composite.")
    slicer.app.processEvents()
    after_nodes = _list_visible_transform_nodes()
    new_nodes = [node for node in after_nodes if str(node.GetID()) not in before_ids]
    return {
        **_transform_summary(transform_node),
        "component_transforms": [_transform_summary(node) for node in new_nodes],
        "n_component_transforms": len(new_nodes),
    }


def _sample_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    reference = args.get("referenceVolume") or args.get("reference") or args.get("reference_volume_id") or args.get("reference_volume_name")
    reference_volume = _resolve_volume_node_ref(ref=reference, fallback_active=True)
    if reference_volume is None:
        raise ValueError("A reference volume is required for sample_transform.")

    output_kind = str(args.get("output_kind") or "grid_transform").strip().lower()
    output_name = str(args.get("output_name") or "") or None
    make_active = bool(args.get("make_active", False))
    logic = _transforms_logic()

    if output_kind == "grid_transform":
        output_transform = _resolve_or_create_transform_output(output_name, default_name=f"{transform_node.GetName()} grid transform")
        sampled_transform = logic.ConvertToGridTransform(transform_node, reference_volume, output_transform)
        if sampled_transform is None:
            raise RuntimeError("ConvertToGridTransform failed.")
        return {
            **_transform_summary(transform_node),
            "sampled_transform_id": str(sampled_transform.GetID()),
            "sampled_transform_name": str(sampled_transform.GetName()),
            "reference_volume_id": str(reference_volume.GetID()),
            "reference_volume_name": str(reference_volume.GetName()),
            "made_active": make_active,
            **({"transform_id": str(sampled_transform.GetID()), "transform_name": str(sampled_transform.GetName())} if make_active else {}),
        }

    if output_kind not in {"vector_displacement_volume", "scalar_displacement_magnitude"}:
        raise ValueError("output_kind must be one of grid_transform, vector_displacement_volume, scalar_displacement_magnitude")

    class_name = "vtkMRMLVectorVolumeNode" if output_kind == "vector_displacement_volume" else "vtkMRMLScalarVolumeNode"
    default_name = (
        f"{transform_node.GetName()} displacement vectors"
        if output_kind == "vector_displacement_volume"
        else f"{transform_node.GetName()} displacement magnitude"
    )
    output_volume = _resolve_or_create_volume_output(output_name, class_name=class_name, default_name=default_name)
    sampled_volume = logic.CreateDisplacementVolumeFromTransform(
        transform_node,
        reference_volume,
        output_kind == "scalar_displacement_magnitude",
        output_volume,
    )
    if sampled_volume is None:
        raise RuntimeError("CreateDisplacementVolumeFromTransform failed.")
    if make_active:
        _set_active_volume(sampled_volume)
    return {
        **_transform_summary(transform_node),
        "sampled_volume_id": str(sampled_volume.GetID()),
        "sampled_volume_name": str(sampled_volume.GetName()),
        "reference_volume_id": str(reference_volume.GetID()),
        "reference_volume_name": str(reference_volume.GetName()),
        "active_volume_id": str(sampled_volume.GetID()) if make_active else None,
        "active_volume_name": str(sampled_volume.GetName()) if make_active else None,
        "made_active": make_active,
    }


def _save_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    transform_node = _resolve_transform_node(
        transform=args.get("transform"),
        transform_id=args.get("transform_id"),
        transform_name=args.get("transform_name"),
    )
    output_path = os.path.abspath(os.path.expanduser(str(args.get("output_path") or "")))
    if not output_path:
        raise ValueError("output_path is required")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = bool(slicer.util.saveNode(transform_node, output_path))
    if not ok:
        raise RuntimeError(f"Failed to save transform to {output_path}")
    return {**_transform_summary(transform_node), "saved_path": output_path}


def _load_transform(args: Dict[str, Any]) -> Dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(str(args.get("path") or "")))
    if not path or not os.path.exists(path):
        raise ValueError(f"Transform file does not exist: {path}")
    transform_node = None
    try:
        transform_node = _transforms_logic().AddTransform(path, slicer.mrmlScene)
    except Exception:
        transform_node = None
    if transform_node is None:
        try:
            transform_node = slicer.util.loadTransform(path)
        except Exception:
            transform_node = None
    if transform_node is None:
        raise RuntimeError(f"Failed to load transform from {path}")
    return {**_transform_summary(transform_node), "loaded_path": path, "made_active": bool(args.get("make_active", True))}


# -----------------------------------------------------------------------------
# BRAINS CLI helpers
# -----------------------------------------------------------------------------


def _get_cli_module(*module_names: str):
    for name in module_names:
        module = getattr(slicer.modules, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"CLI module not available: {', '.join(module_names)}")


def _run_cli_sync(module, parameters: Dict[str, Any]) -> Dict[str, Any]:
    cli_node = slicer.cli.runSync(module, None, parameters)
    try:
        slicer.app.processEvents()
        status_string = str(cli_node.GetStatusString()) if hasattr(cli_node, "GetStatusString") else None
        error_text = None
        if hasattr(cli_node, "GetErrorText"):
            try:
                error_text = str(cli_node.GetErrorText() or "")
            except Exception:
                error_text = None
        output_text = None
        if hasattr(cli_node, "GetOutputText"):
            try:
                output_text = str(cli_node.GetOutputText() or "")
            except Exception:
                output_text = None
        failed = False
        if status_string:
            lowered = status_string.lower()
            failed = ("cancel" in lowered) or ("error" in lowered) or ("fail" in lowered)
            if status_string == "Completed":
                failed = False
        if failed:
            detail = error_text or output_text or status_string or "CLI execution failed"
            raise RuntimeError(detail)
        return {
            "cli_status": status_string,
            "cli_output_text": output_text,
            "cli_error_text": error_text,
        }
    finally:
        try:
            slicer.mrmlScene.RemoveNode(cli_node)
        except Exception:
            pass


def _resolve_volume_arg(*, preferred: Optional[str] = None, canonical: Optional[str] = None, required: bool = False, active_fallback: bool = False):
    node = _resolve_volume_node_ref(ref=preferred or canonical, fallback_active=active_fallback)
    if node is None and required:
        raise ValueError("Required volume node was not found.")
    return node


def _resolve_transform_input_arg(ref: Optional[str]):
    text = _normalize_optional_ref(ref)
    if text is None:
        return None
    node = _find_node_by_ref(text, expected_classes=["vtkMRMLTransformNode"])
    if node is not None:
        return node
    path = os.path.abspath(os.path.expanduser(str(text)))
    if os.path.exists(path):
        return path
    raise ValueError(f"Transform not found: {text}")


def _resolve_mask_output_node(ref_or_name: Optional[str], *, default_name: str):
    ref_or_name = _normalize_optional_ref(ref_or_name)
    node = _find_node_by_ref(ref_or_name, expected_classes=["vtkMRMLLabelMapVolumeNode"]) if ref_or_name else None
    if node is not None:
        return node
    node_name = str(ref_or_name).strip() if ref_or_name else default_name
    return _create_node_by_class("vtkMRMLLabelMapVolumeNode", node_name)


def _volume_output_class_name_like(node) -> str:
    if node is not None and node.IsA("vtkMRMLVectorVolumeNode"):
        return "vtkMRMLVectorVolumeNode"
    if node is not None and node.IsA("vtkMRMLLabelMapVolumeNode"):
        return "vtkMRMLLabelMapVolumeNode"
    return "vtkMRMLScalarVolumeNode"


def _maybe_save_transform_to_temp_path(transform_or_path, *, suffix: str = ".h5", temp_dir: str) -> Any:
    if transform_or_path is None:
        return None
    if isinstance(transform_or_path, str):
        return transform_or_path
    path = os.path.join(temp_dir, f"transform_input{suffix}")
    ok = bool(slicer.util.saveNode(transform_or_path, path))
    if not ok:
        raise RuntimeError(f"Failed to save temporary transform to {path}")
    return path


def _coerce_number_list(value: Optional[Sequence[Any]], *, cast_type=float) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raw_items = [item.strip() for item in str(value).split(",")]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]

    items: List[Any] = []
    for item in raw_items:
        if _normalize_optional_ref(item) is None:
            continue
        items.append(cast_type(item))
    return items


def _build_brains_fit_parameters(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fixed_volume = _resolve_volume_arg(preferred=args.get("fixed"), canonical=args.get("fixedVolume"), required=True)
    moving_volume = _resolve_volume_arg(preferred=args.get("moving"), canonical=args.get("movingVolume"), required=True)
    fixed_volume_2 = _resolve_volume_arg(preferred=None, canonical=args.get("fixedVolume2"), required=False)
    moving_volume_2 = _resolve_volume_arg(preferred=None, canonical=args.get("movingVolume2"), required=False)
    fixed_mask = _resolve_volume_arg(preferred=None, canonical=args.get("fixedBinaryVolume"), required=False)
    moving_mask = _resolve_volume_arg(preferred=None, canonical=args.get("movingBinaryVolume"), required=False)
    output_fixed_roi = _resolve_mask_output_node(args.get("outputFixedVolumeROI"), default_name="BRAINSFit fixed ROI") if args.get("outputFixedVolumeROI") else None
    output_moving_roi = _resolve_mask_output_node(args.get("outputMovingVolumeROI"), default_name="BRAINSFit moving ROI") if args.get("outputMovingVolumeROI") else None
    initial_transform = _resolve_transform_input_arg(args.get("initialTransform"))
    warnings: List[str] = []

    initialize_transform_mode_arg = args.get("initializeTransformMode")
    initialize_transform_mode = str(initialize_transform_mode_arg or "Off")
    if initial_transform is not None and initialize_transform_mode != "Off":
        initialize_transform_mode = "Off"
        warnings.append("initialTransform was provided, so initializeTransformMode was forced to Off.")

    if str(args.get("maskProcessingMode") or "").upper() == "ROI" and fixed_mask is None and moving_mask is None:
        raise ValueError("maskProcessingMode=ROI requires fixedBinaryVolume and/or movingBinaryVolume.")

    transform_type_values, phase_flags_override, defaulted_to_rigid = _effective_brains_phase_names(args)
    if defaulted_to_rigid:
        warnings.append("No registration phase was specified, so BRAINSFit defaulted to a single Rigid phase.")

    wants_nonlinear = bool(args.get("useBSpline")) or bool(args.get("useSyN")) or bool(args.get("useComposite")) or any(
        token.lower() in {"bspline", "syn", "composite"} for token in transform_type_values
    )

    requested_output = args.get("output_transform") or args.get("output")
    output_transform_path = args.get("outputTransform") if _looks_like_path(args.get("outputTransform")) else None
    stripped_output_transform_path = args.get("strippedOutputTransform") if _looks_like_path(args.get("strippedOutputTransform")) else None
    if requested_output is None and args.get("outputTransform") is not None and output_transform_path is None:
        requested_output = args.get("outputTransform")
    linear_output = None
    bspline_output = None
    if args.get("linearTransform"):
        linear_output = _resolve_or_create_transform_output(args.get("linearTransform"), default_name="BRAINSFit linear transform")
    if args.get("bsplineTransform"):
        bspline_output = _resolve_or_create_transform_output(args.get("bsplineTransform"), default_name="BRAINSFit bspline transform")
    if linear_output is None and bspline_output is None and args.get("outputVolume") is None and output_transform_path is None and stripped_output_transform_path is None:
        if wants_nonlinear:
            bspline_output = _resolve_or_create_transform_output(requested_output, default_name="BRAINSFit Transform")
        else:
            linear_output = _resolve_or_create_transform_output(requested_output, default_name="BRAINSFit Transform")

    output_volume = None
    if args.get("outputVolume"):
        output_volume = _resolve_or_create_volume_output(
            args.get("outputVolume"),
            class_name=_volume_output_class_name_like(moving_volume),
            default_name="BRAINSFit output volume",
        )

    params: Dict[str, Any] = {
        "fixedVolume": fixed_volume,
        "movingVolume": moving_volume,
    }
    if fixed_volume_2 is not None:
        params["fixedVolume2"] = fixed_volume_2
    if moving_volume_2 is not None:
        params["movingVolume2"] = moving_volume_2
    if linear_output is not None:
        params["linearTransform"] = linear_output
    if bspline_output is not None:
        params["bsplineTransform"] = bspline_output
    if output_volume is not None:
        params["outputVolume"] = output_volume
    if fixed_mask is not None:
        params["fixedBinaryVolume"] = fixed_mask
    if moving_mask is not None:
        params["movingBinaryVolume"] = moving_mask
    if output_fixed_roi is not None:
        params["outputFixedVolumeROI"] = output_fixed_roi
    if output_moving_roi is not None:
        params["outputMovingVolumeROI"] = output_moving_roi
    if initial_transform is not None:
        params["initialTransform"] = initial_transform

    passthrough_keys = [
        "outputVolumePixelType",
        "backgroundFillValue",
        "scaleOutputValues",
        "initializeRegistrationByCurrentGenericTransform",
        "useRigid",
        "useScaleVersor3D",
        "useScaleSkewVersor3D",
        "useAffine",
        "useBSpline",
        "useSyN",
        "useComposite",
        "transformType",
        "samplingPercentage",
        "numberOfSamples",
        "metricSamplingStrategy",
        "maskProcessingMode",
        "useROIBSpline",
        "histogramMatch",
        "removeIntensityOutliers",
        "numberOfHistogramBins",
        "numberOfMatchPoints",
        "costMetric",
        "maximumStepLength",
        "relaxationFactor",
        "translationScale",
        "reproportionScale",
        "skewScale",
        "maxBSplineDisplacement",
        "fixedVolumeTimeIndex",
        "movingVolumeTimeIndex",
        "interpolationMode",
        "maskInferiorCutOffFromCenter",
        "ROIAutoDilateSize",
        "ROIAutoClosingSize",
        "writeOutputTransformInFloat",
        "numberOfThreads",
        "debugLevel",
        "costFunctionConvergenceFactor",
        "projectedGradientTolerance",
        "maximumNumberOfEvaluations",
        "maximumNumberOfCorrections",
        "failureExitCode",
        "writeTransformOnFailure",
        "promptUser",
        "logFileReport",
    ]
    for key in passthrough_keys:
        value = args.get(key)
        if value is not None:
            params[key] = value
    if output_transform_path is not None:
        params["outputTransform"] = output_transform_path
    if stripped_output_transform_path is not None:
        params["strippedOutputTransform"] = stripped_output_transform_path

    if initialize_transform_mode_arg is not None or initial_transform is not None:
        params["initializeTransformMode"] = initialize_transform_mode

    if not phase_flags_override:
        phase_flag_by_type = {
            "Rigid": "useRigid",
            "ScaleVersor3D": "useScaleVersor3D",
            "ScaleSkewVersor3D": "useScaleSkewVersor3D",
            "Affine": "useAffine",
            "BSpline": "useBSpline",
            "SyN": "useSyN",
            "Composite": "useComposite",
        }
        if all(phase in phase_flag_by_type for phase in transform_type_values):
            for phase in transform_type_values:
                params[phase_flag_by_type[phase]] = True
            phase_flags_override = True
        else:
            params["transformType"] = list(transform_type_values)

    for key, cast_type in (("splineGridSize", int), ("medianFilterSize", int)):
        value = args.get(key)
        if value is not None:
            params[key] = _coerce_number_list(value, cast_type=cast_type)

    phase_count = max(len(transform_type_values), 1)
    number_of_iterations, iterations_repaired = _normalize_phase_parameter_length(
        args.get("numberOfIterations"), target_len=phase_count, cast_type=int
    )
    if number_of_iterations is not None:
        params["numberOfIterations"] = number_of_iterations
    if iterations_repaired:
        warnings.append(
            f"numberOfIterations length did not match the effective phase count ({phase_count}), so it was normalized automatically."
        )

    minimum_step_length, min_step_repaired = _normalize_phase_parameter_length(
        args.get("minimumStepLength"), target_len=phase_count, cast_type=float
    )
    if minimum_step_length is not None:
        params["minimumStepLength"] = minimum_step_length
    if min_step_repaired:
        warnings.append(
            f"minimumStepLength length did not match the effective phase count ({phase_count}), so it was normalized automatically."
        )

    metadata = {
        "fixed_volume": fixed_volume,
        "moving_volume": moving_volume,
        "linear_output": linear_output,
        "bspline_output": bspline_output,
        "output_volume": output_volume,
        "output_fixed_roi": output_fixed_roi,
        "output_moving_roi": output_moving_roi,
        "requested_output": requested_output,
        "wants_nonlinear": wants_nonlinear,
        "warnings": warnings,
        "effective_parameters": {
            "initializeTransformMode": initialize_transform_mode if (initialize_transform_mode_arg is not None or initial_transform is not None) else None,
            "transformType": list(transform_type_values),
            "phase_flags_override": bool(phase_flags_override),
            "numberOfIterations": list(number_of_iterations) if number_of_iterations is not None else None,
            "minimumStepLength": list(minimum_step_length) if minimum_step_length is not None else None,
        },
    }
    return params, metadata


def _brains_fit(args: Dict[str, Any]) -> Dict[str, Any]:
    module = _get_cli_module("brainsfit", "BRAINSFit")
    params, metadata = _build_brains_fit_parameters(args)
    cli_result = _run_cli_sync(module, params)

    result_transform = metadata.get("bspline_output") or metadata.get("linear_output")
    payload: Dict[str, Any] = {
        **cli_result,
        "fixed_volume_id": str(metadata["fixed_volume"].GetID()),
        "fixed_volume_name": str(metadata["fixed_volume"].GetName()),
        "moving_volume_id": str(metadata["moving_volume"].GetID()),
        "moving_volume_name": str(metadata["moving_volume"].GetName()),
    }
    if metadata.get("linear_output") is not None:
        payload["linear_transform_id"] = str(metadata["linear_output"].GetID())
        payload["linear_transform_name"] = str(metadata["linear_output"].GetName())
    if metadata.get("bspline_output") is not None:
        payload["bspline_transform_id"] = str(metadata["bspline_output"].GetID())
        payload["bspline_transform_name"] = str(metadata["bspline_output"].GetName())
    if metadata.get("output_volume") is not None:
        payload["output_volume_id"] = str(metadata["output_volume"].GetID())
        payload["output_volume_name"] = str(metadata["output_volume"].GetName())
    if metadata.get("output_fixed_roi") is not None:
        payload["output_fixed_volume_roi_id"] = str(metadata["output_fixed_roi"].GetID())
        payload["output_fixed_volume_roi_name"] = str(metadata["output_fixed_roi"].GetName())
    if metadata.get("output_moving_roi") is not None:
        payload["output_moving_volume_roi_id"] = str(metadata["output_moving_roi"].GetID())
        payload["output_moving_volume_roi_name"] = str(metadata["output_moving_roi"].GetName())
    if metadata.get("warnings"):
        payload["warnings"] = list(metadata["warnings"])
    if metadata.get("effective_parameters"):
        payload["effective_parameters"] = dict(metadata["effective_parameters"])
    if result_transform is not None:
        payload.update(_transform_summary(result_transform))
    return payload


def _build_brains_resample_parameters(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    input_volume = _resolve_volume_arg(preferred=args.get("input"), canonical=args.get("inputVolume"), required=True)
    reference_volume = _resolve_volume_arg(preferred=args.get("reference"), canonical=args.get("referenceVolume"), required=False)
    if reference_volume is None:
        reference_volume = input_volume
    deformation_volume = _resolve_volume_arg(preferred=None, canonical=args.get("deformationVolume"), required=False)
    transform_input = _resolve_transform_input_arg(args.get("transform") or args.get("warpTransform"))
    if transform_input is None and deformation_volume is None:
        raise ValueError("brains_resample requires transform/warpTransform or deformationVolume.")

    output_volume = _resolve_or_create_volume_output(
        args.get("output") or args.get("outputVolume"),
        class_name=_volume_output_class_name_like(input_volume),
        default_name="BRAINSResample Output",
    )

    params: Dict[str, Any] = {
        "inputVolume": input_volume,
        "referenceVolume": reference_volume,
        "outputVolume": output_volume,
    }
    if deformation_volume is not None:
        params["deformationVolume"] = deformation_volume
    for key in ("pixelType", "interpolationMode", "inverseTransform", "defaultValue", "numberOfThreads"):
        value = args.get(key)
        if value is not None:
            params[key] = value
    if args.get("gridSpacing") is not None:
        params["gridSpacing"] = _coerce_number_list(args.get("gridSpacing"), cast_type=float)

    metadata = {
        "input_volume": input_volume,
        "reference_volume": reference_volume,
        "output_volume": output_volume,
        "transform_input": transform_input,
        "deformation_volume": deformation_volume,
        "make_active": bool(args.get("make_active", True)),
    }
    return params, metadata, transform_input


def _brains_resample(args: Dict[str, Any]) -> Dict[str, Any]:
    module = _get_cli_module("brainsresample", "BRAINSResample")
    params, metadata, transform_input = _build_brains_resample_parameters(args)
    if transform_input is not None:
        params["warpTransform"] = transform_input
    cli_result = _run_cli_sync(module, params)

    cli_output_text = str(cli_result.get("cli_output_text") or "")
    if transform_input is not None and "warptransform is set as identity" in cli_output_text.lower():
        raise RuntimeError("BRAINSResample ran with an identity warp; the requested transform was not applied.")

    output_volume = metadata["output_volume"]
    if metadata["make_active"]:
        _set_active_volume(output_volume)

    payload: Dict[str, Any] = {
        **cli_result,
        "input_volume_id": str(metadata["input_volume"].GetID()),
        "input_volume_name": str(metadata["input_volume"].GetName()),
        "reference_volume_id": str(metadata["reference_volume"].GetID()),
        "reference_volume_name": str(metadata["reference_volume"].GetName()),
        "output_volume_id": str(output_volume.GetID()),
        "output_volume_name": str(output_volume.GetName()),
        "made_active": metadata["make_active"],
        "active_volume_id": str(output_volume.GetID()) if metadata["make_active"] else None,
        "active_volume_name": str(output_volume.GetName()) if metadata["make_active"] else None,
    }
    if metadata.get("deformation_volume") is not None:
        payload["deformation_volume_id"] = str(metadata["deformation_volume"].GetID())
        payload["deformation_volume_name"] = str(metadata["deformation_volume"].GetName())
    if not isinstance(transform_input, str) and transform_input is not None:
        payload["transform_id"] = str(transform_input.GetID())
        payload["transform_name"] = str(transform_input.GetName())
    return payload


def _build_brains_roi_auto_parameters(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_volume = _resolve_volume_arg(preferred=args.get("input"), canonical=args.get("inputVolume"), required=True)
    mask_output_requested = args.get("outputROIMaskVolume")
    crop_output_requested = args.get("outputVolume")
    primary_output = args.get("output")

    mask_output = args.get("maskOutput")
    crop_output = args.get("cropOutput")
    if mask_output is None and crop_output is None:
        mask_output = True if crop_output_requested is None else False
        crop_output = True if crop_output_requested is not None else False
    else:
        mask_output = bool(mask_output) if mask_output is not None else bool(mask_output_requested)
        crop_output = bool(crop_output) if crop_output is not None else bool(crop_output_requested)
    if not mask_output and not crop_output:
        mask_output = True

    if primary_output and mask_output_requested is None and crop_output_requested is None:
        if mask_output and not crop_output:
            mask_output_requested = primary_output
        elif crop_output and not mask_output:
            crop_output_requested = primary_output
        else:
            mask_output_requested = primary_output

    output_mask = _resolve_mask_output_node(mask_output_requested, default_name="BRAINS ROI Mask") if mask_output else None
    output_volume = _resolve_or_create_volume_output(
        crop_output_requested,
        class_name=_volume_output_class_name_like(input_volume),
        default_name="BRAINS ROI Crop",
    ) if crop_output else None

    params: Dict[str, Any] = {
        "inputVolume": input_volume,
        "maskOutput": bool(mask_output),
        "cropOutput": bool(crop_output),
    }
    if output_mask is not None:
        params["outputROIMaskVolume"] = output_mask
    if output_volume is not None:
        params["outputVolume"] = output_volume
    for key in (
        "otsuPercentileThreshold",
        "thresholdCorrectionFactor",
        "closingSize",
        "ROIAutoDilateSize",
        "outputVolumePixelType",
        "numberOfThreads",
    ):
        value = args.get(key)
        if value is not None:
            params[key] = value

    metadata = {
        "input_volume": input_volume,
        "output_mask": output_mask,
        "output_volume": output_volume,
        "make_active": bool(args.get("make_active", False)),
    }
    return params, metadata


def _brains_roi_auto(args: Dict[str, Any]) -> Dict[str, Any]:
    module = _get_cli_module("brainsroiauto", "BRAINSROIAuto")
    params, metadata = _build_brains_roi_auto_parameters(args)
    cli_result = _run_cli_sync(module, params)

    output_volume = metadata.get("output_volume")
    if output_volume is not None and metadata.get("make_active"):
        _set_active_volume(output_volume)

    payload: Dict[str, Any] = {
        **cli_result,
        "input_volume_id": str(metadata["input_volume"].GetID()),
        "input_volume_name": str(metadata["input_volume"].GetName()),
        "made_active": bool(metadata.get("make_active") and output_volume is not None),
    }
    if metadata.get("output_mask") is not None:
        payload["output_roi_mask_volume_id"] = str(metadata["output_mask"].GetID())
        payload["output_roi_mask_volume_name"] = str(metadata["output_mask"].GetName())
    if output_volume is not None:
        payload["output_volume_id"] = str(output_volume.GetID())
        payload["output_volume_name"] = str(output_volume.GetName())
        if metadata.get("make_active"):
            payload["active_volume_id"] = str(output_volume.GetID())
            payload["active_volume_name"] = str(output_volume.GetName())
    return payload


REGISTRATION_BRIDGE_HANDLERS: Dict[str, Any] = {
    "list_transforms": _list_transforms,
    "create_transform": _create_transform,
    "get_transform": _get_transform,
    "edit_transform": _edit_transform,
    "set_transform_center": _set_transform_center,
    "apply_transform": _apply_transform,
    "remove_transform": _remove_transform,
    "harden_transform": _harden_transform,
    "invert_transform": _invert_transform,
    "compose_transforms": _compose_transforms,
    "split_transform": _split_transform,
    "sample_transform": _sample_transform,
    "save_transform": _save_transform,
    "load_transform": _load_transform,
    "brains_fit": _brains_fit,
    "brains_resample": _brains_resample,
    "brains_roi_auto": _brains_roi_auto,
}
