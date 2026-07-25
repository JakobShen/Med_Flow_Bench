from __future__ import annotations

import os
from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import PackSpec, ToolSpec


_TRANSFORM_REF_PROPERTIES = {
    "transform": S.string(description="Preferred transform reference. Use either a loaded transform id or a loaded transform name."),
    "transform_id": S.string(description="Optional explicit transform id override."),
    "transform_name": S.string(description="Optional explicit transform name override."),
}

_VOLUME_REF_PROPERTIES = {
    "volume_ref": S.string(description="Preferred volume reference. Use either a loaded volume id or a loaded volume name."),
    "volume_id": S.string(description="Optional explicit volume id override."),
    "volume_name": S.string(description="Optional explicit volume name override."),
}


def _transform_ref_schema(*, include_preferred: bool = True) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if include_preferred:
        props["transform"] = _TRANSFORM_REF_PROPERTIES["transform"]
    props["transform_id"] = _TRANSFORM_REF_PROPERTIES["transform_id"]
    props["transform_name"] = _TRANSFORM_REF_PROPERTIES["transform_name"]
    return props


def _volume_ref_schema(*, preferred_key: str = "volume_ref", description: str = "Preferred volume reference. Use either a loaded volume id or a loaded volume name.") -> Dict[str, Any]:
    return {
        preferred_key: S.string(description=description),
        "volume_id": _VOLUME_REF_PROPERTIES["volume_id"],
        "volume_name": _VOLUME_REF_PROPERTIES["volume_name"],
    }


def register_registration_pack(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    packs = build.packs

    def _safe_standard_preview(*, out_prefix: str, size: int = 512) -> Dict[str, Any]:
        reset_result = None
        try:
            reset_result = ctx.reset_standard_views_to_initial_mid_slices()
        except Exception as e:
            reset_result = {"ok": False, "error": str(e)}

        try:
            obs = ctx.observe_standard_views(size=int(size), out_prefix=out_prefix)
        except Exception as e:
            preview_error: Dict[str, Any] = {"preview_error": str(e)}
            if reset_result is not None:
                preview_error["reset_to_initial_mid_slice"] = reset_result
            return preview_error

        preview: Dict[str, Any] = {"observation": obs}
        if reset_result is not None:
            preview["reset_to_initial_mid_slice"] = reset_result
        if isinstance(obs, dict):
            png_paths = obs.get("png_paths")
            if png_paths is not None:
                preview["png_paths"] = png_paths
            if isinstance(png_paths, dict):
                first_png = next((str(value) for value in png_paths.values() if value), None)
                if first_png:
                    preview["png_path"] = first_png
        return preview

    def _safe_fusion_preview(*, fixed_ref: Any, moving_ref: Any, out_prefix: str, size: int = 512) -> Dict[str, Any]:
        preview: Dict[str, Any] = {}
        if fixed_ref and moving_ref:
            try:
                preview["preview_fusion"] = ctx.fuse_volumes(
                    background_volume_ref=str(fixed_ref),
                    foreground_volume_ref=str(moving_ref),
                    opacity=0.35,
                )
            except Exception as e:
                preview["preview_fusion_error"] = str(e)
        preview.update(_safe_standard_preview(out_prefix=out_prefix, size=size))
        return preview

    def _merge_preview(result: Dict[str, Any], *, out_prefix: str, size: int = 512) -> Dict[str, Any]:
        merged = dict(result)
        merged.update(_safe_standard_preview(out_prefix=out_prefix, size=size))
        return merged

    def _brains_fit_with_preview(args: Dict[str, Any]) -> Dict[str, Any]:
        result = ctx.brains_fit(**args)
        transform_ref = result.get("transform_id") or result.get("linear_transform_id") or result.get("bspline_transform_id")
        moving_ref = result.get("moving_volume_id") or args.get("moving") or args.get("movingVolume")
        fixed_ref = result.get("fixed_volume_id") or args.get("fixed") or args.get("fixedVolume")

        preview: Dict[str, Any] = {}
        if transform_ref and moving_ref:
            try:
                preview["preview_apply_transform"] = ctx.apply_transform(transform=str(transform_ref), node_ref=str(moving_ref))
                preview["preview_transform_applied"] = True
            except Exception as e:
                preview["preview_apply_transform_error"] = str(e)
        preview.update(_safe_fusion_preview(fixed_ref=fixed_ref, moving_ref=moving_ref, out_prefix="brains_fit"))
        return {**result, **preview}

    def _brains_resample_with_preview(args: Dict[str, Any]) -> Dict[str, Any]:
        result = ctx.brains_resample(**args)
        preview = _safe_fusion_preview(
            fixed_ref=result.get("reference_volume_id") or args.get("reference") or args.get("referenceVolume"),
            moving_ref=result.get("output_volume_id") or args.get("output") or args.get("outputVolume"),
            out_prefix="brains_resample",
        )
        return {**result, **preview}

    def _brains_roi_auto_with_preview(args: Dict[str, Any]) -> Dict[str, Any]:
        result = ctx.brains_roi_auto(**args)
        return _merge_preview(result, out_prefix="brains_roi_auto")

    add(
        ToolSpec(
            name="list_transforms",
            description="List transform nodes in the scene.",
            parameters=S.obj({}),
            handler=lambda args: ctx.list_transforms(),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="create_transform",
            description="Create a new transform node.",
            parameters=S.obj(
                {
                    "name": S.string(),
                    "kind": S.string(default="general", enum=["general", "linear"]),
                    "make_active": S.boolean(default=True),
                }
            ),
            handler=lambda args: ctx.create_transform(
                name=args.get("name"),
                kind=args.get("kind", "general"),
                make_active=bool(args.get("make_active", True)),
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="get_transform",
            description="Get one transform and its current state.",
            parameters=S.obj(_transform_ref_schema()),
            handler=lambda args: ctx.get_transform(
                transform=args.get("transform"),
                transform_id=args.get("transform_id"),
                transform_name=args.get("transform_name"),
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="edit_transform",
            description="Edit a transform and return updated views.",
            parameters=S.obj(
                {
                    **_transform_ref_schema(),
                    "matrix_4x4": S.array(S.num_vec(min_items=4, max_items=4), min_items=4, max_items=4),
                    "translation": S.num_vec(min_items=3, max_items=3),
                    "rotation_degrees": S.num_vec(min_items=3, max_items=3),
                    "scale": S.num_vec(min_items=3, max_items=3),
                    "shear": S.num_vec(
                        min_items=6,
                        max_items=6,
                        description="Six shear factors ordered as [xy, xz, yx, yz, zx, zy].",
                    ),
                    "translation_reference_frame": S.string(default="parent", enum=["parent", "local"]),
                    "reset_to_identity": S.boolean(default=False),
                }
            ),
            handler=lambda args: _merge_preview(
                ctx.edit_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                    matrix_4x4=args.get("matrix_4x4"),
                    translation=args.get("translation"),
                    rotation_degrees=args.get("rotation_degrees"),
                    scale=args.get("scale"),
                    shear=args.get("shear"),
                    translation_reference_frame=args.get("translation_reference_frame", "parent"),
                    reset_to_identity=bool(args.get("reset_to_identity", False)),
                ),
                out_prefix="edit_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="set_transform_center",
            description="Set or reset the center of transformation.",
            parameters=S.obj(
                {
                    **_transform_ref_schema(),
                    "mode": S.string(default="world_ras", enum=["world_ras", "local", "origin", "transformed_nodes", "node_bounds_center"]),
                    "ras": S.num_vec(min_items=3, max_items=3),
                    "local": S.num_vec(min_items=3, max_items=3),
                    "node_ref": S.string(description="A transformed scene node id or name used when mode=node_bounds_center."),
                    "node_id": S.string(),
                    "node_name": S.string(),
                }
            ),
            handler=lambda args: ctx.set_transform_center(
                transform=args.get("transform"),
                transform_id=args.get("transform_id"),
                transform_name=args.get("transform_name"),
                mode=args.get("mode", "world_ras"),
                ras=args.get("ras"),
                local=args.get("local"),
                node_ref=args.get("node_ref"),
                node_id=args.get("node_id"),
                node_name=args.get("node_name"),
            ),
            pack="registration",
        )
    )

    node_list_properties: Dict[str, Any] = {
        "node_refs": S.array(S.string(), description="Preferred scene node references. Each item may be either a node id or a node name."),
        "node_ids": S.array(S.string()),
        "node_names": S.array(S.string()),
        "node_ref": S.string(),
        "node_id": S.string(),
        "node_name": S.string(),
    }

    add(
        ToolSpec(
            name="apply_transform",
            description="Apply a transform to scene nodes and return updated views.",
            parameters=S.obj({**_transform_ref_schema(), **node_list_properties}),
            handler=lambda args: _merge_preview(
                ctx.apply_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                    node_refs=args.get("node_refs"),
                    node_ids=args.get("node_ids"),
                    node_names=args.get("node_names"),
                    node_ref=args.get("node_ref"),
                    node_id=args.get("node_id"),
                    node_name=args.get("node_name"),
                ),
                out_prefix="apply_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="remove_transform",
            description="Remove a transform from scene nodes and return updated views.",
            parameters=S.obj({**_transform_ref_schema(), **node_list_properties}),
            handler=lambda args: _merge_preview(
                ctx.remove_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                    node_refs=args.get("node_refs"),
                    node_ids=args.get("node_ids"),
                    node_names=args.get("node_names"),
                    node_ref=args.get("node_ref"),
                    node_id=args.get("node_id"),
                    node_name=args.get("node_name"),
                ),
                out_prefix="remove_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="harden_transform",
            description="Harden a transform on scene nodes and return updated views.",
            parameters=S.obj({**_transform_ref_schema(), **node_list_properties}),
            handler=lambda args: _merge_preview(
                ctx.harden_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                    node_refs=args.get("node_refs"),
                    node_ids=args.get("node_ids"),
                    node_names=args.get("node_names"),
                    node_ref=args.get("node_ref"),
                    node_id=args.get("node_id"),
                    node_name=args.get("node_name"),
                ),
                out_prefix="harden_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="invert_transform",
            description="Invert one transform and return updated views.",
            parameters=S.obj(_transform_ref_schema()),
            handler=lambda args: _merge_preview(
                ctx.invert_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                ),
                out_prefix="invert_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="compose_transforms",
            description="Set one transform under another transform.",
            parameters=S.obj(
                {
                    "child_transform": S.string(description="Preferred child transform reference. Use either a transform id or a transform name."),
                    "child_transform_id": S.string(),
                    "child_transform_name": S.string(),
                    "parent_transform": S.string(description="Preferred parent transform reference. Use either a transform id or a transform name."),
                    "parent_transform_id": S.string(),
                    "parent_transform_name": S.string(),
                },
                required=["child_transform", "parent_transform"],
            ),
            handler=lambda args: ctx.compose_transforms(
                child_transform=args.get("child_transform"),
                child_transform_id=args.get("child_transform_id"),
                child_transform_name=args.get("child_transform_name"),
                parent_transform=args.get("parent_transform"),
                parent_transform_id=args.get("parent_transform_id"),
                parent_transform_name=args.get("parent_transform_name"),
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="split_transform",
            description="Split a composite transform into component transforms.",
            parameters=S.obj(_transform_ref_schema()),
            handler=lambda args: ctx.split_transform(
                transform=args.get("transform"),
                transform_id=args.get("transform_id"),
                transform_name=args.get("transform_name"),
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="sample_transform",
            description="Sample a transform and return updated views.",
            parameters=S.obj(
                {
                    **_transform_ref_schema(),
                    "reference": S.string(description="Preferred reference volume reference. Use either a loaded volume id or a loaded volume name."),
                    "referenceVolume": S.string(description="Official reference volume parameter. Use a loaded volume id or name."),
                    "reference_volume_id": S.string(),
                    "reference_volume_name": S.string(),
                    "output_kind": S.string(default="grid_transform", enum=["grid_transform", "vector_displacement_volume", "scalar_displacement_magnitude"]),
                    "output_name": S.string(),
                    "make_active": S.boolean(default=False),
                },
                required=["output_kind"],
            ),
            handler=lambda args: _merge_preview(
                ctx.sample_transform(
                    transform=args.get("transform"),
                    transform_id=args.get("transform_id"),
                    transform_name=args.get("transform_name"),
                    reference=args.get("reference"),
                    reference_volume=args.get("referenceVolume"),
                    reference_volume_id=args.get("reference_volume_id"),
                    reference_volume_name=args.get("reference_volume_name"),
                    output_kind=args.get("output_kind", "grid_transform"),
                    output_name=args.get("output_name"),
                    make_active=bool(args.get("make_active", False)),
                ),
                out_prefix="sample_transform",
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="save_transform",
            description="Save a transform node to file.",
            parameters=S.obj(
                {
                    **_transform_ref_schema(),
                    "output_path": S.string(),
                },
                required=["output_path"],
            ),
            handler=lambda args: ctx.save_transform(
                transform=args.get("transform"),
                transform_id=args.get("transform_id"),
                transform_name=args.get("transform_name"),
                output_path=args["output_path"],
            ),
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="load_transform",
            description="Load a transform from file.",
            parameters=S.obj(
                {
                    "path": S.string(),
                    "make_active": S.boolean(default=True),
                },
                required=["path"],
            ),
            handler=lambda args: ctx.load_transform(
                path=args["path"],
                make_active=bool(args.get("make_active", True)),
            ),
            pack="registration",
        )
    )

    brains_fit_properties: Dict[str, Any] = {
        "fixed": S.string(description="Convenience alias for fixedVolume. Use a loaded volume id or name."),
        "moving": S.string(description="Convenience alias for movingVolume. Use a loaded volume id or name."),
        "output": S.string(description="Convenience alias for the main output transform name or reference."),
        "output_transform": S.string(description="Convenience alias for the main output transform name or reference."),
        "fixedVolume": S.string(description="Official fixed volume parameter. Use a loaded volume id or name."),
        "movingVolume": S.string(description="Official moving volume parameter. Use a loaded volume id or name."),
        "fixedVolume2": S.string(description="Optional second fixed volume for multi-image registration."),
        "movingVolume2": S.string(description="Optional second moving volume for multi-image registration."),
        "linearTransform": S.string(description="Official linear transform output node ref or name."),
        "bsplineTransform": S.string(description="Official bspline transform output node ref or name."),
        "outputVolume": S.string(description="Optional official output volume node ref or name."),
        "strippedOutputTransform": S.string(description="Optional stripped output transform file path."),
        "outputTransform": S.string(description="Optional composite output transform file path."),
        "outputVolumePixelType": S.string(enum=["float", "short", "ushort", "int", "uint", "uchar"]),
        "backgroundFillValue": S.number(),
        "scaleOutputValues": S.boolean(),
        "initializeTransformMode": S.string(enum=["Off", "useMomentsAlign", "useCenterOfHeadAlign", "useGeometryAlign", "useCenterOfROIAlign"]),
        "initialTransform": S.string(description="Optional initial transform ref or name."),
        "initializeRegistrationByCurrentGenericTransform": S.boolean(),
        "useRigid": S.boolean(),
        "useScaleVersor3D": S.boolean(),
        "useScaleSkewVersor3D": S.boolean(),
        "useAffine": S.boolean(),
        "useBSpline": S.boolean(),
        "useSyN": S.boolean(),
        "useComposite": S.boolean(),
        "transformType": S.array(S.string()),
        "samplingPercentage": S.number(minimum=0.0, maximum=1.0),
        "numberOfSamples": S.integer(minimum=1),
        "metricSamplingStrategy": S.string(enum=["Random", "RandomCoordinate", "Regular"]),
        "splineGridSize": S.int_vec(min_items=3, max_items=3),
        "maskProcessingMode": S.string(enum=["NOMASK", "ROIAUTO", "ROI"]),
        "fixedBinaryVolume": S.string(description="Fixed mask volume ref or name."),
        "movingBinaryVolume": S.string(description="Moving mask volume ref or name."),
        "outputFixedVolumeROI": S.string(description="Optional fixed ROI output volume ref or name."),
        "outputMovingVolumeROI": S.string(description="Optional moving ROI output volume ref or name."),
        "useROIBSpline": S.boolean(),
        "histogramMatch": S.boolean(),
        "medianFilterSize": S.int_vec(min_items=3, max_items=3),
        "removeIntensityOutliers": S.number(minimum=0.0),
        "numberOfHistogramBins": S.integer(minimum=1),
        "numberOfMatchPoints": S.integer(minimum=1),
        "costMetric": S.string(enum=["MMI", "MSE", "NC", "MI", "MC"]),
        "numberOfIterations": S.array(S.integer(minimum=0), min_items=1),
        "maximumStepLength": S.number(minimum=0.0),
        "minimumStepLength": S.array(S.number(minimum=0.0), min_items=1),
        "relaxationFactor": S.number(minimum=0.0),
        "translationScale": S.number(minimum=0.0),
        "reproportionScale": S.number(minimum=0.0),
        "skewScale": S.number(minimum=0.0),
        "maxBSplineDisplacement": S.number(minimum=0.0),
        "fixedVolumeTimeIndex": S.integer(minimum=0),
        "movingVolumeTimeIndex": S.integer(minimum=0),
        "interpolationMode": S.string(enum=["Linear", "NearestNeighbor", "BSpline", "WindowedSinc", "ResampleInPlace"]),
        "minimumStepLength": S.array(S.number(minimum=0.0), min_items=1),
        "maskInferiorCutOffFromCenter": S.number(),
        "ROIAutoDilateSize": S.number(minimum=0.0),
        "ROIAutoClosingSize": S.number(minimum=0.0),
        "writeOutputTransformInFloat": S.boolean(),
        "numberOfThreads": S.integer(minimum=1),
        "debugLevel": S.integer(minimum=0),
        "costFunctionConvergenceFactor": S.number(minimum=0.0),
        "projectedGradientTolerance": S.number(minimum=0.0),
        "maximumNumberOfEvaluations": S.integer(minimum=0),
        "maximumNumberOfCorrections": S.integer(minimum=0),
        "failureExitCode": S.integer(),
        "writeTransformOnFailure": S.boolean(),
        "promptUser": S.boolean(),
        "logFileReport": S.string(),
    }
    brains_fit_properties.pop("minimumStepLength", None)
    brains_fit_properties["minimumStepLength"] = S.array(S.number(minimum=0.0), min_items=1)

    disable_brains_fit = os.environ.get("SLICER_AGENT_DISABLE_BRAINS_FIT", "").strip().lower() in {"1", "true", "yes", "on"}
    if not disable_brains_fit:
        add(
            ToolSpec(
                name="brains_fit",
                description="Run BRAINSFit registration and return a preview.",
                parameters=S.obj(brains_fit_properties),
                handler=_brains_fit_with_preview,
                pack="registration",
            )
        )

    add(
        ToolSpec(
            name="brains_resample",
            description="Resample a volume with BRAINSResample and return a preview.",
            parameters=S.obj(
                {
                    "input": S.string(description="Convenience alias for inputVolume. Use a loaded volume id or name."),
                    "reference": S.string(description="Convenience alias for referenceVolume. Use a loaded volume id or name."),
                    "transform": S.string(description="Convenience alias for warpTransform. Use a loaded transform id or name."),
                    "output": S.string(description="Convenience alias for outputVolume. Use a loaded volume id or name, or a desired new node name."),
                    "inputVolume": S.string(description="Official input volume parameter. Use a loaded volume id or name."),
                    "referenceVolume": S.string(description="Official reference volume parameter. Use a loaded volume id or name."),
                    "outputVolume": S.string(description="Official output volume node ref or name."),
                    "pixelType": S.string(enum=["binary", "uchar", "short", "ushort", "int", "float", "double"]),
                    "deformationVolume": S.string(description="Optional deformation field volume ref or name."),
                    "warpTransform": S.string(description="Official warp transform ref or name."),
                    "interpolationMode": S.string(enum=["NearestNeighbor", "Linear", "ResampleInPlace", "BSpline", "WindowedSinc", "Hamming", "Cosine", "Welch", "Lanczos", "Blackman"]),
                    "inverseTransform": S.boolean(),
                    "defaultValue": S.number(),
                    "gridSpacing": S.num_vec(min_items=3, max_items=3),
                    "numberOfThreads": S.integer(minimum=1),
                    "make_active": S.boolean(default=True),
                }
            ),
            handler=_brains_resample_with_preview,
            pack="registration",
        )
    )

    add(
        ToolSpec(
            name="brains_roi_auto",
            description="Create an automatic ROI mask or cropped volume and return a preview.",
            parameters=S.obj(
                {
                    "input": S.string(description="Convenience alias for inputVolume. Use a loaded volume id or name."),
                    "output": S.string(description="Convenience alias for the primary ROI output node ref or name."),
                    "inputVolume": S.string(description="Official input volume parameter. Use a loaded volume id or name."),
                    "outputROIMaskVolume": S.string(description="Official ROI mask output node ref or name."),
                    "outputVolume": S.string(description="Official cropped output volume node ref or name."),
                    "maskOutput": S.boolean(),
                    "cropOutput": S.boolean(),
                    "otsuPercentileThreshold": S.number(minimum=0.0, maximum=1.0),
                    "thresholdCorrectionFactor": S.number(),
                    "closingSize": S.number(minimum=0.0),
                    "ROIAutoDilateSize": S.number(minimum=0.0),
                    "outputVolumePixelType": S.string(enum=["float", "short", "ushort", "int", "uint", "uchar"]),
                    "numberOfThreads": S.integer(minimum=1),
                    "make_active": S.boolean(default=False),
                }
            ),
            handler=_brains_roi_auto_with_preview,
            pack="registration",
        )
    )

    tool_names = [
        "list_transforms",
        "create_transform",
        "get_transform",
        "edit_transform",
        "set_transform_center",
        "apply_transform",
        "remove_transform",
        "harden_transform",
        "invert_transform",
        "compose_transforms",
        "split_transform",
        "sample_transform",
        "save_transform",
        "load_transform",
        "brains_resample",
        "brains_roi_auto",
    ]
    if not disable_brains_fit:
        tool_names.insert(-2, "brains_fit")
    packs.append(
        PackSpec(
            name="registration",
            description="Manual transforms plus BRAINS registration and resampling.",
            tool_names=tool_names,
            activation_hint="manual alignment, rigid or affine registration, transform application, controlled resampling, or automatic ROI masks; state-changing tools return preview images",
        )
    )
