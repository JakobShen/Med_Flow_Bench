from __future__ import annotations

from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import PackSpec, ToolSpec


def register_optional_packs(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    packs = build.packs

    add(
        ToolSpec(
            name="roi_stats_ijk",
            description="Compute ROI intensity statistics for an IJK box on the current or specified volume.",
            parameters=S.obj(
                {
                    "ijk_min": S.int_vec(),
                    "ijk_max": S.int_vec(),
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                },
                required=["ijk_min", "ijk_max"],
            ),
            handler=lambda args: ctx.roi_stats_ijk(
                ijk_min=args["ijk_min"],
                ijk_max=args["ijk_max"],
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            ),
            pack="stats",
        )
    )
    add(
        ToolSpec(
            name="roi_stats_ras_box",
            description="Compute ROI intensity statistics for a RAS-aligned box on the current or specified volume.",
            parameters=S.obj(
                {
                    "ras_min": S.num_vec(),
                    "ras_max": S.num_vec(),
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                },
                required=["ras_min", "ras_max"],
            ),
            handler=lambda args: ctx.roi_stats_ras_box(
                ras_min=args["ras_min"],
                ras_max=args["ras_max"],
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            ),
            pack="stats",
        )
    )
    add(
        ToolSpec(
            name="sample_intensity_ras",
            description="Sample image intensity at a RAS point on the current or specified volume.",
            parameters=S.obj(
                {
                    "ras": S.num_vec(),
                    "method": S.string(default="nearest"),
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                },
                required=["ras"],
            ),
            handler=lambda args: ctx.sample_intensity_ras(
                ras=args["ras"],
                method=args.get("method", "nearest"),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            ),
            pack="stats",
        )
    )
    packs.append(
        PackSpec(
            name="stats",
            description="Deterministic ROI statistics and intensity sampling. Activate only when the task requires explicit numeric tissue characterization.",
            tool_names=["roi_stats_ijk", "roi_stats_ras_box", "sample_intensity_ras"],
            activation_hint="numeric ROI summaries, density/intensity checks, or direct probing of a focal point",
        )
    )

    add(
        ToolSpec(
            name="measure_distance_ras",
            description="Measure Euclidean distance in mm between two RAS points.",
            parameters=S.obj(
                {
                    "p1": S.num_vec(),
                    "p2": S.num_vec(),
                },
                required=["p1", "p2"],
            ),
            handler=lambda args: ctx.measure_distance_ras(p1=args["p1"], p2=args["p2"]),
            pack="measurement",
        )
    )
    add(
        ToolSpec(
            name="measure_angle_ras",
            description="Measure angle in degrees at p2 between (p1-p2) and (p3-p2).",
            parameters=S.obj(
                {
                    "p1": S.num_vec(),
                    "p2": S.num_vec(),
                    "p3": S.num_vec(),
                },
                required=["p1", "p2", "p3"],
            ),
            handler=lambda args: ctx.measure_angle_ras(p1=args["p1"], p2=args["p2"], p3=args["p3"]),
            pack="measurement",
        )
    )
    add(
        ToolSpec(
            name="measure_area_polygon_ras",
            description="Measure planar polygon area in mm^2 from a list of RAS points.",
            parameters=S.obj(
                {
                    "points": S.array(S.num_vec()),
                },
                required=["points"],
            ),
            handler=lambda args: ctx.measure_area_polygon_ras(points=args["points"]),
            pack="measurement",
        )
    )
    packs.append(
        PackSpec(
            name="measurement",
            description="Deterministic geometric measurements based on explicit RAS points. If you only know image coordinates from a slice screenshot, first use image_coords_to_ras to convert 0..999 image-space points or boxes into RAS before calling these tools.",
            tool_names=["measure_distance_ras", "measure_angle_ras", "measure_area_polygon_ras"],
            activation_hint="tumor size, lesion distance, or explicit point-based measurement; when starting from screenshot coordinates, first call image_coords_to_ras",
        )
    )

    def _thick_slab(args: Dict[str, Any]) -> Dict[str, Any]:
        slab_res = ctx.set_thick_slab(
            view=args.get("view", "axial"),
            enabled=bool(args.get("enabled", True)),
            thickness_mm=float(args.get("thickness_mm", 12.0)),
            mode=args.get("mode", "mip"),
        )
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        return {"ok": True, "slab": slab_res, "observation": obs, "png_path": obs.get("png_path")}

    add(
        ToolSpec(
            name="thick_slab",
            description="Enable or disable thick slab reconstruction and return an updated observation.",
            parameters=S.obj(
                {
                    "view": S.string(default="axial"),
                    "enabled": S.boolean(default=True),
                    "thickness_mm": S.number(default=12.0),
                    "mode": S.string(default="mip"),
                    "size": S.integer(default=512),
                },
                required=["view"],
            ),
            handler=_thick_slab,
            pack="viewer_advanced",
        )
    )

    def _fusion(args: Dict[str, Any]) -> Dict[str, Any]:
        background_ref = args.get("background_volume_ref")
        foreground_ref = args.get("foreground_volume_ref")
        background_id = args.get("background_volume_id")
        background_name = args.get("background_volume_name")
        foreground_id = args.get("foreground_volume_id")
        foreground_name = args.get("foreground_volume_name")

        if not (background_ref or background_id or background_name) or not (foreground_ref or foreground_id or foreground_name):
            return {
                "ok": False,
                "tool": "fusion",
                "error": (
                    "fusion requires both background and foreground volume references. "
                    "Preferred usage: pass background_volume_ref and foreground_volume_ref, where each ref is either a loaded volume id or a loaded volume name from the prompt-provided scene inventory."
                ),
            }

        fusion_res = ctx.fuse_volumes(
            background_volume_ref=background_ref,
            background_volume_id=background_id,
            background_volume_name=background_name,
            foreground_volume_ref=foreground_ref,
            foreground_volume_id=foreground_id,
            foreground_volume_name=foreground_name,
            opacity=float(args.get("opacity", 0.35)),
        )
        reset_res = ctx.reset_standard_views_to_initial_mid_slices(
            enabled=bool(args.get("reset_to_initial_mid_slice", True))
        )
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix="fusion")
        return {
            "ok": True,
            "fusion": fusion_res,
            "reset_to_initial_mid_slice": reset_res,
            "observation": obs,
            "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None,
        }

    add(
        ToolSpec(
            name="fusion",
            description="Set simple foreground/background slice fusion (for example CT + PET) and return the updated standard observation. Pass background_volume_ref and foreground_volume_ref; each ref may be either a loaded volume id or a loaded volume name taken from the prompt-provided scene inventory.",
            parameters=S.obj(
                {
                    "background_volume_ref": S.string(description="Required preferred background volume reference. Use either a loaded volume id or a loaded volume name from the scene inventory."),
                    "foreground_volume_ref": S.string(description="Required preferred foreground volume reference. Use either a loaded volume id or a loaded volume name from the scene inventory."),
                    "background_volume_id": S.string(description="Optional explicit background volume id override."),
                    "background_volume_name": S.string(description="Optional explicit background volume name override."),
                    "foreground_volume_id": S.string(description="Optional explicit foreground volume id override."),
                    "foreground_volume_name": S.string(description="Optional explicit foreground volume name override."),
                    "opacity": S.number(default=0.35),
                    "size": S.integer(default=512),
                    "reset_to_initial_mid_slice": S.boolean(default=True, description="Reset axial/sagittal/coronal views to the initial case-opening middle slices before capturing the fusion preview."),
                },
                required=["background_volume_ref", "foreground_volume_ref"],
            ),
            handler=_fusion,
            pack="core",
            always_on=True,
        )
    )
    def _clear_fusion(args: Dict[str, Any]) -> Dict[str, Any]:
        fusion_res = ctx.clear_fusion()
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix="fusion_clear")
        return {"ok": True, "fusion": fusion_res, "observation": obs, "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None}

    add(
        ToolSpec(
            name="clear_fusion",
            description="Clear foreground/background fusion overlays and return an updated standard observation.",
            parameters=S.obj({"size": S.integer(default=512)}),
            handler=_clear_fusion,
            pack="viewer_advanced",
        )
    )
    add(
        ToolSpec(
            name="compute_subtraction",
            description="Compute a derived subtraction volume (A - B) into a new volume and make it active.",
            parameters=S.obj(
                {
                    "volume_a_id": S.string(),
                    "volume_b_id": S.string(),
                    "output_name": S.string(default="Subtraction"),
                },
                required=["volume_a_id", "volume_b_id"],
            ),
            handler=lambda args: ctx.compute_subtraction(
                volume_a_id=args["volume_a_id"],
                volume_b_id=args["volume_b_id"],
                output_name=args.get("output_name", "Subtraction"),
            ),
            pack="viewer_advanced",
        )
    )

    def _resample_volume_isotropic(args: Dict[str, Any]) -> Dict[str, Any]:
        resample_res = ctx.resample_volume_isotropic(
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            output_spacing_mm=float(args.get("output_spacing_mm", 1.0)),
            output_name=args.get("output_name"),
            interpolation_type=args.get("interpolation_type", "linear"),
            make_active=bool(args.get("make_active", True)),
        )
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix="resample_isotropic")
        return {"ok": True, "resample": resample_res, "observation": obs, "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None}

    add(
        ToolSpec(
            name="resample_volume_isotropic",
            description="Resample the current or specified scalar volume to isotropic spacing and optionally make the result active. Use this when axial looks acceptable but sagittal/coronal are blurry due to thick spacing.",
            parameters=S.obj(
                {
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "output_spacing_mm": S.number(default=1.0),
                    "output_name": S.string(),
                    "interpolation_type": S.string(default="linear", enum=["linear", "nearest", "bspline", "ws"]),
                    "make_active": S.boolean(default=True),
                    "size": S.integer(default=512),
                }
            ),
            handler=_resample_volume_isotropic,
            pack="viewer_advanced",
        )
    )
    packs.append(
        PackSpec(
            name="viewer_advanced",
            description="Less common viewer operations that are useful for PET/CT review or slab-based inspection, but should not distract the model on every task.",
            tool_names=["thick_slab", "clear_fusion", "compute_subtraction", "resample_volume_isotropic"],
            activation_hint="fusion reset, slab visualization, isotropic resampling for blurry MPR views, or derived subtraction volume",
        )
    )
