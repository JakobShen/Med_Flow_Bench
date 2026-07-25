from __future__ import annotations

from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import PackSpec, ToolSpec


def register_segmentation_pack(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    packs = build.packs
    _capture_overlay_standard_views = build.capture_overlay_standard_views

    add(
        ToolSpec(
            name="list_segmentations",
            description="List segmentation nodes currently loaded in the scene.",
            parameters=S.obj({}),
            handler=lambda args: ctx.list_segmentations(),
            pack="segmentation",
        )
    )
    add(
        ToolSpec(
            name="list_segments",
            description="List segments inside a segmentation node. If no segmentation is given, uses the currently active segmentation if available.",
            parameters=S.obj(
                {
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                }
            ),
            handler=lambda args: ctx.list_segments(
                segmentation_id=args.get("segmentation_id"),
                segmentation_name=args.get("segmentation_name"),
            ),
            pack="segmentation",
        )
    )
    add(
        ToolSpec(
            name="hide_segmentation_overlay",
            description=(
                "Hide Slicer segmentation mask overlays from slice views without deleting them. "
                "Use target='all' after a wrong or distracting segmentation, or pass segmentation_id from a prior segmentation tool result to hide one mask."
            ),
            parameters=S.obj(
                {
                    "target": S.string(default="all", enum=["all", "segmentation_id"]),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                }
            ),
            handler=lambda args: ctx.hide_segmentation(
                target=args.get("target", "all"),
                segmentation_id=args.get("segmentation_id"),
                segmentation_name=args.get("segmentation_name"),
                segment_id=args.get("segment_id"),
                segment_name=args.get("segment_name"),
            ),
            pack="segmentation",
        )
    )

    def _segment_from_threshold(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_from_threshold(
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
            minimum_threshold=float(args["minimum_threshold"]),
            maximum_threshold=float(args["maximum_threshold"]),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_threshold")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_from_threshold",
            description="Create or update one segment by thresholding the current or specified source volume. This is suitable for coarse, stable segmentation only.",
            parameters=S.obj(
                {
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "minimum_threshold": S.number(),
                    "maximum_threshold": S.number(),
                    "size": S.integer(default=512),
                },
                required=["minimum_threshold", "maximum_threshold"],
            ),
            handler=_segment_from_threshold,
            pack="segmentation",
        )
    )

    def _segment_local_threshold(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_local_threshold(
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
            minimum_threshold=float(args["minimum_threshold"]),
            maximum_threshold=float(args["maximum_threshold"]),
            minimum_diameter_mm=float(args.get("minimum_diameter_mm", 3.0)),
            feature_size_mm=float(args.get("feature_size_mm", 3.0)),
            segmentation_algorithm=args.get("segmentation_algorithm", "GrowCut"),
            seed_view=args.get("seed_view"),
            seed_bbox_1000=args.get("seed_bbox_1000"),
            size=int(args.get("size", 512)),
        )
        return seg_res

    add(
        ToolSpec(
            name="segment_local_threshold",
            description=(
                "Run SegmentEditorExtraEffects Local Threshold from a single seed chosen on one fresh slice image. "
                "Use this instead of global threshold for focal soft-tissue lesion illustration. "
                "Primary input contract: provide seed_view plus seed_bbox_1000=[x_min,y_min,x_max,y_max] as INTEGER coordinates in the fixed 0..999 image space with origin at the top-left of the exact cropped slice image you just inspected (outer black/transparent border trimmed). "
                "Make the box tight around the intended lesion center; the backend uses the bbox center as the actual seed. "
                "Do not invent ras_seed values and do not guess physical coordinates. "
                "If the tool returns ok=false or failure_reason=zero_voxels, the segmentation failed and you should inspect the seed preview and retry with a tighter seed_bbox_1000 and/or adjusted thresholds. "
                "The result returns segmentation_id/segment_id, an annotated seed preview, and overlay screenshots so you can verify where the seed landed and whether the mask is plausible. "
                "If the mask is wrong, call hide_segmentation_overlay(target='all') or hide_segmentation_overlay(target='segmentation_id', segmentation_id=...)."
            ),
            parameters=S.obj(
                {
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "minimum_threshold": S.number(),
                    "maximum_threshold": S.number(),
                    "minimum_diameter_mm": S.number(default=3.0),
                    "feature_size_mm": S.number(default=3.0),
                    "segmentation_algorithm": S.string(default="GrowCut", enum=["Masking", "GrowCut", "WaterShed"]),
                    "seed_view": S.string(description="axial, sagittal, or coronal; use the exact view whose image you inspected"),
                    "seed_bbox_1000": S.bbox_1000(
                        description="Tight lesion box [x_min,y_min,x_max,y_max] in integer 0..999 coordinates on the chosen slice image. The backend uses the bbox center as the seed."
                    ),
                    "size": S.integer(default=512),
                },
                required=["minimum_threshold", "maximum_threshold", "seed_view", "seed_bbox_1000"],
            ),
            handler=_segment_local_threshold,
            pack="segmentation",
        )
    )

    def _segment_edit_sphere(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_edit_sphere(
            ras_center=args["ras_center"],
            radius_mm=float(args["radius_mm"]),
            action=args.get("action", "add"),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_sphere")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_edit_sphere",
            description="Locally add to or erase from a segment using a sphere defined by a RAS center point and radius in mm. This is a stable replacement for freehand editing.",
            parameters=S.obj(
                {
                    "ras_center": S.num_vec(),
                    "radius_mm": S.number(),
                    "action": S.string(default="add", enum=["add", "erase"]),
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                },
                required=["ras_center", "radius_mm"],
            ),
            handler=_segment_edit_sphere,
            pack="segmentation",
        )
    )

    def _segment_margin(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_margin(
            margin_size_mm=float(args["margin_size_mm"]),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_margin")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_margin",
            description="Grow or shrink the selected segment by a margin in mm.",
            parameters=S.obj(
                {
                    "margin_size_mm": S.number(),
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                },
                required=["margin_size_mm"],
            ),
            handler=_segment_margin,
            pack="segmentation",
        )
    )

    def _segment_smoothing(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_smoothing(
            smoothing_method=args.get("smoothing_method", "MEDIAN"),
            kernel_size_mm=float(args.get("kernel_size_mm", 3.0)),
            gaussian_std_mm=args.get("gaussian_std_mm"),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_smoothing")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_smoothing",
            description="Apply a stable smoothing method to the selected segment. Supported methods are limited to the most reliable Segment Editor smoothing modes.",
            parameters=S.obj(
                {
                    "smoothing_method": S.string(default="MEDIAN", enum=["MEDIAN", "GAUSSIAN", "MORPHOLOGICAL_CLOSING"]),
                    "kernel_size_mm": S.number(default=3.0),
                    "gaussian_std_mm": S.number(),
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                },
            ),
            handler=_segment_smoothing,
            pack="segmentation",
        )
    )

    def _segment_islands(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_islands(
            operation=args.get("operation", "KEEP_LARGEST_ISLAND"),
            minimum_size=int(args.get("minimum_size", 1000)),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_islands")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_islands",
            description="Keep the largest connected component or remove small islands from the selected segment.",
            parameters=S.obj(
                {
                    "operation": S.string(default="KEEP_LARGEST_ISLAND", enum=["KEEP_LARGEST_ISLAND", "REMOVE_SMALL_ISLANDS"]),
                    "minimum_size": S.integer(default=1000),
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                },
            ),
            handler=_segment_islands,
            pack="segmentation",
        )
    )

    def _segment_logical(args: Dict[str, Any]) -> Dict[str, Any]:
        seg_res = ctx.segment_logical(
            operation=args["operation"],
            modifier_segment_id=args.get("modifier_segment_id"),
            modifier_segment_name=args.get("modifier_segment_name"),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="segment_logical")
        return {"ok": True, "segmentation": seg_res, **overlay}

    add(
        ToolSpec(
            name="segment_logical",
            description="Run a stable logical operator on the selected segment. Use modifier_segment only for COPY, UNION, INTERSECT, or SUBTRACT.",
            parameters=S.obj(
                {
                    "operation": S.string(enum=["COPY", "UNION", "INTERSECT", "SUBTRACT", "INVERT", "CLEAR", "FILL"]),
                    "modifier_segment_id": S.string(),
                    "modifier_segment_name": S.string(),
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                },
                required=["operation"],
            ),
            handler=_segment_logical,
            pack="segmentation",
        )
    )

    add(
        ToolSpec(
            name="segment_statistics",
            description="Return stable per-segment statistics such as voxel count, volume, center RAS, and intensity summaries when a source volume is available.",
            parameters=S.obj(
                {
                    "source_volume_id": S.string(),
                    "source_volume_name": S.string(),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                }
            ),
            handler=lambda args: ctx.segment_statistics(
                source_volume_id=args.get("source_volume_id"),
                source_volume_name=args.get("source_volume_name"),
                segmentation_id=args.get("segmentation_id"),
                segmentation_name=args.get("segmentation_name"),
                segment_id=args.get("segment_id"),
                segment_name=args.get("segment_name"),
            ),
            pack="segmentation",
        )
    )

    def _center_on_segment(args: Dict[str, Any]) -> Dict[str, Any]:
        center_res = ctx.center_on_segment(
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        )
        overlay = _capture_overlay_standard_views(size=int(args.get("size", 512)), out_prefix="center_on_segment")
        return {"ok": True, "center": center_res, **overlay}

    add(
        ToolSpec(
            name="center_on_segment",
            description="Jump all slice views to the center of the selected segment and return updated observations.",
            parameters=S.obj(
                {
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "size": S.integer(default=512),
                }
            ),
            handler=_center_on_segment,
            pack="segmentation",
        )
    )

    add(
        ToolSpec(
            name="export_segmentation_dicom",
            description="Export a segmentation node to DICOM SEG using a reference volume. This is a packaging/export operation, not an editing step.",
            parameters=S.obj(
                {
                    "segmentation_id": S.string(),
                    "reference_volume_id": S.string(),
                    "output_folder": S.string(),
                },
                required=["segmentation_id", "reference_volume_id", "output_folder"],
            ),
            handler=lambda args: ctx.export_segmentation_dicom(
                segmentation_id=args["segmentation_id"],
                reference_volume_id=args["reference_volume_id"],
                output_folder=args["output_folder"],
            ),
            pack="segmentation",
        )
    )
    packs.append(
        PackSpec(
            name="segmentation",
            description="Curated stable segmentation tools. This is intentionally narrower than full Segment Editor: it focuses on threshold-based initialization, Local Threshold seeded growth, local spherical edits, morphology, islands cleanup, logical ops, stats, centering, and DICOM SEG export.",
            tool_names=[
                "list_segmentations",
                "list_segments",
                "hide_segmentation_overlay",
                "segment_from_threshold",
                "segment_local_threshold",
                "segment_edit_sphere",
                "segment_margin",
                "segment_smoothing",
                "segment_islands",
                "segment_logical",
                "segment_statistics",
                "center_on_segment",
                "export_segmentation_dicom",
            ],
            activation_hint="coarse but auditable segmentation tasks; for focal lesions, first inspect one slice, then call segment_local_threshold with seed_view plus a tight seed_bbox_1000 lesion box in 0..999 image coordinates",
        )
    )
