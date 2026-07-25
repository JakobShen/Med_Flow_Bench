from __future__ import annotations

from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import PackSpec, ToolSpec


def register_monai_label_pack(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    packs = build.packs

    add(
        ToolSpec(
            name="monai_label_status",
            description="Check the configured MONAI Label server and return the server/app information from /info/.",
            parameters=S.obj({}),
            handler=lambda args: ctx.monai_label_status(),
            pack="monai_label",
        )
    )

    add(
        ToolSpec(
            name="vista3d_label_catalog",
            description="Return VISTA3D class ids from bundle metadata when available. For NSCLC CT lung tumor, use label_prompt=[23].",
            parameters=S.obj(
                {
                    "query": S.string(description="Optional text/id filter such as 'lung' or 'tumor'."),
                    "limit": S.integer(default=50, minimum=1, maximum=200),
                }
            ),
            handler=lambda args: ctx.vista3d_label_catalog(query=args.get("query"), limit=int(args.get("limit", 50))),
            pack="monai_label",
        )
    )

    def _vista3d_segment(args: Dict[str, Any]) -> Dict[str, Any]:
        return ctx.vista3d_segment(
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            model=args.get("model", "vista3d"),
            label_prompt=args.get("label_prompt"),
            points_ijk=args.get("points_ijk"),
            points_ras=args.get("points_ras"),
            point_labels=args.get("point_labels"),
            seed_view=args.get("seed_view"),
            seed_bbox_1000=args.get("seed_bbox_1000"),
            seed_point_1000=args.get("seed_point_1000"),
            point_label=int(args.get("point_label", 1)),
            segmentation_name=args.get("segmentation_name"),
            size=int(args.get("size", 512)),
            auto_show=bool(args.get("auto_show", True)),
            auto_center=bool(args.get("auto_center", True)),
            opacity=float(args.get("opacity", 0.3)),
            screen_views=args.get("screen_views"),
        )

    add(
        ToolSpec(
            name="vista3d_segment",
            description=(
                "Run MONAI Label/VISTA3D on the current or specified Slicer volume, import the returned labelmap as a Slicer segmentation, "
                "show the mask in Slicer by default, center axial/sagittal/coronal views on the mask, and return normal Slicer viewer images. "
                "Use label_prompt for VISTA3D class ids, for example [23] for NSCLC CT lung tumor. If label_prompt is omitted, VISTA3D runs without a class prompt. "
                "For point prompting, send either points_ijk, points_ras, or a human-style seed_view plus seed_bbox_1000/seed_point_1000."
            ),
            parameters=S.obj(
                {
                    "source_volume_id": S.string(description="Slicer volume node ID. If omitted, uses the active volume."),
                    "source_volume_name": S.string(description="Slicer volume node name. Used only when source_volume_id is omitted."),
                    "model": S.string(default="vista3d", description="MONAI Label model name exposed by the server."),
                    "label_prompt": S.int_vec(description="VISTA3D class index prompt list. For NSCLC CT lung tumor use [23]. Leave omitted for no class prompt."),
                    "points_ijk": S.array(S.num_vec(min_items=3, max_items=3), description="Original image-space IJK point prompts as [[i,j,k], ...]."),
                    "points_ras": S.array(S.num_vec(min_items=3, max_items=3), description="Physical RAS point prompts; the tool converts them to VISTA3D image-space points."),
                    "point_labels": S.int_vec(description="Point labels matching points: 1 foreground, 0 background, -1 ignore."),
                    "seed_view": S.string(enum=["axial", "sagittal", "coronal"], description="View used for human-style seed prompt conversion."),
                    "seed_bbox_1000": S.bbox_1000(description="Tight bbox on the latest seed_view screenshot; its center becomes a foreground/background point."),
                    "seed_point_1000": S.point_1000(description="Point on the latest seed_view screenshot."),
                    "point_label": S.integer(default=1, description="Point label used for seed_bbox_1000/seed_point_1000 when point_labels is omitted."),
                    "segmentation_name": S.string(description="Name for the imported Slicer segmentation node."),
                    "size": S.integer(default=512, description="Returned normal viewer image size."),
                    "auto_show": S.boolean(default=True, description="Show the imported segmentation in Slicer immediately."),
                    "auto_center": S.boolean(default=True, description="Jump axial/sagittal/coronal slice views to the segmentation center immediately."),
                    "opacity": S.number(default=0.3, minimum=0.0, maximum=1.0, description="Viewer mask opacity for the green VISTA3D label overlay."),
                    "screen_views": S.array(S.string(enum=["axial", "sagittal", "coronal"]), default=["axial", "sagittal", "coronal"]),
                }
            ),
            handler=_vista3d_segment,
            pack="monai_label",
        )
    )

    def _show_segmentation(args: Dict[str, Any]) -> Dict[str, Any]:
        return ctx.show_segmentation(
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
            visible=bool(args.get("visible", True)),
            fill=bool(args.get("fill", True)),
            outline=bool(args.get("outline", True)),
            opacity=float(args.get("opacity", 0.3)),
            color=args.get("color"),
        )

    add(
        ToolSpec(
            name="show_segmentation",
            description="Show or hide a Slicer segmentation mask, optionally targeting one segment. Use visible=false to unshow it.",
            parameters=S.obj(
                {
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "visible": S.boolean(default=True),
                    "fill": S.boolean(default=True),
                    "outline": S.boolean(default=True),
                    "opacity": S.number(default=0.3, minimum=0.0, maximum=1.0),
                    "color": S.num_vec(description="RGB color in 0..1, e.g. [0,1,0] for green.", min_items=3, max_items=3),
                }
            ),
            handler=_show_segmentation,
            pack="monai_label",
        )
    )

    add(
        ToolSpec(
            name="hide_segmentation",
            description="LLM-friendly alias for show_segmentation(..., visible=false).",
            parameters=S.obj(
                {
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                }
            ),
            handler=lambda args: ctx.hide_segmentation(
                segmentation_id=args.get("segmentation_id"),
                segmentation_name=args.get("segmentation_name"),
                segment_id=args.get("segment_id"),
                segment_name=args.get("segment_name"),
            ),
            pack="monai_label",
        )
    )

    add(
        ToolSpec(
            name="capture_segmentation_overlay",
            description=(
                "Show and center a segmentation, then return a normal Slicer viewer PNG for the requested view. "
                "The mask is rendered by Slicer as the active segmentation/label layer, not composited after export."
            ),
            parameters=S.obj(
                {
                    "view": S.string(default="axial", enum=["axial", "sagittal", "coronal"]),
                    "segmentation_id": S.string(),
                    "segmentation_name": S.string(),
                    "segment_id": S.string(),
                    "segment_name": S.string(),
                    "out_name": S.string(),
                }
            ),
            handler=lambda args: ctx.capture_segmentation_overlay(
                view=args.get("view", "axial"),
                segmentation_id=args.get("segmentation_id"),
                segmentation_name=args.get("segmentation_name"),
                segment_id=args.get("segment_id"),
                segment_name=args.get("segment_name"),
                out_name=args.get("out_name"),
            ),
            pack="monai_label",
        )
    )

    packs.append(
        PackSpec(
            name="monai_label",
            description="Optional MONAI Label / VISTA3D server-backed segmentation tools.",
            tool_names=(
                "monai_label_status",
                "vista3d_label_catalog",
                "vista3d_segment",
                "show_segmentation",
                "hide_segmentation",
                "capture_segmentation_overlay",
            ),
            activation_hint=(
                "Use these tools only when an external MONAI Label/VISTA3D model can help produce a segmentation. "
                "First call monai_label_status and vista3d_label_catalog, then run vista3d_segment. "
                "For NSCLC CT lung tumor use label_prompt=[23]; VISTA3D is a CT model, so brain MRI is experimental."
            ),
        )
    )
