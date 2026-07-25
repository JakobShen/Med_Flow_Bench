from __future__ import annotations

from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import ToolSpec


def register_core_tools(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    _resolve_case_path = build.resolve_case_path
    _scene_inventory_payload = lambda _ctx=None: build.scene_inventory_payload()

    add(
        ToolSpec(
            name="ping",
            description="Ping the running Slicer WebServer.",
            parameters=S.obj({}),
            handler=lambda args: ctx.ping(),
            always_on=True,
        )
    )

    def _open_case(args: Dict[str, Any]) -> Dict[str, Any]:
        return ctx.open_case(
            _resolve_case_path(args["dicom_dir"]),
            preset=args.get("preset", "mri_brain"),
            layout=args.get("layout", "four_up"),
            linked_slices=bool(args.get("linked_slices", True)),
            interpolation=bool(args.get("interpolation", True)),
            wl_auto=bool(args.get("wl_auto", True)),
            baseline=bool(args.get("baseline", True)),
            size=int(args.get("size", 512)),
            clear_scene_first=bool(args.get("clear_scene_first", True)),
            include_segmentations=bool(args.get("include_segmentations", False)),
        )

    add(
        ToolSpec(
            name="open_case",
            description="(Re)open a case path with the standard viewer preset. Returns baseline images plus the currently loaded scene inventory.",
            parameters=S.obj(
                {
                    "dicom_dir": S.string(description="Case path. DICOM directory or NIfTI file/folder."),
                    "preset": S.string(default="mri_brain"),
                    "layout": S.string(default="four_up"),
                    "linked_slices": S.boolean(default=True),
                    "interpolation": S.boolean(default=True),
                    "wl_auto": S.boolean(default=True),
                    "baseline": S.boolean(default=True),
                    "size": S.integer(default=512),
                    "clear_scene_first": S.boolean(default=True),
                    "include_segmentations": S.boolean(default=False),
                },
                required=["dicom_dir"],
            ),
            handler=_open_case,
            always_on=True,
        )
    )

    def _select_volume(args: Dict[str, Any]) -> Dict[str, Any]:
        volume_id = args.get("volume_id")
        volume_name = args.get("volume_name")
        if not volume_id and not volume_name:
            return {
                "ok": False,
                "tool": "select_volume",
                "error": "Provide volume_id or volume_name from the current scene inventory.",
                "required_any_of": ["volume_id", "volume_name"],
            }
        select_res = ctx.select_volume(volume_id=volume_id, volume_name=volume_name)
        active_volume_id = select_res.get("active_volume_id") if isinstance(select_res, dict) else None
        active_volume_name = select_res.get("active_volume_name") if isinstance(select_res, dict) else None
        recover_res = ctx.recover_standard_views(
            volume_id=active_volume_id,
            volume_name=active_volume_name,
            centered=True,
            fit=True,
        )
        observation = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix="select_volume")
        inventory = _scene_inventory_payload(ctx)
        return {
            "ok": True,
            "select": select_res,
            "recover": recover_res,
            "observation": observation,
            "png_paths": observation.get("png_paths") if isinstance(observation, dict) else None,
            **inventory,
        }

    add(
        ToolSpec(
            name="select_volume",
            description="Select one of the already loaded image volumes by id or name from the current scene inventory, recover standard views, then return an updated 3-plane observation and scene inventory. You must provide volume_id or volume_name.",
            parameters=S.obj(
                {
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                    "size": S.integer(default=512),
                }
            ),
            handler=_select_volume,
            always_on=True,
        )
    )

    add(
        ToolSpec(
            name="observe",
            description="Observe the current standard 3-plane views (axial, sagittal, coronal).",
            parameters=S.obj(
                {
                    "size": S.integer(default=512),
                    "out_prefix": S.string(default="obs"),
                }
            ),
            handler=lambda args: ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix=args.get("out_prefix", "obs")),
            always_on=True,
        )
    )

    image_coords_parameters = S.obj(
        {
            "view": S.string(description="axial, sagittal, or coronal; use the exact view whose slice image you inspected"),
            "bbox_1000": S.bbox_1000(description="Mutually exclusive with point_1000. Bounding box [x_min,y_min,x_max,y_max] in integer 0..999 coordinates. If you send bbox_1000, do not send point_1000; the tool will return bbox-corner RAS values plus center_ras."),
            "point_1000": S.point_1000(description="Mutually exclusive with bbox_1000. Point [x,y] in integer 0..999 coordinates. If you send point_1000, do not send bbox_1000."),
            "size": S.integer(default=512),
        },
        required=["view"],
    )

    add(
        ToolSpec(
            name="image_coords_to_ras",
            description=(
                "Convert image-space localization coordinates from the exact slice image you just inspected into physical RAS coordinates in mm. "
                "Primary contract: provide view plus exactly ONE of bbox_1000=[x_min,y_min,x_max,y_max] or point_1000=[x,y], using integer coordinates in fixed 0..999 space with origin at the top-left of the exact cropped slice image (outer black/transparent border trimmed). Never send both in the same call. "
                "If you want the center of a box, send only bbox_1000; the tool returns the four bbox-corner RAS points and center_ras automatically. Use this before RAS-based measurement tools when you only know image coordinates."
            ),
            parameters=image_coords_parameters,
            handler=lambda args: ctx.image_coords_to_ras(
                view=args["view"],
                bbox_1000=args.get("bbox_1000"),
                point_1000=args.get("point_1000"),
                size=int(args.get("size", 512)),
            ),
            always_on=True,
        )
    )

    add(
        ToolSpec(
            name="get_viewer_state",
            description="Return the current viewer state snapshot (active volume, slice offsets, fusion state, thick slab state, etc.).",
            parameters=S.obj({}),
            handler=lambda args: ctx.get_viewer_state(),
            always_on=True,
        )
    )

    def _wl(args: Dict[str, Any]) -> Dict[str, Any]:
        wl_res = ctx.set_window_level(auto=args.get("auto"), window=args.get("window"), level=args.get("level"))
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        return {"ok": True, "wl": wl_res, "observation": obs, "png_path": obs.get("png_path")}

    add(
        ToolSpec(
            name="wl",
            description="Set window/level and return an updated observation PNG.",
            parameters=S.obj(
                {
                    "auto": S.boolean(),
                    "window": S.number(),
                    "level": S.number(),
                    "view": S.string(default="axial"),
                    "size": S.integer(default=512),
                }
            ),
            handler=_wl,
            always_on=True,
        )
    )

    def _window_preset(args: Dict[str, Any]) -> Dict[str, Any]:
        preset_res = ctx.apply_window_preset(
            preset=args["preset"],
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
        )
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        return {"ok": True, "window_preset": preset_res, "observation": obs, "png_path": obs.get("png_path")}

    add(
        ToolSpec(
            name="window_preset",
            description="Apply a modality-specific window preset (ct_lung, ct_mediastinal, ct_bone, pet_auto) and return an updated observation.",
            parameters=S.obj(
                {
                    "preset": S.string(),
                    "view": S.string(default="axial"),
                    "size": S.integer(default=512),
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                },
                required=["preset"],
            ),
            handler=_window_preset,
            always_on=True,
        )
    )

    def _zoom(args: Dict[str, Any]) -> Dict[str, Any]:
        zoom_res = ctx.zoom_slice_relative(view=args.get("view", "axial"), factor=float(args.get("factor", 1.5)))
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        return {"ok": True, "zoom": zoom_res, "observation": obs, "png_path": obs.get("png_path")}

    add(
        ToolSpec(
            name="zoom",
            description="Zoom the current slice view relative to its current field of view and return an updated observation.",
            parameters=S.obj(
                {
                    "view": S.string(default="axial"),
                    "factor": S.number(default=1.5),
                    "size": S.integer(default=512),
                }
            ),
            handler=_zoom,
            always_on=True,
        )
    )

    def _fit(args: Dict[str, Any]) -> Dict[str, Any]:
        fit_res = ctx.fit_slice(view=args.get("view", "axial"))
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        return {"ok": True, "fit": fit_res, "observation": obs, "png_path": obs.get("png_path")}

    add(
        ToolSpec(
            name="fit",
            description="Fit the selected slice view to the full volume and return an updated observation.",
            parameters=S.obj(
                {
                    "view": S.string(default="axial"),
                    "size": S.integer(default=512),
                }
            ),
            handler=_fit,
            always_on=True,
        )
    )

    def _set_slice_offset(args: Dict[str, Any]) -> Dict[str, Any]:
        set_res = ctx.set_slice_offset(view=args.get("view", "axial"), offset=float(args["offset"]))
        obs = ctx.observe_view(view=args.get("view", "axial"), size=int(args.get("size", 512)))
        try:
            offset_range = ctx.get_slice_offset_range(view=args.get("view", "axial"))
        except Exception:
            offset_range = None
        return {
            "ok": True,
            "set_slice_offset": set_res,
            "slice_offset_range": offset_range,
            "observation": obs,
            "png_path": obs.get("png_path"),
        }

    add(
        ToolSpec(
            name="set_slice_offset",
            description="Set the slice position directly in mm for axial/sagittal/coronal. Use this when scrolling moves the view to a blank slice or you need to recover a specific S/A/L position.",
            parameters=S.obj(
                {
                    "view": S.string(default="axial"),
                    "offset": S.number(),
                    "size": S.integer(default=512),
                },
                required=["view", "offset"],
            ),
            handler=_set_slice_offset,
            always_on=True,
        )
    )

    add(
        ToolSpec(
            name="get_slice_offset_range",
            description="Return the valid min/max slice offset range in mm for a view. Useful before setting S/A/L slice positions directly.",
            parameters=S.obj({"view": S.string(default="axial")}, required=["view"]),
            handler=lambda args: ctx.get_slice_offset_range(view=args.get("view", "axial")),
            always_on=True,
        )
    )

    def _jump(args: Dict[str, Any]) -> Dict[str, Any]:
        jump_res = ctx.jump_to_ras(ras=args["ras"], centered=bool(args.get("centered", True)))
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix=args.get("out_prefix", "jump"))
        any_png = None
        if isinstance(obs, dict):
            png_paths = obs.get("png_paths")
            if isinstance(png_paths, dict):
                any_png = next(iter(png_paths.values()), None)
        return {
            "ok": True,
            "jump": jump_res,
            "observation": obs,
            "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None,
            "png_path": any_png,
        }

    add(
        ToolSpec(
            name="jump",
            description="Jump the linked views to a trusted physical RAS point (mm) and return the updated standard 3-plane observation. This is not a panning tool. Do not use it for casual left/right/up/down exploration; if views become off-center, prefer recover_standard_views instead.",
            parameters=S.obj(
                {
                    "ras": S.num_vec(),
                    "centered": S.boolean(default=True),
                    "size": S.integer(default=512),
                    "out_prefix": S.string(default="jump"),
                },
                required=["ras"],
            ),
            handler=_jump,
            always_on=True,
        )
    )

    def _recover_standard_views(args: Dict[str, Any]) -> Dict[str, Any]:
        recover_res = ctx.recover_standard_views(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
            centered=bool(args.get("centered", True)),
            fit=bool(args.get("fit", True)),
        )
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix=args.get("out_prefix", "recover_standard_views"))
        return {
            "ok": True,
            "recover": recover_res,
            "observation": obs,
            "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None,
            "viewer_state": recover_res.get("viewer_state") if isinstance(recover_res, dict) else None,
        }

    add(
        ToolSpec(
            name="recover_standard_views",
            description="Restore the standard axial/sagittal/coronal views around the center of the active or specified volume, then refit them. Use this when scroll/jump/pan leaves the chest off-center or partially out of view.",
            parameters=S.obj(
                {
                    "volume_id": S.string(),
                    "volume_name": S.string(),
                    "centered": S.boolean(default=True),
                    "fit": S.boolean(default=True),
                    "size": S.integer(default=512),
                    "out_prefix": S.string(default="recover_standard_views"),
                }
            ),
            handler=_recover_standard_views,
            always_on=True,
        )
    )

    def _set_layout(args: Dict[str, Any]) -> Dict[str, Any]:
        layout_res = ctx.set_layout(layout=args.get("layout", "four_up"))
        obs = ctx.observe_standard_views(size=int(args.get("size", 512)), out_prefix="layout")
        return {"ok": True, "layout": layout_res, "observation": obs, "png_paths": obs.get("png_paths") if isinstance(obs, dict) else None}

    add(
        ToolSpec(
            name="set_layout",
            description="Set a standard Slicer layout and return the updated standard observation.",
            parameters=S.obj(
                {
                    "layout": S.string(default="four_up"),
                    "size": S.integer(default=512),
                }
            ),
            handler=_set_layout,
            always_on=True,
        )
    )

    add(
        ToolSpec(
            name="scroll",
            description="Scroll through slices. output='keyframes' returns sampled PNGs and a contact sheet with per-frame physical slice-position labels in mm when available; output='video' returns an MP4. If analyze=true and output='video', Gemini analysis is run in one step.",
            parameters=S.obj(
                {
                    "view": S.string(default="axial"),
                    "orientation": S.string(default="axial"),
                    "start": S.number(default=0.0),
                    "end": S.number(default=1.0),
                    "n_frames": S.integer(default=12),
                    "size": S.integer(default=512),
                    "output": S.string(default="keyframes", enum=["keyframes", "video"]),
                    "fps": S.integer(default=12),
                    "analyze": S.boolean(default=False),
                    "prompt": S.string(),
                    "auto_prompt": S.boolean(default=True),
                }
            ),
            handler=lambda args: ctx.scroll_sweep(
                view=args.get("view", "axial"),
                orientation=args.get("orientation", "axial"),
                start=float(args.get("start", 0.0)),
                end=float(args.get("end", 1.0)),
                n_frames=int(args.get("n_frames", 12)),
                size=int(args.get("size", 512)),
                output=args.get("output", "keyframes"),
                fps=int(args.get("fps", 12)),
                analyze=bool(args.get("analyze", False)),
                prompt=args.get("prompt"),
                auto_prompt=bool(args.get("auto_prompt", True)),
                out_prefix="scroll",
            ),
            always_on=True,
        )
    )

    add(
        ToolSpec(
            name="gemini_analyze_video",
            description="Analyze an existing MP4 cine with Gemini and return a concise text summary. Use this only when you already have a video artifact and need explicit video-model interpretation.",
            parameters=S.obj(
                {
                    "video_path": S.string(),
                    "prompt": S.string(),
                },
                required=["video_path", "prompt"],
            ),
            handler=lambda args: (
                {"ok": False, "error": "Gemini not configured (set GEMINI_API_KEY)."}
                if ctx.gemini is None
                else {"ok": True, "text": ctx.gemini.analyze_mp4(video_path=args["video_path"], prompt=args["prompt"]), "model": ctx.gemini.model}
            ),
            always_on=True,
            expose_to_mcp=False,
        )
    )
