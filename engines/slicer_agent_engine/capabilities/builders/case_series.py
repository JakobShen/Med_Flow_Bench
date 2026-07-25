from __future__ import annotations

from typing import Any, Dict

from .. import schema as S
from ..helpers import ToolSurfaceBuildContext
from ..types import PackSpec, ToolSpec


def register_case_series_pack(build: ToolSurfaceBuildContext) -> None:
    ctx = build.ctx
    add = build.add
    packs = build.packs
    case_series_catalog = build.case_series_catalog

    if case_series_catalog is not None:
        def _load_case_series(args: Dict[str, Any]) -> Dict[str, Any]:
            alias = str(args["alias"])
            paths = case_series_catalog.alias_paths(alias)
            if not paths:
                return {
                    "ok": False,
                    "tool": "load_case_series",
                    "error": f"Unknown or empty alias: {alias}",
                    "available_aliases": sorted(case_series_catalog.alias_map.keys()),
                }
            window_preset = args.get("window_preset") or case_series_catalog.alias_window_preset(alias)
            result = ctx.open_case(
                case_series_catalog.data_root,
                preset="ct_pet_chest",
                layout=args.get("layout", "four_up"),
                linked_slices=True,
                interpolation=True,
                wl_auto=window_preset is None,
                baseline=bool(args.get("baseline", True)),
                size=int(args.get("size", 512)),
                clear_scene_first=bool(args.get("clear_scene_first", True)),
                dicom_series_dirs=paths,
                series_display_names=case_series_catalog.alias_display_names(alias),
                active_prefer=case_series_catalog.alias_active_prefer(alias),
                window_preset=window_preset,
            )
            return {**result, "loaded_alias": alias, "loaded_paths": [str(p) for p in paths], "loaded_display_names": case_series_catalog.alias_display_names(alias)}

        add(
            ToolSpec(
                name="load_case_series",
                description="Load one named curated alias group for the current case (for example diagnostic_ct, pet_primary, pet_ct_core, or core). Returns baseline images and the new scene inventory.",
                parameters=S.obj(
                    {
                        "alias": S.string(),
                        "clear_scene_first": S.boolean(default=True),
                        "baseline": S.boolean(default=True),
                        "size": S.integer(default=512),
                        "layout": S.string(default="four_up"),
                        "window_preset": S.string(),
                    },
                    required=["alias"],
                ),
                handler=_load_case_series,
                pack="case_series",
            )
        )
        packs.append(
            PackSpec(
                name="case_series",
                description="Curated series switching for the current case. The available aliases are already described in the prompt, so you do not need a separate listing tool.",
                tool_names=["load_case_series"],
                activation_hint="switching between curated CT/PET alias groups after reading the prompt-provided case brief",
            )
        )
