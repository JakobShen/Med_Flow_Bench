from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..load_helper import CaseSeriesCatalog, format_catalog_for_prompt
from ..tools import ToolContext


def build_case_series_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "list_case_series",
            "description": "List clinically curated series aliases for the current case. Use this when the raw folder structure is messy.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "load_case_series",
            "description": "Load one named series alias or alias group for the current case (for example: diagnostic_ct, pet_primary, pet_ct_core, core). Returns baseline images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alias": {"type": "string"},
                    "clear_scene_first": {"type": "boolean", "default": True},
                    "baseline": {"type": "boolean", "default": True},
                    "size": {"type": "integer", "default": 512},
                    "layout": {"type": "string", "default": "four_up"},
                    "window_preset": {"type": "string"},
                },
                "required": ["alias"],
            },
        },
    ]


def make_case_series_tool_executor(*, ctx: ToolContext, catalog: CaseSeriesCatalog):
    def run_tool(name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if name == "list_case_series":
            return {"ok": True, **catalog.to_summary(), "catalog_text": format_catalog_for_prompt(catalog)}

        if name == "load_case_series":
            alias = str(args["alias"])
            paths = catalog.alias_paths(alias)
            if not paths:
                return {"ok": False, "tool": name, "error": f"Unknown/empty alias: {alias}", "available_aliases": sorted(catalog.alias_map.keys())}
            window_preset = args.get("window_preset")
            if not window_preset:
                window_preset = catalog.alias_window_preset(alias)
            result = ctx.open_case(
                catalog.data_root,
                preset="ct_pet_chest",
                layout=args.get("layout", "four_up"),
                linked_slices=True,
                interpolation=True,
                wl_auto=window_preset is None,
                baseline=bool(args.get("baseline", True)),
                size=int(args.get("size", 512)),
                clear_scene_first=bool(args.get("clear_scene_first", True)),
                dicom_series_dirs=paths,
                active_prefer=catalog.alias_active_prefer(alias),
                window_preset=window_preset,
            )
            result = {**result, "loaded_alias": alias, "loaded_paths": [str(p) for p in paths]}
            return result

        return None

    return run_tool
