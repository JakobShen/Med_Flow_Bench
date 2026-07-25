from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from ..context_briefs import format_scene_inventory_for_prompt
from ..tools import ToolContext
from .schema import obj
from .types import PackSpec, ToolSpec

if TYPE_CHECKING:
    from ..load_helper import CaseSeriesCatalog


@dataclass
class ToolSurfaceBuildContext:
    ctx: ToolContext
    specs: List[ToolSpec]
    packs: List[PackSpec]
    case_series_catalog: Optional["CaseSeriesCatalog"] = None

    def add(self, spec: ToolSpec) -> None:
        self.specs.append(spec)

    @staticmethod
    def json_object(properties: Dict[str, Any], required: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return obj(properties, required=required)

    @staticmethod
    def resolve_case_path(path_value: Any) -> Any:
        dicom_dir = path_value
        try:
            p = Path(str(dicom_dir)).expanduser()
            if not p.is_absolute():
                case_root = os.environ.get("SLICER_CASE_ROOT")
                if case_root:
                    candidate = (Path(case_root).expanduser() / p).resolve()
                    if candidate.exists():
                        dicom_dir = str(candidate)
        except Exception:
            pass
        return dicom_dir

    def scene_inventory_payload(self) -> Dict[str, Any]:
        try:
            volumes = self.ctx.client.list_volumes()
        except Exception as e:
            return {"scene_inventory_error": str(e)}
        active_volume_id = self.ctx.session.state.get("active_volume_id")
        active_volume_name = self.ctx.session.state.get("active_volume_name")
        return {
            "scene_inventory": volumes,
            "scene_inventory_text": format_scene_inventory_for_prompt(
                volumes,
                active_volume_id=active_volume_id,
                active_volume_name=active_volume_name,
            ),
        }

    def capture_overlay_standard_views(self, *, size: int = 512, out_prefix: str = "segmentation") -> Dict[str, Any]:
        png_paths: Dict[str, str] = {}
        any_png: Optional[str] = None
        for view_name in ("axial", "sagittal", "coronal"):
            cap = self.ctx.observe_view(view=view_name, size=int(size), out_name=f"{out_prefix}_{view_name}_overlay.png")
            if isinstance(cap, dict) and cap.get("png_path"):
                png_paths[view_name] = str(cap["png_path"])
                if any_png is None:
                    any_png = str(cap["png_path"])
        return {"png_paths": png_paths, "png_path": any_png}
