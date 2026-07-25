from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .capabilities import ToolSurfaceBuildContext, ToolSurfaceManager
from .capabilities.builders import (
    register_case_series_pack,
    register_core_tools,
    register_monai_label_pack,
    register_optional_packs,
    register_registration_pack,
    register_segmentation_pack,
)
from .capabilities.types import ACTIVATE_PACK_TOOL_NAME as _ACTIVATE_PACK_TOOL_NAME
from .capabilities.types import PackSpec, ToolHandler, ToolSpec
from .toolpacks import TOOL_PACK_ACTIVATION_MODE_MANUAL
from .tools import ToolContext

if TYPE_CHECKING:
    from .load_helper import CaseSeriesCatalog


# Backward-compatible helper exports. New code should prefer ToolSurfaceBuildContext.
def _json_object(properties: Dict[str, Any], required: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return ToolSurfaceBuildContext.json_object(properties, required)


def _resolve_case_path(path_value: Any) -> Any:
    return ToolSurfaceBuildContext.resolve_case_path(path_value)


def _scene_inventory_payload(ctx: ToolContext) -> Dict[str, Any]:
    build = ToolSurfaceBuildContext(ctx=ctx, specs=[], packs=[])
    return build.scene_inventory_payload()


def build_default_tool_surface(
    *,
    ctx: ToolContext,
    case_series_catalog: Optional["CaseSeriesCatalog"] = None,
    initial_active_packs: Optional[Sequence[str]] = None,
    optional_pack_names: Optional[Sequence[str]] = None,
    pack_activation_mode: str = TOOL_PACK_ACTIVATION_MODE_MANUAL,
) -> ToolSurfaceManager:
    """Build the default tool surface.

    ``optional_pack_names`` uses three modes:
    - ``None``: provide all optional packs (current library default).
    - ``[]`` / empty sequence: provide no optional packs.
    - non-empty sequence: provide only that subset.
    """

    specs: List[ToolSpec] = []
    packs: List[PackSpec] = []
    build = ToolSurfaceBuildContext(
        ctx=ctx,
        specs=specs,
        packs=packs,
        case_series_catalog=case_series_catalog,
    )

    register_core_tools(build)
    register_optional_packs(build)
    register_registration_pack(build)
    register_segmentation_pack(build)
    register_monai_label_pack(build)
    register_case_series_pack(build)

    if optional_pack_names is not None:
        selected = {str(name).strip() for name in optional_pack_names if str(name).strip()}
        packs = [pack for pack in packs if pack.name in selected]

    return ToolSurfaceManager(
        specs=specs,
        packs=packs,
        session_updater=ctx.session.update_state,
        session_logger=ctx.session,
        initial_active_packs=initial_active_packs,
        pack_activation_mode=pack_activation_mode,
    )


__all__ = [
    "PackSpec",
    "ToolHandler",
    "ToolSpec",
    "ToolSurfaceManager",
    "build_default_tool_surface",
    "_ACTIVATE_PACK_TOOL_NAME",
    "_json_object",
    "_resolve_case_path",
    "_scene_inventory_payload",
]
