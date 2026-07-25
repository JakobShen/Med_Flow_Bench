from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from qupath_agent_engine.software.qupath.connector import QuPathConnector
from qupath_agent_engine.software.qupath.direct_control import (
    PAN_DIRECTION_OPTIONS,
    PAN_MODE_OPTIONS,
    PAN_UNIT_OPTIONS,
    QUPATH_IMAGE_TYPE_OPTIONS,
    VIEWER_COORD_MAX,
    ZOOM_DIRECTION_OPTIONS,
    ZOOM_UNIT_OPTIONS,
)


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


DIRECT_TOOL_NAMES = frozenset(
    {
        "load_case",
        "open_project",
        "load_image",
        "set_image_type",
        "zoom",
        "pan",
        "focus_qupath",
        "capture_viewer",
        "sleep",
        "close_qupath",
    }
)
GUI_TOOL_NAMES = frozenset({"qupath_gui_act", "qupath_gui_query", "qupath_gui_assert"})
EVIDENCE_TOOL_NAMES = frozenset({"record_current_view_as_evidence", "viewer_point_to_wsi_coordinate"})
ALL_TOOL_NAMES = frozenset(DIRECT_TOOL_NAMES | GUI_TOOL_NAMES | EVIDENCE_TOOL_NAMES)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]
    handler: ToolHandler
    strict: bool = True

    def normalized_schema(self) -> dict[str, Any]:
        schema = dict(self.args_schema or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", False)
        required = schema.get("required")
        if not isinstance(required, list):
            required = []
        if self.strict:
            property_names = [str(name) for name in schema["properties"].keys()]
            seen: set[str] = set()
            required = [
                name
                for name in [*required, *property_names]
                if isinstance(name, str) and not (name in seen or seen.add(name))
            ]
        schema["required"] = required
        return schema

    def gemini_schema(self) -> dict[str, Any]:
        schema = _normalize_gemini_schema(self.args_schema or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        required = schema.get("required")
        if not isinstance(required, list):
            required = []
        schema["required"] = [str(name) for name in required if isinstance(name, str)]
        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.normalized_schema(),
            "strict": bool(self.strict),
        }

    def to_gemini_declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.gemini_schema(),
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        self._tools = list(tools)
        self._by_name = {tool.name: tool for tool in self._tools}

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools]

    def to_gemini_declarations(self) -> list[dict[str, Any]]:
        return [tool.to_gemini_declaration() for tool in self._tools]


_IMAGE_TYPE_DESCRIPTION = ", ".join(f"{key}={label}" for key, label in QUPATH_IMAGE_TYPE_OPTIONS.items())
_ZOOM_UNIT_DESCRIPTION = ", ".join(
    [
        "magnification=absolute target viewer magnification in x (for example 10 means 10x)",
        "relative_factor=multiply the current magnification by the requested factor",
        "steps=QuPath internal zoom steps",
    ]
)
_VIEWER_COORD_MAX_INT = int(VIEWER_COORD_MAX)

HOST_TOOL_GUIDANCE = {
    "load_case": "Recovery/reset only. The case is usually already loaded before the host loop starts.",
    "open_project": (
        "Direct QuPath control. Open a project without using the GUI. Prefer this over GUI clicks for project loading. "
        "Optionally also open one exact project image entry by name."
    ),
    "load_image": (
        "Direct QuPath control. Load a standalone image from a full filesystem path without using the GUI. "
        "Prefer this over GUI clicks for opening a local image."
    ),
    "set_image_type": (
        "Direct QuPath control. Set the current QuPath image type without using the GUI. "
        f"Supported values: {_IMAGE_TYPE_DESCRIPTION}."
    ),
    "zoom": (
        "Direct QuPath viewer control. Zoom the current viewer without using the GUI. "
        "Prefer unit='magnification' when you want an exact target such as 10x. "
        "Use unit='relative_factor' only when you explicitly want a multiplicative change from the current view. "
        f"Supported units: {_ZOOM_UNIT_DESCRIPTION}. "
        "When unit='magnification', direction is ignored. "
        f"You may optionally provide x and y in the range 0..{_VIEWER_COORD_MAX_INT} on the CURRENT viewer screenshot; "
        "QuPath will zoom and then make that selected point the new viewer center. "
        "If x and y are omitted, the current viewer center is used."
    ),
    "pan": (
        "Direct QuPath viewer control. Pan the current viewer without using the GUI. "
        "Use either directional panning or center-based recentering. "
        "For directional panning, provide direction, amount, and unit. "
        "unit='viewport_fraction' means a fraction of the currently visible region; "
        "unit='image_pixels' means full-resolution image pixels. "
        f"For center-based recentering, provide x and y in the range 0..{_VIEWER_COORD_MAX_INT} on the CURRENT viewer screenshot; "
        "that selected point becomes the new viewer center. "
        "mode='directional' or mode='set_center' is optional; if omitted, the tool infers the mode from the provided arguments."
    ),
    "focus_qupath": "Use when QuPath may have lost focus.",
    "qupath_gui_act": (
        "GUI-only action bridge backed by Midscene. Use for buttons, menus, panes, toolbar tools, dialogs, "
        "viewer navigation, scrolling, typing, and other visible QuPath controls when no direct tool exists. "
        "Do not use it for pathology interpretation. Keep each instruction short and concrete."
    ),
    "qupath_gui_query": (
        "GUI-only inspection bridge backed by Midscene. Use only for extra GUI clarification that is not already covered by the "
        "rule-based QuPath GUI state summary. Ask about visible controls, panels, dialogs, tabs, toolbar state, selected tools, "
        "or whether a GUI element is present. Do not ask it to interpret the pathology image."
    ),
    "qupath_gui_assert": (
        "GUI-only boolean check backed by Midscene. Use when you need confirmation that a GUI condition is true before the next action."
    ),
    "capture_viewer": "Force an extra viewer capture if you explicitly need a labeled artifact.",
    "record_current_view_as_evidence": (
        "Record the current QuPath viewer center as one evidence point. Before calling this tool, center the ROI or lesion region "
        "that you want to cite. The benchmark scores only the CURRENT VIEW CENTER POINT. Ensure the current view center lies inside "
        "the ROI you intend to cite. Some benchmarks require an exact label token and/or a short description; provide them when requested."
    ),
    "viewer_point_to_wsi_coordinate": (
        "Convert a point on the CURRENT QuPath viewer screenshot into a full-resolution QuPath/WSI image coordinate. "
        f"Provide x and y normalized to the current screenshot in the range 0..{_VIEWER_COORD_MAX_INT}, with origin at the top-left. "
        "Use the returned x and y in the final answer when a benchmark requires localization output. "
        "Choose a point on the main tissue image rather than the overview thumbnail, scale bar, or text overlays."
    ),
    "sleep": "Small wait for GUI stabilization.",
    "close_qupath": "Close QuPath without saving. Only use if the task explicitly requires it.",
}


BRACS_LABELS_DEFAULT = ("N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC")


def build_tool_registry(
    connector: QuPathConnector,
    *,
    enabled_tool_names: Iterable[str] | None = None,
    disable_gui_agent: bool = True,
    evidence_label_options: Sequence[str] | None = None,
    evidence_requires_supported_label: bool | None = None,
    evidence_requires_description: bool = True,
) -> ToolRegistry:
    handlers = connector.build_tool_handlers()
    enabled_set = _normalize_enabled_tool_names(enabled_tool_names)
    label_source = BRACS_LABELS_DEFAULT if evidence_label_options is None else evidence_label_options
    labels = [str(label).strip().upper() for label in label_source if str(label).strip()]
    if evidence_requires_supported_label is None:
        evidence_requires_supported_label = bool(labels)

    evidence_supported_label_schema: dict[str, Any]
    if labels:
        evidence_supported_label_schema = {
            "type": "string" if evidence_requires_supported_label else ["string", "null"],
            "enum": labels,
            "description": "Exact benchmark label token supported by the current centered region.",
        }
        evidence_label_note = f" Allowed values: {', '.join(labels)}."
    else:
        evidence_supported_label_schema = {
            "type": "string" if evidence_requires_supported_label else ["string", "null"],
            "description": "Optional benchmark label token supported by the current centered region.",
        }
        evidence_label_note = ""

    evidence_description_schema: dict[str, Any] = {
        "type": "string" if evidence_requires_description else ["string", "null"],
        "description": (
            "Short explanation of why the CURRENT centered region supports the selected label or conclusion. "
            "State the key finding plainly, and make sure the current view center is inside the ROI you intend to cite."
        ),
    }
    evidence_required_fields: list[str] = []
    if evidence_requires_supported_label:
        evidence_required_fields.append("supported_label")
    if evidence_requires_description:
        evidence_required_fields.append("description")

    tools = [
        ToolSpec(
            name="load_case",
            description=(
                "Reload the currently configured case in QuPath. "
                f"{HOST_TOOL_GUIDANCE['load_case']}"
            ),
            args_schema={"type": "object", "properties": {}},
            handler=handlers["load_case"],
        ),
        ToolSpec(
            name="open_project",
            description=HOST_TOOL_GUIDANCE["open_project"],
            args_schema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Filesystem path to a QuPath project file (.qpproj) or a project directory.",
                    },
                    "image_name": {
                        "type": ["string", "null"],
                        "description": "Optional exact project image entry name to open immediately after the project is loaded.",
                    },
                },
                "required": ["project_path", "image_name"],
            },
            handler=handlers["open_project"],
        ),
        ToolSpec(
            name="load_image",
            description=HOST_TOOL_GUIDANCE["load_image"],
            args_schema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Full filesystem path to the image that should be opened directly in QuPath.",
                    }
                },
                "required": ["image_path"],
            },
            handler=handlers["load_image"],
        ),
        ToolSpec(
            name="set_image_type",
            description=HOST_TOOL_GUIDANCE["set_image_type"],
            args_schema={
                "type": "object",
                "properties": {
                    "image_type": {
                        "type": "string",
                        "enum": list(QUPATH_IMAGE_TYPE_OPTIONS.keys()),
                        "description": (
                            "The exact QuPath image type enum value to apply. "
                            f"Options: {_IMAGE_TYPE_DESCRIPTION}."
                        ),
                    }
                },
                "required": ["image_type"],
            },
            handler=handlers["set_image_type"],
        ),
        ToolSpec(
            name="zoom",
            description=HOST_TOOL_GUIDANCE["zoom"],
            args_schema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": list(ZOOM_DIRECTION_OPTIONS),
                        "description": (
                            "Zoom direction. Required by the schema for compatibility. "
                            "Ignored when unit='magnification'."
                        ),
                    },
                    "amount": {
                        "type": "number",
                        "description": (
                            "If unit='magnification', this is the target viewer magnification in x. "
                            "If unit='relative_factor', this is a multiplier > 1.0. "
                            "If unit='steps', this is an integer-like count >= 1."
                        ),
                    },
                    "unit": {
                        "type": "string",
                        "enum": list(ZOOM_UNIT_OPTIONS),
                        "description": f"Zoom amount unit. Options: {_ZOOM_UNIT_DESCRIPTION}.",
                    },
                    "x": {
                        "type": "number",
                        "description": (
                            f"Optional normalized horizontal coordinate on the CURRENT viewer screenshot. "
                            f"0 is the left edge and {_VIEWER_COORD_MAX_INT} is the right edge. "
                            "If omitted, QuPath uses the current viewer center."
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            f"Optional normalized vertical coordinate on the CURRENT viewer screenshot. "
                            f"0 is the top edge and {_VIEWER_COORD_MAX_INT} is the bottom edge. "
                            "If omitted, QuPath uses the current viewer center."
                        ),
                    },
                },
                "required": ["direction", "amount", "unit"],
            },
            handler=handlers["zoom"],
            strict=False,
        ),
        ToolSpec(
            name="pan",
            description=HOST_TOOL_GUIDANCE["pan"],
            args_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(PAN_MODE_OPTIONS),
                        "description": (
                            "Optional pan mode. Use 'directional' for direction+amount+unit panning, "
                            "or 'set_center' to make the selected viewer point the new center. "
                            "If omitted, the tool infers the mode from the provided arguments."
                        ),
                    },
                    "direction": {
                        "type": "string",
                        "enum": list(PAN_DIRECTION_OPTIONS),
                        "description": "Pan direction. Required for mode='directional'.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Positive pan amount. The interpretation depends on unit. Required for mode='directional'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": list(PAN_UNIT_OPTIONS),
                        "description": "Pan amount unit: 'viewport_fraction' or 'image_pixels'. Required for mode='directional'.",
                    },
                    "x": {
                        "type": "number",
                        "description": (
                            f"Normalized horizontal coordinate on the CURRENT viewer screenshot. "
                            f"0 is the left edge and {_VIEWER_COORD_MAX_INT} is the right edge. "
                            "Required for mode='set_center'."
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            f"Normalized vertical coordinate on the CURRENT viewer screenshot. "
                            f"0 is the top edge and {_VIEWER_COORD_MAX_INT} is the bottom edge. "
                            "Required for mode='set_center'."
                        ),
                    },
                },
            },
            handler=handlers["pan"],
            strict=False,
        ),
        ToolSpec(
            name="focus_qupath",
            description=(
                "Bring the QuPath application to the foreground. "
                f"{HOST_TOOL_GUIDANCE['focus_qupath']}"
            ),
            args_schema={"type": "object", "properties": {}},
            handler=handlers["focus_qupath"],
        ),
        ToolSpec(
            name="qupath_gui_act",
            description=HOST_TOOL_GUIDANCE["qupath_gui_act"],
            args_schema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "A short GUI instruction for QuPath, usually one small bounded action sequence.",
                    }
                },
                "required": ["instruction"],
            },
            handler=handlers["qupath_gui_act"],
        ),
        ToolSpec(
            name="qupath_gui_query",
            description=HOST_TOOL_GUIDANCE["qupath_gui_query"],
            args_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A GUI-only question about visible QuPath controls or window state.",
                    }
                },
                "required": ["question"],
            },
            handler=handlers["qupath_gui_query"],
        ),
        ToolSpec(
            name="qupath_gui_assert",
            description=HOST_TOOL_GUIDANCE["qupath_gui_assert"],
            args_schema={
                "type": "object",
                "properties": {
                    "assertion": {
                        "type": "string",
                        "description": "A GUI-only boolean condition that should hold in the visible QuPath window.",
                    }
                },
                "required": ["assertion"],
            },
            handler=handlers["qupath_gui_assert"],
        ),
        ToolSpec(
            name="capture_viewer",
            description=HOST_TOOL_GUIDANCE["capture_viewer"],
            args_schema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": ["string", "null"],
                        "description": "Optional label for the saved viewer capture artifact.",
                    }
                },
                "required": ["label"],
            },
            handler=handlers["capture_viewer"],
        ),
        ToolSpec(
            name="viewer_point_to_wsi_coordinate",
            description=HOST_TOOL_GUIDANCE["viewer_point_to_wsi_coordinate"],
            args_schema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": (
                            f"Normalized horizontal coordinate on the CURRENT viewer screenshot. "
                            f"0 is the left edge and {_VIEWER_COORD_MAX_INT} is the right edge."
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            f"Normalized vertical coordinate on the CURRENT viewer screenshot. "
                            f"0 is the top edge and {_VIEWER_COORD_MAX_INT} is the bottom edge."
                        ),
                    },
                },
                "required": ["x", "y"],
            },
            handler=handlers["viewer_point_to_wsi_coordinate"],
        ),
        ToolSpec(
            name="record_current_view_as_evidence",
            description=(
                HOST_TOOL_GUIDANCE["record_current_view_as_evidence"]
                + evidence_label_note
            ),
            args_schema={
                "type": "object",
                "properties": {
                    "supported_label": evidence_supported_label_schema,
                    "description": evidence_description_schema,
                },
                "required": evidence_required_fields,
            },
            handler=handlers["record_current_view_as_evidence"],
        ),
        ToolSpec(
            name="sleep",
            description=HOST_TOOL_GUIDANCE["sleep"],
            args_schema={
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Number of seconds to wait.",
                    }
                },
                "required": ["seconds"],
            },
            handler=handlers["sleep"],
        ),
        ToolSpec(
            name="close_qupath",
            description=HOST_TOOL_GUIDANCE["close_qupath"],
            args_schema={
                "type": "object",
                "properties": {
                    "save": {
                        "type": ["boolean", "null"],
                        "description": "Whether to save before closing. v1 currently supports close without save only.",
                    }
                },
                "required": ["save"],
            },
            handler=handlers["close_qupath"],
        ),
    ]

    if disable_gui_agent:
        tools = [tool for tool in tools if tool.name not in GUI_TOOL_NAMES]
    if enabled_set is not None:
        tools = [tool for tool in tools if tool.name in enabled_set]
    return ToolRegistry(tools)


def infer_tool_category(tool_name: str | None) -> str:
    name = str(tool_name or "").strip()
    if name in DIRECT_TOOL_NAMES:
        return "direct"
    if name in GUI_TOOL_NAMES:
        return "gui"
    if name in EVIDENCE_TOOL_NAMES:
        return "evidence"
    return "other"


def _normalize_enabled_tool_names(enabled_tool_names: Iterable[str] | None) -> set[str] | None:
    if enabled_tool_names is None:
        return None
    normalized = {str(name).strip() for name in enabled_tool_names if str(name).strip()}
    unknown = sorted(name for name in normalized if name not in ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool names: {unknown}")
    return normalized


def _normalize_gemini_schema(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {
            str(key): _normalize_gemini_schema(item)
            for key, item in value.items()
            if str(key) != "additionalProperties"
        }
        type_value = normalized.get("type")
        if isinstance(type_value, list):
            non_null_types = [item for item in type_value if str(item).lower() != "null"]
            has_null = len(non_null_types) != len(type_value)
            if len(non_null_types) == 1:
                normalized["type"] = non_null_types[0]
                if has_null:
                    normalized["nullable"] = True
            elif len(non_null_types) == 0:
                normalized["type"] = "null"
            else:
                raise ValueError(
                    f"Gemini function schema does not support union type lists: {type_value!r}"
                )
        return normalized
    if isinstance(value, list):
        return [_normalize_gemini_schema(item) for item in value]
    return value
