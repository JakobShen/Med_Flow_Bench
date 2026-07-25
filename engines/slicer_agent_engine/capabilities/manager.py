from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ..toolpacks import TOOL_PACK_ACTIVATION_MODE_AUTO, TOOL_PACK_ACTIVATION_MODE_MANUAL
from .schema import array, obj, string
from .types import ACTIVATE_PACK_TOOL_NAME, PackSpec, ToolSpec

logger = logging.getLogger(__name__)


class ToolSurfaceManager:
    """LLM-facing tool surface with core tools always on and configurable optional packs."""

    def __init__(
        self,
        *,
        specs: Sequence[ToolSpec],
        packs: Sequence[PackSpec],
        session_updater: Optional[Callable[..., None]] = None,
        session_logger: Optional[Any] = None,
        initial_active_packs: Optional[Sequence[str]] = None,
        pack_activation_mode: str = TOOL_PACK_ACTIVATION_MODE_MANUAL,
    ) -> None:
        self._specs: Dict[str, ToolSpec] = {spec.name: spec for spec in specs}
        self._packs: Dict[str, PackSpec] = {pack.name: pack for pack in packs}
        self._session_updater = session_updater
        self._session_logger = session_logger
        self._pack_activation_mode = str(pack_activation_mode or TOOL_PACK_ACTIVATION_MODE_MANUAL).strip().lower()
        if self._pack_activation_mode not in {TOOL_PACK_ACTIVATION_MODE_MANUAL, TOOL_PACK_ACTIVATION_MODE_AUTO}:
            raise ValueError(f"Unsupported pack activation mode: {pack_activation_mode!r}")
        self._active_packs: Set[str] = set()

        if self._pack_activation_mode == TOOL_PACK_ACTIVATION_MODE_AUTO and self._packs:
            self.activate_packs(self.available_packs)
            return

        if initial_active_packs:
            self.activate_packs(initial_active_packs)
        else:
            self._sync_state()

    @property
    def active_packs(self) -> List[str]:
        return sorted(self._active_packs)

    @property
    def available_packs(self) -> List[str]:
        return sorted(self._packs.keys())

    @property
    def pack_activation_mode(self) -> str:
        return self._pack_activation_mode

    def _sync_state(self) -> None:
        if self._session_updater is not None:
            self._session_updater(active_tool_packs=self.active_packs)

    def _log_run_event(self, event_type: str, **payload: Any) -> None:
        if self._session_logger is not None and hasattr(self._session_logger, "log_run_event"):
            try:
                self._session_logger.log_run_event(event_type, **payload)
            except Exception:
                logger.exception("Failed to log run event: %s", event_type)

    def _pack_metadata(self) -> Dict[str, Any]:
        return {
            name: {
                "description": pack.description,
                "tool_names": list(pack.tool_names),
            }
            for name, pack in sorted(self._packs.items())
        }

    def _pack_is_provided(self, pack_name: str) -> bool:
        return str(pack_name or "").strip() in self._packs

    def activate_packs(self, pack_names: Sequence[str]) -> Dict[str, Any]:
        requested = [str(x).strip() for x in pack_names if str(x).strip()]
        activated: List[str] = []
        already_active: List[str] = []
        unknown: List[str] = []
        for name in requested:
            if name not in self._packs:
                unknown.append(name)
                continue
            if name in self._active_packs:
                already_active.append(name)
                continue
            self._active_packs.add(name)
            activated.append(name)
        self._sync_state()
        result = {
            "ok": len(unknown) == 0,
            "requested_packs": requested,
            "activated_packs": activated,
            "already_active_packs": already_active,
            "unknown_packs": unknown,
            "active_packs": self.active_packs,
            "activated_tools": self._tool_names_for_packs(activated),
            "available_packs": self._pack_metadata(),
            "pack_activation_mode": self._pack_activation_mode,
        }
        self._log_run_event("pack_activation", **result)
        return result

    def _tool_names_for_packs(self, pack_names: Sequence[str]) -> List[str]:
        names: List[str] = []
        for pack_name in pack_names:
            pack = self._packs.get(pack_name)
            if pack is None:
                continue
            names.extend([tool_name for tool_name in pack.tool_names if tool_name in self._specs])
        return names

    def capability_index_text(self) -> str:
        if not self._packs:
            return ""
        if self._pack_activation_mode == TOOL_PACK_ACTIVATION_MODE_MANUAL:
            lines: List[str] = [
                "Optional tool packs are available on demand. They are NOT active until you call activate_tool_pack.",
                "Only activate a pack when the current task actually needs it.",
            ]
            for pack_name in self.available_packs:
                pack = self._packs[pack_name]
                lines.append(f"- {pack.name}: {pack.description}")
                lines.append(f"  tools: {', '.join(pack.tool_names)}")
                if pack.activation_hint:
                    lines.append(f"  use when: {pack.activation_hint}")
            lines.append("Call activate_tool_pack with pack='...' or packs=['...'] before using tools from that pack.")
            return "\n".join(lines)

        lines = [
            "The tools from the following optional tool packs are already active in this run.",
            "You may call those tools directly; no activation step is required.",
        ]
        for pack_name in self.available_packs:
            pack = self._packs[pack_name]
            lines.append(f"- {pack.name}: {pack.description}")
            lines.append(f"  tools: {', '.join(pack.tool_names)}")
            if pack.activation_hint:
                lines.append(f"  use when: {pack.activation_hint}")
        return "\n".join(lines)

    def _is_model_available(self, spec: ToolSpec) -> bool:
        return spec.expose_to_model and (spec.always_on or spec.pack in self._active_packs)

    def current_tool_schemas(self) -> List[Dict[str, Any]]:
        tools = [spec.canonical_schema() for spec in self._specs.values() if self._is_model_available(spec)]
        if self._packs and self._pack_activation_mode == TOOL_PACK_ACTIVATION_MODE_MANUAL:
            tools.append(
                {
                    "kind": "function",
                    "name": ACTIVATE_PACK_TOOL_NAME,
                    "description": "Activate one or more optional tool packs for the rest of this run. Newly activated tools become callable in later turns.",
                    "parameters": obj({"pack": string(), "packs": array(string())}),
                }
            )
        return tools

    def mcp_tool_specs(self) -> List[ToolSpec]:
        return [
            spec
            for spec in self._specs.values()
            if spec.expose_to_mcp and (spec.always_on or spec.pack == "core" or self._pack_is_provided(spec.pack))
        ]

    def execute_tool_call_json(self, tool_name: str, tool_args_json: str) -> Dict[str, Any]:
        try:
            args = json.loads(tool_args_json or "{}")
        except Exception as e:
            return {"ok": False, "tool": tool_name, "error": f"Invalid tool arguments JSON: {e}"}
        if not isinstance(args, dict):
            return {"ok": False, "tool": tool_name, "error": "Tool arguments must decode to a JSON object."}
        return self.execute_named_tool(tool_name, args, enforce_model_visibility=True)

    def execute_named_tool(self, tool_name: str, args: Dict[str, Any], *, enforce_model_visibility: bool = False) -> Dict[str, Any]:
        try:
            if tool_name == ACTIVATE_PACK_TOOL_NAME:
                if self._pack_activation_mode != TOOL_PACK_ACTIVATION_MODE_MANUAL:
                    return {
                        "ok": False,
                        "tool": tool_name,
                        "error": "Pack activation is not exposed in this run because optional tool pack tools are managed automatically.",
                        "available_packs": self.available_packs,
                    }
                pack_names: List[str] = []
                if args.get("pack"):
                    pack_names.append(str(args["pack"]))
                if isinstance(args.get("packs"), list):
                    pack_names.extend([str(x) for x in args["packs"]])
                if not pack_names:
                    return {
                        "ok": False,
                        "tool": tool_name,
                        "error": "Provide pack='...' or packs=['...'].",
                        "available_packs": self.available_packs,
                    }
                return self.activate_packs(pack_names)

            spec = self._specs.get(tool_name)
            if spec is None:
                return {"ok": False, "tool": tool_name, "error": f"Unknown tool: {tool_name}"}

            if enforce_model_visibility:
                if not spec.expose_to_model:
                    return {"ok": False, "tool": tool_name, "error": f"Tool {tool_name} is not exposed to the model."}
                if not spec.always_on and not self._pack_is_provided(spec.pack):
                    return {
                        "ok": False,
                        "tool": tool_name,
                        "error": f"Tool {tool_name} is not available in this run.",
                        "active_packs": self.active_packs,
                        "available_packs": self.available_packs,
                    }
                if not self._is_model_available(spec):
                    return {
                        "ok": False,
                        "tool": tool_name,
                        "error": f"Tool {tool_name} is not active. Activate pack '{spec.pack}' first.",
                        "active_packs": self.active_packs,
                        "available_packs": self.available_packs,
                    }
            return spec.handler(args)
        except Exception as e:
            logger.exception("Tool failed: %s", tool_name)
            return {"ok": False, "tool": tool_name, "error": str(e)}
