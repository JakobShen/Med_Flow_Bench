from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
ACTIVATE_PACK_TOOL_NAME = "activate_tool_pack"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler
    pack: str = "core"
    always_on: bool = False
    expose_to_model: bool = True
    expose_to_mcp: bool = True

    def canonical_schema(self) -> Dict[str, Any]:
        return {
            "kind": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }



@dataclass(frozen=True)
class PackSpec:
    name: str
    description: str
    tool_names: Sequence[str]
    activation_hint: str
