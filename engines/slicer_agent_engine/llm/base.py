from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from .types import CanonicalToolSchema, ModelTurnResult, PromptInput, ToolCall


class BaseModelAdapter(ABC):
    """Provider-neutral adapter interface.

    Future providers only need to implement this small boundary:
    1. convert provider-neutral prompt inputs + canonical tool schemas to the provider request format,
    2. parse provider responses into canonical tool calls / final text,
    3. append tool results back into the provider conversation state.
    """

    provider_name: str = "unknown"

    @abstractmethod
    def start_conversation(self, *, inputs: Sequence[PromptInput]) -> Any:
        """Create provider-specific mutable conversation state from provider-neutral inputs."""

    @abstractmethod
    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: Any,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        """Execute one provider request and mutate state with provider-native response items."""

    @abstractmethod
    def append_tool_result(
        self,
        *,
        state: Any,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        """Append a tool result to provider state. Return the number of attached image artifacts."""

    @abstractmethod
    def export_state(self, state: Any) -> List[Any]:
        """Return a serializable snapshot of the provider conversation state."""

    def tool_display_names(
        self,
        *,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        names: List[str] = []
        for tool in tool_schemas:
            if not isinstance(tool, dict):
                continue
            if tool.get("name"):
                names.append(str(tool.get("name")))
            else:
                names.append(f"[{tool.get('kind', 'tool')}]")
        for tool in extra_model_tools or []:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool.get("name")))
            elif isinstance(tool, dict) and tool.get("type"):
                names.append(f"[{tool.get('type')}]")
            else:
                names.append(str(tool))
        return names
