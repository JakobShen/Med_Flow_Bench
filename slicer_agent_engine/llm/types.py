from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class TextPrompt:
    """Provider-neutral user text input."""

    text: str


@dataclass(frozen=True)
class ImagePrompt:
    """Provider-neutral user image input with an optional caption."""

    path: Path
    text: str = ""
    mime: str = "image/png"

    def resolved_path(self) -> Path:
        return self.path.expanduser().resolve()


PromptInput = Union[TextPrompt, ImagePrompt]


@dataclass(frozen=True)
class ToolCall:
    """Canonical tool call emitted by a model adapter."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ModelTurnResult:
    """Canonical model turn result consumed by the provider-neutral tool loop."""

    output_text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    response_id: Optional[str] = None
    output_item_types: List[str] = field(default_factory=list)
    raw_response: Optional[Any] = None


def make_text_prompt(text: str) -> TextPrompt:
    return TextPrompt(text=str(text))


def make_image_prompt(path: Union[str, Path], *, text: str = "", mime: str = "image/png") -> ImagePrompt:
    return ImagePrompt(path=Path(path), text=str(text), mime=str(mime))


CanonicalToolSchema = Dict[str, Any]
