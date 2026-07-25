from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qupath_agent_engine.core.types import ModelDecision, PendingToolResult, StepObservation

if False:  # pragma: no cover
    from qupath_agent_engine.agent.tooling import ToolSpec


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model: str
    max_tokens: int = 1600
    temperature: float = 0.0
    reasoning_effort: str | None = None


class BaseHostModel(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.system_instruction: str = ""
        self.started: bool = False

    @abstractmethod
    def start_session(self, *, system_instruction: str, tools: Sequence["ToolSpec"]) -> None:
        raise NotImplementedError

    @abstractmethod
    def decide_next_step(
        self,
        *,
        observation: StepObservation,
        pending_tool_result: PendingToolResult | None,
        force_final: bool = False,
    ) -> ModelDecision:
        raise NotImplementedError


def encode_image_as_data_url(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def read_image_bytes(path: Path | None) -> tuple[bytes | None, str | None]:
    if path is None or not path.exists():
        return None, None
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "application/octet-stream"
    return path.read_bytes(), mime_type


def safe_serialize(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    for method_name in ("model_dump", "to_json_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
                if isinstance(dumped, dict):
                    serialized = _json_safe(dumped)
                    if isinstance(serialized, dict):
                        return serialized
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            serialized = _json_safe(value)
            if isinstance(serialized, dict):
                return serialized
        except Exception:
            pass
    if isinstance(value, dict):
        serialized = _json_safe(value)
        if isinstance(serialized, dict):
            return serialized
    return {"repr": repr(value)}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return f"<{type(value).__name__}:{len(value)} bytes>"
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except Exception:
            return repr(value)
    if hasattr(value, "to_json_dict") and callable(value.to_json_dict):
        try:
            return value.to_json_dict()
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return repr(value)
    return repr(value)
