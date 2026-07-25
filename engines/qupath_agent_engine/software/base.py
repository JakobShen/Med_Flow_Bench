from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class SoftwareConnector(ABC):
    @abstractmethod
    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        raise NotImplementedError
