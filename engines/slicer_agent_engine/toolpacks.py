from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

TOOL_PACK_ACTIVATION_MODE_MANUAL = "manual"
TOOL_PACK_ACTIVATION_MODE_AUTO = "auto"
TOOL_PACK_ACTIVATION_MODES = (
    TOOL_PACK_ACTIVATION_MODE_MANUAL,
    TOOL_PACK_ACTIVATION_MODE_AUTO,
)


@dataclass(frozen=True)
class ToolPackSelection:
    """Selection of optional tool packs for one run.

    ``provided_tool_packs`` uses three states:
    - ``None``: provide all optional packs.
    - ``()`` / empty tuple: provide no optional packs.
    - non-empty tuple: provide only that subset.
    """

    provided_tool_packs: Optional[Tuple[str, ...]] = None
    activation_mode: str = TOOL_PACK_ACTIVATION_MODE_MANUAL
    prompt_hints: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = str(self.activation_mode or TOOL_PACK_ACTIVATION_MODE_MANUAL).strip().lower()
        if mode not in TOOL_PACK_ACTIVATION_MODES:
            raise ValueError(
                f"Unsupported tool pack activation mode: {self.activation_mode!r}. "
                f"Expected one of: {', '.join(TOOL_PACK_ACTIVATION_MODES)}"
            )
        object.__setattr__(self, "activation_mode", mode)

        packs = self.provided_tool_packs
        if packs is not None:
            normalized_packs = tuple(sorted({str(name).strip() for name in packs if str(name).strip()}))
            object.__setattr__(self, "provided_tool_packs", normalized_packs)

        hints = tuple(dict.fromkeys(str(line).strip() for line in self.prompt_hints if str(line).strip()))
        object.__setattr__(self, "prompt_hints", hints)

    @classmethod
    def none(cls) -> "ToolPackSelection":
        return cls(provided_tool_packs=())

    @property
    def has_optional_tool_packs(self) -> bool:
        packs = self.provided_tool_packs
        return packs is None or bool(packs)

    def provides_pack(self, pack_name: str) -> bool:
        name = str(pack_name or "").strip()
        if not name or not self.has_optional_tool_packs:
            return False
        if self.provided_tool_packs is None:
            return True
        return name in self.provided_tool_packs

    def requires_manual_activation(self, pack_name: Optional[str] = None) -> bool:
        if self.activation_mode != TOOL_PACK_ACTIVATION_MODE_MANUAL:
            return False
        return self.has_optional_tool_packs if pack_name is None else self.provides_pack(pack_name)

    def uses_auto_activation(self, pack_name: Optional[str] = None) -> bool:
        if self.activation_mode != TOOL_PACK_ACTIVATION_MODE_AUTO:
            return False
        return self.has_optional_tool_packs if pack_name is None else self.provides_pack(pack_name)

    def pick_text(self, *, none_text: str, manual_text: str, auto_text: str) -> str:
        if not self.has_optional_tool_packs:
            return none_text
        return manual_text if self.activation_mode == TOOL_PACK_ACTIVATION_MODE_MANUAL else auto_text

    def append_prompt_hints(self, text: str) -> str:
        if not self.prompt_hints:
            return text
        base = str(text or "").rstrip()
        hint_block = "\n".join(self.prompt_hints)
        if not base:
            return hint_block
        return f"{base}\n\n{hint_block}"

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "activation_mode": self.activation_mode,
            "has_optional_tool_packs": self.has_optional_tool_packs,
            "provide_all_optional_tool_packs": self.has_optional_tool_packs and self.provided_tool_packs is None,
            "provided_tool_packs": None if self.provided_tool_packs is None else list(self.provided_tool_packs),
            "prompt_hints": list(self.prompt_hints),
        }


__all__ = [
    "TOOL_PACK_ACTIVATION_MODE_AUTO",
    "TOOL_PACK_ACTIVATION_MODE_MANUAL",
    "TOOL_PACK_ACTIVATION_MODES",
    "ToolPackSelection",
]
