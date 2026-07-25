from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .llm.types import ImagePrompt, TextPrompt, make_image_prompt, make_text_prompt


def format_scene_inventory_for_prompt(
    volumes: Sequence[Dict[str, Any]],
    *,
    active_volume_id: Optional[str] = None,
    active_volume_name: Optional[str] = None,
    heading: str = "Currently loaded scene volumes",
) -> str:
    """Return a compact prompt-friendly description of the loaded scene inventory."""

    lines: List[str] = [f"{heading}:"]
    if not volumes:
        lines.append("- (none)")
        return "\n".join(lines)

    for idx, vol in enumerate(volumes, start=1):
        if not isinstance(vol, dict):
            lines.append(f"- {vol}")
            continue
        vid = str(vol.get("id") or "").strip()
        name = str(vol.get("name") or vid or f"volume_{idx}").strip()
        marker = "*" if (active_volume_id and vid == active_volume_id) or (active_volume_name and name == active_volume_name) else "-"
        if vid:
            lines.append(f"{marker} {idx}. {name} [id={vid}]")
        else:
            lines.append(f"{marker} {idx}. {name}")

    lines.append("Use select_volume(volume_name=...) only for loaded image volumes you want to inspect; it will recover the standard views after switching.")
    return "\n".join(lines)


def make_text_input_message(text: str) -> TextPrompt:
    return make_text_prompt(text)


def make_image_input_message(path: str, *, text: str, mime: str = "image/png") -> ImagePrompt:
    return make_image_prompt(path, text=text, mime=mime)
