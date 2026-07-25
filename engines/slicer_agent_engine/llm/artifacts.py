from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, List, Union


def guess_mime_type(path: Union[str, Path], default: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or default


def file_to_base64(path: Union[str, Path]) -> str:
    p = Path(path).expanduser().resolve()
    return base64.b64encode(p.read_bytes()).decode("ascii")


def file_to_data_url(path: Union[str, Path], mime: str) -> str:
    return f"data:{mime};base64,{file_to_base64(path)}"


def gather_png_paths(result: Any) -> List[Path]:
    """Best-effort extraction of PNG artifacts from a tool result."""

    paths: List[Path] = []
    if not isinstance(result, dict):
        return paths

    p = result.get("png_path")
    if isinstance(p, str) and p:
        paths.append(Path(p))

    pp = result.get("png_paths")
    if isinstance(pp, list):
        for item in pp:
            if isinstance(item, str) and item:
                paths.append(Path(item))

    if isinstance(pp, dict):
        for item in pp.values():
            if isinstance(item, str) and item:
                paths.append(Path(item))

    cs = result.get("contact_sheet_png_path")
    if isinstance(cs, str) and cs:
        paths.insert(0, Path(cs))

    deduped: List[Path] = []
    seen = set()
    for raw in paths:
        try:
            path = raw.expanduser().resolve()
        except Exception:
            path = Path(raw)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def preferred_png_paths(result: Any, *, max_images: int = 4) -> List[Path]:
    pngs = [p for p in gather_png_paths(result) if p.exists()]
    if not pngs:
        return []
    if isinstance(result, dict):
        contact = result.get("contact_sheet_png_path")
        if isinstance(contact, str) and contact:
            p = Path(contact).expanduser().resolve()
            if p.exists():
                return [p]
    return pngs[: max(0, int(max_images))]
