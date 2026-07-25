from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def local_timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(value: str, *, fallback: str = "item") -> str:
    cleaned = _FILENAME_SAFE.sub("_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def compact(value: Any, *, max_chars: int = 1200) -> Any:
    if isinstance(value, dict):
        return {k: compact(v, max_chars=max_chars) for k, v in list(value.items())[:20]}
    if isinstance(value, list):
        return [compact(v, max_chars=max_chars) for v in value[:20]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f" ... <{len(value) - max_chars} more chars>"
    return value


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sleep_seconds(seconds: float) -> None:
    time.sleep(max(0.0, float(seconds)))
