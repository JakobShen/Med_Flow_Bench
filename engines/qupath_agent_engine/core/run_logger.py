from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import compact, utc_ts


@dataclass(slots=True)
class RunLogger:
    log_path: Path

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str, **payload: Any) -> None:
        self.event("info", message=message, **payload)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": utc_ts(),
            "event": event_type,
            **{k: compact(v, max_chars=8000) for k, v in payload.items()},
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
