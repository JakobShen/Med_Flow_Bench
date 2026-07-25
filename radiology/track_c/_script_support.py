from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_repo_imports() -> None:
    engine_root = str(Path(__file__).resolve().parents[2] / "engines")
    if engine_root in sys.path:
        sys.path.remove(engine_root)
    sys.path.insert(0, engine_root)


ensure_repo_imports()

from slicer_agent_engine.llm.registry import default_model_from_env  # noqa: E402,F401


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    token = token.strip("-._")
    return token or "unknown"


def resolve_run_output_dir(
    base_dir: Path | str,
    *,
    model: str,
    timestamp: Optional[datetime] = None,
) -> Path:
    base_path = Path(base_dir).expanduser()
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return (base_path / f"{ts}__{_safe_path_token(model)}").resolve()


def configure_logging(
    out_dir: Path,
    *,
    log_name: str,
    level: int = logging.INFO,
) -> Path:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / log_name
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


__all__ = [
    "configure_logging",
    "default_model_from_env",
    "ensure_repo_imports",
    "resolve_run_output_dir",
]
