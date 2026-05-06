from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def ensure_repo_imports() -> None:
    try:
        import slicer_agent_engine  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


ensure_repo_imports()

from slicer_agent_engine.llm.registry import default_model_from_env, resolve_provider_name  # noqa: E402,F401


def configure_logging(log_dir: Path, *, log_name: str = "benchmark.log", level: int = logging.INFO) -> None:
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def resolve_run_output_dir(base_dir: str | Path, *, model: Optional[str] = None) -> Path:
    import time

    base = Path(base_dir).expanduser().resolve()
    if not model:
        return base
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(model)).strip("_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return base / f"{timestamp}__{safe_model or 'model'}"


__all__ = [
    "configure_logging",
    "default_model_from_env",
    "ensure_repo_imports",
    "resolve_provider_name",
    "resolve_run_output_dir",
]
