from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_imports() -> None:
    engine_root = str(Path(__file__).resolve().parents[2] / "engines")
    if engine_root in sys.path:
        sys.path.remove(engine_root)
    sys.path.insert(0, engine_root)


ensure_repo_imports()

from slicer_agent_engine.app.bootstrap import (  # noqa: E402,F401
    ContextBundle,
    ModelRuntimeBundle,
    ScriptRuntimeBundle,
    bootstrap_runtime,
    build_context_bundle,
    build_model_runtime,
    build_optional_gemini_video,
    build_session_context,
    configure_logging,
    ensure_slicer_runtime,
    resolve_run_output_dir,
    resolve_bridge_dir,
)
from slicer_agent_engine.llm.registry import default_model_from_env, resolve_provider_name  # noqa: E402,F401

__all__ = [
    "ContextBundle",
    "ModelRuntimeBundle",
    "ScriptRuntimeBundle",
    "bootstrap_runtime",
    "build_context_bundle",
    "build_model_runtime",
    "build_optional_gemini_video",
    "build_session_context",
    "configure_logging",
    "default_model_from_env",
    "ensure_repo_imports",
    "ensure_slicer_runtime",
    "resolve_run_output_dir",
    "resolve_bridge_dir",
    "resolve_provider_name",
]
