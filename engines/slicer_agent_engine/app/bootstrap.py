from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from ..gemini_video import GeminiVideoAnalyzer
from ..llm.registry import build_builtin_model_tools, build_model_client, resolve_provider_name
from ..session import SessionManager
from ..slicer_client import SlicerClient
from ..slicer_launcher import ensure_webserver
from ..tools import ToolContext
from ..video_renderer import VideoRenderer


def resolve_bridge_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "radiology"
        / "slicer_track_a_b"
        / "runtime"
        / "slicer_bridge"
    )


def default_slicer_executable() -> Path:
    return Path(os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"))


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    token = token.strip("-._")
    return token or "unknown"


def resolve_run_output_dir(base_dir: Path | str, *, model: str, timestamp: Optional[datetime] = None) -> Path:
    base_path = Path(base_dir).expanduser()
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}__{_safe_path_token(model)}"
    return (base_path / run_name).resolve()


def configure_logging(out_dir: Path, *, log_name: str, level: int = logging.INFO) -> Path:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / log_name
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def build_optional_gemini_video(*, enabled: bool) -> Optional[GeminiVideoAnalyzer]:
    if not enabled or not os.environ.get("GEMINI_API_KEY"):
        return None
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    return GeminiVideoAnalyzer(model=gemini_model)


def ensure_slicer_runtime(
    *,
    base_url: str,
    bridge_dir: Optional[Path] = None,
    slicer_executable: Optional[Path] = None,
    require_exec: bool = True,
    require_slice_render: bool = True,
) -> Path:
    resolved_bridge_dir = bridge_dir or resolve_bridge_dir()
    if slicer_executable is not None:
        os.environ["SLICER_EXECUTABLE"] = str(Path(slicer_executable).expanduser().resolve())
    bootstrap_script = resolved_bridge_dir / "bootstrap_webserver.py"
    parsed_port = urlparse(base_url).port or 2016
    ensure_webserver(
        base_url=base_url,
        bootstrap_script=bootstrap_script,
        start_if_not_running=True,
        port=parsed_port,
        require_exec=require_exec,
        require_slice=require_slice_render,
    )
    return resolved_bridge_dir


def restart_slicer_runtime(
    *,
    base_url: str,
    bridge_dir: Optional[Path] = None,
    slicer_executable: Optional[Path] = None,
    require_exec: bool = True,
    require_slice_render: bool = True,
    logger: Optional[logging.Logger] = None,
    reason: Optional[str] = None,
    attempts: int = 2,
    settle_time_s: float = 3.0,
) -> Path:
    resolved_bridge_dir = bridge_dir or resolve_bridge_dir()
    if slicer_executable is not None:
        os.environ["SLICER_EXECUTABLE"] = str(Path(slicer_executable).expanduser().resolve())
    parsed_port = urlparse(base_url).port or 2016
    last_err: Optional[Exception] = None

    for attempt_idx in range(1, max(1, int(attempts)) + 1):
        if logger is not None:
            suffix = f" ({reason})" if reason else ""
            logger.info(
                "Restarting Slicer at %s%s [attempt %d/%d].",
                base_url,
                suffix,
                attempt_idx,
                max(1, int(attempts)),
            )
        try:
            SlicerClient(base_url=base_url, timeout_s=5.0).shutdown_system()
        except Exception as e:
            if logger is not None:
                logger.info("Port-specific Slicer shutdown via WebServer failed: %s", e)
        try:
            subprocess.run(
                ["fuser", "-k", f"{parsed_port}/tcp"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            if logger is not None:
                logger.info("Port-specific fuser cleanup failed: %s", e)
        time.sleep(settle_time_s)
        try:
            return ensure_slicer_runtime(
                base_url=base_url,
                bridge_dir=resolved_bridge_dir,
                slicer_executable=slicer_executable,
                require_exec=require_exec,
                require_slice_render=require_slice_render,
            )
        except Exception as e:
            last_err = e
            if logger is not None:
                logger.warning(
                    "Slicer restart attempt %d/%d did not reach healthy exec/slice readiness: %s",
                    attempt_idx,
                    max(1, int(attempts)),
                    e,
                )
    if last_err is not None:
        raise last_err
    raise RuntimeError("restart_slicer_runtime exited without a result")


@dataclass(frozen=True)
class ModelRuntimeBundle:
    resolved_provider: str
    model_client: Any
    extra_model_tools: list[dict[str, Any]]


@dataclass
class ContextBundle:
    client: SlicerClient
    session: SessionManager
    video: VideoRenderer
    gemini: Optional[GeminiVideoAnalyzer]
    ctx: ToolContext


@dataclass
class ScriptRuntimeBundle:
    bridge_dir: Path
    out_dir: Path
    log_path: Path
    model: ModelRuntimeBundle
    context: ContextBundle


def build_model_runtime(*, provider: str, model: str, enable_code_interpreter: bool = False) -> ModelRuntimeBundle:
    resolved_provider = resolve_provider_name(provider, model)
    model_client = build_model_client(resolved_provider, model=model)
    extra_model_tools = build_builtin_model_tools(
        resolved_provider,
        model=model,
        enable_code_interpreter=enable_code_interpreter,
    )
    return ModelRuntimeBundle(
        resolved_provider=resolved_provider,
        model_client=model_client,
        extra_model_tools=extra_model_tools,
    )


def build_session_context(
    *,
    client: SlicerClient,
    out_dir: Path,
    bridge_dir: Optional[Path] = None,
    video: Optional[VideoRenderer] = None,
    gemini: Optional[GeminiVideoAnalyzer] = None,
    algorithm_runtime: Any = None,
) -> ContextBundle:
    resolved_bridge_dir = bridge_dir or resolve_bridge_dir()
    session = SessionManager(out_dir=out_dir)
    video_obj = video or VideoRenderer()
    ctx = ToolContext(
        client=client,
        session=session,
        bridge_dir=resolved_bridge_dir,
        video=video_obj,
        gemini=gemini,
        algorithm_runtime=algorithm_runtime,
    )
    return ContextBundle(client=client, session=session, video=video_obj, gemini=gemini, ctx=ctx)

def build_context_bundle(
    *,
    base_url: str,
    out_dir: Path,
    bridge_dir: Optional[Path] = None,
    enable_gemini_video: bool = False,
) -> ContextBundle:
    client = SlicerClient(base_url=base_url)
    gemini = build_optional_gemini_video(enabled=enable_gemini_video)
    return build_session_context(
        client=client,
        out_dir=out_dir,
        bridge_dir=bridge_dir,
        gemini=gemini,
    )


def bootstrap_runtime(
    *,
    out_dir: Path,
    log_name: str,
    base_url: str,
    provider: str,
    model: str,
    enable_code_interpreter: bool = False,
    enable_gemini_video: bool = False,
    ensure_slicer_ready: bool = True,
    slicer_executable: Optional[Path] = None,
    require_exec: bool = True,
    require_slice_render: bool = True,
) -> ScriptRuntimeBundle:
    bridge_dir = resolve_bridge_dir()
    log_path = configure_logging(out_dir, log_name=log_name)
    if ensure_slicer_ready:
        ensure_slicer_runtime(
            base_url=base_url,
            bridge_dir=bridge_dir,
            slicer_executable=slicer_executable,
            require_exec=require_exec,
            require_slice_render=require_slice_render,
        )
    model_bundle = build_model_runtime(provider=provider, model=model, enable_code_interpreter=enable_code_interpreter)
    context_bundle = build_context_bundle(
        base_url=base_url,
        out_dir=out_dir,
        bridge_dir=bridge_dir,
        enable_gemini_video=enable_gemini_video,
    )
    return ScriptRuntimeBundle(
        bridge_dir=bridge_dir,
        out_dir=out_dir,
        log_path=log_path,
        model=model_bundle,
        context=context_bundle,
    )
