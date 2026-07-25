from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def default_qupath_executable() -> Path:
    configured = os.environ.get("QUPATH_EXECUTABLE")
    if configured:
        return Path(configured).expanduser()

    detected = shutil.which("QuPath") or shutil.which("QuPath.exe")
    if detected:
        return Path(detected)

    if platform.system().lower() == "darwin":
        candidates = (
            Path("/Applications/QuPath.app/Contents/MacOS/QuPath"),
            Path("/Applications/QuPath-0.7.0.app/Contents/MacOS/QuPath"),
            Path("/Applications/QuPath-0.7.0-x64.app/Contents/MacOS/QuPath-0.7.0-x64"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    if platform.system().lower() == "windows":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates = (
            program_files / "QuPath" / "QuPath.exe",
            program_files / "QuPath-0.7.0" / "QuPath.exe",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return Path("QuPath")


def default_image_path() -> Path | None:
    configured = os.environ.get("QUPATH_DEFAULT_IMAGE")
    return Path(configured).expanduser() if configured else None


DEFAULT_QUPATH_EXECUTABLE = default_qupath_executable()
DEFAULT_IMAGE_PATH = default_image_path()


@dataclass(slots=True)
class QuPathConfig:
    executable_path: Path = field(default_factory=default_qupath_executable)
    sidecar_dir: Path | None = None
    process_name: str | None = None
    display_id: str | None = None
    display_dpr: float | None = None
    load_wait_seconds: float = 20.0
    viewer_copy_retry_count: int = 8
    viewer_copy_retry_sleep: float = 0.35
    midscene_wait_after_action_ms: int = 500
    midscene_replanning_cycle_limit: int = 10
    close_on_exit: bool = False

    @classmethod
    def from_env(cls) -> "QuPathConfig":
        executable = os.environ.get("QUPATH_EXECUTABLE")
        sidecar = os.environ.get("MIDSCENE_QUPATH_SIDECAR_DIR")
        process_name = os.environ.get("QUPATH_PROCESS_NAME") or None
        display_id = os.environ.get("QUPATH_DISPLAY_ID") or None
        dpr = os.environ.get("QUPATH_DISPLAY_DPR")
        return cls(
            executable_path=Path(executable).expanduser() if executable else default_qupath_executable(),
            sidecar_dir=Path(sidecar).expanduser().resolve() if sidecar else None,
            process_name=process_name,
            display_id=display_id,
            display_dpr=float(dpr) if dpr else None,
        )

    def normalized(self) -> "QuPathConfig":
        executable = self.executable_path.expanduser()
        if executable.is_absolute():
            executable = executable.resolve()
        else:
            detected = shutil.which(str(executable))
            executable = Path(detected).resolve() if detected else executable
        sidecar = self.sidecar_dir.expanduser().resolve() if self.sidecar_dir else None
        process_name = self.process_name or executable.name
        return QuPathConfig(
            executable_path=executable,
            sidecar_dir=sidecar,
            process_name=process_name,
            display_id=self.display_id,
            display_dpr=self.display_dpr,
            load_wait_seconds=self.load_wait_seconds,
            viewer_copy_retry_count=self.viewer_copy_retry_count,
            viewer_copy_retry_sleep=self.viewer_copy_retry_sleep,
            midscene_wait_after_action_ms=self.midscene_wait_after_action_ms,
            midscene_replanning_cycle_limit=self.midscene_replanning_cycle_limit,
            close_on_exit=self.close_on_exit,
        )
