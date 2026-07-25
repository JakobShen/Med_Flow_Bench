from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from qupath_agent_engine.core.utils import sleep_seconds

from .config import QuPathConfig
from .launcher import QuPathLauncher
from .platform import macos


class ViewerCapture:
    def __init__(self, *, config: QuPathConfig, launcher: QuPathLauncher) -> None:
        self.config = config.normalized()
        self.launcher = launcher
        self._platform = platform.system().lower()

    def capture_current_viewer(self, output_path: Path) -> dict[str, Any]:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.launcher.focus()

            if self._platform == "darwin":
                macos.copy_current_viewer_to_clipboard(self.launcher.process_name)
            else:
                # Best-effort fallback: rely on the viewer shortcut after focusing the app.
                self.launcher.focus()
        except Exception:
            return {
                "ok": False,
                "error": (
                    "Failed to capture current QuPath viewer content from clipboard. "
                    "Check focus, menu accessibility permissions, and Pillow clipboard support."
                ),
                "path": str(output_path),
            }

        for attempt in range(1, self.config.viewer_copy_retry_count + 1):
            sleep_seconds(self.config.viewer_copy_retry_sleep)
            clipped = ImageGrab.grabclipboard()
            if clipped is None:
                continue
            if isinstance(clipped, list):
                continue
            clipped.save(output_path, format="PNG")
            return {
                "ok": True,
                "path": str(output_path),
                "attempt": attempt,
                "mode": clipped.mode,
                "size": clipped.size,
            }

        return {
            "ok": False,
            "error": (
                "Failed to capture current QuPath viewer content from clipboard. "
                "Check focus, menu accessibility permissions, and Pillow clipboard support."
            ),
            "path": str(output_path),
        }
