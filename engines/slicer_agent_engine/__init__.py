from __future__ import annotations

from typing import Any


__all__ = ["SessionManager", "SlicerClient", "VideoRenderer"]


def __getattr__(name: str) -> Any:
    if name == "SessionManager":
        from .session import SessionManager

        return SessionManager
    if name == "SlicerClient":
        from .slicer_client import SlicerClient

        return SlicerClient
    if name == "VideoRenderer":
        from .video_renderer import VideoRenderer

        return VideoRenderer
    raise AttributeError(name)
