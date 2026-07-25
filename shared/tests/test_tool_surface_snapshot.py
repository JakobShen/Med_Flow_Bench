from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.capability_registry import build_default_tool_surface


class DummySession:
    def __init__(self) -> None:
        self.state = {}

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def log_run_event(self, event_type: str, **payload):
        return {"event_type": event_type, **payload}


class DummyCtx:
    def __init__(self) -> None:
        self.session = DummySession()
        self.gemini = None


class ToolSurfaceSnapshotTests(unittest.TestCase):
    def test_default_core_tool_order_is_stable(self) -> None:
        surface = build_default_tool_surface(ctx=DummyCtx())
        self.assertEqual(
            [tool["name"] for tool in surface.current_tool_schemas()],
            [
                "ping",
                "open_case",
                "select_volume",
                "observe",
                "image_coords_to_ras",
                "get_viewer_state",
                "wl",
                "window_preset",
                "zoom",
                "fit",
                "set_slice_offset",
                "get_slice_offset_range",
                "jump",
                "recover_standard_views",
                "set_layout",
                "scroll",
                "gemini_analyze_video",
                "fusion",
                "activate_tool_pack",
            ],
        )
        self.assertEqual(
            surface.available_packs,
            ["measurement", "monai_label", "registration", "segmentation", "stats", "viewer_advanced"],
        )


if __name__ == "__main__":
    unittest.main()
