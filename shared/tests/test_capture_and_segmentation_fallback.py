from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.runtime_tools.common import ToolContextBaseMixin
from slicer_agent_engine.runtime_tools.segmentation import ToolContextSegmentationMixin


class _DummySession:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.session_id = "sess"
        self.state = {}

    def log_event(self, **kwargs):
        return kwargs

    def update_state(self, **kwargs):
        self.state.update(kwargs)


class _RetryClient:
    def __init__(self) -> None:
        self.calls = 0

    def exec_bridge(self, **kwargs):
        self.calls += 1
        output_path = Path(kwargs["args"]["output_path"])
        if self.calls >= 2:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"not-a-real-png-but-exists")
        return {"ok": True, "output_path": str(output_path)}


class _CaptureCtx(ToolContextBaseMixin):
    def __init__(self, out_dir: Path) -> None:
        self.session = _DummySession(out_dir)
        self.bridge_dir = out_dir
        self.client = _RetryClient()

    def _crop_focus_png_in_place(self, *, png_path: str):
        return png_path, (0, 0, 10, 10), (10, 10)


class CaptureSliceViewRetryTests(unittest.TestCase):
    def test_capture_slice_view_png_retries_when_bridge_output_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _CaptureCtx(Path(td))
            res = ctx.capture_slice_view_png(view="axial", out_name="cap.png")
            self.assertTrue(Path(res["png_path"]).exists())
            self.assertEqual(ctx.client.calls, 2)


class _SegClient:
    def __init__(self, *, voxel_count: int = 12) -> None:
        self.voxel_count = voxel_count

    def exec_bridge(self, **kwargs):
        if kwargs.get("tool") == "segment_local_threshold":
            return {"ok": True, "segmentation_id": "seg1", "segment_id": "segA", "voxel_count": self.voxel_count}
        raise AssertionError(f"unexpected bridge tool: {kwargs.get('tool')}")


class _SegCtx(ToolContextSegmentationMixin):
    def __init__(self, *, voxel_count: int = 12) -> None:
        self.session = _DummySession(Path("/tmp"))
        self.bridge_dir = Path("/tmp")
        self.client = _SegClient(voxel_count=voxel_count)
        self.capture_calls = []
        self.observe_calls = []

    def _log(self, **kwargs):
        return None

    def _annotate_bbox_preview(self, *, png_path: str, bbox_1000, out_name: str, label=None) -> str:
        return f"annotated::{png_path}"

    def capture_slice_view_png(self, *, view: str = "axial", out_name=None, include_controller: bool = False):
        self.capture_calls.append(view)
        if view == "axial":
            raise FileNotFoundError("missing overlay png")
        return {"png_path": f"{view}.png"}

    def observe_view(self, *, view: str = "axial", size: int = 512, out_name=None):
        self.observe_calls.append(view)
        return {"png_path": f"fallback-{view}.png"}


class SegmentationOverlayFallbackTests(unittest.TestCase):
    def test_segment_local_threshold_does_not_fall_back_to_observe_view_when_overlay_capture_fails(self) -> None:
        ctx = _SegCtx()
        res = ctx.segment_local_threshold(
            minimum_threshold=-800,
            maximum_threshold=400,
            ras_seed=[1.0, 2.0, 3.0],
            size=512,
        )
        self.assertTrue(res["ok"])
        self.assertIn("png_paths", res)
        self.assertEqual(ctx.observe_calls, [])
        self.assertNotIn("axial", res["png_paths"])
        self.assertEqual(res["png_paths"]["sagittal"], "sagittal.png")
        self.assertEqual(res["png_paths"]["coronal"], "coronal.png")

    def test_segment_local_threshold_returns_retryable_failure_for_zero_voxels(self) -> None:
        ctx = _SegCtx(voxel_count=0)
        res = ctx.segment_local_threshold(
            minimum_threshold=-800,
            maximum_threshold=400,
            ras_seed=[1.0, 2.0, 3.0],
            size=512,
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["failure_reason"], "zero_voxels")
        self.assertTrue(res["retryable"])
        self.assertEqual(ctx.observe_calls, [])
        self.assertEqual(ctx.capture_calls, [])


if __name__ == "__main__":
    unittest.main()
