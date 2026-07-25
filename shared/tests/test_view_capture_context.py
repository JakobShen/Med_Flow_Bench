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


class _SegClient:
    def __init__(self) -> None:
        self.calls = []

    def exec_bridge(self, **kwargs):
        self.calls.append(kwargs)
        tool = kwargs["tool"]
        if tool == "image_coords_to_ras":
            return {"ok": True, "center_ras": [1.0, 2.0, 3.0]}
        if tool == "segment_local_threshold":
            return {"ok": True, "segmentation_id": "seg1", "segment_id": "s1"}
        raise AssertionError(f"unexpected tool: {tool}")


class _Ctx(ToolContextBaseMixin, ToolContextSegmentationMixin):
    def __init__(self, out_dir: Path) -> None:
        self.session = _DummySession(out_dir)
        self.bridge_dir = out_dir
        self.client = _SegClient()

    def _log(self, **kwargs):
        return None

    def _annotate_bbox_preview(self, *, png_path: str, bbox_1000, out_name: str, label=None) -> str:
        out = self.session.out_dir / out_name
        out.write_bytes(Path(png_path).read_bytes())
        return str(out)

    def _annotate_point_preview(self, *, png_path: str, point_1000, out_name: str, label=None) -> str:
        out = self.session.out_dir / out_name
        out.write_bytes(Path(png_path).read_bytes())
        return str(out)

    def _capture_current_view_png_raw(self, **kwargs):
        raise AssertionError("fresh capture should not be needed when capture context exists")

    def capture_slice_view_png(self, *, view: str = "axial", out_name=None, include_controller: bool = False):
        p = self.session.out_dir / f"{view}.png"
        if not p.exists():
            p.write_bytes(b"png")
        return {"ok": True, "png_path": str(p)}

    def observe_view(self, *, view: str = "axial", size: int = 512, out_name=None):
        p = self.session.out_dir / f"obs-{view}.png"
        p.write_bytes(b"png")
        return {"ok": True, "png_path": str(p)}


class ViewCaptureContextTests(unittest.TestCase):
    def test_image_coords_to_ras_uses_last_cropped_view_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(10, 20, 110, 220),
                full_size=(200, 300),
                orientation="axial",
                offset=42.5,
            )
            res = ctx.image_coords_to_ras(view="axial", bbox_1000=[0, 0, 999, 999])
            self.assertTrue(res["ok"])
            call = ctx.client.calls[-1]
            self.assertEqual(call["tool"], "image_coords_to_ras")
            self.assertEqual(call["args"]["orientation"], "axial")
            self.assertEqual(call["args"]["offset"], 42.5)
            self.assertIn("mapped_bbox_norm", res)
            self.assertEqual(res["reference_source"], "session_view_capture")

    def test_image_coords_to_ras_ignores_zero_point_placeholder_when_bbox_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(10, 20, 110, 220),
                full_size=(200, 300),
                orientation="axial",
                offset=42.5,
            )
            res = ctx.image_coords_to_ras(view="axial", bbox_1000=[100, 200, 300, 400], point_1000=[0, 0])
            self.assertTrue(res["ok"])
            call = ctx.client.calls[-1]
            self.assertIn("bbox_norm", call["args"])
            self.assertNotIn("point_norm", call["args"])
            self.assertEqual(res["bbox_1000"], [100, 200, 300, 400])

    def test_image_coords_to_ras_ignores_zero_bbox_placeholder_when_point_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(10, 20, 110, 220),
                full_size=(200, 300),
                orientation="axial",
                offset=42.5,
            )
            res = ctx.image_coords_to_ras(view="axial", bbox_1000=[0, 0, 0, 0], point_1000=[321, 654])
            self.assertTrue(res["ok"])
            call = ctx.client.calls[-1]
            self.assertIn("point_norm", call["args"])
            self.assertNotIn("bbox_norm", call["args"])
            self.assertEqual(res["point_1000"], [321, 654])

    def test_image_coords_to_ras_prefers_point_when_bbox_and_point_are_both_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(10, 20, 110, 220),
                full_size=(200, 300),
                orientation="axial",
                offset=42.5,
            )
            res = ctx.image_coords_to_ras(view="axial", bbox_1000=[415, 110, 485, 205], point_1000=[450, 160])
            self.assertTrue(res["ok"])
            self.assertEqual(res.get("contract_fix_applied"), "dropped_bbox_1000_point_preferred_inside_bbox")
            self.assertEqual(res.get("effective_input_kind"), "point_1000")
            call = ctx.client.calls[-1]
            self.assertIn("point_norm", call["args"])
            self.assertNotIn("bbox_norm", call["args"])
            self.assertEqual(res["point_1000"], [450, 160])

    def test_image_coords_to_ras_prefers_point_when_point_and_bbox_disagree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(10, 20, 110, 220),
                full_size=(200, 300),
                orientation="axial",
                offset=42.5,
            )
            res = ctx.image_coords_to_ras(view="axial", bbox_1000=[100, 100, 200, 200], point_1000=[180, 130])
            self.assertTrue(res["ok"])
            self.assertEqual(res.get("contract_fix_applied"), "dropped_bbox_1000_point_preferred_inside_bbox")
            self.assertEqual(res.get("effective_input_kind"), "point_1000")
            call = ctx.client.calls[-1]
            self.assertIn("point_norm", call["args"])
            self.assertNotIn("bbox_norm", call["args"])
            self.assertEqual(res["point_1000"], [180, 130])

    def test_segment_local_threshold_uses_last_cropped_view_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ctx = _Ctx(Path(td))
            img = Path(td) / "axial.png"
            img.write_bytes(b"png")
            ctx._remember_view_capture(
                view="axial",
                png_path=str(img),
                raw_png_path=str(img),
                crop_box=(0, 0, 100, 100),
                full_size=(100, 100),
                orientation="axial",
                offset=12.0,
            )
            res = ctx.segment_local_threshold(
                minimum_threshold=-800,
                maximum_threshold=300,
                seed_view="axial",
                seed_bbox_1000=[200, 200, 400, 400],
            )
            self.assertTrue(res["ok"])
            call = ctx.client.calls[-1]
            self.assertEqual(call["tool"], "segment_local_threshold")
            self.assertEqual(call["args"]["seed_orientation"], "axial")
            self.assertEqual(call["args"]["seed_offset_mm"], 12.0)
            self.assertEqual(res["seed_reference_source"], "session_view_capture")


if __name__ == "__main__":
    unittest.main()
