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
        self.events = []

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def log_run_event(self, event_type: str, **payload):
        rec = {"event_type": event_type, **payload}
        self.events.append(rec)
        return rec


class PreviewCtx:
    def __init__(self) -> None:
        self.session = DummySession()
        self.calls: list[tuple[str, dict]] = []
        self.gemini = None

    def _record(self, name: str, **kwargs):
        self.calls.append((name, dict(kwargs)))

    def observe_standard_views(self, *, size: int = 512, out_prefix: str = "preview"):
        self._record("observe_standard_views", size=size, out_prefix=out_prefix)
        return {
            "ok": True,
            "png_paths": {
                "axial": f"/tmp/{out_prefix}_axial.png",
                "sagittal": f"/tmp/{out_prefix}_sagittal.png",
                "coronal": f"/tmp/{out_prefix}_coronal.png",
            },
        }

    def fuse_volumes(self, *, background_volume_ref: str, foreground_volume_ref: str, opacity: float = 0.35):
        self._record(
            "fuse_volumes",
            background_volume_ref=background_volume_ref,
            foreground_volume_ref=foreground_volume_ref,
            opacity=opacity,
        )
        return {
            "ok": True,
            "background_volume_id": background_volume_ref,
            "foreground_volume_id": foreground_volume_ref,
            "opacity": opacity,
        }

    def apply_transform(self, **kwargs):
        self._record("apply_transform", **kwargs)
        return {"ok": True, "applied_nodes": [{"id": kwargs.get("node_ref"), "name": kwargs.get("node_ref")}]}

    def brains_fit(self, **kwargs):
        self._record("brains_fit", **kwargs)
        return {
            "ok": True,
            "transform_id": "vtkMRMLTransformNode1",
            "transform_name": "RigidTransform",
            "fixed_volume_id": "vtkMRMLScalarVolumeNodeFixed",
            "fixed_volume_name": "fixed",
            "moving_volume_id": "vtkMRMLScalarVolumeNodeMoving",
            "moving_volume_name": "moving",
        }

    def edit_transform(self, **kwargs):
        self._record("edit_transform", **kwargs)
        return {"ok": True, "transform_id": "vtkMRMLTransformNode1", "transform_name": "ManualTransform"}

    def brains_resample(self, **kwargs):
        self._record("brains_resample", **kwargs)
        return {
            "ok": True,
            "reference_volume_id": "vtkMRMLScalarVolumeNodeFixed",
            "reference_volume_name": "fixed",
            "output_volume_id": "vtkMRMLScalarVolumeNodeResampled",
            "output_volume_name": "resampled",
        }

    def brains_roi_auto(self, **kwargs):
        self._record("brains_roi_auto", **kwargs)
        return {"ok": True, "output_volume_id": "vtkMRMLScalarVolumeNodeCrop", "output_volume_name": "crop"}


class RegistrationBuilderPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = PreviewCtx()
        self.surface = build_default_tool_surface(ctx=self.ctx)
        self.surface.activate_packs(["registration"])

    def test_brains_fit_returns_preview_images_and_applies_transform_for_preview(self) -> None:
        result = self.surface.execute_named_tool(
            "brains_fit",
            {"fixed": "fixed", "moving": "moving", "useRigid": True},
            enforce_model_visibility=True,
        )

        self.assertTrue(result["ok"])
        self.assertIn("png_paths", result)
        self.assertIn("preview_fusion", result)
        self.assertIn("preview_apply_transform", result)
        self.assertTrue(result["preview_transform_applied"])

        call_names = [name for name, _ in self.ctx.calls]
        self.assertEqual(call_names[0], "brains_fit")
        self.assertIn("apply_transform", call_names)
        self.assertIn("fuse_volumes", call_names)
        self.assertIn("observe_standard_views", call_names)

    def test_manual_edit_transform_returns_preview_images(self) -> None:
        result = self.surface.execute_named_tool(
            "edit_transform",
            {"transform": "ManualTransform", "translation": [1.0, 0.0, 0.0]},
            enforce_model_visibility=True,
        )

        self.assertTrue(result["ok"])
        self.assertIn("png_paths", result)
        self.assertIn("observation", result)
        self.assertEqual(self.ctx.calls[0][0], "edit_transform")
        self.assertEqual(self.ctx.calls[1][0], "observe_standard_views")

    def test_brains_resample_returns_preview_images(self) -> None:
        result = self.surface.execute_named_tool(
            "brains_resample",
            {"input": "moving", "reference": "fixed"},
            enforce_model_visibility=True,
        )

        self.assertTrue(result["ok"])
        self.assertIn("png_paths", result)
        self.assertIn("preview_fusion", result)


if __name__ == "__main__":
    unittest.main()
