from __future__ import annotations

import json
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
        self.run_events = []

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def log_run_event(self, event_type: str, **payload):
        rec = {"event_type": event_type, **payload}
        self.run_events.append(rec)
        return rec


class DummyCtx:
    def __init__(self) -> None:
        self.session = DummySession()


class SelectVolumeClient:
    def list_volumes(self):
        return [{"id": "vol1", "name": "T1c"}, {"id": "vol2", "name": "FLAIR"}]


class SelectVolumeCtx(DummyCtx):
    def __init__(self) -> None:
        super().__init__()
        self.client = SelectVolumeClient()
        self.calls = []

    def select_volume(self, *, volume_id=None, volume_name=None):
        self.calls.append(("select_volume", volume_id, volume_name))
        self.session.state.update(active_volume_id="vol1", active_volume_name=volume_name or "T1c")
        return {"ok": True, "active_volume_id": "vol1", "active_volume_name": volume_name or "T1c"}

    def recover_standard_views(self, *, volume_id=None, volume_name=None, centered=True, fit=True):
        self.calls.append(("recover_standard_views", volume_id, volume_name, centered, fit))
        return {"ok": True, "volume_id": volume_id or "vol1", "volume_name": volume_name or "T1c"}

    def observe_standard_views(self, *, size=512, out_prefix="std"):
        self.calls.append(("observe_standard_views", size, out_prefix))
        return {"ok": True, "png_paths": {"axial": "a.png", "sagittal": "s.png", "coronal": "c.png"}}


class ToolSurfaceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = build_default_tool_surface(ctx=DummyCtx())

    def _tool_names(self) -> list[str]:
        return [tool["name"] for tool in self.surface.current_tool_schemas()]

    def test_core_surface_is_small_and_provider_neutral(self) -> None:
        names = set(self._tool_names())
        self.assertIn("activate_tool_pack", names)
        self.assertIn("observe", names)
        self.assertIn("fusion", names)
        self.assertNotIn("list_case_series", names)
        self.assertNotIn("roi_stats_ijk", names)
        self.assertNotIn("measure_distance_ras", names)
        self.assertNotIn("brains_fit", names)
        self.assertNotIn("create_transform", names)

        coords_schema = next(tool for tool in self.surface.current_tool_schemas() if tool["name"] == "image_coords_to_ras")
        self.assertEqual(coords_schema["kind"], "function")
        self.assertIn("bbox_1000", coords_schema["parameters"]["properties"])
        self.assertIn("point_1000", coords_schema["parameters"]["properties"])
        self.assertNotIn("oneOf", coords_schema["parameters"])

        fusion_schema = next(tool for tool in self.surface.current_tool_schemas() if tool["name"] == "fusion")
        self.assertIn("background_volume_ref", fusion_schema["parameters"]["properties"])
        self.assertIn("foreground_volume_ref", fusion_schema["parameters"]["properties"])
        self.assertIn("background_volume_ref", fusion_schema["parameters"]["required"])
        self.assertIn("foreground_volume_ref", fusion_schema["parameters"]["required"])

    def test_pack_visibility_is_enforced_in_json_executor(self) -> None:
        inactive = self.surface.execute_tool_call_json(
            "measure_distance_ras",
            json.dumps({"ras_a": [0.0, 0.0, 0.0], "ras_b": [1.0, 1.0, 1.0]}),
        )
        self.assertFalse(inactive["ok"])
        self.assertIn("Activate pack 'measurement'", inactive["error"])

        activated = self.surface.execute_tool_call_json("activate_tool_pack", json.dumps({"pack": "measurement"}))
        self.assertTrue(activated["ok"])
        self.assertIn("measurement", activated["active_packs"])


    def test_registration_pack_activation_exposes_registration_tools(self) -> None:
        names_before = set(self._tool_names())
        self.assertNotIn("brains_fit", names_before)
        self.assertNotIn("create_transform", names_before)

        activated = self.surface.activate_packs(["registration"])
        self.assertTrue(activated["ok"])
        self.assertIn("registration", activated["active_packs"])
        self.assertIn("brains_fit", activated["activated_tools"])
        self.assertIn("sample_transform", activated["activated_tools"])

        names_after = set(self._tool_names())
        self.assertIn("brains_fit", names_after)
        self.assertIn("brains_resample", names_after)
        self.assertIn("brains_roi_auto", names_after)
        self.assertIn("create_transform", names_after)
        self.assertIn("sample_transform", names_after)

        pack_meta = activated["available_packs"]["registration"]
        self.assertIn("Manual transforms", pack_meta["description"])
        self.assertIn("brains_fit", pack_meta["tool_names"])

    def test_no_optional_packs_surface_hides_activation_and_marks_optional_tools_unavailable(self) -> None:
        surface = build_default_tool_surface(ctx=DummyCtx(), optional_pack_names=[])
        names = {tool["name"] for tool in surface.current_tool_schemas()}
        self.assertNotIn("activate_tool_pack", names)
        self.assertNotIn("measure_distance_ras", names)
        self.assertNotIn("list_segmentations", names)

        unavailable = surface.execute_tool_call_json(
            "measure_distance_ras",
            json.dumps({"ras_a": [0.0, 0.0, 0.0], "ras_b": [1.0, 1.0, 1.0]}),
        )
        self.assertFalse(unavailable["ok"])
        self.assertIn("not available in this run", unavailable["error"])

        mcp_names = {spec.name for spec in surface.mcp_tool_specs()}
        self.assertNotIn("measure_distance_ras", mcp_names)

    def test_auto_activation_exposes_selected_optional_tools_immediately(self) -> None:
        surface = build_default_tool_surface(ctx=DummyCtx(), optional_pack_names=["segmentation"], pack_activation_mode="auto")
        self.assertEqual(surface.active_packs, ["segmentation"])

        names = {tool["name"] for tool in surface.current_tool_schemas()}
        self.assertNotIn("activate_tool_pack", names)
        self.assertIn("list_segmentations", names)
        self.assertIn("hide_segmentation_overlay", names)
        self.assertIn("segment_local_threshold", names)
        self.assertNotIn("measure_distance_ras", names)

    def test_fusion_returns_clean_validation_error_when_refs_are_missing(self) -> None:
        res = self.surface.execute_tool_call_json("fusion", json.dumps({"opacity": 0.4}))
        self.assertFalse(res["ok"])
        self.assertIn("background", res["error"])
        self.assertIn("foreground", res["error"])

    def test_mcp_surface_remains_filtered(self) -> None:
        names = {spec.name for spec in self.surface.mcp_tool_specs()}
        self.assertIn("open_case", names)
        self.assertIn("measure_distance_ras", names)
        self.assertNotIn("gemini_analyze_video", names)

    def test_select_volume_recovers_standard_views_before_observation(self) -> None:
        ctx = SelectVolumeCtx()
        surface = build_default_tool_surface(ctx=ctx)
        res = surface.execute_named_tool("select_volume", {"volume_name": "T1c"}, enforce_model_visibility=True)
        self.assertTrue(res["ok"])
        self.assertEqual(ctx.calls[0], ("select_volume", None, "T1c"))
        self.assertEqual(ctx.calls[1][0], "recover_standard_views")
        self.assertEqual(ctx.calls[2][0], "observe_standard_views")
        self.assertIn("recover", res)
        self.assertIn("observation", res)

    def test_select_volume_missing_reference_returns_clean_validation_error(self) -> None:
        ctx = SelectVolumeCtx()
        surface = build_default_tool_surface(ctx=ctx)
        res = surface.execute_named_tool("select_volume", {}, enforce_model_visibility=True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["required_any_of"], ["volume_id", "volume_name"])
        self.assertEqual(ctx.calls, [])


if __name__ == "__main__":
    unittest.main()
