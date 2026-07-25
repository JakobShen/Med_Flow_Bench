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


class RegistrationToolSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = build_default_tool_surface(ctx=DummyCtx())

    def _schema(self, name: str) -> dict:
        return next(tool for tool in self.surface.current_tool_schemas() if tool["name"] == name)

    def test_registration_tools_are_optional_until_pack_activation(self) -> None:
        names_before = {tool["name"] for tool in self.surface.current_tool_schemas()}
        self.assertNotIn("brains_fit", names_before)
        self.assertNotIn("sample_transform", names_before)
        self.assertNotIn("create_transform", names_before)

        activated = self.surface.activate_packs(["registration"])
        self.assertTrue(activated["ok"])
        self.assertEqual(activated["unknown_packs"], [])
        self.assertIn("registration", activated["active_packs"])

        names_after = {tool["name"] for tool in self.surface.current_tool_schemas()}
        self.assertIn("brains_fit", names_after)
        self.assertIn("brains_resample", names_after)
        self.assertIn("brains_roi_auto", names_after)
        self.assertIn("sample_transform", names_after)
        self.assertIn("create_transform", names_after)

    def test_brains_fit_schema_keeps_aliases_and_official_parameters(self) -> None:
        self.surface.activate_packs(["registration"])
        schema = self._schema("brains_fit")
        props = schema["parameters"]["properties"]

        for key in ("fixed", "moving", "output", "output_transform"):
            self.assertIn(key, props)
        for key in ("fixedVolume", "movingVolume", "linearTransform", "bsplineTransform"):
            self.assertIn(key, props)
        self.assertIn("samplingPercentage", props)
        self.assertIn("maskProcessingMode", props)
        self.assertIn("initializeTransformMode", props)
        self.assertNotIn("fixedPoints", props)
        self.assertNotIn("movingPoints", props)

    def test_sample_transform_schema_merges_export_modes(self) -> None:
        self.surface.activate_packs(["registration"])
        schema = self._schema("sample_transform")
        props = schema["parameters"]["properties"]
        self.assertEqual(
            props["output_kind"]["enum"],
            ["grid_transform", "vector_displacement_volume", "scalar_displacement_magnitude"],
        )
        self.assertIn("reference", props)
        self.assertIn("referenceVolume", props)
        self.assertIn("output_name", props)


if __name__ == "__main__":
    unittest.main()
