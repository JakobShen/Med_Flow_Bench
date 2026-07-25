from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SLICER_SIDE_ROOT = REPO_ROOT / "radiology" / "slicer_track_a_b" / "runtime" / "slicer_bridge"
if str(SLICER_SIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(SLICER_SIDE_ROOT))


class BridgeRegistrationParameterNormalizationTests(unittest.TestCase):
    def _import_registration_module(self):
        fake_slicer = types.ModuleType("slicer")
        fake_vtk = types.ModuleType("vtk")
        fake_common = types.ModuleType("bridge_runtime.common")
        fake_core = types.ModuleType("bridge_runtime.core")
        fake_core._get_active_volume_node = lambda: None
        fake_core._get_node_by_id_or_name = lambda *args, **kwargs: None
        fake_core._set_active_volume = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "slicer": fake_slicer,
                "vtk": fake_vtk,
                "bridge_runtime.common": fake_common,
                "bridge_runtime.core": fake_core,
            },
            clear=False,
        ):
            sys.modules.pop("bridge_runtime.registration", None)
            return importlib.import_module("bridge_runtime.registration")

    def test_null_like_refs_and_scalar_lists_are_normalized(self) -> None:
        mod = self._import_registration_module()

        self.assertIsNone(mod._normalize_optional_ref("None"))
        self.assertIsNone(mod._normalize_optional_ref("  null  "))
        self.assertEqual(mod._normalize_optional_ref("vtkMRMLTransformNode1"), "vtkMRMLTransformNode1")
        self.assertEqual(mod._coerce_number_list(250, cast_type=int), [250])
        self.assertEqual(mod._coerce_number_list("200, 100", cast_type=int), [200, 100])

    def test_phase_lists_default_and_repair_to_effective_phase_count(self) -> None:
        mod = self._import_registration_module()

        phases, flags_override, defaulted = mod._effective_brains_phase_names({})
        self.assertEqual(phases, ["Rigid"])
        self.assertFalse(flags_override)
        self.assertTrue(defaulted)

        phases, flags_override, defaulted = mod._effective_brains_phase_names({"transformType": "Rigid,Affine"})
        self.assertEqual(phases, ["Rigid", "Affine"])
        self.assertFalse(flags_override)
        self.assertFalse(defaulted)

        phases, flags_override, defaulted = mod._effective_brains_phase_names({"useRigid": True, "useAffine": True})
        self.assertEqual(phases, ["Rigid", "Affine"])
        self.assertTrue(flags_override)
        self.assertFalse(defaulted)

        repaired, changed = mod._normalize_phase_parameter_length([1500, 750, 250], target_len=2, cast_type=int)
        self.assertEqual(repaired, [1500, 750])
        self.assertTrue(changed)

        repaired, changed = mod._normalize_phase_parameter_length([0.1], target_len=3, cast_type=float)
        self.assertEqual(repaired, [0.1])
        self.assertFalse(changed)

        repaired, changed = mod._normalize_phase_parameter_length([0.2, 0.1], target_len=3, cast_type=float)
        self.assertEqual(repaired, [0.2, 0.1, 0.1])
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
