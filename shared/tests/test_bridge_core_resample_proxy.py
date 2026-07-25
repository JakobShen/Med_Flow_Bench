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


class BridgeCoreResampleProxyTests(unittest.TestCase):
    def test_resample_volume_isotropic_handler_lazy_imports_segmentation_impl(self) -> None:
        fake_slicer = types.ModuleType("slicer")
        fake_segmentation = types.ModuleType("bridge_runtime.segmentation")

        called: dict[str, object] = {}

        def _fake_impl(**kwargs):
            called["kwargs"] = dict(kwargs)
            return {"proxied": True, **kwargs}

        fake_segmentation._resample_volume_isotropic = _fake_impl  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {
                "slicer": fake_slicer,
                "bridge_runtime.segmentation": fake_segmentation,
            },
            clear=False,
        ):
            sys.modules.pop("bridge_runtime.common", None)
            sys.modules.pop("bridge_runtime.core", None)

            core = importlib.import_module("bridge_runtime.core")
            result = core.CORE_BRIDGE_HANDLERS["resample_volume_isotropic"](
                {
                    "output_spacing_mm": 1.0,
                    "source_volume_id": "vtkMRMLScalarVolumeNode1",
                    "source_volume_name": "T1",
                    "output_name": "T1_iso_1mm",
                    "interpolation_type": "linear",
                    "make_active": False,
                }
            )

        self.assertTrue(result["proxied"])
        self.assertEqual(called["kwargs"]["output_spacing_mm"], 1.0)
        self.assertEqual(called["kwargs"]["source_volume_id"], "vtkMRMLScalarVolumeNode1")
        self.assertEqual(called["kwargs"]["source_volume_name"], "T1")
        self.assertEqual(called["kwargs"]["output_name"], "T1_iso_1mm")
        self.assertEqual(called["kwargs"]["interpolation_type"], "linear")
        self.assertFalse(called["kwargs"]["make_active"])


if __name__ == "__main__":
    unittest.main()
