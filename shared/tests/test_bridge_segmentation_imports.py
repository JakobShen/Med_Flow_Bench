from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
SLICER_BRIDGE_ROOT = REPO_ROOT / "radiology" / "slicer_track_a_b" / "runtime" / "slicer_bridge"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))


class BridgeSegmentationImportTests(unittest.TestCase):
    def test_segmentation_bridge_imports_slice_helpers_from_core(self) -> None:
        source_path = SLICER_BRIDGE_ROOT / "bridge_runtime" / "segmentation.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))

        imported_names = set()
        for node in module.body:
            if isinstance(node, ast.ImportFrom) and node.module == "core":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn("_set_slice_orientation", imported_names)
        self.assertIn("_set_slice_offset", imported_names)
        self.assertIn("_set_slice_scroll_to", imported_names)


if __name__ == "__main__":
    unittest.main()
