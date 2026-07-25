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


class BridgeRegistrationImportTests(unittest.TestCase):
    def test_registration_bridge_imports_scene_helpers_from_core(self) -> None:
        source_path = SLICER_BRIDGE_ROOT / "bridge_runtime" / "registration.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))

        imported_names = set()
        for node in module.body:
            if isinstance(node, ast.ImportFrom) and node.module == "core":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn("_get_active_volume_node", imported_names)
        self.assertIn("_get_node_by_id_or_name", imported_names)
        self.assertIn("_set_active_volume", imported_names)

    def test_main_bridge_registers_registration_handlers(self) -> None:
        source_path = SLICER_BRIDGE_ROOT / "slicer_agent_bridge.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("REGISTRATION_BRIDGE_HANDLERS", source)
        self.assertIn("_BRIDGE_HANDLERS.update(REGISTRATION_BRIDGE_HANDLERS)", source)


if __name__ == "__main__":
    unittest.main()
