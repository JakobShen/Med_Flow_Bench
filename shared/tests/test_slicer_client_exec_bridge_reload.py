from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.slicer_client import SlicerClient


class _CaptureExecClient(SlicerClient):
    def __init__(self) -> None:
        super().__init__()
        self.last_source = None

    def exec_python(self, source: str, *, timeout_s=None):  # type: ignore[override]
        self.last_source = source
        return {"ok": True}


class SlicerClientExecBridgeReloadTests(unittest.TestCase):
    def test_exec_bridge_source_clears_bridge_runtime_modules(self) -> None:
        client = _CaptureExecClient()
        result = client.exec_bridge(
            bridge_dir=REPO_ROOT / "3d_slicer" / "runtime" / "slicer_bridge",
            tool="ping",
            args={},
            session_id="test-session",
        )

        self.assertEqual(result, {"ok": True})
        self.assertIsInstance(client.last_source, str)
        self.assertIn('name.startswith("bridge_runtime.")', client.last_source)
        self.assertIn('name in {"slicer_agent_bridge_runtime", "bridge_runtime"}', client.last_source)
        self.assertIn("importlib.invalidate_caches()", client.last_source)


if __name__ == "__main__":
    unittest.main()
