from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.algorithm_runtime import NoopAlgorithmRuntime


class NoopAlgorithmRuntimeTests(unittest.TestCase):
    def test_run_blocking_returns_structured_failure(self) -> None:
        runtime = NoopAlgorithmRuntime()
        res = runtime.run_blocking(operation="monai_label_infer", inputs={"image": "case.nii.gz"})
        self.assertFalse(res.ok)
        self.assertEqual(res.job.backend, "noop")
        self.assertEqual(res.job.metadata["operation"], "monai_label_infer")
        self.assertIn("No algorithm runtime configured", res.error or "")


if __name__ == "__main__":
    unittest.main()
