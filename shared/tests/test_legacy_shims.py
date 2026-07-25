from __future__ import annotations

import unittest


class LegacyShimTests(unittest.TestCase):
    def test_openai_common_shim_imports_legacy_symbols(self) -> None:
        import slicer_agent_engine.openai_common as shim

        self.assertTrue(hasattr(shim, 'file_to_data_url'))

    def test_openai_agent_shim_reexports_legacy_entrypoint(self) -> None:
        import slicer_agent_engine.agents.openai_auto_agent as shim

        self.assertTrue(hasattr(shim, 'run_openai_auto_task'))


if __name__ == '__main__':
    unittest.main()
