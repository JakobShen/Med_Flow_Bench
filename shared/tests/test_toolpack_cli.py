from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.benchmarking.nsclc import NSCLCCase, build_task_prompt
from slicer_agent_engine.benchmarking.toolpack_cli import add_toolpack_cli_arguments, resolve_toolpack_cli_selection
from slicer_agent_engine.toolpacks import ToolPackSelection


class ToolPackCliTests(unittest.TestCase):
    def _parse(self, *argv: str) -> ToolPackSelection:
        parser = argparse.ArgumentParser()
        add_toolpack_cli_arguments(parser)
        return resolve_toolpack_cli_selection(parser.parse_args(list(argv)))

    def test_default_selection_provides_no_optional_toolpacks(self) -> None:
        selection = self._parse()
        self.assertEqual(selection.provided_tool_packs, ())
        self.assertFalse(selection.has_optional_tool_packs)
        self.assertEqual(selection.activation_mode, "manual")
        self.assertEqual(selection.prompt_hints, ())

    def test_all_toolpacks_selection_matches_manual_activation_mode(self) -> None:
        selection = self._parse("--all_toolpacks")
        self.assertIsNone(selection.provided_tool_packs)
        self.assertTrue(selection.has_optional_tool_packs)
        self.assertEqual(selection.activation_mode, "manual")

    def test_segmentation_selection_supports_auto_activation_and_prompt_hint(self) -> None:
        selection = self._parse("--segmentation_toolpacks", "--activate_toolpacks", "auto")
        self.assertEqual(selection.provided_tool_packs, ("segmentation",))
        self.assertTrue(selection.has_optional_tool_packs)
        self.assertEqual(selection.activation_mode, "auto")
        self.assertTrue(any("SegmentationTools" in hint for hint in selection.prompt_hints))


class NsclcToolPackPromptTests(unittest.TestCase):
    def _case(self) -> NSCLCCase:
        return NSCLCCase(
            case_id="CASE-001",
            case_root=Path("/tmp/case"),
            catalog=object(),
            gt_location=None,
            gt_t_stage=None,
            gt_n_stage=None,
            gt_histology=None,
            gt_grade=None,
        )

    def test_nsclc_prompt_omits_case_series_hint_when_pack_is_not_provided(self) -> None:
        prompt = build_task_prompt(self._case(), toolpack_selection=ToolPackSelection.none())
        self.assertNotIn("case_series tool pack", prompt)
        self.assertIn("core fusion tool directly", prompt)

    def test_nsclc_prompt_uses_manual_case_series_hint_when_pack_requires_activation(self) -> None:
        prompt = build_task_prompt(
            self._case(),
            toolpack_selection=ToolPackSelection(provided_tool_packs=("case_series",), activation_mode="manual"),
        )
        self.assertIn("first activate the case_series tool pack", prompt)
        self.assertIn("prefer the curated alias pet_ct_core after activating the case_series tool pack", prompt)

    def test_nsclc_prompt_uses_direct_case_series_hint_when_pack_is_auto_active(self) -> None:
        prompt = build_task_prompt(
            self._case(),
            toolpack_selection=ToolPackSelection(provided_tool_packs=("case_series",), activation_mode="auto"),
        )
        self.assertNotIn("first activate the case_series tool pack", prompt)
        self.assertIn("call load_case_series(alias)", prompt)
        self.assertIn("prefer the curated alias pet_ct_core via load_case_series(alias)", prompt)


if __name__ == "__main__":
    unittest.main()
