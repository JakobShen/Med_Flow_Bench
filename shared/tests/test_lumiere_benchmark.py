from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.benchmarking.judge import extract_choice_from_set, extract_numbered_answers
from slicer_agent_engine.benchmarking.lumiere import build_question_bank, stage_question_inputs


RATING_LABEL_COL = (
    "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, "
    "CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)"
)
RATING_RATIONALE_COL = (
    "Rating rationale (CRET: complete resection of the enhancing tumor, PRET: partial resection of the "
    "enhancing tumor, T2-Progr.: T2-Progression, L: Lesion)"
)


class LUMIEREBenchmarkTests(unittest.TestCase):
    def _touch_study(self, root: Path, patient: str, timepoint: str) -> None:
        study_dir = root / "Imaging" / patient / timepoint / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
        study_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ct1_skull_strip.nii.gz", "t1_skull_strip.nii.gz", "t2_skull_strip.nii.gz", "flair_skull_strip.nii.gz"):
            (study_dir / name).write_bytes(b"test")

    def _write_ratings(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Patient",
                    "Date",
                    "LessThan3Months",
                    "NonMeasurableLesions",
                    RATING_LABEL_COL,
                    RATING_RATIONALE_COL,
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_completeness(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Patient",
                    "Timepoint",
                    "CT1",
                    "T1",
                    "T2",
                    "FLAIR",
                    "DeepBraTumIA",
                    "HD-GLIO-AUTO",
                    "DeepBraTumIA-CoLlAGe",
                    "HD-GLIO-AUTO-CoLlAGe",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_build_question_bank_prefers_postop_as_old_reference_when_same_week_contains_preop_and_postop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patient = "Patient-001"
            for tp in ("week-000-1", "week-000", "week-004"):
                self._touch_study(root, patient, tp)

            ratings_csv = root / "ratings.csv"
            self._write_ratings(
                ratings_csv,
                [
                    {"Patient": patient, "Date": "week-000", RATING_LABEL_COL: "Post-Op", RATING_RATIONALE_COL: "CRET", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-000-1", RATING_LABEL_COL: "Pre-Op", RATING_RATIONALE_COL: "", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-004", RATING_LABEL_COL: "SD", RATING_RATIONALE_COL: "", "LessThan3Months": "", "NonMeasurableLesions": ""},
                ],
            )
            completeness_csv = root / "datacompleteness.csv"
            self._write_completeness(
                completeness_csv,
                [
                    {"Patient": patient, "Timepoint": "week-000-1", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                    {"Patient": patient, "Timepoint": "week-000", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                    {"Patient": patient, "Timepoint": "week-004", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                ],
            )

            bank = build_question_bank(data_root=root, ratings_csv=ratings_csv, datacompleteness_csv=completeness_csv)
            self.assertEqual(len(bank.questions), 1)
            q = bank.questions[0]
            self.assertEqual(q.old_visit.timepoint, "week-000")
            self.assertEqual(q.current_visit.timepoint, "week-004")
            self.assertEqual(q.gt_rano_label, "SD")
            self.assertEqual(q.gt_choice, "B")

    def test_build_question_bank_skips_conflicting_duplicate_same_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patient = "Patient-002"
            for tp in ("week-000", "week-010", "week-020"):
                self._touch_study(root, patient, tp)

            ratings_csv = root / "ratings.csv"
            self._write_ratings(
                ratings_csv,
                [
                    {"Patient": patient, "Date": "week-000", RATING_LABEL_COL: "Post-Op", RATING_RATIONALE_COL: "CRET", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-010", RATING_LABEL_COL: "SD", RATING_RATIONALE_COL: "", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-010", RATING_LABEL_COL: "PD", RATING_RATIONALE_COL: "T2-Progr.", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-020", RATING_LABEL_COL: "PD", RATING_RATIONALE_COL: "Target L.", "LessThan3Months": "", "NonMeasurableLesions": ""},
                ],
            )
            completeness_csv = root / "datacompleteness.csv"
            self._write_completeness(
                completeness_csv,
                [
                    {"Patient": patient, "Timepoint": "week-000", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                    {"Patient": patient, "Timepoint": "week-010", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                    {"Patient": patient, "Timepoint": "week-020", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                ],
            )

            bank = build_question_bank(data_root=root, ratings_csv=ratings_csv, datacompleteness_csv=completeness_csv)
            self.assertEqual(len(bank.questions), 0)
            self.assertEqual(bank.stats["n_ambiguous_duplicate_groups"], 1)

    def test_stage_question_inputs_creates_unique_old_and_current_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patient = "Patient-003"
            for tp in ("week-000", "week-008"):
                self._touch_study(root, patient, tp)

            ratings_csv = root / "ratings.csv"
            self._write_ratings(
                ratings_csv,
                [
                    {"Patient": patient, "Date": "week-000", RATING_LABEL_COL: "Post-Op", RATING_RATIONALE_COL: "PRET", "LessThan3Months": "", "NonMeasurableLesions": ""},
                    {"Patient": patient, "Date": "week-008", RATING_LABEL_COL: "PR", RATING_RATIONALE_COL: "", "LessThan3Months": "", "NonMeasurableLesions": ""},
                ],
            )
            completeness_csv = root / "datacompleteness.csv"
            self._write_completeness(
                completeness_csv,
                [
                    {"Patient": patient, "Timepoint": "week-000", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                    {"Patient": patient, "Timepoint": "week-008", "CT1": "x", "T1": "x", "T2": "x", "FLAIR": "x", "DeepBraTumIA": "x", "HD-GLIO-AUTO": "x", "DeepBraTumIA-CoLlAGe": "", "HD-GLIO-AUTO-CoLlAGe": ""},
                ],
            )

            bank = build_question_bank(data_root=root, ratings_csv=ratings_csv, datacompleteness_csv=completeness_csv)
            q = bank.questions[0]
            staged = stage_question_inputs(q, root / "staged")
            self.assertEqual(len(staged.all_files), 8)
            names = sorted(path.name for path in staged.all_files)
            self.assertTrue(any(name.startswith("OLD__week-000__CT1") for name in names))
            self.assertTrue(any(name.startswith("CURRENT__week-008__FLAIR") for name in names))
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(path.exists() for path in staged.all_files))

    def test_extract_choice_from_set_supports_d(self) -> None:
        self.assertEqual(extract_choice_from_set("Reasoning\nANSWER: D", valid_choices=("A", "B", "C", "D")), "D")
        self.assertEqual(extract_choice_from_set('{"answer": "B"}', valid_choices=("A", "B", "C", "D")), "B")

    def test_extract_choice_from_set_ignores_options_word(self) -> None:
        text = (
            "These features support Glioblastoma while other options like Oligodendroglioma are less likely.\n\n"
            "ANSWER: A"
        )
        self.assertEqual(extract_choice_from_set(text, valid_choices=("A", "B", "C")), "A")

    def test_extract_choice_from_set_prefers_final_answer_over_discussed_option(self) -> None:
        text = (
            "Oligodendroglioma (option B) is possible, but glioblastoma is more likely.\n"
            "FINAL ANSWER: A\n"
            "EVIDENCE:\nKEY_SLICE_AXIAL_MM: 50.0"
        )
        self.assertEqual(extract_choice_from_set(text, valid_choices=("A", "B", "C")), "A")

    def test_extract_choice_from_set_supports_verbose_boxed_final_answer(self) -> None:
        text = "Reasoning mentions option B.\nFinal Answer: The final answer is $\\boxed{C}$"
        self.assertEqual(extract_choice_from_set(text, valid_choices=("A", "B", "C", "D")), "C")

    def test_extract_numbered_answers_prefers_complete_explicit_answer(self) -> None:
        text = (
            "3) Pathological N stage: unavailable.\n"
            "4) Histology: indeterminate.\n"
            "Final Answer:\nANSWER: 1A2E3D4E5F"
        )
        self.assertEqual(extract_numbered_answers(text, question_ids=(1, 2, 3, 4, 5)), "1A2E3D4E5F")

    def test_extract_numbered_answers_rejects_partial_prose_pairs(self) -> None:
        text = "2) Pathological T stage\n3) Pathological N stage\nFinal Answer: A"
        self.assertIsNone(extract_numbered_answers(text, question_ids=(1, 2, 3, 4, 5)))

    def test_extract_numbered_answers_accepts_standalone_compact_line_before_long_check(self) -> None:
        text = "FINAL ANSWER:\n1A2B3D4D5F\n" + "\n".join(f"check line {index}" for index in range(20))
        self.assertEqual(extract_numbered_answers(text, question_ids=(1, 2, 3, 4, 5)), "1A2B3D4D5F")


if __name__ == "__main__":
    unittest.main()
