from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from radiology.track_c.benchmark_LUMIERE_track_c import (
    _balance_questions_by_label,
    _filter_labels_with_min_count,
    build_parser,
)
from radiology.track_c.benchmark_NSCLC_track_c import _build_case_message as build_nsclc_message
from radiology_track_c.prompting import build_case_message as build_ucsf_message


def _questions(*labels: str):
    return [SimpleNamespace(gt_rano_label=label) for label in labels]


def test_lumiere_label_filter_drops_only_underrepresented_labels() -> None:
    questions = _questions("PD", "PD", "SD", "SD", "PR", "CR")
    kept, n_filtered, selected, counts = _filter_labels_with_min_count(questions, min_count=2)
    assert [question.gt_rano_label for question in kept] == ["PD", "PD", "SD", "SD"]
    assert n_filtered == 2
    assert selected == ["PD", "SD"]
    assert counts == {"PD": 2, "SD": 2, "PR": 1, "CR": 1}


def test_lumiere_balance_interleaves_without_truncating() -> None:
    questions = _questions("PD", "PD", "PD", "SD", "SD")
    balanced = _balance_questions_by_label(questions)
    assert [question.gt_rano_label for question in balanced] == ["PD", "SD", "PD", "SD", "PD"]


def test_lumiere_track_c_reproduction_defaults_match_protocol() -> None:
    args = build_parser().parse_args([])
    assert args.balance_rano_labels is True
    assert args.min_scorable_evidence_fields == 2
    assert args.min_label_count_before_balance == 10


def test_nsclc_prompt_requests_evidence_without_exposing_paths() -> None:
    private_root = Path("/private/user/data/R01-001")
    case = SimpleNamespace(case_id="R01-001", case_root=private_root)
    record = SimpleNamespace(
        display_name="CT diagnostic",
        role="diagnostic_ct",
        modality="CT",
        abs_path=private_root / "series",
    )
    message = build_nsclc_message(
        case=case,
        method="compressed_figures",
        selected_alias="core",
        selected_records=[record],
        artifact_summary_lines=["one montage"],
    )
    assert str(private_root) not in message
    assert "EVIDENCE:" in message
    assert "KEY_SLICE_AXIAL_MM" in message
    assert "ANSWER: 1A2B3C4D5E" in message


def test_ucsf_prompt_uses_file_names_without_exposing_case_path() -> None:
    private_root = Path("/private/user/data/UCSF-PDGM-0001")
    message = build_ucsf_message(
        case_id="UCSF-PDGM-0001",
        case_path=private_root,
        nifti_files=[private_root / "case_T1c.nii.gz"],
        method="compressed_figures",
        artifact_summary_lines=["one montage"],
    )
    assert str(private_root) not in message
    assert "case_T1c.nii.gz" in message
