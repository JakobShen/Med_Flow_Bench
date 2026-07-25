from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional


LUMIERE_EVIDENCE_FIELDS = (
    "T2_PROGRESSION",
    "CURRENT_TARGET_LESION",
    "NEW_TARGET_LESION",
    "CURRENT_NON_TARGET_LESION",
    "NEW_NON_TARGET_LESION",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average_from_summary(summary: Mapping[str, Any], *, average_key: str, count_keys: tuple[str, ...]) -> Optional[float]:
    average_value = _safe_float(_first_present(summary, average_key))
    if average_value is not None:
        return average_value

    total_tool_calls = _safe_float(_first_present(summary, "total_tool_calls"))
    if total_tool_calls is None:
        return None

    for count_key in count_keys:
        count_value = _safe_float(_first_present(summary, count_key))
        if count_value and count_value > 0:
            return total_tool_calls / count_value
    return None


def _is_track_c_summary(summary: Mapping[str, Any]) -> bool:
    script_name = str(_first_present(summary, "script") or "").lower()
    return "track_c" in script_name


def _calls_for_summary(summary: Mapping[str, Any], *, average_key: str, count_keys: tuple[str, ...]) -> Optional[float]:
    if _is_track_c_summary(summary):
        return 0.0
    return _average_from_summary(summary, average_key=average_key, count_keys=count_keys)


def _budget_fields(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "MaxRounds": _safe_float(_first_present(summary, "max_rounds")),
        "MaxTools": _safe_float(_first_present(summary, "max_tools")),
    }


def build_lumiere_paper_table_data(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    evidence_fields = _as_mapping(_as_mapping(summary.get("evidence")).get("fields"))
    per_class = _as_mapping(summary.get("per_class"))
    payload: Dict[str, Optional[float]] = {
        "TaskAcc": _safe_float(_first_present(per_class, "balanced_accuracy", "BalancedAcc")),
        "Strict": _safe_float(_first_present(summary, "strict_avg_evidence_correct")),
        "Calls": _calls_for_summary(
            summary,
            average_key="average_tool_calls_per_question",
            count_keys=("n_questions", "n_scored"),
        ),
    }
    if payload["Strict"] is None:
        payload["Strict"] = _safe_float(
            _first_present(
                _as_mapping(summary.get("evidence")),
                "correct_and_all_scored_evidence_correct_accuracy",
            )
        )

    for field in LUMIERE_EVIDENCE_FIELDS:
        payload[f"{field}_Acc"] = _safe_float(_first_present(_as_mapping(evidence_fields.get(field)), "accuracy"))
    payload.update(_budget_fields(summary))
    return payload


def build_ucsf_pdgm_paper_table_data(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    per_subtype = _as_mapping(_first_present(summary, "per_subtype"))
    payload = {
        "Task": _safe_float(_first_present(per_subtype, "balanced_accuracy", "BalancedAcc", "task_balanced_accuracy"))
        or _safe_float(_first_present(summary, "accuracy")),
        "Strict": _safe_float(_first_present(per_subtype, "balanced_strict_accuracy", "StrictBalancedAcc", "strict_balanced_accuracy"))
        or _safe_float(_first_present(summary, "accuracy_with_evidence")),
        "Loc": _safe_float(_first_present(summary, "Localization_acc", "localization_acc")),
        "Calls": _calls_for_summary(
            summary,
            average_key="average_tool_calls_per_case",
            count_keys=("n_cases", "n_scored"),
        ),
    }
    payload.update(_budget_fields(summary))
    return payload


def build_nsclc_paper_table_data(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    question_metrics = _as_mapping(_first_present(summary, "question_metrics", "per_question"))

    def _question_accuracy(question_id: int) -> Optional[float]:
        return _safe_float(_first_present(_as_mapping(question_metrics.get(str(question_id))), "accuracy"))

    payload = {
        "Task": _safe_float(_first_present(summary, "question_accuracy", "question_micro_accuracy")),
        "Tumor Loc": _question_accuracy(1),
        "T Stage": _question_accuracy(2),
        "Path N": _question_accuracy(3),
        "Hist": _question_accuracy(4),
        "Grade": _question_accuracy(5),
        "Strict": _safe_float(_first_present(summary, "case_with_evidence_partial_accuracy")),
        "Loc (RAS)": _safe_float(_first_present(summary, "Localization_acc", "localization_acc")),
        "Calls": _calls_for_summary(
            summary,
            average_key="average_tool_calls_per_case",
            count_keys=("n_cases", "n_case_scored", "n_scored"),
        ),
    }
    payload.update(_budget_fields(summary))
    return payload


def write_paper_table_data(log_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(log_dir) / "Paper_Table_Data.json"
    output_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return output_path
