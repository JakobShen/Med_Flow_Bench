from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
from typing import Any, Iterable

from qupath_agent_engine.benchmarking.common import (
    build_output_localization_records,
    collect_step_stats,
    extract_json_object,
    normalize_confidence,
    normalize_output_localizations,
    safe_mean,
    safe_rate,
)
from qupath_agent_engine.software.qupath.connector import QuPathConnector

from .dataset import BRACS_LABELS, BracsCaseRecord, normalize_bracs_label
from .prompting import BRACS_LABEL_TO_OPTION, BRACS_OPTION_TO_LABEL

BRACS_NEGATIVE_LABEL = "N"
BRACS_PAPER_SUPERCLASSES = ("benign", "atyp", "malig")
BRACS_LABEL_TO_SUPERCLASS = {
    "N": "benign",
    "PB": "benign",
    "UDH": "benign",
    "FEA": "atyp",
    "ADH": "atyp",
    "DCIS": "malig",
    "IC": "malig",
}


@dataclass(slots=True)
class ParsedBracsAnswer:
    raw_text: str
    selected_option: str | None
    wsi_label: str | None
    confidence: float | None
    brief_rationale: str | None
    localizations: list[dict[str, Any]]
    parse_error: str | None

    @property
    def ok(self) -> bool:
        return self.wsi_label in BRACS_LABELS and self.selected_option in BRACS_OPTION_TO_LABEL

    def summary_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "selected_option": self.selected_option,
            "wsi_label": self.wsi_label,
            "confidence": self.confidence,
            "brief_rationale": self.brief_rationale,
            "localizations": list(self.localizations),
            "parse_error": self.parse_error,
            "ok": self.ok,
        }


def parse_bracs_final_answer(text: str) -> ParsedBracsAnswer:
    raw_text = str(text or "").strip()
    payload = extract_json_object(raw_text)
    selected_option: str | None = None
    wsi_label: str | None = None
    confidence: float | None = None
    brief_rationale: str | None = None
    localizations: list[dict[str, Any]] = []
    parse_error: str | None = None

    if payload is not None:
        selected_option = _normalize_option(payload.get("selected_option"))
        wsi_label = normalize_bracs_label(payload.get("wsi_label"))
        confidence = normalize_confidence(payload.get("confidence"))
        rationale_value = payload.get("brief_rationale")
        brief_rationale = str(rationale_value).strip() if rationale_value is not None else None
        localizations = normalize_output_localizations(payload.get("localizations"))
    else:
        selected_option = _extract_option_from_text(raw_text)
        wsi_label = _extract_label_from_text(raw_text)
        confidence = _extract_confidence_from_text(raw_text)
        brief_rationale = raw_text or None
        parse_error = "The final answer was not valid JSON."

    if wsi_label is None and selected_option is not None:
        wsi_label = BRACS_OPTION_TO_LABEL.get(selected_option)
    if selected_option is None and wsi_label is not None:
        selected_option = BRACS_LABEL_TO_OPTION.get(wsi_label)

    if selected_option is not None and wsi_label is not None:
        expected_label = BRACS_OPTION_TO_LABEL.get(selected_option)
        if expected_label is not None and expected_label != wsi_label:
            parse_error = (
                f"selected_option={selected_option} maps to {expected_label}, but wsi_label={wsi_label}."
            )

    if wsi_label is None:
        parse_error = parse_error or "Could not extract a valid BRACS label from the final answer."
    if selected_option is None and wsi_label is not None:
        selected_option = BRACS_LABEL_TO_OPTION.get(wsi_label)

    return ParsedBracsAnswer(
        raw_text=raw_text,
        selected_option=selected_option,
        wsi_label=wsi_label,
        confidence=confidence,
        brief_rationale=brief_rationale,
        localizations=localizations,
        parse_error=parse_error,
    )


def score_bracs_case(
    *,
    case: BracsCaseRecord,
    final_text: str,
    steps: list[dict[str, Any]],
    connector: QuPathConnector,
    recorded_evidence: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parsed = parse_bracs_final_answer(final_text)
    predicted_label = parsed.wsi_label
    gt_positive_case = case.gt_label != BRACS_NEGATIVE_LABEL
    wsi_label_correct = bool(predicted_label is not None and predicted_label == case.gt_label)
    negative_prediction = predicted_label == BRACS_NEGATIVE_LABEL

    output_localizations = [dict(item) for item in parsed.localizations]
    recorded = build_output_localization_records(output_localizations)
    step_stats = collect_step_stats(steps)
    n_recorded_evidence = len(recorded)

    evidence_results: list[dict[str, Any]] = []
    evidence_errors: list[dict[str, Any]] = []
    evidence_scored = case.qpdata_path is not None
    evidence_label_correct: bool | None = False if evidence_scored else None
    localization_correct = False
    evidence_skip_reason: str | None = None

    if negative_prediction:
        evidence_skip_reason = (
            "negative_predictions_must_not_record_evidence"
            if recorded
            else "not_required_for_negative_prediction"
        )
        evidence_label_correct = not recorded
    elif case.qpdata_path is not None:
        if not recorded:
            evidence_skip_reason = "required_but_missing"
        for evidence in recorded:
            x = evidence.get("center_pixel_x")
            y = evidence.get("center_pixel_y")
            evidence_id = str(
                evidence.get("evidence_id")
                or f"evidence_{len(evidence_results) + len(evidence_errors) + 1:04d}"
            )
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                evidence_errors.append(
                    {
                        "evidence_id": evidence_id,
                        "error": "Evidence center coordinates are missing or invalid.",
                    }
                )
                continue

            score_result = connector.score_point_against_qpdata(
                qpdata_path=case.qpdata_path,
                x=float(x),
                y=float(y),
            )
            if not isinstance(score_result, dict) or not score_result.get("ok"):
                evidence_errors.append(
                    {
                        "evidence_id": evidence_id,
                        "error": (
                            (score_result or {}).get("error")
                            if isinstance(score_result, dict)
                            else "Unknown evidence scoring error."
                        ),
                    }
                )
                continue

            hits = score_result.get("hits") or []
            normalized_hit_labels: list[str] = []
            hit_summaries: list[dict[str, Any]] = []
            if isinstance(hits, list):
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    normalized = normalize_annotation_label(
                        hit.get("classification_normalized") or hit.get("classification")
                    )
                    if normalized:
                        normalized_hit_labels.append(normalized)
                    hit_summaries.append(
                        {
                            "classification": hit.get("classification"),
                            "classification_normalized": normalized,
                            "name": hit.get("name"),
                            "object_id": hit.get("object_id"),
                            "centroid_x": hit.get("centroid_x"),
                            "centroid_y": hit.get("centroid_y"),
                        }
                    )

            supports_gt_label = bool(case.gt_label in normalized_hit_labels)
            evidence_results.append(
                {
                    "evidence_id": evidence_id,
                    "supported_label": evidence.get("supported_label"),
                    "description": evidence.get("description"),
                    "center_pixel_x": x,
                    "center_pixel_y": y,
                    "hit_any_roi": bool(score_result.get("hit_any")),
                    "normalized_hit_labels": normalized_hit_labels,
                    "supports_gt_label": supports_gt_label,
                    "hits": hit_summaries,
                    "localization": evidence.get("localization"),
                }
            )
            if supports_gt_label:
                evidence_label_correct = True
                if gt_positive_case:
                    localization_correct = True
    else:
        evidence_skip_reason = "missing_qpdata"

    return {
        "case_id": case.case_id,
        "excel_row_index": case.excel_row_index,
        "split": case.split,
        "gt_label": case.gt_label,
        "declared_roi_count": case.declared_roi_count,
        "svs_path": str(case.svs_path) if case.svs_path else None,
        "qpdata_path": str(case.qpdata_path) if case.qpdata_path else None,
        "parsed_answer": parsed.summary_dict(),
        "predicted_label": predicted_label,
        "wsi_label_correct": wsi_label_correct,
        "evidence_skip_reason": evidence_skip_reason,
        "evidence_scored": evidence_scored,
        "evidence_label_correct": evidence_label_correct,
        "localization_correct": localization_correct,
        "evidence_results": evidence_results,
        "evidence_errors": evidence_errors,
        "output_localizations": output_localizations,
        "n_output_localizations": len(output_localizations),
        "n_recorded_evidence": n_recorded_evidence,
        "recorded_evidence": recorded,
        **step_stats,
        "joint_correct": (wsi_label_correct and bool(evidence_label_correct)) if evidence_scored else None,
    }


def aggregate_bracs_case_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in results]
    n_cases = len(rows)
    n_wsi_correct = sum(1 for row in rows if row.get("wsi_label_correct"))
    gt_positive_rows = [row for row in rows if row.get("gt_label") != BRACS_NEGATIVE_LABEL]
    n_localization_correct = sum(1 for row in gt_positive_rows if row.get("localization_correct"))
    evidence_scored_rows = [row for row in rows if row.get("evidence_scored")]
    n_evidence_scored = len(evidence_scored_rows)
    n_evidence_correct = sum(1 for row in evidence_scored_rows if row.get("evidence_label_correct"))
    joint_scored_rows = [row for row in evidence_scored_rows if row.get("joint_correct") is not None]
    n_joint_correct = sum(1 for row in joint_scored_rows if row.get("joint_correct"))

    return {
        "n_cases": n_cases,
        "wsi_label_accuracy": safe_rate(n_wsi_correct, n_cases),
        "n_wsi_label_correct": n_wsi_correct,
        "localization_correct": safe_rate(n_localization_correct, len(gt_positive_rows)),
        "n_localization_correct": n_localization_correct,
        "n_gt_positive_cases": len(gt_positive_rows),
        "n_evidence_scored": n_evidence_scored,
        "n_cases_without_qpdata": sum(1 for row in rows if not row.get("evidence_scored")),
        "evidence_label_accuracy": safe_rate(n_evidence_correct, n_evidence_scored),
        "n_evidence_label_correct": n_evidence_correct,
        "joint_accuracy": safe_rate(n_joint_correct, len(joint_scored_rows)),
        "n_joint_correct": n_joint_correct,
        "avg_steps": safe_mean(row.get("n_steps") for row in rows),
        "avg_direct_tools": safe_mean(row.get("n_direct_tools") for row in rows),
        "avg_gui_tools": safe_mean(row.get("n_gui_tools") for row in rows),
        "avg_tool_calls": safe_mean(row.get("n_tool_calls") for row in rows),
        "avg_roi_output_count": safe_mean(row.get("n_recorded_evidence") for row in rows),
    }


def build_bracs_paper_data(
    results: Iterable[dict[str, Any]],
    *,
    benchmark_root: str,
) -> dict[str, Any]:
    rows = [dict(item) for item in results]
    superclass_pairs = [
        (
            BRACS_LABEL_TO_SUPERCLASS.get(str(row.get("gt_label") or "").strip().upper()),
            BRACS_LABEL_TO_SUPERCLASS.get(str(row.get("predicted_label") or "").strip().upper()),
        )
        for row in rows
    ]
    superclass_metrics = _compute_macro_classification_metrics(
        superclass_pairs,
        labels=BRACS_PAPER_SUPERCLASSES,
    )
    superclass_recalls = superclass_metrics["per_label_recall"]

    paper_metrics = {
        "Task": safe_rate(sum(1 for row in rows if row.get("wsi_label_correct")), len(rows)),
        "bAcc": superclass_metrics["balanced_accuracy"],
        "Prec.": superclass_metrics["macro_precision"],
        "Rec.": superclass_metrics["macro_recall"],
        "Benign": superclass_recalls.get("benign"),
        "Atyp.": superclass_recalls.get("atyp"),
        "Malig.": superclass_recalls.get("malig"),
        "Evd.": safe_rate(
            sum(1 for row in rows if row.get("evidence_label_correct") is True),
            sum(1 for row in rows if row.get("evidence_scored")),
        ),
        "Loc.": safe_rate(
            sum(1 for row in rows if row.get("gt_label") != BRACS_NEGATIVE_LABEL and row.get("localization_correct")),
            sum(1 for row in rows if row.get("gt_label") != BRACS_NEGATIVE_LABEL),
        ),
        "Calls": safe_mean(row.get("n_tool_calls") for row in rows),
    }

    return {
        **paper_metrics,
        "benchmark_track": "bracs",
        "benchmark_root": str(benchmark_root),
        "paper_metrics": paper_metrics,
        "metric_notes": {
            "Task": "Exact BRACS slide-label accuracy.",
            "bAcc": "Balanced accuracy over BRACS paper superclasses: benign={N, PB, UDH}, atyp={FEA, ADH}, malig={DCIS, IC}.",
            "Prec.": "Macro precision over the same BRACS paper superclasses.",
            "Rec.": "Macro recall over the same BRACS paper superclasses.",
            "Benign": "Recall for the benign superclass.",
            "Atyp.": "Recall for the atypical superclass.",
            "Malig.": "Recall for the malignant superclass.",
            "Evd.": "Evidence-label accuracy on evidence-scored cases.",
            "Loc.": "Localization hit rate on gt-positive cases.",
            "Calls": "Average tool calls per case (direct + GUI + evidence tools).",
        },
        "counts": {
            "n_cases": len(rows),
            "n_gt_positive_cases": sum(1 for row in rows if row.get("gt_label") != BRACS_NEGATIVE_LABEL),
            "n_evidence_scored_cases": sum(1 for row in rows if row.get("evidence_scored")),
            "n_case_errors": sum(1 for row in rows if row.get("error")),
        },
        "superclass_mapping": {
            "benign": ["N", "PB", "UDH"],
            "atyp": ["FEA", "ADH"],
            "malig": ["DCIS", "IC"],
        },
        "raw_support": {
            "per_superclass_gt_counts": superclass_metrics["per_label_gt_counts"],
            "per_superclass_pred_counts": superclass_metrics["per_label_pred_counts"],
        },
    }


def normalize_annotation_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = normalize_bracs_label(text)
    if normalized is not None:
        return normalized
    text_up = text.upper().replace("_", "-")
    prefix = text_up.split("-", 1)[0]
    normalized = normalize_bracs_label(prefix)
    if normalized is not None:
        return normalized
    tokenized = re.sub(r"[^A-Z0-9]+", " ", text_up)
    for label in sorted(BRACS_LABELS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", tokenized):
            return label
    return None


def _normalize_option(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in BRACS_OPTION_TO_LABEL:
        return text
    return None


def _extract_option_from_text(text: str) -> str | None:
    patterns = [
        r'"selected_option"\s*:\s*"?([A-G])"?',
        r"selected[_\s-]*option\s*[:=]\s*([A-G])",
        r"\boption\s*([A-G])\b",
        r"\b([A-G])\b",
    ]
    upper = str(text or "").upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def _extract_label_from_text(text: str) -> str | None:
    raw = str(text or "")
    direct = re.search(r'"wsi_label"\s*:\s*"?([A-Za-z]+)"?', raw, flags=re.IGNORECASE)
    if direct:
        normalized = normalize_bracs_label(direct.group(1))
        if normalized is not None:
            return normalized
    tokenized = re.sub(r"[^A-Za-z0-9]+", " ", raw.upper())
    for label in sorted(BRACS_LABELS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", tokenized):
            return label
    return None


def _extract_confidence_from_text(text: str) -> float | None:
    match = re.search(r'"confidence"\s*:\s*([0-9.]+)', str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_confidence(match.group(1))


def _compute_macro_classification_metrics(
    pairs: Iterable[tuple[str | None, str | None]],
    *,
    labels: Iterable[str],
) -> dict[str, Any]:
    ordered_labels = [str(label) for label in labels]
    per_label_recall: dict[str, float | None] = {}
    per_label_precision: dict[str, float | None] = {}
    per_label_gt_counts: dict[str, int] = {}
    per_label_pred_counts: dict[str, int] = {}

    valid_pairs = [(gt, pred) for gt, pred in pairs if gt in ordered_labels]
    for label in ordered_labels:
        gt_count = sum(1 for gt, _ in valid_pairs if gt == label)
        pred_count = sum(1 for _, pred in valid_pairs if pred == label)
        tp = sum(1 for gt, pred in valid_pairs if gt == label and pred == label)
        per_label_gt_counts[label] = gt_count
        per_label_pred_counts[label] = pred_count
        per_label_recall[label] = safe_rate(tp, gt_count)
        per_label_precision[label] = safe_rate(tp, pred_count)

    recall_values = [value for value in per_label_recall.values() if value is not None]
    precision_values = [value for value in per_label_precision.values() if value is not None]
    macro_recall = mean(recall_values) if recall_values else None
    macro_precision = mean(precision_values) if precision_values else None

    return {
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_label_recall": per_label_recall,
        "per_label_precision": per_label_precision,
        "per_label_gt_counts": per_label_gt_counts,
        "per_label_pred_counts": per_label_pred_counts,
    }
