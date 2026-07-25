from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from PIL import Image

from qupath_agent_engine.benchmarking.common import (
    build_output_localization_records,
    collect_step_stats,
    extract_json_object,
    normalize_confidence,
    normalize_output_localizations,
    safe_mean,
    safe_rate,
)

from .dataset import CAMELYON17_PATIENT_STAGES, CAMELYON17_SLIDE_LABELS, Camelyon17SlideRecord, normalize_camelyon17_slide_label
from .prompting import (
    CAMELYON17_SLIDE_LABEL_TO_OPTION,
    CAMELYON17_SLIDE_OPTION_TO_LABEL,
    CAMELYON17_TUMOR_PRESENCE_LABEL_TO_OPTION,
    CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL,
)


Image.MAX_IMAGE_PIXELS = None
CAMELYON17_TUMOR_PRESENCE_LABELS = tuple(CAMELYON17_TUMOR_PRESENCE_LABEL_TO_OPTION.keys())
CAMELYON17_POSITIVE_SLIDE_LABELS = ("itc", "micro", "macro")
CAMELYON17_EVIDENCE_STATUS_VALUES = (
    "not_required",
    "gt_missing",
    "required_but_missing",
    "forbidden_for_negative_prediction",
    "hit",
    "miss",
)


@dataclass(slots=True)
class ParsedCamelyon17SlideAnswer:
    raw_text: str
    tumor_presence_option: str | None
    tumor_presence: str | None
    largest_metastasis_option: str | None
    largest_metastasis: str | None
    confidence: float | None
    brief_rationale: str | None
    localizations: list[dict[str, Any]]
    parse_error: str | None

    @property
    def predicted_positive(self) -> bool | None:
        if self.tumor_presence == "tumor_present":
            return True
        if self.tumor_presence == "no_tumor":
            return False
        if self.largest_metastasis in CAMELYON17_POSITIVE_SLIDE_LABELS:
            return True
        if self.largest_metastasis == "negative":
            return False
        return None

    @property
    def ok(self) -> bool:
        return (
            self.tumor_presence in CAMELYON17_TUMOR_PRESENCE_LABELS
            and self.largest_metastasis in CAMELYON17_SLIDE_LABEL_TO_OPTION
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "tumor_presence_option": self.tumor_presence_option,
            "tumor_presence": self.tumor_presence,
            "largest_metastasis_option": self.largest_metastasis_option,
            "largest_metastasis": self.largest_metastasis,
            "predicted_positive": self.predicted_positive,
            "confidence": self.confidence,
            "brief_rationale": self.brief_rationale,
            "localizations": list(self.localizations),
            "parse_error": self.parse_error,
            "ok": self.ok,
        }


def parse_camelyon17_slide_final_answer(text: str) -> ParsedCamelyon17SlideAnswer:
    raw_text = str(text or "").strip()
    payload = extract_json_object(raw_text)
    tumor_presence_option: str | None = None
    tumor_presence: str | None = None
    largest_metastasis_option: str | None = None
    largest_metastasis: str | None = None
    confidence: float | None = None
    brief_rationale: str | None = None
    localizations: list[dict[str, Any]] = []
    parse_error: str | None = None

    if payload is not None:
        tumor_presence_option = _normalize_presence_option(payload.get("tumor_presence_option"))
        tumor_presence = _normalize_tumor_presence(payload.get("tumor_presence"))
        largest_metastasis_option = _normalize_slide_option(payload.get("largest_metastasis_option"))
        largest_metastasis = normalize_camelyon17_slide_label(payload.get("largest_metastasis"))
        confidence = normalize_confidence(payload.get("confidence"))
        rationale_value = payload.get("brief_rationale")
        brief_rationale = str(rationale_value).strip() if rationale_value is not None else None
        localizations = normalize_output_localizations(payload.get("localizations"))
    else:
        tumor_presence_option = _extract_presence_option_from_text(raw_text)
        tumor_presence = _extract_presence_from_text(raw_text)
        largest_metastasis_option = _extract_slide_option_from_text(raw_text)
        largest_metastasis = _extract_slide_label_from_text(raw_text)
        confidence = _extract_confidence_from_text(raw_text)
        brief_rationale = raw_text or None
        parse_error = "The final answer was not valid JSON."

    if tumor_presence is None and largest_metastasis is not None:
        tumor_presence = "no_tumor" if largest_metastasis == "negative" else "tumor_present"
    if largest_metastasis is None and tumor_presence == "no_tumor":
        largest_metastasis = "negative"

    if tumor_presence is None and tumor_presence_option is not None:
        tumor_presence = CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL.get(tumor_presence_option)
    if tumor_presence_option is None and tumor_presence is not None:
        tumor_presence_option = CAMELYON17_TUMOR_PRESENCE_LABEL_TO_OPTION.get(tumor_presence)
    if largest_metastasis is None and largest_metastasis_option is not None:
        largest_metastasis = CAMELYON17_SLIDE_OPTION_TO_LABEL.get(largest_metastasis_option)
    if largest_metastasis_option is None and largest_metastasis is not None:
        largest_metastasis_option = CAMELYON17_SLIDE_LABEL_TO_OPTION.get(largest_metastasis)

    if tumor_presence_option is not None and tumor_presence is not None:
        expected_presence = CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL.get(tumor_presence_option)
        if expected_presence is not None and expected_presence != tumor_presence:
            parse_error = (
                f"tumor_presence_option={tumor_presence_option} maps to {expected_presence}, "
                f"but tumor_presence={tumor_presence}."
            )

    if largest_metastasis_option is not None and largest_metastasis is not None:
        expected_label = CAMELYON17_SLIDE_OPTION_TO_LABEL.get(largest_metastasis_option)
        if expected_label is not None and expected_label != largest_metastasis:
            parse_error = (
                f"largest_metastasis_option={largest_metastasis_option} maps to {expected_label}, "
                f"but largest_metastasis={largest_metastasis}."
            )

    if tumor_presence == "no_tumor" and largest_metastasis in CAMELYON17_POSITIVE_SLIDE_LABELS:
        parse_error = "tumor_presence=no_tumor is inconsistent with a positive largest_metastasis label."
    if tumor_presence == "tumor_present" and largest_metastasis == "negative":
        parse_error = "tumor_presence=tumor_present is inconsistent with largest_metastasis=negative."

    if tumor_presence is None:
        parse_error = parse_error or "Could not extract a valid tumor_presence answer from the final response."
    if largest_metastasis is None:
        parse_error = parse_error or "Could not extract a valid largest_metastasis answer from the final response."

    return ParsedCamelyon17SlideAnswer(
        raw_text=raw_text,
        tumor_presence_option=tumor_presence_option,
        tumor_presence=tumor_presence,
        largest_metastasis_option=largest_metastasis_option,
        largest_metastasis=largest_metastasis,
        confidence=confidence,
        brief_rationale=brief_rationale,
        localizations=localizations,
        parse_error=parse_error,
    )


def score_camelyon17_slide_case(
    *,
    case: Camelyon17SlideRecord,
    final_text: str,
    steps: list[dict[str, Any]],
    recorded_evidence: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_camelyon17_slide_final_answer(final_text)
    predicted_slide_label = parsed.largest_metastasis
    predicted_tumor_presence = parsed.tumor_presence
    predicted_positive = parsed.predicted_positive
    gt_tumor_presence = "tumor_present" if case.gt_tumor_present else "no_tumor"

    raw_tumor_presence_correct = bool(predicted_tumor_presence is not None and predicted_tumor_presence == gt_tumor_presence)
    raw_slide_label_correct = bool(predicted_slide_label is not None and predicted_slide_label == case.gt_slide_label)

    output_localizations = [dict(item) for item in parsed.localizations]
    recorded = build_output_localization_records(output_localizations)
    step_stats = collect_step_stats(steps)
    n_recorded_evidence = len(recorded)

    evidence_gt_available = case.has_evidence_source
    evidence_required_by_prediction = bool(predicted_positive)
    evidence_scored = bool(evidence_required_by_prediction and evidence_gt_available)
    forbidden_negative_evidence = bool(not evidence_required_by_prediction and n_recorded_evidence > 0)
    evidence_skip_reason = None

    evidence_results: list[dict[str, Any]] = []
    evidence_errors: list[dict[str, Any]] = []
    any_evidence_hit_tumor = False

    if evidence_gt_available and recorded:
        for evidence in recorded:
            evidence_id = str(
                evidence.get("evidence_id")
                or f"evidence_{len(evidence_results) + len(evidence_errors) + 1:04d}"
            )
            x = evidence.get("center_pixel_x")
            y = evidence.get("center_pixel_y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                evidence_errors.append(
                    {
                        "evidence_id": evidence_id,
                        "error": "Evidence center coordinates are missing or invalid.",
                    }
                )
                continue
            judge = judge_camelyon17_evidence(case=case, evidence=evidence)
            judge["evidence_id"] = evidence_id
            judge["supported_label"] = evidence.get("supported_label")
            judge["description"] = evidence.get("description")
            judge["localization"] = evidence.get("localization")
            evidence_results.append(judge)
            if judge.get("hit_tumor"):
                any_evidence_hit_tumor = True

    supported_tumor_presence_correct: bool | None
    supported_slide_label_correct: bool | None
    evidence_status: str
    evidence_correct: bool | None
    if forbidden_negative_evidence:
        evidence_status = "forbidden_for_negative_prediction"
        evidence_correct = False
        supported_tumor_presence_correct = None
        supported_slide_label_correct = None
        evidence_skip_reason = "negative_predictions_must_not_record_evidence"
    elif not evidence_required_by_prediction:
        evidence_status = "not_required"
        evidence_correct = None
        supported_tumor_presence_correct = None
        supported_slide_label_correct = None
        evidence_skip_reason = "not_required"
    elif not evidence_gt_available:
        evidence_status = "gt_missing"
        evidence_correct = None
        supported_tumor_presence_correct = None
        supported_slide_label_correct = None
        evidence_skip_reason = "missing_mask_and_xml"
    elif n_recorded_evidence == 0:
        evidence_status = "required_but_missing"
        evidence_correct = False
        supported_tumor_presence_correct = False
        supported_slide_label_correct = False
        evidence_skip_reason = "required_but_missing"
    else:
        evidence_correct = bool(case.gt_tumor_present and any_evidence_hit_tumor)
        evidence_status = "hit" if evidence_correct else "miss"
        supported_tumor_presence_correct = bool(raw_tumor_presence_correct and evidence_correct)
        supported_slide_label_correct = bool(raw_slide_label_correct and evidence_correct)

    if evidence_status not in CAMELYON17_EVIDENCE_STATUS_VALUES:
        raise ValueError(f"Unsupported CAMELYON17 evidence_status: {evidence_status}")

    case_result = {
        "case_id": case.case_id,
        "csv_row_index": case.csv_row_index,
        "slide_filename": case.slide_filename,
        "patient_id": case.patient_id,
        "node_index": case.node_index,
        "center": case.center,
        "patient_stage": case.patient_stage,
        "gt_slide_label": case.gt_slide_label,
        "gt_tumor_presence": gt_tumor_presence,
        "image_path": str(case.image_path) if case.image_path else None,
        "mask_path": str(case.mask_path) if case.mask_path else None,
        "xml_path": str(case.xml_path) if case.xml_path else None,
        "parsed_answer": parsed.summary_dict(),
        "predicted_slide_label": predicted_slide_label,
        "predicted_tumor_presence": predicted_tumor_presence,
        "predicted_positive": predicted_positive,
        "raw_tumor_presence_correct": raw_tumor_presence_correct,
        "raw_slide_label_correct": raw_slide_label_correct,
        "evidence_gt_available": evidence_gt_available,
        "evidence_required_by_prediction": evidence_required_by_prediction,
        "evidence_scored": evidence_scored,
        "evidence_status": evidence_status,
        "evidence_skip_reason": evidence_skip_reason,
        "evidence_correct": evidence_correct,
        "any_evidence_hit_tumor": any_evidence_hit_tumor if evidence_gt_available else None,
        "evidence_results": evidence_results,
        "evidence_errors": evidence_errors,
        "supported_tumor_presence_correct": supported_tumor_presence_correct,
        "supported_slide_label_correct": supported_slide_label_correct,
        "output_localizations": output_localizations,
        "n_output_localizations": len(output_localizations),
        "n_recorded_evidence": n_recorded_evidence,
        "recorded_evidence": recorded,
        **step_stats,
    }

    judge_log = {
        "case_id": case.case_id,
        "slide_filename": case.slide_filename,
        "gt_slide_label": case.gt_slide_label,
        "gt_tumor_presence": gt_tumor_presence,
        "predicted_slide_label": predicted_slide_label,
        "predicted_tumor_presence": predicted_tumor_presence,
        "predicted_positive": predicted_positive,
        "evidence_required_by_prediction": evidence_required_by_prediction,
        "evidence_gt_available": evidence_gt_available,
        "evidence_scored": evidence_scored,
        "evidence_status": evidence_status,
        "evidence_skip_reason": evidence_skip_reason,
        "any_evidence_hit_tumor": any_evidence_hit_tumor if evidence_gt_available else None,
        "evidence_correct": evidence_correct,
        "evidence_results": evidence_results,
        "evidence_errors": evidence_errors,
        "raw_tumor_presence_correct": raw_tumor_presence_correct,
        "raw_slide_label_correct": raw_slide_label_correct,
        "supported_tumor_presence_correct": supported_tumor_presence_correct,
        "supported_slide_label_correct": supported_slide_label_correct,
        "parsed_answer": parsed.summary_dict(),
    }

    return case_result, judge_log


def aggregate_camelyon17_slide_case_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in results]
    n_cases = len(rows)
    n_raw_tumor_presence_correct = sum(1 for row in rows if row.get("raw_tumor_presence_correct"))
    n_raw_slide_label_correct = sum(1 for row in rows if row.get("raw_slide_label_correct"))
    gt_positive_rows = [row for row in rows if row.get("gt_tumor_presence") == "tumor_present"]
    n_question_correct = sum(
        1
        for row in rows
        if row.get("raw_tumor_presence_correct") and row.get("raw_slide_label_correct")
    )
    n_correct_with_evidence = sum(
        1
        for row in rows
        if row.get("raw_tumor_presence_correct") and row.get("raw_slide_label_correct") and row.get("evidence_correct")
    )
    evidence_scored_rows = [row for row in rows if row.get("evidence_scored")]
    positive_supported_rows = [
        row
        for row in rows
        if row.get("evidence_required_by_prediction") and row.get("evidence_gt_available")
    ]

    return {
        "n_cases": n_cases,
        "raw_tumor_presence_accuracy": safe_rate(n_raw_tumor_presence_correct, n_cases),
        "n_raw_tumor_presence_correct": n_raw_tumor_presence_correct,
        "raw_slide_label_accuracy": safe_rate(n_raw_slide_label_correct, n_cases),
        "n_raw_slide_label_correct": n_raw_slide_label_correct,
        "Question_Acc": safe_rate(n_question_correct, n_cases),
        "n_question_correct": n_question_correct,
        "Correct_with_Evidence": safe_rate(n_correct_with_evidence, n_cases),
        "n_correct_with_evidence": n_correct_with_evidence,
        "localization_correct": safe_rate(
            sum(1 for row in gt_positive_rows if row.get("evidence_correct")),
            len(gt_positive_rows),
        ),
        "n_localization_correct": sum(1 for row in gt_positive_rows if row.get("evidence_correct")),
        "n_gt_positive_cases": len(gt_positive_rows),
        "n_cases_with_evidence_gt": sum(1 for row in rows if row.get("evidence_gt_available")),
        "n_predicted_positive": sum(1 for row in rows if row.get("predicted_positive") is True),
        "n_predicted_positive_with_evidence_gt": len(evidence_scored_rows),
        "n_predicted_positive_without_evidence_gt": sum(
            1 for row in rows if row.get("evidence_required_by_prediction") and not row.get("evidence_gt_available")
        ),
        "evidence_hit_rate_predicted_positive_scored": safe_rate(
            sum(1 for row in evidence_scored_rows if row.get("evidence_correct")),
            len(evidence_scored_rows),
        ),
        "n_evidence_correct": sum(1 for row in evidence_scored_rows if row.get("evidence_correct")),
        "positive_supported_tumor_presence_accuracy": safe_rate(
            sum(1 for row in positive_supported_rows if row.get("supported_tumor_presence_correct")),
            len(positive_supported_rows),
        ),
        "n_positive_supported_tumor_presence_correct": sum(
            1 for row in positive_supported_rows if row.get("supported_tumor_presence_correct")
        ),
        "positive_supported_slide_label_accuracy": safe_rate(
            sum(1 for row in positive_supported_rows if row.get("supported_slide_label_correct")),
            len(positive_supported_rows),
        ),
        "n_positive_supported_slide_label_correct": sum(
            1 for row in positive_supported_rows if row.get("supported_slide_label_correct")
        ),
        "n_positive_supported_cases": len(positive_supported_rows),
        "n_positive_supported_metric_excluded_cases": n_cases - len(positive_supported_rows),
        "avg_steps": safe_mean(row.get("n_steps") for row in rows),
        "avg_direct_tools": safe_mean(row.get("n_direct_tools") for row in rows),
        "avg_gui_tools": safe_mean(row.get("n_gui_tools") for row in rows),
        "avg_tool_calls": safe_mean(row.get("n_tool_calls") for row in rows),
        "avg_roi_output_count": safe_mean(row.get("n_recorded_evidence") for row in rows),
    }


def build_camelyon17_slide_paper_data(
    results: Iterable[dict[str, Any]],
    *,
    benchmark_root: str,
) -> dict[str, Any]:
    rows = [dict(item) for item in results]
    tumor_presence_pairs = [
        (
            str(row.get("gt_tumor_presence") or "").strip(),
            str(row.get("predicted_tumor_presence") or "").strip() or None,
        )
        for row in rows
    ]
    tumor_presence_metrics = _compute_balanced_binary_metrics(
        tumor_presence_pairs,
        negative_label="no_tumor",
        positive_label="tumor_present",
    )
    patient_stage_summary = _build_camelyon17_patient_stage_summary(rows)

    paper_metrics = {
        "Task": safe_rate(sum(1 for row in rows if row.get("raw_slide_label_correct")), len(rows)),
        "Pat.QWK": patient_stage_summary["patient_qwk"],
        "bAcc": tumor_presence_metrics["balanced_accuracy"],
        "Prec.": tumor_presence_metrics["positive_precision"],
        "Rec.": tumor_presence_metrics["positive_recall"],
        "Evd.": safe_rate(
            sum(1 for row in rows if row.get("evidence_correct") is True),
            sum(1 for row in rows if row.get("evidence_scored")),
        ),
        "Loc.": safe_rate(
            sum(1 for row in rows if row.get("gt_tumor_presence") == "tumor_present" and row.get("evidence_correct")),
            sum(1 for row in rows if row.get("gt_tumor_presence") == "tumor_present"),
        ),
        "Calls": safe_mean(row.get("n_tool_calls") for row in rows),
    }

    return {
        **paper_metrics,
        "benchmark_track": "camelyon17_slide",
        "benchmark_root": str(benchmark_root),
        "paper_metrics": paper_metrics,
        "metric_notes": {
            "Task": "Exact CAMELYON17 slide-label accuracy.",
            "Pat.QWK": "Quadratic weighted kappa over patient stages inferred from complete 5-node patient predictions only.",
            "bAcc": "Balanced accuracy for binary tumor-presence detection (tumor_present vs no_tumor).",
            "Prec.": "Positive-class precision for binary tumor-presence detection.",
            "Rec.": "Positive-class recall for binary tumor-presence detection.",
            "Evd.": "Evidence hit rate on evidence-scored predicted-positive cases.",
            "Loc.": "Localization hit rate on gt-positive slides.",
            "Calls": "Average tool calls per case (direct + GUI + evidence tools).",
        },
        "counts": {
            "n_cases": len(rows),
            "n_gt_positive_cases": sum(1 for row in rows if row.get("gt_tumor_presence") == "tumor_present"),
            "n_evidence_scored_cases": sum(1 for row in rows if row.get("evidence_scored")),
            "n_case_errors": sum(1 for row in rows if row.get("error")),
            "n_patient_qwk_scored": patient_stage_summary["n_patient_qwk_scored"],
            "n_complete_patients_seen": patient_stage_summary["n_complete_patients_seen"],
        },
        "patient_stage_summary": patient_stage_summary,
        "raw_support": {
            "tumor_presence_gt_counts": tumor_presence_metrics["gt_counts"],
            "tumor_presence_pred_counts": tumor_presence_metrics["pred_counts"],
        },
    }


def judge_camelyon17_evidence(*, case: Camelyon17SlideRecord, evidence: dict[str, Any]) -> dict[str, Any]:
    x = float(evidence.get("center_pixel_x"))
    y = float(evidence.get("center_pixel_y"))
    image_width, image_height = _resolve_image_dimensions(case=case, evidence=evidence)

    source_results: dict[str, dict[str, Any]] = {}
    source_errors: list[Exception] = []
    if case.mask_path is not None:
        try:
            source_results["mask"] = _judge_point_against_mask(
                mask_path=case.mask_path,
                x=x,
                y=y,
                image_width=image_width,
                image_height=image_height,
            )
        except Exception as exc:
            source_results["mask"] = {
                "ok": False,
                "source_path": str(case.mask_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            source_errors.append(exc)
    if case.xml_path is not None:
        try:
            source_results["xml"] = _judge_point_against_xml(xml_path=case.xml_path, x=x, y=y)
        except Exception as exc:
            source_results["xml"] = {
                "ok": False,
                "source_path": str(case.xml_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            source_errors.append(exc)

    successful_sources = [source for source in source_results.values() if source.get("ok") is not False]
    if not successful_sources and source_errors:
        raise source_errors[0]

    hit_tumor = any(bool(source.get("hit_tumor")) for source in successful_sources)
    return {
        "center_pixel_x": x,
        "center_pixel_y": y,
        "image_width": image_width,
        "image_height": image_height,
        "hit_tumor": hit_tumor,
        "judged_sources": source_results,
    }


@lru_cache(maxsize=256)
def _cached_xml_polygons(path_text: str) -> dict[str, list[dict[str, Any]]]:
    xml_path = Path(path_text)
    root = ET.parse(xml_path).getroot()
    tumor: list[dict[str, Any]] = []
    exclusion: list[dict[str, Any]] = []
    for annotation in root.findall(".//Annotation"):
        group = str(annotation.attrib.get("PartOfGroup") or "").strip().lower()
        name = str(annotation.attrib.get("Name") or "")
        coords = []
        for coord in annotation.findall(".//Coordinate"):
            try:
                x = float(coord.attrib["X"])
                y = float(coord.attrib["Y"])
            except Exception:
                continue
            coords.append((x, y))
        if len(coords) < 3:
            continue
        bbox = _polygon_bbox(coords)
        payload = {
            "name": name,
            "points": coords,
            "bbox": bbox,
        }
        if group == "tumor":
            tumor.append(payload)
        elif group == "exclusion":
            exclusion.append(payload)
    return {"tumor": tumor, "exclusion": exclusion}


def _judge_point_against_xml(*, xml_path: Path, x: float, y: float) -> dict[str, Any]:
    polygons = _cached_xml_polygons(str(xml_path.resolve()))
    tumor_hits = []
    exclusion_hits = []
    for polygon in polygons.get("tumor", []):
        if _bbox_contains_point(polygon["bbox"], x, y) and _point_in_polygon(x, y, polygon["points"]):
            tumor_hits.append(polygon["name"])
    for polygon in polygons.get("exclusion", []):
        if _bbox_contains_point(polygon["bbox"], x, y) and _point_in_polygon(x, y, polygon["points"]):
            exclusion_hits.append(polygon["name"])
    return {
        "ok": True,
        "source_path": str(xml_path),
        "n_tumor_polygons": len(polygons.get("tumor", [])),
        "n_exclusion_polygons": len(polygons.get("exclusion", [])),
        "tumor_hit_names": tumor_hits,
        "exclusion_hit_names": exclusion_hits,
        "hit_tumor": bool(tumor_hits) and not bool(exclusion_hits),
    }


def _judge_point_against_mask(
    *,
    mask_path: Path,
    x: float,
    y: float,
    image_width: int | None,
    image_height: int | None,
) -> dict[str, Any]:
    candidate_frame_indices = _select_mask_frame_candidates(str(mask_path.resolve()))
    pixel_value = None
    mask_width = 0
    mask_height = 0
    mask_x = 0
    mask_y = 0
    mask_frame_index = None
    mask_frame_count = 0
    last_error: Exception | None = None
    with Image.open(mask_path) as image:
        mask_frame_count = int(getattr(image, "n_frames", 1) or 1)
        for frame_index in candidate_frame_indices:
            try:
                image.seek(frame_index)
            except EOFError:
                continue
            mask_width, mask_height = image.size
            mask_x = _rescale_coordinate(x, src_extent=image_width, dst_extent=mask_width)
            mask_y = _rescale_coordinate(y, src_extent=image_height, dst_extent=mask_height)
            try:
                pixel_value = image.getpixel((mask_x, mask_y))
                mask_frame_index = int(frame_index)
                break
            except OSError as exc:
                last_error = exc
                continue
    if pixel_value is None:
        if last_error is not None:
            raise last_error
        raise OSError(f"Failed to read CAMELYON17 mask pixel from {mask_path}")
    if isinstance(pixel_value, tuple):
        pixel_scalar = pixel_value[0]
    else:
        pixel_scalar = pixel_value
    try:
        pixel_scalar_int = int(pixel_scalar)
    except Exception:
        pixel_scalar_int = None
    return {
        "ok": True,
        "source_path": str(mask_path),
        "mask_width": mask_width,
        "mask_height": mask_height,
        "mask_pixel_x": mask_x,
        "mask_pixel_y": mask_y,
        "mask_frame_index": mask_frame_index,
        "mask_frame_count": mask_frame_count,
        "pixel_value": pixel_scalar_int,
        "hit_tumor": bool(pixel_scalar_int is not None and pixel_scalar_int > 0),
    }


@lru_cache(maxsize=256)
def _select_mask_frame_candidates(path_text: str) -> tuple[int, ...]:
    mask_path = Path(path_text)
    with Image.open(mask_path) as image:
        n_frames = int(getattr(image, "n_frames", 1) or 1)
        preferred: list[int] = []
        fallback: list[int] = []
        for frame_index in range(n_frames):
            try:
                image.seek(frame_index)
            except EOFError:
                continue
            if _mask_frame_has_complete_storage(image):
                preferred.append(frame_index)
            else:
                fallback.append(frame_index)
    ordered = [*preferred, *fallback]
    return tuple(ordered or [0])


def _mask_frame_has_complete_storage(image: Image.Image) -> bool:
    tag_v2 = getattr(image, "tag_v2", None)
    if tag_v2 is None:
        return True
    tile_byte_counts = tag_v2.get(325)
    tile_offsets = tag_v2.get(324)
    if tile_byte_counts is None or tile_offsets is None:
        return True
    return _all_positive_tag_values(tile_byte_counts) and _all_positive_tag_values(tile_offsets)


def _all_positive_tag_values(values: Any) -> bool:
    if isinstance(values, (list, tuple)):
        return all(_tag_value_positive(value) for value in values)
    return _tag_value_positive(values)


def _tag_value_positive(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False


def _resolve_image_dimensions(*, case: Camelyon17SlideRecord, evidence: dict[str, Any]) -> tuple[int | None, int | None]:
    width = evidence.get("image_width")
    height = evidence.get("image_height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return int(round(width)), int(round(height))
    image_path = case.image_path
    if image_path is None:
        return None, None
    with Image.open(image_path) as image:
        try:
            image.seek(0)
        except EOFError:
            pass
        return int(image.size[0]), int(image.size[1])


def _rescale_coordinate(coord: float, *, src_extent: int | None, dst_extent: int) -> int:
    if dst_extent <= 1:
        return 0
    if src_extent is None or src_extent <= 1:
        return int(max(0, min(dst_extent - 1, round(coord))))
    clamped = max(0.0, min(float(coord), float(src_extent - 1)))
    return int(round(clamped * float(dst_extent - 1) / float(src_extent - 1)))


def _polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_contains_point(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    min_x, min_y, max_x, max_y = bbox
    return min_x <= x <= max_x and min_y <= y <= max_y


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = ((y1 > y) != (y2 > y))
        if not intersects:
            continue
        denominator = y2 - y1
        if denominator == 0:
            continue
        x_intersection = (x2 - x1) * (y - y1) / denominator + x1
        if x_intersection == x:
            return True
        if x_intersection > x:
            inside = not inside
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float, eps: float = 1e-6) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -eps:
        return False
    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot - squared_length > eps:
        return False
    return True


def _normalize_presence_option(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL else None


def _normalize_slide_option(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in CAMELYON17_SLIDE_OPTION_TO_LABEL else None


def _normalize_tumor_presence(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "no_tumor": "no_tumor",
        "no tumor": "no_tumor",
        "negative": "no_tumor",
        "tumor_present": "tumor_present",
        "tumor present": "tumor_present",
        "positive": "tumor_present",
        "tumor": "tumor_present",
    }
    return aliases.get(text)


def _extract_presence_option_from_text(text: str) -> str | None:
    patterns = [
        r'"tumor_presence_option"\s*:\s*"?([AB])"?',
        r"tumor[_\s-]*presence[_\s-]*option\s*[:=]\s*([AB])",
        r"\bpresence[_\s-]*option\s*([AB])\b",
    ]
    upper = str(text or "").upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def _extract_presence_from_text(text: str) -> str | None:
    raw = str(text or "")
    direct = re.search(r'"tumor_presence"\s*:\s*"?([A-Za-z_ ]+)"?', raw, flags=re.IGNORECASE)
    if direct:
        normalized = _normalize_tumor_presence(direct.group(1))
        if normalized is not None:
            return normalized
    upper = raw.upper()
    if re.search(r"\b(NO[ _-]?TUMOR|NEGATIVE)\b", upper):
        return "no_tumor"
    if re.search(r"\b(TUMOR[ _-]?PRESENT|POSITIVE)\b", upper):
        return "tumor_present"
    return None


def _extract_slide_option_from_text(text: str) -> str | None:
    patterns = [
        r'"largest_metastasis_option"\s*:\s*"?([A-D])"?',
        r"largest[_\s-]*metastasis[_\s-]*option\s*[:=]\s*([A-D])",
    ]
    upper = str(text or "").upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def _extract_slide_label_from_text(text: str) -> str | None:
    raw = str(text or "")
    direct = re.search(r'"largest_metastasis"\s*:\s*"?([A-Za-z_ ]+)"?', raw, flags=re.IGNORECASE)
    if direct:
        normalized = normalize_camelyon17_slide_label(direct.group(1))
        if normalized is not None:
            return normalized
    tokenized = re.sub(r"[^A-Za-z0-9]+", " ", raw.lower())
    for label in sorted(CAMELYON17_SLIDE_LABEL_TO_OPTION, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", tokenized):
            return label
    return None


def _extract_confidence_from_text(text: str) -> float | None:
    match = re.search(r'"confidence"\s*:\s*([0-9.]+)', str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_confidence(match.group(1))


def _build_camelyon17_patient_stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patient_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patient_id = str(row.get("patient_id") or "").strip().lower()
        if not patient_id:
            continue
        patient_rows.setdefault(patient_id, []).append(row)

    patient_predictions: list[dict[str, Any]] = []
    gt_stages: list[str] = []
    predicted_stages: list[str] = []

    for patient_id, items in sorted(patient_rows.items()):
        gt_stage = str(items[0].get("patient_stage") or "").strip().lower() or None
        unique_nodes = {
            int(node_index)
            for node_index in (item.get("node_index") for item in items)
            if isinstance(node_index, int)
        }
        complete_coverage = unique_nodes == {0, 1, 2, 3, 4}
        predicted_slide_labels = [str(item.get("predicted_slide_label") or "").strip().lower() for item in items]
        has_complete_predictions = complete_coverage and all(label in CAMELYON17_SLIDE_LABELS for label in predicted_slide_labels)
        predicted_stage = _predict_camelyon17_patient_stage(predicted_slide_labels) if has_complete_predictions else None
        patient_predictions.append(
            {
                "patient_id": patient_id,
                "gt_stage": gt_stage,
                "predicted_stage": predicted_stage,
                "n_slides_seen": len(items),
                "unique_nodes_seen": sorted(unique_nodes),
                "complete_coverage": complete_coverage,
            }
        )
        if gt_stage in CAMELYON17_PATIENT_STAGES and predicted_stage in CAMELYON17_PATIENT_STAGES:
            gt_stages.append(gt_stage)
            predicted_stages.append(predicted_stage)

    return {
        "patient_qwk": _quadratic_weighted_kappa(
            gt_stages,
            predicted_stages,
            labels=CAMELYON17_PATIENT_STAGES,
        ),
        "n_patient_qwk_scored": len(gt_stages),
        "n_complete_patients_seen": sum(1 for row in patient_predictions if row.get("complete_coverage")),
        "patients": patient_predictions,
    }


def _predict_camelyon17_patient_stage(predicted_slide_labels: Iterable[str]) -> str | None:
    labels = [str(label).strip().lower() for label in predicted_slide_labels if str(label).strip()]
    if len(labels) != 5 or any(label not in CAMELYON17_SLIDE_LABELS for label in labels):
        return None

    macro_count = sum(1 for label in labels if label == "macro")
    micro_count = sum(1 for label in labels if label == "micro")
    itc_count = sum(1 for label in labels if label == "itc")
    positive_node_count = macro_count + micro_count

    if macro_count > 0:
        return "pn2" if positive_node_count >= 4 else "pn1"
    if micro_count > 0:
        return "pn1mi"
    if itc_count > 0:
        return "pn0(i+)"
    return "pn0"


def _compute_balanced_binary_metrics(
    pairs: Iterable[tuple[str | None, str | None]],
    *,
    negative_label: str,
    positive_label: str,
) -> dict[str, Any]:
    valid_pairs = [
        (gt, pred)
        for gt, pred in pairs
        if gt in {negative_label, positive_label}
    ]
    tp = sum(1 for gt, pred in valid_pairs if gt == positive_label and pred == positive_label)
    gt_negative_total = sum(1 for gt, _ in valid_pairs if gt == negative_label)
    gt_positive_total = sum(1 for gt, _ in valid_pairs if gt == positive_label)
    pred_positive_total = sum(1 for _, pred in valid_pairs if pred == positive_label)
    tn = sum(1 for gt, pred in valid_pairs if gt == negative_label and pred == negative_label)

    negative_recall = safe_rate(tn, gt_negative_total)
    positive_recall = safe_rate(tp, gt_positive_total)
    recall_values = [value for value in (negative_recall, positive_recall) if value is not None]

    return {
        "balanced_accuracy": mean(recall_values) if recall_values else None,
        "positive_precision": safe_rate(tp, pred_positive_total),
        "positive_recall": positive_recall,
        "gt_counts": {
            negative_label: gt_negative_total,
            positive_label: gt_positive_total,
        },
        "pred_counts": {
            negative_label: sum(1 for _, pred in valid_pairs if pred == negative_label),
            positive_label: pred_positive_total,
        },
    }


def _quadratic_weighted_kappa(
    gt_labels: Iterable[str],
    predicted_labels: Iterable[str],
    *,
    labels: Iterable[str],
) -> float | None:
    ordered_labels = [str(label) for label in labels]
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    pairs = [
        (label_to_index[gt], label_to_index[pred])
        for gt, pred in zip(gt_labels, predicted_labels)
        if gt in label_to_index and pred in label_to_index
    ]
    if not pairs:
        return None

    n_labels = len(ordered_labels)
    observed = [[0.0 for _ in range(n_labels)] for _ in range(n_labels)]
    gt_hist = [0.0 for _ in range(n_labels)]
    pred_hist = [0.0 for _ in range(n_labels)]
    for gt_index, pred_index in pairs:
        observed[gt_index][pred_index] += 1.0
        gt_hist[gt_index] += 1.0
        pred_hist[pred_index] += 1.0

    total = float(len(pairs))
    if total <= 0:
        return None

    numerator = 0.0
    denominator = 0.0
    normalizer = float((n_labels - 1) ** 2) if n_labels > 1 else 1.0
    for i in range(n_labels):
        for j in range(n_labels):
            weight = ((i - j) ** 2) / normalizer
            expected = (gt_hist[i] * pred_hist[j]) / total
            numerator += weight * observed[i][j]
            denominator += weight * expected

    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return 1.0 - (numerator / denominator)
