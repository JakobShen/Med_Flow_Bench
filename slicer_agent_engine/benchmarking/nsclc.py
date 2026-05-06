from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .check_segment import SegmentGroundTruth
from ..load_helper import CaseSeriesCatalog, build_nsclc_case_catalog
from ..toolpacks import ToolPackSelection


QUESTION_IDS = [1, 2, 3, 4, 5]
QUESTION_NAMES = {
    1: "Tumor location",
    2: "Pathological T stage",
    3: "Pathological N stage",
    4: "Histology",
    5: "Histopathological Grade",
}
EVIDENCE_VIEWS = ["axial", "sagittal", "coronal"]

LOCATION_OPTIONS = {
    "A": "Right upper lobe",
    "B": "Right middle lobe",
    "C": "Right lower lobe",
    "D": "Left upper lobe",
    "E": "Left lower lobe",
    "F": "Lingula",
    "G": "No tumor",
}

T_STAGE_OPTIONS = {
    "A": "T1",
    "B": "T2",
    "C": "T3",
    "D": "T4",
    "E": "Not available / not collected",
    "F": "No tumor",
}

N_STAGE_OPTIONS = {
    "A": "N0",
    "B": "N1",
    "C": "N2",
    "D": "Not available / not collected",
    "E": "No tumor",
}

HISTOLOGY_OPTIONS = {
    "A": "Adenocarcinoma",
    "B": "Squamous cell carcinoma",
    "C": "Small cell carcinoma",
    "D": "Other / indeterminate NSCLC",
    "E": "No tumor",
}

GRADE_OPTIONS = {
    "A": "G1 Well differentiated",
    "B": "G2 Moderately differentiated",
    "C": "G3 Poorly differentiated",
    "D": "Other, Type I: Well to moderately differentiated",
    "E": "Other, Type II: Moderately to poorly differentiated",
    "F": "No tumor",
}


def _split_numbered_answers(compact: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    s = str(compact or "").upper()
    i = 0
    while i + 1 < len(s):
        if s[i].isdigit() and s[i + 1].isalpha():
            out[int(s[i])] = s[i + 1]
            i += 2
            continue
        i += 1
    return out


def _compact_from_map(answer_map: Dict[int, Optional[str]]) -> Optional[str]:
    parts: List[str] = []
    for qid in QUESTION_IDS:
        a = answer_map.get(qid)
        if a is None:
            continue
        parts.append(f"{qid}{a}")
    return "".join(parts) if parts else None


def _score_questions(
    *,
    gt_map: Dict[int, Optional[str]],
    pred_compact: Optional[str],
    q_scored: Dict[int, int],
    q_correct: Dict[int, int],
) -> Tuple[Dict[str, Any], int, int]:
    pred_map = _split_numbered_answers(pred_compact)
    per_question: Dict[str, Any] = {}
    n_questions_scored = 0
    n_questions_correct = 0
    for qid in QUESTION_IDS:
        gt_q = gt_map.get(qid)
        pred_q = pred_map.get(qid)
        scored_q = gt_q is not None
        correct_q = bool(scored_q and pred_q == gt_q)
        if scored_q:
            q_scored[qid] += 1
            n_questions_scored += 1
            if correct_q:
                q_correct[qid] += 1
                n_questions_correct += 1
        per_question[str(qid)] = {
            "name": QUESTION_NAMES[qid],
            "gt": gt_q,
            "pred": pred_q,
            "scored": scored_q,
            "correct": correct_q,
        }
    return per_question, n_questions_scored, n_questions_correct


def _segment_gt_summary(gt: Optional[SegmentGroundTruth]) -> Optional[Dict[str, Any]]:
    if gt is None:
        return None
    return {
        "has_segment": bool(gt.has_segment),
        "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
        "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
        "segmentation_name": gt.segmentation_name,
        "segment_name": gt.segment_name,
        "source_volume_name": gt.source_volume_name,
        "source_modality": gt.source_modality,
        "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in EVIDENCE_VIEWS},
        "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
        "voxel_count": gt.voxel_count,
        "segment_total_voxel_count": gt.segment_total_voxel_count,
        "component_count": gt.component_count,
        "notes": list(gt.notes),
    }


def _emit_segment_debug(case_log: logging.Logger, *, case_id: str, segment_check: Dict[str, Any]) -> None:
    if not isinstance(segment_check, dict):
        return
    for line in segment_check.get("debug_lines") or []:
        case_log.info("[segment-check][%s] %s", case_id, line)


def _update_segment_metrics(
    *,
    gt: Optional[SegmentGroundTruth],
    segment_check: Dict[str, Any],
    key_slice_scored: Dict[str, int],
    key_slice_correct: Dict[str, int],
    rsa_metrics: Dict[str, int],
    provenance_metrics: Dict[str, int],
    evidence_case_metrics: Dict[str, int],
) -> None:
    if gt is None or not gt.has_segment:
        return
    evidence_case_metrics["n_cases_with_segment_gt"] += 1
    if bool(segment_check.get("geometry_exact_correct", False)):
        evidence_case_metrics["n_cases_with_geometry_exact_correct"] += 1
    if bool(segment_check.get("evidence_exact_correct", False)):
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] += 1

    key_slice_payload = segment_check.get("key_slice") or {}
    for view in EVIDENCE_VIEWS:
        key_slice_scored[view] += 1
        if bool((key_slice_payload.get(view) or {}).get("correct", False)):
            key_slice_correct[view] += 1

    rsa_metrics["n_scored"] += 1
    if bool((segment_check.get("rsa") or {}).get("correct", False)):
        rsa_metrics["n_correct"] += 1

    provenance_metrics["n_scored"] += 1
    if bool((segment_check.get("provenance") or {}).get("ok", False)):
        provenance_metrics["n_correct"] += 1


@dataclass(frozen=True)
class NSCLCCase:
    case_id: str
    case_root: Path
    catalog: CaseSeriesCatalog
    gt_location: Optional[str]
    gt_t_stage: Optional[str]
    gt_n_stage: Optional[str]
    gt_histology: Optional[str]
    gt_grade: Optional[str]

    @property
    def gt_answer(self) -> Optional[str]:
        if not all([self.gt_location, self.gt_t_stage, self.gt_n_stage, self.gt_histology, self.gt_grade]):
            return None
        return f"1{self.gt_location}2{self.gt_t_stage}3{self.gt_n_stage}4{self.gt_histology}5{self.gt_grade}"


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    csv_path = Path(csv_path).expanduser().resolve()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_get(row: Dict[str, str], *keys: str) -> Optional[str]:
    for k in keys:
        if k in row:
            return row.get(k)
    return None


def _location_letter(row: Dict[str, str]) -> Optional[str]:
    mapping = [
        ("Tumor Location (choice=RUL)", "A"),
        ("Tumor Location (choice=RML)", "B"),
        ("Tumor Location (choice=RLL)", "C"),
        ("Tumor Location (choice=LUL)", "D"),
        ("Tumor Location (choice=LLL)", "E"),
        ("Tumor Location (choice=L Lingula)", "F"),
    ]
    for col, letter in mapping:
        if str(row.get(col) or "").strip().lower() == "checked":
            return letter
    return "G" if str(row.get("Tumor Location (choice=Unknown)") or "").strip().lower() == "checked" else None


def _t_stage_letter(value: Optional[str]) -> Optional[str]:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if "not collected" in v or v in {"tx", "na", "n/a", "unknown", "not available", "not applicable"}:
        return None
    if v in {"t1", "t1a", "t1b", "tis"}:
        return "A"
    if v in {"t2", "t2a", "t2b"}:
        return "B"
    if v == "t3":
        return "C"
    if v == "t4":
        return "D"
    if "no tumor" in v:
        return "F"
    return None


def _n_stage_letter(value: Optional[str]) -> Optional[str]:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if "not collected" in v or v in {"nx", "na", "n/a", "unknown", "not available", "not applicable"}:
        return None
    if v == "n0":
        return "A"
    if v == "n1":
        return "B"
    if v == "n2":
        return "C"
    if "no tumor" in v:
        return "E"
    return None


def _histology_letter(value: Optional[str]) -> Optional[str]:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if "not collected" in v or v in {"na", "n/a", "unknown", "not available", "not applicable"}:
        return None
    if "adeno" in v:
        return "A"
    if "squamous" in v:
        return "B"
    if "small cell" in v:
        return "C"
    if "nsclc nos" in v or "not otherwise specified" in v:
        return "D"
    if "no tumor" in v or "normal" in v:
        return "E"
    if "other" in v or "indeterminate" in v:
        return "D"
    return None


def _grade_letter(value: Optional[str]) -> Optional[str]:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if "not collected" in v or v in {"na", "n/a", "unknown", "not available", "not applicable"}:
        return None
    if "no tumor" in v or "normal" in v:
        return "F"
    if "type ii" in v or "moderately to poorly" in v:
        return "E"
    if "type i" in v or "well to moderately" in v:
        return "D"
    if v.startswith("g1") or v == "well differentiated":
        return "A"
    if v.startswith("g2") or v == "moderately differentiated":
        return "B"
    if v.startswith("g3") or v == "poorly differentiated":
        return "C"
    return None


def iter_cases(
    *,
    data_root: Path,
    metadata_csv: Path,
    labels_csv: Path,
    limit: Optional[int] = None,
) -> Iterable[NSCLCCase]:
    data_root = Path(data_root).expanduser().resolve()
    labels = {str(r.get("Case ID") or "").strip(): r for r in _read_csv_rows(labels_csv)}
    metadata_rows = _read_csv_rows(metadata_csv)
    metadata_subject_ids = {str(r.get("Subject ID") or "").strip() for r in metadata_rows if str(r.get("Subject ID") or "").strip()}
    subject_ids = sorted([sid for sid in labels.keys() if sid and sid in metadata_subject_ids])
    if limit and limit > 0:
        subject_ids = subject_ids[: int(limit)]

    for sid in subject_ids:
        catalog = build_nsclc_case_catalog(data_root=data_root, metadata_csv=metadata_csv, subject_id=sid)
        row = labels.get(sid, {})
        case_root = catalog.subject_root()
        yield NSCLCCase(
            case_id=sid,
            case_root=case_root,
            catalog=catalog,
            gt_location=_location_letter(row),
            gt_t_stage=_t_stage_letter(_row_get(row, "Pathological T stage")),
            gt_n_stage=_n_stage_letter(_row_get(row, "Pathological N stage")),
            gt_histology=_histology_letter(_row_get(row, "Histology ", "Histology")),
            gt_grade=_grade_letter(_row_get(row, "Histopathological Grade")),
        )


def _format_segment_requirement(segment_evidence: Mapping[str, Any]) -> str:
    source_volume_name = str(segment_evidence.get("source_volume_name") or "").strip()
    source_modality = str(segment_evidence.get("source_modality") or "").strip()
    segment_name = str(segment_evidence.get("segment_name") or "").strip()
    modality_phrase = source_modality if source_modality else "the hidden source series"
    if source_volume_name:
        source_phrase = f'the loaded source series "{source_volume_name}"'
        if source_modality:
            source_phrase += f" ({source_modality})"
    else:
        source_phrase = f"the hidden source series ({modality_phrase})" if source_modality else "the hidden source series"

    segment_phrase = f" for segment \"{segment_name}\"" if segment_name else ""
    return (
        "Additional evidence output is required for this case.\n"
        f"The benchmark has a hidden lesion segmentation{segment_phrase}. Report the localization evidence on {source_phrase} only. "
        "Do not reuse coordinates from any other loaded series even if the anatomy looks aligned. If there are multiple disconnected visible tumor foci, localize the largest tumor component.\n"
        "Before producing the evidence block, explicitly switch/select the loaded source series that matches this hidden source volume and keep that source series active while localizing the tumor.\n"
        "After your brief rationale, output the following lines exactly once before the final compact answer string:\n"
        "EVIDENCE:\n"
        "KEY_SLICE_AXIAL_MM: <float>\n"
        "KEY_SLICE_SAGITTAL_MM: <float>\n"
        "KEY_SLICE_CORONAL_MM: <float>\n"
        "RSA: [<R>, <A>, <S>]\n"
        "Definitions:\n"
        "- Each KEY_SLICE_*_MM value must use canonical RAS-axis mm from the image corner overlay on the active hidden-source series: axial=S, sagittal=R, coronal=A. Do not use the native viewer-header sign convention if it differs from the corner overlay.\n"
        "- axial/sagittal/coronal refer to the three standard slice viewers, not slice indices.\n"
        "- RSA must be a single point strictly inside the visible primary tumor on that same hidden-source series.\n"
        "- If you use image_coords_to_ras, pass exactly one of bbox_1000 or point_1000. Never send both in the same call. If you want the center of a box, send only bbox_1000 and use center_ras from the tool output.\n"
        "- Keep the active source series on the correct volume when producing the evidence. If needed, explicitly call select_volume(volume_name=...) on the source series before the evidence block.\n"
    )

def build_task_prompt(
    case: NSCLCCase,
    *,
    toolpack_selection: ToolPackSelection,
    segment_evidence: Optional[Mapping[str, Any]] = None,
) -> str:
    case_series_intro = ""
    case_series_requirement = "- If CT and PET are already loaded and alignment is acceptable, you may use the core fusion tool directly.\n"
    if toolpack_selection.provides_pack("case_series"):
        case_series_intro = (
            " The prompt also lists additional curated CT/PET alias groups as reference groupings; if you need to switch to one of them explicitly, first activate the case_series tool pack and then call load_case_series(alias)."
            if toolpack_selection.requires_manual_activation("case_series")
            else " The prompt also lists additional curated CT/PET alias groups as reference groupings; if you need to switch to one of them explicitly, call load_case_series(alias)."
        )
        case_series_requirement = (
            "- If you need aligned PET/CT, prefer the curated alias pet_ct_core after activating the case_series tool pack.\n"
            if toolpack_selection.requires_manual_activation("case_series")
            else "- If you need aligned PET/CT, prefer the curated alias pet_ct_core via load_case_series(alias).\n"
        )

    evidence_block = ""
    final_answer_requirement = "- Explain your reasoning briefly in English, then end with the final compact answer string only once.\n"
    if segment_evidence and bool(segment_evidence.get("has_segment")):
        evidence_block = _format_segment_requirement(segment_evidence) + "\n"
        final_answer_requirement = (
            "- Explain your reasoning briefly in English.\n"
            "- If the evidence block is required above, include it exactly once before the final compact answer string.\n"
            "- End with the final compact answer string only once.\n"
        )

    return (
        f"Case: {case.case_id}\n\n"
        + "This is a multimodal thoracic oncology case. All resolved imaging series across all available timepoints are preloaded."
        + case_series_intro
        + "\n\n"
        + "Answer the following five multiple-choice questions based on the imaging only.\n"
        + "Return your final answer in one single compact string, with NO spaces, exactly like: 1C2D3B4A5B\n\n"
        + "Q1. Tumor location\n"
        + "1A Right upper lobe\n"
        + "1B Right middle lobe\n"
        + "1C Right lower lobe\n"
        + "1D Left upper lobe\n"
        + "1E Left lower lobe\n"
        + "1F Lingula\n"
        + "1G No tumor\n\n"
        + "Q2. Pathological T stage\n"
        + "2A T1\n"
        + "2B T2\n"
        + "2C T3\n"
        + "2D T4\n"
        + "2E Not available / not collected\n"
        + "2F No tumor\n\n"
        + "Q3. Pathological N stage\n"
        + "3A N0\n"
        + "3B N1\n"
        + "3C N2\n"
        + "3D Not available / not collected\n"
        + "3E No tumor\n\n"
        + "Q4. Histology\n"
        + "4A Adenocarcinoma\n"
        + "4B Squamous cell carcinoma\n"
        + "4C Small cell carcinoma\n"
        + "4D Other / indeterminate NSCLC\n"
        + "4E No tumor\n\n"
        + "Q5. Histopathological Grade\n"
        + "5A G1 Well differentiated\n"
        + "5B G2 Moderately differentiated\n"
        + "5C G3 Poorly differentiated\n"
        + "5D Other, Type I: Well to moderately differentiated\n"
        + "5E Other, Type II: Moderately to poorly differentiated\n"
        + "5F No tumor\n\n"
        + evidence_block
        + "Requirements:\n"
        + "- Start from CT morphology.\n"
        + "- Use PET as adjunctive evidence for metabolic activity and nodal survey.\n"
        + case_series_requirement
        + final_answer_requirement
    )
