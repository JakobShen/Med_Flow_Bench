from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RANO_LABEL_TO_CHOICE: Dict[str, str] = {
    "PD": "A",
    "SD": "B",
    "PR": "C",
    "CR": "D",
}
CHOICE_TO_RANO_LABEL: Dict[str, str] = {v: k for k, v in RANO_LABEL_TO_CHOICE.items()}
ANSWER_CHOICES = tuple(CHOICE_TO_RANO_LABEL.keys())
VALID_RANO_LABELS: Tuple[str, ...] = tuple(RANO_LABEL_TO_CHOICE.keys())
NON_BENCHMARK_LABELS: Tuple[str, ...] = ("Pre-Op", "Post-Op", "Post-Op/PD")
ATLAS_MODALITIES: Tuple[str, ...] = ("CT1", "T1", "T2", "FLAIR")
LUMIERE_IMAGE_SPACE_REGISTERED = "registered"
LUMIERE_IMAGE_SPACE_UNREGISTERED = "unregistered"
LUMIERE_IMAGE_SPACES: Tuple[str, ...] = (
    LUMIERE_IMAGE_SPACE_REGISTERED,
    LUMIERE_IMAGE_SPACE_UNREGISTERED,
)
_ATLAS_RELATIVE_PATHS: Dict[str, Tuple[str, ...]] = {
    "CT1": ("DeepBraTumIA-segmentation", "atlas", "skull_strip", "ct1_skull_strip.nii.gz"),
    "T1": ("DeepBraTumIA-segmentation", "atlas", "skull_strip", "t1_skull_strip.nii.gz"),
    "T2": ("DeepBraTumIA-segmentation", "atlas", "skull_strip", "t2_skull_strip.nii.gz"),
    "FLAIR": ("DeepBraTumIA-segmentation", "atlas", "skull_strip", "flair_skull_strip.nii.gz"),
}
EVIDENCE_FIELDS: Tuple[str, ...] = (
    "NON_MEASURABLE_LESION",
    "T2_PROGRESSION",
    "CURRENT_TARGET_LESION",
    "NEW_TARGET_LESION",
    "CURRENT_NON_TARGET_LESION",
    "NEW_NON_TARGET_LESION",
)
REPORT_TABLE_EVIDENCE_FIELDS: Tuple[str, ...] = tuple(field for field in EVIDENCE_FIELDS if field != "NON_MEASURABLE_LESION")
_EVIDENCE_LINE_RE = re.compile(
    r"^[\s>*\-\u2022]*([A-Z][A-Z0-9_]+)\s*:\s*(YES|NO)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_T2_PROGRESSION_YES_RE = re.compile(r"\b(?:discrete\s+)?t2(?:w)?\s*-?\s*progr(?:ession)?\.?", re.IGNORECASE)
_NEW_TARGET_YES_RE = re.compile(
    r"\b(?:new\s+(?:measurable\s+)?target\s+(?:l\.?|lesion)|new\s+tl)\b",
    re.IGNORECASE,
)
_CURRENT_TARGET_YES_PATTERNS: Tuple[re.Pattern[str], ...] = (
    _NEW_TARGET_YES_RE,
    re.compile(r"\bprogressing\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    re.compile(r"\bregressing\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    re.compile(r"\bnon-?measurable\s+(?:l\.?|lesion)\s+to\s+target\s+(?:l\.?|lesion)\b", re.IGNORECASE),
    re.compile(r"(?<!non-)(?<!non )\btarget\s+l\.?\s*:?\s*\d", re.IGNORECASE),
)
_CURRENT_NON_TARGET_NO_RE = re.compile(r"\bno\s+non-?target\s+(?:l\.?|lesion(?:s)?)\b", re.IGNORECASE)
_CURRENT_NON_TARGET_YES_RE = re.compile(
    r"\b(?:new\s+|regressing\s+|progressing\s+)?(?:non-?measurable\s+)?non-?target\s+(?:l\.?|lesion(?:s)?)\b",
    re.IGNORECASE,
)
_NEW_NON_TARGET_YES_RE = re.compile(
    r"\b(?:\d+\s+)?new\s+(?:non-?measurable\s+)?non-?target\s+(?:l\.?|lesion(?:s)?)\b",
    re.IGNORECASE,
)


def _strip(value: Optional[str]) -> str:
    return str(value or "").strip()


def normalize_rano_label(value: Optional[str]) -> Optional[str]:
    text = _strip(value)
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text)
    if normalized in set(VALID_RANO_LABELS) | set(NON_BENCHMARK_LABELS):
        return normalized
    return normalized


def parse_x_flag(value: Optional[str]) -> bool:
    return _strip(value).lower() == "x"


_TIMEPOINT_RE = re.compile(r"^week-(\d{3})(?:-(\d+))?$")
_PATIENT_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class LUMIEREStudyFiles:
    study_dir: Path
    atlas_mri_paths: Dict[str, Path]
    native_mri_paths: Dict[str, Path]
    deepbratumia_atlas_segmentation: Optional[Path]
    deepbratumia_native_segmentations: Dict[str, Path]
    hd_glio_registered_segmentation: Optional[Path]


@dataclass(frozen=True)
class LUMIEREVisit:
    patient_id: str
    timepoint: str
    label: Optional[str]
    rationale: Optional[str]
    less_than_3_months: bool
    non_measurable_lesions: bool
    relative_week: int
    intraweek_index: Optional[int]
    sort_key: Tuple[int, int, str]
    study_files: Optional[LUMIEREStudyFiles]
    completeness_flags: Dict[str, bool]
    duplicate_count: int = 1
    duplicate_note: Optional[str] = None
    ambiguous_duplicate: bool = False


@dataclass(frozen=True)
class LUMIEREQuestion:
    question_id: str
    patient_id: str
    old_visit: LUMIEREVisit
    current_visit: LUMIEREVisit
    gt_rano_label: str
    gt_choice: str
    question_index_for_patient: int


@dataclass(frozen=True)
class StagedQuestionInputs:
    stage_dir: Path
    old_files: Dict[str, Path]
    current_files: Dict[str, Path]
    all_files: List[Path]


@dataclass(frozen=True)
class QuestionBank:
    questions: List[LUMIEREQuestion]
    stats: Dict[str, Any]


@dataclass(frozen=True)
class _MergedRatingRow:
    patient_id: str
    timepoint: str
    label: Optional[str]
    rationale: Optional[str]
    less_than_3_months: bool
    non_measurable_lesions: bool
    duplicate_count: int
    duplicate_note: Optional[str]
    ambiguous_duplicate: bool


def patient_sort_key(patient_id: str) -> Tuple[int, str]:
    m = _PATIENT_RE.search(_strip(patient_id))
    if m:
        return int(m.group(1)), _strip(patient_id)
    return 10**9, _strip(patient_id)


def parse_timepoint_components(timepoint: str) -> Tuple[int, Optional[int]]:
    m = _TIMEPOINT_RE.match(_strip(timepoint))
    if not m:
        raise ValueError(f"Unsupported LUMIERE timepoint format: {timepoint!r}")
    week = int(m.group(1))
    intraweek = int(m.group(2)) if m.group(2) is not None else None
    return week, intraweek


def timepoint_sort_key(timepoint: str, label: Optional[str]) -> Tuple[int, int, str]:
    week, intraweek = parse_timepoint_components(timepoint)
    normalized_label = normalize_rano_label(label)
    if normalized_label == "Pre-Op":
        phase_rank = 0
    elif intraweek is not None:
        phase_rank = 10 + int(intraweek)
    elif normalized_label in {"Post-Op", "Post-Op/PD"}:
        phase_rank = 90
    else:
        phase_rank = 50
    return week, phase_rank, _strip(timepoint)


def read_datacompleteness(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, bool]]:
    csv_path = Path(csv_path).expanduser().resolve()
    result: Dict[Tuple[str, str], Dict[str, bool]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = _strip(row.get("Patient"))
            timepoint = _strip(row.get("Timepoint"))
            if not patient or not timepoint:
                continue
            result[(patient, timepoint)] = {str(k): parse_x_flag(v) for k, v in row.items() if k not in {"Patient", "Timepoint"}}
    return result


def _resolve_study_dir(data_root: Path, patient_id: str, timepoint: str) -> Optional[Path]:
    data_root = Path(data_root).expanduser().resolve()
    candidates = [
        data_root / "Imaging" / patient_id / timepoint,
        data_root / patient_id / timepoint,
    ]
    if data_root.name == "Imaging":
        candidates.insert(0, data_root / patient_id / timepoint)
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    return None


def discover_study_files(data_root: Path, patient_id: str, timepoint: str) -> Optional[LUMIEREStudyFiles]:
    study_dir = _resolve_study_dir(data_root, patient_id, timepoint)
    if study_dir is None:
        return None

    atlas_paths: Dict[str, Path] = {}
    for modality, parts in _ATLAS_RELATIVE_PATHS.items():
        atlas_paths[modality] = study_dir.joinpath(*parts)
    native_paths = {modality: study_dir / f"{modality}.nii.gz" for modality in ATLAS_MODALITIES}

    deep_native = {
        "CT1": study_dir / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "ct1_seg_mask.nii.gz",
        "T1": study_dir / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "t1_seg_mask.nii.gz",
        "T2": study_dir / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "t2_seg_mask.nii.gz",
        "FLAIR": study_dir / "DeepBraTumIA-segmentation" / "native" / "segmentation" / "flair_seg_mask.nii.gz",
    }
    deep_atlas_seg = study_dir / "DeepBraTumIA-segmentation" / "atlas" / "segmentation" / "seg_mask.nii.gz"
    hd_glio_registered_seg = study_dir / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz"

    return LUMIEREStudyFiles(
        study_dir=study_dir,
        atlas_mri_paths=atlas_paths,
        native_mri_paths=native_paths,
        deepbratumia_atlas_segmentation=deep_atlas_seg if deep_atlas_seg.exists() else None,
        deepbratumia_native_segmentations={k: v for k, v in deep_native.items() if v.exists()},
        hd_glio_registered_segmentation=hd_glio_registered_seg if hd_glio_registered_seg.exists() else None,
    )


def _has_required_deepbratumia_atlas_mri(
    *,
    completeness_flags: Dict[str, bool],
    study_files: Optional[LUMIEREStudyFiles],
) -> bool:
    if completeness_flags:
        required_flags = [
            completeness_flags.get("CT1", False),
            completeness_flags.get("T1", False),
            completeness_flags.get("T2", False),
            completeness_flags.get("FLAIR", False),
            completeness_flags.get("DeepBraTumIA", False),
        ]
        if not all(required_flags):
            return False
    if study_files is None:
        return False
    return all(study_files.atlas_mri_paths.get(modality, Path("/does/not/exist")).exists() for modality in ATLAS_MODALITIES)


def _has_required_native_mri(
    *,
    completeness_flags: Dict[str, bool],
    study_files: Optional[LUMIEREStudyFiles],
) -> bool:
    if completeness_flags:
        required_flags = [
            completeness_flags.get("CT1", False),
            completeness_flags.get("T1", False),
            completeness_flags.get("T2", False),
            completeness_flags.get("FLAIR", False),
        ]
        if not all(required_flags):
            return False
    if study_files is None:
        return False
    return all(study_files.native_mri_paths.get(modality, Path("/does/not/exist")).exists() for modality in ATLAS_MODALITIES)


def _has_required_mri(
    *,
    completeness_flags: Dict[str, bool],
    study_files: Optional[LUMIEREStudyFiles],
    image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED,
) -> bool:
    if image_space == LUMIERE_IMAGE_SPACE_UNREGISTERED:
        return _has_required_native_mri(completeness_flags=completeness_flags, study_files=study_files)
    return _has_required_deepbratumia_atlas_mri(completeness_flags=completeness_flags, study_files=study_files)


def _coalesce_unique_text(items: Sequence[str]) -> Optional[str]:
    unique: List[str] = []
    seen = set()
    for item in items:
        text = _strip(item)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            unique.append(text)
    if not unique:
        return None
    return " | ".join(unique)


def _rating_columns(fieldnames: Sequence[str]) -> Tuple[str, str]:
    label_col = None
    rationale_col = None
    for name in fieldnames:
        if name.startswith("Rating (according to RANO"):
            label_col = name
        elif name.startswith("Rating rationale"):
            rationale_col = name
    if not label_col or not rationale_col:
        raise KeyError("Could not locate LUMIERE rating/ rationale columns in the ratings CSV.")
    return label_col, rationale_col


def read_ratings(csv_path: Path) -> Tuple[List[_MergedRatingRow], Dict[str, Any]]:
    csv_path = Path(csv_path).expanduser().resolve()
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    n_rows_raw = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Ratings CSV has no header: {csv_path}")
        label_col, rationale_col = _rating_columns(reader.fieldnames)
        for row in reader:
            n_rows_raw += 1
            patient_id = _strip(row.get("Patient"))
            timepoint = _strip(row.get("Date"))
            if not patient_id or not timepoint:
                continue
            grouped.setdefault((patient_id, timepoint), []).append(
                {
                    "label": normalize_rano_label(row.get(label_col)),
                    "rationale": _strip(row.get(rationale_col)) or None,
                    "less_than_3_months": parse_x_flag(row.get("LessThan3Months")),
                    "non_measurable_lesions": parse_x_flag(row.get("NonMeasurableLesions")),
                }
            )

    merged_rows: List[_MergedRatingRow] = []
    n_ambiguous_duplicates = 0
    n_duplicate_groups = 0
    for (patient_id, timepoint), items in sorted(grouped.items(), key=lambda kv: (patient_sort_key(kv[0][0]), kv[0][1])):
        n_items = len(items)
        if n_items > 1:
            n_duplicate_groups += 1
        labels = {item["label"] for item in items if item.get("label")}
        ambiguous = len(labels) > 1
        if ambiguous:
            n_ambiguous_duplicates += 1
            merged_rows.append(
                _MergedRatingRow(
                    patient_id=patient_id,
                    timepoint=timepoint,
                    label=None,
                    rationale=_coalesce_unique_text([str(item.get("rationale") or "") for item in items]),
                    less_than_3_months=any(bool(item.get("less_than_3_months")) for item in items),
                    non_measurable_lesions=any(bool(item.get("non_measurable_lesions")) for item in items),
                    duplicate_count=n_items,
                    duplicate_note=f"Conflicting labels for the same patient/date: {sorted(labels)}",
                    ambiguous_duplicate=True,
                )
            )
            continue

        merged_rows.append(
            _MergedRatingRow(
                patient_id=patient_id,
                timepoint=timepoint,
                label=next(iter(labels)) if labels else None,
                rationale=_coalesce_unique_text([str(item.get("rationale") or "") for item in items]),
                less_than_3_months=any(bool(item.get("less_than_3_months")) for item in items),
                non_measurable_lesions=any(bool(item.get("non_measurable_lesions")) for item in items),
                duplicate_count=n_items,
                duplicate_note=(f"Merged {n_items} duplicate rows with identical label." if n_items > 1 else None),
                ambiguous_duplicate=False,
            )
        )

    stats = {
        "n_rows_raw": n_rows_raw,
        "n_rows_after_merge": len(merged_rows),
        "n_duplicate_groups": n_duplicate_groups,
        "n_ambiguous_duplicate_groups": n_ambiguous_duplicates,
    }
    return merged_rows, stats


def _build_visit(
    *,
    data_root: Path,
    row: _MergedRatingRow,
    completeness_index: Dict[Tuple[str, str], Dict[str, bool]],
) -> LUMIEREVisit:
    week, intraweek = parse_timepoint_components(row.timepoint)
    sort_key = timepoint_sort_key(row.timepoint, row.label)
    completeness_flags = completeness_index.get((row.patient_id, row.timepoint), {})
    study_files = discover_study_files(data_root, row.patient_id, row.timepoint)
    return LUMIEREVisit(
        patient_id=row.patient_id,
        timepoint=row.timepoint,
        label=row.label,
        rationale=row.rationale,
        less_than_3_months=bool(row.less_than_3_months),
        non_measurable_lesions=bool(row.non_measurable_lesions),
        relative_week=week,
        intraweek_index=intraweek,
        sort_key=sort_key,
        study_files=study_files,
        completeness_flags=dict(completeness_flags),
        duplicate_count=row.duplicate_count,
        duplicate_note=row.duplicate_note,
        ambiguous_duplicate=bool(row.ambiguous_duplicate),
    )


def _balance_questions_by_label(questions: Sequence[LUMIEREQuestion]) -> List[LUMIEREQuestion]:
    buckets: Dict[str, List[LUMIEREQuestion]] = {label: [] for label in VALID_RANO_LABELS}
    other_questions: List[LUMIEREQuestion] = []
    for question in questions:
        label = str(question.gt_rano_label or "").strip()
        if label in buckets:
            buckets[label].append(question)
        else:
            other_questions.append(question)

    balanced: List[LUMIEREQuestion] = []
    ordered_labels = list(VALID_RANO_LABELS)
    while any(buckets[label] for label in ordered_labels):
        for label in ordered_labels:
            if buckets[label]:
                balanced.append(buckets[label].pop(0))
    balanced.extend(other_questions)
    return balanced


def build_question_bank(
    *,
    data_root: Path,
    ratings_csv: Path,
    datacompleteness_csv: Optional[Path] = None,
    limit_patient: Optional[int] = None,
    limit_question: Optional[int] = None,
    balance_rano_labels: bool = False,
    image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED,
) -> QuestionBank:
    if limit_patient and limit_patient > 0 and limit_question and limit_question > 0:
        raise ValueError("Only one of limit_patient or limit_question may be set.")

    completeness_index = read_datacompleteness(datacompleteness_csv) if datacompleteness_csv else {}
    merged_rows, rating_stats = read_ratings(ratings_csv)

    visits_by_patient: Dict[str, List[LUMIEREVisit]] = {}
    for row in merged_rows:
        visit = _build_visit(data_root=data_root, row=row, completeness_index=completeness_index)
        visits_by_patient.setdefault(visit.patient_id, []).append(visit)

    questions: List[LUMIEREQuestion] = []
    per_label_counts_pre_limit = {label: 0 for label in VALID_RANO_LABELS}
    stats: Dict[str, Any] = {
        **rating_stats,
        "n_patients_total": len(visits_by_patient),
        "n_patients_with_questions_before_limit": 0,
        "n_questions_total_before_limit": 0,
        "n_skipped_current_non_benchmark_label": 0,
        "n_skipped_missing_required_old_or_current_imaging": 0,
        "n_skipped_ambiguous_duplicate": 0,
        "n_skipped_missing_previous_visit": 0,
        "n_skipped_missing_label": 0,
        "per_label_counts_before_limit": {label: 0 for label in VALID_RANO_LABELS},
        "per_label_counts_after_limit": {label: 0 for label in VALID_RANO_LABELS},
    }

    patients_with_questions: List[str] = []
    for patient_id in sorted(visits_by_patient.keys(), key=patient_sort_key):
        visits = sorted(visits_by_patient[patient_id], key=lambda item: item.sort_key)
        patient_questions: List[LUMIEREQuestion] = []
        for index in range(1, len(visits)):
            old_visit = visits[index - 1]
            current_visit = visits[index]

            if current_visit.label is None:
                stats["n_skipped_missing_label"] += 1
                continue
            if current_visit.label not in VALID_RANO_LABELS:
                stats["n_skipped_current_non_benchmark_label"] += 1
                continue
            if old_visit.ambiguous_duplicate or current_visit.ambiguous_duplicate:
                stats["n_skipped_ambiguous_duplicate"] += 1
                continue
            if not _has_required_mri(
                completeness_flags=old_visit.completeness_flags,
                study_files=old_visit.study_files,
                image_space=image_space,
            ) or not _has_required_mri(
                completeness_flags=current_visit.completeness_flags,
                study_files=current_visit.study_files,
                image_space=image_space,
            ):
                stats["n_skipped_missing_required_old_or_current_imaging"] += 1
                continue

            question = LUMIEREQuestion(
                question_id=f"{patient_id}__{old_visit.timepoint}__to__{current_visit.timepoint}",
                patient_id=patient_id,
                old_visit=old_visit,
                current_visit=current_visit,
                gt_rano_label=current_visit.label,
                gt_choice=RANO_LABEL_TO_CHOICE[current_visit.label],
                question_index_for_patient=len(patient_questions) + 1,
            )
            patient_questions.append(question)
            per_label_counts_pre_limit[current_visit.label] += 1

        if patient_questions:
            patients_with_questions.append(patient_id)
            questions.extend(patient_questions)

    stats["n_patients_with_questions_before_limit"] = len(patients_with_questions)
    stats["n_questions_total_before_limit"] = len(questions)
    stats["per_label_counts_before_limit"] = dict(per_label_counts_pre_limit)

    if limit_patient and limit_patient > 0:
        allowed_patients = {pid for pid in sorted(patients_with_questions, key=patient_sort_key)[: int(limit_patient)]}
        questions = [q for q in questions if q.patient_id in allowed_patients]
    if balance_rano_labels:
        questions = _balance_questions_by_label(questions)
    if limit_question and limit_question > 0:
        questions = questions[: int(limit_question)]

    stats["limit_patient"] = int(limit_patient or 0)
    stats["limit_question"] = int(limit_question or 0)
    stats["balance_rano_labels"] = bool(balance_rano_labels)
    stats["n_patients_after_limit"] = len({q.patient_id for q in questions})
    stats["n_questions_after_limit"] = len(questions)
    per_label_counts_after_limit = {label: 0 for label in VALID_RANO_LABELS}
    for question in questions:
        per_label_counts_after_limit[question.gt_rano_label] += 1
    stats["per_label_counts_after_limit"] = per_label_counts_after_limit
    return QuestionBank(questions=questions, stats=stats)


def _safe_timepoint_token(timepoint: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", _strip(timepoint))


def staged_volume_stem(*, side: str, timepoint: str, modality: str) -> str:
    return f"{side}__{_safe_timepoint_token(timepoint)}__{modality}"


def _link_or_copy(src: Path, dst: Path) -> None:
    src = Path(src).expanduser().resolve()
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def stage_question_inputs(
    question: LUMIEREQuestion,
    stage_dir: Path,
    *,
    image_space: str = LUMIERE_IMAGE_SPACE_REGISTERED,
) -> StagedQuestionInputs:
    if question.old_visit.study_files is None or question.current_visit.study_files is None:
        raise ValueError(f"Question {question.question_id} does not have resolved study files.")

    stage_dir = Path(stage_dir).expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)

    old_files: Dict[str, Path] = {}
    current_files: Dict[str, Path] = {}
    all_files: List[Path] = []

    for side, visit, dest_map in (
        ("OLD", question.old_visit, old_files),
        ("CURRENT", question.current_visit, current_files),
    ):
        if visit.study_files is None:
            raise ValueError(f"Visit {visit.patient_id} {visit.timepoint} has no study files.")
        for modality in ATLAS_MODALITIES:
            src_map = (
                visit.study_files.native_mri_paths
                if image_space == LUMIERE_IMAGE_SPACE_UNREGISTERED
                else visit.study_files.atlas_mri_paths
            )
            src = src_map[modality]
            if not src.exists():
                raise FileNotFoundError(f"Missing required {image_space} MRI: {src}")
            dst = stage_dir / f"{staged_volume_stem(side=side, timepoint=visit.timepoint, modality=modality)}.nii.gz"
            _link_or_copy(src, dst)
            dest_map[modality] = dst
            all_files.append(dst)

    return StagedQuestionInputs(
        stage_dir=stage_dir,
        old_files=old_files,
        current_files=current_files,
        all_files=all_files,
    )


def format_loaded_volume_guide(question: LUMIEREQuestion) -> str:
    old_lines = [f"  - {staged_volume_stem(side='OLD', timepoint=question.old_visit.timepoint, modality=modality)}" for modality in ATLAS_MODALITIES]
    current_lines = [f"  - {staged_volume_stem(side='CURRENT', timepoint=question.current_visit.timepoint, modality=modality)}" for modality in ATLAS_MODALITIES]
    return (
        f"OLD study ({question.old_visit.timepoint}) loaded volumes:\n"
        + "\n".join(old_lines)
        + "\n\n"
        + f"CURRENT study ({question.current_visit.timepoint}) loaded volumes:\n"
        + "\n".join(current_lines)
    )


def _per_class_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {label: 0 for label in VALID_RANO_LABELS}
    correct = {label: 0 for label in VALID_RANO_LABELS}
    for rec in results:
        gt_label = str(rec.get("gt_rano_label") or "").strip()
        if gt_label not in counts:
            continue
        counts[gt_label] += 1
        if bool(rec.get("correct", False)):
            correct[gt_label] += 1
    accuracy = {
        label: ((correct[label] / counts[label]) if counts[label] else 0.0)
        for label in VALID_RANO_LABELS
    }
    observed = [accuracy[label] for label in VALID_RANO_LABELS if counts[label] > 0]
    balanced_accuracy = (sum(observed) / len(observed)) if observed else 0.0
    return {
        "counts": counts,
        "correct": correct,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def _normalize_rationale_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_evidence_predictions(text: str) -> Dict[str, Optional[bool]]:
    found: Dict[str, bool] = {}
    if text:
        for match in _EVIDENCE_LINE_RE.finditer(text):
            raw_name = str(match.group(1) or "").strip().upper()
            if raw_name.startswith("EVIDENCE_"):
                raw_name = raw_name[len("EVIDENCE_") :]
            if raw_name not in EVIDENCE_FIELDS:
                continue
            found[raw_name] = str(match.group(2) or "").strip().upper() == "YES"
    return {field: found.get(field) for field in EVIDENCE_FIELDS}


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _infer_evidence_ground_truth(question: LUMIEREQuestion) -> Tuple[Dict[str, Optional[bool]], Dict[str, Optional[str]]]:
    rationale = _normalize_rationale_text(question.current_visit.rationale)
    gt: Dict[str, Optional[bool]] = {field: None for field in EVIDENCE_FIELDS}
    source: Dict[str, Optional[str]] = {field: None for field in EVIDENCE_FIELDS}

    gt["NON_MEASURABLE_LESION"] = bool(question.current_visit.non_measurable_lesions)
    source["NON_MEASURABLE_LESION"] = "csv.NonMeasurableLesions"

    if not rationale:
        return gt, source

    if _T2_PROGRESSION_YES_RE.search(rationale):
        gt["T2_PROGRESSION"] = True
        source["T2_PROGRESSION"] = "rationale:T2-Progr"
    if _matches_any(rationale, _CURRENT_TARGET_YES_PATTERNS):
        gt["CURRENT_TARGET_LESION"] = True
        source["CURRENT_TARGET_LESION"] = "rationale:target lesion mention"
    if _NEW_TARGET_YES_RE.search(rationale):
        gt["NEW_TARGET_LESION"] = True
        source["NEW_TARGET_LESION"] = "rationale:new target lesion"
    if _CURRENT_NON_TARGET_NO_RE.search(rationale):
        gt["CURRENT_NON_TARGET_LESION"] = False
        source["CURRENT_NON_TARGET_LESION"] = "rationale:no non-target lesion"
    elif _CURRENT_NON_TARGET_YES_RE.search(rationale):
        gt["CURRENT_NON_TARGET_LESION"] = True
        source["CURRENT_NON_TARGET_LESION"] = "rationale:non-target lesion mention"
    if _NEW_NON_TARGET_YES_RE.search(rationale):
        gt["NEW_NON_TARGET_LESION"] = True
        source["NEW_NON_TARGET_LESION"] = "rationale:new non-target lesion"
    return gt, source


def _count_reporttable_evidence_fields(question: LUMIEREQuestion) -> int:
    gt, _ = _infer_evidence_ground_truth(question)
    return sum(1 for field in REPORT_TABLE_EVIDENCE_FIELDS if gt.get(field) is not None)


def _format_bool_label(value: Optional[bool], *, unknown_label: str = "UNKNOWN") -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return unknown_label


def _score_evidence_fields(
    *,
    gt: Dict[str, Optional[bool]],
    pred: Dict[str, Optional[bool]],
    source: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    details: Dict[str, Dict[str, Any]] = {}
    n_scored = 0
    n_correct = 0
    for field in EVIDENCE_FIELDS:
        gt_val = gt.get(field)
        pred_val = pred.get(field)
        scored = gt_val is not None
        correct = bool(scored and pred_val is not None and pred_val == gt_val)
        if scored:
            n_scored += 1
            n_correct += int(correct)
        details[field] = {
            "gt": gt_val,
            "gt_label": _format_bool_label(gt_val),
            "pred": pred_val,
            "pred_label": _format_bool_label(pred_val, unknown_label="MISSING"),
            "source": source.get(field),
            "scored": scored,
            "correct": correct,
        }
    return details, n_scored, n_correct


def _aggregate_evidence_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_field: Dict[str, Dict[str, Any]] = {}
    total_scored = 0
    total_correct = 0
    n_questions_with_any_evidence_scored = 0
    n_questions_all_scored_evidence_correct = 0
    n_questions_correct_and_all_scored_evidence_correct = 0

    for rec in results:
        evidence = rec.get("evidence") or {}
        case_has_scored_evidence = any(bool((evidence.get(field) or {}).get("scored", False)) for field in EVIDENCE_FIELDS)
        if case_has_scored_evidence:
            n_questions_with_any_evidence_scored += 1
            if int(rec.get("n_evidence_fields_scored") or 0) == int(rec.get("n_evidence_fields_correct") or 0):
                n_questions_all_scored_evidence_correct += 1
                if bool(rec.get("correct")):
                    n_questions_correct_and_all_scored_evidence_correct += 1

    for field in EVIDENCE_FIELDS:
        n_scored = 0
        n_correct = 0
        n_gt_yes = 0
        n_gt_no = 0
        n_pred_missing = 0
        for rec in results:
            item = (rec.get("evidence") or {}).get(field) or {}
            if not bool(item.get("scored", False)):
                continue
            n_scored += 1
            gt_val = item.get("gt")
            pred_val = item.get("pred")
            if gt_val is True:
                n_gt_yes += 1
            elif gt_val is False:
                n_gt_no += 1
            if pred_val is None:
                n_pred_missing += 1
            if bool(item.get("correct", False)):
                n_correct += 1
        accuracy = (n_correct / n_scored) if n_scored else 0.0
        per_field[field] = {
            "n_scored": n_scored,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "n_gt_yes": n_gt_yes,
            "n_gt_no": n_gt_no,
            "n_pred_missing": n_pred_missing,
        }
        total_scored += n_scored
        total_correct += n_correct

    micro_accuracy = (total_correct / total_scored) if total_scored else 0.0
    all_scored_evidence_correct_accuracy = (
        n_questions_all_scored_evidence_correct / n_questions_with_any_evidence_scored
        if n_questions_with_any_evidence_scored
        else 0.0
    )
    correct_and_all_scored_evidence_correct_accuracy = (
        n_questions_correct_and_all_scored_evidence_correct / n_questions_with_any_evidence_scored
        if n_questions_with_any_evidence_scored
        else 0.0
    )
    return {
        "fields": per_field,
        "total_scored": total_scored,
        "total_correct": total_correct,
        "micro_accuracy": micro_accuracy,
        "n_questions_with_any_evidence_scored": n_questions_with_any_evidence_scored,
        "n_questions_all_scored_evidence_correct": n_questions_all_scored_evidence_correct,
        "all_scored_evidence_correct_accuracy": all_scored_evidence_correct_accuracy,
        "n_questions_correct_and_all_scored_evidence_correct": n_questions_correct_and_all_scored_evidence_correct,
        "correct_and_all_scored_evidence_correct_accuracy": correct_and_all_scored_evidence_correct_accuracy,
    }
