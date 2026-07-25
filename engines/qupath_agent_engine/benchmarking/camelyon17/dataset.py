from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable


CAMELYON17_SLIDE_LABELS = ("negative", "itc", "micro", "macro")
CAMELYON17_PATIENT_STAGES = ("pn0", "pn0(i+)", "pn1mi", "pn1", "pn2")
_CAMELYON_IMAGE_RE = re.compile(r"^(patient_\d{3}_node_([0-4]))\.tif$", re.IGNORECASE)
_CAMELYON_MASK_RE = re.compile(r"^(patient_\d{3}_node_[0-4])_mask\.tif$", re.IGNORECASE)
_CAMELYON_XML_RE = re.compile(r"^(patient_\d{3}_node_[0-4])\.xml$", re.IGNORECASE)
_CAMELYON_PATIENT_RE = re.compile(r"^(patient_\d{3})\.zip$", re.IGNORECASE)


@dataclass(slots=True)
class Camelyon17PatientRecord:
    patient_id: str
    patient_archive_name: str
    patient_stage: str
    center: str | None
    slide_case_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "patient_archive_name": self.patient_archive_name,
            "patient_stage": self.patient_stage,
            "center": self.center,
            "slide_case_ids": list(self.slide_case_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Camelyon17SlideRecord:
    csv_row_index: int
    case_id: str
    slide_filename: str
    patient_id: str
    node_index: int
    center: str | None
    patient_stage: str | None
    gt_slide_label: str
    image_path: Path | None
    mask_path: Path | None
    xml_path: Path | None
    duplicate_image_paths: list[str] = field(default_factory=list)
    duplicate_mask_paths: list[str] = field(default_factory=list)
    duplicate_xml_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_runnable(self) -> bool:
        return self.image_path is not None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None

    @property
    def has_xml(self) -> bool:
        return self.xml_path is not None

    @property
    def has_evidence_source(self) -> bool:
        return self.has_mask or self.has_xml

    @property
    def gt_tumor_present(self) -> bool:
        return self.gt_slide_label != "negative"

    def summary_dict(self) -> dict[str, Any]:
        return {
            "csv_row_index": self.csv_row_index,
            "case_id": self.case_id,
            "slide_filename": self.slide_filename,
            "patient_id": self.patient_id,
            "node_index": self.node_index,
            "center": self.center,
            "patient_stage": self.patient_stage,
            "gt_slide_label": self.gt_slide_label,
            "gt_tumor_present": self.gt_tumor_present,
            "image_path": str(self.image_path) if self.image_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "xml_path": str(self.xml_path) if self.xml_path else None,
            "duplicate_image_paths": list(self.duplicate_image_paths),
            "duplicate_mask_paths": list(self.duplicate_mask_paths),
            "duplicate_xml_paths": list(self.duplicate_xml_paths),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Camelyon17DatasetManifest:
    data_root: Path
    stages_csv_path: Path
    slides: list[Camelyon17SlideRecord]
    patients: list[Camelyon17PatientRecord]
    indexed_image_count: int
    indexed_mask_count: int
    indexed_xml_count: int
    duplicate_filename_counts: dict[str, int] = field(default_factory=dict)

    def runnable_slides(
        self,
        *,
        allowed_centers: Iterable[str] | None = None,
        allowed_slide_labels: Iterable[str] | None = None,
        allowed_patient_ids: Iterable[str] | None = None,
    ) -> list[Camelyon17SlideRecord]:
        center_filter = _normalize_generic_filter(allowed_centers)
        label_filter = _normalize_generic_filter(allowed_slide_labels)
        patient_filter = _normalize_generic_filter(allowed_patient_ids)
        slides = [slide for slide in self.slides if slide.is_runnable]
        if center_filter is not None:
            slides = [slide for slide in slides if str(slide.center or "").strip().lower() in center_filter]
        if label_filter is not None:
            slides = [slide for slide in slides if slide.gt_slide_label in label_filter]
        if patient_filter is not None:
            slides = [slide for slide in slides if slide.patient_id.lower() in patient_filter]
        return slides

    def skipped_missing_image_slides(
        self,
        *,
        allowed_centers: Iterable[str] | None = None,
        allowed_slide_labels: Iterable[str] | None = None,
        allowed_patient_ids: Iterable[str] | None = None,
    ) -> list[Camelyon17SlideRecord]:
        center_filter = _normalize_generic_filter(allowed_centers)
        label_filter = _normalize_generic_filter(allowed_slide_labels)
        patient_filter = _normalize_generic_filter(allowed_patient_ids)
        slides = [slide for slide in self.slides if not slide.is_runnable]
        if center_filter is not None:
            slides = [slide for slide in slides if str(slide.center or "").strip().lower() in center_filter]
        if label_filter is not None:
            slides = [slide for slide in slides if slide.gt_slide_label in label_filter]
        if patient_filter is not None:
            slides = [slide for slide in slides if slide.patient_id.lower() in patient_filter]
        return slides

    def patient_map(self) -> dict[str, Camelyon17PatientRecord]:
        return {patient.patient_id: patient for patient in self.patients}

    def summary_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "stages_csv_path": str(self.stages_csv_path),
            "n_patient_rows": len(self.patients),
            "n_slide_rows": len(self.slides),
            "n_runnable_slides": sum(1 for slide in self.slides if slide.is_runnable),
            "n_gt_positive_slides": sum(1 for slide in self.slides if slide.gt_tumor_present),
            "n_slides_with_mask": sum(1 for slide in self.slides if slide.has_mask),
            "n_slides_with_xml": sum(1 for slide in self.slides if slide.has_xml),
            "n_slides_with_evidence_source": sum(1 for slide in self.slides if slide.has_evidence_source),
            "indexed_image_count": self.indexed_image_count,
            "indexed_mask_count": self.indexed_mask_count,
            "indexed_xml_count": self.indexed_xml_count,
            "duplicate_filename_counts": dict(self.duplicate_filename_counts),
        }


def load_camelyon17_manifest(data_root: str | Path) -> Camelyon17DatasetManifest:
    root = Path(data_root).expanduser().resolve()
    stages_csv_path = _resolve_camelyon17_stages_csv(root)

    indexed = _index_camelyon17_files(root)
    duplicate_filename_counts = {
        name: len(paths)
        for name, paths in {**indexed["images"], **indexed["masks"], **indexed["xmls"]}.items()
        if len(paths) > 1
    }

    patient_records: list[Camelyon17PatientRecord] = []
    patient_stage_map: dict[str, Camelyon17PatientRecord] = {}
    slide_records: list[Camelyon17SlideRecord] = []

    with stages_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"patient", "stage"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"stages.csv is missing required columns: {missing}")

        for csv_row_index, row in enumerate(reader, start=2):
            patient_field = str(row.get("patient") or "").strip()
            stage_field = row.get("stage")
            center_value = _string_or_none(row.get("center"))
            if not patient_field:
                continue

            patient_match = _CAMELYON_PATIENT_RE.match(patient_field)
            if patient_match:
                patient_id = patient_match.group(1).lower()
                patient_stage = normalize_camelyon17_patient_stage(stage_field)
                if patient_stage is None:
                    raise ValueError(
                        f"Unsupported CAMELYON17 patient stage at CSV row {csv_row_index}: {stage_field!r}"
                    )
                record = Camelyon17PatientRecord(
                    patient_id=patient_id,
                    patient_archive_name=patient_field,
                    patient_stage=patient_stage,
                    center=center_value,
                    metadata={"csv_row_index": csv_row_index},
                )
                patient_stage_map[patient_id] = record
                patient_records.append(record)
                continue

            slide_match = _CAMELYON_IMAGE_RE.match(patient_field)
            if not slide_match:
                continue

            case_id = slide_match.group(1).lower()
            patient_id = case_id.rsplit("_node_", 1)[0]
            node_index = int(slide_match.group(2))
            gt_slide_label = normalize_camelyon17_slide_label(stage_field)
            if gt_slide_label is None:
                raise ValueError(
                    f"Unsupported CAMELYON17 slide label at CSV row {csv_row_index}: {stage_field!r}"
                )
            image_name = f"{case_id}.tif"
            mask_name = f"{case_id}_mask.tif"
            xml_name = f"{case_id}.xml"
            image_matches = indexed["images"].get(image_name.lower(), [])
            mask_matches = indexed["masks"].get(mask_name.lower(), [])
            xml_matches = indexed["xmls"].get(xml_name.lower(), [])
            patient_record = patient_stage_map.get(patient_id)
            if patient_record is not None:
                patient_record.slide_case_ids.append(case_id)
            slide_records.append(
                Camelyon17SlideRecord(
                    csv_row_index=csv_row_index,
                    case_id=case_id,
                    slide_filename=patient_field,
                    patient_id=patient_id,
                    node_index=node_index,
                    center=center_value,
                    patient_stage=patient_record.patient_stage if patient_record else None,
                    gt_slide_label=gt_slide_label,
                    image_path=image_matches[0] if image_matches else None,
                    mask_path=mask_matches[0] if mask_matches else None,
                    xml_path=xml_matches[0] if xml_matches else None,
                    duplicate_image_paths=[str(path) for path in image_matches[1:]],
                    duplicate_mask_paths=[str(path) for path in mask_matches[1:]],
                    duplicate_xml_paths=[str(path) for path in xml_matches[1:]],
                    metadata={
                        "csv_row_index": csv_row_index,
                    },
                )
            )

    return Camelyon17DatasetManifest(
        data_root=root,
        stages_csv_path=stages_csv_path,
        slides=slide_records,
        patients=patient_records,
        indexed_image_count=sum(len(paths) for paths in indexed["images"].values()),
        indexed_mask_count=sum(len(paths) for paths in indexed["masks"].values()),
        indexed_xml_count=sum(len(paths) for paths in indexed["xmls"].values()),
        duplicate_filename_counts=duplicate_filename_counts,
    )


def _resolve_camelyon17_stages_csv(root: Path) -> Path:
    search_roots = [root]
    current = root
    for _ in range(3):
        parent = current.parent
        if parent == current:
            break
        search_roots.append(parent)
        current = parent

    for candidate_root in search_roots:
        candidate = candidate_root / "stages.csv"
        if candidate.exists():
            return candidate

    for candidate_root in search_roots:
        candidate = _find_named_file_nearby(candidate_root, "stages.csv", max_depth=5)
        if candidate is not None:
            return candidate

    searched = [str(candidate_root / "stages.csv") for candidate_root in search_roots]
    raise FileNotFoundError(
        "stages.csv was not found. Searched: " + ", ".join(searched)
    )


def _find_named_file_nearby(root: Path, filename: str, *, max_depth: int) -> Path | None:
    if max_depth < 1 or not root.exists() or not root.is_dir():
        return None
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda path: path.name.lower())
        except Exception:
            continue
        for child in children:
            if child.is_file() and child.name.lower() == filename.lower():
                return child
        for child in children:
            if child.is_dir():
                queue.append((child, depth + 1))
    return None


def normalize_camelyon17_slide_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "negative": "negative",
        "neg": "negative",
        "itc": "itc",
        "micro": "micro",
        "macro": "macro",
    }
    return aliases.get(text)


def normalize_camelyon17_patient_stage(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "pn0": "pn0",
        "pn0(i+)": "pn0(i+)",
        "pn1mi": "pn1mi",
        "pn1": "pn1",
        "pn2": "pn2",
    }
    return aliases.get(text)


def _normalize_generic_filter(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized or None


def _index_camelyon17_files(root: Path) -> dict[str, dict[str, list[Path]]]:
    images: dict[str, list[Path]] = {}
    masks: dict[str, list[Path]] = {}
    xmls: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("._"):
            continue
        lower_name = name.lower()
        if _CAMELYON_IMAGE_RE.match(name):
            images.setdefault(lower_name, []).append(path.resolve())
            continue
        if _CAMELYON_MASK_RE.match(name):
            masks.setdefault(lower_name, []).append(path.resolve())
            continue
        if _CAMELYON_XML_RE.match(name):
            xmls.setdefault(lower_name, []).append(path.resolve())
            continue
    return {"images": images, "masks": masks, "xmls": xmls}


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
