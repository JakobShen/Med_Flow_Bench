from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


IMAGE_MODALITIES = {"CT", "PT", "PET", "MR", "NM", "US", "CR", "DX", "XA", "MG"}


@dataclass(frozen=True)
class SeriesRecord:
    subject_id: str
    study_uid: str
    study_date: str
    study_description: str
    series_uid: str
    series_description: str
    modality: str
    n_images: int
    abs_path: Path
    raw_file_location: str
    role: str
    score: float

    @property
    def display_name(self) -> str:
        bits: List[str] = []
        if self.study_date:
            bits.append(self.study_date)
        if self.modality:
            bits.append(self.modality)
        if self.series_description:
            bits.append(self.series_description)
        elif self.abs_path.name:
            bits.append(self.abs_path.name)
        return " | ".join(bits) if bits else str(self.abs_path)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "study_uid": self.study_uid,
            "study_date": self.study_date,
            "study_description": self.study_description,
            "series_uid": self.series_uid,
            "series_description": self.series_description,
            "modality": self.modality,
            "n_images": self.n_images,
            "abs_path": str(self.abs_path),
            "role": self.role,
            "score": round(float(self.score), 3),
            "display_name": self.display_name,
        }


@dataclass
class CaseSeriesCatalog:
    subject_id: str
    data_root: Path
    series: List[SeriesRecord]
    alias_map: Dict[str, List[SeriesRecord]]
    default_aliases: List[str]
    notes: List[str]

    def to_summary(self) -> Dict[str, Any]:
        aliases: Dict[str, Any] = {}
        for alias, records in self.alias_map.items():
            aliases[alias] = [r.as_dict() for r in records]
        return {
            "subject_id": self.subject_id,
            "data_root": str(self.data_root),
            "default_aliases": list(self.default_aliases),
            "aliases": aliases,
            "notes": list(self.notes),
            "all_series": [r.as_dict() for r in self.series],
            "subject_root": str(self.subject_root()),
        }

    def alias_paths(self, alias: str) -> List[Path]:
        recs = self.alias_map.get(alias, [])
        return [r.abs_path for r in recs]

    def alias_display_names(self, alias: str) -> List[str]:
        return [r.display_name for r in self.alias_map.get(alias, [])]

    def alias_active_prefer(self, alias: str) -> List[str]:
        alias_l = alias.lower()
        if alias_l in {"diagnostic_ct", "ct_only", "core"}:
            return ["lung", "thorax", "chest", "ct", "b70", "b45", "1 mm"]
        if alias_l in {"pet_primary", "pet_nac", "pet_ac", "pet_only"}:
            return ["suvbw", "body weight", "standardized uptake value", "nac", "noac", "mac", "pet", "ramla", "wb"]
        if alias_l in {"pet_ct_core", "ctac"}:
            return ["ctac", "ct images", "ldct", "fusion", "pet", "suvbw", "wb"]
        if alias_l == "fusion_ct":
            return ["fusion", "ct"]
        return []

    def alias_window_preset(self, alias: str) -> Optional[str]:
        alias_l = alias.lower()
        if alias_l in {"diagnostic_ct", "ct_only", "core", "pet_ct_core", "ctac", "fusion_ct"}:
            return "ct_lung"
        if alias_l in {"pet_primary", "pet_nac", "pet_ac", "pet_only"}:
            return "pet_auto"
        return None

    def all_series_records(self, *, prefer_latest: bool = True) -> List[SeriesRecord]:
        unique: Dict[str, SeriesRecord] = {}
        for rec in self.series:
            unique[str(rec.abs_path)] = rec
        records = list(unique.values())
        if prefer_latest:
            return sorted(records, key=lambda r: (r.study_date, r.score, r.n_images, r.series_description), reverse=True)
        return sorted(records, key=lambda r: (r.study_date, r.modality, r.series_description, str(r.abs_path)))

    def all_series_paths(self, *, prefer_latest: bool = True) -> List[Path]:
        return [r.abs_path for r in self.all_series_records(prefer_latest=prefer_latest)]

    def all_series_display_names(self, *, prefer_latest: bool = True) -> List[str]:
        return [r.display_name for r in self.all_series_records(prefer_latest=prefer_latest)]

    def series_by_study_date(self, *, prefer_latest: bool = False) -> Dict[str, List[SeriesRecord]]:
        grouped: Dict[str, List[SeriesRecord]] = defaultdict(list)
        for rec in self.series:
            grouped[rec.study_date or "unknown"].append(rec)
        out: Dict[str, List[SeriesRecord]] = {}
        keys = sorted(grouped.keys(), reverse=prefer_latest)
        for key in keys:
            out[key] = sorted(grouped[key], key=lambda r: (r.modality, -r.score, r.series_description, str(r.abs_path)))
        return out

    def subject_root(self) -> Path:
        if not self.series:
            return (self.data_root / self.subject_id).resolve()
        try:
            common = Path(os.path.commonpath([str(r.abs_path) for r in self.series]))
            return common.resolve()
        except Exception:
            return self.series[0].abs_path.parent.resolve()


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    csv_path = Path(csv_path).expanduser().resolve()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_tcia_file_location(file_location: str, data_root: Path) -> Path:
    """Resolve TCIA metadata `File Location` against a local download root.

    Handles common cases:
    - metadata says `./NSCLC Radiogenomics/...`
    - local directory is `NSCLC_Radiogenomics/...`
    - caller passes either the manifest root or the dataset root itself
    - metadata points into alternate collection roots such as `Lung-PET-CT-Dx/...`
    """

    data_root = Path(data_root).expanduser().resolve()
    raw = str(file_location or "").strip()
    if not raw:
        raise ValueError("Empty file_location")

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p.resolve()

    rel = raw
    if rel.startswith("./"):
        rel = rel[2:]
    parts = list(Path(rel).parts)

    candidates: List[Path] = []

    def add_candidate(root: Path, tail_parts: Sequence[str]) -> None:
        candidates.append((root / Path(*tail_parts)).resolve())

    if parts:
        first = parts[0]
        tail = parts[1:]
        add_candidate(data_root, parts)

        if data_root.name in {"NSCLC_Radiogenomics", "NSCLC Radiogenomics", "Lung-PET-CT-Dx"}:
            if first == data_root.name:
                add_candidate(data_root, tail)
            else:
                add_candidate(data_root, parts)
        else:
            dataset_roots = (
                "NSCLC_Radiogenomics",
                "NSCLC Radiogenomics",
                "Lung-PET-CT-Dx",
            )
            if first in dataset_roots:
                for ds_root in dataset_roots:
                    if first != ds_root:
                        continue
                    add_candidate(data_root / ds_root, tail)
                    add_candidate(data_root, [ds_root, *tail])
            else:
                for ds_root in dataset_roots:
                    add_candidate(data_root, [ds_root, *parts])
                    add_candidate(data_root / ds_root, parts)

        if first == "NSCLC Radiogenomics":
            add_candidate(data_root, ["NSCLC_Radiogenomics", *tail])
        elif first == "NSCLC_Radiogenomics":
            add_candidate(data_root, ["NSCLC Radiogenomics", *tail])

    for c in candidates:
        if c.exists():
            return c

    tail = Path(*parts[-3:]) if len(parts) >= 3 else Path(*parts)
    if data_root.exists():
        for root in [
            data_root,
            data_root / "NSCLC_Radiogenomics",
            data_root / "NSCLC Radiogenomics",
            data_root / "Lung-PET-CT-Dx",
        ]:
            if not root.exists():
                continue
            probe = root / tail
            if probe.exists():
                return probe.resolve()

    raise FileNotFoundError(f"Could not resolve file location {file_location!r} under {data_root}")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _classify_nsclc_series(row: Mapping[str, Any], abs_path: Path) -> SeriesRecord:
    desc = str(row.get("Series Description") or "").strip()
    desc_l = desc.lower()
    study_desc = str(row.get("Study Description") or "").strip()
    study_l = study_desc.lower()
    modality = str(row.get("Modality") or "").strip().upper()
    n_images = _safe_int(row.get("Number of Images"), 0)
    role = "other"
    score = 0.0

    if modality == "CT":
        if n_images <= 2 or any(k in desc_l for k in ["scout", "topogram"]):
            role = "localizer"
            score = -100.0
        elif "fusion" in desc_l:
            role = "fusion_ct"
            score = 10.0
        elif "ctac" in desc_l or "ldct" in desc_l or ("ct images" in desc_l and "pet" in study_l):
            role = "ctac"
            score = 50.0
        elif any(k in desc_l for k in ["coronal", "mpr", "coronals", "in reach", "reference only"]):
            role = "reformat_ct"
            score = 15.0
        else:
            role = "diagnostic_ct"
            score = 60.0
            if any(k in desc_l for k in ["lung", "thorax", "chest"]):
                score += 20.0
            if re.search(r"\b1\s*mm\b", desc_l) or "1mm" in desc_l or "1.0" in desc_l or "5.0.0" in desc_l or "b70" in desc_l or "b45" in desc_l:
                score += 12.0
            if any(k in desc_l for k in ["contrast", "with con"]):
                score += 3.0
            if "pet" in study_l:
                score -= 12.0
        score += min(n_images, 500) / 100.0

    elif modality in {"PT", "PET"}:
        if "nac" in desc_l or "noac" in desc_l:
            role = "pet_nac"
            score = 60.0
        elif "mac" in desc_l or "ctac" in desc_l:
            role = "pet_ac"
            score = 65.0
        else:
            role = "pet_recon"
            score = 45.0
            if "ramla" in desc_l:
                score += 10.0
        score += min(n_images, 500) / 100.0

    return SeriesRecord(
        subject_id=str(row.get("Subject ID") or row.get("Case ID") or ""),
        study_uid=str(row.get("Study UID") or ""),
        study_date=str(row.get("Study Date") or ""),
        study_description=study_desc,
        series_uid=str(row.get("Series UID") or ""),
        series_description=desc,
        modality=modality,
        n_images=n_images,
        abs_path=abs_path,
        raw_file_location=str(row.get("File Location") or ""),
        role=role,
        score=score,
    )


def _pick_best(records: Sequence[SeriesRecord], role: str) -> Optional[SeriesRecord]:
    subset = [r for r in records if r.role == role]
    if not subset:
        return None
    return sorted(subset, key=lambda r: (r.score, r.n_images, r.study_date, r.series_description), reverse=True)[0]


def build_nsclc_case_catalog(*, data_root: Path, metadata_csv: Path, subject_id: str) -> CaseSeriesCatalog:
    rows = _read_csv_rows(metadata_csv)
    subject_rows = [r for r in rows if str(r.get("Subject ID") or "").strip() == str(subject_id).strip()]
    all_records: List[SeriesRecord] = []
    notes: List[str] = []
    for row in subject_rows:
        try:
            abs_path = resolve_tcia_file_location(str(row.get("File Location") or ""), Path(data_root))
        except Exception as e:
            notes.append(f"Skipped unresolved series {row.get('Series Description')!r}: {e}")
            continue
        all_records.append(_classify_nsclc_series(row, abs_path))

    # Keep the visible case catalog strictly imaging-only so hidden support objects such as SEG/SR
    # are never preloaded into the agent-facing scene inventory or prompt.
    records = [r for r in all_records if str(r.modality or "").upper() in IMAGE_MODALITIES]

    alias_map: Dict[str, List[SeriesRecord]] = {}

    diagnostic_ct = _pick_best(records, "diagnostic_ct") or _pick_best(records, "ctac")
    pet_nac = _pick_best(records, "pet_nac")
    pet_ac = _pick_best(records, "pet_ac")
    pet_primary = pet_ac or pet_nac or _pick_best(records, "pet_recon")
    ctac = _pick_best(records, "ctac")
    fusion_ct = _pick_best(records, "fusion_ct")

    if diagnostic_ct:
        alias_map["diagnostic_ct"] = [diagnostic_ct]
        alias_map["ct_only"] = [diagnostic_ct]
    if pet_primary:
        alias_map["pet_primary"] = [pet_primary]
        alias_map["pet_only"] = [pet_primary]
    if pet_nac:
        alias_map["pet_nac"] = [pet_nac]
    if pet_ac:
        alias_map["pet_ac"] = [pet_ac]
    if ctac:
        alias_map["ctac"] = [ctac]
    if fusion_ct:
        alias_map["fusion_ct"] = [fusion_ct]

    core: List[SeriesRecord] = []
    if diagnostic_ct:
        core.append(diagnostic_ct)
    if pet_primary and all(p.abs_path != pet_primary.abs_path for p in core):
        core.append(pet_primary)
    if core:
        alias_map["core"] = core

    pet_ct_core: List[SeriesRecord] = []
    pet_ct_ref = ctac or fusion_ct or diagnostic_ct
    if pet_ct_ref:
        pet_ct_core.append(pet_ct_ref)
    if pet_primary and all(p.abs_path != pet_primary.abs_path for p in pet_ct_core):
        pet_ct_core.append(pet_primary)
    if pet_ct_core:
        alias_map["pet_ct_core"] = pet_ct_core

    default_aliases: List[str] = []
    if "core" in alias_map:
        default_aliases.append("core")
    elif "diagnostic_ct" in alias_map:
        default_aliases.append("diagnostic_ct")
    elif "pet_primary" in alias_map:
        default_aliases.append("pet_primary")

    if diagnostic_ct and pet_primary and diagnostic_ct.study_date != pet_primary.study_date:
        notes.append(
            f"Diagnostic CT date ({diagnostic_ct.study_date}) differs from PET date ({pet_primary.study_date}); use PET mainly for adjunctive assessment, and use pet_ct_core for aligned PET/CT when needed."
        )

    return CaseSeriesCatalog(
        subject_id=subject_id,
        data_root=Path(data_root).expanduser().resolve(),
        series=sorted(records, key=lambda r: (r.study_date, r.modality, r.series_description, str(r.abs_path))),
        alias_map=alias_map,
        default_aliases=default_aliases,
        notes=notes,
    )


def format_catalog_for_prompt(catalog: CaseSeriesCatalog) -> str:
    lines: List[str] = []
    lines.append(f"All resolved imaging series for {catalog.subject_id} (all timepoints that exist on disk):")
    by_date = catalog.series_by_study_date(prefer_latest=False)
    for study_date, records in by_date.items():
        lines.append(f"- Study {study_date}:")
        for r in records:
            lines.append(f"  - {r.modality} | {r.series_description} | role={r.role} | images={r.n_images}")
    if catalog.alias_map:
        lines.append("")
        lines.append("Curated alias groups:")
        for alias, records in catalog.alias_map.items():
            rec_bits = []
            for r in records:
                rec_bits.append(r.display_name)
            lines.append(f"- {alias}: {' ; '.join(rec_bits)}")
    if catalog.notes:
        lines.append("Notes:")
        for n in catalog.notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def iter_nsclc_subject_ids(metadata_csv: Path) -> List[str]:
    rows = _read_csv_rows(metadata_csv)
    ids = sorted({str(r.get("Subject ID") or "").strip() for r in rows if str(r.get("Subject ID") or "").strip()})
    return ids
