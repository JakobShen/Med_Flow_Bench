from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .helpers import method_display_name


def build_system_instructions(
    *,
    method: str,
    uses_attached_images: bool,
    include_evidence: bool,
) -> str:
    method_name = method_display_name(method)
    parts = [
        "You are a neuroradiology benchmark model running in MEDOPENCLAW Track C / Open-Method mode. ",
        f"The study representation comes from the Track3 method '{method_name}'. ",
        "Your task is a 3-way multiple-choice diagnosis on pre-operative brain MRI. ",
        "Choose exactly one label: A, B, or C. ",
        "Do not output JSON. Write a brief rationale, then end with a single final line exactly formatted as 'ANSWER: <A|B|C>'. ",
        "If uncertain, still choose the most likely option. ",
    ]
    if uses_attached_images:
        parts.extend(
            [
                "You will receive attached figures or slice images. ",
                "Whenever a tile caption contains a sequence name, plane name, or canonical axis coordinate in mm, use that annotation directly rather than inferring a different coordinate system. ",
            ]
        )
    else:
        parts.extend(
            [
                "You will receive one MRI volume at a time through the backend model interface. ",
                "Base your answer only on the current volume under review. ",
            ]
        )
    if include_evidence:
        parts.extend(
            [
                "If an EVIDENCE block is requested, include it exactly once before the final answer line. ",
                "KEY_SLICE_AXIAL_MM refers to the axial S coordinate in canonical RAS space; KEY_SLICE_SAGITTAL_MM refers to the sagittal R coordinate; KEY_SLICE_CORONAL_MM refers to the coronal A coordinate. ",
                "RSA must be a single [R, A, S] point inside the lesion, not background. ",
            ]
        )
    return "".join(parts)


def _format_segment_requirement(segment_evidence: Dict[str, Any]) -> str:
    source_name = str(segment_evidence.get("source_volume_name") or "").strip()
    source_line = (
        f"A geometry-matched reference MRI volume is '{source_name}'. You may use it as a convenient anchor if relevant.\n"
        if source_name
        else ""
    )
    return (
        "Additional evidence output is required for this case.\n"
        "The benchmark has a hidden tumor segmentation in a shared co-registered MRI space.\n"
        + source_line
        + "If there are multiple disconnected tumor foci, localize the largest connected tumor component.\n"
        + "After your brief rationale, output the following block exactly once before the final answer line:\n"
        + "EVIDENCE:\n"
        + "KEY_SLICE_AXIAL_MM: <float>\n"
        + "KEY_SLICE_SAGITTAL_MM: <float>\n"
        + "KEY_SLICE_CORONAL_MM: <float>\n"
        + "RSA: [<R>, <A>, <S>]\n"
    )


def build_case_message(
    *,
    case_id: str,
    case_path: Path,
    nifti_files: Sequence[Path],
    method: str,
    artifact_summary_lines: Sequence[str],
    segment_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    method_name = method_display_name(method)
    series_lines = "\n".join(f"- {path.name}" for path in nifti_files) if nifti_files else "- <none>"
    artifact_block = "\n".join(f"- {line}" for line in artifact_summary_lines) if artifact_summary_lines else "- No artifact summary available."
    evidence_block = ""
    if segment_evidence and bool(segment_evidence.get("has_segment")):
        evidence_block = _format_segment_requirement(segment_evidence) + "\n"
    return (
        f"Case ID: {case_id}\n"
        f"Absolute case path: {case_path.expanduser().resolve()}\n"
        f"Track3 method: {method_name}\n\n"
        "Available MRI series for this case:\n"
        f"{series_lines}\n\n"
        "Representation summary for the current Track3 method:\n"
        f"{artifact_block}\n\n"
        "Task: classify the case into one option:\n"
        "  A) Glioblastoma\n"
        "  B) Oligodendroglioma / Astrocytoma\n"
        "  C) No tumor\n\n"
        + evidence_block
        + "Write a brief rationale, then end with a single final line exactly as: ANSWER: <A|B|C>."
    )


__all__ = ["build_case_message", "build_system_instructions"]
