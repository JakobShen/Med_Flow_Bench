from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from medflow_prompts import render_prompt

from .helpers import method_display_name


PROMPT_REF = "track_c/ucsf_pdgm.json"


def build_system_instructions(
    *,
    method: str,
    uses_attached_images: bool,
    include_evidence: bool,
) -> str:
    method_name = method_display_name(method)
    return render_prompt(
        PROMPT_REF,
        "templates.system",
        method_name=method_name,
        delivery_guidance=render_prompt(
            PROMPT_REF,
            "templates.fragments.delivery.attached_images"
            if uses_attached_images
            else "templates.fragments.delivery.external_worker",
        ),
        evidence_guidance=render_prompt(
            PROMPT_REF,
            f"templates.fragments.evidence.{'enabled' if include_evidence else 'disabled'}",
        ),
    )


def _format_segment_requirement(segment_evidence: Dict[str, Any]) -> str:
    source_name = str(segment_evidence.get("source_volume_name") or "").strip()
    source_line = (
        render_prompt(
            PROMPT_REF,
            "templates.fragments.segment_source_line",
            source_name=source_name,
        )
        if source_name
        else ""
    )
    return render_prompt(PROMPT_REF, "templates.segment_requirement", source_line=source_line)


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
    return render_prompt(
        PROMPT_REF,
        "templates.user",
        case_id=case_id,
        method_name=method_name,
        series_lines=series_lines,
        artifact_block=artifact_block,
        evidence_block=evidence_block,
    )


__all__ = ["PROMPT_REF", "build_case_message", "build_system_instructions"]
