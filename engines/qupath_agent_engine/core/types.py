from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class CaseSpec:
    """A software-agnostic case description.

    In v1, QuPath uses this conservatively:
    - if project_path is set, that project is opened;
    - otherwise the first image path is opened;
    - any remaining image paths are preserved in metadata for future project workflows.
    """

    image_paths: list[Path] = field(default_factory=list)
    project_path: Path | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "CaseSpec":
        project = self.project_path.expanduser().resolve() if self.project_path else None
        images = [Path(p).expanduser().resolve() for p in self.image_paths]
        case_id = self.case_id or self._infer_case_id(project=project, images=images)
        return CaseSpec(
            image_paths=images,
            project_path=project,
            case_id=case_id,
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        if self.project_path is None and not self.image_paths:
            raise ValueError("CaseSpec requires at least one image path or a project path.")
        if self.project_path and not self.project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {self.project_path}")
        for image in self.image_paths:
            if not image.exists():
                raise FileNotFoundError(f"Image path does not exist: {image}")

    @property
    def primary_image(self) -> Path | None:
        return self.image_paths[0] if self.image_paths else None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "project_path": str(self.project_path) if self.project_path else None,
            "primary_image": str(self.primary_image) if self.primary_image else None,
            "additional_images": [str(p) for p in self.image_paths[1:]],
            "metadata": dict(self.metadata),
        }

    def summary_text(self) -> str:
        normalized = self.normalized()
        normalized.validate()
        lines = [f"Case ID: {normalized.case_id}"]
        if normalized.project_path:
            lines.append(f"Project: {normalized.project_path}")
        if normalized.primary_image:
            lines.append(f"Primary image: {normalized.primary_image}")
        if len(normalized.image_paths) > 1:
            lines.append("Additional images:")
            lines.extend(f"- {path}" for path in normalized.image_paths[1:])
        if normalized.metadata:
            lines.append("Metadata:")
            for key, value in normalized.metadata.items():
                lines.append(f"- {key}: {value}")
        lines.append(
            "v1 loading policy: open the project if provided; otherwise open the primary image and keep"
            " additional image paths in host context for future manual opening or later project workflows."
        )
        return "\n".join(lines)

    @staticmethod
    def _infer_case_id(*, project: Path | None, images: list[Path]) -> str:
        if project is not None:
            return project.stem or project.name
        if images:
            return images[0].stem
        return "case"


@dataclass(slots=True)
class StepObservation:
    round_idx: int
    task: str
    case_summary: str
    viewer_capture: dict[str, Any]
    viewer_image_path: Path | None
    qupath_state: dict[str, Any] = field(default_factory=dict)
    qupath_state_summary: str = ""
    gui_state: dict[str, Any] = field(default_factory=dict)
    gui_summary: str = ""

    def summary_dict(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "viewer_capture": dict(self.viewer_capture),
            "viewer_image_path": str(self.viewer_image_path) if self.viewer_image_path else None,
            "qupath_state": dict(self.qupath_state),
            "qupath_state_summary": self.qupath_state_summary,
            "gui_state": dict(self.gui_state),
            "gui_summary": self.gui_summary,
        }


@dataclass(slots=True)
class PendingToolResult:
    call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    result: dict[str, Any]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "result": dict(self.result),
        }


@dataclass(slots=True)
class ModelDecision:
    kind: Literal["tool", "final"]
    raw_response: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    assistant_text: str | None = None
    provider_response_id: str | None = None
