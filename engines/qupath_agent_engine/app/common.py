from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qupath_agent_engine.core.run_context import RunContext
from qupath_agent_engine.core.run_logger import RunLogger
from qupath_agent_engine.core.run_paths import RunPaths
from qupath_agent_engine.core.types import CaseSpec, StepObservation
from qupath_agent_engine.software.qupath.config import QuPathConfig
from qupath_agent_engine.software.qupath.connector import QuPathConnector


@dataclass(slots=True)
class PreparedLoadedCaseSession:
    connector: QuPathConnector
    bootstrap_result: dict[str, Any]
    load_result: dict[str, Any]
    initial_image_type_result: dict[str, Any] | None
    initial_observation: StepObservation


def build_run_context(*, runs_dir: str | Path, run_prefix: str, save_evidence: bool) -> RunContext:
    run_paths = RunPaths.create(base_dir=Path(runs_dir), run_prefix=run_prefix)
    run_logger = RunLogger(log_path=run_paths.logs_dir / "runlog.jsonl")
    return RunContext(run_paths=run_paths, logger=run_logger, save_evidence=bool(save_evidence))


def describe_run_artifacts(run_context: RunContext) -> dict[str, str]:
    return {
        "run_dir": str(run_context.run_paths.root),
        "runlog_path": str(run_context.run_paths.logs_dir / "runlog.jsonl"),
        "evidence_dir": str(run_context.run_paths.evidence_dir),
        "summary_path": str(run_context.run_paths.root / "summary.json"),
    }


def build_case_spec(*, image_args: Sequence[str] | None, project: str | None) -> CaseSpec:
    images = [Path(item) for item in (image_args or []) if item]
    case = CaseSpec(
        image_paths=images,
        project_path=Path(project) if project else None,
    ).normalized()
    case.validate()
    return case


def build_qupath_connector(
    *,
    run_context: RunContext,
    case: CaseSpec,
    qupath_executable: str | Path | None,
    sidecar_dir: str | Path | None = None,
    close_on_exit: bool = False,
    redact_model_state: bool = False,
) -> QuPathConnector:
    config = QuPathConfig.from_env()
    if qupath_executable is not None:
        config.executable_path = Path(qupath_executable)
    config.sidecar_dir = Path(sidecar_dir).expanduser().resolve() if sidecar_dir else config.sidecar_dir
    config.close_on_exit = bool(close_on_exit)
    return QuPathConnector(
        config=config,
        run_context=run_context,
        case_spec=case,
        redact_model_state=bool(redact_model_state),
    )


def prepare_loaded_case_session(
    *,
    connector: QuPathConnector,
    task: str,
    case_summary: str,
    initial_label: str = "after_load_case",
    initial_image_type: str | None = None,
) -> PreparedLoadedCaseSession:
    bootstrap_result = {
        "ok": True,
        "skipped": True,
        "reason": "This flow loads a case directly. Use bootstrap() only for open-only flows.",
    }
    load_result = connector.load_case()
    initial_image_type_result: dict[str, Any] | None = None
    if initial_image_type:
        try:
            initial_image_type_result = connector.set_image_type(image_type=initial_image_type)
        except Exception as exc:
            initial_image_type_result = {
                "ok": False,
                "error": f"Failed to set initial image type to {initial_image_type}: {exc}",
            }
            connector.run_context.logger.event(
                "initial_image_type_error",
                requested_image_type=initial_image_type,
                result=initial_image_type_result,
            )

    initial_observation = connector.observe_current_state(
        round_idx=1,
        task=task,
        case_summary=case_summary,
        label=initial_label,
    )

    connector.run_context.summary["bootstrap"] = bootstrap_result
    connector.run_context.summary["load_case"] = load_result
    connector.run_context.summary["initial_image_type_result"] = initial_image_type_result
    connector.run_context.summary["initial_observation"] = initial_observation.summary_dict()
    connector.run_context.logger.event(
        "prepared_loaded_case_session",
        bootstrap=bootstrap_result,
        load=load_result,
        initial_image_type_result=initial_image_type_result,
        initial_observation=initial_observation.summary_dict(),
    )
    return PreparedLoadedCaseSession(
        connector=connector,
        bootstrap_result=bootstrap_result,
        load_result=load_result,
        initial_image_type_result=initial_image_type_result,
        initial_observation=initial_observation,
    )
