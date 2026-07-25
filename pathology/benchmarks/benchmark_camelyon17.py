#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Callable, TypeVar

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from medflow_prompts import prompt_metadata
from qupath_agent_engine.agent.loop import BoundedAgent
from qupath_agent_engine.agent.tooling import build_tool_registry
from qupath_agent_engine.app.common import (
    build_case_spec,
    build_qupath_connector,
    build_run_context,
    describe_run_artifacts,
    prepare_loaded_case_session,
)
from qupath_agent_engine.benchmarking.camelyon17 import (
    CAMELYON17_BENCHMARK_TOOL_NAMES,
    CAMELYON17_SLIDE_LABELS,
    aggregate_camelyon17_slide_case_results,
    build_camelyon17_slide_paper_data,
    build_camelyon17_slide_case_summary,
    build_camelyon17_slide_system_instruction,
    build_camelyon17_slide_task_prompt,
    load_camelyon17_manifest,
    score_camelyon17_slide_case,
)
from qupath_agent_engine.core.run_logger import RunLogger
from qupath_agent_engine.core.utils import dump_json, local_timestamp_slug, sanitize_filename
from qupath_agent_engine.llm.factory import build_model


DEFAULT_RUNS_DIR = "./runs/benchmark_camelyon17"
DEFAULT_INITIAL_IMAGE_TYPE = "BRIGHTFIELD_H_E"
PROMPT_REF = "pathology/camelyon17.json"
COMMON_PROMPT_REF = "pathology/common.json"
CONSOLE_SUMMARY_EXCLUDED_KEYS = frozenset({"selected_missing_image_case_ids"})
STATIC_MODEL_ERROR_MARKERS = (
    "Provider/model mismatch",
    "model_not_found",
    "does not exist",
    "unknown model",
    "invalid_function_parameters",
    "Invalid schema for function",
    "additional_properties",
    "Cannot find field",
)
_T = TypeVar("_T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CAMELYON17 QuPath benchmark (slide-level only in v1).")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing stages.csv and nested images/masks/annotations.")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--qupath-executable",
        type=str,
        default=None,
        help="QuPath executable. Defaults to QUPATH_EXECUTABLE or platform discovery.",
    )
    parser.add_argument("--sidecar-dir", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--disable-gui-agent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide Midscene GUI-agent tools from the model and keep only direct QuPath controls.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N runnable slide cases after filtering. 0 means all runnable cases.")
    parser.add_argument("--center", dest="centers", action="append", default=None, help="Optional center filter. May be repeated.")
    parser.add_argument("--slide-label", dest="slide_labels", action="append", default=None, help="Optional slide-label filter. May be repeated, e.g. --slide-label micro")
    parser.add_argument("--patient-id", dest="patient_ids", action="append", default=None, help="Optional exact patient id filter. May be repeated, e.g. --patient-id patient_000")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=None, help="Optional exact slide case id filter. May be repeated, e.g. --case-id patient_000_node_4")
    parser.add_argument(
        "--balance-subtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance sampled cases across CAMELYON17 slide labels as evenly as possible before applying --limit.",
    )
    parser.add_argument(
        "--include-no-evidence",
        dest="require_evidence",
        action="store_false",
        help="Include slides without mask/xml evidence. By default, only evidence-backed slides are benchmarked.",
    )
    parser.add_argument("--restart-between-cases", action="store_true", help="Close QuPath after each case for a cleaner runtime state.")
    parser.add_argument("--leave-open", action="store_true", help="Leave QuPath open after the benchmark finishes.")
    parser.set_defaults(require_evidence=True)
    return parser


def _balanced_interleave(items: list[_T], *, key_fn: Callable[[_T], str | None], ordered_keys: tuple[str, ...]) -> list[_T]:
    buckets: dict[str, list[_T]] = {key: [] for key in ordered_keys}
    other_items: list[_T] = []
    for item in items:
        key = str(key_fn(item) or "").strip().lower()
        if key in buckets:
            buckets[key].append(item)
        else:
            other_items.append(item)
    balanced: list[_T] = []
    while any(buckets[key] for key in ordered_keys):
        for key in ordered_keys:
            if buckets[key]:
                balanced.append(buckets[key].pop(0))
    balanced.extend(other_items)
    return balanced


def _count_by_key(items: list[_T], *, key_fn: Callable[[_T], str | None], ordered_keys: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {key: 0 for key in ordered_keys}
    other = 0
    for item in items:
        key = str(key_fn(item) or "").strip().lower()
        if key in counts:
            counts[key] += 1
        else:
            other += 1
    if other:
        counts["__other__"] = other
    return counts


def _detect_static_model_error(steps: list[dict[str, object]]) -> str | None:
    if not steps:
        return None
    errors: list[str] = []
    for step in steps:
        if step.get("tool") is not None:
            return None
        tool_result = step.get("tool_result")
        if not isinstance(tool_result, dict):
            return None
        error_text = str(tool_result.get("error") or "").strip()
        if not error_text:
            return None
        errors.append(error_text)
    if not errors:
        return None
    if all(any(marker in error for marker in STATIC_MODEL_ERROR_MARKERS) for error in errors):
        return errors[0]
    return None


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    manifest = load_camelyon17_manifest(args.data_root)
    benchmark_root = Path(args.runs_dir).expanduser().resolve() / f"camelyon17_slide_benchmark_{local_timestamp_slug()}"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    cases_root = benchmark_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    benchmark_logger = RunLogger(log_path=benchmark_root / "benchmark_runlog.jsonl")

    runnable_cases = manifest.runnable_slides(
        allowed_centers=args.centers,
        allowed_slide_labels=args.slide_labels,
        allowed_patient_ids=args.patient_ids,
    )
    skipped_missing_images = manifest.skipped_missing_image_slides(
        allowed_centers=args.centers,
        allowed_slide_labels=args.slide_labels,
        allowed_patient_ids=args.patient_ids,
    )

    if args.require_evidence:
        runnable_cases = [case for case in runnable_cases if case.has_evidence_source]
        skipped_missing_images = [case for case in skipped_missing_images if case.has_evidence_source]

    if args.case_ids:
        case_filter = {str(case_id).strip().lower() for case_id in args.case_ids if str(case_id).strip()}
        runnable_cases = [case for case in runnable_cases if case.case_id.lower() in case_filter]
        skipped_missing_images = [case for case in skipped_missing_images if case.case_id.lower() in case_filter]

    if args.balance_subtypes:
        runnable_cases = _balanced_interleave(
            runnable_cases,
            key_fn=lambda case: case.gt_slide_label,
            ordered_keys=CAMELYON17_SLIDE_LABELS,
        )

    if args.limit and args.limit > 0:
        runnable_cases = runnable_cases[: int(args.limit)]

    manifest_payload = manifest.summary_dict()
    manifest_payload["requested_centers"] = list(args.centers or [])
    manifest_payload["requested_slide_labels"] = list(args.slide_labels or [])
    manifest_payload["requested_patient_ids"] = list(args.patient_ids or [])
    manifest_payload["requested_case_ids"] = list(args.case_ids or [])
    manifest_payload["require_evidence"] = bool(args.require_evidence)
    manifest_payload["balance_subtypes"] = bool(args.balance_subtypes)
    manifest_payload["disable_gui_agent"] = bool(args.disable_gui_agent)
    manifest_payload["n_selected_runnable_cases"] = len(runnable_cases)
    manifest_payload["n_selected_missing_image_cases"] = len(skipped_missing_images)
    manifest_payload["selected_slide_label_counts"] = _count_by_key(
        runnable_cases,
        key_fn=lambda case: case.gt_slide_label,
        ordered_keys=CAMELYON17_SLIDE_LABELS,
    )
    manifest_payload["selected_missing_image_case_ids"] = [case.case_id for case in skipped_missing_images]
    dump_json(benchmark_root / "manifest.json", manifest_payload)

    benchmark_logger.event("benchmark_start", benchmark_root=str(benchmark_root), manifest=manifest_payload)

    task_text = build_camelyon17_slide_task_prompt()
    system_instruction = build_camelyon17_slide_system_instruction()
    case_results: list[dict[str, object]] = []
    last_connector = None

    print(f"Benchmark root: {benchmark_root}", flush=True)
    print(f"Runnable slide cases: {len(runnable_cases)}", flush=True)
    if skipped_missing_images:
        print(f"Skipped because slide image was not found: {len(skipped_missing_images)}", flush=True)

    for index, case in enumerate(runnable_cases, start=1):
        case_alias = f"case_{index:04d}"
        run_context = build_run_context(
            runs_dir=cases_root,
            run_prefix=sanitize_filename(case.case_id.lower(), fallback="case"),
            save_evidence=True,
        )
        run_context.benchmark_logger = benchmark_logger
        case_artifacts = describe_run_artifacts(run_context)
        connector = None
        case_result: dict[str, object]
        judge_log: dict[str, object] | None = None
        try:
            print(f"[{index}/{len(runnable_cases)}] {case.case_id}", flush=True)
            benchmark_logger.event("case_start", case=case.summary_dict(), ordinal=index, run_artifacts=case_artifacts)

            case_spec = build_case_spec(image_args=[str(case.image_path)], project=None)
            connector = build_qupath_connector(
                run_context=run_context,
                case=case_spec,
                qupath_executable=args.qupath_executable,
                sidecar_dir=args.sidecar_dir,
                close_on_exit=False,
                redact_model_state=True,
            )
            last_connector = connector
            case_summary = build_camelyon17_slide_case_summary(case_alias=case_alias)
            prepared = prepare_loaded_case_session(
                connector=connector,
                task=task_text,
                case_summary=case_summary,
                initial_label="after_load_case",
                initial_image_type=DEFAULT_INITIAL_IMAGE_TYPE,
            )

            run_context.summary["benchmark_case"] = case.summary_dict()
            run_context.summary["benchmark_case_alias"] = case_alias
            run_context.summary["benchmark_track"] = "camelyon17_slide"
            run_context.summary["task"] = task_text
            run_context.summary["case_summary_text"] = case_summary
            run_context.summary["run_artifacts"] = case_artifacts

            model = build_model(provider=args.provider, model=args.model)
            tool_registry = build_tool_registry(
                connector,
                enabled_tool_names=CAMELYON17_BENCHMARK_TOOL_NAMES,
                disable_gui_agent=args.disable_gui_agent,
                evidence_label_options=(),
                evidence_requires_supported_label=False,
                evidence_requires_description=False,
            )
            agent = BoundedAgent(
                model=model,
                connector=connector,
                run_context=run_context,
                max_steps=args.max_steps,
                system_instruction=system_instruction,
                tool_registry=tool_registry,
                prompt_metadata={
                    "definitions": [
                        prompt_metadata(COMMON_PROMPT_REF),
                        prompt_metadata(PROMPT_REF),
                    ]
                },
            )
            agent_result = agent.run(
                task=task_text,
                case_summary=case_summary,
                initial_observation=prepared.initial_observation,
            )
            static_model_error = _detect_static_model_error(agent_result.steps)
            if static_model_error is not None:
                raise RuntimeError(static_model_error)
            case_result, judge_log = score_camelyon17_slide_case(
                case=case,
                final_text=agent_result.final_answer,
                steps=agent_result.steps,
            )
            case_result.update(case_artifacts)
            case_result["final_answer_text"] = agent_result.final_answer

            run_context.summary["benchmark_result"] = case_result
            run_context.summary["judge_log"] = judge_log
            dump_json(run_context.run_paths.root / "benchmark_result.json", case_result)
            dump_json(run_context.run_paths.root / "judge_log.json", judge_log)
            run_context.logger.event("judge_case", judge_log=judge_log)
            benchmark_logger.event(
                "case_judged",
                case_id=case.case_id,
                judge_log=judge_log,
                run_dir=case_artifacts["run_dir"],
            )
            benchmark_logger.event(
                "case_end",
                case_id=case.case_id,
                result=case_result,
                run_dir=case_artifacts["run_dir"],
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            case_result = {
                "case_id": case.case_id,
                "csv_row_index": case.csv_row_index,
                "slide_filename": case.slide_filename,
                "patient_id": case.patient_id,
                "node_index": case.node_index,
                "center": case.center,
                "patient_stage": case.patient_stage,
                "gt_slide_label": case.gt_slide_label,
                "image_path": str(case.image_path) if case.image_path else None,
                "mask_path": str(case.mask_path) if case.mask_path else None,
                "xml_path": str(case.xml_path) if case.xml_path else None,
                "error": error_text,
                "traceback": traceback.format_exc(),
                "raw_tumor_presence_correct": False,
                "raw_slide_label_correct": False,
                "evidence_gt_available": case.has_evidence_source,
                "evidence_required_by_prediction": None,
                "evidence_scored": False,
                "evidence_status": None,
                "evidence_correct": None,
                "supported_tumor_presence_correct": None,
                "supported_slide_label_correct": None,
                "n_steps": None,
                "n_direct_tools": None,
                "n_gui_tools": None,
                "n_output_localizations": None,
                "n_recorded_evidence": None,
            }
            judge_log = {
                "case_id": case.case_id,
                "error": error_text,
                "traceback": traceback.format_exc(),
            }
            case_result.update(case_artifacts)
            run_context.summary["benchmark_error"] = case_result
            run_context.summary["judge_log"] = judge_log
            dump_json(run_context.run_paths.root / "benchmark_result.json", case_result)
            dump_json(run_context.run_paths.root / "judge_log.json", judge_log)
            run_context.logger.event("judge_case_error", judge_log=judge_log)
            benchmark_logger.event("case_error", case_id=case.case_id, error=case_result)
            print(f"  error: {error_text}", flush=True)
        finally:
            case_results.append(case_result)
            run_context.finalize()
            if connector is not None and args.restart_between_cases:
                try:
                    connector.close_qupath(save=False)
                except Exception as close_exc:
                    benchmark_logger.event("case_close_error", case_id=case.case_id, error=str(close_exc))

    if last_connector is not None and not args.restart_between_cases and not args.leave_open:
        try:
            last_connector.close_qupath(save=False)
        except Exception as exc:
            benchmark_logger.event("benchmark_close_error", error=str(exc))

    aggregate = aggregate_camelyon17_slide_case_results(case_results)
    aggregate["benchmark_root"] = str(benchmark_root)
    aggregate["benchmark_track"] = "camelyon17_slide"
    aggregate["n_selected_runnable_cases"] = len(runnable_cases)
    aggregate["n_selected_missing_image_cases"] = len(skipped_missing_images)
    aggregate["selected_missing_image_case_ids"] = [case.case_id for case in skipped_missing_images]
    aggregate["n_cases_with_mask"] = sum(1 for case in runnable_cases if case.mask_path is not None)
    aggregate["n_cases_with_xml"] = sum(1 for case in runnable_cases if case.xml_path is not None)
    aggregate["n_case_errors"] = sum(1 for row in case_results if row.get("error"))
    paper_data = build_camelyon17_slide_paper_data(case_results, benchmark_root=str(benchmark_root))

    dump_json(benchmark_root / "benchmark_summary.json", aggregate)
    dump_json(benchmark_root / "paper_data.json", paper_data)
    (benchmark_root / "benchmark_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in case_results) + ("\n" if case_results else ""),
        encoding="utf-8",
    )
    benchmark_logger.event("benchmark_end", aggregate=aggregate)

    print("\n=== CAMELYON17 Slide Benchmark Summary ===", flush=True)
    console_summary = {key: value for key, value in aggregate.items() if key not in CONSOLE_SUMMARY_EXCLUDED_KEYS}
    print(json.dumps(console_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results saved to: {benchmark_root}", flush=True)


if __name__ == "__main__":
    main()
