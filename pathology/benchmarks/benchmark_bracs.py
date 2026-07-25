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
from qupath_agent_engine.benchmarking.bracs import (
    BRACS_BENCHMARK_TOOL_NAMES,
    BRACS_LABELS,
    aggregate_bracs_case_results,
    build_bracs_paper_data,
    build_bracs_case_summary,
    build_bracs_system_instruction,
    build_bracs_task_prompt,
    load_bracs_manifest,
    score_bracs_case,
)
from qupath_agent_engine.core.run_logger import RunLogger
from qupath_agent_engine.core.utils import dump_json, local_timestamp_slug, sanitize_filename
from qupath_agent_engine.llm.factory import build_model


DEFAULT_RUNS_DIR = "./runs/benchmark_bracs"
DEFAULT_INITIAL_IMAGE_TYPE = "BRIGHTFIELD_H_E"
PROMPT_REF = "pathology/bracs.json"
COMMON_PROMPT_REF = "pathology/common.json"
CONSOLE_SUMMARY_EXCLUDED_KEYS = frozenset({"selected_missing_svs_case_ids"})
_T = TypeVar("_T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BRACS QuPath benchmark.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing BRACS.xlsx and nested slide/annotation files.")
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
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N runnable cases after filtering. 0 means all runnable cases.")
    parser.add_argument("--set", dest="sets", action="append", default=None, help="Optional split filter. May be repeated, e.g. --set Testing --set Validation")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=None, help="Optional exact BRACS case ID filter. May be repeated.")
    parser.add_argument(
        "--forced_qpdata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require qpdata-backed BRACS cases only. Runnable cases without qpdata are skipped before balancing and --limit.",
    )
    parser.add_argument(
        "--balance-subtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance sampled BRACS cases across slide labels as evenly as possible before applying --limit.",
    )
    parser.add_argument("--restart-between-cases", action="store_true", help="Close QuPath after each case for a cleaner runtime state.")
    parser.add_argument("--leave-open", action="store_true", help="Leave QuPath open after the benchmark finishes.")
    return parser


def _balanced_interleave(items: list[_T], *, key_fn: Callable[[_T], str | None], ordered_keys: tuple[str, ...]) -> list[_T]:
    buckets: dict[str, list[_T]] = {key: [] for key in ordered_keys}
    other_items: list[_T] = []
    for item in items:
        key = str(key_fn(item) or "").strip().upper()
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
        key = str(key_fn(item) or "").strip().upper()
        if key in counts:
            counts[key] += 1
        else:
            other += 1
    if other:
        counts["__other__"] = other
    return counts


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    manifest = load_bracs_manifest(args.data_root)
    benchmark_root = Path(args.runs_dir).expanduser().resolve() / f"bracs_benchmark_{local_timestamp_slug()}"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    cases_root = benchmark_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    benchmark_logger = RunLogger(log_path=benchmark_root / "benchmark_runlog.jsonl")

    runnable_cases = manifest.runnable_cases(allowed_splits=args.sets)
    skipped_missing_svs = manifest.skipped_missing_svs_cases(allowed_splits=args.sets)
    skipped_missing_qpdata: list = []

    if args.case_ids:
        case_filter = {str(case_id).strip().upper() for case_id in args.case_ids if str(case_id).strip()}
        runnable_cases = [case for case in runnable_cases if case.case_id.upper() in case_filter]
        skipped_missing_svs = [case for case in skipped_missing_svs if case.case_id.upper() in case_filter]

    if args.forced_qpdata:
        skipped_missing_qpdata = [case for case in runnable_cases if case.qpdata_path is None]
        runnable_cases = [case for case in runnable_cases if case.qpdata_path is not None]

    if args.balance_subtypes:
        runnable_cases = _balanced_interleave(
            runnable_cases,
            key_fn=lambda case: case.gt_label,
            ordered_keys=BRACS_LABELS,
        )

    if args.limit and args.limit > 0:
        runnable_cases = runnable_cases[: int(args.limit)]

    manifest_payload = manifest.summary_dict()
    manifest_payload["requested_sets"] = list(args.sets or [])
    manifest_payload["requested_case_ids"] = list(args.case_ids or [])
    manifest_payload["balance_subtypes"] = bool(args.balance_subtypes)
    manifest_payload["disable_gui_agent"] = bool(args.disable_gui_agent)
    manifest_payload["forced_qpdata"] = bool(args.forced_qpdata)
    manifest_payload["n_selected_runnable_cases"] = len(runnable_cases)
    manifest_payload["n_selected_missing_svs_cases"] = len(skipped_missing_svs)
    manifest_payload["n_selected_missing_qpdata_cases"] = len(skipped_missing_qpdata)
    manifest_payload["selected_label_counts"] = _count_by_key(
        runnable_cases,
        key_fn=lambda case: case.gt_label,
        ordered_keys=BRACS_LABELS,
    )
    manifest_payload["selected_missing_svs_case_ids"] = [case.case_id for case in skipped_missing_svs]
    dump_json(benchmark_root / "manifest.json", manifest_payload)

    benchmark_logger.event("benchmark_start", benchmark_root=str(benchmark_root), manifest=manifest_payload)

    task_text = build_bracs_task_prompt()
    system_instruction = build_bracs_system_instruction()
    case_results: list[dict[str, object]] = []
    last_connector = None

    print(f"Benchmark root: {benchmark_root}", flush=True)
    print(f"Runnable cases: {len(runnable_cases)}", flush=True)
    if skipped_missing_svs:
        print(f"Skipped because .svs was not found: {len(skipped_missing_svs)}", flush=True)
    if skipped_missing_qpdata:
        print(f"Skipped because .qpdata was not found: {len(skipped_missing_qpdata)}", flush=True)

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
        case_result: dict[str, object] = {
            "case_id": case.case_id,
            "excel_row_index": case.excel_row_index,
            "split": case.split,
            "gt_label": case.gt_label,
            "svs_path": str(case.svs_path) if case.svs_path else None,
            "qpdata_path": str(case.qpdata_path) if case.qpdata_path else None,
            "error": "Case execution did not complete.",
            "traceback": None,
            "wsi_label_correct": False,
            "evidence_scored": False,
            "evidence_label_correct": None,
            "n_steps": None,
            "n_direct_tools": None,
            "n_gui_tools": None,
            "n_output_localizations": None,
            "n_recorded_evidence": None,
        }
        try:
            print(f"[{index}/{len(runnable_cases)}] {case.case_id}", flush=True)
            benchmark_logger.event("case_start", case=case.summary_dict(), ordinal=index, run_artifacts=case_artifacts)

            case_spec = build_case_spec(image_args=[str(case.svs_path)], project=None)
            connector = build_qupath_connector(
                run_context=run_context,
                case=case_spec,
                qupath_executable=args.qupath_executable,
                sidecar_dir=args.sidecar_dir,
                close_on_exit=False,
                redact_model_state=True,
            )
            last_connector = connector
            case_summary = build_bracs_case_summary(case_alias=case_alias)
            prepared = prepare_loaded_case_session(
                connector=connector,
                task=task_text,
                case_summary=case_summary,
                initial_label="after_load_case",
                initial_image_type=DEFAULT_INITIAL_IMAGE_TYPE,
            )

            run_context.summary["benchmark_case"] = case.summary_dict()
            run_context.summary["benchmark_case_alias"] = case_alias
            run_context.summary["task"] = task_text
            run_context.summary["case_summary_text"] = case_summary
            run_context.summary["run_artifacts"] = case_artifacts

            model = build_model(provider=args.provider, model=args.model)
            tool_registry = build_tool_registry(
                connector,
                enabled_tool_names=BRACS_BENCHMARK_TOOL_NAMES,
                disable_gui_agent=args.disable_gui_agent,
                evidence_label_options=BRACS_LABELS,
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
            case_result = score_bracs_case(
                case=case,
                final_text=agent_result.final_answer,
                steps=agent_result.steps,
                connector=connector,
            )
            case_result.update(case_artifacts)
            case_result["final_answer_text"] = agent_result.final_answer
            judge_log = {
                "case_id": case.case_id,
                "gt_label": case_result.get("gt_label"),
                "predicted_label": case_result.get("predicted_label"),
                "wsi_label_correct": case_result.get("wsi_label_correct"),
                "evidence_scored": case_result.get("evidence_scored"),
                "evidence_label_correct": case_result.get("evidence_label_correct"),
                "joint_correct": case_result.get("joint_correct"),
                "parsed_answer": case_result.get("parsed_answer"),
                "evidence_results": case_result.get("evidence_results"),
                "evidence_errors": case_result.get("evidence_errors"),
            }

            run_context.summary["benchmark_result"] = case_result
            run_context.summary["judge_log"] = judge_log
            dump_json(run_context.run_paths.root / "benchmark_result.json", case_result)
            dump_json(run_context.run_paths.root / "judge_log.json", judge_log)
            run_context.logger.event("judge_case", judge_log=judge_log)
            benchmark_logger.event("case_judged", case_id=case.case_id, judge_log=judge_log, run_dir=case_artifacts["run_dir"])
            benchmark_logger.event(
                "case_end",
                case_id=case.case_id,
                result=case_result,
                run_dir=case_artifacts["run_dir"],
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            case_result = {
                **case_result,
                "error": error_text,
                "traceback": traceback.format_exc(),
            }
            case_result.update(case_artifacts)
            judge_log = {
                "case_id": case.case_id,
                "error": error_text,
                "traceback": case_result.get("traceback"),
            }
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

    aggregate = aggregate_bracs_case_results(case_results)
    aggregate["benchmark_root"] = str(benchmark_root)
    aggregate["n_selected_runnable_cases"] = len(runnable_cases)
    aggregate["n_selected_missing_svs_cases"] = len(skipped_missing_svs)
    aggregate["n_selected_missing_qpdata_cases"] = len(skipped_missing_qpdata)
    aggregate["n_cases_with_qpdata"] = sum(1 for case in runnable_cases if case.qpdata_path is not None)
    aggregate["n_case_errors"] = sum(1 for row in case_results if row.get("error"))
    paper_data = build_bracs_paper_data(case_results, benchmark_root=str(benchmark_root))

    dump_json(benchmark_root / "benchmark_summary.json", aggregate)
    dump_json(benchmark_root / "paper_data.json", paper_data)
    (benchmark_root / "benchmark_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in case_results) + ("\n" if case_results else ""),
        encoding="utf-8",
    )
    benchmark_logger.event("benchmark_end", aggregate=aggregate)

    print("\n=== BRACS Benchmark Summary ===", flush=True)
    console_summary = {key: value for key, value in aggregate.items() if key not in CONSOLE_SUMMARY_EXCLUDED_KEYS}
    print(json.dumps(console_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results saved to: {benchmark_root}", flush=True)


if __name__ == "__main__":
    main()
