#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from qupath_agent_engine.agent.loop import BoundedAgent
from qupath_agent_engine.agent.tooling import build_tool_registry
from qupath_agent_engine.app.common import (
    build_case_spec,
    build_qupath_connector,
    build_run_context,
    describe_run_artifacts,
    prepare_loaded_case_session,
)
from qupath_agent_engine.llm.factory import build_model
from qupath_agent_engine.software.qupath.config import default_image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the QuPath demo agent.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--image", type=str, action="append", default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument(
        "--qupath-executable",
        type=str,
        default=None,
        help="QuPath executable. Defaults to QUPATH_EXECUTABLE or platform discovery.",
    )
    parser.add_argument("--sidecar-dir", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--disable-gui-agent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide Midscene GUI-agent tools from the model and keep only direct QuPath controls.",
    )
    parser.add_argument("--save", action="store_true", help="Persist evidence images instead of treating them as disposable.")
    parser.add_argument("--close-on-exit", action="store_true", help="Close QuPath without saving at the end.")
    parser.add_argument("--leave-open", action="store_true", help="Leave QuPath open even if --close-on-exit is absent.")
    parser.add_argument("--runs-dir", type=str, default="./runs/demo")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    task = args.task or input("Task for the agent: ").strip()
    if not task:
        raise SystemExit("A non-empty task is required.")

    image_args = list(args.image or [])
    fallback_image = default_image_path()
    if not image_args and fallback_image is not None:
        image_args.append(str(fallback_image))
    if not image_args and not args.project:
        raise SystemExit(
            "No QuPath case was provided. Pass --image PATH or --project PATH "
            "(or set QUPATH_DEFAULT_IMAGE)."
        )
    run_context = build_run_context(runs_dir=args.runs_dir, run_prefix="demo", save_evidence=bool(args.save))
    case = build_case_spec(image_args=image_args, project=args.project)
    connector = build_qupath_connector(
        run_context=run_context,
        case=case,
        qupath_executable=args.qupath_executable,
        sidecar_dir=args.sidecar_dir,
        close_on_exit=bool(args.close_on_exit),
    )

    prepared = prepare_loaded_case_session(
        connector=connector,
        task=task,
        case_summary=case.summary_text(),
        initial_label="after_load_case",
    )
    run_context.summary["case_summary"] = case.summary_dict()

    model = build_model(provider=args.provider, model=args.model)
    tool_registry = build_tool_registry(connector, disable_gui_agent=args.disable_gui_agent)
    agent = BoundedAgent(
        model=model,
        connector=connector,
        run_context=run_context,
        max_steps=args.max_steps,
        tool_registry=tool_registry,
    )

    result = agent.run(task=task, case_summary=case.summary_text(), initial_observation=prepared.initial_observation)
    run_artifacts = describe_run_artifacts(run_context)
    print("\n=== Final Answer ===\n")
    print(result.final_answer)
    print(f"\nRun directory: {run_artifacts['run_dir']}")
    print(f"Run log: {run_artifacts['runlog_path']}")

    if args.close_on_exit and not args.leave_open:
        connector.close_qupath(save=False)

    run_context.finalize()


if __name__ == "__main__":
    main()
