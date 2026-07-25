#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from qupath_agent_engine.app.common import build_case_spec, build_qupath_connector, build_run_context
from qupath_agent_engine.software.qupath.config import default_image_path
from qupath_agent_engine.software.qupath.direct_control import (
    PAN_DIRECTION_OPTIONS,
    PAN_UNIT_OPTIONS,
    QUPATH_IMAGE_TYPE_OPTIONS,
    ZOOM_DIRECTION_OPTIONS,
    ZOOM_UNIT_OPTIONS,
)


DEFAULT_ZOOM_TOLERANCE = 0.15



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-LLM local smoke test for QuPath launch/load/direct-control/capture/rule-based state."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--project-image-name", type=str, default=None)
    parser.add_argument(
        "--qupath-executable",
        type=str,
        default=None,
        help="QuPath executable. Defaults to QUPATH_EXECUTABLE or platform discovery.",
    )
    parser.add_argument("--sidecar-dir", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--close-on-exit", action="store_true")
    parser.add_argument("--with-midscene", action="store_true", help="Also run one explicit Midscene GUI query tool call.")
    parser.add_argument("--image-type", type=str, default="BRIGHTFIELD_H_E", choices=sorted(QUPATH_IMAGE_TYPE_OPTIONS.keys()))
    parser.add_argument("--zoom-direction", type=str, default="in", choices=list(ZOOM_DIRECTION_OPTIONS))
    parser.add_argument("--zoom-amount", type=float, default=10.0)
    parser.add_argument("--zoom-unit", type=str, default="magnification", choices=list(ZOOM_UNIT_OPTIONS))
    parser.add_argument("--zoom-tolerance", type=float, default=DEFAULT_ZOOM_TOLERANCE)
    parser.add_argument("--pan-direction", type=str, default="right", choices=list(PAN_DIRECTION_OPTIONS))
    parser.add_argument("--pan-amount", type=float, default=0.20)
    parser.add_argument("--pan-unit", type=str, default="viewport_fraction", choices=list(PAN_UNIT_OPTIONS))
    parser.add_argument("--runs-dir", type=str, default="./runs/local_smoke")
    return parser



def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if not args.image:
        fallback_image = default_image_path()
        args.image = str(fallback_image) if fallback_image is not None else None
    if not args.image and not args.project:
        raise SystemExit(
            "No QuPath case was provided. Pass --image PATH or --project PATH "
            "(or set QUPATH_DEFAULT_IMAGE)."
        )

    run_context = build_run_context(runs_dir=args.runs_dir, run_prefix="local_smoke", save_evidence=bool(args.save))
    case = build_case_spec(image_args=[args.image] if args.image else [], project=args.project)
    connector = build_qupath_connector(
        run_context=run_context,
        case=case,
        qupath_executable=args.qupath_executable,
        sidecar_dir=args.sidecar_dir,
        close_on_exit=bool(args.close_on_exit),
    )

    bootstrap_result = {
        "ok": True,
        "skipped": True,
        "reason": "This smoke test loads a case directly. Use bootstrap() only for open-only flows.",
    }
    load_result = connector.load_case()
    state_after_load = connector.query_qupath_state()

    open_project_result = None
    if args.project:
        open_project_result = connector.open_project(project_path=args.project, image_name=args.project_image_name)

    load_image_tool_result = None
    if args.image:
        load_image_tool_result = connector.load_image(image_path=args.image)

    set_image_type_result = connector.set_image_type(image_type=args.image_type)
    zoom_result = connector.zoom(direction=args.zoom_direction, amount=args.zoom_amount, unit=args.zoom_unit)
    pan_result = connector.pan(direction=args.pan_direction, amount=args.pan_amount, unit=args.pan_unit)
    state_after_navigation = connector.query_qupath_state()
    viewer_capture = connector.capture_viewer(label="after_direct_tools")
    observation = connector.observe_current_state(
        round_idx=1,
        task="local smoke test",
        case_summary=case.summary_text(),
        label="after_direct_tools_state_probe",
    )

    explicit_midscene_query = None
    if args.with_midscene:
        explicit_midscene_query = connector.qupath_gui_query(
            question="Describe only the currently visible QuPath controls, tabs, toolbars, and dialogs. Ignore pathology content."
        )

    _assert_ok("load_case", load_result)
    _assert_ok("state_after_load", state_after_load)
    _assert_ok("set_image_type", set_image_type_result)
    _assert_ok("zoom", zoom_result)
    _assert_ok("pan", pan_result)
    _assert_ok("state_after_navigation", state_after_navigation)
    _assert_ok("viewer_capture", viewer_capture)

    _assert_direct_tool_result_shape("set_image_type", set_image_type_result)
    _assert_direct_tool_result_shape("zoom", zoom_result)
    _assert_direct_tool_result_shape("pan", pan_result)
    _assert_rule_based_gui_state(observation.summary_dict())
    _assert_zoom_behavior(
        state_after_navigation=state_after_navigation,
        requested_amount=float(args.zoom_amount),
        requested_unit=str(args.zoom_unit),
        tolerance=float(args.zoom_tolerance),
    )

    print("Bootstrap:", bootstrap_result)
    print("Load case:", load_result)
    print("State after initial load:", state_after_load)
    print("Open project tool:", open_project_result)
    print("Load image tool:", load_image_tool_result)
    print("Set image type tool:", set_image_type_result)
    print("Zoom tool:", zoom_result)
    print("Pan tool:", pan_result)
    print("State after navigation:", state_after_navigation)
    print("Viewer capture:", viewer_capture)
    print("Rule-based observation:", observation.summary_dict())
    if explicit_midscene_query is not None:
        print("Explicit Midscene GUI query:", explicit_midscene_query)
    print(f"Run directory: {run_context.run_paths.root}")
    print(f"Run log: {run_context.run_paths.logs_dir / 'runlog.jsonl'}")

    run_context.summary["bootstrap"] = bootstrap_result
    run_context.summary["load_case"] = load_result
    run_context.summary["state_after_load"] = state_after_load
    run_context.summary["open_project_result"] = open_project_result
    run_context.summary["load_image_tool_result"] = load_image_tool_result
    run_context.summary["set_image_type_result"] = set_image_type_result
    run_context.summary["zoom_result"] = zoom_result
    run_context.summary["pan_result"] = pan_result
    run_context.summary["state_after_navigation"] = state_after_navigation
    run_context.summary["viewer_capture"] = viewer_capture
    run_context.summary["observation"] = observation.summary_dict()
    run_context.summary["explicit_midscene_query"] = explicit_midscene_query

    if args.close_on_exit:
        connector.close_qupath(save=False)

    run_context.finalize()



def _assert_ok(name: str, payload: dict) -> None:
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise SystemExit(f"Smoke test failed: {name} returned a non-ok payload: {payload}")



def _assert_rule_based_gui_state(observation: dict) -> None:
    gui_state = observation.get("gui_state") or {}
    gui_summary = str(observation.get("gui_summary") or "").strip()
    if not isinstance(gui_state, dict) or not gui_state.get("ok"):
        raise SystemExit(f"Smoke test failed: rule-based gui_state is unavailable: {gui_state}")
    if not gui_summary:
        raise SystemExit("Smoke test failed: gui_summary is empty.")



def _assert_zoom_behavior(*, state_after_navigation: dict, requested_amount: float, requested_unit: str, tolerance: float) -> None:
    if requested_unit != "magnification":
        return
    state = state_after_navigation.get("state") or {}
    viewer = state.get("viewer") or {}
    magnification = viewer.get("magnification")
    if not isinstance(magnification, (int, float)) or not math.isfinite(float(magnification)):
        raise SystemExit(
            f"Smoke test failed: viewer magnification missing after zoom-to-magnification: {state_after_navigation}"
        )
    if abs(float(magnification) - requested_amount) > tolerance:
        raise SystemExit(
            "Smoke test failed: zoom-to-magnification did not reach the requested target. "
            f"requested={requested_amount}x, actual={float(magnification):.3f}x, tolerance={tolerance}"
        )


def _assert_direct_tool_result_shape(name: str, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise SystemExit(f"Smoke test failed: {name} returned a non-dict payload: {payload}")
    forbidden_keys = {"state", "state_before", "state_after"}
    leaked = sorted(key for key in forbidden_keys if key in payload)
    if leaked:
        raise SystemExit(
            f"Smoke test failed: {name} leaked observation/context fields into the immediate tool result: {leaked}; payload={payload}"
        )
    text_output = str(payload.get("text_output") or "").strip()
    if not text_output:
        raise SystemExit(f"Smoke test failed: {name} result is missing text_output: {payload}")


if __name__ == "__main__":
    main()
