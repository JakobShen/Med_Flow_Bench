#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from medflow_prompts import prompt_metadata, render_prompt
from slicer_agent_engine.agents.auto_agent import run_auto_task
from slicer_agent_engine.algorithm_runtime import MonaiLabelRuntime
from slicer_agent_engine.app.bootstrap import (
    build_model_runtime,
    build_session_context,
    configure_logging,
    ensure_slicer_runtime,
    resolve_bridge_dir,
)
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.toolpacks import TOOL_PACK_ACTIVATION_MODE_AUTO


PROMPT_REF = "3d_slicer/runtime.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MONAI Label / VISTA3D Slicer-agent demo.")
    parser.add_argument("--nifti_file", required=True, help="Input NIfTI volume for the demo.")
    parser.add_argument("--out_dir", required=True, help="Output directory for traces and artifacts.")
    parser.add_argument("--base_url", default=os.environ.get("SLICER_BASE_URL", "http://127.0.0.1:2018"))
    parser.add_argument("--monai_url", default=os.environ.get("MONAI_LABEL_SERVER_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--slicer_executable", default=os.environ.get("SLICER_EXECUTABLE"))
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-preview"))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--monai_model", default=os.environ.get("MONAI_LABEL_MODEL", "vista3d"))
    parser.add_argument("--label_prompt", default="3", help="Comma-separated VISTA3D label ids.")
    parser.add_argument("--point_prompt_smoke", action="store_true", help="Also run one direct VISTA3D point-prompt smoke using points_ijk/point_labels.")
    parser.add_argument("--run_agent", action="store_true", help="Also run a minimal Gemini tool-calling prompt.")
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--slicer_timeout_s", type=float, default=240.0)
    parser.add_argument("--monai_timeout_s", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nifti_path = Path(args.nifti_file).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"nifti_file does not exist: {nifti_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir, log_name="demo_monai_vista3d.log")

    slicer_executable = Path(args.slicer_executable).expanduser().resolve() if args.slicer_executable else None
    bridge_dir = ensure_slicer_runtime(
        base_url=args.base_url,
        bridge_dir=resolve_bridge_dir(),
        slicer_executable=slicer_executable,
        require_exec=True,
        require_slice_render=True,
    )
    client = SlicerClient(base_url=args.base_url, timeout_s=float(args.slicer_timeout_s))
    monai_runtime = MonaiLabelRuntime(
        server_url=args.monai_url,
        tmp_dir=out_dir / "monai_tmp",
        timeout_s=float(args.monai_timeout_s),
    )
    context = build_session_context(
        client=client,
        out_dir=out_dir,
        bridge_dir=bridge_dir,
        algorithm_runtime=monai_runtime,
    )
    ctx = context.ctx

    label_prompt = [int(x.strip()) for x in str(args.label_prompt).split(",") if x.strip()]
    result: Dict[str, Any] = {"nifti_file": str(nifti_path), "label_prompt": label_prompt}
    result["monai_label_status"] = ctx.monai_label_status()
    result["load_nifti"] = ctx.load_nifti(str(nifti_path), clear_scene_first=True)
    result["vista3d_segment"] = ctx.vista3d_segment(
        model=args.monai_model,
        label_prompt=label_prompt or [3],
        segmentation_name=f"VISTA3D {','.join(str(x) for x in (label_prompt or [3]))}",
    )
    if args.point_prompt_smoke:
        result["vista3d_point_prompt"] = ctx.vista3d_segment(
            model=args.monai_model,
            label_prompt=(label_prompt[:1] if label_prompt else None),
            points_ijk=[[128, 128, 32]],
            point_labels=[1],
            segmentation_name="VISTA3D point prompt smoke",
            auto_show=True,
            auto_center=True,
        )
    result["segmentations"] = ctx.list_segmentations()

    if args.run_agent:
        model_runtime = build_model_runtime(provider=args.provider, model=args.model)
        agent = run_auto_task(
            provider=args.provider,
            model_client=model_runtime.model_client,
            model=args.model,
            ctx=ctx,
            case_path=nifti_path,
            instructions=render_prompt(PROMPT_REF, "templates.vista3d_demo_system"),
            user_message=render_prompt(PROMPT_REF, "templates.vista3d_demo_user"),
            max_rounds=int(args.max_rounds),
            reuse_current_scene=True,
            provided_tool_packs=("monai_label",),
            initial_active_packs=("monai_label",),
            tool_pack_activation_mode=TOOL_PACK_ACTIVATION_MODE_AUTO,
            extra_model_tools=model_runtime.extra_model_tools,
            prompt_metadata=prompt_metadata(PROMPT_REF),
        )
        result["agent_output"] = agent.text
        (out_dir / "agent_output.txt").write_text(agent.text, encoding="utf-8")

    (out_dir / "demo_result.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
