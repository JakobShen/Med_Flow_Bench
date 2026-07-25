from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from medflow_prompts import prompt_metadata as load_prompt_metadata
from medflow_prompts import render_prompt

from ..capability_registry import build_default_tool_surface
from ..context_briefs import format_scene_inventory_for_prompt, make_image_input_message, make_text_input_message
from ..llm.loop import run_model_tool_loop
from ..llm.registry import build_model_adapter, resolve_provider_name
from ..toolpacks import TOOL_PACK_ACTIVATION_MODE_MANUAL
from ..tools import ToolContext

if TYPE_CHECKING:
    from ..load_helper import CaseSeriesCatalog

logger = logging.getLogger(__name__)
RUNTIME_PROMPT_REF = "3d_slicer/runtime.json"


@dataclass
class AutoAgentResult:
    """Result of a provider-neutral auto-viewer run."""

    text: str
    open_case: Dict[str, Any]
    input_messages: List[Any]


def attach_baseline_images(*, input_list: List[Any], baseline: Dict[str, str]) -> None:
    """Attach baseline axial/sagittal/coronal PNGs to the running input list."""

    for view_name in ("axial", "sagittal", "coronal"):
        png_path = baseline.get(view_name)
        if not png_path:
            continue
        path = Path(png_path)
        if not path.exists():
            continue
        input_list.append(
            make_image_input_message(
                str(path),
                text=render_prompt(
                    RUNTIME_PROMPT_REF,
                    "templates.agent.baseline_image",
                    view_name=view_name,
                ),
            )
        )


def run_auto_task(
    *,
    provider: str,
    model_client: Any,
    model: str,
    ctx: ToolContext,
    case_path: Union[str, Path],
    instructions: str,
    user_message: str,
    max_rounds: int = 12,
    max_tools: Optional[int] = None,
    attach_images: bool = True,
    max_images_per_tool: int = 4,
    open_case_kwargs: Optional[Dict[str, Any]] = None,
    pre_messages: Optional[List[Any]] = None,
    case_brief_text: Optional[str] = None,
    case_series_catalog: Optional["CaseSeriesCatalog"] = None,
    initial_active_packs: Optional[Sequence[str]] = None,
    provided_tool_packs: Optional[Sequence[str]] = None,
    tool_pack_activation_mode: str = TOOL_PACK_ACTIVATION_MODE_MANUAL,
    extra_tools_schema: Optional[Sequence[Dict[str, Any]]] = None,
    extra_tool_executor: Optional[Any] = None,
    reuse_current_scene: bool = False,
    extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
    prompt_metadata: Optional[Dict[str, Any]] = None,
    model_request_timeout_sec: Optional[float] = None,
    model_request_retry_on_timeout: int = 0,
    model_adapter: Optional[Any] = None,
) -> AutoAgentResult:
    """Run a provider-neutral tool-calling session against a single case."""

    if extra_tools_schema or extra_tool_executor:
        logger.warning(
            "extra_tools_schema/extra_tool_executor are deprecated and ignored; use case_brief_text + case_series_catalog + optional packs instead."
        )

    resolved_provider = resolve_provider_name(provider, model=model)

    tool_budget_note = render_prompt(
        RUNTIME_PROMPT_REF,
        "templates.agent.tool_budget",
        max_rounds=int(max_rounds),
    )
    image_coords_contract_note = render_prompt(
        RUNTIME_PROMPT_REF,
        "templates.agent.image_coords_contract",
    )
    instruction_parts: List[str] = []
    if instructions:
        instruction_parts.append(instructions.rstrip())
    instruction_parts.append(image_coords_contract_note)
    instruction_parts.append(tool_budget_note)
    effective_instructions = "\n\n".join(part for part in instruction_parts if part)
    caller_prompt_metadata = dict(prompt_metadata or {})
    if isinstance(caller_prompt_metadata.get("definitions"), list):
        prompt_definitions = list(caller_prompt_metadata["definitions"])
    elif caller_prompt_metadata:
        prompt_definitions = [caller_prompt_metadata]
    else:
        prompt_definitions = []
    runtime_metadata = load_prompt_metadata(RUNTIME_PROMPT_REF)
    if not any(
        isinstance(item, dict) and item.get("prompt_ref") == runtime_metadata["prompt_ref"]
        for item in prompt_definitions
    ):
        prompt_definitions.append(runtime_metadata)
    effective_prompt_metadata = {"definitions": prompt_definitions}

    if hasattr(ctx.session, "log_run_event"):
        ctx.session.log_run_event(
            "auto_task_start",
            provider=resolved_provider,
            model=model,
            case_path=str(Path(case_path).expanduser().resolve()),
            max_rounds=max_rounds,
            instructions=effective_instructions,
            raw_instructions=instructions,
            user_message=user_message,
            case_brief_text=case_brief_text,
            prompt_metadata=effective_prompt_metadata,
            initial_active_packs=list(initial_active_packs or []),
            provided_tool_packs=None if provided_tool_packs is None else list(provided_tool_packs),
            tool_pack_activation_mode=tool_pack_activation_mode,
            reuse_current_scene=bool(reuse_current_scene),
            reasoning_effort=reasoning_effort,
            model_request_timeout_sec=model_request_timeout_sec,
            model_request_retry_on_timeout=model_request_retry_on_timeout,
        )

    open_kwargs: Dict[str, Any] = {"baseline": True, "size": 512}
    if open_case_kwargs:
        open_kwargs.update(open_case_kwargs)

    if reuse_current_scene:
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("auto_task_reuse_current_scene_start", open_case_kwargs=open_kwargs)
        baseline = None
        try:
            baseline_obs = ctx.observe_standard_views(size=int(open_kwargs.get("size", 512)), out_prefix="continue_scene")
            baseline = baseline_obs.get("png_paths") if isinstance(baseline_obs, dict) else None
        except Exception:
            baseline = None
        try:
            viewer_state = ctx.get_viewer_state()
        except Exception as e:
            viewer_state = {"ok": False, "error": str(e)}
        try:
            scene_inventory = ctx.client.list_volumes()
        except Exception as e:
            scene_inventory = []
            scene_inventory_text = f"Currently loaded scene volumes: unavailable ({e})"
        else:
            scene_inventory_text = format_scene_inventory_for_prompt(
                scene_inventory,
                active_volume_id=ctx.session.state.get("active_volume_id"),
                active_volume_name=ctx.session.state.get("active_volume_name"),
            )
        ctx.session.update_state(scene_inventory=scene_inventory, scene_inventory_text=scene_inventory_text)
        open_summary = {
            "ok": True,
            "reused_current_scene": True,
            "baseline": baseline,
            "viewer_state": viewer_state,
            "scene_inventory": scene_inventory,
            "scene_inventory_text": scene_inventory_text,
        }
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event(
                "auto_task_reuse_current_scene_done",
                open_case_summary=ctx.session.summarize_tool_result(open_summary) if hasattr(ctx.session, "summarize_tool_result") else open_summary,
            )
    else:
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("auto_task_open_case_start", open_case_kwargs=open_kwargs)
        open_summary = ctx.open_case(case_path, **open_kwargs)
        baseline = open_summary.get("baseline") if isinstance(open_summary, dict) else None

        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event(
                "auto_task_open_case_done",
                open_case_summary=ctx.session.summarize_tool_result(open_summary) if hasattr(ctx.session, "summarize_tool_result") else open_summary,
            )

    surface = build_default_tool_surface(
        ctx=ctx,
        case_series_catalog=case_series_catalog,
        initial_active_packs=initial_active_packs,
        optional_pack_names=provided_tool_packs,
        pack_activation_mode=tool_pack_activation_mode,
    )

    input_list: List[Any] = []
    if pre_messages:
        input_list.extend(pre_messages)

    capability_index = surface.capability_index_text()
    if capability_index:
        input_list.append(make_text_input_message(capability_index))
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("capability_index_attached", capability_index=capability_index)

    if case_brief_text:
        input_list.append(make_text_input_message(case_brief_text))
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("case_brief_attached", case_brief_text=case_brief_text)

    if isinstance(open_summary, dict) and open_summary.get("scene_inventory_text"):
        input_list.append(make_text_input_message(str(open_summary["scene_inventory_text"])))
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("scene_inventory_attached", scene_inventory_text=str(open_summary["scene_inventory_text"]))

    input_list.append(make_text_input_message(user_message))
    if isinstance(baseline, dict):
        attach_baseline_images(input_list=input_list, baseline=baseline)
        if hasattr(ctx.session, "log_run_event"):
            ctx.session.log_run_event("baseline_images_attached", baseline=baseline)

    adapter = model_adapter or build_model_adapter(resolved_provider, model=model)
    report_text, final_input = run_model_tool_loop(
        adapter=adapter,
        client=model_client,
        model=model,
        tool_schemas=surface.current_tool_schemas,
        initial_inputs=input_list,
        instructions=effective_instructions,
        tool_executor=surface.execute_tool_call_json,
        max_rounds=max_rounds,
        max_tools=max_tools,
        attach_images=attach_images,
        max_images_per_tool=max_images_per_tool,
        session=ctx.session,
        extra_model_tools=extra_model_tools,
        reasoning_effort=reasoning_effort,
        model_request_timeout_sec=model_request_timeout_sec,
        model_request_retry_on_timeout=model_request_retry_on_timeout,
    )

    if hasattr(ctx.session, "log_run_event"):
        ctx.session.log_run_event("auto_task_finished", provider=resolved_provider, final_text=report_text)

    return AutoAgentResult(text=report_text, open_case=open_summary, input_messages=final_input)
