from __future__ import annotations

from medflow_prompts import render_prompt

from qupath_agent_engine.core.types import PendingToolResult, StepObservation


COMMON_PROMPT_REF = "pathology/common.json"
HOST_SYSTEM_PROMPT = render_prompt(COMMON_PROMPT_REF, "templates.system")


def render_observation_text(
    *,
    observation: StepObservation,
    pending_tool_result: PendingToolResult | None,
    is_first_turn: bool,
    force_final: bool = False,
) -> str:
    if is_first_turn:
        opening = render_prompt(
            COMMON_PROMPT_REF,
            "templates.observation_first",
            task=observation.task,
            case_summary=observation.case_summary,
        )
    else:
        opening = render_prompt(
            COMMON_PROMPT_REF,
            "templates.observation_continue",
            round_idx=observation.round_idx,
        )

    tool_notice = ""
    if pending_tool_result is not None:
        tool_notice = render_prompt(COMMON_PROMPT_REF, "templates.tool_result_notice")

    state = render_prompt(
        COMMON_PROMPT_REF,
        "templates.observation_state",
        qupath_state_summary=observation.qupath_state_summary or "QuPath direct state unavailable.",
        gui_summary=observation.gui_summary or "GUI summary unavailable.",
    )
    if force_final:
        closing = render_prompt(COMMON_PROMPT_REF, "templates.force_final")
    else:
        closing = render_prompt(COMMON_PROMPT_REF, "templates.decide_next")
    return f"{opening}{tool_notice}{state}{closing}".strip()
