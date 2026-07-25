from __future__ import annotations

from medflow_prompts import render_prompt

from qupath_agent_engine.agent.prompts import HOST_SYSTEM_PROMPT


CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL = {
    "A": "no_tumor",
    "B": "tumor_present",
}
CAMELYON17_TUMOR_PRESENCE_LABEL_TO_OPTION = {
    label: option for option, label in CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL.items()
}
CAMELYON17_SLIDE_OPTION_TO_LABEL = {
    "A": "negative",
    "B": "itc",
    "C": "micro",
    "D": "macro",
}
CAMELYON17_SLIDE_LABEL_TO_OPTION = {
    label: option for option, label in CAMELYON17_SLIDE_OPTION_TO_LABEL.items()
}
CAMELYON17_BENCHMARK_TOOL_NAMES = (
    "zoom",
    "pan",
    "focus_qupath",
    "capture_viewer",
    "viewer_point_to_wsi_coordinate",
    "sleep",
)
PROMPT_REF = "pathology/camelyon17.json"


def build_camelyon17_slide_system_instruction() -> str:
    extra = render_prompt(PROMPT_REF, "templates.system_extra")
    return f"{HOST_SYSTEM_PROMPT}\n\n{extra}".strip()


def build_camelyon17_slide_case_summary(*, case_alias: str) -> str:
    return render_prompt(PROMPT_REF, "templates.case_summary", case_alias=case_alias)


def build_camelyon17_slide_task_prompt() -> str:
    tumor_options = "\n".join(
        f"{option}. {label}" for option, label in CAMELYON17_TUMOR_PRESENCE_OPTION_TO_LABEL.items()
    )
    slide_options = "\n".join(
        f"{option}. {label}" for option, label in CAMELYON17_SLIDE_OPTION_TO_LABEL.items()
    )
    return render_prompt(
        PROMPT_REF,
        "templates.task",
        tumor_options=tumor_options,
        slide_options=slide_options,
    )
