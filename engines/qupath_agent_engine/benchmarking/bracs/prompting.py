from __future__ import annotations

from medflow_prompts import render_prompt

from qupath_agent_engine.agent.prompts import HOST_SYSTEM_PROMPT

from .dataset import BRACS_LABELS


BRACS_OPTION_TO_LABEL = {
    "A": "N",
    "B": "PB",
    "C": "UDH",
    "D": "FEA",
    "E": "ADH",
    "F": "DCIS",
    "G": "IC",
}
BRACS_LABEL_TO_OPTION = {label: option for option, label in BRACS_OPTION_TO_LABEL.items()}
BRACS_BENCHMARK_TOOL_NAMES = (
    "zoom",
    "pan",
    "focus_qupath",
    "capture_viewer",
    "viewer_point_to_wsi_coordinate",
    "sleep",
)
PROMPT_REF = "pathology/bracs.json"


def build_bracs_system_instruction() -> str:
    extra = render_prompt(PROMPT_REF, "templates.system_extra")
    return f"{HOST_SYSTEM_PROMPT}\n\n{extra}".strip()


def build_bracs_case_summary(*, case_alias: str) -> str:
    return render_prompt(PROMPT_REF, "templates.case_summary", case_alias=case_alias)


def build_bracs_task_prompt() -> str:
    options = "\n".join(f"{option}. {label}" for option, label in BRACS_OPTION_TO_LABEL.items())
    allowed_labels = ", ".join(BRACS_LABELS)
    return render_prompt(
        PROMPT_REF,
        "templates.task",
        options=options,
        allowed_labels=allowed_labels,
    )
