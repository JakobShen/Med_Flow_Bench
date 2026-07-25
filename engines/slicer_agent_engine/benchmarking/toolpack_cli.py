from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Tuple

from medflow_prompts import render_prompt

from ..toolpacks import (
    TOOL_PACK_ACTIVATION_MODE_MANUAL,
    TOOL_PACK_ACTIVATION_MODES,
    ToolPackSelection,
)

TOOLPACK_PROMPT_REF = "3d_slicer/toolpacks.json"


@dataclass(frozen=True)
class BenchmarkToolPackFlag:
    arg_name: str
    pack_names: Tuple[str, ...]
    help: str
    prompt_template_keys: Tuple[str, ...] = ()


_BENCHMARK_TOOLPACK_FLAGS: Tuple[BenchmarkToolPackFlag, ...] = (
    BenchmarkToolPackFlag(
        arg_name="segmentation_toolpacks",
        pack_names=("segmentation",),
        help="Provide the segmentation tool pack in addition to the baseline/core tools.",
        prompt_template_keys=("templates.benchmark_hints.segmentation",),
    ),
    BenchmarkToolPackFlag(
        arg_name="registration_toolpacks",
        pack_names=("registration",),
        help="Provide the registration tool pack in addition to the baseline/core tools.",
        prompt_template_keys=("templates.benchmark_hints.registration",),
    ),
    BenchmarkToolPackFlag(
        arg_name="monai_label_toolpacks",
        pack_names=("monai_label",),
        help="Provide MONAI Label / VISTA3D tools in addition to the baseline/core tools.",
        prompt_template_keys=("templates.benchmark_hints.monai_label",),
    ),
)


def add_toolpack_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--activate_toolpacks",
        choices=TOOL_PACK_ACTIVATION_MODES,
        default=TOOL_PACK_ACTIVATION_MODE_MANUAL,
        help=(
            "How provided optional tool packs become usable. "
            "manual: the agent must call activate_tool_pack first. "
            "auto: all tools from the provided tool packs are exposed immediately."
        ),
    )
    parser.add_argument(
        "--all_toolpacks",
        action="store_true",
        help="Provide all optional tool packs in addition to the baseline/core tools.",
    )
    for spec in _BENCHMARK_TOOLPACK_FLAGS:
        parser.add_argument(f"--{spec.arg_name}", action="store_true", help=spec.help)


def resolve_toolpack_cli_selection(args: Any) -> ToolPackSelection:
    selected_packs = set()
    prompt_hints = []
    for spec in _BENCHMARK_TOOLPACK_FLAGS:
        if bool(getattr(args, spec.arg_name, False)):
            selected_packs.update(spec.pack_names)
            prompt_hints.extend(
                render_prompt(TOOLPACK_PROMPT_REF, template_key)
                for template_key in spec.prompt_template_keys
            )

    provided_tool_packs = None if bool(getattr(args, "all_toolpacks", False)) else tuple(sorted(selected_packs))
    return ToolPackSelection(
        provided_tool_packs=provided_tool_packs,
        activation_mode=str(getattr(args, "activate_toolpacks", TOOL_PACK_ACTIVATION_MODE_MANUAL) or TOOL_PACK_ACTIVATION_MODE_MANUAL),
        prompt_hints=tuple(prompt_hints),
    )


__all__ = [
    "BenchmarkToolPackFlag",
    "TOOLPACK_PROMPT_REF",
    "add_toolpack_cli_arguments",
    "resolve_toolpack_cli_selection",
]
