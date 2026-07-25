from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from medflow_prompts import prompt_metadata as load_prompt_metadata

from qupath_agent_engine.core.run_context import RunContext
from qupath_agent_engine.core.types import PendingToolResult, StepObservation
from qupath_agent_engine.core.utils import compact
from qupath_agent_engine.llm.base import BaseHostModel
from qupath_agent_engine.software.qupath.connector import QuPathConnector

from .prompts import COMMON_PROMPT_REF, HOST_SYSTEM_PROMPT
from .tooling import ToolRegistry, build_tool_registry


@dataclass(slots=True)
class AgentResult:
    final_answer: str
    steps: list[dict[str, Any]]


class BoundedAgent:
    def __init__(
        self,
        *,
        model: BaseHostModel,
        connector: QuPathConnector,
        run_context: RunContext,
        max_steps: int = 16,
        system_instruction: str | None = None,
        tool_registry: ToolRegistry | None = None,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.connector = connector
        self.run_context = run_context
        self.max_steps = max(1, int(max_steps))
        self.tool_registry: ToolRegistry = tool_registry or build_tool_registry(self.connector)
        self.system_instruction = str(system_instruction or HOST_SYSTEM_PROMPT).strip()
        self.prompt_metadata = dict(
            prompt_metadata
            or {"definitions": [load_prompt_metadata(COMMON_PROMPT_REF)]}
        )
        self.run_context.summary["system_instruction"] = self.system_instruction
        self.run_context.summary["prompt_metadata"] = self.prompt_metadata
        self.model.start_session(system_instruction=self.system_instruction, tools=self.tool_registry.tools)

    def run(
        self,
        *,
        task: str,
        case_summary: str,
        initial_observation: StepObservation | None = None,
    ) -> AgentResult:
        transcript: list[dict[str, Any]] = []
        pending_tool_result: PendingToolResult | None = None
        observation = initial_observation or self.connector.observe_current_state(
            round_idx=1,
            task=task,
            case_summary=case_summary,
            label="round_0001_view",
        )
        self.run_context.logger.event(
            "agent_run_start",
            task=task,
            case_summary=case_summary,
            system_instruction=self.system_instruction,
            prompt_metadata=self.prompt_metadata,
            max_steps=self.max_steps,
        )

        for round_idx in range(1, self.max_steps + 1):
            if observation.round_idx != round_idx:
                observation = self.connector.observe_current_state(
                    round_idx=round_idx,
                    task=task,
                    case_summary=case_summary,
                    label=f"round_{round_idx:04d}_view",
                )

            self.run_context.logger.event(
                "agent_round_start",
                round_idx=round_idx,
                pending_tool_result=pending_tool_result.summary_dict() if pending_tool_result else None,
                observation=observation.summary_dict(),
            )

            try:
                decision = self.model.decide_next_step(
                    observation=observation,
                    pending_tool_result=pending_tool_result,
                    force_final=False,
                )
            except Exception as exc:
                error = {
                    "ok": False,
                    "error": f"Model step failed: {exc}",
                }
                self.run_context.logger.event("agent_generation_error", round_idx=round_idx, error=error)
                transcript.append(
                    {
                        "round": round_idx,
                        "tool": None,
                        "args": {},
                        "tool_result": error,
                    }
                )
                if _is_static_model_config_error(exc):
                    final_answer = (
                        "The host stopped before acting because the model configuration is invalid. "
                        "Check the run log for the provider/model mismatch, missing model, or tool schema error."
                    )
                    self.run_context.summary["final_answer"] = final_answer
                    self.run_context.summary["steps"] = transcript
                    self.run_context.logger.event(
                        "agent_static_generation_error",
                        round_idx=round_idx,
                        error=error,
                    )
                    return AgentResult(final_answer=final_answer, steps=transcript)
                pending_tool_result = None
                observation = self.connector.observe_current_state(
                    round_idx=round_idx + 1,
                    task=task,
                    case_summary=case_summary,
                    label=f"round_{round_idx + 1:04d}_view",
                )
                continue

            pending_tool_result = None
            self.run_context.logger.event(
                "agent_decision",
                round_idx=round_idx,
                decision=compact(decision.raw_response),
                parsed={
                    "kind": decision.kind,
                    "tool_name": decision.tool_name,
                    "tool_args": decision.tool_args,
                    "tool_call_id": decision.tool_call_id,
                    "final_answer": decision.final_answer,
                    "provider_response_id": decision.provider_response_id,
                },
            )

            if decision.kind == "final":
                final_answer = str(decision.final_answer or "")
                final_answer = _guard_final_answer_against_unmet_magnification(
                    task=task,
                    observation=observation,
                    final_answer=final_answer,
                )
                self.run_context.summary["final_answer"] = final_answer
                self.run_context.summary["steps"] = transcript
                self.run_context.logger.event("agent_final_answer", round_idx=round_idx, final_answer=final_answer)
                return AgentResult(final_answer=final_answer, steps=transcript)

            tool_name = str(decision.tool_name or "")
            tool_args = dict(decision.tool_args or {})
            tool = self.tool_registry.get(tool_name)
            if tool is None:
                error = {"ok": False, "error": f"Unknown tool: {tool_name}"}
                self.run_context.logger.event("tool_lookup_error", round_idx=round_idx, tool_name=tool_name, error=error)
                next_observation = self.connector.observe_current_state(
                    round_idx=round_idx + 1,
                    task=task,
                    case_summary=case_summary,
                    label=f"after_unknown_{tool_name or 'tool'}",
                )
                pending_tool_result = PendingToolResult(
                    call_id=decision.tool_call_id or f"local_call_{round_idx:04d}",
                    tool_name=tool_name or "unknown_tool",
                    tool_args=tool_args,
                    result=error,
                )
                transcript.append(
                    {
                        "round": round_idx,
                        "tool": tool_name,
                        "args": compact(tool_args),
                        "tool_result": compact(error),
                        "post_observation": compact(next_observation.summary_dict()),
                    }
                )
                observation = next_observation
                continue

            self.run_context.logger.event(
                "tool_call_start",
                round_idx=round_idx,
                tool_name=tool_name,
                tool_args=tool_args,
            )
            try:
                tool_result = tool.handler(dict(tool_args))
            except Exception as exc:
                tool_result = {"ok": False, "error": f"Tool execution failed: {exc}"}
                self.run_context.logger.event(
                    "tool_call_exception",
                    round_idx=round_idx,
                    tool_name=tool_name,
                    error=tool_result,
                )

            next_observation = self.connector.observe_current_state(
                round_idx=round_idx + 1,
                task=task,
                case_summary=case_summary,
                label=f"after_{tool_name}",
            )
            self.run_context.logger.event(
                "tool_call_end",
                round_idx=round_idx,
                tool_name=tool_name,
                tool_result=tool_result,
                post_observation=next_observation.summary_dict(),
            )

            pending_tool_result = PendingToolResult(
                call_id=decision.tool_call_id or f"local_call_{round_idx:04d}",
                tool_name=tool_name,
                tool_args=tool_args,
                result=tool_result,
            )
            transcript.append(
                {
                    "round": round_idx,
                    "tool": tool_name,
                    "args": compact(tool_args),
                    "tool_result": compact(tool_result),
                    "post_observation": compact(next_observation.summary_dict()),
                    "provider_response_id": decision.provider_response_id,
                }
            )
            observation = next_observation

        fallback = (
            "The host reached the step limit and the forced-final pass did not return a usable final answer. "
            "Check the run log and saved evidence for the latest QuPath state."
        )
        self.run_context.logger.event(
            "agent_step_limit_exhausted",
            round_idx=observation.round_idx,
            pending_tool_result=pending_tool_result.summary_dict() if pending_tool_result else None,
            observation=observation.summary_dict(),
            steps=transcript,
        )
        try:
            force_final_decision = self.model.decide_next_step(
                observation=observation,
                pending_tool_result=pending_tool_result,
                force_final=True,
            )
        except Exception as exc:
            error = {
                "ok": False,
                "error": f"Forced final answer failed: {exc}",
            }
            self.run_context.logger.event(
                "agent_forced_final_error",
                round_idx=observation.round_idx,
                error=error,
            )
        else:
            self.run_context.logger.event(
                "agent_forced_final_decision",
                round_idx=observation.round_idx,
                decision=compact(force_final_decision.raw_response),
                parsed={
                    "kind": force_final_decision.kind,
                    "tool_name": force_final_decision.tool_name,
                    "tool_args": force_final_decision.tool_args,
                    "tool_call_id": force_final_decision.tool_call_id,
                    "final_answer": force_final_decision.final_answer,
                    "provider_response_id": force_final_decision.provider_response_id,
                },
            )
            if force_final_decision.kind == "final":
                final_answer = str(force_final_decision.final_answer or "")
                final_answer = _guard_final_answer_against_unmet_magnification(
                    task=task,
                    observation=observation,
                    final_answer=final_answer,
                )
                self.run_context.summary["final_answer"] = final_answer
                self.run_context.summary["steps"] = transcript
                self.run_context.logger.event(
                    "agent_final_answer",
                    round_idx=observation.round_idx,
                    final_answer=final_answer,
                    forced_final=True,
                )
                return AgentResult(final_answer=final_answer, steps=transcript)
            self.run_context.logger.event(
                "agent_forced_final_unexpected_tool",
                round_idx=observation.round_idx,
                tool_name=force_final_decision.tool_name,
                tool_args=force_final_decision.tool_args,
                provider_response_id=force_final_decision.provider_response_id,
            )
        self.run_context.summary["final_answer"] = fallback
        self.run_context.summary["steps"] = transcript
        self.run_context.logger.event(
            "agent_final_answer",
            round_idx=observation.round_idx,
            final_answer=fallback,
            forced_final=False,
        )
        return AgentResult(final_answer=fallback, steps=transcript)



def _is_static_model_config_error(exc: Exception) -> bool:
    text = str(exc)
    markers = (
        "invalid_function_parameters",
        "Invalid schema for function",
        "model_not_found",
        "does not exist",
        "unknown model",
        "Provider/model mismatch",
        "additional_properties",
        "Cannot find field",
    )
    return any(marker in text for marker in markers)


def _guard_final_answer_against_unmet_magnification(
    *,
    task: str,
    observation: StepObservation,
    final_answer: str,
) -> str:
    target_magnification = _extract_requested_magnification(task)
    if target_magnification is None:
        return final_answer
    current_magnification = _extract_current_magnification(observation)
    if current_magnification is None:
        return final_answer

    tolerance = max(0.25, target_magnification * 0.05)
    if math.fabs(current_magnification - target_magnification) <= tolerance:
        return final_answer

    return (
        "The run did not reach the requested viewer magnification. "
        f"Requested: {target_magnification:.3f}x. "
        f"Current: {current_magnification:.3f}x. "
        "Check the run log and restart QuPath before retrying."
    )


def _extract_requested_magnification(task: str) -> float | None:
    text = str(task or "")
    keyword_hits = (
        "zoom" in text.lower()
        or "magnification" in text.lower()
        or "放大" in text
        or "倍率" in text
        or "倍" in text
    )
    if not keyword_hits:
        return None

    patterns = (
        r"(\d+(?:\.\d+)?)\s*[xX]\b",
        r"(\d+(?:\.\d+)?)\s*倍",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _extract_current_magnification(observation: StepObservation) -> float | None:
    qupath_state = observation.qupath_state if isinstance(observation.qupath_state, dict) else {}
    state = qupath_state.get("state") or {}
    if not isinstance(state, dict):
        return None
    viewer = state.get("viewer") or {}
    if not isinstance(viewer, dict):
        return None
    value = viewer.get("magnification")
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric
