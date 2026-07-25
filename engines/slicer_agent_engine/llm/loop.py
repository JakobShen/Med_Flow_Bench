from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .base import BaseModelAdapter
from .types import CanonicalToolSchema

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, str], Dict[str, Any]]
ToolSchemaSource = Union[Sequence[CanonicalToolSchema], Callable[[], Sequence[CanonicalToolSchema]]]
ExtraModelToolsSource = Union[Sequence[Dict[str, Any]], Callable[[], Sequence[Dict[str, Any]]]]


def _force_final_instruction(
    base_instructions: str,
    *,
    max_rounds: int,
    max_tools: Optional[int] = None,
    tools_used: Optional[int] = None,
) -> str:
    if max_tools is not None:
        used = max_tools if tools_used is None else int(tools_used)
        suffix = (
            f"\n\nTool budget exhausted after {used}/{int(max_tools)} executed tool calls. "
            "Do not call any more tools. You must now provide your best final answer using only the evidence already in context."
        )
    else:
        suffix = (
            f"\n\nTool budget exhausted after {max_rounds} tool-calling rounds. "
            "Do not call any more tools. You must now provide your best final answer using only the evidence already in context."
        )
    if base_instructions:
        return f"{base_instructions.rstrip()}{suffix}"
    return suffix.strip()


def run_model_tool_loop(
    *,
    adapter: BaseModelAdapter,
    client: Any,
    model: str,
    tool_schemas: ToolSchemaSource,
    initial_inputs: Sequence[Any],
    instructions: str,
    tool_executor: ToolExecutor,
    max_rounds: int = 12,
    max_tools: Optional[int] = None,
    attach_images: bool = True,
    max_images_per_tool: int = 4,
    session: Optional[Any] = None,
    extra_model_tools: Optional[ExtraModelToolsSource] = None,
    reasoning_effort: Optional[str] = None,
    model_request_timeout_sec: Optional[float] = None,
    model_request_retry_on_timeout: int = 0,
) -> Tuple[str, List[Any]]:
    """Provider-neutral host-side tool loop.

    The host owns:
    - tool visibility,
    - tool execution,
    - artifact attachment policy,
    - tracing.

    The provider adapter owns only request/response translation.
    """

    max_rounds = max(0, int(max_rounds))
    max_tools_limit: Optional[int] = None
    if max_tools is not None:
        try:
            candidate_max_tools = int(max_tools)
        except (TypeError, ValueError):
            candidate_max_tools = 0
        if candidate_max_tools > 0:
            max_tools_limit = candidate_max_tools
    tools_used = 0

    def _current_tool_schemas() -> List[CanonicalToolSchema]:
        if callable(tool_schemas):
            return list(tool_schemas())
        return list(tool_schemas)

    def _current_extra_tools() -> List[Dict[str, Any]]:
        if extra_model_tools is None:
            return []
        if callable(extra_model_tools):
            return list(extra_model_tools())
        return list(extra_model_tools)

    def _log_run_event(event_type: str, **payload: Any) -> None:
        if session is not None and hasattr(session, "log_run_event"):
            try:
                session.log_run_event(event_type, **payload)
            except Exception:
                logger.exception("Failed logging run event: %s", event_type)

    def _compact(value: Any) -> Any:
        if session is not None and hasattr(session, "_compact"):
            try:
                return session._compact(value, max_str=500, max_list=15)  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(value, dict):
            return {k: _compact(v) for k, v in list(value.items())[:20]}
        if isinstance(value, list):
            return [_compact(v) for v in value[:15]]
        if isinstance(value, str) and len(value) > 500:
            return value[:500] + f" ... <{len(value) - 500} more chars>"
        return value

    state = adapter.start_conversation(inputs=initial_inputs)
    provider_tag = str(getattr(adapter, "provider_name", "model")).upper()

    def _request_with_timing(
        stage: str,
        *,
        round_idx: int,
        tool_schemas_override: Optional[Sequence[CanonicalToolSchema]] = None,
        extra_tools_override: Optional[Sequence[Dict[str, Any]]] = None,
        instructions_override: Optional[str] = None,
    ):
        visible_tool_schemas = list(tool_schemas_override) if tool_schemas_override is not None else _current_tool_schemas()
        visible_extra_tools = list(extra_tools_override) if extra_tools_override is not None else _current_extra_tools()
        effective_instructions = instructions if instructions_override is None else instructions_override
        visible_tools = adapter.tool_display_names(
            tool_schemas=visible_tool_schemas,
            extra_model_tools=visible_extra_tools,
        )
        _log_run_event(
            "model_request_start",
            stage=stage,
            round_idx=round_idx,
            model=model,
            provider=getattr(adapter, "provider_name", None),
            visible_tools=visible_tools,
            visible_tool_count=len(visible_tools),
            input_item_count=len(adapter.export_state(state)),
        )
        logger.info("[%s] waiting start: %s", provider_tag, stage)
        t0 = time.monotonic()
        try:
            response = adapter.request_turn(
                client=client,
                model=model,
                instructions=effective_instructions,
                state=state,
                tool_schemas=visible_tool_schemas,
                extra_model_tools=visible_extra_tools,
                reasoning_effort=reasoning_effort,
                timeout_sec=model_request_timeout_sec,
                retry_on_timeout=model_request_retry_on_timeout,
            )
        except Exception:
            dt = time.monotonic() - t0
            _log_run_event(
                "model_request_failed",
                stage=stage,
                round_idx=round_idx,
                model=model,
                provider=getattr(adapter, "provider_name", None),
                duration_ms=round(dt * 1000.0, 1),
            )
            logger.exception("[%s] waiting failed after %.2fs: %s", provider_tag, dt, stage)
            raise

        dt = time.monotonic() - t0
        _log_run_event(
            "model_request_done",
            stage=stage,
            round_idx=round_idx,
            model=model,
            provider=getattr(adapter, "provider_name", None),
            duration_ms=round(dt * 1000.0, 1),
            response_id=response.response_id,
            output_text_preview=_compact(response.output_text),
            function_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
            output_item_types=list(response.output_item_types),
        )
        logger.info("[%s] waiting done in %.2fs: %s", provider_tag, dt, stage)
        return response

    response = _request_with_timing("initial", round_idx=0)

    round_idx = 0
    budget_exhausted = False
    while True:
        if not response.tool_calls:
            break

        if round_idx >= max_rounds:
            budget_exhausted = True
            _log_run_event(
                "tool_budget_exhausted",
                model=model,
                provider=getattr(adapter, "provider_name", None),
                max_rounds=max_rounds,
                max_tools=max_tools_limit,
                tools_used=tools_used,
                pending_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
            )
            budget_error = {
                "ok": False,
                "error": f"Tool budget exhausted after {max_rounds} tool-calling rounds. No further tool calls are allowed.",
                "final_answer_required": True,
            }
            for call in response.tool_calls:
                adapter.append_tool_result(
                    state=state,
                    call=call,
                    result={**budget_error, "tool": call.name},
                    attach_images=False,
                    max_images=0,
                )

            response = _request_with_timing(
                "forced_final",
                round_idx=round_idx + 1,
                tool_schemas_override=[],
                extra_tools_override=[],
                instructions_override=_force_final_instruction(instructions, max_rounds=max_rounds),
            )
            if response.tool_calls:
                _log_run_event(
                    "forced_final_returned_tool_calls",
                    model=model,
                    provider=getattr(adapter, "provider_name", None),
                    returned_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
                )
            break

        tool_calls_to_execute = list(response.tool_calls)
        if max_tools_limit is not None:
            remaining_tools = max_tools_limit - tools_used
            if remaining_tools <= 0:
                budget_exhausted = True
                _log_run_event(
                    "tool_budget_exhausted",
                    model=model,
                    provider=getattr(adapter, "provider_name", None),
                    max_rounds=max_rounds,
                    max_tools=max_tools_limit,
                    tools_used=tools_used,
                    pending_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
                )
                if hasattr(adapter, "prune_unexecuted_tool_calls"):
                    adapter.prune_unexecuted_tool_calls(state=state, keep_call_ids=[])
                response = _request_with_timing(
                    "forced_final",
                    round_idx=round_idx + 1,
                    tool_schemas_override=[],
                    extra_tools_override=[],
                    instructions_override=_force_final_instruction(
                        instructions,
                        max_rounds=max_rounds,
                        max_tools=max_tools_limit,
                        tools_used=tools_used,
                    ),
                )
                if response.tool_calls:
                    _log_run_event(
                        "forced_final_returned_tool_calls",
                        model=model,
                        provider=getattr(adapter, "provider_name", None),
                        returned_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
                    )
                break
            if len(tool_calls_to_execute) > remaining_tools:
                skipped_tool_calls = tool_calls_to_execute[remaining_tools:]
                tool_calls_to_execute = tool_calls_to_execute[:remaining_tools]
                _log_run_event(
                    "tool_calls_skipped_max_tools",
                    model=model,
                    provider=getattr(adapter, "provider_name", None),
                    max_tools=max_tools_limit,
                    tools_used=tools_used,
                    tools_remaining=remaining_tools,
                    executed_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in tool_calls_to_execute],
                    skipped_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in skipped_tool_calls],
                )
                if hasattr(adapter, "prune_unexecuted_tool_calls"):
                    adapter.prune_unexecuted_tool_calls(
                        state=state,
                        keep_call_ids=[tc.call_id for tc in tool_calls_to_execute],
                    )

        round_idx += 1
        for call in tool_calls_to_execute:
            try:
                parsed_args = json.loads(call.arguments_json or "{}")
            except Exception:
                parsed_args = call.arguments_json

            _log_run_event(
                "tool_call_start",
                round_idx=round_idx,
                call_id=call.call_id,
                tool_name=call.name,
                arguments=_compact(parsed_args),
            )
            tool_t0 = time.monotonic()
            try:
                result = tool_executor(call.name, call.arguments_json)
            except Exception as e:
                logger.exception("Tool executor crashed: %s", call.name)
                result = {"ok": False, "tool": call.name, "error": str(e)}
            tool_dt = time.monotonic() - tool_t0

            tools_used += 1
            if max_tools_limit is not None:
                budget_payload = {
                    "max_tools": max_tools_limit,
                    "tools_used": tools_used,
                    "tools_remaining": max(max_tools_limit - tools_used, 0),
                }
                if isinstance(result, dict):
                    result = {**result, "tool_budget": budget_payload}
                else:
                    result = {"ok": True, "value": result, "tool_budget": budget_payload}

            if session is not None and hasattr(session, "summarize_tool_result"):
                try:
                    summary = session.summarize_tool_result(result)
                except Exception:
                    summary = _compact(result)
            else:
                summary = _compact(result)

            _log_run_event(
                "tool_call_end",
                round_idx=round_idx,
                call_id=call.call_id,
                tool_name=call.name,
                duration_ms=round(tool_dt * 1000.0, 1),
                ok=(bool(result.get("ok", True)) if isinstance(result, dict) else True),
                result_summary=summary,
                max_tools=max_tools_limit,
                tools_used=tools_used,
                tools_remaining=(max(max_tools_limit - tools_used, 0) if max_tools_limit is not None else None),
            )

            attached_image_count = 0
            try:
                attached_image_count = int(
                    adapter.append_tool_result(
                        state=state,
                        call=call,
                        result=result,
                        attach_images=attach_images,
                        max_images=max_images_per_tool,
                    )
                    or 0
                )
            except Exception:
                logger.exception("Failed appending tool result for %s", call.name)
                raise

            if attached_image_count:
                _log_run_event(
                    "tool_artifacts_attached",
                    round_idx=round_idx,
                    call_id=call.call_id,
                    tool_name=call.name,
                    image_count=attached_image_count,
                )

        if max_tools_limit is not None and tools_used >= max_tools_limit:
            budget_exhausted = True
            _log_run_event(
                "tool_budget_exhausted",
                model=model,
                provider=getattr(adapter, "provider_name", None),
                max_rounds=max_rounds,
                max_tools=max_tools_limit,
                tools_used=tools_used,
                pending_calls=[],
            )
            response = _request_with_timing(
                "forced_final",
                round_idx=round_idx + 1,
                tool_schemas_override=[],
                extra_tools_override=[],
                instructions_override=_force_final_instruction(
                    instructions,
                    max_rounds=max_rounds,
                    max_tools=max_tools_limit,
                    tools_used=tools_used,
                ),
            )
            if response.tool_calls:
                _log_run_event(
                    "forced_final_returned_tool_calls",
                    model=model,
                    provider=getattr(adapter, "provider_name", None),
                    returned_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
                )
            break

        response = _request_with_timing(f"round_{round_idx}", round_idx=round_idx)

    final_text = str(response.output_text or "").strip()
    if budget_exhausted and not final_text:
        final_text = "Tool budget exhausted before the model produced a final answer."
        _log_run_event(
            "forced_final_missing_text",
            model=model,
            provider=getattr(adapter, "provider_name", None),
            max_rounds=max_rounds,
            max_tools=max_tools_limit,
            tools_used=tools_used,
        )

    _log_run_event(
        "model_loop_finished",
        model=model,
        provider=getattr(adapter, "provider_name", None),
        rounds_executed=round_idx,
        budget_exhausted=budget_exhausted,
        max_tools=max_tools_limit,
        tools_used=tools_used,
        final_text_preview=_compact(final_text),
    )

    return final_text, adapter.export_state(state)
