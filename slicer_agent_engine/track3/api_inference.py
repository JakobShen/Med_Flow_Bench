from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from slicer_agent_engine.llm.registry import build_model_adapter, build_model_client, resolve_provider_name
from slicer_agent_engine.llm.types import make_image_prompt, make_text_prompt

from .artifacts import Track3Artifact


def run_multimodal_prompt(
    *,
    provider: str,
    model: str,
    instructions: str,
    user_message: str,
    artifacts: Sequence[Track3Artifact],
    reasoning_effort: Optional[str] = None,
    timeout_sec: float = 300.0,
    retry_on_timeout: int = 1,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    resolved_provider = resolve_provider_name(provider, model)
    adapter = build_model_adapter(resolved_provider, model=model)
    client = build_model_client(resolved_provider, model=model)

    inputs = [make_text_prompt(user_message)]
    for artifact in artifacts:
        inputs.append(make_image_prompt(Path(artifact.path), text=str(artifact.description or ""), mime="image/png"))

    if session is not None and hasattr(session, "log_run_event"):
        session.log_run_event(
            "track3_model_request_start",
            provider=resolved_provider,
            model=model,
            image_count=len(artifacts),
            image_paths=[str(artifact.path) for artifact in artifacts],
            reasoning_effort=reasoning_effort,
            timeout_sec=float(timeout_sec),
        )

    state = adapter.start_conversation(inputs=inputs)
    response = adapter.request_turn(
        client=client,
        model=model,
        instructions=instructions,
        state=state,
        tool_schemas=[],
        extra_model_tools=None,
        reasoning_effort=reasoning_effort,
        timeout_sec=timeout_sec,
        retry_on_timeout=retry_on_timeout,
    )

    if session is not None and hasattr(session, "log_run_event"):
        session.log_run_event(
            "track3_model_request_done",
            provider=resolved_provider,
            model=model,
            response_id=response.response_id,
            output_text_preview=str(response.output_text or "")[:1200],
            output_item_types=list(response.output_item_types or []),
            unexpected_tool_calls=[{"call_id": tc.call_id, "name": tc.name} for tc in response.tool_calls],
        )

    return {
        "provider": resolved_provider,
        "model": model,
        "text": str(response.output_text or ""),
        "response_id": response.response_id,
        "output_item_types": list(response.output_item_types or []),
        "unexpected_tool_calls": [
            {"call_id": tc.call_id, "name": tc.name, "arguments_json": tc.arguments_json}
            for tc in response.tool_calls
        ],
    }


__all__ = ["run_multimodal_prompt"]
