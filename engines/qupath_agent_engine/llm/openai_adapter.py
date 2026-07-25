from __future__ import annotations

import json
import os
from typing import Any, Sequence, TYPE_CHECKING

from qupath_agent_engine.agent.prompts import render_observation_text
from qupath_agent_engine.core.types import ModelDecision, PendingToolResult, StepObservation

from .base import BaseHostModel, encode_image_as_data_url, safe_serialize

if TYPE_CHECKING:  # pragma: no cover
    from qupath_agent_engine.agent.tooling import ToolSpec


class OpenAIHostModel(BaseHostModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL") or None
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        self.client: Any | None = None
        self.previous_response_id: str | None = None
        self.openai_tools: list[dict[str, Any]] = []

    def start_session(self, *, system_instruction: str, tools: Sequence["ToolSpec"]) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.system_instruction = system_instruction
        self.openai_tools = [tool.to_openai_tool() for tool in tools]
        self.previous_response_id = None
        self.started = True

    def decide_next_step(
        self,
        *,
        observation: StepObservation,
        pending_tool_result: PendingToolResult | None,
        force_final: bool = False,
    ) -> ModelDecision:
        if not self.started or self.client is None:
            raise RuntimeError("OpenAI host model session has not been started.")

        message_text = render_observation_text(
            observation=observation,
            pending_tool_result=pending_tool_result,
            is_first_turn=self.previous_response_id is None,
            force_final=force_final,
        )

        input_items: list[dict[str, Any]] = []
        if pending_tool_result is not None:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": pending_tool_result.call_id,
                    "output": json.dumps(pending_tool_result.result, ensure_ascii=False),
                }
            )

        content_parts: list[dict[str, Any]] = [{"type": "input_text", "text": message_text}]
        image_data_url = encode_image_as_data_url(observation.viewer_image_path)
        if image_data_url:
            content_parts.append({"type": "input_image", "image_url": image_data_url})

        input_items.append({"role": "user", "content": content_parts})

        params: dict[str, Any] = {
            "model": self.config.model,
            "instructions": self.system_instruction,
            "input": input_items,
            "max_output_tokens": self.config.max_tokens,
        }
        if not force_final:
            params["tools"] = self.openai_tools
            params["parallel_tool_calls"] = False
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            params["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.previous_response_id:
            params["previous_response_id"] = self.previous_response_id

        response = self.client.responses.create(**params)
        raw_response = safe_serialize(response)
        self.previous_response_id = self._read_attr(response, "id")

        function_calls = self._extract_function_calls(response)
        if function_calls:
            first_call = function_calls[0]
            return ModelDecision(
                kind="tool",
                raw_response=raw_response,
                tool_name=first_call["name"],
                tool_args=first_call["args"],
                tool_call_id=first_call["call_id"],
                assistant_text=self._extract_output_text(response),
                provider_response_id=self.previous_response_id,
            )

        final_text = (self._extract_output_text(response) or self._extract_message_text(response) or "").strip()
        return ModelDecision(
            kind="final",
            raw_response=raw_response,
            final_answer=final_text,
            assistant_text=final_text,
            provider_response_id=self.previous_response_id,
        )

    def _extract_function_calls(self, response: Any) -> list[dict[str, Any]]:
        output = self._read_attr(response, "output") or []
        results: list[dict[str, Any]] = []
        for item in output:
            item_type = self._read_attr(item, "type")
            if item_type != "function_call":
                continue
            name = self._read_attr(item, "name")
            call_id = self._read_attr(item, "call_id")
            arguments_raw = self._read_attr(item, "arguments") or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except Exception:
                arguments = {"_raw": str(arguments_raw)}
            if isinstance(name, str) and isinstance(call_id, str):
                results.append({"name": name, "call_id": call_id, "args": arguments})
        return results

    def _extract_output_text(self, response: Any) -> str:
        text = self._read_attr(response, "output_text")
        if isinstance(text, str):
            return text
        return ""

    def _extract_message_text(self, response: Any) -> str:
        output = self._read_attr(response, "output") or []
        text_parts: list[str] = []
        for item in output:
            if self._read_attr(item, "type") != "message":
                continue
            content = self._read_attr(item, "content") or []
            for part in content:
                part_text = self._read_attr(part, "text")
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text)
        return "\n".join(text_parts).strip()

    @staticmethod
    def _read_attr(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)
