from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..artifacts import file_to_base64, preferred_png_paths
from ..base import BaseModelAdapter
from ..types import CanonicalToolSchema, ImagePrompt, ModelTurnResult, PromptInput, TextPrompt, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ClaudeConversationState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    pending_tool_results: Optional[List[Dict[str, Any]]] = None


_REASONING_BUDGET_MAP = {
    "none": None,
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 12288,
}


def _to_plain_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict(exclude_none=True)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _to_plain_dict(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _ensure_schema_no_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(schema or {"type": "object", "properties": {}, "required": []}))

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            node.setdefault("type", "object")
            node.setdefault("properties", {})
            node.setdefault("additionalProperties", False)
            for child in node.get("properties", {}).values():
                _walk(child)
        if node_type == "array" and isinstance(node.get("items"), dict):
            _walk(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(node.get(key), list):
                for child in node[key]:
                    _walk(child)

    _walk(out)
    return out


def _canonical_tool_to_claude_tool(tool: CanonicalToolSchema) -> Dict[str, Any]:
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "input_schema": _ensure_schema_no_additional_properties(dict(tool.get("parameters") or {"type": "object", "properties": {}, "required": []})),
    }


def build_claude_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    del enable_code_interpreter
    return []


class ClaudeNativeAdapter(BaseModelAdapter):
    provider_name = "claude"

    def start_conversation(self, *, inputs: Sequence[PromptInput]) -> ClaudeConversationState:
        state = ClaudeConversationState()
        for item in inputs:
            state.messages.append({"role": "user", "content": self._serialize_prompt(item)})
        return state

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: ClaudeConversationState,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        del timeout_sec, retry_on_timeout  # SDK/client transport policy is configured at client creation time.
        state.pending_tool_results = None

        tools = [_canonical_tool_to_claude_tool(tool) for tool in tool_schemas]
        for tool in extra_model_tools or []:
            if isinstance(tool, dict) and tool.get("name") and tool.get("input_schema"):
                tools.append(dict(tool))

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": list(state.messages),
            "tools": tools,
        }
        if instructions:
            create_kwargs["system"] = instructions
        budget = _REASONING_BUDGET_MAP.get(str(reasoning_effort).lower()) if reasoning_effort else None
        if budget:
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        response = client.messages.create(**create_kwargs)

        assistant_content = [_to_plain_dict(block) for block in (getattr(response, "content", None) or [])]
        state.messages.append({"role": "assistant", "content": assistant_content})

        output_text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        output_item_types: List[str] = []
        for block in assistant_content:
            block_type = str(block.get("type") or "")
            output_item_types.append(block_type or "block")
            if block_type == "text":
                text = str(block.get("text") or "")
                if text.strip():
                    output_text_parts.append(text)
                continue
            if block_type != "tool_use":
                continue
            tool_calls.append(
                ToolCall(
                    call_id=str(block.get("id") or f"claude_call_{len(tool_calls)+1}"),
                    name=str(block.get("name") or ""),
                    arguments_json=json.dumps(block.get("input") or {}, ensure_ascii=False),
                )
            )

        return ModelTurnResult(
            output_text="\n".join(part for part in output_text_parts if part).strip(),
            tool_calls=tool_calls,
            response_id=getattr(response, "id", None),
            output_item_types=output_item_types,
            raw_response=response,
        )

    def append_tool_result(
        self,
        *,
        state: ClaudeConversationState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        pngs = preferred_png_paths(result, max_images=max_images) if attach_images else []
        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(_to_plain_dict(result), ensure_ascii=False)}]
        for path in pngs:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": file_to_base64(path),
                    },
                }
            )

        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": call.call_id,
            "content": content_blocks,
            "is_error": bool(isinstance(result, dict) and (result.get("ok") is False)),
        }

        if state.pending_tool_results is None:
            state.pending_tool_results = [tool_result_block]
            state.messages.append({"role": "user", "content": state.pending_tool_results})
        else:
            state.pending_tool_results.append(tool_result_block)
        return len(pngs)

    def export_state(self, state: ClaudeConversationState) -> List[Any]:
        return list(state.messages)

    def _serialize_prompt(self, item: PromptInput) -> List[Dict[str, Any]]:
        if isinstance(item, TextPrompt):
            return [{"type": "text", "text": item.text}]
        if isinstance(item, ImagePrompt):
            path = item.resolved_path()
            blocks: List[Dict[str, Any]] = []
            if item.text:
                blocks.append({"type": "text", "text": item.text})
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.mime,
                        "data": file_to_base64(path),
                    },
                }
            )
            return blocks
        raise TypeError(f"Unsupported prompt input: {type(item)!r}")
