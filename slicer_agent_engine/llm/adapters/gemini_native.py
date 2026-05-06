from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..artifacts import file_to_base64, preferred_png_paths
from ..base import BaseModelAdapter
from ..types import CanonicalToolSchema, ImagePrompt, ModelTurnResult, PromptInput, TextPrompt, ToolCall

logger = logging.getLogger(__name__)


def _looks_like_quota_retryable_error(exc: Exception) -> bool:
    text = " ".join(
        str(part)
        for part in (
            exc,
            getattr(exc, "message", None),
            getattr(exc, "status", None),
            getattr(exc, "code", None),
            getattr(exc, "status_code", None),
        )
        if part is not None
    ).lower()
    return any(
        marker in text
        for marker in (
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
            "status': 'resource_exhausted'",
            "status\": \"resource_exhausted\"",
            " 429",
            "code: 429",
            "code 429",
        )
    )


def _call_with_quota_backoff(fn: Any, *, op_name: str, max_attempts: int = 10, base_delay_s: float = 1.0) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_attempts or not _looks_like_quota_retryable_error(exc):
                raise
            delay_s = base_delay_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            logger.warning(
                "[GEMINI] retrying quota-limited request op=%s attempt=%d/%d sleep=%.2fs error=%s",
                op_name,
                attempt,
                max_attempts,
                delay_s,
                exc,
            )
            time.sleep(delay_s)


@dataclass
class GeminiConversationState:
    contents: List[Any] = field(default_factory=list)
    pending_user_parts: Optional[List[Any]] = None


_REASONING_LEVEL_MAP = {
    "none": "minimal",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

_ARTIFACT_PATH_KEYS = {
    "png_path",
    "png_paths",
    "contact_sheet_png_path",
    "gif_path",
    "video_path",
    "mp4_path",
    "webm_path",
}

_ARTIFACT_FILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm")


def _to_plain_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
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


def _get_attr(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _looks_like_artifact_string(value: str) -> bool:
    lower = value.strip().lower()
    if not lower:
        return False
    return lower.endswith(_ARTIFACT_FILE_SUFFIXES) or any(sep in lower for sep in ('/', '\\')) and any(lower.endswith(sfx) for sfx in _ARTIFACT_FILE_SUFFIXES)


def _strip_unsupported_gemini_parameter_keywords(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a canonical JSON schema for Gemini FunctionDeclaration.parameters.

    The Gemini FunctionDeclaration `parameters` field uses the Schema object shape shown in
    the function-calling guide examples. The API reference exposes a separate
    `parametersJsonSchema` field for full JSON Schema values. In practice, sending
    `additionalProperties` inside `parameters` causes 400 INVALID_ARGUMENT errors on the
    Gemini native endpoint, so we drop JSON-Schema-only keywords here and keep a conservative,
    official-example-compatible object schema.
    """

    out = json.loads(json.dumps(schema or {"type": "object", "properties": {}, "required": []}))

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return

        for unsupported_key in ("additionalProperties", "propertyOrdering", "unevaluatedProperties", "$schema"):
            node.pop(unsupported_key, None)

        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            node.setdefault("type", "object")
            node.setdefault("properties", {})
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


def _canonical_tool_to_gemini_function(tool: CanonicalToolSchema) -> Dict[str, Any]:
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": _strip_unsupported_gemini_parameter_keywords(
            dict(tool.get("parameters") or {"type": "object", "properties": {}, "required": []})
        ),
    }


def build_gemini_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if enable_code_interpreter:
        tools.append({"type": "code_execution", "code_execution": {}})
    return tools


class GeminiNativeAdapter(BaseModelAdapter):
    provider_name = "gemini"

    def start_conversation(self, *, inputs: Sequence[PromptInput]) -> GeminiConversationState:
        state = GeminiConversationState()
        for item in inputs:
            state.contents.append(self._serialize_prompt(item))
        return state

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: GeminiConversationState,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        del timeout_sec, retry_on_timeout  # google-genai handles its own transport config; keep host loop simple.
        state.pending_user_parts = None

        function_declarations = [_canonical_tool_to_gemini_function(tool) for tool in tool_schemas]
        tools_payload: List[Dict[str, Any]] = []
        has_server_side_builtin_tools = False
        if function_declarations:
            tools_payload.append({"function_declarations": function_declarations})
        for tool in extra_model_tools or []:
            if not isinstance(tool, dict):
                continue
            payload = dict(tool)
            payload.pop("type", None)
            if payload:
                tools_payload.append(payload)
                has_server_side_builtin_tools = True

        config: Dict[str, Any] = {"automatic_function_calling": {"disable": True}}
        if tools_payload:
            config["tools"] = tools_payload
        if function_declarations and has_server_side_builtin_tools:
            config["tool_config"] = {"include_server_side_tool_invocations": True}
        if reasoning_effort:
            config["thinking_config"] = {"thinking_level": _REASONING_LEVEL_MAP.get(str(reasoning_effort).lower(), "medium")}
        if instructions:
            config["system_instruction"] = instructions

        response = _call_with_quota_backoff(
            lambda: client.models.generate_content(
                model=model,
                contents=list(state.contents),
                config=config or None,
            ),
            op_name=f"generate_content:{model}",
        )

        assistant_content = self._extract_assistant_content(response)
        if assistant_content is not None:
            state.contents.append(assistant_content)

        output_text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for idx, part in enumerate(self._iter_parts(assistant_content), start=1):
            text = _get_attr(part, "text")
            if isinstance(text, str) and text.strip():
                output_text_parts.append(text)
                continue
            function_call = _get_attr(part, "function_call", "functionCall")
            if function_call is None:
                continue
            name = str(_get_attr(function_call, "name") or "")
            args = _to_plain_dict(_get_attr(function_call, "args", default={}) or {})
            call_id = _get_attr(function_call, "id", "call_id", "callId")
            if not call_id:
                call_id = f"gemini_call_{len(tool_calls)+1}_{idx}"
            tool_calls.append(
                ToolCall(
                    call_id=str(call_id),
                    name=name,
                    arguments_json=json.dumps(args, ensure_ascii=False),
                )
            )

        if not tool_calls:
            for idx, function_call in enumerate(_get_attr(response, "function_calls", default=[]) or [], start=1):
                name = str(_get_attr(function_call, "name") or "")
                args = _to_plain_dict(_get_attr(function_call, "args", default={}) or {})
                call_id = _get_attr(function_call, "id", "call_id", "callId") or f"gemini_call_{idx}"
                tool_calls.append(
                    ToolCall(
                        call_id=str(call_id),
                        name=name,
                        arguments_json=json.dumps(args, ensure_ascii=False),
                    )
                )

        return ModelTurnResult(
            output_text="\n".join(part for part in output_text_parts if part).strip(),
            tool_calls=tool_calls,
            response_id=getattr(response, "id", None),
            output_item_types=self._output_item_types(assistant_content),
            raw_response=response,
        )

    def append_tool_result(
        self,
        *,
        state: GeminiConversationState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        part, image_count = self._build_function_response_part(call=call, result=result, attach_images=attach_images, max_images=max_images)
        if state.pending_user_parts is None:
            state.pending_user_parts = [part]
            state.contents.append({"role": "user", "parts": state.pending_user_parts})
        else:
            state.pending_user_parts.append(part)
        return image_count

    def export_state(self, state: GeminiConversationState) -> List[Any]:
        return list(state.contents)

    def _serialize_prompt(self, item: PromptInput) -> Dict[str, Any]:
        if isinstance(item, TextPrompt):
            return {"role": "user", "parts": [{"text": item.text}]}
        if isinstance(item, ImagePrompt):
            path = item.resolved_path()
            image_part = {
                "inlineData": {
                    "mimeType": item.mime,
                    "data": file_to_base64(path),
                }
            }
            parts: List[Dict[str, Any]] = [image_part]
            if item.text:
                parts.append({"text": item.text})
            return {"role": "user", "parts": parts}
        raise TypeError(f"Unsupported prompt input: {type(item)!r}")

    def _extract_assistant_content(self, response: Any) -> Any:
        candidates = _get_attr(response, "candidates", default=[]) or []
        if not candidates:
            return None
        first = candidates[0]
        return _get_attr(first, "content")

    def _iter_parts(self, assistant_content: Any) -> List[Any]:
        if assistant_content is None:
            return []
        parts = _get_attr(assistant_content, "parts", default=[])
        if isinstance(parts, list):
            return parts
        return []

    def _output_item_types(self, assistant_content: Any) -> List[str]:
        out: List[str] = []
        for part in self._iter_parts(assistant_content):
            if _get_attr(part, "function_call", "functionCall") is not None:
                out.append("function_call")
            elif _get_attr(part, "text") is not None:
                out.append("text")
            else:
                out.append("part")
        return out

    def _sanitize_tool_result_for_response(self, value: Any) -> Any:
        value = _to_plain_dict(value)
        if isinstance(value, str):
            return "[artifact omitted]" if _looks_like_artifact_string(value) else value
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                key = str(k)
                key_l = key.lower()
                if key in _ARTIFACT_PATH_KEYS or key_l in _ARTIFACT_PATH_KEYS:
                    continue
                if key_l.endswith("_png_path") or key_l.endswith("_png_paths"):
                    continue
                if key_l.endswith("_gif_path") or key_l.endswith("_mp4_path") or key_l.endswith("_video_path"):
                    continue
                out[key] = self._sanitize_tool_result_for_response(v)
            return out
        if isinstance(value, list):
            return [self._sanitize_tool_result_for_response(v) for v in value]
        return value

    def _build_function_response_part(
        self,
        *,
        call: ToolCall,
        result: Any,
        attach_images: bool,
        max_images: int,
    ) -> Tuple[Any, int]:
        pngs = preferred_png_paths(result, max_images=max_images) if attach_images else []
        structured_result = self._sanitize_tool_result_for_response(result)

        # Gemini 3 supports multimodal function responses by nesting inlineData parts inside
        # the functionResponse block. Referencing those parts from the structured response via
        # {"$ref": "<displayName>"} is optional according to the official guide. In practice,
        # the google-genai SDK path has produced strict validation failures in the field for
        # otherwise well-formed responses. To keep the loop stable, we always emit the raw REST
        # shape and attach images as multimodal parts only, without $ref indirection.
        response_payload: Dict[str, Any] = {"result": structured_result}
        image_paths: List[Path] = [Path(path) for path in pngs]

        function_response: Dict[str, Any] = {
            "name": call.name,
            "id": str(call.call_id),
            "response": response_payload,
        }
        if image_paths:
            function_response["parts"] = [
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": file_to_base64(path),
                    }
                }
                for path in image_paths
            ]
            response_payload["attached_image_count"] = len(image_paths)

        return {"functionResponse": function_response}, len(image_paths)
