from __future__ import annotations

import ast
import json
import logging
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..artifacts import file_to_data_url, gather_png_paths
from ..base import BaseModelAdapter
from ..types import CanonicalToolSchema, ImagePrompt, ModelTurnResult, PromptInput, TextPrompt, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConversationState:
    input_items: List[Any] = field(default_factory=list)
    manual_tool_protocol: bool = False
    manual_tool_call_counter: int = 0
    text_tool_protocol: bool = False


class _ModelRequestTimedOut(Exception):
    pass


def _canonical_tool_to_responses_function(tool: CanonicalToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": dict(tool.get("parameters") or {"type": "object", "properties": {}, "required": []}),
    }


# Backward-compatible alias retained for tests/imports that still use the old helper name.
_canonical_tool_to_openai_function = _canonical_tool_to_responses_function


def build_openai_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if enable_code_interpreter:
        tools.append({"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "4g"}})
    return tools


def build_qwen_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if enable_code_interpreter:
        tools.append({"type": "code_interpreter"})
    return tools


def build_local_vllm_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    del enable_code_interpreter
    return []


def build_doubao_builtin_tools(*, enable_code_interpreter: bool = False) -> List[Dict[str, Any]]:
    del enable_code_interpreter
    return []


def _item_type(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        raw = item.get("type")
    else:
        raw = getattr(item, "type", None)
    return str(raw) if raw is not None else None


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain_data(model_dump(mode="python"))
        except TypeError:
            return _to_plain_data(model_dump())
    if hasattr(value, "__dict__"):
        return {str(k): _to_plain_data(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return value


def _sanitize_code_interpreter_item_for_replay(item: Any) -> Any:
    if _item_type(item) != "code_interpreter_call":
        return item
    payload = _to_plain_data(item)
    if not isinstance(payload, dict):
        return {"type": "code_interpreter_call"}
    replay_item: Dict[str, Any] = {"type": "code_interpreter_call", "outputs": None}
    for key in ("id", "container_id", "status"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            replay_item[key] = value
    code_value = payload.get("code")
    if isinstance(code_value, str) and code_value and len(code_value) <= 100_000:
        replay_item["code"] = code_value
    return replay_item


def _extract_response_failure_message(resp: Any) -> Optional[str]:
    status = str(getattr(resp, "status", "") or "").strip().lower()
    if status != "failed":
        return None
    error = getattr(resp, "error", None)
    if error is None:
        return "response status=failed"
    code = str(getattr(error, "code", "") or "").strip()
    message = str(getattr(error, "message", "") or "").strip()
    if code and message:
        return f"{code}: {message}"
    if message:
        return message
    if code:
        return code
    return "response status=failed"


_MANUAL_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_CODE_FENCE_RE = re.compile(r"```(?:tool_code|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _tool_schemas_by_name(tool_schemas: Sequence[CanonicalToolSchema]) -> Dict[str, CanonicalToolSchema]:
    out: Dict[str, CanonicalToolSchema] = {}
    for tool in tool_schemas:
        if isinstance(tool, dict) and tool.get("name"):
            out[str(tool.get("name"))] = tool
    return out


def _ast_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_ast_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {_ast_literal(k): _ast_literal(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"Unsupported AST literal: {type(node)!r}")


def _tool_call_from_python_code(
    text: str,
    *,
    tool_schemas: Sequence[CanonicalToolSchema],
    call_id_prefix: str,
) -> List[ToolCall]:
    schema_by_name = _tool_schemas_by_name(tool_schemas)
    if not schema_by_name:
        return []
    snippets = [m.group(1).strip() for m in _TOOL_CODE_FENCE_RE.finditer(text or "") if str(m.group(1) or "").strip()]
    if not snippets and text:
        snippets = [str(text)]
    calls: List[ToolCall] = []
    for snippet in snippets:
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            continue
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "print" and len(expr.args) == 1:
                inner = expr.args[0]
                if isinstance(inner, ast.Call):
                    expr = inner
            if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
                continue
            name = str(expr.func.id)
            schema = schema_by_name.get(name)
            if schema is None:
                continue
            arguments: Dict[str, Any] = {}
            if expr.args:
                param_names = list(((schema.get("parameters") or {}).get("properties") or {}).keys())
                if len(expr.args) > len(param_names):
                    continue
                try:
                    for idx, arg in enumerate(expr.args):
                        arguments[param_names[idx]] = _ast_literal(arg)
                except Exception:
                    continue
            try:
                for kw in expr.keywords:
                    if kw.arg is None:
                        raise ValueError("Unsupported **kwargs tool call")
                    arguments[str(kw.arg)] = _ast_literal(kw.value)
            except Exception:
                continue
            calls.append(
                ToolCall(
                    call_id=f"{call_id_prefix}_{len(calls) + 1}",
                    name=name,
                    arguments_json=json.dumps(arguments, ensure_ascii=False),
                )
            )
            return calls
    return calls


def _tool_call_from_manual_tag(
    text: str,
    *,
    tool_schemas: Sequence[CanonicalToolSchema],
    call_id_prefix: str,
) -> List[ToolCall]:
    allowed_names = set(_tool_schemas_by_name(tool_schemas).keys())
    calls: List[ToolCall] = []
    for match in _MANUAL_TOOL_TAG_RE.finditer(str(text or "")):
        payload_text = str(match.group(1) or "").strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except Exception:
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name or (allowed_names and name not in allowed_names):
                continue
            arguments = entry.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(
                    call_id=f"{call_id_prefix}_{len(calls) + 1}",
                    name=name,
                    arguments_json=json.dumps(arguments, ensure_ascii=False),
                )
            )
    return calls


def _fallback_tool_calls_from_text(
    text: str,
    *,
    tool_schemas: Sequence[CanonicalToolSchema],
    call_id_prefix: str = "text_tool_call",
) -> List[ToolCall]:
    calls = _tool_call_from_manual_tag(text, tool_schemas=tool_schemas, call_id_prefix=call_id_prefix)
    if calls:
        return calls
    return _tool_call_from_python_code(text, tool_schemas=tool_schemas, call_id_prefix=call_id_prefix)


class ResponsesCompatibleAdapter(BaseModelAdapter):
    """Shared adapter for providers that expose an OpenAI-compatible Responses API."""

    provider_name = "responses"
    request_text_type = "input_text"
    request_image_type = "input_image"
    text_prompt_as_plain_string = False
    include_code_interpreter_outputs = False

    def start_conversation(self, *, inputs: Sequence[PromptInput]) -> OpenAIConversationState:
        state = OpenAIConversationState()
        for item in inputs:
            state.input_items.append(self._serialize_prompt(item))
        return state

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: OpenAIConversationState,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        tools = [_canonical_tool_to_responses_function(tool) for tool in tool_schemas]
        if extra_model_tools:
            tools.extend(list(extra_model_tools))

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "tools": tools,
            "input": self._prepare_input_items(state.input_items),
            "instructions": instructions,
        }
        self._apply_reasoning_hints(create_kwargs, reasoning_effort)
        if self._should_request_code_interpreter_outputs(extra_model_tools):
            create_kwargs["include"] = ["code_interpreter_call.outputs"]

        resp = self._responses_create_with_optional_timeout(
            client=client,
            create_kwargs=create_kwargs,
            timeout_sec=timeout_sec,
            retry_on_timeout=retry_on_timeout,
        )
        failure_message = _extract_response_failure_message(resp)
        if failure_message:
            raise RuntimeError(f"{self.provider_name} responses.create failed: {failure_message}")
        output_items = [_sanitize_code_interpreter_item_for_replay(item) for item in getattr(resp, "output", [])]
        structured_tool_calls = [
            ToolCall(
                call_id=str(getattr(item, "call_id", "")),
                name=str(getattr(item, "name", "")),
                arguments_json=str(getattr(item, "arguments", "") or "{}"),
            )
            for item in getattr(resp, "output", [])
            if getattr(item, "type", None) == "function_call"
        ]
        output_text = str(getattr(resp, "output_text", "") or "")
        recovered_text_tool_calls: List[ToolCall] = []
        if not structured_tool_calls and tool_schemas:
            recovered_text_tool_calls = _fallback_tool_calls_from_text(output_text, tool_schemas=tool_schemas)
            if recovered_text_tool_calls:
                logger.warning(
                    "[%s] recovered %d tool call(s) from assistant text fallback",
                    str(self.provider_name).upper(),
                    len(recovered_text_tool_calls),
                )
        tool_calls = structured_tool_calls or recovered_text_tool_calls
        text_tool_protocol = bool(getattr(state, "text_tool_protocol", False))
        if recovered_text_tool_calls:
            state.text_tool_protocol = True
            text_tool_protocol = True

        if text_tool_protocol:
            assistant_text = output_text.strip()
            if not assistant_text and structured_tool_calls:
                assistant_text = "\n".join(
                    f"Calling tool: {call.name}" for call in structured_tool_calls
                )
            if assistant_text:
                state.input_items.append({"role": "assistant", "content": assistant_text})
        else:
            state.input_items += output_items
        return ModelTurnResult(
            output_text=output_text,
            tool_calls=tool_calls,
            response_id=getattr(resp, "id", None),
            output_item_types=[getattr(item, "type", None) for item in getattr(resp, "output", [])],
            raw_response=resp,
        )

    def append_tool_result(
        self,
        *,
        state: OpenAIConversationState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        if bool(getattr(state, "text_tool_protocol", False)):
            return self._append_tool_result_manual_protocol(
                state=state,
                call=call,
                result=result,
                attach_images=attach_images,
                max_images=max_images,
            )
        state.input_items.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
        if not attach_images:
            return 0

        pngs = [p for p in gather_png_paths(result) if p.exists()]
        if not pngs:
            return 0

        contact = None
        if isinstance(result, dict):
            cs = result.get("contact_sheet_png_path")
            if isinstance(cs, str) and cs:
                p = Path(cs)
                if p.exists():
                    contact = p
        if contact is not None:
            pngs = [contact]
        else:
            pngs = pngs[:max_images]

        for p in pngs:
            state.input_items.append(
                self._serialize_user_image_message(
                    text=f"Rendered image from tool {call.name}: {p.name}",
                    image_url=file_to_data_url(p, "image/png"),
                )
            )
        return len(pngs)

    def _append_tool_result_manual_protocol(
        self,
        *,
        state: OpenAIConversationState,
        call: ToolCall,
        result: Any,
        attach_images: bool,
        max_images: int,
    ) -> int:
        state.input_items.append(self._serialize_user_text_message(self._manual_tool_result_prompt(call=call, result=result)))
        if not attach_images:
            return 0

        pngs = [p for p in gather_png_paths(result) if p.exists()]
        if not pngs:
            return 0

        contact = None
        if isinstance(result, dict):
            cs = result.get("contact_sheet_png_path")
            if isinstance(cs, str) and cs:
                p = Path(cs)
                if p.exists():
                    contact = p
        if contact is not None:
            pngs = [contact]
        else:
            pngs = pngs[:max_images]

        for p in pngs:
            state.input_items.append(
                self._serialize_user_image_message(
                    text=f"Rendered image from tool {call.name}: {p.name}",
                    image_url=file_to_data_url(p, "image/png"),
                )
            )
        return len(pngs)

    def export_state(self, state: OpenAIConversationState) -> List[Any]:
        return list(state.input_items)

    def _prepare_input_items(self, items: Sequence[Any]) -> List[Any]:
        merged: List[Any] = []
        pending_user: Optional[Dict[str, Any]] = None
        for item in items:
            if not isinstance(item, dict) or str(item.get("role") or "") != "user":
                if pending_user is not None:
                    merged.append(pending_user)
                    pending_user = None
                merged.append(item)
                continue
            content = item.get("content")
            if isinstance(content, str):
                content_items: List[Any] = [{"type": self.request_text_type, "text": content}]
            elif isinstance(content, list):
                content_items = list(content)
            else:
                if pending_user is not None:
                    merged.append(pending_user)
                    pending_user = None
                merged.append(item)
                continue
            if pending_user is None:
                pending_user = {"role": "user", "content": []}
            pending_user["content"].extend(content_items)
        if pending_user is not None:
            merged.append(pending_user)
        return merged

    def _serialize_prompt(self, item: PromptInput) -> Dict[str, Any]:
        if isinstance(item, TextPrompt):
            return self._serialize_user_text_message(item.text)
        if isinstance(item, ImagePrompt):
            path = item.resolved_path()
            return self._serialize_user_image_message(
                text=item.text,
                image_url=file_to_data_url(path, item.mime),
            )
        raise TypeError(f"Unsupported prompt input: {type(item)!r}")

    def _serialize_user_text_message(self, text: str) -> Dict[str, Any]:
        if self.text_prompt_as_plain_string:
            return {"role": "user", "content": text}
        return {"role": "user", "content": [self._text_content_item(text)]}

    def _serialize_user_image_message(self, *, text: str, image_url: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                self._text_content_item(text),
                self._image_content_item(image_url),
            ],
        }

    def _text_content_item(self, text: str) -> Dict[str, Any]:
        return {"type": self.request_text_type, "text": text}

    def _image_content_item(self, image_url: str) -> Dict[str, Any]:
        item = {"type": self.request_image_type, "image_url": image_url}
        if self.request_image_type == "input_image":
            item["detail"] = "auto"
        return item

    def _apply_reasoning_hints(self, create_kwargs: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
        del create_kwargs, reasoning_effort

    def _should_request_code_interpreter_outputs(self, extra_model_tools: Optional[Sequence[Dict[str, Any]]]) -> bool:
        if not self.include_code_interpreter_outputs:
            return False
        return any(isinstance(tool, dict) and tool.get("type") == "code_interpreter" for tool in extra_model_tools or [])

    def _set_extra_body_fields(self, create_kwargs: Dict[str, Any], extra_fields: Dict[str, Any]) -> None:
        if not extra_fields:
            return
        current = dict(create_kwargs.get("extra_body") or {})
        current.update(extra_fields)
        create_kwargs["extra_body"] = current

    def _responses_create_with_optional_timeout(
        self,
        *,
        client: Any,
        create_kwargs: Dict[str, Any],
        timeout_sec: Optional[float],
        retry_on_timeout: int,
    ) -> Any:
        timeout_value = float(timeout_sec or 0.0)
        attempts = max(1, int(retry_on_timeout or 0) + 1)
        last_timeout: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if timeout_value <= 0:
                    return client.responses.create(**create_kwargs)
                return self._responses_create_with_timeout(client=client, create_kwargs=create_kwargs, timeout_sec=timeout_value)
            except _ModelRequestTimedOut as e:
                last_timeout = e
                logger.warning(
                    "[%s] responses.create timed out after %.2fs (attempt %d/%d)",
                    str(self.provider_name).upper(),
                    timeout_value,
                    attempt,
                    attempts,
                )
                if attempt >= attempts:
                    raise TimeoutError(str(e))
        if last_timeout is not None:
            raise TimeoutError(str(last_timeout))
        raise RuntimeError("responses.create retry loop exited unexpectedly")

    def _responses_create_with_timeout(self, *, client: Any, create_kwargs: Dict[str, Any], timeout_sec: float) -> Any:
        q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                resp = client.responses.create(**create_kwargs)
            except Exception as e:
                q.put(("error", e))
                return
            q.put(("ok", resp))

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout_sec)
        if th.is_alive():
            raise _ModelRequestTimedOut(f"responses.create exceeded {timeout_sec:.1f}s timeout")
        try:
            status, payload = q.get_nowait()
        except queue.Empty:
            raise RuntimeError("responses.create worker returned no result")
        if status == "error":
            raise payload
        return payload

    def _manual_tool_result_prompt(self, *, call: ToolCall, result: Any) -> str:
        rendered = json.dumps(result, ensure_ascii=False)
        return (
            f"Tool {call.name} returned the following JSON result.\n"
            f"{rendered}\n"
            "Continue from this result. If another tool is needed, call it. Otherwise provide the final answer normally."
        )


class OpenAIResponsesAdapter(ResponsesCompatibleAdapter):
    provider_name = "openai"
    request_text_type = "input_text"
    request_image_type = "input_image"
    include_code_interpreter_outputs = True

    def _apply_reasoning_hints(self, create_kwargs: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
        if reasoning_effort:
            create_kwargs["reasoning"] = {"effort": str(reasoning_effort)}


class ManualToolResponsesAdapter(ResponsesCompatibleAdapter):
    """Responses adapter using a model-neutral text tool-call protocol."""

    provider_name = "responses_manual"
    manual_protocol_label = "Model"
    request_text_type = "input_text"
    request_image_type = "input_image"
    text_prompt_as_plain_string = True

    _MANUAL_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: OpenAIConversationState,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        if not self._should_use_manual_tool_protocol(state):
            return super().request_turn(
                client=client,
                model=model,
                instructions=instructions,
                state=state,
                tool_schemas=tool_schemas,
                extra_model_tools=extra_model_tools,
                reasoning_effort=reasoning_effort,
                timeout_sec=timeout_sec,
                retry_on_timeout=retry_on_timeout,
            )
        return self._request_turn_manual_tool_protocol(
            client=client,
            model=model,
            instructions=instructions,
            state=state,
            tool_schemas=tool_schemas,
            extra_model_tools=extra_model_tools,
            reasoning_effort=reasoning_effort,
            timeout_sec=timeout_sec,
            retry_on_timeout=retry_on_timeout,
        )

    def append_tool_result(
        self,
        *,
        state: OpenAIConversationState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        if not bool(getattr(state, "manual_tool_protocol", False)):
            return super().append_tool_result(
                state=state,
                call=call,
                result=result,
                attach_images=attach_images,
                max_images=max_images,
            )

        state.input_items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": self._manual_tool_result_prompt(call=call, result=result),
                    }
                ],
            }
        )
        if not attach_images:
            return 0

        pngs = [p for p in gather_png_paths(result) if p.exists()]
        if not pngs:
            return 0

        contact = None
        if isinstance(result, dict):
            cs = result.get("contact_sheet_png_path")
            if isinstance(cs, str) and cs:
                p = Path(cs)
                if p.exists():
                    contact = p
        if contact is not None:
            pngs = [contact]
        else:
            pngs = pngs[:max_images]

        for p in pngs:
            state.input_items.append(
                self._serialize_user_image_message(
                    text=f"Rendered image from tool {call.name}: {p.name}",
                    image_url=file_to_data_url(p, "image/png"),
                )
            )
        return len(pngs)

    def _request_turn_manual_tool_protocol(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: OpenAIConversationState,
        tool_schemas: Sequence[CanonicalToolSchema],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        unsupported_extra_tools = list(extra_model_tools or [])
        if unsupported_extra_tools:
            logger.warning(
                "Ignoring unsupported %s Responses manual tools: %s",
                self.manual_protocol_label,
                unsupported_extra_tools,
            )

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self._prepare_input_items(state.input_items),
            "instructions": self._manual_tool_protocol_instructions(
                instructions=instructions,
                tool_schemas=tool_schemas,
            ),
        }
        self._apply_reasoning_hints(create_kwargs, reasoning_effort)
        resp = self._responses_create_with_optional_timeout(
            client=client,
            create_kwargs=create_kwargs,
            timeout_sec=timeout_sec,
            retry_on_timeout=retry_on_timeout,
        )
        failure_message = _extract_response_failure_message(resp)
        if failure_message:
            raise RuntimeError(f"{self.provider_name} responses.create failed: {failure_message}")

        output_text = str(getattr(resp, "output_text", "") or "")
        tool_calls = self._parse_manual_tool_calls(state=state, output_text=output_text, tool_schemas=tool_schemas)
        replay_text = self._tool_calls_for_replay(tool_calls) if tool_calls else output_text
        state.input_items.append({"role": "assistant", "content": replay_text})
        output_item_types = ["message"]
        if tool_calls:
            output_item_types.append("manual_tool_call")
        return ModelTurnResult(
            output_text=output_text,
            tool_calls=tool_calls,
            response_id=getattr(resp, "id", None),
            output_item_types=output_item_types,
            raw_response=resp,
        )

    def _tool_calls_for_replay(self, tool_calls: Sequence[ToolCall]) -> str:
        blocks = []
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments_json or "{}")
            except Exception:
                arguments = {}
            payload = {"name": call.name, "arguments": arguments if isinstance(arguments, dict) else {}}
            blocks.append(
                "<tool_call>"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "</tool_call>"
            )
        return "\n".join(blocks)

    def _should_use_manual_tool_protocol(self, state: OpenAIConversationState) -> bool:
        if bool(getattr(state, "manual_tool_protocol", False)):
            return True
        for item in getattr(state, "input_items", []):
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "")
                if part_type in {"input_image", "image_url"}:
                    state.manual_tool_protocol = True
                    return True
        return False

    def _manual_tool_protocol_instructions(
        self,
        *,
        instructions: str,
        tool_schemas: Sequence[CanonicalToolSchema],
    ) -> str:
        base = str(instructions or "").strip()
        lines: List[str] = []
        if base:
            lines.append(base)
        lines.append(f"{self.manual_protocol_label} manual tool protocol:")
        if tool_schemas:
            lines.append("If you need a tool, reply with exactly one XML block and no other text:")
            lines.append('<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>')
            tool_call_limit = self._manual_tool_call_limit()
            if tool_call_limit == 1:
                lines.append("Call only one tool per turn. Wait for its result before choosing the next tool.")
            else:
                lines.append("If multiple tools are needed in the same round, emit a JSON array inside the same tag.")
            lines.append("When a tool result is provided later, continue reasoning from it. If no more tools are needed, give the final answer normally.")
            lines.append("Available tools:")
            for tool in tool_schemas:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name") or "")
                description = str(tool.get("description") or "")
                parameters = json.dumps(tool.get("parameters") or {"type": "object"}, ensure_ascii=False, sort_keys=True)
                lines.append(f"- {name}: {description} | parameters={parameters}")
        else:
            lines.append("No tools are available in this turn. Do not emit <tool_call>.")
        return "\n".join(line for line in lines if line)

    def _manual_tool_call_limit(self) -> Optional[int]:
        return None

    def _parse_manual_tool_calls(
        self,
        *,
        state: OpenAIConversationState,
        output_text: str,
        tool_schemas: Sequence[CanonicalToolSchema],
    ) -> List[ToolCall]:
        allowed_names = {
            str(tool.get("name") or "")
            for tool in tool_schemas
            if isinstance(tool, dict) and tool.get("name")
        }
        calls: List[ToolCall] = []
        for match in self._MANUAL_TOOL_TAG_RE.finditer(str(output_text or "")):
            payload_text = str(match.group(1) or "").strip()
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except Exception:
                logger.warning(
                    "Failed parsing %s manual tool payload: %s",
                    self.manual_protocol_label,
                    payload_text,
                )
                continue
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name or (allowed_names and name not in allowed_names):
                    continue
                arguments = entry.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                state.manual_tool_call_counter += 1
                calls.append(
                    ToolCall(
                        call_id=f"manual_call_{state.manual_tool_call_counter}",
                        name=name,
                        arguments_json=json.dumps(arguments, ensure_ascii=False),
                    )
                )
                tool_call_limit = self._manual_tool_call_limit()
                if tool_call_limit is not None and len(calls) >= tool_call_limit:
                    return calls
        return calls

    def _manual_tool_result_prompt(self, *, call: ToolCall, result: Any) -> str:
        rendered = json.dumps(result, ensure_ascii=False)
        return (
            f"Tool {call.name} returned the following JSON result.\n"
            f"{rendered}\n"
            "Continue from this result. If another tool is needed, emit <tool_call>...</tool_call> exactly. "
            "Otherwise provide the final answer normally."
        )


class QwenResponsesAdapter(ManualToolResponsesAdapter):
    provider_name = "qwen"
    manual_protocol_label = "Qwen"

    def _apply_reasoning_hints(self, create_kwargs: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
        if reasoning_effort is None:
            return
        self._set_extra_body_fields(
            create_kwargs,
            {"enable_thinking": str(reasoning_effort).strip().lower() != "none"},
        )


class LocalVLLMResponsesAdapter(ManualToolResponsesAdapter):
    """OpenAI-compatible vLLM adapter with model-neutral text tool calling."""

    provider_name = "local_vllm"
    manual_protocol_label = "Local vLLM"

    def _manual_tool_call_limit(self) -> Optional[int]:
        raw_limit = str(os.environ.get("LOCAL_VLLM_MAX_TOOL_CALLS_PER_TURN", "1")).strip()
        try:
            tool_call_limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                "LOCAL_VLLM_MAX_TOOL_CALLS_PER_TURN must be a positive integer"
            ) from exc
        if tool_call_limit <= 0:
            raise ValueError("LOCAL_VLLM_MAX_TOOL_CALLS_PER_TURN must be a positive integer")
        return tool_call_limit

    @staticmethod
    def _positive_env_int(name: str, default: str) -> int:
        raw_value = str(os.environ.get(name, default)).strip()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _trim_manual_tool_history(self, items: Sequence[Any]) -> List[Any]:
        history = list(items)
        max_rounds = self._positive_env_int("LOCAL_VLLM_MAX_HISTORY_TOOL_ROUNDS", "3")
        assistant_positions = [
            index
            for index, item in enumerate(history)
            if isinstance(item, dict) and str(item.get("role") or "") == "assistant"
        ]
        if len(assistant_positions) <= max_rounds:
            return history

        first_assistant = assistant_positions[0]
        keep_from = assistant_positions[-max_rounds]
        omitted_rounds = len(assistant_positions) - max_rounds
        bounded = history[:first_assistant]
        bounded.append(
            self._serialize_user_text_message(
                f"{omitted_rounds} earlier tool round(s) were omitted to stay within the local model context. "
                "The Slicer scene still contains their effects. Continue from the latest tool results and images."
            )
        )
        bounded.extend(history[keep_from:])
        logger.info(
            "Trimmed %d old local vLLM tool round(s); retaining the latest %d",
            omitted_rounds,
            max_rounds,
        )
        return bounded

    def _prepare_input_items(self, items: Sequence[Any]) -> List[Any]:
        prepared = super()._prepare_input_items(self._trim_manual_tool_history(items))
        image_limit = self._positive_env_int(
            "LOCAL_VLLM_MAX_INPUT_IMAGES",
            os.environ.get("VLLM_IMAGE_LIMIT", "16"),
        )

        image_positions: List[tuple[int, int]] = []
        for item_index, item in enumerate(prepared):
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_index, part in enumerate(content):
                if isinstance(part, dict) and str(part.get("type") or "") in {
                    "input_image",
                    "image_url",
                }:
                    image_positions.append((item_index, content_index))
        excess = len(image_positions) - image_limit
        if excess <= 0:
            return prepared

        removals: Dict[int, set[int]] = {}
        for item_index, content_index in image_positions[:excess]:
            removals.setdefault(item_index, set()).add(content_index)
        for item_index, content_indexes in removals.items():
            content = prepared[item_index].get("content")
            prepared[item_index]["content"] = [
                part for index, part in enumerate(content) if index not in content_indexes
            ]
        logger.info(
            "Trimmed %d old image(s) from local vLLM context to enforce the %d-image limit",
            excess,
            image_limit,
        )
        return prepared

    def _manual_tool_result_prompt(self, *, call: ToolCall, result: Any) -> str:
        max_chars = self._positive_env_int("LOCAL_VLLM_MAX_TOOL_RESULT_CHARS", "8000")
        rendered = json.dumps(result, ensure_ascii=False)
        if len(rendered) > max_chars:
            omitted_chars = len(rendered) - max_chars
            rendered = (
                rendered[:max_chars]
                + f"\n[Tool result truncated by {omitted_chars} characters to fit the local model context.]"
            )
        return (
            f"Tool {call.name} returned the following JSON result.\n"
            f"{rendered}\n"
            "Continue from this result. If another tool is needed, emit <tool_call>...</tool_call> exactly. "
            "Otherwise provide the final answer normally."
        )

    def _apply_reasoning_hints(self, create_kwargs: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
        del reasoning_effort
        raw_limit = str(os.environ.get("LOCAL_VLLM_MAX_OUTPUT_TOKENS", "2048")).strip()
        try:
            max_output_tokens = int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                f"LOCAL_VLLM_MAX_OUTPUT_TOKENS must be a positive integer, got {raw_limit!r}"
            ) from exc
        if max_output_tokens <= 0:
            raise ValueError("LOCAL_VLLM_MAX_OUTPUT_TOKENS must be a positive integer")
        create_kwargs["max_output_tokens"] = max_output_tokens


class DoubaoResponsesAdapter(ResponsesCompatibleAdapter):
    provider_name = "doubao"
    request_text_type = "input_text"
    request_image_type = "input_image"

    def _apply_reasoning_hints(self, create_kwargs: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
        if reasoning_effort is None:
            return
        thinking_type = "disabled" if str(reasoning_effort).strip().lower() == "none" else "enabled"
        self._set_extra_body_fields(create_kwargs, {"thinking": {"type": thinking_type}})
