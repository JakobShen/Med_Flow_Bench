from __future__ import annotations

import os
import time
from typing import Any, Sequence, TYPE_CHECKING

from qupath_agent_engine.agent.prompts import render_observation_text
from qupath_agent_engine.core.types import ModelDecision, PendingToolResult, StepObservation

from .base import BaseHostModel, read_image_bytes, safe_serialize

if TYPE_CHECKING:  # pragma: no cover
    from qupath_agent_engine.agent.tooling import ToolSpec


class GeminiHostModel(BaseHostModel):
    def __init__(self, config):
        super().__init__(config)
        self.gemini_api_key = self._first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.vertex_api_key = self._first_env("VERTEX_API_KEY", "VERTEXAI_API_KEY")
        self.vertex_project = self._first_env("WORKSTATION_AGENT_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT")
        self.vertex_location = self._first_env("WORKSTATION_AGENT_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION")
        self.backend = self._resolve_backend()
        self.model_name = self._normalize_model_name(config.model, backend=self.backend)
        self.request_timeout_ms = self._int_env("WORKSTATION_AGENT_GEMINI_TIMEOUT_MS", default=60000)
        self.request_retries = max(1, self._int_env("WORKSTATION_AGENT_GEMINI_REQUEST_RETRIES", default=2))
        if self.backend == "gemini_api" and not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for provider=gemini when using the Gemini Developer API. "
                "Set WORKSTATION_AGENT_GEMINI_BACKEND=vertexai or export VERTEX_API_KEY to use Vertex AI instead."
            )
        self.client: Any | None = None
        self.types: Any | None = None
        self.generate_config: Any | None = None
        self.generate_config_no_tools: Any | None = None
        self.history: list[Any] = []
        self.pending_model_content: Any | None = None

    def start_session(self, *, system_instruction: str, tools: Sequence["ToolSpec"]) -> None:
        from google import genai
        from google.genai import types

        client_kwargs: dict[str, Any] = {}
        if self.request_timeout_ms > 0:
            client_kwargs["http_options"] = types.HttpOptions(timeout=self.request_timeout_ms)
        if self.backend == "vertexai":
            client_kwargs["vertexai"] = True
            if self.vertex_api_key:
                client_kwargs["api_key"] = self.vertex_api_key
            else:
                if self.vertex_project:
                    client_kwargs["project"] = self.vertex_project
                if self.vertex_location:
                    client_kwargs["location"] = self.vertex_location
        else:
            client_kwargs["api_key"] = self.gemini_api_key

        try:
            self.client = genai.Client(**client_kwargs)
        except Exception as exc:
            if self.backend == "vertexai":
                raise RuntimeError(
                    "Failed to initialize Gemini on Vertex AI. "
                    "For express mode, export VERTEX_API_KEY and use genai.Client(vertexai=True, api_key=...). "
                    "For ADC-based auth, provide GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION or log in with gcloud. "
                    f"Original error: {exc}"
                ) from exc
            raise RuntimeError(
                "Failed to initialize the Gemini Developer API client. "
                "Export GEMINI_API_KEY or GOOGLE_API_KEY. "
                f"Original error: {exc}"
            ) from exc
        self.types = types
        self.system_instruction = system_instruction

        function_declarations = [
            types.FunctionDeclaration(
                **tool.to_gemini_declaration(),
            )
            for tool in tools
        ]
        tool_block = types.Tool(function_declarations=function_declarations)
        self.generate_config = types.GenerateContentConfig(
            tools=[tool_block],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=self.system_instruction,
            temperature=self.config.temperature,
        )
        self.generate_config_no_tools = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=self.system_instruction,
            temperature=self.config.temperature,
        )
        self.history = []
        self.pending_model_content = None
        self.started = True

    def decide_next_step(
        self,
        *,
        observation: StepObservation,
        pending_tool_result: PendingToolResult | None,
        force_final: bool = False,
    ) -> ModelDecision:
        if (
            not self.started
            or self.client is None
            or self.types is None
            or self.generate_config is None
            or self.generate_config_no_tools is None
        ):
            raise RuntimeError("Gemini host model session has not been started.")

        message_text = render_observation_text(
            observation=observation,
            pending_tool_result=pending_tool_result,
            is_first_turn=not self.history and self.pending_model_content is None,
            force_final=force_final,
        )

        if pending_tool_result is not None:
            if self.pending_model_content is not None:
                self.history.append(self.pending_model_content)
                self.pending_model_content = None
            self.history.append(
                self.types.Content(
                    role="tool",
                    parts=[self._build_function_response_part(pending_tool_result)],
                )
            )

        observation_content = self._build_observation_content(message_text, observation.viewer_image_path)
        self.history.append(observation_content)

        response = self._generate_content_with_retries(
            contents=self.history,
            config=self.generate_config_no_tools if force_final else self.generate_config,
        )

        assistant_content = self._extract_model_content(response)
        raw_response = safe_serialize(response)
        provider_response_id = self._read_attr(response, "response_id") or self._read_attr(response, "id")
        assistant_text = self._extract_text(assistant_content)

        function_calls = self._extract_function_calls(response, assistant_content)
        if function_calls:
            function_call = function_calls[0]
            self.pending_model_content = assistant_content
            return ModelDecision(
                kind="tool",
                raw_response=raw_response,
                tool_name=self._read_attr(function_call, "name"),
                tool_args=self._read_attr(function_call, "args") or {},
                tool_call_id=self._read_attr(function_call, "call_id") or self._read_attr(function_call, "id"),
                assistant_text=assistant_text,
                provider_response_id=str(provider_response_id) if provider_response_id is not None else None,
            )

        self.pending_model_content = None
        final_text = assistant_text.strip()
        return ModelDecision(
            kind="final",
            raw_response=raw_response,
            final_answer=final_text,
            assistant_text=final_text,
            provider_response_id=str(provider_response_id) if provider_response_id is not None else None,
        )

    def _build_observation_content(self, text: str, image_path) -> Any:
        parts: list[Any] = [self.types.Part(text=text)]
        image_bytes, mime_type = read_image_bytes(image_path)
        if image_bytes is not None and mime_type is not None:
            parts.append(self.types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        return self.types.Content(role="user", parts=parts)

    def _build_function_response_part(self, pending_tool_result: PendingToolResult) -> Any:
        kwargs = {
            "name": pending_tool_result.tool_name,
            "response": pending_tool_result.result,
        }
        try:
            return self.types.Part.from_function_response(
                id=pending_tool_result.call_id,
                **kwargs,
            )
        except TypeError:
            return self.types.Part.from_function_response(**kwargs)

    def _extract_model_content(self, response: Any) -> Any:
        candidates = self._read_attr(response, "candidates") or []
        if candidates:
            first_candidate = candidates[0]
            content = self._read_attr(first_candidate, "content")
            if content is not None:
                return content
        return None

    def _extract_function_calls(self, response: Any, assistant_content: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, part in enumerate(self._iter_parts(assistant_content), start=1):
            function_call = self._read_attr(part, "function_call") or self._read_attr(part, "functionCall")
            if function_call is None:
                continue
            call_id = self._read_attr(function_call, "id") or self._read_attr(function_call, "call_id")
            if not call_id:
                call_id = f"gemini_call_{len(results) + 1}_{index}"
            results.append(
                {
                    "name": self._read_attr(function_call, "name"),
                    "args": self._to_plain_value(self._read_attr(function_call, "args") or {}),
                    "call_id": call_id,
                }
            )
        if results:
            return results

        function_calls = list(self._read_attr(response, "function_calls") or [])
        for index, function_call in enumerate(function_calls, start=1):
            results.append(
                {
                    "name": self._read_attr(function_call, "name"),
                    "args": self._to_plain_value(self._read_attr(function_call, "args") or {}),
                    "call_id": self._read_attr(function_call, "id")
                    or self._read_attr(function_call, "call_id")
                    or f"gemini_call_{index}",
                }
            )
        return results

    def _extract_text(self, assistant_content: Any) -> str:
        parts: list[str] = []
        for part in self._iter_parts(assistant_content):
            part_text = self._read_attr(part, "text")
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text)
        return "\n".join(parts)

    def _iter_parts(self, content: Any) -> list[Any]:
        if content is None:
            return []
        parts = self._read_attr(content, "parts") or []
        if isinstance(parts, list):
            return parts
        return list(parts)

    def _to_plain_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_plain_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_plain_value(item) for item in value]
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return self._to_plain_value(value.model_dump(exclude_none=True))
            except Exception:
                return repr(value)
        if hasattr(value, "dict") and callable(value.dict):
            try:
                return self._to_plain_value(value.dict(exclude_none=True))
            except Exception:
                return repr(value)
        if hasattr(value, "items") and callable(value.items):
            try:
                return {str(key): self._to_plain_value(item) for key, item in value.items()}
            except Exception:
                return repr(value)
        if hasattr(value, "__dict__"):
            try:
                return {
                    str(key): self._to_plain_value(item)
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            except Exception:
                return repr(value)
        return repr(value)

    @staticmethod
    def _read_attr(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _first_env(*names: str) -> str | None:
        for name in names:
            value = os.environ.get(name)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return None

    @staticmethod
    def _env_truthy(name: str) -> bool:
        value = str(os.environ.get(name, "")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _int_env(name: str, *, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return int(default)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return int(default)

    def _resolve_backend(self) -> str:
        explicit_backend = self._first_env("WORKSTATION_AGENT_GEMINI_BACKEND")
        if explicit_backend:
            normalized = explicit_backend.lower().replace("-", "_")
            if normalized in {"vertex", "vertexai", "vertex_ai"}:
                return "vertexai"
            if normalized in {"gemini", "gemini_api", "developer", "developer_api"}:
                return "gemini_api"
            raise RuntimeError(
                "Unsupported WORKSTATION_AGENT_GEMINI_BACKEND value. "
                "Use one of: vertexai, gemini_api."
            )

        if self.vertex_api_key:
            return "vertexai"
        if self._env_truthy("GOOGLE_GENAI_USE_VERTEXAI"):
            return "vertexai"
        if self.vertex_project or self.vertex_location:
            return "vertexai"
        return "gemini_api"

    @staticmethod
    def _normalize_model_name(model_name: str, *, backend: str) -> str:
        normalized = str(model_name or "").strip()
        if backend != "vertexai":
            return normalized

        aliases = {
            "gemini-3.1-flash-preview": "gemini-3-flash-preview",
            "models/gemini-3.1-flash-preview": "gemini-3-flash-preview",
            "gemini-3.1-pro-preview": "gemini-3-pro-preview",
            "models/gemini-3.1-pro-preview": "gemini-3-pro-preview",
        }
        return aliases.get(normalized, normalized)

    def _generate_content_with_retries(self, *, contents: list[Any], config: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.request_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.request_retries or not self._is_retryable_generation_error(exc):
                    raise
                time.sleep(min(2** (attempt - 1), 4))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini generation failed without returning a response or an error.")

    @staticmethod
    def _is_retryable_generation_error(exc: Exception) -> bool:
        status_code = getattr(exc, "code", None)
        if isinstance(status_code, int):
            return status_code in {408, 429, 500, 502, 503, 504}
        message = str(exc).strip().lower()
        transient_fragments = (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "connection refused",
            "remote protocol error",
            "server disconnected",
            "read error",
            "eof occurred in violation of protocol",
        )
        return any(fragment in message for fragment in transient_fragments)
