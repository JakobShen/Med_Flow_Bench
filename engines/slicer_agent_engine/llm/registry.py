from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .adapters import (
    ClaudeNativeAdapter,
    DoubaoResponsesAdapter,
    GeminiNativeAdapter,
    LocalVLLMResponsesAdapter,
    OpenAIResponsesAdapter,
    QwenResponsesAdapter,
    build_claude_builtin_tools,
    build_doubao_builtin_tools,
    build_gemini_builtin_tools,
    build_local_vllm_builtin_tools,
    build_openai_builtin_tools,
    build_qwen_builtin_tools,
)
from .base import BaseModelAdapter


@dataclass(frozen=True)
class ProviderRegistration:
    name: str
    adapter_factory: Callable[[], BaseModelAdapter]
    client_factory: Callable[..., Any]
    builtin_tool_factory: Callable[..., List[Dict[str, Any]]]


class UnsupportedProviderError(ValueError):
    pass


_PROVIDER_ALIASES = {
    "openai": "openai",
    "open-ai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "claude": "claude",
    "anthropic": "claude",
    "qwen": "qwen",
    "dashscope": "qwen",
    "model-studio": "qwen",
    "modelstudio": "qwen",
    "aliyun": "qwen",
    "alibaba": "qwen",
    "local_vllm": "local_vllm",
    "local-vllm": "local_vllm",
    "vllm": "local_vllm",
    "doubao": "doubao",
    "byteplus": "doubao",
    "modelark": "doubao",
    "ark": "doubao",
    "volcengine": "doubao",
}

_PROVIDER_MODEL_ENV_HINTS: Dict[str, Sequence[str]] = {
    "openai": ("OPENAI_MODEL",),
    "gemini": ("GEMINI_MODEL",),
    "claude": ("ANTHROPIC_MODEL", "CLAUDE_MODEL"),
    "qwen": ("QWEN_MODEL", "DASHSCOPE_MODEL"),
    "local_vllm": ("LOCAL_VLLM_MODEL",),
    "doubao": ("DOUBAO_MODEL", "ARK_MODEL"),
}

_FALLBACK_MODEL_ENV_HINTS: Sequence[str] = (
    "OPENAI_MODEL",
    "GEMINI_MODEL",
    "ANTHROPIC_MODEL",
    "CLAUDE_MODEL",
    "QWEN_MODEL",
    "DASHSCOPE_MODEL",
    "LOCAL_VLLM_MODEL",
    "DOUBAO_MODEL",
    "ARK_MODEL",
)

_QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DOUBAO_DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
_LOCAL_VLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def _iter_distinct(items: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        yield item


def _normalize_qwen_base_url(base_url: Optional[str]) -> Optional[str]:
    value = str(base_url or "").strip()
    if not value:
        return None
    value = value.rstrip("/")
    legacy_suffix = "/api/v2/apps/protocols/compatible-mode/v1"
    if value.endswith(legacy_suffix):
        return value[: -len(legacy_suffix)] + "/compatible-mode/v1"
    return value


def _openai_compatible_client_factory(
    *,
    default_api_key_env: Optional[str],
    base_url_envs: Sequence[str],
    default_base_url: Optional[str],
    base_url_normalizer: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    **kwargs: Any,
) -> Any:
    from openai import OpenAI

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and k != "model"}
    if not clean_kwargs.get("api_key") and default_api_key_env:
        api_key = os.environ.get(default_api_key_env)
        if api_key:
            clean_kwargs["api_key"] = api_key
    if not clean_kwargs.get("base_url"):
        for env_name in base_url_envs:
            candidate = os.environ.get(env_name)
            if candidate:
                clean_kwargs["base_url"] = candidate
                break
        else:
            if default_base_url:
                clean_kwargs["base_url"] = default_base_url
    if base_url_normalizer:
        normalized = base_url_normalizer(clean_kwargs.get("base_url"))
        if normalized:
            clean_kwargs["base_url"] = normalized
    return OpenAI(**clean_kwargs)


def _openai_client_factory(**kwargs: Any) -> Any:
    return _openai_compatible_client_factory(
        default_api_key_env="OPENAI_API_KEY",
        base_url_envs=("OPENAI_BASE_URL",),
        default_base_url=None,
        **kwargs,
    )


def _qwen_client_factory(**kwargs: Any) -> Any:
    return _openai_compatible_client_factory(
        default_api_key_env="DASHSCOPE_API_KEY",
        base_url_envs=("QWEN_BASE_URL", "DASHSCOPE_BASE_URL"),
        default_base_url=_QWEN_DEFAULT_BASE_URL,
        base_url_normalizer=_normalize_qwen_base_url,
        **kwargs,
    )


def _local_vllm_client_factory(**kwargs: Any) -> Any:
    if not kwargs.get("api_key"):
        kwargs["api_key"] = os.environ.get("LOCAL_VLLM_API_KEY", "EMPTY")
    return _openai_compatible_client_factory(
        default_api_key_env="LOCAL_VLLM_API_KEY",
        base_url_envs=("LOCAL_VLLM_BASE_URL",),
        default_base_url=_LOCAL_VLLM_DEFAULT_BASE_URL,
        **kwargs,
    )


def _doubao_client_factory(**kwargs: Any) -> Any:
    return _openai_compatible_client_factory(
        default_api_key_env="ARK_API_KEY",
        base_url_envs=("DOUBAO_BASE_URL", "ARK_BASE_URL"),
        default_base_url=_DOUBAO_DEFAULT_BASE_URL,
        **kwargs,
    )


def _gemini_client_factory(**kwargs: Any) -> Any:
    from google import genai  # type: ignore

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in {"model", "base_url"}}
    use_vertex = str(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() in {"1", "true", "yes", "on"}
    if use_vertex:
        clean_kwargs["vertexai"] = True
        if not clean_kwargs.get("api_key"):
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                clean_kwargs["api_key"] = api_key
        if clean_kwargs.get("api_key"):
            clean_kwargs.pop("project", None)
            clean_kwargs.pop("location", None)
        else:
            if not clean_kwargs.get("project"):
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if project:
                    clean_kwargs["project"] = project
            if not clean_kwargs.get("location"):
                location = os.environ.get("GOOGLE_CLOUD_LOCATION")
                if location:
                    clean_kwargs["location"] = location
    return genai.Client(**clean_kwargs)


def _claude_client_factory(**kwargs: Any) -> Any:
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in {"model", "base_url"}}
    use_vertex = str(
        os.environ.get("ANTHROPIC_USE_VERTEXAI")
        or os.environ.get("CLAUDE_USE_VERTEXAI")
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if use_vertex:
        from anthropic import AnthropicVertex  # type: ignore

        clean_kwargs.pop("api_key", None)
        project_id = (
            clean_kwargs.pop("project_id", None)
            or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        region = (
            clean_kwargs.pop("region", None)
            or os.environ.get("ANTHROPIC_VERTEX_REGION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or "global"
        )
        if not project_id:
            raise ValueError("Claude Vertex requires GOOGLE_CLOUD_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID")
        return AnthropicVertex(project_id=project_id, region=region, **clean_kwargs)

    from anthropic import Anthropic  # type: ignore

    return Anthropic(**clean_kwargs)


_PROVIDER_REGISTRY = {
    "openai": ProviderRegistration(
        name="openai",
        adapter_factory=OpenAIResponsesAdapter,
        client_factory=_openai_client_factory,
        builtin_tool_factory=build_openai_builtin_tools,
    ),
    "gemini": ProviderRegistration(
        name="gemini",
        adapter_factory=GeminiNativeAdapter,
        client_factory=_gemini_client_factory,
        builtin_tool_factory=build_gemini_builtin_tools,
    ),
    "claude": ProviderRegistration(
        name="claude",
        adapter_factory=ClaudeNativeAdapter,
        client_factory=_claude_client_factory,
        builtin_tool_factory=build_claude_builtin_tools,
    ),
    "qwen": ProviderRegistration(
        name="qwen",
        adapter_factory=QwenResponsesAdapter,
        client_factory=_qwen_client_factory,
        builtin_tool_factory=build_qwen_builtin_tools,
    ),
    "local_vllm": ProviderRegistration(
        name="local_vllm",
        adapter_factory=LocalVLLMResponsesAdapter,
        client_factory=_local_vllm_client_factory,
        builtin_tool_factory=build_local_vllm_builtin_tools,
    ),
    "doubao": ProviderRegistration(
        name="doubao",
        adapter_factory=DoubaoResponsesAdapter,
        client_factory=_doubao_client_factory,
        builtin_tool_factory=build_doubao_builtin_tools,
    ),
}


def registered_providers() -> List[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def default_model_from_env() -> str:
    llm_model = os.environ.get("LLM_MODEL")
    if llm_model:
        return llm_model

    provider_hint = _PROVIDER_ALIASES.get(str(os.environ.get("LLM_PROVIDER") or "").strip().lower(), "")
    env_order = list(_PROVIDER_MODEL_ENV_HINTS.get(provider_hint, ())) + list(_FALLBACK_MODEL_ENV_HINTS)
    for env_name in _iter_distinct(env_order):
        value = os.environ.get(env_name)
        if value:
            return value
    return "gpt-5-mini"


def detect_provider_from_model(model: Optional[str]) -> str:
    name = str(model or "").strip().lower()
    if not name:
        return "openai"
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "claude"
    if any(token in name for token in ("qwen", "qwq", "qvq")):
        return "qwen"
    if any(token in name for token in ("seed-2-0", "seed_2_0", "dola-seed", "doubao")):
        return "doubao"
    return "openai"


def resolve_provider_name(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    key = str(provider or "").strip().lower()
    if not key or key == "auto":
        return detect_provider_from_model(model)
    key = _PROVIDER_ALIASES.get(key, key)
    if key not in _PROVIDER_REGISTRY:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider: {provider!r}. Supported: auto, {', '.join(registered_providers())}"
        )
    return key


def _require_provider(provider: Optional[str] = None, model: Optional[str] = None) -> ProviderRegistration:
    key = resolve_provider_name(provider, model)
    try:
        return _PROVIDER_REGISTRY[key]
    except KeyError as e:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider: {provider!r}. Supported: auto, {', '.join(registered_providers())}"
        ) from e


def build_model_adapter(provider: Optional[str] = None, *, model: Optional[str] = None) -> BaseModelAdapter:
    return _require_provider(provider, model).adapter_factory()


def build_model_client(provider: Optional[str] = None, *, model: Optional[str] = None, **kwargs: Any) -> Any:
    return _require_provider(provider, model).client_factory(model=model, **kwargs)


def build_builtin_model_tools(provider: Optional[str] = None, *, model: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
    return list(_require_provider(provider, model).builtin_tool_factory(**kwargs))
