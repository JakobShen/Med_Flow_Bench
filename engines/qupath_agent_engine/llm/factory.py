from __future__ import annotations

import os

from .base import BaseHostModel, ModelConfig
from .gemini_adapter import GeminiHostModel
from .openai_adapter import OpenAIHostModel


DEFAULT_PROVIDER = os.environ.get("WORKSTATION_AGENT_PROVIDER", "openai")
DEFAULT_MODEL = os.environ.get("WORKSTATION_AGENT_MODEL") or None
OPENAI_DEFAULT_MODEL = os.environ.get("WORKSTATION_AGENT_OPENAI_MODEL", "gpt-5.4-mini")
GEMINI_DEFAULT_MODEL = os.environ.get("WORKSTATION_AGENT_GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_REASONING_EFFORT = os.environ.get("WORKSTATION_AGENT_REASONING_EFFORT") or None
DEFAULT_MAX_TOKENS = int(os.environ.get("WORKSTATION_AGENT_MAX_TOKENS", "1600"))
DEFAULT_TEMPERATURE = float(os.environ.get("WORKSTATION_AGENT_TEMPERATURE", "0.0"))
OPENAI_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
    "computer-use-preview",
)
GEMINI_MODEL_PREFIXES = (
    "gemini-",
    "models/gemini-",
)


def build_model(*, provider: str | None = None, model: str | None = None) -> BaseHostModel:
    if model is not None and str(model).strip():
        model_name = str(model).strip()
    elif DEFAULT_MODEL:
        model_name = DEFAULT_MODEL.strip()
    else:
        model_name = ""

    inferred_provider = _infer_provider_from_model_name(model_name) if model_name else None
    explicit_provider = str(provider or "").strip().lower() or None
    provider_name = explicit_provider or inferred_provider or DEFAULT_PROVIDER.strip().lower()

    if explicit_provider and inferred_provider and explicit_provider != inferred_provider:
        raise ValueError(
            f"Provider/model mismatch: provider={explicit_provider!r}, model={model_name!r}. "
            f"Model {model_name!r} looks like a {inferred_provider} model."
        )

    if not model_name:
        if provider_name == "gemini":
            model_name = GEMINI_DEFAULT_MODEL.strip()
        else:
            model_name = OPENAI_DEFAULT_MODEL.strip()

    config = ModelConfig(
        provider=provider_name,
        model=model_name,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )

    if provider_name == "openai":
        return OpenAIHostModel(config)
    if provider_name == "gemini":
        return GeminiHostModel(config)

    raise ValueError(
        f"Unsupported provider: {provider_name}. This refactor implements host adapters for openai and gemini only."
    )


def _infer_provider_from_model_name(model_name: str) -> str | None:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return None
    if normalized.startswith(GEMINI_MODEL_PREFIXES):
        return "gemini"
    if normalized.startswith(OPENAI_MODEL_PREFIXES):
        return "openai"
    return None
