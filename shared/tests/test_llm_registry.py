from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.llm.adapters.claude_native import ClaudeNativeAdapter
from slicer_agent_engine.llm.adapters.gemini_native import GeminiNativeAdapter
from slicer_agent_engine.llm.adapters.openai_native import (
    DoubaoResponsesAdapter,
    LocalVLLMResponsesAdapter,
    OpenAIResponsesAdapter,
    QwenResponsesAdapter,
)
from slicer_agent_engine.llm.registry import (
    UnsupportedProviderError,
    build_builtin_model_tools,
    build_model_adapter,
    build_model_client,
    default_model_from_env,
    detect_provider_from_model,
    registered_providers,
    resolve_provider_name,
)


class RegistryTests(unittest.TestCase):
    def test_core_providers_are_registered(self) -> None:
        self.assertIn("openai", registered_providers())
        self.assertIn("gemini", registered_providers())
        self.assertIn("claude", registered_providers())
        self.assertIn("qwen", registered_providers())
        self.assertIn("doubao", registered_providers())
        self.assertIn("local_vllm", registered_providers())
        self.assertIsInstance(build_model_adapter("openai"), OpenAIResponsesAdapter)
        self.assertIsInstance(build_model_adapter("gemini"), GeminiNativeAdapter)
        self.assertIsInstance(build_model_adapter("claude"), ClaudeNativeAdapter)
        self.assertIsInstance(build_model_adapter("qwen"), QwenResponsesAdapter)
        self.assertIsInstance(build_model_adapter("doubao"), DoubaoResponsesAdapter)
        self.assertIsInstance(build_model_adapter("vllm"), LocalVLLMResponsesAdapter)
        self.assertEqual(build_builtin_model_tools("openai", enable_code_interpreter=True)[0]["type"], "code_interpreter")
        self.assertEqual(build_builtin_model_tools("gemini", enable_code_interpreter=True)[0]["type"], "code_execution")
        self.assertEqual(build_builtin_model_tools("claude", enable_code_interpreter=True), [])
        self.assertEqual(build_builtin_model_tools("qwen", enable_code_interpreter=True)[0]["type"], "code_interpreter")
        self.assertEqual(build_builtin_model_tools("doubao", enable_code_interpreter=True), [])
        self.assertEqual(build_builtin_model_tools("local_vllm", enable_code_interpreter=True), [])

    def test_provider_detection_from_model_name(self) -> None:
        self.assertEqual(detect_provider_from_model("gemini-3.1-pro-preview"), "gemini")
        self.assertEqual(detect_provider_from_model("claude-opus-4-1"), "claude")
        self.assertEqual(detect_provider_from_model("gpt-5-mini"), "openai")
        self.assertEqual(detect_provider_from_model("qwen3-vl-plus"), "qwen")
        self.assertEqual(detect_provider_from_model("seed-2-0-lite-260228"), "doubao")
        self.assertEqual(resolve_provider_name("auto", "gemini-3.1-flash"), "gemini")
        self.assertEqual(resolve_provider_name("anthropic", "claude-3-7-sonnet"), "claude")
        self.assertEqual(resolve_provider_name("google", "gemini-3.1-pro-preview"), "gemini")
        self.assertEqual(resolve_provider_name("dashscope", "qwen3.5-plus"), "qwen")
        self.assertEqual(resolve_provider_name("ark", "seed-2-0-pro-260328"), "doubao")
        self.assertEqual(resolve_provider_name("local-vllm", "google/medgemma-1.5-4b-it"), "local_vllm")

    def test_default_model_from_env_respects_provider_hint(self) -> None:
        env = {
            "LLM_PROVIDER": "qwen",
            "QWEN_MODEL": "qwen3.5-plus",
            "OPENAI_MODEL": "gpt-5-mini",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_model_from_env(), "qwen3.5-plus")

        env = {
            "LLM_PROVIDER": "doubao",
            "ARK_MODEL": "seed-2-0-lite-260228",
            "OPENAI_MODEL": "gpt-5-mini",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_model_from_env(), "seed-2-0-lite-260228")

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(UnsupportedProviderError):
            build_model_adapter("does-not-exist")

    def test_qwen_client_defaults_to_cn_compatible_mode_base_url(self) -> None:
        captured: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}, clear=True):
            with mock.patch("openai.OpenAI", FakeOpenAI):
                build_model_client("qwen", model="qwen3.5-35b-a3b")

        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_qwen_client_normalizes_legacy_compatible_mode_path(self) -> None:
        captured: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        env = {
            "DASHSCOPE_API_KEY": "sk-test",
            "DASHSCOPE_BASE_URL": "https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("openai.OpenAI", FakeOpenAI):
                build_model_client("qwen", model="qwen3.5-35b-a3b")

        self.assertEqual(captured["base_url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

    def test_local_vllm_client_defaults_to_local_responses_endpoint(self) -> None:
        captured: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("openai.OpenAI", FakeOpenAI):
                build_model_client("local_vllm", model="google/medgemma-1.5-4b-it")

        self.assertEqual(captured["api_key"], "EMPTY")
        self.assertEqual(captured["base_url"], "http://127.0.0.1:8000/v1")


if __name__ == "__main__":
    unittest.main()
