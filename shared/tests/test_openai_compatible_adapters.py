from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.llm.adapters.openai_native import (
    DoubaoResponsesAdapter,
    LocalVLLMResponsesAdapter,
    QwenResponsesAdapter,
)
from slicer_agent_engine.llm.types import ToolCall, make_image_prompt, make_text_prompt


class FakeResponses:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            output_text="observed",
            output=[SimpleNamespace(type="function_call", call_id="call_1", name="observe", arguments='{"size": 256}')],
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FakeManualResponses:
    def __init__(self, output_text: str = '<tool_call>{"name":"observe","arguments":{"size":256}}</tool_call>') -> None:
        self.calls: List[Dict[str, Any]] = []
        self.output_text = output_text

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_manual",
            status="completed",
            output_text=self.output_text,
            output=[SimpleNamespace(type="message")],
            error=None,
        )


class FakeManualClient:
    def __init__(self, output_text: str = '<tool_call>{"name":"observe","arguments":{"size":256}}</tool_call>') -> None:
        self.responses = FakeManualResponses(output_text=output_text)


class QwenAdapterTests(unittest.TestCase):
    def test_qwen_uses_qwen_response_shapes(self) -> None:
        adapter = QwenResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])
        client = FakeClient()

        result = adapter.request_turn(
            client=client,
            model="qwen3.5-plus",
            instructions="inspect",
            state=state,
            tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
            extra_model_tools=[{"type": "code_interpreter"}],
            reasoning_effort="medium",
        )

        self.assertEqual(result.output_text, "observed")
        self.assertEqual(state.input_items[0], {"role": "user", "content": "hello world"})
        create_kwargs = client.responses.calls[0]
        self.assertEqual(create_kwargs["tools"][0]["type"], "function")
        self.assertEqual(create_kwargs["tools"][1]["type"], "code_interpreter")
        self.assertEqual(create_kwargs["extra_body"], {"enable_thinking": True})
        self.assertNotIn("reasoning", create_kwargs)
        self.assertNotIn("include", create_kwargs)

    def test_qwen_attaches_tool_images_with_text_parts(self) -> None:
        adapter = QwenResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello")])
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            count = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="call_1", name="observe", arguments_json="{}"),
                result={"ok": True, "png_path": str(png_path)},
            )
        self.assertEqual(count, 1)
        self.assertEqual(state.input_items[2]["content"][0]["type"], "input_text")
        self.assertEqual(state.input_items[2]["content"][1]["type"], "input_image")

    def test_qwen_multimodal_falls_back_to_manual_tool_protocol(self) -> None:
        adapter = QwenResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(inputs=[make_image_prompt(png_path, text="inspect this image")])
            client = FakeManualClient()

            result = adapter.request_turn(
                client=client,
                model="qwen3.5-plus",
                instructions="inspect",
                state=state,
                tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
                extra_model_tools=[{"type": "code_interpreter"}],
                reasoning_effort="none",
            )

        create_kwargs = client.responses.calls[0]
        self.assertNotIn("tools", create_kwargs)
        self.assertIn("Qwen manual tool protocol", create_kwargs["instructions"])
        self.assertIn("<tool_call>", create_kwargs["instructions"])
        self.assertEqual(create_kwargs["extra_body"], {"enable_thinking": False})
        self.assertEqual(result.tool_calls[0].name, "observe")
        self.assertEqual(result.tool_calls[0].arguments_json, '{"size": 256}')
        self.assertTrue(state.manual_tool_protocol)
        self.assertEqual(state.input_items[-1]["role"], "assistant")

    def test_qwen_manual_protocol_appends_tool_results_as_user_messages(self) -> None:
        adapter = QwenResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(inputs=[make_image_prompt(png_path, text="hello")])
            state.manual_tool_protocol = True
            count = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="call_1", name="observe", arguments_json="{}"),
                result={"ok": True, "png_path": str(png_path)},
            )
        self.assertEqual(count, 1)
        self.assertEqual(state.input_items[1]["role"], "user")
        self.assertEqual(state.input_items[1]["content"][0]["type"], "input_text")
        self.assertIn("Tool observe returned", state.input_items[1]["content"][0]["text"])
        self.assertEqual(state.input_items[2]["content"][0]["type"], "input_text")
        self.assertEqual(state.input_items[2]["content"][1]["type"], "input_image")


class LocalVLLMAdapterTests(unittest.TestCase):
    def test_multimodal_request_uses_model_neutral_manual_protocol(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(inputs=[make_image_prompt(png_path, text="inspect")])
            client = FakeManualClient()
            result = adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object"}}],
                reasoning_effort="medium",
            )

        create_kwargs = client.responses.calls[0]
        self.assertIn("Local vLLM manual tool protocol", create_kwargs["instructions"])
        self.assertNotIn("extra_body", create_kwargs)
        self.assertEqual(create_kwargs["max_output_tokens"], 2048)
        self.assertEqual(result.tool_calls[0].name, "observe")

    def test_multimodal_request_merges_consecutive_user_messages(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(
                inputs=[
                    make_text_prompt("task"),
                    make_image_prompt(png_path, text="axial image"),
                ]
            )
            client = FakeManualClient()
            adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object"}}],
                reasoning_effort="medium",
            )

        input_items = client.responses.calls[0]["input"]
        self.assertEqual(len(input_items), 1)
        self.assertEqual(input_items[0]["role"], "user")
        self.assertEqual(
            [item["type"] for item in input_items[0]["content"]],
            ["input_text", "input_text", "input_image"],
        )

    def test_output_token_limit_can_be_overridden(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("task")])
        client = FakeManualClient()
        with mock.patch.dict("os.environ", {"LOCAL_VLLM_MAX_OUTPUT_TOKENS": "512"}):
            adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[],
            )
        self.assertEqual(client.responses.calls[0]["max_output_tokens"], 512)

    def test_tool_round_replay_drops_unneeded_reasoning_text(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(inputs=[make_image_prompt(png_path, text="task")])
            client = FakeManualClient(
                output_text=(
                    "Long reasoning that should not be replayed.\n"
                    '<tool_call>{"name":"observe","arguments":{"size":256}}</tool_call>'
                )
            )
            result = adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object"}}],
            )
        self.assertEqual(result.tool_calls[0].name, "observe")
        self.assertEqual(
            state.input_items[-1]["content"],
            '<tool_call>{"name":"observe","arguments":{"size":256}}</tool_call>',
        )

    def test_manual_protocol_executes_one_tool_per_turn(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            state = adapter.start_conversation(inputs=[make_image_prompt(png_path, text="task")])
            client = FakeManualClient(
                output_text=(
                    '<tool_call>{"name":"observe","arguments":{"size":256}}</tool_call>\n'
                    '<tool_call>{"name":"observe","arguments":{"size":512}}</tool_call>'
                )
            )
            result = adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object"}}],
            )
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments_json, '{"size": 256}')
        self.assertIn("Call only one tool per turn", client.responses.calls[0]["instructions"])

    def test_request_trims_old_images_to_configured_limit(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_prompts = []
            for index in range(3):
                png_path = Path(tmpdir) / f"obs-{index}.png"
                png_path.write_bytes(b"fakepng")
                image_prompts.append(make_image_prompt(png_path, text=f"image {index}"))
            state = adapter.start_conversation(inputs=image_prompts)
            client = FakeManualClient()
            with mock.patch.dict("os.environ", {"LOCAL_VLLM_MAX_INPUT_IMAGES": "2"}):
                adapter.request_turn(
                    client=client,
                    model="google/medgemma-1.5-4b-it",
                    instructions="inspect",
                    state=state,
                    tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object"}}],
                )
        content = client.responses.calls[0]["input"][0]["content"]
        self.assertEqual(sum(item["type"] == "input_image" for item in content), 2)
        self.assertEqual(sum(item["type"] == "input_text" for item in content), 3)

    def test_request_keeps_initial_prompt_and_recent_tool_rounds(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("original task")])
        state.manual_tool_protocol = True
        for index in range(5):
            state.input_items.append(
                {
                    "role": "assistant",
                    "content": f'<tool_call>{{"name":"observe","arguments":{{"size":{index}}}}}</tool_call>',
                }
            )
            state.input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"tool result {index}"}],
                }
            )
        client = FakeManualClient(output_text="done")
        with mock.patch.dict("os.environ", {"LOCAL_VLLM_MAX_HISTORY_TOOL_ROUNDS": "2"}):
            adapter.request_turn(
                client=client,
                model="google/medgemma-1.5-4b-it",
                instructions="inspect",
                state=state,
                tool_schemas=[],
            )

        rendered_input = json.dumps(client.responses.calls[0]["input"])
        self.assertIn("original task", rendered_input)
        self.assertIn("3 earlier tool round(s)", rendered_input)
        self.assertNotIn("tool result 2", rendered_input)
        self.assertIn("tool result 3", rendered_input)
        self.assertIn("tool result 4", rendered_input)

    def test_local_tool_result_is_truncated_to_configured_limit(self) -> None:
        adapter = LocalVLLMResponsesAdapter()
        with mock.patch.dict("os.environ", {"LOCAL_VLLM_MAX_TOOL_RESULT_CHARS": "20"}):
            prompt = adapter._manual_tool_result_prompt(
                call=ToolCall(call_id="call_1", name="observe", arguments_json="{}"),
                result={"payload": "x" * 100},
            )
        self.assertIn("Tool result truncated", prompt)
        self.assertLess(len(prompt), 500)


class DoubaoAdapterTests(unittest.TestCase):
    def test_doubao_uses_ark_reasoning_shape(self) -> None:
        adapter = DoubaoResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])
        client = FakeClient()

        result = adapter.request_turn(
            client=client,
            model="seed-2-0-lite-260228",
            instructions="inspect",
            state=state,
            tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
            reasoning_effort="none",
        )

        self.assertEqual(result.output_text, "observed")
        self.assertEqual(state.input_items[0]["content"][0]["type"], "input_text")
        create_kwargs = client.responses.calls[0]
        self.assertEqual(create_kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning", create_kwargs)
        self.assertNotIn("include", create_kwargs)


if __name__ == "__main__":
    unittest.main()
