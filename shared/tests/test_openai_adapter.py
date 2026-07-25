from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.llm.adapters.openai_native import OpenAIResponsesAdapter
from slicer_agent_engine.llm.types import make_text_prompt, ToolCall


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


class OpenAIAdapterTests(unittest.TestCase):
    def test_request_turn_converts_canonical_tools_and_inputs(self) -> None:
        adapter = OpenAIResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])
        client = FakeClient()

        result = adapter.request_turn(
            client=client,
            model="gpt-test",
            instructions="inspect",
            state=state,
            tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
            extra_model_tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            reasoning_effort="low",
        )

        self.assertEqual(result.output_text, "observed")
        self.assertEqual(result.tool_calls[0].name, "observe")
        create_kwargs = client.responses.calls[0]
        self.assertEqual(create_kwargs["model"], "gpt-test")
        self.assertEqual(create_kwargs["tools"][0]["type"], "function")
        self.assertEqual(create_kwargs["tools"][0]["name"], "observe")
        self.assertIn("include", create_kwargs)
        self.assertEqual(create_kwargs["reasoning"], {"effort": "low"})
        self.assertEqual(state.input_items[0]["role"], "user")
        self.assertEqual(state.input_items[-1].type, "function_call")

    def test_append_tool_result_attaches_function_output_and_image(self) -> None:
        adapter = OpenAIResponsesAdapter()
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
        self.assertEqual(state.input_items[1]["type"], "function_call_output")
        self.assertEqual(state.input_items[2]["role"], "user")
        self.assertEqual(state.input_items[2]["content"][1]["type"], "input_image")

    def test_request_turn_sanitizes_code_interpreter_outputs_before_replay(self) -> None:
        adapter = OpenAIResponsesAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])

        huge_url = "data:image/png;base64," + ("A" * 1_325_418)
        response = SimpleNamespace(
            id="resp_test",
            output_text="observed",
            output=[
                SimpleNamespace(
                    type="code_interpreter_call",
                    id="ci_1",
                    status="completed",
                    outputs=[
                        SimpleNamespace(type="logs", text="done"),
                        SimpleNamespace(type="image", mime_type="image/png", url=huge_url),
                    ],
                ),
                SimpleNamespace(type="function_call", call_id="call_1", name="observe", arguments='{"size": 256}'),
            ],
        )
        client = FakeClient()
        client.responses.create = lambda **kwargs: response

        result = adapter.request_turn(
            client=client,
            model="gpt-test",
            instructions="inspect",
            state=state,
            tool_schemas=[],
            extra_model_tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
            reasoning_effort="low",
        )

        self.assertEqual(result.tool_calls[0].name, "observe")
        replay_item = state.input_items[1]
        self.assertIsInstance(replay_item, dict)
        self.assertEqual(replay_item["type"], "code_interpreter_call")
        self.assertIsNone(replay_item["outputs"])


if __name__ == "__main__":
    unittest.main()
