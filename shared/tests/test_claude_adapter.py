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

from slicer_agent_engine.llm.adapters.claude_native import ClaudeNativeAdapter
from slicer_agent_engine.llm.types import ToolCall, make_text_prompt


class FakeClaudeMessages:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="claude_resp_1",
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="planning"),
                SimpleNamespace(type="tool_use", id="toolu_1", name="observe", input={"size": 256}),
            ],
        )


class FakeClaudeClient:
    def __init__(self) -> None:
        self.messages = FakeClaudeMessages()


class ClaudeAdapterTests(unittest.TestCase):
    def test_request_turn_extracts_tool_use_blocks(self) -> None:
        adapter = ClaudeNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])
        client = FakeClaudeClient()

        result = adapter.request_turn(
            client=client,
            model="claude-opus-4-1",
            instructions="inspect",
            state=state,
            tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
            reasoning_effort="low",
        )

        self.assertEqual(result.tool_calls[0].name, "observe")
        self.assertEqual(result.tool_calls[0].call_id, "toolu_1")
        create_kwargs = client.messages.calls[0]
        self.assertEqual(create_kwargs["model"], "claude-opus-4-1")
        self.assertNotIn("strict", create_kwargs["tools"][0])
        self.assertEqual(create_kwargs["tools"][0]["input_schema"]["additionalProperties"], False)
        self.assertEqual(create_kwargs["thinking"]["type"], "enabled")
        self.assertEqual(state.messages[-1]["role"], "assistant")

    def test_append_tool_result_builds_tool_result_blocks_with_images(self) -> None:
        adapter = ClaudeNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello")])
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            count = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="toolu_1", name="observe", arguments_json="{}"),
                result={"ok": True, "png_path": str(png_path)},
            )
            adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="toolu_2", name="zoom", arguments_json="{}"),
                result={"ok": False, "error": "bad"},
                attach_images=False,
            )
        self.assertEqual(count, 1)
        last = state.messages[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"][0]["type"], "tool_result")
        self.assertEqual(last["content"][1]["tool_use_id"], "toolu_2")
        self.assertEqual(last["content"][1]["is_error"], True)


if __name__ == "__main__":
    unittest.main()
