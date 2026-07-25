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

from slicer_agent_engine.llm.adapters.gemini_native import GeminiNativeAdapter, _canonical_tool_to_gemini_function
from slicer_agent_engine.llm.types import ToolCall, make_text_prompt


class FakeGeminiResponse:
    def __init__(self, content: Any) -> None:
        self.id = "gem_resp_1"
        self.candidates = [SimpleNamespace(content=content)]

    @property
    def text(self) -> str:  # pragma: no cover - adapter should never touch this
        raise AssertionError("Gemini adapter should inspect candidates.content.parts directly, not response.text")


class FakeGeminiModels:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = SimpleNamespace(
            role="model",
            parts=[
                SimpleNamespace(function_call=SimpleNamespace(name="observe", args={"size": 320}, id="gem_call_1")),
                SimpleNamespace(text="inspecting"),
            ],
        )
        return FakeGeminiResponse(content)


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


class GeminiAdapterTests(unittest.TestCase):
    def test_request_turn_reads_candidates_content_parts_not_response_text(self) -> None:
        adapter = GeminiNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello world")])
        client = FakeGeminiClient()

        result = adapter.request_turn(
            client=client,
            model="gemini-3.1-pro-preview",
            instructions="inspect",
            state=state,
            tool_schemas=[{"kind": "function", "name": "observe", "description": "Observe", "parameters": {"type": "object", "properties": {"size": {"type": "integer"}}}}],
            extra_model_tools=[{"type": "code_execution", "code_execution": {}}],
            reasoning_effort="low",
        )

        self.assertEqual(result.tool_calls[0].name, "observe")
        self.assertEqual(result.tool_calls[0].call_id, "gem_call_1")
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-pro-preview")
        self.assertEqual(call["config"]["thinking_config"]["thinking_level"], "low")
        self.assertEqual(call["config"]["automatic_function_calling"]["disable"], True)
        self.assertEqual(call["config"]["tools"][0]["function_declarations"][0]["name"], "observe")
        self.assertEqual(call["config"]["tools"][1]["code_execution"], {})
        self.assertEqual(getattr(state.contents[-1], "role", None), "model")

    def test_canonical_gemini_function_schema_drops_additional_properties_for_parameters_field(self) -> None:
        fn = _canonical_tool_to_gemini_function(
            {
                "kind": "function",
                "name": "fusion",
                "description": "Fuse two volumes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "background_volume_ref": {"type": "string"},
                        "foreground_volume_ref": {"type": "string"},
                    },
                    "required": ["background_volume_ref", "foreground_volume_ref"],
                    "additionalProperties": False,
                },
            }
        )
        self.assertNotIn("additionalProperties", fn["parameters"])
        self.assertEqual(fn["parameters"]["required"], ["background_volume_ref", "foreground_volume_ref"])

    def test_append_tool_result_nests_inline_data_inside_function_response_without_refs(self) -> None:
        adapter = GeminiNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello")])
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "select_volume_1.png"
            png_path.write_bytes(b"fakepng")
            count = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="gem_call_1", name="select_volume", arguments_json="{}"),
                result={"ok": True, "png_path": str(png_path), "note": "updated"},
            )
        self.assertEqual(count, 1)
        last = state.contents[-1]
        self.assertEqual(last["role"], "user")
        part = last["parts"][0]
        self.assertIn("functionResponse", part)
        fr = part["functionResponse"]
        self.assertIn("parts", fr)
        self.assertEqual(fr["id"], "gem_call_1")
        self.assertIn("parts", fr)
        self.assertEqual(fr["parts"][0]["inlineData"]["mimeType"], "image/png")
        self.assertNotIn("displayName", fr["parts"][0]["inlineData"])
        self.assertEqual(fr["response"]["attached_image_count"], 1)
        self.assertNotIn("png_path", fr["response"]["result"])
        self.assertNotIn("image_1", fr["response"])

    def test_append_tool_result_sanitizes_artifact_like_strings_from_structured_response(self) -> None:
        adapter = GeminiNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello")])
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "select_volume_1.png"
            png_path.write_bytes(b"fakepng")
            adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="gem_call_1", name="select_volume", arguments_json="{}"),
                result={
                    "ok": True,
                    "png_path": str(png_path),
                    "artifact_name": "select_volume_artifact_3.png",
                    "nested": {"preview": "/tmp/select_volume_artifact_3.png"},
                },
            )
        fr = state.contents[-1]["parts"][0]["functionResponse"]
        self.assertEqual(fr["response"]["result"]["artifact_name"], "[artifact omitted]")
        self.assertEqual(fr["response"]["result"]["nested"]["preview"], "[artifact omitted]")

    def test_append_tool_result_coalesces_multiple_function_responses(self) -> None:
        adapter = GeminiNativeAdapter()
        state = adapter.start_conversation(inputs=[make_text_prompt("hello")])
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "obs.png"
            png_path.write_bytes(b"fakepng")
            count1 = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="gem_call_1", name="observe", arguments_json="{}"),
                result={"ok": True, "png_path": str(png_path)},
            )
            count2 = adapter.append_tool_result(
                state=state,
                call=ToolCall(call_id="gem_call_2", name="zoom", arguments_json="{}"),
                result={"ok": True},
                attach_images=False,
            )
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 0)
        last = state.contents[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(len(last["parts"]), 2)


if __name__ == "__main__":
    unittest.main()
