from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engines"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from slicer_agent_engine.llm.base import BaseModelAdapter
from slicer_agent_engine.llm.loop import run_model_tool_loop
from slicer_agent_engine.llm.types import ModelTurnResult, TextPrompt, ToolCall


@dataclass
class DummyState:
    items: List[Any] = field(default_factory=list)
    requests: int = 0


class DummyAdapter(BaseModelAdapter):
    provider_name = "dummy"

    def start_conversation(self, *, inputs: Sequence[Any]) -> DummyState:
        return DummyState(items=list(inputs))

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: DummyState,
        tool_schemas: Sequence[Dict[str, Any]],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        state.requests += 1
        if state.requests == 1:
            return ModelTurnResult(
                output_text="",
                tool_calls=[ToolCall(call_id="call_1", name="dummy_tool", arguments_json='{"value": 7}')],
                response_id="resp_1",
                output_item_types=["function_call"],
            )
        return ModelTurnResult(
            output_text="final answer",
            tool_calls=[],
            response_id="resp_2",
            output_item_types=["message"],
        )

    def append_tool_result(
        self,
        *,
        state: DummyState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        state.items.append({"call_id": call.call_id, "name": call.name, "result": result})
        return 0

    def export_state(self, state: DummyState) -> List[Any]:
        return list(state.items)


class BudgetAdapter(BaseModelAdapter):
    provider_name = "budget"

    def start_conversation(self, *, inputs: Sequence[Any]) -> DummyState:
        return DummyState(items=list(inputs))

    def request_turn(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        state: DummyState,
        tool_schemas: Sequence[Dict[str, Any]],
        extra_model_tools: Optional[Sequence[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        retry_on_timeout: int = 0,
    ) -> ModelTurnResult:
        state.requests += 1
        if state.requests == 1:
            return ModelTurnResult(
                output_text="",
                tool_calls=[ToolCall(call_id="call_1", name="dummy_tool", arguments_json='{"value": 1}')],
                response_id="resp_1",
                output_item_types=["function_call"],
            )
        if tool_schemas:
            return ModelTurnResult(
                output_text="",
                tool_calls=[ToolCall(call_id="call_2", name="dummy_tool", arguments_json='{"value": 2}')],
                response_id="resp_2",
                output_item_types=["function_call"],
            )
        return ModelTurnResult(
            output_text="forced final answer",
            tool_calls=[],
            response_id="resp_final",
            output_item_types=["message"],
        )

    def append_tool_result(
        self,
        *,
        state: DummyState,
        call: ToolCall,
        result: Any,
        attach_images: bool = True,
        max_images: int = 4,
    ) -> int:
        state.items.append({"call_id": call.call_id, "name": call.name, "result": result})
        return 0

    def export_state(self, state: DummyState) -> List[Any]:
        return list(state.items)


class DummySession:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_run_event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"event_type": event_type, **payload})


class ProviderBoundaryLoopTests(unittest.TestCase):
    def test_generic_loop_runs_with_non_openai_adapter(self) -> None:
        session = DummySession()
        adapter = DummyAdapter()

        def executor(name: str, arguments_json: str) -> Dict[str, Any]:
            self.assertEqual(name, "dummy_tool")
            self.assertEqual(arguments_json, '{"value": 7}')
            return {"ok": True, "echo": 7}

        final_text, snapshot = run_model_tool_loop(
            adapter=adapter,
            client=object(),
            model="dummy-model",
            tool_schemas=[{"kind": "function", "name": "dummy_tool", "description": "demo", "parameters": {"type": "object", "properties": {}}}],
            initial_inputs=[TextPrompt(text="hello")],
            instructions="demo instructions",
            tool_executor=executor,
            session=session,
        )

        self.assertEqual(final_text, "final answer")
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot[1]["result"]["echo"], 7)
        event_types = [evt["event_type"] for evt in session.events]
        self.assertIn("model_request_start", event_types)
        self.assertIn("tool_call_start", event_types)
        self.assertIn("model_loop_finished", event_types)

    def test_budget_exhaustion_forces_final_answer_without_executing_more_tools(self) -> None:
        session = DummySession()
        adapter = BudgetAdapter()
        executed: List[str] = []

        def executor(name: str, arguments_json: str) -> Dict[str, Any]:
            executed.append(arguments_json)
            return {"ok": True, "echo": arguments_json}

        final_text, snapshot = run_model_tool_loop(
            adapter=adapter,
            client=object(),
            model="budget-model",
            tool_schemas=[{"kind": "function", "name": "dummy_tool", "description": "demo", "parameters": {"type": "object", "properties": {}}}],
            initial_inputs=[TextPrompt(text="hello")],
            instructions="demo instructions",
            tool_executor=executor,
            session=session,
            max_rounds=1,
        )

        self.assertEqual(final_text, "forced final answer")
        self.assertEqual(executed, ['{"value": 1}'])
        self.assertEqual(snapshot[-1]["result"]["tool"], "dummy_tool")
        self.assertFalse(snapshot[-1]["result"]["ok"])
        event_types = [evt["event_type"] for evt in session.events]
        self.assertIn("tool_budget_exhausted", event_types)
        self.assertIn("model_loop_finished", event_types)


if __name__ == "__main__":
    unittest.main()
