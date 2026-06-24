"""Unit tests for the Pydantic AI governance adapter.

These tests use real ``pydantic_ai`` message parts (``UserPromptPart`` etc.)
so the part-extraction logic is exercised against the actual types, plus the
adapter's model-wrapping attach/detach against a real ``Agent`` (driven by the
offline ``TestModel``).

The package is configured with ``asyncio_mode = "auto"``, so ``async def``
tests run without an explicit marker.
"""

from __future__ import annotations

import logging
from typing import Any, List

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_pydantic_ai.governance.adapter import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
    GovernanceModel,
    PydanticAIAdapter,
    _coerce_args,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeEvaluator:
    """Records evaluate_* calls; optionally BLOCKs on a named hook."""

    def __init__(self, block_on: str | None = None) -> None:
        self.block_on = block_on
        self.calls: List[tuple[str, dict]] = []

    def _record(self, hook: str, **kwargs: Any) -> None:
        self.calls.append((hook, kwargs))
        if self.block_on == hook:
            raise GovernanceBlockException("blocked")  # type: ignore[call-arg]

    def evaluate_before_agent(self, **kwargs: Any) -> None:
        self._record("before_agent", **kwargs)

    def evaluate_after_agent(self, **kwargs: Any) -> None:
        self._record("after_agent", **kwargs)

    def evaluate_before_model(self, **kwargs: Any) -> None:
        self._record("before_model", **kwargs)

    def evaluate_after_model(self, **kwargs: Any) -> None:
        self._record("after_model", **kwargs)

    def evaluate_tool_call(self, **kwargs: Any) -> None:
        self._record("tool_call", **kwargs)

    def evaluate_after_tool(self, **kwargs: Any) -> None:
        self._record("after_tool", **kwargs)


def _make_callbacks(ev: FakeEvaluator) -> GovernanceCallbacks:
    return GovernanceCallbacks(evaluator=ev, agent_name="agent-1", session_id="sess-1")


def _hooks(ev: FakeEvaluator) -> List[str]:
    return [h for h, _ in ev.calls]


# --------------------------------------------------------------------------
# can_handle
# --------------------------------------------------------------------------


def test_can_handle_agent():
    assert PydanticAIAdapter().can_handle(Agent(model=TestModel())) is True


def test_can_handle_rejects_non_agent():
    from types import SimpleNamespace

    # A duck-typed look-alike (model/run/iter) must NOT be claimed — only a real Agent.
    look_alike = SimpleNamespace(model=object(), run=lambda: None, iter=lambda: None)
    assert PydanticAIAdapter().can_handle(look_alike) is False
    assert PydanticAIAdapter().can_handle(object()) is False


# --------------------------------------------------------------------------
# attach / detach
# --------------------------------------------------------------------------


def test_attach_wraps_model_and_detach_restores():
    agent = Agent(model=TestModel())
    original = agent.model
    adapter = PydanticAIAdapter()
    returned = adapter.attach(agent, agent_id="x", session_id="s", evaluator=FakeEvaluator())
    assert returned is agent
    assert isinstance(agent.model, GovernanceModel)
    adapter.detach(agent)
    assert agent.model is original


def test_attach_is_idempotent():
    agent = Agent(model=TestModel())
    adapter = PydanticAIAdapter()
    ev = FakeEvaluator()
    adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
    wrapped = agent.model
    adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
    assert agent.model is wrapped  # not double-wrapped


def test_attach_warns_when_no_bound_model(caplog):
    agent = Agent()  # no model bound
    with caplog.at_level(logging.WARNING):
        PydanticAIAdapter().attach(agent, agent_id="x", session_id="s", evaluator=FakeEvaluator())
    assert any("no bound Model" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# on_request → BEFORE_MODEL + AFTER_TOOL
# --------------------------------------------------------------------------


def test_on_request_fires_before_model_with_latest_user_prompt():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    messages = [
        ModelRequest(parts=[UserPromptPart(content="old turn")]),
        ModelRequest(parts=[UserPromptPart(content="the question")]),
    ]
    cb.on_request(messages)
    assert _hooks(ev) == ["before_model"]
    assert ev.calls[0][1]["model_input"] == "the question"


def test_on_request_fires_after_tool_for_tool_return():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    messages = [
        ModelRequest(
            parts=[ToolReturnPart(tool_name="lookup", content={"balance": "1000"}, tool_call_id="c1")]
        )
    ]
    cb.on_request(messages)
    # both BEFORE_MODEL (tool result is the model's new input) and AFTER_TOOL fire
    assert "before_model" in _hooks(ev)
    after_tool = [kw for h, kw in ev.calls if h == "after_tool"]
    assert after_tool and after_tool[0]["tool_name"] == "lookup"
    assert "1000" in after_tool[0]["tool_result"]


def test_on_request_caps_text():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    cb.on_request([ModelRequest(parts=[UserPromptPart(content=huge)])])
    assert len(ev.calls[0][1]["model_input"]) <= _BEFORE_MODEL_TEXT_CAP


def test_on_request_empty():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.on_request([])
    assert ev.calls[0][1]["model_input"] == ""


# --------------------------------------------------------------------------
# on_response → AFTER_MODEL + TOOL_CALL
# --------------------------------------------------------------------------


def test_on_response_fires_after_model_and_tool_call():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    response = ModelResponse(
        parts=[
            TextPart(content="thinking out loud"),
            ToolCallPart(tool_name="transfer", args={"amount": 50}, tool_call_id="c1"),
        ]
    )
    cb.on_response(response)
    assert "after_model" in _hooks(ev) and "tool_call" in _hooks(ev)
    after_model = [kw for h, kw in ev.calls if h == "after_model"][0]
    assert after_model["model_output"] == "thinking out loud"
    tool_call = [kw for h, kw in ev.calls if h == "tool_call"][0]
    assert tool_call["tool_name"] == "transfer"
    assert tool_call["tool_args"] == {"amount": 50}
    assert tool_call["session_state"]["tool_calls"] == 1


def test_on_response_coerces_json_string_args():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    response = ModelResponse(
        parts=[ToolCallPart(tool_name="t", args='{"x": 1}', tool_call_id="c1")]
    )
    cb.on_response(response)
    tool_call = [kw for h, kw in ev.calls if h == "tool_call"][0]
    assert tool_call["tool_args"] == {"x": 1}


# --------------------------------------------------------------------------
# GovernanceModel.request brackets a wrapped model
# --------------------------------------------------------------------------


async def test_governance_model_request_brackets_call():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    order: List[str] = []

    class FakeWrapped:
        async def request(self, messages, settings, params):
            order.append("MODEL_CALL")
            return ModelResponse(parts=[TextPart(content="Your balance is 1000.")])

    gm = GovernanceModel.__new__(GovernanceModel)  # bypass WrapperModel init
    gm.wrapped = FakeWrapped()  # type: ignore[attr-defined]
    gm._callbacks = cb
    messages = [ModelRequest(parts=[UserPromptPart(content="What is my balance?")])]
    await gm.request(messages, None, None)

    assert order == ["MODEL_CALL"]
    assert _hooks(ev) == ["before_model", "after_model"]
    assert ev.calls[0][1]["model_input"] == "What is my balance?"
    assert ev.calls[1][1]["model_output"] == "Your balance is 1000."


# --------------------------------------------------------------------------
# helpers + enforcement
# --------------------------------------------------------------------------


def test_coerce_args_variants():
    assert _coerce_args({"a": 1}) == {"a": 1}
    assert _coerce_args('{"a": 1}') == {"a": 1}
    assert _coerce_args(None) == {}
    assert _coerce_args("not json") == {}


def test_block_in_before_model_propagates():
    cb = _make_callbacks(FakeEvaluator(block_on="before_model"))
    with pytest.raises(GovernanceBlockException):
        cb.on_request([ModelRequest(parts=[UserPromptPart(content="hi")])])


def test_block_in_tool_call_propagates():
    cb = _make_callbacks(FakeEvaluator(block_on="tool_call"))
    with pytest.raises(GovernanceBlockException):
        cb.on_response(
            ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")])
        )


def test_non_block_exception_is_swallowed(caplog):
    class Boom:
        def evaluate_before_model(self, **_: Any) -> None:
            raise RuntimeError("evaluator bug")

    cb = GovernanceCallbacks(evaluator=Boom(), agent_name="a", session_id="s")  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        cb.on_request([ModelRequest(parts=[UserPromptPart(content="x")])])
    assert any("governance check failed" in r.message for r in caplog.records)