"""Unit tests for the Pydantic AI governance model wrapper.

These tests use real ``pydantic_ai`` message parts (``UserPromptPart`` etc.)
so the part-extraction logic is exercised against the actual types, plus the
adapter's model-wrapping attach/detach against a real ``Agent`` (driven by the
offline ``TestModel``).

The package is configured with ``asyncio_mode = "auto"``, so ``async def``
tests run without an explicit marker.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
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

from uipath_pydantic_ai.governance.model import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
    GovernanceModel,
    _coerce_args,
    install_governance,
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
# install_governance
# --------------------------------------------------------------------------


def test_install_governance_wraps_model():
    agent = Agent(model=TestModel())
    returned = install_governance(agent, FakeEvaluator(), agent_name="x", session_id="s")
    assert returned is agent
    assert isinstance(agent.model, GovernanceModel)


def test_install_governance_is_idempotent():
    agent = Agent(model=TestModel())
    ev = FakeEvaluator()
    install_governance(agent, ev, agent_name="x", session_id="s")
    wrapped = agent.model
    install_governance(agent, ev, agent_name="x", session_id="s")
    assert agent.model is wrapped  # not double-wrapped


def test_install_governance_warns_when_no_bound_model(caplog):
    agent = Agent()  # no model bound
    with caplog.at_level(logging.WARNING):
        install_governance(agent, FakeEvaluator(), agent_name="x", session_id="s")
    assert any("no bound Model" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Factory wiring — the evaluator kwarg drives install_governance
# --------------------------------------------------------------------------


def _factory_without_init():
    """A factory instance that skips __init__ (avoids config/IO)."""
    from uipath_pydantic_ai.runtime.factory import UiPathPydanticAIRuntimeFactory

    return UiPathPydanticAIRuntimeFactory.__new__(UiPathPydanticAIRuntimeFactory)


async def test_factory_installs_governance_when_evaluator_supplied(monkeypatch):
    from uipath_pydantic_ai.runtime import factory as factory_mod

    monkeypatch.setattr(factory_mod, "UiPathPydanticAIRuntime", lambda **kw: SimpleNamespace(**kw))
    agent = Agent(model=TestModel())
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e", evaluator=FakeEvaluator()
    )
    assert isinstance(agent.model, GovernanceModel)


async def test_factory_skips_governance_without_evaluator(monkeypatch):
    from uipath_pydantic_ai.runtime import factory as factory_mod

    monkeypatch.setattr(factory_mod, "UiPathPydanticAIRuntime", lambda **kw: SimpleNamespace(**kw))
    agent = Agent(model=TestModel())
    original = agent.model
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e"
    )
    assert agent.model is original


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


async def test_governance_model_request_stream_block_propagates():
    # A DENY during the after-stream check must abort the run, exactly like the
    # non-streaming request() path — it must not be swallowed by the catch-all.
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    cb = _make_callbacks(FakeEvaluator(block_on="tool_call"))
    denied = ModelResponse(
        parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")]
    )

    class FakeWrapped:
        @asynccontextmanager
        async def request_stream(self, *_a, **_k):
            yield SimpleNamespace(get=lambda: denied)

    gm = GovernanceModel.__new__(GovernanceModel)  # bypass WrapperModel init
    gm.wrapped = FakeWrapped()  # type: ignore[attr-defined]
    gm._callbacks = cb
    messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    with pytest.raises(GovernanceBlockException):
        async with gm.request_stream(messages, None, None) as stream:
            assert stream is not None


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