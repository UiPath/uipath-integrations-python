"""Unit tests for the LlamaIndex governance adapter.

The adapter governs via the LlamaIndex instrumentation dispatcher, so these
tests exercise the real event types (``LLMChatStartEvent`` etc.) routed
through :class:`GovernanceEventHandler`, plus the adapter's register/detach on
the dispatcher. The dispatcher is process-global, so each dispatcher test
cleans up after itself via ``detach``.
"""

from __future__ import annotations

import logging
from typing import Any, List

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.tools.types import ToolMetadata
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_llamaindex.governance.adapter import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
    GovernanceEventHandler,
    LlamaIndexAdapter,
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


class FakeWorkflow:
    """Duck-typed LlamaIndex workflow stand-in."""

    async def run(self, *_a: Any, **_k: Any) -> None:
        return None


def _make_callbacks(ev: FakeEvaluator) -> GovernanceCallbacks:
    return GovernanceCallbacks(evaluator=ev, agent_name="agent-1", session_id="sess-1")


def _handler(ev: FakeEvaluator) -> GovernanceEventHandler:
    return GovernanceEventHandler(callbacks=_make_callbacks(ev))


# --------------------------------------------------------------------------
# can_handle
# --------------------------------------------------------------------------


def test_can_handle_workflow_like():
    assert LlamaIndexAdapter().can_handle(FakeWorkflow()) is True


def test_can_handle_rejects_plain_object():
    assert LlamaIndexAdapter().can_handle(object()) is False


# --------------------------------------------------------------------------
# attach / detach (real dispatcher)
# --------------------------------------------------------------------------


def _gov_handlers() -> list:
    return [
        h
        for h in get_dispatcher().event_handlers
        if isinstance(h, GovernanceEventHandler)
    ]


def test_attach_registers_handler_then_detach_removes():
    adapter = LlamaIndexAdapter()
    agent = FakeWorkflow()
    try:
        returned = adapter.attach(agent, agent_id="x", session_id="s", evaluator=FakeEvaluator())
        assert returned is agent
        assert len(_gov_handlers()) == 1
    finally:
        adapter.detach(agent)
    assert _gov_handlers() == []


def test_attach_is_idempotent():
    adapter = LlamaIndexAdapter()
    agent = FakeWorkflow()
    ev = FakeEvaluator()
    try:
        adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
        adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
        assert len(_gov_handlers()) == 1
    finally:
        adapter.detach(agent)


# --------------------------------------------------------------------------
# event routing through the handler
# --------------------------------------------------------------------------


def test_handler_routes_llm_chat_start_to_before_model():
    ev = FakeEvaluator()
    h = _handler(ev)
    event = LLMChatStartEvent(
        messages=[ChatMessage(role="user", content="old"),
                  ChatMessage(role="user", content="the question")],
        additional_kwargs={},
        model_dict={},
    )
    h.handle(event)
    hook, kwargs = ev.calls[-1]
    assert hook == "before_model"
    assert kwargs["model_input"] == "the question"  # latest only


def test_handler_routes_llm_chat_end_to_after_model():
    ev = FakeEvaluator()
    h = _handler(ev)
    event = LLMChatEndEvent(
        messages=[ChatMessage(role="user", content="q")],
        response=ChatResponse(message=ChatMessage(role="assistant", content="the answer")),
    )
    h.handle(event)
    hook, kwargs = ev.calls[-1]
    assert hook == "after_model"
    assert kwargs["model_output"] == "the answer"


def test_handler_routes_tool_call():
    ev = FakeEvaluator()
    h = _handler(ev)
    event = AgentToolCallEvent(
        tool=ToolMetadata(description="d", name="transfer"),
        arguments='{"amount": 50}',
    )
    h.handle(event)
    hook, kwargs = ev.calls[-1]
    assert hook == "tool_call"
    assert kwargs["tool_name"] == "transfer"
    assert kwargs["tool_args"] == {"amount": 50}
    assert kwargs["session_state"]["tool_calls"] == 1


def test_handler_ignores_unrelated_events():
    ev = FakeEvaluator()
    h = _handler(ev)
    h.handle(object())  # not a governance-relevant event
    assert ev.calls == []


# --------------------------------------------------------------------------
# text / arg extraction
# --------------------------------------------------------------------------


def test_before_model_caps_text():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    cb.before_model([ChatMessage(role="user", content=huge)])
    assert len(ev.calls[-1][1]["model_input"]) <= _BEFORE_MODEL_TEXT_CAP


def test_before_model_empty():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.before_model([])
    assert ev.calls[-1][1]["model_input"] == ""


def test_coerce_args_json_string():
    assert _coerce_args('{"a": 1}') == {"a": 1}


def test_coerce_args_dict_passthrough():
    assert _coerce_args({"a": 1}) == {"a": 1}


def test_coerce_args_none_and_bad():
    assert _coerce_args(None) == {}
    assert _coerce_args("not json") == {}


# --------------------------------------------------------------------------
# enforcement semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook,invoke",
    [
        ("before_model", lambda cb: cb.before_model([ChatMessage(role="user", content="hi")])),
        ("after_model", lambda cb: cb.after_model(ChatResponse(message=ChatMessage(role="assistant", content="o")))),
        ("tool_call", lambda cb: cb.tool_call(ToolMetadata(description="d", name="t"), "{}")),
    ],
)
def test_block_exception_propagates(hook, invoke):
    cb = _make_callbacks(FakeEvaluator(block_on=hook))
    with pytest.raises(GovernanceBlockException):
        invoke(cb)


def test_non_block_exception_is_swallowed(caplog):
    class Boom:
        def evaluate_before_model(self, **_: Any) -> None:
            raise RuntimeError("evaluator bug")

    cb = GovernanceCallbacks(evaluator=Boom(), agent_name="a", session_id="s")  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        cb.before_model([ChatMessage(role="user", content="x")])
    assert any("governance check failed" in r.message for r in caplog.records)