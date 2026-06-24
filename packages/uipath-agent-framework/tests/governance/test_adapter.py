"""Unit tests for the Microsoft Agent Framework governance adapter.

The middleware classes subclass ``agent_framework`` base classes (the
framework routes middleware by ``isinstance``), so importing the adapter
requires ``agent-framework-core`` — but the messages / responses / tools /
contexts under test are lightweight duck-typed fakes.

The package is configured with ``asyncio_mode = "auto"``, so ``async def``
tests run without an explicit marker.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_agent_framework.governance.adapter import (
    _BEFORE_MODEL_TEXT_CAP,
    AgentFrameworkAdapter,
    GovernanceCallbacks,
    GovernanceChatMiddleware,
    GovernanceFunctionMiddleware,
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


class FakeAgent:
    """Minimal stand-in for an ``agent_framework`` Agent (duck-typed)."""

    def __init__(self, name: str = "agent"):
        self.name = name
        self.middleware: Any = None

    async def run(self, *_a: Any, **_k: Any) -> None:  # marks it as an agent
        return None


class FakeWorkflowAgent:
    """Stand-in for ``WorkflowAgent`` exposing inner agents via executors."""

    def __init__(self, inner_agents: List[Any]):
        self.middleware: Any = None
        executors = {
            f"e{i}": SimpleNamespace(_agent=a) for i, a in enumerate(inner_agents)
        }
        self.workflow = SimpleNamespace(executors=executors)


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def _msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _make_callbacks(ev: FakeEvaluator) -> GovernanceCallbacks:
    return GovernanceCallbacks(evaluator=ev, agent_name="agent-1", session_id="sess-1")


async def _noop_next() -> None:
    return None


# --------------------------------------------------------------------------
# can_handle
# --------------------------------------------------------------------------


def test_can_handle_real_agent():
    from agent_framework import BaseAgent

    assert AgentFrameworkAdapter().can_handle(BaseAgent(name="t")) is True


def test_can_handle_rejects_non_agent():
    # Duck-typed look-alikes (middleware + run/workflow) must NOT be claimed —
    # only a real agent_framework BaseAgent is.
    assert AgentFrameworkAdapter().can_handle(FakeAgent()) is False
    assert AgentFrameworkAdapter().can_handle(FakeWorkflowAgent([])) is False
    assert AgentFrameworkAdapter().can_handle(object()) is False


# --------------------------------------------------------------------------
# attach / detach
# --------------------------------------------------------------------------


def test_attach_appends_both_middleware():
    agent = FakeAgent()
    returned = AgentFrameworkAdapter().attach(
        agent, agent_id="x", session_id="s", evaluator=FakeEvaluator()
    )
    assert returned is agent
    kinds = [type(m) for m in agent.middleware]
    assert GovernanceChatMiddleware in kinds
    assert GovernanceFunctionMiddleware in kinds


def test_attach_installs_on_workflow_inner_agents():
    a, b = FakeAgent("a"), FakeAgent("b")
    root = FakeWorkflowAgent([a, b])
    AgentFrameworkAdapter().attach(root, agent_id="x", session_id="s", evaluator=FakeEvaluator())
    for node in (a, b):
        assert any(isinstance(m, GovernanceChatMiddleware) for m in node.middleware)


def test_attach_is_idempotent():
    agent = FakeAgent()
    adapter = AgentFrameworkAdapter()
    ev = FakeEvaluator()
    adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
    adapter.attach(agent, agent_id="x", session_id="s", evaluator=ev)
    assert sum(isinstance(m, GovernanceChatMiddleware) for m in agent.middleware) == 1


def test_attach_preserves_existing_middleware_and_runs_governance_first():
    user_mw = object()
    agent = FakeAgent()
    agent.middleware = [user_mw]
    AgentFrameworkAdapter().attach(agent, agent_id="x", session_id="s", evaluator=FakeEvaluator())
    # governance prepended → runs first; user middleware preserved at the end
    assert isinstance(agent.middleware[0], GovernanceChatMiddleware)
    assert agent.middleware[-1] is user_mw


def test_detach_removes_governance_middleware():
    user_mw = object()
    agent = FakeAgent()
    agent.middleware = [user_mw]
    adapter = AgentFrameworkAdapter()
    adapter.attach(agent, agent_id="x", session_id="s", evaluator=FakeEvaluator())
    adapter.detach(agent)
    assert agent.middleware == [user_mw]


def test_attach_warns_when_no_agent(caplog):
    with caplog.at_level(logging.WARNING):
        AgentFrameworkAdapter().attach(
            object(), agent_id="x", session_id="s", evaluator=FakeEvaluator()
        )
    assert any("no agent" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# ChatMiddleware → BEFORE_MODEL / AFTER_MODEL
# --------------------------------------------------------------------------


async def test_chat_middleware_brackets_call_with_before_and_after():
    ev = FakeEvaluator()
    mw = GovernanceChatMiddleware(_make_callbacks(ev))
    order: List[str] = []

    async def call_next() -> None:
        order.append("model_call")

    context = SimpleNamespace(
        messages=[_msg("old"), _msg("the question")],
        result=SimpleNamespace(text="the answer"),
    )
    await mw.process(context, call_next)

    hooks = [h for h, _ in ev.calls]
    assert hooks == ["before_model", "after_model"]
    assert order == ["model_call"]
    assert ev.calls[0][1]["model_input"] == "the question"  # latest only
    assert ev.calls[1][1]["model_output"] == "the answer"


async def test_chat_middleware_caps_text():
    ev = FakeEvaluator()
    mw = GovernanceChatMiddleware(_make_callbacks(ev))
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    context = SimpleNamespace(messages=[_msg(huge)], result=SimpleNamespace(text=""))
    await mw.process(context, _noop_next)
    assert len(ev.calls[0][1]["model_input"]) <= _BEFORE_MODEL_TEXT_CAP


# --------------------------------------------------------------------------
# FunctionMiddleware → TOOL_CALL / AFTER_TOOL
# --------------------------------------------------------------------------


async def test_function_middleware_passes_name_args_and_result():
    ev = FakeEvaluator()
    mw = GovernanceFunctionMiddleware(_make_callbacks(ev))
    order: List[str] = []

    async def call_next() -> None:
        order.append("tool_call")

    context = SimpleNamespace(
        function=FakeTool("transfer"),
        arguments={"amount": 50},
        result={"status": "ok"},
    )
    await mw.process(context, call_next)

    hooks = [h for h, _ in ev.calls]
    assert hooks == ["tool_call", "after_tool"]
    assert order == ["tool_call"]
    assert ev.calls[0][1]["tool_name"] == "transfer"
    assert ev.calls[0][1]["tool_args"] == {"amount": 50}
    assert ev.calls[0][1]["session_state"]["tool_calls"] == 1
    assert "ok" in ev.calls[1][1]["tool_result"]


async def test_function_middleware_coerces_pydantic_args():
    ev = FakeEvaluator()
    mw = GovernanceFunctionMiddleware(_make_callbacks(ev))
    args = SimpleNamespace(model_dump=lambda: {"x": 1})
    context = SimpleNamespace(function=FakeTool("t"), arguments=args, result=None)
    await mw.process(context, _noop_next)
    assert ev.calls[0][1]["tool_args"] == {"x": 1}
    assert ev.calls[1][1]["tool_result"] == ""  # None result → ""


# --------------------------------------------------------------------------
# enforcement semantics
# --------------------------------------------------------------------------


async def test_block_in_before_model_aborts_before_call_next():
    ev = FakeEvaluator(block_on="before_model")
    mw = GovernanceChatMiddleware(_make_callbacks(ev))
    called = {"next": False}

    async def call_next() -> None:
        called["next"] = True

    context = SimpleNamespace(messages=[_msg("hi")], result=None)
    with pytest.raises(GovernanceBlockException):
        await mw.process(context, call_next)
    assert called["next"] is False  # tool/model never ran


async def test_block_in_before_tool_aborts_before_call_next():
    ev = FakeEvaluator(block_on="tool_call")
    mw = GovernanceFunctionMiddleware(_make_callbacks(ev))
    called = {"next": False}

    async def call_next() -> None:
        called["next"] = True

    context = SimpleNamespace(function=FakeTool("t"), arguments={}, result=None)
    with pytest.raises(GovernanceBlockException):
        await mw.process(context, call_next)
    assert called["next"] is False


async def test_non_block_exception_is_swallowed(caplog):
    class Boom:
        def evaluate_before_model(self, **_: Any) -> None:
            raise RuntimeError("evaluator bug")

    cb = GovernanceCallbacks(evaluator=Boom(), agent_name="a", session_id="s")  # type: ignore[arg-type]
    mw = GovernanceChatMiddleware(cb)
    with caplog.at_level(logging.WARNING):
        await mw.process(SimpleNamespace(messages=[_msg("x")], result=None), _noop_next)
    assert any("governance check failed" in r.message for r in caplog.records)