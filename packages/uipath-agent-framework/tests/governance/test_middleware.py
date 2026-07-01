"""Unit tests for the Microsoft Agent Framework governance middleware.

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

from uipath_agent_framework.governance.middleware import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
    GovernanceChatMiddleware,
    GovernanceFunctionMiddleware,
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
# install_governance
# --------------------------------------------------------------------------


def test_install_governance_appends_both_middleware():
    agent = FakeAgent()
    returned = install_governance(
        agent, FakeEvaluator(), agent_name="x", session_id="s"
    )
    assert returned is agent
    kinds = [type(m) for m in agent.middleware]
    assert GovernanceChatMiddleware in kinds
    assert GovernanceFunctionMiddleware in kinds


def test_install_governance_installs_on_workflow_inner_agents():
    a, b = FakeAgent("a"), FakeAgent("b")
    root = FakeWorkflowAgent([a, b])
    install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")
    for node in (a, b):
        assert any(isinstance(m, GovernanceChatMiddleware) for m in node.middleware)


def test_install_governance_recurses_into_nested_workflow():
    """A WorkflowAgent inside a WorkflowAgent (workflow-of-workflows): the deep
    leaf agent must still be governed, not left one level below the walk."""
    leaf = FakeAgent("leaf")
    inner = FakeWorkflowAgent([leaf])
    root = FakeWorkflowAgent([inner])
    install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")
    assert any(isinstance(m, GovernanceChatMiddleware) for m in leaf.middleware)


def test_install_governance_is_cycle_safe():
    """A workflow whose executor points back at itself must not loop forever."""
    w = FakeWorkflowAgent([])
    w.workflow.executors = {"self": SimpleNamespace(_agent=w)}
    # completes (id-visited set breaks the cycle) and governs w exactly once
    install_governance(w, FakeEvaluator(), agent_name="x", session_id="s")
    assert sum(isinstance(m, GovernanceChatMiddleware) for m in w.middleware) == 1


def test_install_governance_is_idempotent():
    agent = FakeAgent()
    ev = FakeEvaluator()
    install_governance(agent, ev, agent_name="x", session_id="s")
    install_governance(agent, ev, agent_name="x", session_id="s")
    assert sum(isinstance(m, GovernanceChatMiddleware) for m in agent.middleware) == 1


def test_install_governance_preserves_existing_middleware_and_runs_first():
    user_mw = object()
    agent = FakeAgent()
    agent.middleware = [user_mw]
    install_governance(agent, FakeEvaluator(), agent_name="x", session_id="s")
    # governance prepended → runs first; user middleware preserved at the end
    assert isinstance(agent.middleware[0], GovernanceChatMiddleware)
    assert agent.middleware[-1] is user_mw


def test_install_governance_warns_when_no_agent(caplog):
    with caplog.at_level(logging.WARNING):
        install_governance(object(), FakeEvaluator(), agent_name="x", session_id="s")
    assert any("no agent" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Factory wiring — the evaluator kwarg drives install_governance
# --------------------------------------------------------------------------


def _factory_without_init():
    """A factory instance that skips __init__ (avoids config/IO)."""
    from uipath_agent_framework.runtime.factory import (
        UiPathAgentFrameworkRuntimeFactory,
    )

    return UiPathAgentFrameworkRuntimeFactory.__new__(
        UiPathAgentFrameworkRuntimeFactory
    )


def _stub_factory_runtime(monkeypatch, factory_mod):
    """Stub storage + runtime constructions so only the governance branch runs."""
    monkeypatch.setattr(factory_mod, "ScopedCheckpointStorage", lambda *a, **k: None)
    monkeypatch.setattr(
        factory_mod, "UiPathAgentFrameworkRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        factory_mod, "UiPathResumableRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(factory_mod, "UiPathResumeTriggerHandler", lambda *a, **k: None)

    async def _storage(self):
        return SimpleNamespace(checkpoint_storage=object())

    monkeypatch.setattr(
        factory_mod.UiPathAgentFrameworkRuntimeFactory, "_get_storage", _storage
    )


async def test_factory_installs_governance_when_evaluator_supplied(monkeypatch):
    from uipath_agent_framework.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    agent = FakeAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e", evaluator=FakeEvaluator()
    )
    assert any(isinstance(m, GovernanceChatMiddleware) for m in agent.middleware)


async def test_factory_skips_governance_without_evaluator(monkeypatch):
    from uipath_agent_framework.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    agent = FakeAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e"
    )
    assert agent.middleware is None


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


async def test_after_model_runs_even_when_model_call_raises():
    """AFTER_MODEL must fire from the finally so audit/rules still observe the
    turn, and the underlying error must still propagate."""
    ev = FakeEvaluator()
    mw = GovernanceChatMiddleware(_make_callbacks(ev))

    async def boom_next() -> None:
        raise RuntimeError("model exploded")

    context = SimpleNamespace(messages=[_msg("hi")], result=SimpleNamespace(text=""))
    with pytest.raises(RuntimeError, match="model exploded"):
        await mw.process(context, boom_next)
    assert [h for h, _ in ev.calls] == ["before_model", "after_model"]


async def test_streaming_governs_finalized_response_via_result_hook():
    """Streaming: context.result is a ResponseStream after call_next, so
    AFTER_MODEL runs from a stream_result_hook on the finalized ChatResponse."""
    ev = FakeEvaluator()
    mw = GovernanceChatMiddleware(_make_callbacks(ev))
    context = SimpleNamespace(
        messages=[_msg("the question")],
        stream=True,
        stream_result_hooks=[],
        result=None,
    )
    await mw.process(context, _noop_next)

    # BEFORE_MODEL fired; AFTER_MODEL deferred to the registered hook.
    assert [h for h, _ in ev.calls] == ["before_model"]
    assert len(context.stream_result_hooks) == 1

    finalized = SimpleNamespace(text="the streamed answer")
    returned = context.stream_result_hooks[0](finalized)
    assert returned is finalized  # hook returns the response unchanged
    assert [h for h, _ in ev.calls] == ["before_model", "after_model"]
    assert ev.calls[-1][1]["model_output"] == "the streamed answer"


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


async def test_after_tool_runs_even_when_tool_call_raises():
    ev = FakeEvaluator()
    mw = GovernanceFunctionMiddleware(_make_callbacks(ev))

    async def boom_next() -> None:
        raise RuntimeError("tool exploded")

    context = SimpleNamespace(function=FakeTool("t"), arguments={}, result=None)
    with pytest.raises(RuntimeError, match="tool exploded"):
        await mw.process(context, boom_next)
    assert [h for h, _ in ev.calls] == ["tool_call", "after_tool"]


def test_blocked_before_tool_does_not_increment_counter():
    """A DENY raises before the counter bump, so the count is not inflated."""
    ev = FakeEvaluator(block_on="tool_call")
    cb = _make_callbacks(ev)
    with pytest.raises(GovernanceBlockException):
        cb.before_tool(FakeTool("t"), {})
    assert ev.calls[-1][1]["session_state"]["tool_calls"] == 0
    assert cb._session_state["tool_calls"] == 0


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
