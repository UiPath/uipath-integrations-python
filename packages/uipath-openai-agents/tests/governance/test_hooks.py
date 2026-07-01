"""Unit tests for the OpenAI Agents governance hooks.

``can_handle`` is tested against a real ``agents.Agent``; everything else
duck-types the OpenAI Agents payloads (response input/output items, tools)
with lightweight fakes so the real code paths are exercised without a live
LLM. ``GovernanceAgentHooks`` subclasses ``agents.AgentHooks`` (the SDK
type-checks ``agent.hooks``), so importing the adapter requires
``openai-agents`` either way.

The package is configured with ``asyncio_mode = "auto"``, so ``async def``
tests run without an explicit marker.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest
from agents import Agent
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_openai_agents.governance.hooks import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceAgentHooks,
    _content_text,
    _item_text,
    _latest_input_text,
    _model_response_text,
    _stringify,
    install_governance,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeEvaluator:
    """Records evaluate_* calls; optionally BLOCKs on a named hook."""

    def __init__(self, block_on: str | None = None) -> None:
        self.block_on = block_on
        self.calls: List[tuple[str, dict[str, Any]]] = []

    def _record(self, hook: str, **kwargs: Any) -> None:
        self.calls.append((hook, kwargs))
        if self.block_on == hook:
            raise GovernanceBlockException("blocked")  # type: ignore[call-arg]

    def evaluate_before_agent(self, *args: Any, **kwargs: Any) -> Any:
        self._record("before_agent", **kwargs)

    def evaluate_after_agent(self, *args: Any, **kwargs: Any) -> Any:
        self._record("after_agent", **kwargs)

    def evaluate_before_model(self, *args: Any, **kwargs: Any) -> Any:
        self._record("before_model", **kwargs)

    def evaluate_after_model(self, *args: Any, **kwargs: Any) -> Any:
        self._record("after_model", **kwargs)

    def evaluate_tool_call(self, *args: Any, **kwargs: Any) -> Any:
        self._record("tool_call", **kwargs)

    def evaluate_after_tool(self, *args: Any, **kwargs: Any) -> Any:
        self._record("after_tool", **kwargs)


class FakeAgent(Agent):  # type: ignore[type-arg]
    """A real ``agents.Agent`` — the graph walk isinstance-checks ``Agent``, so
    a bare duck-typed stand-in would be (correctly) skipped by
    ``install_governance``. Subclassing keeps the construction lightweight while
    remaining a genuine ``Agent`` instance."""

    def __init__(self, name: str = "agent", handoffs: List[Any] | None = None):
        super().__init__(name=name, handoffs=handoffs or [])


class FakeTool:
    def __init__(self, name: str):
        self.name = name


class RecordingHooks:
    """A user-supplied AgentHooks-like object that records delegated calls."""

    def __init__(self) -> None:
        self.seen: List[str] = []

    async def on_llm_start(self, *_a: Any) -> None:
        self.seen.append("on_llm_start")

    async def on_llm_end(self, *_a: Any) -> None:
        self.seen.append("on_llm_end")

    async def on_tool_start(self, *_a: Any) -> None:
        self.seen.append("on_tool_start")

    async def on_tool_end(self, *_a: Any) -> None:
        self.seen.append("on_tool_end")


def _msg(text: str, role: str = "user") -> dict[str, Any]:
    """A response input item carrying plain string content."""
    return {"role": role, "content": text}


def _msg_parts(*texts: str, role: str = "user") -> dict[str, Any]:
    """A response input item carrying a list of text parts."""
    return {"role": role, "content": [{"type": "input_text", "text": t} for t in texts]}


def _function_call(name: str, arguments: str) -> dict[str, Any]:
    return {"type": "function_call", "name": name, "arguments": arguments}


def _output_message(*texts: str) -> SimpleNamespace:
    """A ModelResponse output message item with text parts."""
    parts = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(role="assistant", content=parts)


def _make_hooks(evaluator: FakeEvaluator, inner: Any = None) -> GovernanceAgentHooks:
    return GovernanceAgentHooks(
        evaluator=evaluator, agent_name="agent-1", session_id="sess-1", inner=inner
    )


# --------------------------------------------------------------------------
# install_governance
# --------------------------------------------------------------------------


def test_install_governance_installs_on_all_agents_in_handoff_graph():
    leaf_a = FakeAgent("a")
    leaf_b = FakeAgent("b")
    root = FakeAgent("root", handoffs=[leaf_a, leaf_b])

    returned = install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")

    assert returned is root  # original returned, not a proxy
    for node in (root, leaf_a, leaf_b):
        assert isinstance(node.hooks, GovernanceAgentHooks)


def test_install_governance_follows_handoff_wrapper_objects():
    target = FakeAgent("target")
    handoff = SimpleNamespace(agent=target)  # Handoff-shaped wrapper
    root = FakeAgent("root", handoffs=[handoff])
    install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")
    assert isinstance(target.hooks, GovernanceAgentHooks)


def test_install_governance_is_idempotent():
    agent = FakeAgent()
    ev = FakeEvaluator()
    install_governance(agent, ev, agent_name="x", session_id="s")
    first = agent.hooks
    install_governance(agent, ev, agent_name="x", session_id="s")
    assert agent.hooks is first  # not re-wrapped


def test_install_governance_chains_existing_hooks():
    agent = FakeAgent()
    user_hooks = RecordingHooks()
    agent.hooks = user_hooks  # type: ignore[assignment]  # test double, not a real AgentHooks
    install_governance(agent, FakeEvaluator(), agent_name="x", session_id="s")
    assert isinstance(agent.hooks, GovernanceAgentHooks)
    assert agent.hooks._inner is user_hooks


_HOOKS_LOGGER = "uipath_openai_agents.governance.hooks"


@contextmanager
def _capture_hooks_logs(caplog: Any) -> Iterator[None]:
    """Attach caplog's handler straight to the hooks logger.

    Some sibling suites configure an ancestor ``uipath*`` logger with
    ``propagate=False``, which silently breaks caplog's default root-handler
    capture. Attaching directly to the target logger is propagation-independent.
    """
    logger = logging.getLogger(_HOOKS_LOGGER)
    logger.addHandler(caplog.handler)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(prev)


def test_install_governance_warns_when_no_agent(caplog):
    with _capture_hooks_logs(caplog):
        install_governance(object(), FakeEvaluator(), agent_name="x", session_id="s")  # type: ignore[arg-type]
    assert any("no Agent" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# on_llm_start (BEFORE_MODEL)
# --------------------------------------------------------------------------


async def test_on_llm_start_scopes_to_latest_item():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    items = [_msg("OLD turn — secret leak here"), _msg("the new question")]
    await cb.on_llm_start(None, FakeAgent(), "system", items)
    hook, kwargs = ev.calls[-1]
    assert hook == "before_model"
    assert kwargs["model_input"] == "the new question"
    assert "OLD turn" not in kwargs["model_input"]


async def test_on_llm_start_extracts_list_parts():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_llm_start(None, FakeAgent(), None, [_msg_parts("part one", "part two")])
    out = ev.calls[-1][1]["model_input"]
    assert "part one" in out and "part two" in out


async def test_on_llm_start_extracts_function_call_when_latest():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    items = [_function_call("lookup", '{"balance": "1000"}')]
    await cb.on_llm_start(None, FakeAgent(), None, items)
    out = ev.calls[-1][1]["model_input"]
    assert "lookup" in out and "1000" in out


async def test_on_llm_start_caps_text():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    await cb.on_llm_start(None, FakeAgent(), None, [_msg(huge)])
    assert len(ev.calls[-1][1]["model_input"]) <= _BEFORE_MODEL_TEXT_CAP


async def test_on_llm_start_empty_input():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_llm_start(None, FakeAgent(), None, [])
    assert ev.calls[-1][1]["model_input"] == ""


# --------------------------------------------------------------------------
# on_llm_end (AFTER_MODEL)
# --------------------------------------------------------------------------


async def test_on_llm_end_extracts_text_and_function_call():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    response = SimpleNamespace(
        output=[
            _output_message("thinking"),
            SimpleNamespace(
                type="function_call",
                name="submit_answer",
                arguments='{"content": "final reply"}',
            ),
        ]
    )
    await cb.on_llm_end(None, FakeAgent(), response)
    out = ev.calls[-1][1]["model_output"]
    assert "thinking" in out and "submit_answer" in out and "final reply" in out


async def test_on_llm_end_empty_response():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_llm_end(None, FakeAgent(), SimpleNamespace(output=[]))
    assert ev.calls[-1][1]["model_output"] == ""


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


async def test_on_tool_start_passes_name_and_session_state():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_tool_start(None, FakeAgent(), FakeTool("transfer"))
    hook, kwargs = ev.calls[-1]
    assert hook == "tool_call"
    assert kwargs["tool_name"] == "transfer"
    assert kwargs["tool_args"] == {}  # OpenAI SDK does not surface args here
    assert kwargs["session_state"]["tool_calls"] == 1


async def test_on_tool_end_stringifies_dict_result():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_tool_end(None, FakeAgent(), FakeTool("lookup"), {"x": 1})
    out = ev.calls[-1][1]["tool_result"]
    assert "x" in out and "1" in out


async def test_on_tool_end_none_result():
    ev = FakeEvaluator()
    cb = _make_hooks(ev)
    await cb.on_tool_end(None, FakeAgent(), FakeTool("noop"), None)
    assert ev.calls[-1][1]["tool_result"] == ""


async def test_reports_live_agent_name_not_install_time_name():
    """After a handoff the executing agent differs from the graph entrypoint
    the factory named us with; governance should attribute the live agent."""
    ev = FakeEvaluator()
    cb = _make_hooks(ev)  # install-time name is "agent-1"
    await cb.on_llm_start(None, FakeAgent("billing_specialist"), None, [_msg("hi")])
    assert ev.calls[-1][1]["agent_name"] == "billing_specialist"


async def test_blocked_call_does_not_increment_counter():
    """A DENY raises before the counter bump, so the count is not inflated."""
    ev = FakeEvaluator(block_on="tool_call")
    cb = _make_hooks(ev)
    with pytest.raises(GovernanceBlockException):
        await cb.on_tool_start(None, FakeAgent(), FakeTool("t"))
    # evaluator saw the pre-call count (0) and the block prevented the bump
    assert ev.calls[-1][1]["session_state"]["tool_calls"] == 0
    assert cb._session_state["tool_calls"] == 0


# --------------------------------------------------------------------------
# chaining to user hooks
# --------------------------------------------------------------------------


async def test_governance_delegates_to_inner_hooks():
    inner = RecordingHooks()
    cb = _make_hooks(FakeEvaluator(), inner=inner)
    await cb.on_llm_start(None, FakeAgent(), None, [_msg("hi")])
    await cb.on_llm_end(None, FakeAgent(), SimpleNamespace(output=[]))
    await cb.on_tool_start(None, FakeAgent(), FakeTool("t"))
    await cb.on_tool_end(None, FakeAgent(), FakeTool("t"), {})
    assert inner.seen == ["on_llm_start", "on_llm_end", "on_tool_start", "on_tool_end"]


# --------------------------------------------------------------------------
# enforcement semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook,invoke",
    [
        (
            "before_model",
            lambda cb: cb.on_llm_start(None, FakeAgent(), None, [_msg("hi")]),
        ),
        (
            "after_model",
            lambda cb: cb.on_llm_end(None, FakeAgent(), SimpleNamespace(output=[])),
        ),
        ("tool_call", lambda cb: cb.on_tool_start(None, FakeAgent(), FakeTool("t"))),
        (
            "after_tool",
            lambda cb: cb.on_tool_end(None, FakeAgent(), FakeTool("t"), {"r": 1}),
        ),
    ],
)
async def test_block_exception_propagates(hook, invoke):
    cb = _make_hooks(FakeEvaluator(block_on=hook))
    with pytest.raises(GovernanceBlockException):
        await invoke(cb)


async def test_non_block_exception_is_swallowed(caplog):
    class Boom:
        def evaluate_before_model(self, **_: Any) -> None:
            raise RuntimeError("evaluator bug")

    cb = GovernanceAgentHooks(
        evaluator=Boom(),  # type: ignore[arg-type]
        agent_name="a",
        session_id="s",
    )
    with _capture_hooks_logs(caplog):
        # must NOT raise — a governance bug can't break the agent run
        await cb.on_llm_start(None, FakeAgent(), None, [_msg("x")])
    assert any("governance check failed" in r.message for r in caplog.records)


async def test_hooks_return_none():
    # hooks are pass-through (return None) — they never short-circuit the run.
    # (the inline type: ignores below silence mypy's func-returns-value on the
    # None-returning hooks; the runtime assert documents the contract.)
    cb = _make_hooks(FakeEvaluator())
    assert await cb.on_llm_start(None, FakeAgent(), None, []) is None  # type: ignore[func-returns-value]
    assert await cb.on_llm_end(None, FakeAgent(), SimpleNamespace(output=[])) is None  # type: ignore[func-returns-value]
    assert await cb.on_tool_start(None, FakeAgent(), FakeTool("t")) is None  # type: ignore[func-returns-value]
    assert await cb.on_tool_end(None, FakeAgent(), FakeTool("t"), {}) is None  # type: ignore[func-returns-value]


# --------------------------------------------------------------------------
# coverage: swallow paths on every hook, boundary delegation, extraction edges
# --------------------------------------------------------------------------


class _Boom:
    """Evaluator whose every evaluate_* raises a non-block error."""

    def __getattr__(self, _name: str) -> Any:
        def _raise(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("evaluator bug")

        return _raise


@pytest.mark.parametrize(
    "invoke",
    [
        lambda cb: cb.on_llm_end(None, FakeAgent(), SimpleNamespace(output=[])),
        lambda cb: cb.on_tool_start(None, FakeAgent(), FakeTool("t")),
        lambda cb: cb.on_tool_end(None, FakeAgent(), FakeTool("t"), {"r": 1}),
    ],
)
async def test_model_and_tool_hooks_swallow_non_block_errors(invoke, caplog):
    cb = GovernanceAgentHooks(evaluator=_Boom(), agent_name="a", session_id="s")  # type: ignore[arg-type]
    with _capture_hooks_logs(caplog):
        await invoke(cb)  # must NOT raise — a governance bug can't break the run
    assert any("governance check failed" in r.message for r in caplog.records)


class _InnerBoundary:
    def __init__(self) -> None:
        self.seen: List[str] = []

    async def on_start(self, *_a: Any) -> None:
        self.seen.append("on_start")

    async def on_end(self, *_a: Any) -> None:
        self.seen.append("on_end")

    async def on_handoff(self, *_a: Any) -> None:
        self.seen.append("on_handoff")


async def test_boundary_hooks_delegate_to_inner():
    inner = _InnerBoundary()
    cb = _make_hooks(FakeEvaluator(), inner=inner)
    await cb.on_start(None, FakeAgent())
    await cb.on_end(None, FakeAgent(), "out")
    await cb.on_handoff(None, FakeAgent(), FakeAgent())
    assert inner.seen == ["on_start", "on_end", "on_handoff"]


async def test_delegate_swallows_inner_hook_error(caplog):
    class _BadInner:
        async def on_llm_start(self, *_a: Any) -> None:
            raise RuntimeError("inner boom")

    cb = _make_hooks(FakeEvaluator(), inner=_BadInner())
    with _capture_hooks_logs(caplog):
        await cb.on_llm_start(None, FakeAgent(), None, [_msg("x")])  # must not raise
    assert any("chained user hook" in r.message for r in caplog.records)


def test_extraction_edges():
    # _stringify: str passthrough; circular ref → str() fallback (not a crash)
    assert _stringify("hi") == "hi"
    circular: dict[str, Any] = {}
    circular["self"] = circular
    assert isinstance(_stringify(circular), str)
    # _item_text: tool-result output-only item
    assert "42" in _item_text({"output": {"balance": 42}})
    # _content_text: object exposing .text, and a bare-string part in a list
    assert _content_text(SimpleNamespace(text="hello")) == "hello"
    assert "raw" in _content_text(["raw", {"text": "block"}])
    # _model_response_text: response with no .output → falls back to item text
    assert _model_response_text(SimpleNamespace(content="direct")) == "direct"
    # _latest_input_text: single (non-list) item
    assert _latest_input_text(_msg("solo")) == "solo"


# --------------------------------------------------------------------------
# Factory wiring — the evaluator kwarg drives install_governance
# --------------------------------------------------------------------------


def _factory_without_init():
    """A factory instance that skips __init__ (avoids SDK instrumentation)."""
    from uipath_openai_agents.runtime.factory import UiPathOpenAIAgentRuntimeFactory

    return UiPathOpenAIAgentRuntimeFactory.__new__(UiPathOpenAIAgentRuntimeFactory)


async def test_factory_installs_governance_when_evaluator_supplied(monkeypatch):
    from uipath_openai_agents.runtime import factory as factory_mod

    # Stub the runtime so we don't introspect a real Agent.
    monkeypatch.setattr(
        factory_mod, "UiPathOpenAIAgentRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    agent = FakeAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e", evaluator=FakeEvaluator()
    )
    assert isinstance(agent.hooks, GovernanceAgentHooks)


async def test_factory_skips_governance_without_evaluator(monkeypatch):
    from uipath_openai_agents.runtime import factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "UiPathOpenAIAgentRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    agent = FakeAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e"
    )
    assert agent.hooks is None
