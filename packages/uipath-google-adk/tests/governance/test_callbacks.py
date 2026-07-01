"""Unit tests for the Google ADK governance callbacks.

``can_handle`` is tested against a real ``google.adk`` ``LlmAgent`` (the
adapter detects agents with ``isinstance(..., BaseAgent)``). The remaining
tests duck-type the ADK payloads — lightweight fakes for ``Part`` /
``Content`` / ``LlmRequest`` / ``LlmResponse`` / tool / agent — so the
callback code paths are exercised without driving the heavy ADK runtime.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_google_adk.governance.callbacks import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
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


class FakeLlmAgent:
    """Minimal stand-in for ``google.adk.agents.LlmAgent``."""

    def __init__(
        self,
        name: str = "agent",
        sub_agents: List[Any] | None = None,
        tools: List[Any] | None = None,
    ):
        self.name = name
        self.before_model_callback: Any = None
        self.after_model_callback: Any = None
        self.before_tool_callback: Any = None
        self.after_tool_callback: Any = None
        self.sub_agents = sub_agents or []
        self.tools = tools or []


class FakeAgentTool:
    """Stand-in for ``google.adk.tools.agent_tool.AgentTool`` — wraps an agent."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.name = getattr(agent, "name", "agent_tool")


class FakeContainerAgent:
    """Container agent (Sequential/Parallel) with no model callbacks."""

    def __init__(self, name: str, sub_agents: List[Any]):
        self.name = name
        self.sub_agents = sub_agents


class FakeTool:
    def __init__(self, name: str):
        self.name = name


def _part(
    text: str | None = None,
    function_call: Any = None,
    function_response: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        function_call=function_call,
        function_response=function_response,
    )


def _content(parts: List[Any], role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(role=role, parts=parts)


def _make_callbacks(evaluator: FakeEvaluator) -> GovernanceCallbacks:
    return GovernanceCallbacks(
        evaluator=evaluator, agent_name="agent-1", session_id="sess-1"
    )


# --------------------------------------------------------------------------
# install_governance
# --------------------------------------------------------------------------


def test_install_governance_installs_on_all_llm_agents_in_tree():
    leaf_a = FakeLlmAgent("a")
    leaf_b = FakeLlmAgent("b")
    root = FakeContainerAgent("root", [leaf_a, leaf_b])

    returned = install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")

    assert returned is root  # original returned, not a proxy
    for leaf in (leaf_a, leaf_b):
        assert isinstance(leaf.before_model_callback, list)
        assert len(leaf.before_model_callback) == 1
        assert leaf.after_model_callback and leaf.before_tool_callback
        assert leaf.after_tool_callback
    # the container agent has no model-callback surface → must NOT be decorated
    assert not hasattr(root, "before_model_callback")


def test_install_governance_is_idempotent():
    agent = FakeLlmAgent()
    ev = FakeEvaluator()
    install_governance(agent, ev, agent_name="x", session_id="s")
    install_governance(agent, ev, agent_name="x", session_id="s")
    assert len(agent.before_model_callback) == 1


def test_install_governance_preserves_existing_callback_and_runs_first():
    def user_cb(*_a, **_k):
        return None

    agent = FakeLlmAgent()
    agent.before_model_callback = user_cb
    install_governance(agent, FakeEvaluator(), agent_name="x", session_id="s")
    cbs = agent.before_model_callback
    assert isinstance(cbs, list) and len(cbs) == 2
    # governance prepended → runs first
    assert getattr(cbs[0], "__self__", None).__class__ is GovernanceCallbacks
    assert cbs[1] is user_cb


def test_install_governance_warns_when_no_llm_agent(caplog):
    container = FakeContainerAgent("root", [])
    with caplog.at_level(logging.WARNING):
        install_governance(container, FakeEvaluator(), agent_name="x", session_id="s")
    assert any("no LlmAgent" in r.message for r in caplog.records)


def test_install_governance_follows_agent_tool_wrapped_agents():
    """An agent exposed to another agent via AgentTool lives in ``tools``, not
    ``sub_agents`` — it must still be governed."""
    wrapped = FakeLlmAgent("researcher")
    root = FakeLlmAgent("root", tools=[FakeAgentTool(wrapped)])
    install_governance(root, FakeEvaluator(), agent_name="x", session_id="s")
    assert isinstance(wrapped.before_model_callback, list)
    assert len(wrapped.before_model_callback) == 1


def test_install_governance_rebinds_session_on_cached_agent_reuse():
    """The factory caches agents by entrypoint; a second new_runtime reuses the
    same agent, so governance metadata must refresh to the new session."""
    agent = FakeLlmAgent()
    install_governance(agent, FakeEvaluator(), agent_name="a", session_id="session-1")
    gov = agent.before_model_callback[0].__self__
    assert gov._session_id == "session-1"

    ev2 = FakeEvaluator()
    install_governance(agent, ev2, agent_name="a", session_id="session-2")
    # same callback object, not re-stacked, but re-pointed at the new run
    assert len(agent.before_model_callback) == 1
    assert agent.before_model_callback[0].__self__ is gov
    assert gov._session_id == "session-2"
    assert gov._evaluator is ev2


# --------------------------------------------------------------------------
# Factory wiring — the evaluator kwarg drives install_governance
# --------------------------------------------------------------------------


class _FakeRuntime:
    APP_NAME = "app"
    USER_ID = "user"

    def __init__(self, **kw: Any) -> None:
        pass


class _FakeSessionService:
    async def get_session(self, **kw: Any) -> Any:
        return None

    async def create_session(self, **kw: Any) -> Any:
        return object()


def _factory_without_init():
    """A factory instance that skips __init__ (avoids config/IO)."""
    from uipath_google_adk.runtime.factory import UiPathGoogleADKRuntimeFactory

    return UiPathGoogleADKRuntimeFactory.__new__(UiPathGoogleADKRuntimeFactory)


def _stub_factory_runtime(monkeypatch, factory_mod):
    """Stub Runner + runtime + session service so only the governance branch runs."""
    monkeypatch.setattr(factory_mod, "Runner", lambda **kw: None)
    monkeypatch.setattr(factory_mod, "UiPathGoogleADKRuntime", _FakeRuntime)

    async def _session_service(self):
        return _FakeSessionService()

    monkeypatch.setattr(
        factory_mod.UiPathGoogleADKRuntimeFactory,
        "_get_session_service",
        _session_service,
    )


async def test_factory_installs_governance_when_evaluator_supplied(monkeypatch):
    from uipath_google_adk.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    agent = FakeLlmAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e", evaluator=FakeEvaluator()
    )
    assert isinstance(agent.before_model_callback, list)


async def test_factory_skips_governance_without_evaluator(monkeypatch):
    from uipath_google_adk.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    agent = FakeLlmAgent()
    await _factory_without_init()._create_runtime_instance(
        agent=agent, runtime_id="r", entrypoint="e"
    )
    assert agent.before_model_callback is None


# --------------------------------------------------------------------------
# before_model
# --------------------------------------------------------------------------


def test_before_model_scopes_to_latest_content():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    req = SimpleNamespace(
        contents=[
            _content([_part(text="OLD turn — secret leak here")]),
            _content([_part(text="the new question")]),
        ]
    )
    cb.before_model(callback_context=None, llm_request=req)
    hook, kwargs = ev.calls[-1]
    assert hook == "before_model"
    assert kwargs["model_input"] == "the new question"
    assert "OLD turn" not in kwargs["model_input"]


def test_before_model_extracts_function_response_when_latest():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    fr = SimpleNamespace(name="lookup", response={"balance": "1000"})
    req = SimpleNamespace(contents=[_content([_part(function_response=fr)])])
    cb.before_model(callback_context=None, llm_request=req)
    assert "1000" in ev.calls[-1][1]["model_input"]


def test_before_model_caps_text():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    req = SimpleNamespace(contents=[_content([_part(text=huge)])])
    cb.before_model(callback_context=None, llm_request=req)
    assert len(ev.calls[-1][1]["model_input"]) <= _BEFORE_MODEL_TEXT_CAP


def test_before_model_empty_contents():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.before_model(callback_context=None, llm_request=SimpleNamespace(contents=[]))
    assert ev.calls[-1][1]["model_input"] == ""


# --------------------------------------------------------------------------
# after_model
# --------------------------------------------------------------------------


def test_after_model_skips_partial():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    resp = SimpleNamespace(partial=True, content=_content([_part(text="chunk")]))
    cb.after_model(callback_context=None, llm_response=resp)
    assert ev.calls == []


def test_after_model_extracts_text_and_function_call():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    fc = SimpleNamespace(name="submit_answer", args={"content": "final reply"})
    resp = SimpleNamespace(
        partial=False,
        content=_content(
            [_part(text="thinking"), _part(function_call=fc)], role="model"
        ),
    )
    cb.after_model(callback_context=None, llm_response=resp)
    out = ev.calls[-1][1]["model_output"]
    assert "thinking" in out and "submit_answer" in out and "final reply" in out


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_before_tool_passes_args_and_session_state():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.before_tool(FakeTool("transfer"), {"amount": 50}, tool_context=None)
    hook, kwargs = ev.calls[-1]
    assert hook == "tool_call"
    assert kwargs["tool_name"] == "transfer"
    assert kwargs["tool_args"] == {"amount": 50}
    assert kwargs["session_state"]["tool_calls"] == 1


def test_before_tool_caps_huge_args():
    """A huge arg blob must not reach the evaluator uncapped (contrast with the
    small-args case, which passes through unchanged)."""
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    huge = "x" * (_BEFORE_MODEL_TEXT_CAP + 5000)
    cb.before_tool(FakeTool("t"), {"blob": huge}, tool_context=None)
    tool_args = ev.calls[-1][1]["tool_args"]
    assert set(tool_args) == {"_truncated"}
    assert len(tool_args["_truncated"]) <= _BEFORE_MODEL_TEXT_CAP


def test_after_tool_stringifies_dict_response():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.after_tool(FakeTool("lookup"), {}, tool_context=None, tool_response={"x": 1})
    out = ev.calls[-1][1]["tool_result"]
    assert "x" in out and "1" in out


def test_after_tool_none_response():
    ev = FakeEvaluator()
    cb = _make_callbacks(ev)
    cb.after_tool(FakeTool("noop"), {}, tool_context=None, tool_response=None)
    assert ev.calls[-1][1]["tool_result"] == ""


def test_blocked_tool_call_does_not_increment_counter():
    """A DENY raises before the counter bump, so the count is not inflated."""
    ev = FakeEvaluator(block_on="tool_call")
    cb = _make_callbacks(ev)
    with pytest.raises(GovernanceBlockException):
        cb.before_tool(FakeTool("t"), {}, tool_context=None)
    assert ev.calls[-1][1]["session_state"]["tool_calls"] == 0
    assert cb._session_state["tool_calls"] == 0


# --------------------------------------------------------------------------
# enforcement semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook,invoke",
    [
        (
            "before_model",
            lambda cb: cb.before_model(
                None, SimpleNamespace(contents=[_content([_part(text="hi")])])
            ),
        ),
        (
            "after_model",
            lambda cb: cb.after_model(
                None,
                SimpleNamespace(partial=False, content=_content([_part(text="o")])),
            ),
        ),
        ("tool_call", lambda cb: cb.before_tool(FakeTool("t"), {}, None)),
        (
            "after_tool",
            lambda cb: cb.after_tool(FakeTool("t"), {}, None, {"r": 1}),
        ),
    ],
)
def test_block_exception_propagates(hook, invoke):
    cb = _make_callbacks(FakeEvaluator(block_on=hook))
    with pytest.raises(GovernanceBlockException):
        invoke(cb)


def test_non_block_exception_is_swallowed(caplog):
    class Boom:
        def evaluate_before_model(self, **_):
            raise RuntimeError("evaluator bug")

    cb = GovernanceCallbacks(
        evaluator=Boom(),
        agent_name="a",
        session_id="s",  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.WARNING):
        # must NOT raise — a governance bug can't break the agent run
        cb.before_model(None, SimpleNamespace(contents=[_content([_part(text="x")])]))
    assert any("governance check failed" in r.message for r in caplog.records)


def test_callbacks_return_none():
    cb = _make_callbacks(FakeEvaluator())
    assert cb.before_model(None, SimpleNamespace(contents=[])) is None
    assert cb.after_model(None, SimpleNamespace(partial=False, content=None)) is None
    assert cb.before_tool(FakeTool("t"), {}, None) is None
    assert cb.after_tool(FakeTool("t"), {}, None, {}) is None
