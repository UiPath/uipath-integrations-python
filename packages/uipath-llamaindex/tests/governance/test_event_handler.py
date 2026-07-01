"""Unit tests for the LlamaIndex governance event handler.

The adapter governs via the LlamaIndex instrumentation dispatcher, so these
tests exercise the real event types (``LLMChatStartEvent`` etc.) routed
through :class:`GovernanceEventHandler`, plus the adapter's register/detach on
the dispatcher. The dispatcher is process-global, so each dispatcher test
cleans up after itself via ``detach``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.instrumentation import (  # type: ignore[attr-defined]
    get_dispatcher,
)
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.tools.types import ToolMetadata
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_llamaindex.governance.event_handler import (
    _BEFORE_MODEL_TEXT_CAP,
    GovernanceCallbacks,
    GovernanceEventHandler,
    _coerce_args,
    install_governance,
    uninstall_governance,
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
            raise GovernanceBlockException("blocked")

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


class FakeWorkflow:
    """Duck-typed LlamaIndex workflow stand-in."""

    async def run(self, *_a: Any, **_k: Any) -> None:
        return None


def _make_callbacks(ev: FakeEvaluator) -> GovernanceCallbacks:
    return GovernanceCallbacks(evaluator=ev, agent_name="agent-1", session_id="sess-1")


def _handler(ev: FakeEvaluator) -> GovernanceEventHandler:
    return GovernanceEventHandler(callbacks=_make_callbacks(ev))


# --------------------------------------------------------------------------
# install_governance (real dispatcher)
# --------------------------------------------------------------------------


def _gov_handlers() -> list[Any]:
    return [
        h
        for h in get_dispatcher().event_handlers
        if isinstance(h, GovernanceEventHandler)
    ]


def _clear_gov_handlers() -> None:
    # Use the adapter's own public detach rather than mutating the dispatcher.
    uninstall_governance()


def test_install_governance_registers_handler():
    agent = FakeWorkflow()
    try:
        returned = install_governance(
            agent, FakeEvaluator(), agent_name="x", session_id="s"
        )
        assert returned is agent
        assert len(_gov_handlers()) == 1
    finally:
        _clear_gov_handlers()
    assert _gov_handlers() == []


def test_install_governance_reinstall_rebinds_single_handler():
    """The dispatcher is process-global: a second install keeps one handler but
    rebinds it to the new run's evaluator / session (last install wins)."""
    try:
        install_governance(
            FakeWorkflow(), FakeEvaluator(), agent_name="a", session_id="s1"
        )
        handlers = _gov_handlers()
        assert len(handlers) == 1
        gov = handlers[0]
        assert gov._callbacks._session_id == "s1"

        ev2 = FakeEvaluator()
        install_governance(FakeWorkflow(), ev2, agent_name="b", session_id="s2")
        handlers = _gov_handlers()
        assert len(handlers) == 1  # not stacked
        assert handlers[0] is gov  # same handler, rebound
        assert gov._callbacks._session_id == "s2"
        assert gov._callbacks._evaluator is ev2
    finally:
        _clear_gov_handlers()


def test_uninstall_governance_removes_handler():
    install_governance(FakeWorkflow(), FakeEvaluator(), agent_name="x", session_id="s")
    assert len(_gov_handlers()) == 1
    uninstall_governance()
    assert _gov_handlers() == []
    # safe to call again when nothing is registered
    uninstall_governance()
    assert _gov_handlers() == []


# --------------------------------------------------------------------------
# Factory wiring — the evaluator kwarg drives install_governance
# --------------------------------------------------------------------------


def _factory_without_init():
    """A factory instance that skips __init__ (avoids config/IO)."""
    from uipath_llamaindex.runtime.factory import UiPathLlamaIndexRuntimeFactory

    f = UiPathLlamaIndexRuntimeFactory.__new__(UiPathLlamaIndexRuntimeFactory)
    f.context = SimpleNamespace(command="run")  # type: ignore[assignment]  # read for debug_mode
    return f


def _stub_factory_runtime(monkeypatch, factory_mod):
    """Stub the runtime constructions + storage so only the governance branch runs."""
    monkeypatch.setattr(
        factory_mod, "UiPathLlamaIndexRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        factory_mod, "UiPathResumableRuntime", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(factory_mod, "UiPathResumeTriggerHandler", lambda *a, **k: None)

    async def _no_storage(self):
        return None

    monkeypatch.setattr(
        factory_mod.UiPathLlamaIndexRuntimeFactory, "_get_storage", _no_storage
    )


async def test_factory_installs_governance_when_evaluator_supplied(monkeypatch):
    from uipath_llamaindex.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    try:
        await _factory_without_init()._create_runtime_instance(
            workflow=FakeWorkflow(),
            runtime_id="r",
            entrypoint="e",
            evaluator=FakeEvaluator(),
        )
        assert len(_gov_handlers()) == 1
    finally:
        _clear_gov_handlers()


async def test_factory_skips_governance_without_evaluator(monkeypatch):
    from uipath_llamaindex.runtime import factory as factory_mod

    _stub_factory_runtime(monkeypatch, factory_mod)
    await _factory_without_init()._create_runtime_instance(
        workflow=FakeWorkflow(), runtime_id="r", entrypoint="e"
    )
    assert _gov_handlers() == []


# --------------------------------------------------------------------------
# event routing through the handler
# --------------------------------------------------------------------------


def test_handler_routes_llm_chat_start_to_before_model():
    ev = FakeEvaluator()
    h = _handler(ev)
    event = LLMChatStartEvent(
        messages=[
            ChatMessage(role="user", content="old"),
            ChatMessage(role="user", content="the question"),
        ],
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
        response=ChatResponse(
            message=ChatMessage(role="assistant", content="the answer")
        ),
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
    # malformed JSON is preserved raw (not dropped) so policies can still scan it
    assert _coerce_args("not json") == {"_raw": "not json"}


def test_coerce_args_preserves_list_shaped_args():
    # list-shaped tool args (common with MCP tools) must not be dropped to {}
    assert _coerce_args(["a", "b"]) == {"_": ["a", "b"]}
    assert _coerce_args('["a", "b"]') == {"_": ["a", "b"]}


def test_message_text_walks_blocks_when_content_empty():
    # a multimodal message whose .content is empty falls back to its text
    # blocks, not str(message) (which would serialize a pydantic repr)
    from uipath_llamaindex.governance.event_handler import _message_text

    msg = SimpleNamespace(
        content=None,
        blocks=[SimpleNamespace(text="block one"), SimpleNamespace(text="block two")],
    )
    assert _message_text(msg) == "block one\nblock two"


# --------------------------------------------------------------------------
# enforcement semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook,invoke",
    [
        (
            "before_model",
            lambda cb: cb.before_model([ChatMessage(role="user", content="hi")]),
        ),
        (
            "after_model",
            lambda cb: cb.after_model(
                ChatResponse(message=ChatMessage(role="assistant", content="o"))
            ),
        ),
        (
            "tool_call",
            lambda cb: cb.tool_call(ToolMetadata(description="d", name="t"), "{}"),
        ),
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
    # Attach caplog's handler directly to the module logger: other suites in the
    # full run can configure an ancestor ``uipath*`` logger with
    # propagate=False, which breaks caplog's default root-handler capture.
    logger = logging.getLogger("uipath_llamaindex.governance.event_handler")
    logger.addHandler(caplog.handler)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    try:
        cb.before_model([ChatMessage(role="user", content="x")])
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(prev)
    assert any("governance check failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# coverage: swallow on after_model/tool_call + extraction edges
# --------------------------------------------------------------------------


class _Boom:
    """Evaluator whose every evaluate_* raises a non-block error."""

    def __getattr__(self, _name: str) -> Any:
        def _raise(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("evaluator bug")

        return _raise


def test_after_model_and_tool_call_swallow_non_block_errors(caplog):
    cb = GovernanceCallbacks(evaluator=_Boom(), agent_name="a", session_id="s")
    logger = logging.getLogger("uipath_llamaindex.governance.event_handler")
    logger.addHandler(caplog.handler)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    try:
        cb.after_model(SimpleNamespace(message=SimpleNamespace(content="x")))
        cb.tool_call(SimpleNamespace(name="t"), {})
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(prev)
    assert sum("governance check failed" in r.message for r in caplog.records) >= 2


def test_extraction_edges():
    from uipath_llamaindex.governance.event_handler import (
        _latest_message_text,
        _message_text,
        _response_text,
    )

    # _message_text: None / str / object with no content or blocks -> str()
    assert _message_text(None) == ""
    assert _message_text("plain") == "plain"
    assert isinstance(_message_text(SimpleNamespace(content=None, blocks=None)), str)
    # _latest_message_text: single (non-list) message
    assert _latest_message_text(SimpleNamespace(content="solo")) == "solo"
    # _response_text: None / .message / .text fallback / str() fallback
    assert _response_text(None) == ""
    assert (
        _response_text(SimpleNamespace(message=SimpleNamespace(content="viamsg")))
        == "viamsg"
    )
    assert _response_text(SimpleNamespace(message=None, text="viatext")) == "viatext"
    assert isinstance(_response_text(SimpleNamespace(message=None, text=None)), str)
