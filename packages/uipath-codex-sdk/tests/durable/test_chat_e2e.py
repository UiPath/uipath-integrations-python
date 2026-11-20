"""Offline end-to-end tests of conversational (chat) agents.

Real factory → ``UiPathCodexResumableRuntime`` → ``UiPathCodexChatRuntime`` →
bundled codex 0.161 → gateway shim (mock model, streaming text deltas) + tool
host, with fake Orchestrator jobs. A "new job" gets a fresh runtime dir with
only ``state.db`` copied over, as on another robot.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from uipath.core.chat import UiPathConversationMessageEvent
from uipath.core.triggers import UiPathResumeTriggerName
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeMessageEvent

from uipath_codex_sdk.runtime.chat_runtime import UiPathCodexChatRuntime
from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.runtime import CONTINUE_PROMPT
from uipath_codex_sdk.runtime.session_store import CodexSessionStore
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage

from .._typing import as_result, attrs, sqlite_storage
from ..contract.mock_responses import (
    advertised_tools,
    find_tool,
    function_call_item,
    input_items,
    message_item,
)
from .fakes import TRIGGER
from .test_attachments_e2e import files  # noqa: F401 (fixture)

pytestmark = pytest.mark.contract

CONVERSATION = "conv-0001"

CHAT_MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool


class AddIn(BaseModel):
    a: int
    b: int


agent = UiPathCodexAgent(
    model="mock-model",
    base_instructions="You are a helpful test assistant.",
    tools=[
        uipath_process_tool(
            "a_plus_b",
            folder_path="Shared/Durable",
            description="Adds a and b.",
            input_schema=AddIn,
        )
    ],
    durable_settle_timeout=20.0,
)
"""


def user_message(text: str, message_id: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "user",
        "contentParts": [{"mimeType": "text/plain", "data": {"inline": text}}],
    }
    if message_id:
        message["messageId"] = message_id
    return {"messages": [message]}


def user_texts(body: dict[str, Any]) -> list[str]:
    """The user messages a model request carries (developer context excluded)."""
    out: list[str] = []
    for item in input_items(body):
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        parts = item.get("content") or []
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.lstrip().startswith("<"):  # environment context etc.
            out.append(text)
    return out


class ChatScript:
    """Mock model: echo the newest user text; call ``a_plus_b`` on TRIGGER.

    After the call's output arrives (the durable resume), answer with it.
    """

    def __init__(self) -> None:
        self.call_id = "call_chat000000000000000001"
        self.histories: list[list[str]] = []

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]]:
        items = input_items(body)
        texts = user_texts(body)
        self.histories.append(texts)
        outputs = {
            it["call_id"]: it.get("output")
            for it in items
            if it.get("type") == "function_call_output"
        }
        called = any(
            it.get("type") == "function_call" and it.get("call_id") == self.call_id
            for it in items
        )
        last = texts[-1] if texts else ""
        if TRIGGER in last and not called:
            name, ns = find_tool(advertised_tools(body), "a_plus_b")
            assert name, "a_plus_b is not advertised"
            return [
                message_item("Let me add them up.", idx),
                function_call_item(self.call_id, name, {"a": 17, "b": 25}, ns),
            ]
        if self.call_id in outputs and last == CONTINUE_PROMPT:
            return [message_item(f"The sum is {outputs[self.call_id]}.", idx)]
        return [message_item(f"You said: {last}", idx)]


@pytest.fixture
def chat_project(tmp_path: Path, monkeypatch) -> Path:
    proj = tmp_path / "chat"
    proj.mkdir()
    (proj / "codex.json").write_text('{"agents": {"agent": "main.py:agent"}}')
    (proj / "main.py").write_text(CHAT_MAIN)
    (proj / "uipath.json").write_text(
        json.dumps({"runtimeOptions": {"isConversational": True}})
    )
    monkeypatch.chdir(proj)
    return proj


def _context(run_dir: Path, *, resume: bool) -> UiPathRuntimeContext:
    return UiPathRuntimeContext(
        runtime_dir=str(run_dir), state_file="state.db", resume=resume
    )


async def _exchange(
    run_dir: Path, input: Any, *, resume: bool
) -> tuple[list[Any], UiPathRuntimeResult, list[Any]]:
    factory = UiPathCodexRuntimeFactory(_context(run_dir, resume=resume))
    runtime = await factory.new_runtime("agent", CONVERSATION)
    events: list[Any] = []
    try:
        async for event in runtime.stream(input, UiPathStreamOptions(resume=resume)):
            events.append(event)
        storage = await sqlite_storage(factory)
        triggers = await storage.get_triggers(CONVERSATION) or []
    finally:
        await runtime.dispose()
        await factory.dispose()
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult), result
    return events[:-1], result, triggers


def _new_job(previous: Path, root: Path, n: int) -> Path:
    """A fresh runtime dir holding only the previous job's state.db."""
    run_dir = root / f"job{n}"
    run_dir.mkdir()
    shutil.copy(previous / "state.db", run_dir / "state.db")
    return run_dir


def chat_payloads(events: list[Any]) -> list[dict[str, Any]]:
    out = []
    for event in events:
        if isinstance(event, UiPathRuntimeMessageEvent):
            assert isinstance(event.payload, UiPathConversationMessageEvent), event
            out.append(event.payload.model_dump(by_alias=True, exclude_none=True))
    return out


def shape(payload: dict[str, Any]) -> str:
    """A compact name of a chat event (``startMessage+startContentPart``...)."""
    keys: list[str] = [k for k in ("startMessage", "endMessage") if k in payload]
    part = payload.get("contentPart") or {}
    keys += [k for k in ("startContentPart", "chunk", "endContentPart") if k in part]
    call = payload.get("toolCall") or {}
    keys += [
        k for k in ("startToolCall", "executingToolCall", "endToolCall") if k in call
    ]
    return "+".join(keys)


def chunks(payloads: list[dict[str, Any]]) -> list[str]:
    return [
        p["contentPart"]["chunk"]["data"]
        for p in payloads
        if "chunk" in (p.get("contentPart") or {})
    ]


async def test_three_exchanges_remember_across_jobs(
    chat_project, upstream, world, tmp_path
):
    upstream.deltas = True
    script = ChatScript()
    upstream.script = script
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = jobs / "job0"

    texts = ["My name is Ada.", "I live in Cluj.", "What do you know about me?"]
    seen_payloads: list[list[dict[str, Any]]] = []
    for n, text in enumerate(texts):
        if n:
            run_dir = _new_job(run_dir, jobs, n)
        events, result, triggers = await _exchange(
            run_dir, user_message(text, f"m{n}"), resume=bool(n)
        )
        assert result.status == UiPathRuntimeStatus.SUSPENDED, result
        assert result.output is None and not result.triggers and not triggers
        payloads = chat_payloads(events)
        seen_payloads.append(payloads)
        assert [shape(p) for p in payloads] == (
            ["startMessage+startContentPart"]
            + ["chunk"] * len(upstream.sent_deltas[-1])
            + ["endContentPart", "endMessage"]
        ), [shape(p) for p in payloads]
        assert chunks(payloads) == upstream.sent_deltas[-1]
        assert "".join(chunks(payloads)) == f"You said: {text}"

    # The third request carried the first two exchanges: restored history.
    assert script.histories[-1] == texts, script.histories
    ids = {p["messageId"] for ps in seen_payloads for p in ps}
    assert len(ids) == 3  # one assistant message per exchange


async def test_durable_call_inside_an_exchange(chat_project, upstream, world, tmp_path):
    upstream.deltas = True
    script = ChatScript()
    upstream.script = script
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = jobs / "job0"

    # --- exchange 1 calls the durable tool: the exchange stays open -------
    events, result, triggers = await _exchange(
        run_dir, user_message(f"{TRIGGER}: add 17 and 25", "m1"), resume=False
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    [trigger] = triggers
    assert trigger.interrupt_id == script.call_id
    assert trigger.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert [t.interrupt_id for t in result.triggers or []] == [script.call_id]
    payloads = chat_payloads(events)
    shapes = [shape(p) for p in payloads]
    assert shapes[0] == "startMessage+startContentPart"
    assert shapes[-3:] == ["endContentPart", "startToolCall", "executingToolCall"]
    assert "endMessage" not in shapes
    assert "".join(chunks(payloads)) == "Let me add them up."
    message_id = payloads[0]["messageId"]
    start = payloads[-2]["toolCall"]
    assert payloads[-2]["messageId"] == message_id
    assert start["toolCallId"] == script.call_id
    assert start["startToolCall"]["toolName"] == "a_plus_b"
    assert start["startToolCall"]["input"] == {"a": 17, "b": 25}
    n_requests = len(upstream.snapshot())

    # --- the job is still running: the resume parks again ------------------
    run_dir = _new_job(run_dir, jobs, 1)
    events, result, triggers = await _exchange(run_dir, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in triggers] == [script.call_id]
    assert chat_payloads(events) == []

    # --- the job finished: the exchange continues and ends -----------------
    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    run_dir = _new_job(run_dir, jobs, 2)
    events, result, triggers = await _exchange(run_dir, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert not result.triggers and not triggers
    payloads = chat_payloads(events)
    shapes = [shape(p) for p in payloads]
    assert shapes[:2] == ["endToolCall", "endMessage"], shapes
    assert payloads[0]["messageId"] == message_id
    end = payloads[0]["toolCall"]
    assert end["toolCallId"] == script.call_id
    assert end["endToolCall"]["output"] == {"sum": 42}
    assert end["endToolCall"]["isError"] is False
    assert shapes[2] == "startMessage+startContentPart"
    assert shapes[-2:] == ["endContentPart", "endMessage"]
    assert "".join(chunks(payloads)) == 'The sum is {"sum": 42}.'
    assert len(upstream.snapshot()) > n_requests
    assert len(world.invocations) == 1

    # --- the next exchange remembers the durable one ---------------------
    run_dir = _new_job(run_dir, jobs, 3)
    events, result, _ = await _exchange(
        run_dir, user_message("Thanks!", "m2"), resume=True
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert "".join(chunks(chat_payloads(events))) == "You said: Thanks!"
    assert script.histories[-1] == [
        f"{TRIGGER}: add 17 and 25",
        CONTINUE_PROMPT,
        "Thanks!",
    ]
    last = upstream.snapshot()[-1].body
    outputs = [
        it["output"]
        for it in input_items(last)
        if it.get("type") == "function_call_output"
    ]
    assert outputs == [json.dumps({"sum": 42})]


async def test_message_while_a_durable_call_is_pending_resuspends(
    chat_project, upstream, world, tmp_path
):
    upstream.deltas = True
    script = ChatScript()
    upstream.script = script
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = jobs / "job0"
    _, result, triggers = await _exchange(
        run_dir, user_message(f"{TRIGGER}: add 17 and 25", "m1"), resume=False
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    [trigger] = triggers
    n_requests = len(upstream.snapshot())

    # A new message arrives (CAS resumes the job with it): not processed.
    run_dir = _new_job(run_dir, jobs, 1)
    events, result, triggers = await _exchange(
        run_dir, user_message("Are you there?", "m2"), resume=True
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert chat_payloads(events) == []
    assert len(upstream.snapshot()) == n_requests  # no model call
    # The trigger is still stored (and returned to the host) for the call.
    [kept] = triggers
    assert kept.interrupt_id == script.call_id
    assert kept.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert kept.item_key == trigger.item_key
    assert [t.interrupt_id for t in result.triggers or []] == [script.call_id]

    # The job's original input again (a resumed job may be handed it): it is
    # this exchange's own message, so the jobs are read; still running.
    run_dir = _new_job(run_dir, jobs, 2)
    _, result, triggers = await _exchange(
        run_dir, user_message(f"{TRIGGER}: add 17 and 25", "m1"), resume=True
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in triggers] == [script.call_id]

    # A plain-text resume input (``uipath run agent '"hello?"' --resume``).
    # ``uipath dev`` never does this: it polls the job and drops what is
    # typed meanwhile (``test_uipath_dev_resumes_a_durable_exchange_by_itself``).
    run_dir = _new_job(run_dir, jobs, 3)
    _, result, triggers = await _exchange(run_dir, "hello?", resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in triggers] == [script.call_id]
    assert len(upstream.snapshot()) == n_requests

    # Once the job is done, ``{}`` (``uipath run --resume``: "check the jobs")
    # finishes the exchange as usual.
    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    run_dir = _new_job(run_dir, jobs, 4)
    events, result, triggers = await _exchange(run_dir, {}, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not triggers
    payloads = chat_payloads(events)
    assert payloads[0]["toolCall"]["endToolCall"]["output"] == {"sum": 42}
    assert "".join(chunks(payloads)) == 'The sum is {"sum": 42}.'
    # The ignored messages never reached the model.
    assert all(
        "Are you there?" not in h and "hello?" not in h for h in script.histories
    )


async def test_dev_exchanges_share_one_factory_without_resume(
    chat_project, upstream, world
):
    """``uipath dev``: every exchange is a fresh execute (resume=False) with
    the same runtime id, on one factory."""
    script = ChatScript()  # no deltas: one chunk per message
    upstream.script = script
    run_dir = chat_project / "__uipath"
    factory = UiPathCodexRuntimeFactory(_context(run_dir, resume=False))
    try:
        for text in ["first", "second"]:
            runtime = await factory.new_runtime("agent", CONVERSATION)
            assert isinstance(runtime.delegate, UiPathCodexChatRuntime)  # type: ignore[attr-defined]
            events = [
                e
                async for e in runtime.stream(
                    user_message(text), UiPathStreamOptions(resume=False)
                )
            ]
            await runtime.dispose()
            result = as_result(events[-1])
            assert result.status == UiPathRuntimeStatus.SUSPENDED
            payloads = chat_payloads(events)
            assert [shape(p) for p in payloads] == [
                "startMessage+startContentPart",
                "chunk",
                "endContentPart",
                "endMessage",
            ]
            assert chunks(payloads) == [f"You said: {text}"]
    finally:
        await factory.dispose()
    assert script.histories[-1] == ["first", "second"]


def _cli_uipath_json(proj: Path, exchange_id: str) -> None:
    (proj / "uipath.json").write_text(
        json.dumps(
            {
                "runtimeOptions": {"isConversational": True},
                "fpsProperties": {
                    "conversationalService.conversationId": CONVERSATION,
                    "conversationalService.exchangeId": exchange_id,
                },
            }
        )
    )


def _envelopes(log: Path, offset: int) -> list[dict[str, Any]]:
    """The CAS envelopes the bridge logged after ``offset``.

    With ``CAS_WEBSOCKET_DISABLED`` it logs each one instead of sending it;
    a job's logs go to ``__uipath/execution.log``.
    """
    marker = "Not sending event: "
    text = log.read_text(encoding="utf-8")[offset:] if log.exists() else ""
    return [
        json.loads(line.split(marker, 1)[1])
        for line in text.splitlines()
        if marker in line
    ]


def test_cli_exchanges_reach_the_chat_bridge(cli_env, monkeypatch):
    """``uipath run`` as a deployed chat job runs it: a job key, the
    conversation ids in ``fpsProperties`` and the bridge in debug mode (it
    logs each CAS envelope instead of sending it)."""
    from click.testing import CliRunner
    from uipath._cli import cli

    proj, server = cli_env
    server.deltas = True
    server.script = ChatScript()
    (proj / "main.py").write_text(CHAT_MAIN)
    (proj / "input.json").write_text(json.dumps(user_message("Hello", "m1")))
    (proj / "next.json").write_text(json.dumps(user_message("Again", "m2")))
    monkeypatch.setenv("UIPATH_JOB_KEY", "chat-job-key")
    monkeypatch.setenv("CAS_WEBSOCKET_DISABLED", "true")
    monkeypatch.setenv("UIPATH_URL", "https://127.0.0.1:9/org/tenant")
    monkeypatch.setenv("UIPATH_TRACING_ENABLED", "false")
    log = proj / "__uipath" / "execution.log"

    for n, (file, exchange) in enumerate(
        [("input.json", "ex-1"), ("next.json", "ex-2")]
    ):
        _cli_uipath_json(proj, exchange)
        offset = len(log.read_text(encoding="utf-8")) if log.exists() else 0
        args = ["run", "agent", "--file", file] + (["--resume"] if n else [])
        res = CliRunner().invoke(cli, args, catch_exceptions=False)
        assert res.exit_code == 0, res.output
        out = json.loads((proj / "__uipath" / "output.json").read_text())
        assert out["status"].lower() == "suspended", out
        assert not out.get("resumeTriggers"), out
        envelopes = _envelopes(log, offset)
        assert envelopes, res.output
        assert {e["conversationId"] for e in envelopes} == {CONVERSATION}
        assert {e["exchange"]["exchangeId"] for e in envelopes} == {exchange}
        assert envelopes[-1]["exchange"] == {"exchangeId": exchange, "endExchange": {}}
        messages = [e["exchange"]["message"] for e in envelopes[:-1]]
        assert "startMessage" in messages[0] and "endMessage" in messages[-1]
        text = "".join(
            m["contentPart"]["chunk"]["data"]
            for m in messages
            if "chunk" in (m.get("contentPart") or {})
        )
        assert text == ("You said: Hello" if n == 0 else "You said: Again")
    assert server.script.histories[-1] == ["Hello", "Again"]


@pytest.fixture
def spans():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(  # type: ignore[attr-defined]
        SimpleSpanProcessor(exporter)
    )
    yield exporter
    exporter.shutdown()


def _exchange_spans(exporter) -> list[Any]:
    return [
        s
        for s in exporter.get_finished_spans()
        if s.name.startswith("Conversational agent run - ")
    ]


async def test_each_exchange_is_one_conversational_agent_run(
    chat_project, upstream, world, tmp_path, spans
):
    script = ChatScript()
    upstream.script = script
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = jobs / "job0"

    # Two plain exchanges: one finished span each.
    for n, text in enumerate(["hello", "again"]):
        if n:
            run_dir = _new_job(run_dir, jobs, n)
        await _exchange(run_dir, user_message(text, f"m{n}"), resume=bool(n))
    plain = _exchange_spans(spans)
    assert [s.name for s in plain] == ["Conversational agent run - agent"] * 2
    first, second = (attrs(s) for s in plain)
    for a in (first, second):
        assert a["isConversational"] is True
        assert a["span_type"] == "agentRun"
        assert a["conversationId"] == CONVERSATION
        assert a["status"] == "successful"
        # Not even the second, a job resume in Orchestrator: a new exchange.
        assert "userPrompt" not in a and "resumed" not in a
    assert first["threadId"] == second["threadId"]
    assert json.loads(first["input"]) == user_message("hello", "m0")
    assert plain[0].context.span_id != plain[1].context.span_id

    # A durable exchange: the span outlives the suspension, then completes.
    run_dir = _new_job(run_dir, jobs, 2)
    await _exchange(run_dir, user_message(f"{TRIGGER}: add", "m2"), resume=True)
    assert len(_exchange_spans(spans)) == 2  # still open
    storage = SqliteResumableStorage(str(run_dir / "state.db"))
    try:
        saved = await CodexSessionStore(storage, CONVERSATION).get_trace()
    finally:
        await storage.dispose()
    assert saved is not None
    assert saved["name"] == "Conversational agent run - agent"
    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    run_dir = _new_job(run_dir, jobs, 3)
    await _exchange(run_dir, None, resume=True)
    done = _exchange_spans(spans)
    assert len(done) == 3
    durable = done[-1]
    assert format(durable.context.span_id, "016x") == saved["span_id"]
    assert attrs(durable)["status"] == "successful"
    assert attrs(durable)["threadId"] == first["threadId"]


async def test_the_chat_host_keeps_a_parked_exchange_open(
    chat_project, upstream, world, tmp_path
):
    """Through ``UiPathChatRuntime``: a durable call's trigger is not an API
    one, so the exchange is not ended; the resume that finishes it is."""
    from uipath.runtime.chat import UiPathChatRuntime

    from ..test_chat import FakeBridge

    upstream.deltas = True
    upstream.script = ChatScript()
    jobs = tmp_path / "jobs"
    jobs.mkdir()

    async def host_exchange(run_dir: Path, input: Any, resume: bool):
        factory = UiPathCodexRuntimeFactory(_context(run_dir, resume=resume))
        bridge = FakeBridge()
        runtime = await factory.new_runtime("agent", CONVERSATION)
        try:
            host = UiPathChatRuntime(delegate=runtime, chat_bridge=bridge)  # type: ignore[arg-type]
            events = [
                e async for e in host.stream(input, UiPathStreamOptions(resume=resume))
            ]
        finally:
            await runtime.dispose()
            await factory.dispose()
        return events[-1], bridge

    result, bridge = await host_exchange(
        jobs / "job0", user_message(f"{TRIGGER}: add", "m1"), False
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.trigger_name for t in result.triggers] == [
        UiPathResumeTriggerName.JOB_RAW
    ]
    assert bridge.exchange_ended == 0 and bridge.interrupts == []
    sent = [m.model_dump(by_alias=True, exclude_none=True) for m in bridge.messages]
    assert [shape(p) for p in sent][-2:] == ["startToolCall", "executingToolCall"]

    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    result, bridge = await host_exchange(_new_job(jobs / "job0", jobs, 1), None, True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert bridge.exchange_ended == 1 and bridge.errors == []
    sent = [m.model_dump(by_alias=True, exclude_none=True) for m in bridge.messages]
    assert [shape(p) for p in sent][:2] == ["endToolCall", "endMessage"]
    assert [shape(p) for p in sent][-1] == "endMessage"


def _dev_service(factory: UiPathCodexRuntimeFactory, shown: list[Any]) -> Any:
    """The run service of ``uipath dev``'s web UI, wired as it wires it.

    With the debug bridge it installs, every run is
    ``Debug(Chat(Resumable(Codex)))``: the debug runtime polls a durable
    call's job trigger and resumes the run itself.
    """
    from uipath.core.tracing import UiPathTraceManager
    from uipath.dev.server.debug_bridge import WebDebugBridge
    from uipath.dev.services.run_service import RunService

    return RunService(
        runtime_factory=factory,
        trace_manager=UiPathTraceManager(),
        on_chat=shown.append,
        debug_bridge_factory=lambda mode: WebDebugBridge(mode=mode),
    )


def _assistant_texts(run: Any) -> list[str]:
    return [
        "".join(
            str(p.data.inline) for p in m.content_parts if hasattr(p.data, "inline")
        )
        for m in run.chat_events.messages.values()
        if m.role == "assistant"
    ]


async def test_uipath_dev_chat_renders_the_exchanges(chat_project, upstream, world):
    """The ``uipath dev`` chat: its run service drives each exchange (a
    fresh execute per message, one factory) and folds the events into the
    messages its chat panel shows."""
    pytest.importorskip("uipath.dev")
    from uipath.dev.models.chat import get_user_message
    from uipath.dev.models.execution import ExecutionMode, ExecutionRun

    upstream.deltas = True
    upstream.script = ChatScript()
    factory = UiPathCodexRuntimeFactory(
        _context(chat_project / "__uipath", resume=False)
    )
    shown: list[Any] = []
    service = _dev_service(factory, shown)
    run = ExecutionRun("agent", {}, ExecutionMode.CHAT)
    service.register_run(run)
    try:
        for text in ["My name is Ada.", "What is my name?"]:
            run.input_data = {
                "messages": [get_user_message(text).model_dump(by_alias=True)]
            }
            await service.execute(run)
            assert run.status == "completed", (run.status, run.error)
    finally:
        await factory.dispose()
    assistant = [m for m in run.chat_events.messages.values() if m.role == "assistant"]
    assert _assistant_texts(run) == [
        "You said: My name is Ada.",
        "You said: What is my name?",
    ]
    assert shown and all(
        m.content_parts[0].mime_type == "text/markdown" for m in assistant
    )
    assert upstream.script.histories[-1] == ["My name is Ada.", "What is my name?"]


async def test_uipath_dev_resumes_a_durable_exchange_by_itself(
    chat_project, upstream, world, monkeypatch
):
    """In ``uipath dev`` a durable exchange does not show as suspended: the
    debug runtime polls the job (every 5 s; faster here) while the run keeps
    running, then resumes it once the job is done. The card closes with the
    result and the reply continues, with no message sent."""
    pytest.importorskip("uipath.dev")
    from uipath.dev.models.chat import get_user_message
    from uipath.dev.models.execution import ExecutionMode, ExecutionRun
    from uipath.runtime.debug import UiPathDebugRuntime

    polls: list[int] = []

    async def poll_wait(self: Any) -> None:
        polls.append(1)
        await asyncio.sleep(0.05)

    monkeypatch.setattr(UiPathDebugRuntime, "_wait_with_quit_check", poll_wait)
    upstream.deltas = True
    upstream.script = ChatScript()
    factory = UiPathCodexRuntimeFactory(
        _context(chat_project / "__uipath", resume=False)
    )
    service = _dev_service(factory, [])
    run = ExecutionRun("agent", {}, ExecutionMode.CHAT)
    service.register_run(run)
    run.input_data = {
        "messages": [
            get_user_message(f"{TRIGGER}: add 17 and 25").model_dump(by_alias=True)
        ]
    }
    task = asyncio.create_task(service.execute(run))
    try:
        for _ in range(600):
            if polls or task.done():
                break
            await asyncio.sleep(0.05)
        assert polls, "the debug runtime never polled the job"
        # Waiting on the job: still running, not suspended.
        assert not task.done() and run.status == "running", run.status
        world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
        await asyncio.wait_for(task, 60)
    finally:
        if not task.done():
            task.cancel()
        await factory.dispose()
    assert run.status == "completed", (run.status, run.error)
    assert _assistant_texts(run) == ["Let me add them up.", 'The sum is {"sum": 42}.']
    calls = [
        call for m in run.chat_events.messages.values() for call in (m.tool_calls or [])
    ]
    assert [c.name for c in calls] == ["a_plus_b"]
    assert calls[0].result is not None and calls[0].result.output == {"sum": 42}
    assert len(world.invocations) == 1


SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "chat-agent"


class RoomsScript:
    """Mock model for the sample: look the rooms up, then answer from them."""

    def __init__(self) -> None:
        self.histories: list[list[str]] = []

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]]:
        items = input_items(body)
        texts = user_texts(body)
        self.histories.append(texts)
        outputs = [
            it.get("output") for it in items if it.get("type") == "function_call_output"
        ]
        if len(texts) == 1 and not outputs:
            name, ns = find_tool(advertised_tools(body), "find_rooms")
            assert name, "find_rooms is not advertised"
            return [
                message_item("Checking the rooms.", idx),
                function_call_item("call_rooms01", name, {"people": 6}, ns),
            ]
        if len(texts) == 1:
            return [
                message_item("Beehive seats 8: it fits the Kingfisher review.", idx)
            ]
        return [message_item("I suggested Beehive, for the Kingfisher review.", idx)]


async def test_the_chat_agent_sample(upstream, world, tmp_path, monkeypatch):
    proj = tmp_path / "chat-agent"
    shutil.copytree(
        SAMPLE, proj, ignore=shutil.ignore_patterns("__uipath", "__pycache__", ".venv")
    )
    monkeypatch.chdir(proj)
    upstream.deltas = True
    upstream.script = RoomsScript()
    first = json.loads((proj / "input.json").read_text())
    events, result, _ = await _exchange(proj / "__uipath", first, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    payloads = chat_payloads(events)
    starts = [
        p["toolCall"] for p in payloads if "startToolCall" in p.get("toolCall", {})
    ]
    ends = [p["toolCall"] for p in payloads if "endToolCall" in p.get("toolCall", {})]
    assert [s["startToolCall"]["toolName"] for s in starts] == ["find_rooms"]
    assert starts[0]["startToolCall"]["input"] == {"people": 6}
    rooms = ends[0]["endToolCall"]["output"]["rooms"]
    assert [r["name"] for r in rooms] == ["Beehive", "Cedar Hall"]
    assert "".join(chunks(payloads)) == (
        "Checking the rooms.Beehive seats 8: it fits the Kingfisher review."
    )

    run_dir = _new_job(proj / "__uipath", tmp_path, 1)
    second = json.loads((proj / "next_message.json").read_text())
    events, result, _ = await _exchange(run_dir, second, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert "Beehive" in "".join(chunks(chat_payloads(events)))
    assert len(upstream.script.histories[-1]) == 2


async def test_a_file_uploaded_in_the_chat_reaches_analyze_files(
    upstream,
    files,  # noqa: F811 (the fixture imported above)
    tmp_path,
    monkeypatch,
):
    """A chat file (``urn:uipath:cas:file:orchestrator:<id>``) is a job
    attachment the run knows: the prompt lists it and the model passes its ID
    to Analyze Files (it stays known in later exchanges: ``test_chat``)."""
    from .test_attachments_e2e import CALL_ID, DOC_ID, Script

    proj = tmp_path / "chat-agent"
    shutil.copytree(
        SAMPLE, proj, ignore=shutil.ignore_patterns("__uipath", "__pycache__", ".venv")
    )
    monkeypatch.chdir(proj)
    script = Script(
        "Analyze_Files",
        {"attachments": [{"ID": DOC_ID}], "analysisTask": "What is the agenda?"},
        lambda outputs: f"Agenda: {outputs.get(CALL_ID)}",
    )
    upstream.script = script
    message = {
        "messages": [
            {
                "role": "user",
                "contentParts": [
                    {"mimeType": "text/plain", "data": {"inline": "Read this"}},
                    {
                        "mimeType": "application/pdf",
                        "name": "agenda.pdf",
                        "data": {"uri": f"urn:uipath:cas:file:orchestrator:{DOC_ID}"},
                    },
                ],
            }
        ]
    }
    events, result, _ = await _exchange(proj / "__uipath", message, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    prompt = script.prompts[0]
    assert prompt.startswith("Read this\n\nInput attachments (JobAttachment objects):")
    assert DOC_ID in prompt and "agenda.pdf" in prompt
    payloads = chat_payloads(events)
    [end] = [p["toolCall"] for p in payloads if "endToolCall" in p.get("toolCall", {})]
    assert end["endToolCall"]["isError"] is False
    assert end["endToolCall"]["output"] == {"analysisResult": "Total: 42"}
    [request] = files.requests
    assert request["input"][0]["content"][1]["filename"] == "agenda-pdf"
