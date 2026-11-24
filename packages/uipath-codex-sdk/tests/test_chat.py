"""Conversational (chat) agents: factory switch, input, chat events, exchanges.

The exchange tests run the real factory, resumable wrapper and chat runtime
over a fake ``AsyncCodex`` (scripted notifications, a rollout file per
thread) and a fake gateway shim; the offline end-to-end tests with the real
codex binary are in ``durable/test_chat_e2e.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    TurnCompletedNotification,
)
from pydantic import BaseModel
from uipath.core.chat import UiPathConversationMessageEvent
from uipath.core.triggers import UiPathResumeTriggerName
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.chat import UiPathChatRuntime
from uipath.runtime.events import UiPathRuntimeMessageEvent, UiPathRuntimeStateEvent

import uipath_codex_sdk.runtime.runtime as runtime_mod
from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool
from uipath_codex_sdk.runtime.chat_events import (
    ATTACHMENT_ONLY_TEXT,
    CodexChatEventMapper,
    chat_attachments,
    extract_user_message,
    is_chat_input,
    message_fingerprint,
    user_text,
)
from uipath_codex_sdk.runtime.chat_runtime import UiPathCodexChatRuntime
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.factory import (
    UiPathCodexResumableRuntime,
    UiPathCodexRuntimeFactory,
)
from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime
from uipath_codex_sdk.runtime.schema import default_messages_schema
from uipath_codex_sdk.runtime.session_store import (
    STATE_STARTED,
    CodexSessionStore,
    DanglingCall,
    EntrypointRecord,
    PendingCall,
    ThreadRecord,
    encode_rollout,
)
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage
from uipath_codex_sdk.runtime.telemetry import start_agent_run

from ._typing import as_result
from .durable.fakes import FakeOrchestrator

RT = "conv-1"
FILE_ID = "a940a416-b97b-4146-3089-08de5f4d0a87"


def msg(text: str | None = None, *, role: str = "user", **extra: Any) -> dict:
    parts = []
    if text is not None:
        parts.append({"mimeType": "text/plain", "data": {"inline": text}})
    return {"role": role, "contentParts": parts, **extra}


def file_part(name: str = "report.pdf") -> dict:
    return {
        "mimeType": "application/pdf",
        "name": name,
        "data": {"uri": f"urn:uipath:cas:file:orchestrator:{FILE_ID}"},
    }


def payloads(events: list[Any]) -> list[dict[str, Any]]:
    out = []
    for event in events:
        if isinstance(event, UiPathRuntimeMessageEvent):
            assert isinstance(event.payload, UiPathConversationMessageEvent), event
            out.append(event.payload.model_dump(by_alias=True, exclude_none=True))
    return out


def shape(payload: dict[str, Any]) -> str:
    keys: list[str] = [k for k in ("startMessage", "endMessage") if k in payload]
    part = payload.get("contentPart") or {}
    keys += [k for k in ("startContentPart", "chunk", "endContentPart") if k in part]
    call = payload.get("toolCall") or {}
    keys += [
        k for k in ("startToolCall", "executingToolCall", "endToolCall") if k in call
    ]
    return "+".join(keys)


# --------------------------------------------------------------------------
# factory switch and schema
# --------------------------------------------------------------------------
def _factory(tmp_path: Path, **context: Any) -> UiPathCodexRuntimeFactory:
    return UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(tmp_path / "__uipath"), state_file="state.db", **context
        )
    )


def test_conversational_from_uipath_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    assert factory._is_conversational() is False  # no uipath.json
    (tmp_path / "uipath.json").write_text("{not json")
    assert factory._is_conversational() is False
    (tmp_path / "uipath.json").write_text('{"runtimeOptions": {}}')
    assert factory._is_conversational() is False
    (tmp_path / "uipath.json").write_text(
        '{"runtimeOptions": {"isConversational": true}}'
    )
    assert factory._is_conversational() is True


def test_conversational_from_the_chat_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _factory(tmp_path, conversation_id="c1")._is_conversational() is True


def _project(tmp_path: Path, main: str, conversational: bool = True) -> Path:
    (tmp_path / "codex.json").write_text('{"agents": {"agent": "main.py:agent"}}')
    (tmp_path / "main.py").write_text(main)
    (tmp_path / "uipath.json").write_text(
        json.dumps({"runtimeOptions": {"isConversational": conversational}})
    )
    return tmp_path


CHAT_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(model="m", base_instructions="Be brief.")
"""


async def test_factory_builds_the_chat_runtime_with_the_messages_schema(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(_project(tmp_path, CHAT_MAIN))
    factory = _factory(tmp_path)
    try:
        runtime = await factory.new_runtime("agent", RT)
        assert isinstance(runtime, UiPathCodexResumableRuntime)
        assert runtime.conversational
        assert isinstance(runtime.delegate, UiPathCodexChatRuntime)
        # A run-scoped CODEX_HOME, as for durable agents.
        assert runtime.delegate._codex_home is not None
        schema = await runtime.get_schema()
        assert schema.input == default_messages_schema() == schema.output
        assert schema.input["required"] == []
        assert "messages" in schema.input["properties"]
    finally:
        await factory.dispose()


async def test_factory_keeps_the_standard_runtime_otherwise(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path, CHAT_MAIN, conversational=False))
    factory = _factory(tmp_path)
    try:
        runtime = await factory.new_runtime("agent", RT)
        assert type(runtime.delegate) is UiPathCodexRuntime  # type: ignore[attr-defined]
        assert not runtime.conversational  # type: ignore[attr-defined]
    finally:
        await factory.dispose()


class _In(BaseModel):
    topic: str


@pytest.mark.parametrize(
    "fields, refused",
    [
        ({"prompt": "Hi {name}"}, "prompt"),
        ({"input_schema": _In}, "input_schema"),
        ({"output_schema": _In}, "output_schema"),
    ],
)
def test_a_chat_agent_refuses_prompt_and_schemas(tmp_path, fields, refused):
    with pytest.raises(UiPathCodexRuntimeError) as info:
        UiPathCodexChatRuntime(
            agent=UiPathCodexAgent(model="m", **fields),
            runtime_id=RT,
            entrypoint="agent",
            file_path=str(tmp_path / "main.py"),
        )
    assert "CONFIG_INVALID" in str(info.value.error_info.code)
    assert refused in info.value.error_info.detail


# --------------------------------------------------------------------------
# the chat input
# --------------------------------------------------------------------------
def test_extract_takes_the_newest_user_message():
    data = {
        "messages": [
            msg("first"),
            msg("an answer", role="assistant"),
            msg("second", messageId="m2"),
            msg("from the system", role="system"),
        ]
    }
    message = extract_user_message(data)
    assert message is not None and user_text(message) == "second"
    assert message_fingerprint(message) == "m2"
    # role-less counts as the user's; empty messages are skipped
    assert (
        user_text(extract_user_message({"messages": [msg("x", role="")]}) or {}) == "x"
    )
    assert extract_user_message({"messages": [msg("x"), msg()]}) is not None
    assert extract_user_message({"messages": []}) is None
    assert extract_user_message({"messages": [msg(role="assistant")]}) is None
    assert extract_user_message(None) is None and extract_user_message("hi") is None


def test_extract_joins_text_parts_and_ignores_a_resume_map():
    message = {
        "role": "user",
        "contentParts": [
            {"mimeType": "text/plain", "data": {"inline": "line one"}},
            {"mimeType": "text/markdown", "data": {"inline": "line two"}},
            {"mimeType": "application/json", "data": {"inline": "{}"}},
            {"data": {"inline": 42}},
        ],
    }
    found = extract_user_message({"messages": [message]})
    assert found is not None
    assert user_text(found) == "line one\nline two"
    assert is_chat_input({"messages": []})
    assert not is_chat_input({"c1": {"sum": 1}}) and not is_chat_input(None)
    # A durable call's job output may be chat-shaped (a conversational agent
    # run as a process): it is still that call's result, not a user message.
    assert extract_user_message({"c1": {"messages": [message]}}) is None
    assert not is_chat_input({"c1": {"messages": [message]}})


def test_files_become_job_attachments():
    message = msg("Summarize this", contentParts=None)
    message["contentParts"] = [
        {"mimeType": "text/plain", "data": {"inline": "Summarize this"}},
        file_part(),
        {"mimeType": "image/png", "data": {"uri": "https://elsewhere/x.png"}},
        {"data": {"uri": "urn:uipath:cas:file:orchestrator:not-a-uuid"}},
    ]
    [attachment] = chat_attachments(message)
    assert str(attachment.id) == FILE_ID
    assert attachment.full_name == "report.pdf"
    assert attachment.mime_type == "application/pdf"
    # A message with only a file still counts.
    only_file = {"role": "user", "contentParts": [file_part()]}
    assert extract_user_message({"messages": [only_file]}) == only_file
    assert message_fingerprint(only_file).startswith("sha256:")


# --------------------------------------------------------------------------
# the mapper
# --------------------------------------------------------------------------
def _agent_item(item_id: str, text: str = "", phase: str | None = None) -> dict:
    return {"type": "agentMessage", "id": item_id, "text": text, "phase": phase}


def _mcp_item(item_id: str, status: str = "inProgress", **extra: Any) -> dict:
    return {
        "type": "mcpToolCall",
        "id": item_id,
        "server": "rooms",
        "tool": "find_rooms",
        "arguments": {"people": 6},
        "status": status,
        **extra,
    }


def test_deltas_stream_into_one_message():
    mapper = CodexChatEventMapper()
    events = mapper.on_item_started(_agent_item("i1", phase="commentary"))
    events += mapper.on_delta("i1", "Hel")
    events += mapper.on_delta("i1", "lo")
    events += mapper.on_item_completed(_agent_item("i1", "Hello"))
    events += mapper.close_turn()
    ps = payloads(events)
    assert [shape(p) for p in ps] == [
        "startMessage+startContentPart",
        "chunk",
        "chunk",
        "endContentPart",
        "endMessage",
    ]
    mid = ps[0]["messageId"]
    assert {p["messageId"] for p in ps} == {mid}
    assert ps[0]["startMessage"]["role"] == "assistant"
    assert ps[0]["startMessage"]["timestamp"].endswith("Z")
    assert ps[0]["contentPart"] == {
        "contentPartId": f"chunk-{mid}-0",
        "startContentPart": {"mimeType": "text/markdown"},
    }
    assert [p["contentPart"]["chunk"]["data"] for p in ps[1:3]] == ["Hel", "lo"]


def test_without_deltas_the_whole_text_is_one_chunk():
    mapper = CodexChatEventMapper()
    events = mapper.on_item_started(_agent_item("i1"))
    events += mapper.on_item_completed(_agent_item("i1", "All at once"))
    ps = payloads(events + mapper.close_turn())
    assert [shape(p) for p in ps] == [
        "startMessage+startContentPart",
        "chunk",
        "endContentPart",
        "endMessage",
    ]
    assert ps[1]["contentPart"]["chunk"]["data"] == "All at once"
    # A delta with no item/started opens the message lazily.
    mapper = CodexChatEventMapper()
    assert shape(payloads(mapper.on_delta("i9", "x"))[0]) == (
        "startMessage+startContentPart"
    )


def test_tool_calls_attach_to_the_message_and_end_it():
    mapper = CodexChatEventMapper()
    events = mapper.on_item_started(_agent_item("i1"))
    events += mapper.on_item_completed(_agent_item("i1", "Looking."))
    events += mapper.on_item_started(_mcp_item("call_1"))
    events += mapper.on_item_started(
        {
            "type": "commandExecution",
            "id": "cmd_1",
            "command": "ls -la",
            "status": "inProgress",
        }
    )
    events += mapper.on_item_completed(
        _mcp_item(
            "call_1",
            "completed",
            result={"content": [{"type": "text", "text": '{"room": "Beehive"}'}]},
        )
    )
    first = payloads(events)
    assert "endMessage" not in [shape(p) for p in first]  # the command still runs
    events = mapper.on_item_completed(
        {
            "type": "commandExecution",
            "id": "cmd_1",
            "command": "ls -la",
            "status": "failed",
            "exit_code": 2,
            "aggregated_output": "ls: nope",
        }
    )
    events += mapper.on_item_started(_agent_item("i2"))
    events += mapper.on_item_completed(_agent_item("i2", "Beehive fits."))
    ps = first + payloads(events + mapper.close_turn())
    assert [shape(p) for p in ps] == [
        "startMessage+startContentPart",
        "chunk",
        "endContentPart",
        "startToolCall",
        "executingToolCall",
        "startToolCall",
        "executingToolCall",
        "endToolCall",
        "endToolCall",
        "endMessage",
        "startMessage+startContentPart",
        "chunk",
        "endContentPart",
        "endMessage",
    ]
    m1, m2 = ps[0]["messageId"], ps[-1]["messageId"]
    assert m1 != m2
    assert all(p["messageId"] == m1 for p in ps[:10])
    start = ps[3]["toolCall"]
    assert start["toolCallId"] == "call_1"
    assert start["startToolCall"]["toolName"] == "find_rooms"
    assert start["startToolCall"]["input"] == {"people": 6}
    assert start["startToolCall"]["metaData"] == {"server": "rooms"}
    assert ps[4]["toolCall"]["executingToolCall"]["input"] == {"people": 6}
    shell = ps[5]["toolCall"]["startToolCall"]
    assert shell["toolName"] == "shell" and shell["input"] == {"command": "ls -la"}
    assert ps[7]["toolCall"]["endToolCall"]["output"] == {"room": "Beehive"}
    assert ps[7]["toolCall"]["endToolCall"]["isError"] is False
    assert ps[8]["toolCall"]["endToolCall"] == {
        "timestamp": ps[8]["toolCall"]["endToolCall"]["timestamp"],
        "output": "ls: nope",
        "isError": True,
    }


def test_a_tool_call_without_a_message_opens_a_bare_one():
    mapper = CodexChatEventMapper()
    events = mapper.on_item_started(_mcp_item("call_1"))
    events += mapper.on_item_completed(
        _mcp_item("call_1", "failed", error={"message": "boom"})
    )
    ps = payloads(events)
    assert [shape(p) for p in ps] == [
        "startMessage",
        "startToolCall",
        "executingToolCall",
        "endToolCall",
        "endMessage",
    ]
    assert ps[3]["toolCall"]["endToolCall"]["output"] == "boom"
    assert ps[3]["toolCall"]["endToolCall"]["isError"] is True
    assert payloads(mapper.close_turn()) == []


def test_reasoning_and_file_changes_are_not_chat_events():
    mapper = CodexChatEventMapper()
    for item in (
        {"type": "reasoning", "id": "r1", "summary": ["x"]},
        {"type": "fileChange", "id": "f1", "changes": [], "status": "completed"},
        {"type": "webSearch", "id": "w1", "query": "q"},
    ):
        assert mapper.on_item_started(item) == []
        assert mapper.on_item_completed(item) == []


def test_state_round_trips_and_ends_the_call_on_resume():
    mapper = CodexChatEventMapper()
    ps = payloads(
        mapper.on_item_started(_agent_item("i1"))
        + mapper.on_item_completed(_agent_item("i1", "Starting the job."))
        + mapper.on_item_started(
            {**_mcp_item("call_d"), "server": "uipath_tools", "tool": "a_plus_b"}
        )
        + mapper.on_item_started(_mcp_item("call_cut"))
    )
    mid = ps[0]["messageId"]
    # The suspension: a sibling cut off by it is cancelled, the durable kept.
    cancelled = payloads(mapper.cancel_calls(keep={"call_d"}))
    assert [shape(p) for p in cancelled] == ["endToolCall"]
    assert cancelled[0]["toolCall"]["endToolCall"]["cancelled"] is True
    state = json.loads(json.dumps(mapper.state()))
    assert state["tool_calls"] == {"call_d": mid}

    resumed = CodexChatEventMapper.from_state(state)
    ps = payloads(
        resumed.on_resumed_call("call_d", "a_plus_b", {}, '{"sum": 42}', False)
    )
    assert [shape(p) for p in ps] == ["endToolCall", "endMessage"]
    assert {p["messageId"] for p in ps} == {mid}
    assert ps[0]["toolCall"]["endToolCall"]["output"] == {"sum": 42}
    assert resumed.state() == {"tool_calls": {}, "open_messages": []}
    # Nothing saved for a call (an older state): shown whole.
    ps = payloads(
        CodexChatEventMapper.from_state({}).on_resumed_call(
            "c9", "a_plus_b", {"a": 1}, "oops", True
        )
    )
    assert [shape(p) for p in ps] == [
        "startMessage",
        "startToolCall",
        "executingToolCall",
        "endToolCall",
        "endMessage",
    ]


# --------------------------------------------------------------------------
# exchanges over a fake codex
# --------------------------------------------------------------------------
class FakeTurn:
    def __init__(self, thread: FakeThread, prompt: str) -> None:
        self.thread = thread
        self.prompt = prompt

    async def stream(self):
        thread = self.thread
        codex = thread.codex
        tid, turn = thread.id, f"turn-{len(codex.prompts)}"
        base = {"threadId": tid, "turnId": turn}
        for spec in codex.script(self.prompt, codex):
            kind, data = spec
            if kind == "do":  # a side effect mid-turn (an awaitable)
                await data
                continue
            if kind == "fail":  # the turn fails (codex reports an error)
                yield SimpleNamespace(
                    payload=TurnCompletedNotification.model_validate(
                        {
                            "threadId": tid,
                            "turn": {
                                "id": turn,
                                "items": [],
                                "status": "failed",
                                "error": {"message": data},
                            },
                        }
                    )
                )
                return
            if kind == "started":
                payload: Any = ItemStartedNotification.model_validate(
                    {**base, "item": data, "startedAtMs": 0}
                )
            elif kind == "completed":
                payload = ItemCompletedNotification.model_validate(
                    {**base, "item": data, "completedAtMs": 0}
                )
            else:
                payload = AgentMessageDeltaNotification.model_validate(
                    {**base, "itemId": data[0], "delta": data[1]}
                )
            yield SimpleNamespace(payload=payload)
        thread.append(self.prompt)
        yield SimpleNamespace(
            payload=TurnCompletedNotification.model_validate(
                {
                    "threadId": tid,
                    "turn": {"id": turn, "items": [], "status": "completed"},
                }
            )
        )


class FakeThread:
    def __init__(self, codex: FakeCodex, thread_id: str) -> None:
        self.codex = codex
        self.id = thread_id

    def rollout(self) -> Path:
        home = Path(self.codex.config.env["CODEX_HOME"])
        found = list(home.glob(f"sessions/**/rollout-*{self.id}.jsonl"))
        if found:
            return found[0]
        path = home / "sessions" / "2026" / "10" / "22" / f"rollout-x-{self.id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return path

    def history(self) -> list[str]:
        return [
            json.loads(line)["prompt"]
            for line in self.rollout().read_text().splitlines()
            if line
        ]

    def append(self, prompt: str) -> None:
        with self.rollout().open("a") as f:
            f.write(json.dumps({"prompt": prompt}) + "\n")

    async def turn(self, prompt: str, **kwargs: Any) -> FakeTurn:
        self.codex.prompts.append(prompt)
        self.codex.histories.append(self.history())
        return FakeTurn(self, prompt)


def reply(prompt: str, codex: FakeCodex) -> list[tuple[str, Any]]:
    """Default script: one agent message streamed in two deltas."""
    item = f"msg-{len(codex.prompts)}"
    text = f"You said: {prompt}"
    half = len(text) // 2
    return [
        ("started", _agent_item(item)),
        ("delta", (item, text[:half])),
        ("delta", (item, text[half:])),
        ("completed", _agent_item(item, text, "final_answer")),
    ]


class FakeCodex:
    """Stands in for ``openai_codex.AsyncCodex`` (class-level bookkeeping)."""

    instances: list[FakeCodex] = []
    script = staticmethod(reply)
    starts: list[str] = []
    resumes: list[str] = []
    prompts: list[str] = []
    histories: list[list[str]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.injected: list[dict[str, Any]] = []
        # ``thread/inject_items`` goes through the raw request of the client.
        self._client = SimpleNamespace(
            _sync=SimpleNamespace(
                _request_raw=lambda method, params: self.injected.append(params)
            )
        )
        FakeCodex.instances.append(self)

    async def __aenter__(self) -> FakeCodex:
        return self

    async def close(self) -> None:
        pass

    async def thread_start(self, **kwargs: Any) -> FakeThread:
        thread = FakeThread(self, str(uuid.uuid4()))
        FakeCodex.starts.append(thread.id)
        thread.rollout()
        return thread

    async def thread_resume(self, thread_id: str, **kwargs: Any) -> FakeThread:
        FakeCodex.resumes.append(thread_id)
        return FakeThread(self, thread_id)


class FakeShim:
    def __init__(self, **kwargs: Any) -> None:
        self.last_activity = 0.0
        self.inflight = 0
        self.mcp_url = "http://127.0.0.1:1/mcp"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def env(self) -> dict[str, str]:
        return {}

    def config_overrides(self) -> tuple[str, ...]:
        return ()


@pytest.fixture
def codex(monkeypatch):
    import openai_codex

    for name in ("instances", "starts", "resumes", "prompts", "histories"):
        monkeypatch.setattr(FakeCodex, name, [])
    monkeypatch.setattr(FakeCodex, "script", staticmethod(reply))
    monkeypatch.setattr(openai_codex, "AsyncCodex", FakeCodex)
    monkeypatch.setattr(runtime_mod, "CodexGatewayShim", FakeShim)
    return FakeCodex


@pytest.fixture
def chat_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(_project(tmp_path, CHAT_MAIN))
    return tmp_path


async def _exchange(
    project: Path,
    input: Any,
    *,
    resume: bool,
    run_dir: str = "__uipath",
    factory: UiPathCodexRuntimeFactory | None = None,
) -> tuple[list[Any], UiPathRuntimeResult]:
    """One exchange; a new factory (a new job) unless one is given (``uipath dev``)."""
    own = factory is None
    if factory is None:
        factory = UiPathCodexRuntimeFactory(
            UiPathRuntimeContext(
                runtime_dir=str(project / run_dir), state_file="state.db", resume=resume
            )
        )
    runtime = await factory.new_runtime("agent", RT)
    try:
        events = [
            e async for e in runtime.stream(input, UiPathStreamOptions(resume=resume))
        ]
    finally:
        await runtime.dispose()
        if own:
            await factory.dispose()
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    return events[:-1], result


async def _store(project: Path, run_dir: str = "__uipath"):
    storage = SqliteResumableStorage(str(project / run_dir / "state.db"))
    return storage, CodexSessionStore(storage, RT)


async def test_first_exchange_starts_a_thread_and_suspends_on_nothing(chat_dir, codex):
    events, result = await _exchange(
        chat_dir, {"messages": [msg("Hi there")]}, resume=False
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert result.output is None and not result.triggers
    assert len(codex.starts) == 1 and codex.resumes == []
    assert codex.prompts == ["Hi there"]
    ps = payloads(events)
    assert (
        "".join(p["contentPart"]["chunk"]["data"] for p in ps if shape(p) == "chunk")
        == "You said: Hi there"
    )
    # No plain-dict message event (the base runtime's) reaches the host.
    assert all(
        isinstance(e.payload, UiPathConversationMessageEvent)
        for e in events
        if isinstance(e, UiPathRuntimeMessageEvent)
    )
    storage, store = await _store(chat_dir)
    try:
        thread = await store.get_thread()
        assert thread is not None and thread.thread_id == codex.starts[0]
        assert thread.rollout_rel is not None
        rollout = await store.get_rollout()
        assert rollout is not None and b"Hi there" in rollout.content()
        assert (await store.get_chat())["user_message"].startswith("sha256:")
    finally:
        await storage.dispose()
    # A run-scoped, hermetic codex home.
    config = codex.instances[0].config
    assert config.env["CODEX_HOME"] == str(chat_dir / "__uipath" / "codex_home")
    assert any("features.plugins=false" in o for o in config.config_overrides)


@pytest.mark.parametrize("resume", [False, True], ids=["dev", "orchestrator"])
async def test_next_exchange_resumes_the_thread_with_the_new_message(
    chat_dir, codex, resume, monkeypatch
):
    # ``uipath dev`` runs every exchange on one factory, without resume.
    dev = _factory(chat_dir) if not resume else None
    await _exchange(chat_dir, {"messages": [msg("one")]}, resume=False, factory=dev)
    cleared: list[bool] = []
    original = CodexSessionStore.clear_all

    async def clear_all(self):
        cleared.append(True)
        await original(self)

    monkeypatch.setattr(CodexSessionStore, "clear_all", clear_all)
    # The whole history replayed by the host: only the newest message counts.
    history = {
        "messages": [msg("one"), msg("You said: one", role="assistant"), msg("two")]
    }
    if resume:  # a new job: only state.db carries over
        os.makedirs(chat_dir / "job2")
        os.replace(chat_dir / "__uipath" / "state.db", chat_dir / "job2" / "state.db")
    try:
        events, result = await _exchange(
            chat_dir,
            history,
            resume=resume,
            run_dir="job2" if resume else "__uipath",
            factory=dev,
        )
    finally:
        if dev is not None:
            await dev.dispose()
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.resumes == codex.starts  # the same thread
    assert codex.prompts == ["one", "two"]
    assert codex.histories[-1] == ["one"]  # restored rollout
    assert cleared == []


async def test_an_exchange_without_a_user_message_is_refused(chat_dir, codex):
    with pytest.raises(UiPathCodexRuntimeError) as info:
        await _exchange(chat_dir, {"messages": []}, resume=False)
    assert "INPUT_INVALID" in str(info.value.error_info.code)


async def test_chat_files_reach_the_prompt_and_stay_known(chat_dir, codex):
    only_file = {"role": "user", "contentParts": [file_part()]}
    await _exchange(chat_dir, {"messages": [only_file]}, resume=False)
    prompt = codex.prompts[0]
    assert prompt.startswith(ATTACHMENT_ONLY_TEXT)
    assert FILE_ID in prompt and "report.pdf" in prompt
    await _exchange(chat_dir, {"messages": [msg("And now?")]}, resume=True)
    assert codex.prompts[1] == "And now?"
    storage = SqliteResumableStorage(str(chat_dir / "__uipath" / "state.db"))
    try:
        raw = await storage.get_value(RT, "codex_run", "attachments")
    finally:
        await storage.dispose()
    assert [a["ID"] for a in raw["value"]] == [FILE_ID]


class FakeBridge:
    """A chat bridge that records what the host sends."""

    def __init__(self) -> None:
        self.messages: list[UiPathConversationMessageEvent] = []
        self.exchange_ended = 0
        self.errors: list[Exception] = []
        self.interrupts: list[Any] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def emit_message_event(self, event: Any) -> None:
        self.messages.append(event)

    async def emit_interrupt_event(self, trigger: Any) -> None:
        self.interrupts.append(trigger)

    async def emit_executing_tool_call_event(self, **kwargs: Any) -> None:
        pass

    async def emit_exchange_end_event(self) -> None:
        self.exchange_ended += 1

    async def emit_exchange_error_event(self, error: Exception) -> None:
        self.errors.append(error)

    async def wait_for_resume(self) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("no API trigger expected")


async def test_the_chat_host_gets_message_events_and_the_exchange_end(chat_dir, codex):
    def script(prompt: str, codex: FakeCodex) -> list[tuple[str, Any]]:
        return [
            ("started", _agent_item("c1", phase="commentary")),
            ("completed", _agent_item("c1", "Checking rooms.", "commentary")),
            ("started", _mcp_item("call_1")),
            (
                "completed",
                _mcp_item(
                    "call_1",
                    "completed",
                    result={"content": [{"type": "text", "text": "Beehive"}]},
                ),
            ),
            ("started", {"type": "reasoning", "id": "r1", "summary": []}),
            ("completed", {"type": "reasoning", "id": "r1", "summary": ["think"]}),
            *reply(prompt, codex),
        ]

    codex.script = staticmethod(script)
    factory = _factory(chat_dir)
    bridge = FakeBridge()
    try:
        runtime = await factory.new_runtime("agent", RT)
        host = UiPathChatRuntime(delegate=runtime, chat_bridge=bridge)  # type: ignore[arg-type]
        events = [
            e
            async for e in host.stream(
                {"messages": [msg("A room for 6?")]}, UiPathStreamOptions()
            )
        ]
        await runtime.dispose()
    finally:
        await factory.dispose()
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert bridge.exchange_ended == 1 and bridge.errors == []
    sent = [m.model_dump(by_alias=True, exclude_none=True) for m in bridge.messages]
    assert all(isinstance(m, UiPathConversationMessageEvent) for m in bridge.messages)
    assert [shape(p) for p in sent] == [
        "startMessage+startContentPart",
        "chunk",
        "endContentPart",
        "startToolCall",
        "executingToolCall",
        "endToolCall",
        "endMessage",
        "startMessage+startContentPart",
        "chunk",
        "chunk",
        "endContentPart",
        "endMessage",
    ]
    # Tools and reasoning still reach the debugger as state events.
    states = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
    assert [s.metadata["item_type"] for s in states if s.metadata] == [
        "mcpToolCall",
        "reasoning",
    ]


# --------------------------------------------------------------------------
# a user message while a durable call is pending
# --------------------------------------------------------------------------
DURABLE_CHAT = UiPathCodexAgent(
    model="m", tools=[uipath_process_tool("a_plus_b")], base_instructions="b"
)


@pytest.fixture
def world(monkeypatch) -> FakeOrchestrator:
    w = FakeOrchestrator()
    import uipath.platform.resume_triggers._protocol as protocol

    import uipath_codex_sdk.tools as tools

    monkeypatch.setattr(tools, "_uipath_client", w.client)
    monkeypatch.setattr(protocol, "UiPath", w.client)
    monkeypatch.setattr(runtime_mod, "RESUSPEND_BACKOFF_SEC", 0.0)
    return w


async def _parked(storage, world) -> PendingCall:
    """State as left by a chat exchange parked on one durable call."""
    store = CodexSessionStore(storage, RT)
    await store.set_entrypoint(EntrypointRecord("agent"))
    rel = "sessions/2026/10/22/rollout-x-thread-1.jsonl"
    await store.set_thread(ThreadRecord("thread-1", rel))
    await store.set_rollout(encode_rollout(rel, b'{"prompt": "add"}\n'))
    job = Job.model_validate({"Id": 1, "Key": "job-1", "State": "Running"})
    world.jobs["job-1"] = job
    value = WaitJobRaw(job=job, process_folder_key="fk")
    call = PendingCall("c1", "a_plus_b", {"a": 1}, STATE_STARTED, "WaitJobRaw", value)
    await store.put_pending(call)
    await store.set_dangling([DanglingCall("c1", "function_call", durable=True)])
    await store.set_chat(
        {
            "tool_calls": {"c1": "m-1"},
            "open_messages": [{"id": "m-1", "text_open": False, "calls": ["c1"]}],
            "user_message": "msg-1",
            "parked": True,
            "exchange_id": "ex-1",
        }
    )
    return call


def _chat_delegate(
    storage, tmp_path, exchange_id: str | None = None
) -> UiPathCodexChatRuntime:
    return UiPathCodexChatRuntime(
        agent=DURABLE_CHAT,
        runtime_id=RT,
        entrypoint="agent",
        file_path=str(tmp_path / "main.py"),
        storage=storage,
        exchange_id=exchange_id,
    )


@pytest.fixture
async def storage(tmp_path):
    s = SqliteResumableStorage(str(tmp_path / "state.db"))
    yield s
    await s.dispose()


@pytest.mark.parametrize("wrapper", ["stock", "codex"])
async def test_a_message_while_a_call_is_pending_resuspends(
    storage, world, tmp_path, codex, wrapper
):
    """Re-suspended on the same call, its trigger saved again; nothing runs.

    The stock ``UiPathResumableRuntime`` deletes the stored trigger when it
    is given explicit input, so the runtime returns the pending call again
    and the trigger is minted anew; the codex wrapper keeps it stored.
    """
    await _parked(storage, world)
    handler = UiPathResumeTriggerHandler()
    seed = UiPathResumableRuntime(
        delegate=_chat_delegate(storage, tmp_path),
        storage=storage,
        trigger_manager=handler,
        runtime_id=RT,
    )
    await seed._handle_suspension(
        UiPathRuntimeResult(
            output={
                "c1": (await CodexSessionStore(storage, RT).get_pending())[
                    "c1"
                ].suspend_value()
            },
            status=UiPathRuntimeStatus.SUSPENDED,
        )
    )
    [before] = await storage.get_triggers(RT)
    kwargs: dict[str, Any] = {
        "delegate": _chat_delegate(storage, tmp_path),
        "storage": storage,
        "trigger_manager": handler,
        "runtime_id": RT,
    }
    runtime = (
        UiPathResumableRuntime(**kwargs)
        if wrapper == "stock"
        else UiPathCodexResumableRuntime(**kwargs, conversational=True)
    )
    events = [
        e
        async for e in runtime.stream(
            {"messages": [msg("Still there?", messageId="msg-2")]},
            UiPathStreamOptions(resume=True),
        )
    ]
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in result.triggers or []] == ["c1"]
    [after] = await storage.get_triggers(RT)
    assert after.interrupt_id == "c1"
    assert after.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert after.item_key == before.item_key == "job-1"
    assert codex.instances == []  # codex never ran
    assert payloads(events) == []
    store = CodexSessionStore(storage, RT)
    assert list(await store.get_pending()) == ["c1"]
    assert (await store.get_chat())["tool_calls"] == {"c1": "m-1"}

    # The job finishes: the resume ends the call on its message, continues.
    world.finish("job-1", "Successful", {"sum": 2})
    events = [e async for e in runtime.stream(None, UiPathStreamOptions(resume=True))]
    ps = payloads(events)
    assert [shape(p) for p in ps][:2] == ["endToolCall", "endMessage"]
    assert {p["messageId"] for p in ps[:2]} == {"m-1"}
    assert ps[0]["toolCall"]["endToolCall"]["output"] == {"sum": 2}
    assert codex.prompts == [runtime_mod.CONTINUE_PROMPT]
    assert codex.histories == [["add"]]  # the restored rollout
    [injected] = codex.instances[0].injected
    assert injected["items"] == [
        {"type": "function_call_output", "call_id": "c1", "output": '{"sum": 2}'}
    ]
    result = as_result(events[-1])
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers


async def _seed_trigger(storage, tmp_path) -> None:
    """Store the trigger of the parked call, as the resumable wrapper did."""
    seed = UiPathResumableRuntime(
        delegate=_chat_delegate(storage, tmp_path),
        storage=storage,
        trigger_manager=UiPathResumeTriggerHandler(),
        runtime_id=RT,
    )
    pending = await CodexSessionStore(storage, RT).get_pending()
    await seed._handle_suspension(
        UiPathRuntimeResult(
            output={"c1": pending["c1"].suspend_value()},
            status=UiPathRuntimeStatus.SUSPENDED,
        )
    )


async def test_a_chat_shaped_call_result_is_no_user_message(
    storage, world, tmp_path, codex
):
    """A resume value of a durable call that holds chat messages (the output
    of a conversational agent run as a process, given as ``{call_id: value}``)
    is that call's result: the exchange continues with it instead of taking
    it for a new user message and suspending again on the call."""
    await _parked(storage, world)
    await _seed_trigger(storage, tmp_path)
    child = {"messages": [msg("Hello from the child agent")]}
    runtime = UiPathCodexResumableRuntime(
        delegate=_chat_delegate(storage, tmp_path),
        storage=storage,
        trigger_manager=UiPathResumeTriggerHandler(),
        runtime_id=RT,
        conversational=True,
    )

    async def drain() -> list[Any]:
        return [
            e
            async for e in runtime.stream(
                {"c1": child}, UiPathStreamOptions(resume=True)
            )
        ]

    events = await asyncio.wait_for(drain(), 30)
    result = as_result(events[-1])
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.prompts == [runtime_mod.CONTINUE_PROMPT]
    [injected] = codex.instances[0].injected
    [item] = injected["items"]
    assert item["call_id"] == "c1" and json.loads(item["output"]) == child
    assert payloads(events)[0]["toolCall"]["endToolCall"]["output"] == child
    assert await CodexSessionStore(storage, RT).get_pending() == {}
    assert not await storage.get_triggers(RT)


@pytest.mark.parametrize("first", [True, False], ids=["first", "later"])
async def test_an_exchange_that_faulted_leaves_nothing_parked(
    chat_dir, codex, first, caplog
):
    """The pending call of a job an exchange started before it faulted is not
    waited on: the next message reaches the model (``uipath dev``: one
    factory, the state kept)."""
    factory = _factory(chat_dir)
    try:
        if not first:
            await _exchange(
                chat_dir, {"messages": [msg("one")]}, resume=False, factory=factory
            )
        store = CodexSessionStore(await factory._get_storage(), RT)
        job = Job.model_validate({"Id": 1, "Key": "job-1", "State": "Running"})
        orphan = PendingCall(
            "c1",
            "a_plus_b",
            {"a": 1},
            STATE_STARTED,
            "WaitJobRaw",
            WaitJobRaw(job=job, process_folder_key="fk"),
        )

        def faulting(prompt: str, codex: FakeCodex) -> list[tuple[str, Any]]:
            # The job is started (its call saved), then the turn fails.
            return [("do", store.put_pending(orphan)), ("fail", "model down")]

        codex.script = staticmethod(faulting)
        with pytest.raises(UiPathCodexRuntimeError):
            await _exchange(
                chat_dir, {"messages": [msg("add")]}, resume=False, factory=factory
            )
        assert list(await store.get_pending()) == ["c1"]
        codex.script = staticmethod(reply)
        events, result = await _exchange(
            chat_dir, {"messages": [msg("three")]}, resume=False, factory=factory
        )
        assert await store.get_pending() == {}
    finally:
        await factory.dispose()
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.prompts[-1] == "three"
    text = "".join(
        p["contentPart"]["chunk"]["data"]
        for p in payloads(events)
        if shape(p) == "chunk"
    )
    assert text == "You said: three"
    if first:
        # The faulted first exchange saved no conversation: a new thread.
        assert len(codex.starts) == 2 and codex.resumes == []
        assert "has no saved conversation" in caplog.text
    else:
        assert codex.resumes == [codex.starts[0]] * 2
        assert codex.histories[-1] == ["one"]  # the last checkpoint


async def test_a_thread_without_a_saved_rollout_starts_a_new_one(
    chat_dir, codex, caplog
):
    os.makedirs(chat_dir / "__uipath")
    storage, store = await _store(chat_dir)
    try:
        await store.set_entrypoint(EntrypointRecord("agent"))
        await store.set_thread(ThreadRecord("thread-lost"))
    finally:
        await storage.dispose()
    _, result = await _exchange(chat_dir, {"messages": [msg("hi")]}, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.resumes == [] and len(codex.starts) == 1
    assert codex.prompts == ["hi"]
    assert "thread-lost has no saved conversation" in caplog.text
    storage, store = await _store(chat_dir)
    try:
        thread = await store.get_thread()
        assert thread is not None and thread.thread_id == codex.starts[0]
    finally:
        await storage.dispose()


async def test_a_new_exchange_clears_a_stale_trigger(chat_dir, codex):
    """A trigger left by an earlier suspension would keep the exchange open
    (the host waits on it), and the wrapper would hand it back."""
    from uipath.core.triggers import UiPathResumeTrigger, UiPathResumeTriggerType

    await _exchange(chat_dir, {"messages": [msg("one")]}, resume=False)
    storage, _ = await _store(chat_dir)
    try:
        stale = UiPathResumeTrigger(
            interrupt_id="old-call",
            trigger_type=UiPathResumeTriggerType.JOB,
            trigger_name=UiPathResumeTriggerName.JOB_RAW,
            item_key="old-job",
        )
        await storage.save_triggers(RT, [stale])
        assert len(await storage.get_triggers(RT) or []) == 1
    finally:
        await storage.dispose()
    _, result = await _exchange(chat_dir, {"messages": [msg("two")]}, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.prompts == ["one", "two"]
    storage, _ = await _store(chat_dir)
    try:
        assert not await storage.get_triggers(RT)
    finally:
        await storage.dispose()


async def test_a_message_after_every_call_finished_is_the_prompt(
    storage, world, tmp_path, codex
):
    """Parked with the results of every call (none still running): the
    results go in and the new message is the prompt. The old message is the
    parked exchange's: it ends without chat events in this one."""
    await _parked(storage, world)
    store = CodexSessionStore(storage, RT)
    await store.remove_pending(["c1"])
    await store.set_completed({"c1": '{"sum": 2}'})
    runtime = _chat_delegate(storage, tmp_path)
    events = [
        e
        async for e in runtime.stream(
            {"messages": [msg("And now?", messageId="msg-2")]},
            UiPathStreamOptions(resume=True),
        )
    ]
    result = as_result(events[-1])
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.prompts == ["And now?"]
    [injected] = codex.instances[0].injected
    assert injected["items"] == [
        {"type": "function_call_output", "call_id": "c1", "output": '{"sum": 2}'}
    ]
    ps = payloads(events)
    assert ps and all(p["messageId"] != "m-1" for p in ps)
    assert [shape(p) for p in ps][0] == "startMessage+startContentPart"
    chat = await store.get_chat()
    assert chat["user_message"] == "msg-2"
    assert chat["parked"] is False and chat["open_messages"] == []
    assert await store.get_completed() == {} and await store.get_dangling() == []


@pytest.mark.parametrize("exchange", ["ex-1", "ex-2"])
async def test_the_parked_message_ends_only_in_its_own_exchange(
    storage, world, tmp_path, codex, exchange
):
    """The chat host sends every event under the exchange it serves: the
    parked message's ``endToolCall`` / ``endMessage`` go out only when the
    resume serves the exchange that parked (``ex-1``)."""
    await _parked(storage, world)
    world.finish("job-1", "Successful", {"sum": 2})
    runtime = _chat_delegate(storage, tmp_path, exchange_id=exchange)
    events = [e async for e in runtime.stream({}, UiPathStreamOptions(resume=True))]
    result = as_result(events[-1])
    assert result.status == UiPathRuntimeStatus.SUSPENDED and not result.triggers
    assert codex.prompts == [runtime_mod.CONTINUE_PROMPT]
    shapes = [shape(p) for p in payloads(events)]
    ids = [p["messageId"] for p in payloads(events)]
    if exchange == "ex-1":
        assert shapes[:2] == ["endToolCall", "endMessage"] and ids[:2] == ["m-1"] * 2
        assert shapes[2] == "startMessage+startContentPart"
    else:
        assert "m-1" not in ids
        assert shapes[0] == "startMessage+startContentPart"
    assert shapes[-1] == "endMessage"
    # The durable result is a state event either way.
    states = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
    assert any((s.metadata or {}).get("call_id") == "c1" for s in states)


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------
def test_the_exchange_span_is_named_as_low_code_names_it():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    import uipath_codex_sdk.runtime.telemetry as telemetry

    original = telemetry.get_tracer
    telemetry.get_tracer = lambda: provider.get_tracer("t")  # type: ignore[assignment]
    try:
        for conversational in (True, False):
            start_agent_run(
                "agent",
                agent_name="Rooms",
                model="m",
                input={"messages": []},
                system_prompt="s",
                user_prompt="hi",
                input_schema={},
                output_schema={},
                resumed=True,
                is_conversational=conversational,
                extra={"conversationId": "c1", "exchangeId": None},
            ).end()
    finally:
        telemetry.get_tracer = original  # type: ignore[assignment]
    chat, plain = exporter.get_finished_spans()
    assert chat.name == "Conversational agent run - Rooms"
    assert plain.name == "Agent run - agent"
    assert chat.attributes is not None and plain.attributes is not None
    assert chat.attributes["isConversational"] is True
    assert chat.attributes["conversationId"] == "c1"
    assert "exchangeId" not in chat.attributes and "userPrompt" not in chat.attributes
    assert plain.attributes["isConversational"] is False
    assert plain.attributes["userPrompt"] == "hi"
    # Low-code's agent-run span has no ``resumed``; a plain run keeps it.
    assert "resumed" not in chat.attributes
    assert plain.attributes["resumed"] is True
