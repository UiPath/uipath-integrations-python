"""Tests for the conversational runtime: event ordering, suspension, session resume."""

from __future__ import annotations

from pathlib import Path

from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.events import UiPathRuntimeMessageEvent

from tests.conftest import make_result_message
from uipath_claude_sdk.runtime.conversational_runtime import (
    UiPathClaudeSDKConversationalRuntime,
)
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage


def _make_runtime(agent, tmp_path: Path, storage: SqliteResumableStorage):
    return UiPathClaudeSDKConversationalRuntime(
        agent=agent,
        session_store=ClaudeSessionStore(storage, "rt-1"),
        workspace_root=tmp_path / "workspace",
        runtime_id="rt-1",
    )


def _conversation_input(text: str) -> dict:
    return {
        "messages": [{"role": "user", "contentParts": [{"data": {"inline": text}}]}]
    }


async def test_exchange_streams_and_suspends(plain_agent, fake_client, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    events = [event async for event in runtime.stream(_conversation_input("hi there"))]

    assert fake_client.last_query == "hi there"

    # Final event is a SUSPENDED result with one interrupt entry.
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert len(result.output) == 1

    # Message event lifecycle: start (with content part start) → chunk → end.
    message_events = [
        e.payload for e in events if isinstance(e, UiPathRuntimeMessageEvent)
    ]
    assert message_events[0].start is not None
    assert message_events[0].start.role == "assistant"
    assert message_events[0].content_part.start is not None
    assert message_events[1].content_part.chunk.data == "Hello"
    assert message_events[-1].end is not None

    # Session id persisted for the next exchange.
    session_store = ClaudeSessionStore(storage, "rt-1")
    assert await session_store.get_session_id() == "session-1"
    await storage.dispose()


async def test_resume_reattaches_session(plain_agent, fake_client, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    async for _ in runtime.stream(_conversation_input("first message")):
        pass
    assert fake_client.last_options.resume is None

    fake_client.scripted_messages = [
        make_result_message(result="again", session_id="session-2")
    ]

    # Second exchange: input arrives as the resume map {interrupt_id: resume_data}.
    resume_input = {"interrupt-1": _conversation_input("second message")}
    async for _ in runtime.stream(resume_input):
        pass

    assert fake_client.last_query == "second message"
    assert fake_client.last_options.resume == "session-1"

    session_store = ClaudeSessionStore(storage, "rt-1")
    assert await session_store.get_session_id() == "session-2"
    await storage.dispose()


async def test_workspace_is_stable_across_exchanges(plain_agent, fake_client, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    async for _ in runtime.stream(_conversation_input("one")):
        pass
    first_cwd = fake_client.last_options.cwd

    async for _ in runtime.stream({"i": _conversation_input("two")}):
        pass
    assert fake_client.last_options.cwd == first_cwd
    assert Path(first_cwd).exists()
    await storage.dispose()


async def test_conversational_schema(plain_agent, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)
    schema = await runtime.get_schema()
    assert "messages" in schema.input["properties"]
    assert "messages" in schema.output["properties"]
    await storage.dispose()
