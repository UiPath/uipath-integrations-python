"""Tests for the conversational runtime: event ordering, suspension, session resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
)
from uipath.core.chat import UiPathConversationMessageEvent
from uipath.core.triggers import UiPathResumeTrigger
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.chat import UiPathChatRuntime
from uipath.runtime.events import UiPathRuntimeMessageEvent

from tests.conftest import make_result_message, make_session_paths
from uipath_claude_sdk.interrupts import PendingSuspend
from uipath_claude_sdk.runtime.conversational_runtime import (
    UiPathClaudeSDKConversationalRuntime,
)
from uipath_claude_sdk.runtime.runtime import RESUME_PROMPT
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage


def _make_runtime(agent, tmp_path: Path, storage: SqliteResumableStorage):
    return UiPathClaudeSDKConversationalRuntime(
        agent=agent,
        session_store=ClaudeSessionStore(storage, "rt-1"),
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
    )


def _conversation_input(text: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "contentParts": [{"data": {"inline": text}}]}]
    }


class RecordingChatBridge:
    """A chat host that records what an exchange told it."""

    def __init__(self) -> None:
        self.messages: list[UiPathConversationMessageEvent] = []
        self.errors: list[Exception] = []
        self.exchange_ended = False

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def emit_message_event(
        self, message_event: UiPathConversationMessageEvent
    ) -> None:
        self.messages.append(message_event)

    async def emit_interrupt_event(self, resume_trigger: UiPathResumeTrigger) -> None:
        return None

    async def emit_executing_tool_call_event(
        self, tool_call_id: str, tool_input: dict[str, Any] | None = None
    ) -> None:
        return None

    async def emit_exchange_end_event(self) -> None:
        self.exchange_ended = True

    async def emit_exchange_error_event(self, error: Exception) -> None:
        self.errors.append(error)

    async def wait_for_resume(self) -> dict[str, Any]:
        raise AssertionError("the host was asked to answer a turn that was over")


def _chat_stack(
    agent,
    tmp_path: Path,
    storage: SqliteResumableStorage,
    bridge: RecordingChatBridge,
) -> UiPathChatRuntime:
    """The runtime stack a conversational job actually runs on."""
    return UiPathChatRuntime(
        delegate=UiPathResumableRuntime(
            delegate=_make_runtime(agent, tmp_path, storage),
            storage=storage,
            trigger_manager=UiPathResumeTriggerHandler(),
            runtime_id="rt-1",
        ),
        chat_bridge=bridge,
    )


async def test_exchange_streams_and_suspends(plain_agent, fake_client, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    events = [event async for event in runtime.stream(_conversation_input("hi there"))]

    assert fake_client.last_query == "hi there"

    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert not result.output

    message_events = [
        e.payload for e in events if isinstance(e, UiPathRuntimeMessageEvent)
    ]
    assert message_events[0].start is not None
    assert message_events[0].start.role == "assistant"
    assert message_events[0].content_part.start is not None
    assert message_events[1].content_part.chunk.data == "Hello"
    assert message_events[-1].end is not None

    session_store = ClaudeSessionStore(storage, "rt-1")
    assert await session_store.get_session_id() == "session-1"
    await storage.dispose()


async def test_exchange_waits_for_background_subagent_follow_up(
    plain_agent, fake_client, tmp_path
):
    fake_client.scripted_messages = [
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="agent-1",
            description="review",
            uuid="task-start-1",
            session_id="session-1",
            task_type="local_agent",
        ),
        make_result_message(result="Review launched."),
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="agent-1",
            status="completed",
            output_file="agent-1.output",
            summary="Review complete",
            uuid="task-end-1",
            session_id="session-1",
        ),
        AssistantMessage(
            content=[TextBlock(text="Review complete: all good.")],
            model="claude-sonnet-4-5",
        ),
        make_result_message(result="Review complete: all good."),
    ]
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    events = [event async for event in runtime.stream(_conversation_input("review"))]

    chunks = [
        event.payload.content_part.chunk.data
        for event in events
        if isinstance(event, UiPathRuntimeMessageEvent)
        and event.payload.content_part is not None
        and event.payload.content_part.chunk is not None
    ]
    assert chunks == ["Review complete: all good."]
    assert isinstance(events[-1], UiPathRuntimeResult)
    assert events[-1].status == UiPathRuntimeStatus.SUSPENDED
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


async def test_next_exchange_carries_the_user_message(
    plain_agent, fake_client, tmp_path
):
    """Every exchange after the first arrives as a resume, message and all.

    The base runtime replaces the prompt with a fixed continuation whenever it
    resumes, because there a resume only ever delivers a parked tool call. A
    conversation resumes to say something new, so this pins that the second
    turn reaches the model instead of "Continue.".
    """
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    async for _ in runtime.stream(_conversation_input("first message")):
        pass

    fake_client.scripted_messages = [
        make_result_message(result="again", session_id="session-1")
    ]
    resume_input = {"interrupt-1": _conversation_input("second message")}
    async for _ in runtime.stream(
        resume_input, UiPathStreamOptions(resume=True, execution_id="x")
    ):
        pass

    assert fake_client.last_query == "second message"
    assert fake_client.last_options.resume == "session-1"
    await storage.dispose()


async def test_resume_of_a_parked_call_adds_no_prompt(
    plain_agent, fake_client, tmp_path
):
    """A payload answering an interrupt is the tool's result, not a new turn."""
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    session_store = ClaudeSessionStore(storage, "rt-1")
    await session_store.set_session_id("session-1")
    await session_store.set_pending_suspend(
        PendingSuspend(
            interrupt_id="toolu_1",
            tool_name="mcp__approvals__ask",
            value="Approve?",
            tool_use_id="toolu_1",
        )
    )
    runtime = _make_runtime(plain_agent, tmp_path, storage)

    async for _ in runtime.stream(
        {"toolu_1": {"answer": "yes"}},
        UiPathStreamOptions(resume=True, execution_id="x"),
    ):
        pass

    assert fake_client.last_query == RESUME_PROMPT
    await storage.dispose()


async def test_transcript_is_stored_after_every_exchange(
    plain_agent, fake_client, tmp_path
):
    """Only the state database crosses a suspension, so each turn stores one."""
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)
    transcript = runtime._session_paths.transcript_path("session-1")
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"type": "user"}\n', encoding="utf-8")

    async for _ in runtime.stream(_conversation_input("hi")):
        pass

    record = await ClaudeSessionStore(storage, "rt-1").get_transcript()
    assert record is not None
    assert record.session_id == "session-1"
    await storage.dispose()


async def test_a_finished_exchange_is_closed_on_the_chat_host(
    plain_agent, fake_client, tmp_path, storage
):
    """The turn is only over once the host has been told so.

    ``UiPathChatRuntime`` ends an exchange on a result that carries no
    interrupt, and treats one carrying an API interrupt as a tool call it still
    owes an answer to. Suspending on a synthetic interrupt therefore leaves the
    composer waiting on a turn that already finished.
    """
    bridge = RecordingChatBridge()
    events = [
        event
        async for event in _chat_stack(plain_agent, tmp_path, storage, bridge).stream(
            _conversation_input("hi")
        )
    ]

    assert bridge.exchange_ended
    assert not bridge.errors
    assert bridge.messages

    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert not result.triggers


async def test_the_next_message_reaches_the_agent_through_the_chat_host(
    plain_agent, fake_client, tmp_path, storage
):
    """A second exchange is a second process over the same state database."""
    async for _ in _chat_stack(
        plain_agent, tmp_path, storage, RecordingChatBridge()
    ).stream(_conversation_input("first message")):
        pass

    fake_client.scripted_messages = [
        make_result_message(result="again", session_id="session-1")
    ]
    bridge = RecordingChatBridge()
    async for _ in _chat_stack(plain_agent, tmp_path, storage, bridge).stream(
        _conversation_input("second message"), UiPathStreamOptions(resume=True)
    ):
        pass

    assert fake_client.last_query == "second message"
    assert fake_client.last_options.resume == "session-1"
    assert bridge.exchange_ended
    assert not bridge.errors


async def test_conversational_schema(plain_agent, tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = _make_runtime(plain_agent, tmp_path, storage)
    schema = await runtime.get_schema()
    assert "messages" in schema.input["properties"]
    assert "messages" in schema.output["properties"]
    await storage.dispose()
