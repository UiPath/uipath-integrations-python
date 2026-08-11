"""Tests for what `uipath debug` needs from this runtime.

The debug command wraps the runtime in ``UiPathDebugRuntime``. These tests
drive that real stack over a scripted bridge, so the contract is checked
rather than assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pytest
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.debug import UiPathBreakpointResult, UiPathDebugRuntime
from uipath.runtime.events import UiPathRuntimeStateEvent

from tests.conftest import make_session_paths
from tests.test_suspend_resume import InterruptClient, hitl_options  # noqa: F401
from uipath_claude_sdk import ClaudeAgent
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage


class ScriptedBridge:
    """A debug bridge that answers the first wait and quits after that."""

    def __init__(self, resume_payload: Any = None) -> None:
        self.log: list[str] = []
        self.state_events: list[UiPathRuntimeStateEvent] = []
        self.resume_payload = resume_payload
        self._resumes = 0

    async def connect(self) -> None:
        self.log.append("connect")

    async def disconnect(self) -> None:
        self.log.append("disconnect")

    async def emit_execution_started(self, **kwargs: Any) -> None:
        self.log.append("started")

    async def emit_state_update(self, state_event: UiPathRuntimeStateEvent) -> None:
        self.state_events.append(state_event)

    async def emit_breakpoint_hit(self, breakpoint_result: UiPathBreakpointResult):
        self.log.append("breakpoint")

    async def emit_execution_suspended(self, runtime_result: UiPathRuntimeResult):
        trigger = runtime_result.trigger
        assert trigger is not None
        self.log.append(f"suspended:{trigger.trigger_type.name}")

    async def emit_execution_resumed(self, resume_data: Any) -> None:
        self.log.append("resumed")

    async def emit_execution_completed(self, runtime_result: UiPathRuntimeResult):
        self.log.append(f"completed:{runtime_result.status.name}")

    async def emit_execution_error(self, error: str) -> None:
        self.log.append(f"error:{error}")

    async def wait_for_resume(self) -> Any:
        self._resumes += 1
        return self.resume_payload

    async def wait_for_terminate(self) -> None:
        return None

    def get_breakpoints(self) -> list[str] | Literal["*"]:
        return []


def _debug_stack(
    agent: ClaudeAgent,
    tmp_path: Path,
    storage: SqliteResumableStorage,
    bridge: ScriptedBridge,
) -> UiPathDebugRuntime:
    runtime = UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=ClaudeSessionStore(storage, "rt-1"),
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )
    return UiPathDebugRuntime(
        delegate=UiPathResumableRuntime(
            delegate=runtime,
            storage=storage,
            trigger_manager=UiPathResumeTriggerHandler(),
            runtime_id="rt-1",
        ),
        debug_bridge=bridge,
    )


async def _drain(runtime: UiPathDebugRuntime, input: dict[str, Any]) -> list[Any]:
    return [event async for event in runtime.stream(input, UiPathStreamOptions())]


async def test_a_human_in_the_loop_run_completes_through_the_debug_stack(
    interrupt_client, tmp_path, storage
):
    """The interactive loop `uipath debug` runs: suspend, answer, finish."""
    agent = ClaudeAgent(options=hitl_options())
    bridge = ScriptedBridge(resume_payload="ship it")
    events = await _drain(
        _debug_stack(agent, tmp_path, storage, bridge), {"input": "ask me"}
    )

    results = [e for e in events if isinstance(e, UiPathRuntimeResult)]
    assert [r.status for r in results] == [
        UiPathRuntimeStatus.SUSPENDED,
        UiPathRuntimeStatus.SUCCESSFUL,
    ]
    assert bridge.log == [
        "connect",
        "started",
        "suspended:API",
        "resumed",
        "completed:SUCCESSFUL",
    ]
    assert interrupt_client.tool_results == ["ship it"]


async def test_state_events_reach_the_debug_bridge(
    structured_agent, fake_client, tmp_path, storage
):
    bridge = ScriptedBridge()
    await _drain(
        _debug_stack(structured_agent, tmp_path, storage, bridge), {"topic": "otters"}
    )

    assert [e.node_name for e in bridge.state_events] == ["assistant", "tool_call"]


async def test_breakpoints_are_reported_as_unsupported_once(
    structured_agent, fake_client, tmp_path, session_store, caplog
):
    runtime = UiPathClaudeSDKRuntime(
        agent=structured_agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )
    with caplog.at_level(logging.WARNING, logger="uipath_claude_sdk.runtime.runtime"):
        async for _ in runtime.stream(
            {"topic": "otters"}, UiPathStreamOptions(breakpoints="*")
        ):
            pass

    warnings = [
        r for r in caplog.records if "Breakpoints are not supported" in r.getMessage()
    ]
    assert len(warnings) == 1


async def test_no_breakpoints_means_no_warning(
    structured_agent, fake_client, tmp_path, session_store, caplog
):
    runtime = UiPathClaudeSDKRuntime(
        agent=structured_agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )
    with caplog.at_level(logging.WARNING, logger="uipath_claude_sdk.runtime.runtime"):
        async for _ in runtime.stream(
            {"topic": "otters"}, UiPathStreamOptions(breakpoints=[])
        ):
            pass

    assert [
        r for r in caplog.records if "Breakpoints are not supported" in r.getMessage()
    ] == []


async def test_cli_stderr_is_logged(
    structured_agent, fake_client, tmp_path, session_store, caplog
):
    runtime = UiPathClaudeSDKRuntime(
        agent=structured_agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )
    async for _ in runtime.stream({"topic": "otters"}):
        pass

    stderr = fake_client.last_options.stderr
    assert stderr is not None
    with caplog.at_level(logging.WARNING):
        stderr("Error: something went wrong")
    assert "Error: something went wrong" in caplog.text


async def test_a_developer_stderr_callback_is_kept(
    structured_agent, fake_client, tmp_path, session_store
):
    seen: list[str] = []

    def developer_stderr(line: str) -> None:
        seen.append(line)

    structured_agent.options.stderr = developer_stderr

    runtime = UiPathClaudeSDKRuntime(
        agent=structured_agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )
    async for _ in runtime.stream({"topic": "otters"}):
        pass

    assert fake_client.last_options.stderr is developer_stderr


@pytest.fixture
def interrupt_client(monkeypatch: pytest.MonkeyPatch):
    from tests.test_suspend_resume import InterruptClient as Client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("uipath_claude_sdk.runtime.runtime.ClaudeSDKClient", Client)
    Client.last_options = None
    Client.last_query = None
    Client.tool_results = []
    Client.session_id = "session-1"
    return Client
