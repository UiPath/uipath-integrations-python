"""Tests for OpenAI session integration in runtime."""

from typing import Any

import pytest
from agents import Agent
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus

from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime


class DummySession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_execute_forwards_session_to_runner(monkeypatch):
    captured: dict[str, Any] = {}
    session = DummySession()

    class FakeRunResult:
        final_output = {"result": "ok"}

    async def fake_run(*args, **kwargs):
        captured["session"] = kwargs.get("session")
        return FakeRunResult()

    monkeypatch.setattr("uipath_openai_agents.runtime.runtime.Runner.run", fake_run)

    runtime = UiPathOpenAIAgentRuntime(
        agent=Agent(name="test-agent", instructions="test"),
        runtime_id="runtime-1",
        entrypoint="agent",
        session=session,  # type: ignore[arg-type]
    )

    result = await runtime.execute(input={"messages": "hello"})

    assert captured["session"] is session
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert result.output == {"result": "ok"}

    await runtime.dispose()
    assert session.closed is True


@pytest.mark.asyncio
async def test_stream_forwards_session_to_runner(monkeypatch):
    captured: dict[str, Any] = {}
    session = DummySession()

    class FakeStreamingResult:
        final_output = {"answer": "done"}

        async def stream_events(self):
            if False:
                yield None

    def fake_run_streamed(*args, **kwargs):
        captured["session"] = kwargs.get("session")
        return FakeStreamingResult()

    monkeypatch.setattr(
        "uipath_openai_agents.runtime.runtime.Runner.run_streamed",
        fake_run_streamed,
    )

    runtime = UiPathOpenAIAgentRuntime(
        agent=Agent(name="test-agent", instructions="test"),
        runtime_id="runtime-2",
        entrypoint="agent",
        session=session,  # type: ignore[arg-type]
    )

    events = [event async for event in runtime.stream(input={"messages": "hello"})]

    assert captured["session"] is session
    assert len(events) == 1
    assert isinstance(events[0], UiPathRuntimeResult)
    assert events[0].status == UiPathRuntimeStatus.SUCCESSFUL
    assert events[0].output == {"answer": "done"}
