"""Tests for UiPathOpenAIAgentRuntime execution and event conversion."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from agents import Agent
from pydantic import BaseModel
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.errors import UiPathErrorCode
from uipath.runtime.events import (
    UiPathRuntimeMessageEvent,
    UiPathRuntimeStateEvent,
)

from uipath_openai_agents.runtime.errors import (
    UiPathOpenAIAgentsErrorCode,
    UiPathOpenAIAgentsRuntimeError,
)
from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime


class _Item(BaseModel):
    text: str


@pytest.fixture
def runtime() -> UiPathOpenAIAgentRuntime:
    agent = Agent(name="echo", instructions="echo")
    return UiPathOpenAIAgentRuntime(agent=agent, entrypoint="echo")


@pytest.mark.asyncio
async def test_execute_wraps_scalar_output_in_result_dict(
    runtime: UiPathOpenAIAgentRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """execute() serializes a non-dict final_output under a 'result' key."""

    async def fake_run(**kwargs: Any) -> Any:
        return SimpleNamespace(final_output="hello there")

    monkeypatch.setattr("uipath_openai_agents.runtime.runtime.Runner.run", fake_run)

    result = await runtime.execute({"messages": "hi"})

    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert result.output == {"result": "hello there"}


@pytest.mark.asyncio
async def test_stream_emits_message_event_then_final_result(
    runtime: UiPathOpenAIAgentRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stream() yields converted events and a terminal success result."""
    stream_event = SimpleNamespace(
        type="run_item_stream_event",
        name="message_output_created",
        item=_Item(text="hi"),
    )

    class FakeStreaming:
        final_output = {"answer": 42}

        async def stream_events(self):
            yield stream_event

    monkeypatch.setattr(
        "uipath_openai_agents.runtime.runtime.Runner.run_streamed",
        lambda **kwargs: FakeStreaming(),
    )

    events = [e async for e in runtime.stream({"messages": "hi"})]

    assert isinstance(events[0], UiPathRuntimeMessageEvent)
    assert events[0].metadata == {"event_name": "message_output_created"}
    assert isinstance(events[-1], UiPathRuntimeResult)
    assert events[-1].output == {"answer": 42}


def test_convert_agent_updated_event_to_state_event(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """agent_updated_stream_event maps to a state event carrying the new agent name."""
    event = SimpleNamespace(
        type="agent_updated_stream_event",
        name=None,
        new_agent=SimpleNamespace(name="specialist"),
    )

    converted = runtime._convert_stream_event_to_runtime_event(event)

    assert isinstance(converted, UiPathRuntimeStateEvent)
    assert converted.payload == {"agent_name": "specialist"}


def test_convert_raw_event_is_filtered_out(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """Unrecognized (raw) events are filtered and produce no runtime event."""
    event = SimpleNamespace(type="raw_response_event", name=None)

    assert runtime._convert_stream_event_to_runtime_event(event) is None


def test_prepare_input_defaults_when_no_input(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """Missing input yields empty message string and no context."""
    assert runtime._prepare_agent_input_and_context(None) == ("", None)


def test_prepare_input_coerces_non_string_messages_to_empty(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """A non-str/non-list 'messages' value is coerced to an empty string."""
    messages, context = runtime._prepare_agent_input_and_context({"messages": 123})

    assert messages == ""
    assert context is None


def test_runtime_error_maps_json_decode_error(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """A JSONDecodeError is classified as INPUT_INVALID_JSON."""
    err = json.JSONDecodeError("bad", "doc", 0)

    mapped = runtime._create_runtime_error(err)

    assert mapped.error_info.code.endswith(UiPathErrorCode.INPUT_INVALID_JSON.value)


def test_runtime_error_maps_timeout(runtime: UiPathOpenAIAgentRuntime) -> None:
    """A TimeoutError is classified as TIMEOUT_ERROR."""
    mapped = runtime._create_runtime_error(TimeoutError("slow"))

    assert mapped.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.TIMEOUT_ERROR.value
    )


def test_runtime_error_passes_through_own_error(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """An existing runtime error is returned unchanged, not re-wrapped."""
    original = UiPathOpenAIAgentsRuntimeError(
        UiPathOpenAIAgentsErrorCode.AGENT_EXECUTION_FAILURE, "t", "d"
    )

    assert runtime._create_runtime_error(original) is original


@pytest.mark.asyncio
async def test_get_schema_reports_agent_entrypoint_and_messages_input(
    runtime: UiPathOpenAIAgentRuntime,
) -> None:
    """get_schema exposes the entrypoint path, agent type and messages input."""
    schema = await runtime.get_schema()

    assert schema.file_path == "echo"
    assert schema.type == "agent"
    assert "messages" in schema.input["properties"]
