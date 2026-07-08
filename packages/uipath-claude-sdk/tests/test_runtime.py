"""Tests for the standard Claude SDK runtime."""

from __future__ import annotations

import pytest
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.events import UiPathRuntimeStateEvent

from tests.conftest import make_result_message
from uipath_claude_sdk.runtime.errors import UiPathClaudeSDKRuntimeError
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime


async def test_execute_structured(structured_agent, fake_client):
    runtime = UiPathClaudeSDKRuntime(agent=structured_agent, runtime_id="rt-1")
    result = await runtime.execute({"topic": "penguins"})

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert result.output == {"summary": "Hello"}
    assert fake_client.last_query == "Summarize penguins"
    assert fake_client.last_options is not None
    # Structured output is requested natively from the SDK.
    assert fake_client.last_options.output_format == {
        "type": "json_schema",
        "schema": structured_agent.output_schema.model_json_schema(),
    }
    # Isolated from user/project/local Claude config by default.
    assert fake_client.last_options.setting_sources == []


async def test_execute_plain_result(plain_agent, fake_client):
    fake_client.scripted_messages = [make_result_message(result="plain text")]
    runtime = UiPathClaudeSDKRuntime(agent=plain_agent, runtime_id="rt-1")
    result = await runtime.execute({"input": "hi"})

    assert result.output == {"result": "plain text"}
    assert fake_client.last_query == "hi"


async def test_stream_yields_state_events_then_result(structured_agent, fake_client):
    runtime = UiPathClaudeSDKRuntime(agent=structured_agent, runtime_id="rt-1")
    events = [event async for event in runtime.stream({"topic": "penguins"})]

    assert isinstance(events[-1], UiPathRuntimeResult)
    state_events = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
    node_names = [e.node_name for e in state_events]
    assert "assistant" in node_names
    assert "tool_call" in node_names


async def test_input_validation_error(structured_agent, fake_client):
    runtime = UiPathClaudeSDKRuntime(agent=structured_agent, runtime_id="rt-1")
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="input schema"):
        await runtime.execute({"wrong_field": 1})


async def test_error_result_raises(structured_agent, fake_client):
    fake_client.scripted_messages = [make_result_message(result="boom", is_error=True)]
    runtime = UiPathClaudeSDKRuntime(agent=structured_agent, runtime_id="rt-1")
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="boom"):
        await runtime.execute({"topic": "penguins"})


async def test_missing_credentials(plain_agent, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("UIPATH_URL", raising=False)
    monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)
    runtime = UiPathClaudeSDKRuntime(agent=plain_agent, runtime_id="rt-1")
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="ANTHROPIC_API_KEY"):
        await runtime.execute({"input": "hi"})


async def test_native_output_format_result(fake_client):
    from claude_agent_sdk import ClaudeAgentOptions

    from uipath_claude_sdk import ClaudeAgent

    agent = ClaudeAgent(
        options=ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            output_format={
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"summary": {}}},
            },
        )
    )
    fake_client.scripted_messages = [
        make_result_message(result="text", structured_output={"summary": "native"})
    ]
    runtime = UiPathClaudeSDKRuntime(agent=agent, runtime_id="rt-1")
    result = await runtime.execute({"input": "hi"})

    assert result.output == {"summary": "native"}
    # The user's own output_format is preserved, not overwritten.
    assert fake_client.last_options.output_format["schema"]["properties"] == {
        "summary": {}
    }
