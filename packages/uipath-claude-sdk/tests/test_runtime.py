"""Tests for the standard Claude SDK runtime."""

from __future__ import annotations

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
)
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.events import UiPathRuntimeStateEvent

from tests.conftest import make_result_message, make_session_paths
from uipath_claude_sdk import UiPathModel
from uipath_claude_sdk.runtime.errors import UiPathClaudeSDKRuntimeError
from uipath_claude_sdk.runtime.gateway import ModelNotInCatalogError
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime


class FakeShim:
    """Stands in for GatewayShim, recording lifecycle calls."""

    def __init__(self, factory: "FakeShimFactory") -> None:
        self._factory = factory
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        if self._factory.start_error is not None:
            raise self._factory.start_error
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:1234"

    @property
    def api_key(self) -> str:
        return "run-secret"

    @property
    def resolved_model(self) -> str:
        return "resolved-sonnet"

    def build_env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_API_KEY": self.api_key,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.resolved_model,
        }


class FakeShimFactory:
    def __init__(self) -> None:
        self.instances: list[FakeShim] = []
        self.start_error: Exception | None = None

    def __call__(self, llm: object, **kwargs: object) -> FakeShim:
        shim = FakeShim(self)
        self.instances.append(shim)
        return shim


@pytest.fixture
def fake_shim(monkeypatch: pytest.MonkeyPatch) -> FakeShimFactory:
    factory = FakeShimFactory()
    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.runtime.GatewayShim", factory, raising=True
    )
    return factory


def _make_runtime(agent, tmp_path, session_store, runtime_id: str = "rt-1"):
    return UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path, runtime_id),
        runtime_id=runtime_id,
    )


async def test_execute_structured(
    structured_agent, fake_client, tmp_path, session_store
):
    runtime = _make_runtime(structured_agent, tmp_path, session_store)
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


async def test_execute_plain_result(plain_agent, fake_client, tmp_path, session_store):
    fake_client.scripted_messages = [make_result_message(result="plain text")]
    runtime = _make_runtime(plain_agent, tmp_path, session_store)
    result = await runtime.execute({"input": "hi"})

    assert result.output == {"result": "plain text"}
    assert fake_client.last_query == "hi"


async def test_stream_yields_state_events_then_result(
    structured_agent, fake_client, tmp_path, session_store
):
    runtime = _make_runtime(structured_agent, tmp_path, session_store)
    events = [event async for event in runtime.stream({"topic": "penguins"})]

    assert isinstance(events[-1], UiPathRuntimeResult)
    state_events = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
    node_names = [e.node_name for e in state_events]
    assert "assistant" in node_names
    assert "tool_call" in node_names


async def test_waits_for_parallel_background_agent_and_workflow_tasks(
    plain_agent, fake_client, tmp_path, session_store
):
    fake_client.scripted_messages = [
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="agent-1",
            description="research",
            uuid="task-start-1",
            session_id="session-1",
            task_type="local_agent",
        ),
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="workflow-1",
            description="verification",
            uuid="task-start-2",
            session_id="session-1",
            task_type="local_workflow",
        ),
        make_result_message(result="Both tasks were launched."),
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="agent-1",
            status="completed",
            output_file="agent-1.output",
            summary="Research complete",
            uuid="task-end-1",
            session_id="session-1",
        ),
        TaskUpdatedMessage(
            subtype="task_updated",
            data={},
            task_id="workflow-1",
            patch={"status": "completed"},
            status="completed",
            session_id="session-1",
            uuid="task-end-2",
        ),
        AssistantMessage(
            content=[TextBlock(text="Final answer after both tasks.")],
            model="claude-sonnet-4-5",
        ),
        make_result_message(result="Final answer after both tasks."),
    ]
    runtime = _make_runtime(plain_agent, tmp_path, session_store)

    events = [event async for event in runtime.stream({"input": "delegate"})]

    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.output == {"result": "Final answer after both tasks."}
    task_events = [
        event
        for event in events
        if isinstance(event, UiPathRuntimeStateEvent) and event.node_name == "task"
    ]
    assert [event.payload.get("status") for event in task_events] == [
        None,
        None,
        "completed",
        "completed",
    ]


async def test_does_not_wait_for_unbounded_background_shell(
    plain_agent, fake_client, tmp_path, session_store
):
    fake_client.scripted_messages = [
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="shell-1",
            description="development server",
            uuid="task-start-1",
            session_id="session-1",
            task_type="local_bash",
        ),
        make_result_message(result="Server is running."),
        make_result_message(result="This result must not be consumed."),
    ]
    runtime = _make_runtime(plain_agent, tmp_path, session_store)

    result = await runtime.execute({"input": "start server"})

    assert result.output == {"result": "Server is running."}


async def test_input_validation_error(
    structured_agent, fake_client, tmp_path, session_store
):
    runtime = _make_runtime(structured_agent, tmp_path, session_store)
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="input schema"):
        await runtime.execute({"wrong_field": 1})


async def test_error_result_raises(
    structured_agent, fake_client, tmp_path, session_store
):
    fake_client.scripted_messages = [make_result_message(result="boom", is_error=True)]
    runtime = _make_runtime(structured_agent, tmp_path, session_store)
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="boom"):
        await runtime.execute({"topic": "penguins"})


async def test_no_uipath_llm_injects_no_env(
    plain_agent, fake_client, tmp_path, session_store
):
    runtime = _make_runtime(plain_agent, tmp_path, session_store)
    await runtime.execute({"input": "hi"})

    injected = fake_client.last_options.env
    assert "ANTHROPIC_BASE_URL" not in injected
    assert "ANTHROPIC_API_KEY" not in injected
    assert fake_client.last_options.model == plain_agent.options.model


async def test_uipath_llm_starts_gateway_and_overrides_model(
    plain_agent, fake_client, fake_shim, tmp_path, session_store
):
    plain_agent.uipath_llm = UiPathModel("claude-sonnet-4-5")
    runtime = _make_runtime(plain_agent, tmp_path, session_store)
    await runtime.execute({"input": "hi"})

    assert fake_client.last_options.model == "resolved-sonnet"
    assert fake_client.last_options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
    assert fake_client.last_options.env["ANTHROPIC_API_KEY"] == "run-secret"
    # The shim is stopped by the runtime's own finally, not only by dispose().
    assert fake_shim.instances[0].stopped == 1
    assert runtime._shim is None


async def test_gateway_start_failure_is_mapped(
    plain_agent, fake_shim, tmp_path, session_store
):
    fake_shim.start_error = ModelNotInCatalogError("nope", ["a", "b"])
    plain_agent.uipath_llm = UiPathModel("nope")
    runtime = _make_runtime(plain_agent, tmp_path, session_store)

    with pytest.raises(UiPathClaudeSDKRuntimeError, match="not available"):
        await runtime.execute({"input": "hi"})


async def test_native_output_format_result(fake_client, tmp_path, session_store):
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
    runtime = _make_runtime(agent, tmp_path, session_store)
    result = await runtime.execute({"input": "hi"})

    assert result.output == {"summary": "native"}
    # The user's own output_format is preserved, not overwritten.
    assert fake_client.last_options.output_format["schema"]["properties"] == {
        "summary": {}
    }
