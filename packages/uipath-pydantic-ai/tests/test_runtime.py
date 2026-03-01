"""Tests for PydanticAI runtime execution."""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

from uipath_pydantic_ai.runtime.errors import (
    UiPathPydanticAIErrorCode,
    UiPathPydanticAIRuntimeError,
)
from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime

# ============= HELPERS =============


def _uipath_message(text: str, role: str = "user") -> dict[str, Any]:
    """Build a UiPath conversation message dict."""
    return {
        "role": role,
        "contentParts": [{"data": {"inline": text}}],
    }


def _uipath_input(text: str) -> dict[str, Any]:
    """Build a UiPath input dict with a single user message."""
    return {"messages": [_uipath_message(text)]}


# ============= ERROR TESTS =============


def test_error_handling():
    """Test that error handling works correctly."""
    error = UiPathPydanticAIRuntimeError(
        code=UiPathPydanticAIErrorCode.AGENT_EXECUTION_ERROR,
        title="Test error",
        detail="This is a test error",
    )

    # Verify error can be created and contains the detail message
    assert isinstance(error, UiPathPydanticAIRuntimeError)
    assert "This is a test error" in str(error)

    # Verify error can be raised
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc_info:
        raise error

    assert "This is a test error" in str(exc_info.value)


def test_error_codes():
    """Test that all error codes are accessible."""
    assert (
        UiPathPydanticAIErrorCode.AGENT_EXECUTION_ERROR.value == "AGENT_EXECUTION_ERROR"
    )
    assert UiPathPydanticAIErrorCode.AGENT_TIMEOUT.value == "AGENT_TIMEOUT"
    assert (
        UiPathPydanticAIErrorCode.SERIALIZE_OUTPUT_ERROR.value
        == "SERIALIZE_OUTPUT_ERROR"
    )
    assert UiPathPydanticAIErrorCode.STREAM_ERROR.value == "STREAM_ERROR"
    assert UiPathPydanticAIErrorCode.CONFIG_MISSING.value == "CONFIG_MISSING"
    assert UiPathPydanticAIErrorCode.CONFIG_INVALID.value == "CONFIG_INVALID"
    assert UiPathPydanticAIErrorCode.AGENT_NOT_FOUND.value == "AGENT_NOT_FOUND"
    assert UiPathPydanticAIErrorCode.AGENT_TYPE_ERROR.value == "AGENT_TYPE_ERROR"
    assert UiPathPydanticAIErrorCode.AGENT_VALUE_ERROR.value == "AGENT_VALUE_ERROR"
    assert UiPathPydanticAIErrorCode.AGENT_LOAD_FAILURE.value == "AGENT_LOAD_FAILURE"
    assert UiPathPydanticAIErrorCode.AGENT_IMPORT_ERROR.value == "AGENT_IMPORT_ERROR"
    assert (
        UiPathPydanticAIErrorCode.SCHEMA_INFERENCE_ERROR.value
        == "SCHEMA_INFERENCE_ERROR"
    )


# ============= INPUT PREPARATION TESTS (UiPath format) =============


def test_runtime_input_uipath_message():
    """Test that runtime extracts text from UiPath conversation messages."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(_uipath_input("Hello"))
    assert prompt == "Hello"
    assert deps is None


def test_runtime_input_multiple_messages_takes_last_user():
    """Test that runtime picks the last user message from a conversation."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(
        {
            "messages": [
                _uipath_message("First question"),
                _uipath_message("I am the assistant", role="assistant"),
                _uipath_message("Second question"),
            ]
        }
    )
    assert prompt == "Second question"
    assert deps is None


def test_runtime_input_multipart_content():
    """Test that runtime concatenates multiple content parts."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(
        {
            "messages": [
                {
                    "role": "user",
                    "contentParts": [
                        {"data": {"inline": "Hello "}},
                        {"data": {"inline": "World"}},
                    ],
                }
            ]
        }
    )
    assert prompt == "Hello World"
    assert deps is None


def test_runtime_input_empty():
    """Test empty input returns empty prompt."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(None)
    assert prompt == ""
    assert deps is None

    prompt, deps = runtime._prepare_input({})
    assert prompt == ""
    assert deps is None


def test_runtime_input_empty_messages():
    """Test empty messages array returns empty prompt."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input({"messages": []})
    assert prompt == ""
    assert deps is None


def test_runtime_input_non_list_messages():
    """Test non-list messages returns empty prompt."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input({"messages": 123})
    assert prompt == ""
    assert deps is None


def test_runtime_input_falls_back_to_last_message():
    """Test that when no user message exists, falls back to last message."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(
        {"messages": [_uipath_message("Assistant reply", role="assistant")]}
    )
    assert prompt == "Assistant reply"
    assert deps is None


# ============= STRUCTURED INPUT TESTS =============


def test_runtime_structured_input_preparation():
    """Test that runtime correctly prepares structured deps input."""

    class MyInput(BaseModel):
        query: str
        max_results: int = 5

    agent = Agent("test", name="test_agent", deps_type=MyInput)
    runtime = UiPathPydanticAIRuntime(agent=agent)

    # Test structured input -> deps model
    prompt, deps = runtime._prepare_input({"query": "test", "max_results": 3})
    assert prompt == ""
    assert isinstance(deps, MyInput)
    assert deps.query == "test"
    assert deps.max_results == 3


def test_runtime_structured_input_with_messages():
    """Test that deps model with a 'messages' field uses it as prompt."""

    class ReviewInput(BaseModel):
        messages: str
        review_type: str

    agent = Agent("test", name="test_agent", deps_type=ReviewInput)
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(
        {"messages": "Review this code", "review_type": "security"}
    )
    assert prompt == "Review this code"
    assert isinstance(deps, ReviewInput)
    assert deps.review_type == "security"


# ============= GET_SCHEMA TESTS =============


@pytest.mark.asyncio
async def test_get_schema_conversational_agent():
    """Test that get_schema() returns UiPath conversation message format for a plain agent."""
    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    schema = await runtime.get_schema()

    # Input should use UiPath conversation messages
    assert "messages" in schema.input["properties"]
    messages_prop = schema.input["properties"]["messages"]
    assert messages_prop["type"] == "array"
    assert messages_prop["items"]["properties"]["role"]["type"] == "string"
    assert "contentParts" in messages_prop["items"]["properties"]

    # Output should also use UiPath conversation messages
    assert "messages" in schema.output["properties"]
    out_messages = schema.output["properties"]["messages"]
    assert out_messages["type"] == "array"


@pytest.mark.asyncio
async def test_get_schema_structured_output_agent():
    """Test that get_schema() returns structured output when output_type is set."""

    class MyOutput(BaseModel):
        answer: str
        confidence: float

    agent = Agent("test", name="test_agent", output_type=MyOutput)
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    schema = await runtime.get_schema()

    # Input should still be UiPath conversation messages
    assert "messages" in schema.input["properties"]

    # Output should be the structured model
    assert "answer" in schema.output["properties"]
    assert "confidence" in schema.output["properties"]


@pytest.mark.asyncio
async def test_get_schema_structured_input_agent():
    """Test that get_schema() returns structured input when deps_type is set."""

    class MyInput(BaseModel):
        query: str

    agent = Agent("test", name="test_agent", deps_type=MyInput)
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    schema = await runtime.get_schema()

    # Input should be the structured model
    assert "query" in schema.input["properties"]

    # Output should be UiPath conversation messages
    assert "messages" in schema.output["properties"]


# ============= E2E TESTS WITH MOCKED LLM =============


@pytest.mark.asyncio
async def test_e2e_execute_with_uipath_messages():
    """E2E test: execute a conversational agent with UiPath message input."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    result = await runtime.execute(input=_uipath_input("What is 2+2?"))

    assert result.status.value == "successful"
    assert isinstance(result.output, dict)
    assert "messages" in result.output
    messages = result.output["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert isinstance(messages[0]["contentParts"][0]["data"]["inline"], str)


@pytest.mark.asyncio
async def test_e2e_execute_with_multi_turn_messages():
    """E2E test: execute with multi-turn conversation, picks last user message."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    result = await runtime.execute(
        input={
            "messages": [
                _uipath_message("Hello"),
                _uipath_message("I'm an assistant", role="assistant"),
                _uipath_message("What is the weather?"),
            ]
        }
    )

    assert result.status.value == "successful"
    assert isinstance(result.output, dict)
    assert "messages" in result.output
    assert result.output["messages"][0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_e2e_execute_structured_output():
    """E2E test: execute agent with structured output_type."""
    from pydantic_ai.models.test import TestModel

    class CityInfo(BaseModel):
        name: str
        population: int

    agent = Agent(
        TestModel(custom_output_args={"name": "Paris", "population": 2161000}),
        name="city_agent",
        output_type=CityInfo,
    )
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    result = await runtime.execute(input=_uipath_input("Tell me about Paris"))

    assert result.status.value == "successful"
    assert isinstance(result.output, dict)
    assert result.output["name"] == "Paris"
    assert result.output["population"] == 2161000


@pytest.mark.asyncio
async def test_e2e_execute_with_tools():
    """E2E test: execute agent with tools using UiPath message input."""
    from pydantic_ai.models.test import TestModel

    def get_answer(ctx, question: str) -> str:
        """Answer a question.

        Args:
            ctx: The agent context.
            question: The question to answer.

        Returns:
            The answer.
        """
        return "42"

    agent = Agent(
        TestModel(),
        name="tool_agent",
        tools=[get_answer],
    )
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    result = await runtime.execute(input=_uipath_input("What is the meaning of life?"))

    assert result.status.value == "successful"
    assert isinstance(result.output, dict)
    assert "messages" in result.output
    assert result.output["messages"][0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_e2e_execute_structured_deps():
    """E2E test: execute agent with structured deps_type."""
    from pydantic_ai.models.test import TestModel

    class ResearchInput(BaseModel):
        topic: str
        max_sources: int = 3

    agent = Agent(
        TestModel(),
        name="research_agent",
        deps_type=ResearchInput,
    )
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    result = await runtime.execute(
        input={"topic": "quantum computing", "max_sources": 5}
    )

    assert result.status.value == "successful"
    assert isinstance(result.output, dict)
    assert "messages" in result.output
    assert result.output["messages"][0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_e2e_stream_with_uipath_messages():
    """E2E test: stream a conversational agent with UiPath message input."""
    from pydantic_ai.models.test import TestModel
    from uipath.runtime import UiPathRuntimeResult

    agent = Agent(TestModel(), name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    events = []
    async for event in runtime.stream(input=_uipath_input("Tell me a joke")):
        events.append(event)

    # Last event should be the result
    assert len(events) > 0
    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status.value == "successful"


@pytest.mark.asyncio
async def test_e2e_stream_events_no_duplicates():
    """E2E test: verify stream events have no duplicates and proper STARTED/COMPLETED pairing."""
    from pydantic_ai.models.test import TestModel
    from uipath.runtime.events import (
        UiPathRuntimeStateEvent,
        UiPathRuntimeStatePhase,
    )

    agent = Agent(TestModel(), name="my_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    state_events: list[tuple[str, UiPathRuntimeStatePhase]] = []
    async for event in runtime.stream(input=_uipath_input("Hello")):
        if isinstance(event, UiPathRuntimeStateEvent) and event.node_name:
            state_events.append((event.node_name, event.phase))

    # Should have exactly one STARTED/COMPLETED pair for the agent node
    agent_events = [e for e in state_events if e[0] == "my_agent"]
    assert len(agent_events) >= 2  # At least one STARTED + COMPLETED

    # Every STARTED must be followed by a COMPLETED for the same node
    open_nodes: dict[str, int] = {}
    for node_name, phase in state_events:
        if phase == UiPathRuntimeStatePhase.STARTED:
            open_nodes[node_name] = open_nodes.get(node_name, 0) + 1
        elif phase == UiPathRuntimeStatePhase.COMPLETED:
            assert open_nodes.get(node_name, 0) > 0, (
                f"COMPLETED without STARTED for {node_name}"
            )
            open_nodes[node_name] -= 1

    # All nodes should be closed (no dangling STARTED without COMPLETED)
    for node_name, count in open_nodes.items():
        assert count == 0, f"Dangling STARTED for {node_name}"

    # No consecutive duplicate events
    for i in range(1, len(state_events)):
        assert state_events[i] != state_events[i - 1], (
            f"Consecutive duplicate event: {state_events[i]}"
        )


@pytest.mark.asyncio
async def test_e2e_stream_with_tools_events():
    """E2E test: verify tools node events when agent has tools."""
    from pydantic_ai.models.test import TestModel
    from uipath.runtime.events import (
        UiPathRuntimeStateEvent,
        UiPathRuntimeStatePhase,
    )

    def my_tool(ctx, query: str) -> str:
        """Search for something.

        Args:
            ctx: The agent context.
            query: The search query.

        Returns:
            Search results.
        """
        return f"Result for {query}"

    agent = Agent(TestModel(), name="tool_agent", tools=[my_tool])
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    state_events = []
    async for event in runtime.stream(input=_uipath_input("Search for cats")):
        if isinstance(event, UiPathRuntimeStateEvent):
            state_events.append((event.node_name, event.phase))

    # Should have agent events
    agent_events = [(n, p) for n, p in state_events if n == "tool_agent"]
    assert len(agent_events) >= 2

    # Should have tools events (TestModel calls tools by default)
    tools_events = [(n, p) for n, p in state_events if n == "tool_agent_tools"]
    assert len(tools_events) >= 2  # At least one STARTED + COMPLETED pair

    # Tools events should be properly paired
    for i in range(0, len(tools_events), 2):
        assert tools_events[i][1] == UiPathRuntimeStatePhase.STARTED
        assert tools_events[i + 1][1] == UiPathRuntimeStatePhase.COMPLETED

    # Agent events should also be properly paired
    for i in range(0, len(agent_events), 2):
        assert agent_events[i][1] == UiPathRuntimeStatePhase.STARTED
        assert agent_events[i + 1][1] == UiPathRuntimeStatePhase.COMPLETED


@pytest.mark.asyncio
async def test_e2e_stream_event_payloads():
    """E2E test: verify state events carry meaningful payloads."""
    from pydantic_ai.models.test import TestModel
    from uipath.runtime.events import (
        UiPathRuntimeStateEvent,
        UiPathRuntimeStatePhase,
    )

    def my_tool(ctx, query: str) -> str:
        """Search for something.

        Args:
            ctx: The agent context.
            query: The search query.

        Returns:
            Search results.
        """
        return f"Result for {query}"

    agent = Agent(TestModel(), name="payload_agent", tools=[my_tool])
    runtime = UiPathPydanticAIRuntime(agent=agent, runtime_id="test", entrypoint="test")

    state_events: list[UiPathRuntimeStateEvent] = []
    async for event in runtime.stream(input=_uipath_input("Search for cats")):
        if isinstance(event, UiPathRuntimeStateEvent):
            state_events.append(event)

    # Agent COMPLETED events should have model_name and usage
    agent_completed = [
        e
        for e in state_events
        if e.node_name == "payload_agent"
        and e.phase == UiPathRuntimeStatePhase.COMPLETED
    ]
    assert len(agent_completed) >= 1
    for event in agent_completed:
        assert "model_name" in event.payload or "usage" in event.payload

    # Tools STARTED events should have tool_calls with tool_name
    tools_started = [
        e
        for e in state_events
        if e.node_name == "payload_agent_tools"
        and e.phase == UiPathRuntimeStatePhase.STARTED
    ]
    assert len(tools_started) >= 1
    for event in tools_started:
        assert "tool_calls" in event.payload
        assert len(event.payload["tool_calls"]) > 0
        assert "tool_name" in event.payload["tool_calls"][0]

    # Tools COMPLETED events should have tool_results with content
    tools_completed = [
        e
        for e in state_events
        if e.node_name == "payload_agent_tools"
        and e.phase == UiPathRuntimeStatePhase.COMPLETED
    ]
    assert len(tools_completed) >= 1
    for event in tools_completed:
        assert "tool_results" in event.payload
        assert len(event.payload["tool_results"]) > 0
        result = event.payload["tool_results"][0]
        assert "tool_name" in result
        assert "content" in result
