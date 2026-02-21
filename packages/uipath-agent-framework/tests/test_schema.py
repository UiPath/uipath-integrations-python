"""Tests for schema extraction utilities."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

from agent_framework import BaseAgent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from conftest import make_chunked_streaming_response, make_mock_response
from pydantic import BaseModel
from uipath.runtime import UiPathRuntimeResult
from uipath.runtime.result import UiPathRuntimeStatus

from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime
from uipath_agent_framework.runtime.schema import (
    get_entrypoints_schema,
)


def _make_agent(name="test_agent", tools=None) -> BaseAgent:
    """Create a mock BaseAgent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = name
    agent.default_options = {"tools": tools or []}
    return agent


class TestGetEntrypointsSchema:
    """Tests for get_entrypoints_schema function."""

    def test_agent_has_messages_input(self):
        """Agents get messages-based input."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        assert "input" in schema
        assert "output" in schema
        assert "messages" in schema["input"]["properties"]
        assert "messages" in schema["input"]["required"]

    def test_agent_has_messages_output(self):
        """Agents get messages-based output (UiPath conversation format)."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        assert "messages" in schema["output"]["properties"]
        assert "messages" in schema["output"]["required"]

    def test_input_messages_is_array(self):
        """Messages input is an array of conversation message objects."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        messages_schema = schema["input"]["properties"]["messages"]
        assert messages_schema["type"] == "array"
        assert "items" in messages_schema
        assert messages_schema["items"]["type"] == "object"

    def test_output_messages_is_array(self):
        """Messages output is an array of conversation message objects."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        messages_schema = schema["output"]["properties"]["messages"]
        assert messages_schema["type"] == "array"
        assert "items" in messages_schema
        assert messages_schema["items"]["type"] == "object"

    def test_message_item_has_role_and_content_parts(self):
        """Message items require role and contentParts."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        item_schema = schema["input"]["properties"]["messages"]["items"]
        assert "role" in item_schema["properties"]
        assert "contentParts" in item_schema["properties"]
        assert "role" in item_schema["required"]
        assert "contentParts" in item_schema["required"]

    def test_input_output_schema_match(self):
        """Input and output use the same conversation message schema."""
        agent = _make_agent()
        schema = get_entrypoints_schema(agent)

        assert schema["input"] == schema["output"]


class CityInfo(BaseModel):
    """Structured output for city information."""

    city: str
    country: str
    description: str
    population_estimate: str
    famous_for: list[str]


class TestStructuredOutputSchema:
    """E2E tests for structured output schema inference via the runtime."""

    def _make_structured_output_agent(self):
        """Create the structured-output sample workflow agent."""
        os.environ.setdefault("UIPATH_URL", "https://fake.uipath.com")
        os.environ.setdefault("UIPATH_ACCESS_TOKEN", "fake")
        from uipath_agent_framework.chat import UiPathOpenAIChatClient

        client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")
        city_agent = client.as_agent(
            name="city_agent",
            instructions="You are a helpful agent that describes cities.",
            default_options={"response_format": CityInfo},
        )
        workflow = WorkflowBuilder(start_executor=city_agent).build()
        return workflow.as_agent(name="structured_output_workflow")

    def test_runtime_get_schema_has_structured_output(self):
        """Runtime.get_schema() returns the CityInfo schema as output."""
        agent = self._make_structured_output_agent()
        runtime = UiPathAgentFrameworkRuntime(agent=agent, entrypoint="agent")

        schema = asyncio.run(runtime.get_schema())

        # Output should be CityInfo, not default messages
        assert schema.output["type"] == "object"
        assert "city" in schema.output["properties"]
        assert "country" in schema.output["properties"]
        assert "description" in schema.output["properties"]
        assert "population_estimate" in schema.output["properties"]
        assert "famous_for" in schema.output["properties"]
        assert schema.output["properties"]["famous_for"]["type"] == "array"
        assert "messages" not in schema.output.get("properties", {})

    def test_runtime_get_schema_input_still_messages(self):
        """Runtime.get_schema() input remains the default messages schema."""
        agent = self._make_structured_output_agent()
        runtime = UiPathAgentFrameworkRuntime(agent=agent, entrypoint="agent")

        schema = asyncio.run(runtime.get_schema())

        assert "messages" in schema.input["properties"]
        assert schema.input["properties"]["messages"]["type"] == "array"

    def test_runtime_get_schema_without_response_format(self):
        """Runtime.get_schema() falls back to messages output without response_format."""
        os.environ.setdefault("UIPATH_URL", "https://fake.uipath.com")
        os.environ.setdefault("UIPATH_ACCESS_TOKEN", "fake")
        from uipath_agent_framework.chat import UiPathOpenAIChatClient

        client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")
        agent = client.as_agent(name="basic_agent", instructions="test")
        workflow = WorkflowBuilder(start_executor=agent).build()
        wa = workflow.as_agent(name="basic_workflow")
        runtime = UiPathAgentFrameworkRuntime(agent=wa, entrypoint="agent")

        schema = asyncio.run(runtime.get_schema())

        assert "messages" in schema.output["properties"]
        assert schema.input == schema.output


CITY_INFO_JSON = json.dumps(
    {
        "city": "Tokyo",
        "country": "Japan",
        "description": "Capital of Japan",
        "population_estimate": "14 million",
        "famous_for": ["sushi", "technology", "cherry blossoms"],
    }
)


class TestStructuredOutputResult:
    """E2E tests for structured output runtime result."""

    def test_execute_returns_structured_output_dict(self):
        """Runtime.execute() returns structured dict, not messages wrapper."""
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = make_mock_response(
            CITY_INFO_JSON
        )

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)
        city_agent = client.as_agent(
            name="city_agent",
            instructions="Describe cities in structured format.",
            default_options={"response_format": CityInfo},
        )
        workflow = WorkflowBuilder(start_executor=city_agent).build()
        agent = workflow.as_agent(name="structured_output_workflow")

        runtime = UiPathAgentFrameworkRuntime(agent=agent, entrypoint="agent")
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "Tell me about Tokyo"

        result = asyncio.run(
            runtime.execute(
                input={"messages": [{"role": "user", "contentParts": [{"data": {"inline": "Tell me about Tokyo"}}]}]}
            )
        )

        assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        assert isinstance(result.output, dict)
        assert result.output["city"] == "Tokyo"
        assert result.output["country"] == "Japan"
        assert result.output["famous_for"] == ["sushi", "technology", "cherry blossoms"]
        assert "messages" not in result.output

    def test_execute_without_response_format_returns_messages(self):
        """Runtime.execute() wraps plain text output in messages."""
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = make_mock_response(
            "Tokyo is the capital of Japan."
        )

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)
        agent = client.as_agent(
            name="basic_agent",
            instructions="Describe cities.",
        )
        workflow = WorkflowBuilder(start_executor=agent).build()
        wa = workflow.as_agent(name="basic_workflow")

        runtime = UiPathAgentFrameworkRuntime(agent=wa, entrypoint="agent")
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "Tell me about Tokyo"

        result = asyncio.run(
            runtime.execute(
                input={"messages": [{"role": "user", "contentParts": [{"data": {"inline": "Tell me about Tokyo"}}]}]}
            )
        )

        assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        assert "messages" in result.output

    def test_streaming_returns_structured_output_dict(self):
        """Runtime.stream() returns structured dict from streaming tokens.

        Matches production behavior where get_outputs() returns many
        AgentResponseUpdate tokens instead of a single AgentResponse.
        """
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.side_effect = (
            lambda *args, **kwargs: make_chunked_streaming_response(CITY_INFO_JSON)
        )

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)
        city_agent = client.as_agent(
            name="city_agent",
            instructions="Describe cities in structured format.",
            default_options={"response_format": CityInfo},
        )
        workflow = WorkflowBuilder(start_executor=city_agent).build()
        agent = workflow.as_agent(name="structured_output_workflow")

        runtime = UiPathAgentFrameworkRuntime(agent=agent, entrypoint="agent")
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "Tell me about Tokyo"
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        async def run_stream():
            result = None
            async for event in runtime.stream(
                input={"messages": [{"role": "user", "contentParts": [{"data": {"inline": "Tell me about Tokyo"}}]}]}
            ):
                if isinstance(event, UiPathRuntimeResult):
                    result = event
            return result

        result = asyncio.run(run_stream())

        assert result is not None
        assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        assert isinstance(result.output, dict)
        assert result.output["city"] == "Tokyo"
        assert result.output["country"] == "Japan"
        assert result.output["famous_for"] == ["sushi", "technology", "cherry blossoms"]
        assert "messages" not in result.output
