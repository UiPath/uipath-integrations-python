"""Tests for schema extraction utilities."""

from unittest.mock import MagicMock

from agent_framework import BaseAgent

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
