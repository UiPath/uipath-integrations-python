"""Tests for schema and graph extraction."""

from __future__ import annotations

from uipath_claude_sdk.runtime.schema import (
    get_agent_graph,
    get_entrypoints_schema,
)


def test_structured_schema(structured_agent):
    schema = get_entrypoints_schema(structured_agent)
    assert schema["input"]["properties"]["topic"]["type"] == "string"
    assert schema["input"]["required"] == ["topic"]
    assert schema["output"]["properties"]["summary"]["type"] == "string"


def test_default_schema(plain_agent):
    schema = get_entrypoints_schema(plain_agent)
    assert "input" in schema["input"]["properties"]
    assert "result" in schema["output"]["properties"]


def test_conversational_schema(structured_agent):
    schema = get_entrypoints_schema(structured_agent, conversational=True)
    assert "messages" in schema["input"]["properties"]
    assert "messages" in schema["output"]["properties"]


def test_graph_nodes_unique_and_connected(structured_agent):
    graph = get_agent_graph(structured_agent)
    ids = [node.id for node in graph.nodes]
    assert len(ids) == len(set(ids))
    assert "__start__" in ids and "__end__" in ids and "research" in ids
    for edge in graph.edges:
        assert edge.source in ids
        assert edge.target in ids
    model_node = next(n for n in graph.nodes if n.id == "research")
    assert model_node.type == "model"
    assert model_node.metadata == {
        "model_name": "anthropic.claude-sonnet-4-5-20250929-v1:0"
    }


def test_schema_from_native_output_format():
    from claude_agent_sdk import ClaudeAgentOptions

    from uipath_claude_sdk import ClaudeAgent

    native_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    agent = ClaudeAgent(
        options=ClaudeAgentOptions(
            model="claude-sonnet-4-5",
            output_format={"type": "json_schema", "schema": native_schema},
        )
    )
    schema = get_entrypoints_schema(agent)
    assert schema["output"] == native_schema
