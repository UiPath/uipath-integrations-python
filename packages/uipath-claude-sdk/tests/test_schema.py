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


def test_a_conversation_requires_nothing_of_its_caller(structured_agent):
    """An empty payload has to validate.

    The host supplies the conversation, so the run form starts out empty, and
    an exchange answers over the chat bridge rather than in its output. Marking
    either side required makes the platform reject a payload the runtime is
    happy with.
    """
    schema = get_entrypoints_schema(structured_agent, conversational=True)
    assert schema["input"]["required"] == []
    assert schema["output"]["required"] == []


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


def test_graph_model_name_prefers_uipath_llm(structured_agent):
    """The graph label comes from uipath_llm, not options.model.

    get_agent_graph runs at init time, before any run has resolved a model into
    options.model, so a gateway-routed agent would otherwise render with no
    model on its node in Studio Web.
    """
    from dataclasses import replace

    from claude_agent_sdk import ClaudeAgentOptions

    from uipath_claude_sdk import UiPathModel

    agent = replace(
        structured_agent,
        options=ClaudeAgentOptions(),
        uipath_llm=UiPathModel("claude-sonnet-4-5"),
    )
    node = next(n for n in get_agent_graph(agent).nodes if n.id == "research")
    assert node.type == "model"
    assert node.metadata == {"model_name": "claude-sonnet-4-5"}


def test_graph_model_name_falls_back_to_options_model(structured_agent):
    from dataclasses import replace

    agent = replace(structured_agent, uipath_llm=None)
    node = next(n for n in get_agent_graph(agent).nodes if n.id == "research")
    assert node.metadata == {"model_name": "anthropic.claude-sonnet-4-5-20250929-v1:0"}
