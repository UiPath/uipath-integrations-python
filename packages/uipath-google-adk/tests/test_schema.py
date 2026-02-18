"""Tests for schema extraction utilities."""

from unittest.mock import MagicMock

import pytest
from google.adk.agents import BaseAgent, LlmAgent
from pydantic import BaseModel

from uipath_google_adk.runtime.schema import (
    get_agent_graph,
    get_entrypoints_schema,
    resolve_output_schema,
)


class InputModel(BaseModel):
    """Test input model."""

    query: str
    limit: int = 10


class OutputModel(BaseModel):
    """Test output model."""

    answer: str
    confidence: float


def _make_llm_agent(
    name="test_agent",
    tools=None,
    sub_agents=None,
    input_schema=None,
    output_schema=None,
    output_key=None,
) -> LlmAgent:
    """Create a real LlmAgent for testing."""
    agent = LlmAgent(
        name=name,
        model="gemini-2.0-flash",
        tools=tools or [],
        sub_agents=sub_agents or [],
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
    )
    return agent


def _make_base_agent(
    name="test_agent",
    sub_agents=None,
) -> BaseAgent:
    """Create a mock BaseAgent (non-LlmAgent) for testing composite agents."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = name
    agent.sub_agents = sub_agents or []
    # Ensure isinstance check fails for LlmAgent
    agent.__class__ = BaseAgent  # type: ignore[assignment]
    return agent


class TestGetEntrypointsSchema:
    """Tests for get_entrypoints_schema function."""

    def test_composite_agent_has_messages_input(self):
        """Composite agents (non-LlmAgent) get messages-based input."""
        agent = _make_base_agent()
        schema = get_entrypoints_schema(agent)

        assert "input" in schema
        assert "output" in schema
        assert "messages" in schema["input"]["properties"]
        assert "messages" in schema["input"]["required"]

    def test_composite_agent_has_default_output(self):
        """Composite agents get default result output."""
        agent = _make_base_agent()
        schema = get_entrypoints_schema(agent)

        assert "result" in schema["output"]["properties"]
        assert "result" in schema["output"]["required"]

    def test_llm_agent_without_schemas_has_messages(self):
        """LlmAgent without input_schema falls back to messages."""
        agent = _make_llm_agent()
        schema = get_entrypoints_schema(agent)

        assert "messages" in schema["input"]["properties"]

    def test_llm_agent_without_schemas_has_default_output(self):
        """LlmAgent without output_schema falls back to default output."""
        agent = _make_llm_agent()
        schema = get_entrypoints_schema(agent)

        assert "result" in schema["output"]["properties"]

    def test_input_schema_replaces_messages(self):
        """LlmAgent with input_schema uses it as the full input schema."""
        agent = _make_llm_agent(input_schema=InputModel)
        schema = get_entrypoints_schema(agent)

        # input_schema replaces messages
        assert "query" in schema["input"]["properties"]
        assert "limit" in schema["input"]["properties"]
        assert "messages" not in schema["input"]["properties"]
        assert "query" in schema["input"]["required"]

    def test_output_schema_from_pydantic(self):
        """LlmAgent with output_schema uses it as the output."""
        agent = _make_llm_agent(output_schema=OutputModel)
        schema = get_entrypoints_schema(agent)

        assert "answer" in schema["output"]["properties"]
        assert "confidence" in schema["output"]["properties"]
        # Default 'result' should not be present
        assert "result" not in schema["output"]["properties"]

    def test_output_key_creates_simple_schema(self):
        """LlmAgent with output_key creates a simple string output schema."""
        agent = _make_llm_agent(output_key="response")
        schema = get_entrypoints_schema(agent)

        assert "response" in schema["output"]["properties"]
        assert schema["output"]["properties"]["response"]["type"] == "string"
        assert "response" in schema["output"]["required"]

    def test_output_schema_takes_precedence_over_output_key(self):
        """output_schema takes precedence over output_key."""
        agent = _make_llm_agent(
            output_schema=OutputModel, output_key="response"
        )
        schema = get_entrypoints_schema(agent)

        assert "answer" in schema["output"]["properties"]
        assert "response" not in schema["output"]["properties"]


class TestResolveOutputSchema:
    """Tests for resolve_output_schema validation."""

    def test_output_schema_without_tools_returns_schema(self):
        """LlmAgent with output_schema and no tools/sub_agents returns it."""
        agent = _make_llm_agent(output_schema=OutputModel)
        assert resolve_output_schema(agent) is OutputModel

    def test_output_schema_with_tools_raises(self):
        """LlmAgent with output_schema AND tools raises ValueError."""
        tool = MagicMock()
        tool.name = "search"
        agent = _make_llm_agent(output_schema=OutputModel, tools=[tool])

        with pytest.raises(ValueError, match="has output_schema set but also has tools"):
            resolve_output_schema(agent)

    def test_output_schema_with_sub_agents_raises(self):
        """LlmAgent with output_schema AND sub_agents raises ValueError."""
        sub = _make_llm_agent(name="sub")
        agent = _make_llm_agent(
            name="root", output_schema=OutputModel, sub_agents=[sub]
        )

        with pytest.raises(
            ValueError, match="has output_schema set but also has sub_agents"
        ):
            resolve_output_schema(agent)

    def test_output_schema_with_tools_and_sub_agents_raises(self):
        """LlmAgent with output_schema AND both tools and sub_agents raises."""
        tool = MagicMock()
        tool.name = "search"
        sub = _make_llm_agent(name="sub")
        agent = _make_llm_agent(
            name="root", output_schema=OutputModel, tools=[tool], sub_agents=[sub]
        )

        with pytest.raises(
            ValueError, match="has output_schema set but also has tools and sub_agents"
        ):
            resolve_output_schema(agent)

    def test_composite_agent_recurses_to_last_sub_agent(self):
        """Composite agent resolves output_schema from the last sub_agent."""
        formatter = _make_llm_agent(name="formatter", output_schema=OutputModel)
        worker = _make_llm_agent(name="worker")
        pipeline = _make_base_agent(
            name="pipeline", sub_agents=[worker, formatter]
        )

        assert resolve_output_schema(pipeline) is OutputModel

    def test_composite_agent_raises_if_last_sub_agent_invalid(self):
        """Composite agent raises if last sub_agent has output_schema + tools."""
        tool = MagicMock()
        tool.name = "search"
        bad_agent = _make_llm_agent(
            name="bad", output_schema=OutputModel, tools=[tool]
        )
        pipeline = _make_base_agent(name="pipeline", sub_agents=[bad_agent])

        with pytest.raises(ValueError, match="has output_schema set but also has tools"):
            resolve_output_schema(pipeline)


class TestGetAgentGraph:
    """Tests for get_agent_graph function."""

    def test_single_agent_graph(self):
        """Test graph for a single agent with no tools or sub-agents."""
        agent = _make_llm_agent(name="root")
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "__start__" in node_ids
        assert "__end__" in node_ids
        assert "root" in node_ids

        # Check start->root and root->end edges
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("__start__", "root") in edge_pairs
        assert ("root", "__end__") in edge_pairs

    def test_agent_with_tools(self):
        """Test graph for LlmAgent with regular tools."""
        tool1 = MagicMock()
        tool1.name = "search"
        tool1.__class__.__name__ = "FunctionTool"

        tool2 = MagicMock()
        tool2.name = "calculator"
        tool2.__class__.__name__ = "FunctionTool"

        agent = _make_llm_agent(name="root", tools=[tool1, tool2])
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "root_tools" in node_ids

        # Find tools node and check metadata
        tools_node = next(n for n in graph.nodes if n.id == "root_tools")
        assert tools_node.type == "tool"
        assert tools_node.metadata is not None
        assert tools_node.metadata["tool_count"] == 2
        assert "search" in tools_node.metadata["tool_names"]
        assert "calculator" in tools_node.metadata["tool_names"]

    def test_composite_agent_with_sub_agents(self):
        """Test graph for composite agent with sub-agents."""
        sub = _make_base_agent(name="sub_agent")
        agent = _make_base_agent(name="root", sub_agents=[sub])
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "root" in node_ids
        assert "sub_agent" in node_ids

        # Check bidirectional edges
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("root", "sub_agent") in edge_pairs
        assert ("sub_agent", "root") in edge_pairs

    def test_llm_agent_with_sub_agents(self):
        """Test graph for LlmAgent with sub-agents."""
        sub = _make_llm_agent(name="sub_agent")
        agent = _make_llm_agent(name="root", sub_agents=[sub])
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "root" in node_ids
        assert "sub_agent" in node_ids

    def test_composite_agent_no_tools_node(self):
        """Composite agents don't produce a tools node (they have no tools)."""
        agent = _make_base_agent(name="root")
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "root_tools" not in node_ids
