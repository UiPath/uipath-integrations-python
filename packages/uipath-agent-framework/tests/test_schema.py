"""Tests for schema extraction utilities."""

from unittest.mock import MagicMock

from agent_framework import BaseAgent

from uipath_agent_framework.runtime.schema import (
    get_agent_graph,
    get_entrypoints_schema,
)


def _make_agent(name="test_agent", tools=None) -> BaseAgent:
    """Create a mock BaseAgent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = name
    agent.tools = tools or []
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


class TestGetAgentGraph:
    """Tests for get_agent_graph function."""

    def test_single_agent_graph(self):
        """Test graph for a single agent with no tools."""
        agent = _make_agent(name="root")
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
        """Test graph for agent with regular tools."""
        def search(): pass
        search.__name__ = "search"

        def calculator(): pass
        calculator.__name__ = "calculator"

        tool1 = search
        tool2 = calculator

        agent = _make_agent(name="root", tools=[tool1, tool2])
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

    def test_agent_without_tools_has_no_tools_node(self):
        """Agent without tools has no tools node."""
        agent = _make_agent(name="root", tools=[])
        graph = get_agent_graph(agent)

        node_ids = [n.id for n in graph.nodes]
        assert "root_tools" not in node_ids

    def test_graph_edges_are_bidirectional_for_tools(self):
        """Tools node has bidirectional edges with agent."""
        def my_tool(): pass
        my_tool.__name__ = "my_tool"
        tool = my_tool

        agent = _make_agent(name="root", tools=[tool])
        graph = get_agent_graph(agent)

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("root", "root_tools") in edge_pairs
        assert ("root_tools", "root") in edge_pairs

    def test_graph_has_correct_node_count_no_tools(self):
        """Graph with no tools has 3 nodes: __start__, agent, __end__."""
        agent = _make_agent(name="root")
        graph = get_agent_graph(agent)
        assert len(graph.nodes) == 3

    def test_graph_has_correct_node_count_with_tools(self):
        """Graph with tools has 4 nodes: __start__, agent, tools, __end__."""
        def my_tool(): pass
        my_tool.__name__ = "my_tool"
        tool = my_tool

        agent = _make_agent(name="root", tools=[tool])
        graph = get_agent_graph(agent)
        assert len(graph.nodes) == 4
