"""Tests for graph building utilities."""

from unittest.mock import MagicMock

from agent_framework import BaseAgent

from uipath_agent_framework.runtime.schema import get_agent_graph


def _make_agent(name="test_agent", tools=None) -> BaseAgent:
    """Create a mock BaseAgent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = name
    agent.default_options = {"tools": tools or []}
    return agent


class TestGraphStructure:
    """Tests for graph structure correctness."""

    def test_start_and_end_nodes_present(self):
        """Graph always has __start__ and __end__ nodes."""
        agent = _make_agent(name="my_agent")
        graph = get_agent_graph(agent)

        node_types = {n.type for n in graph.nodes}
        assert "__start__" in node_types
        assert "__end__" in node_types

    def test_agent_node_type_is_node(self):
        """Agent nodes have type 'node'."""
        agent = _make_agent(name="my_agent")
        graph = get_agent_graph(agent)

        agent_node = next(n for n in graph.nodes if n.id == "my_agent")
        assert agent_node.type == "node"

    def test_tools_node_type_is_tool(self):
        """Tools nodes have type 'tool'."""

        def search():
            pass

        search.__name__ = "search"
        tool = search

        agent = _make_agent(name="my_agent", tools=[tool])
        graph = get_agent_graph(agent)

        tools_node = next(n for n in graph.nodes if n.id == "my_agent_tools")
        assert tools_node.type == "tool"

    def test_start_connects_to_agent(self):
        """__start__ connects to the agent with 'input' label."""
        agent = _make_agent(name="my_agent")
        graph = get_agent_graph(agent)

        start_edge = next(e for e in graph.edges if e.source == "__start__")
        assert start_edge.target == "my_agent"
        assert start_edge.label == "input"

    def test_agent_connects_to_end(self):
        """Agent connects to __end__ with 'output' label."""
        agent = _make_agent(name="my_agent")
        graph = get_agent_graph(agent)

        end_edge = next(e for e in graph.edges if e.target == "__end__")
        assert end_edge.source == "my_agent"
        assert end_edge.label == "output"

    def test_tools_metadata_contains_names(self):
        """Tools node metadata includes tool names and count."""

        def search():
            pass

        search.__name__ = "search"

        def calculator():
            pass

        calculator.__name__ = "calculator"
        tool1 = search
        tool2 = calculator

        agent = _make_agent(name="agent", tools=[tool1, tool2])
        graph = get_agent_graph(agent)

        tools_node = next(n for n in graph.nodes if n.id == "agent_tools")
        assert tools_node.metadata is not None
        assert tools_node.metadata["tool_names"] == ["search", "calculator"]
        assert tools_node.metadata["tool_count"] == 2

    def test_agent_name_fallback(self):
        """Agent without name falls back to 'agent'."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = None
        agent.default_options = {"tools": []}

        graph = get_agent_graph(agent)
        node_ids = [n.id for n in graph.nodes]
        assert "agent" in node_ids

    def test_no_subgraph_or_metadata_on_simple_nodes(self):
        """Simple agent/start/end nodes have no subgraph or metadata."""
        agent = _make_agent(name="test")
        graph = get_agent_graph(agent)

        for node in graph.nodes:
            assert node.subgraph is None
            if node.type != "tool":
                assert node.metadata is None
