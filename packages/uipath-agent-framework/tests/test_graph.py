"""Tests for graph building utilities."""

from typing import Any
from unittest.mock import MagicMock

from agent_framework import (
    AgentExecutor,
    BaseAgent,
    Edge,
    Executor,
    Workflow,
    WorkflowAgent,
)

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


def _make_edge(
    source_id: str, target_id: str, condition_name: str | None = None
) -> Edge:
    """Create a mock Edge."""
    edge = MagicMock(spec=Edge)
    edge.source_id = source_id
    edge.target_id = target_id
    edge.condition_name = condition_name
    return edge


def _make_edge_group(edges: list[Edge], group_type: str = "SingleEdgeGroup"):
    """Create a mock EdgeGroup with the given type name."""
    group = MagicMock()
    group.edges = edges
    type(group).__name__ = group_type
    return group


def _make_executor(exec_id: str, agent: BaseAgent | None = None) -> Executor:
    """Create a mock Executor (AgentExecutor if agent is provided)."""
    if agent is not None:
        executor = MagicMock(spec=AgentExecutor)
        executor._agent = agent
    else:
        executor = MagicMock(spec=Executor)
    executor.id = exec_id
    return executor


def _make_workflow(
    executors: dict[str, Executor],
    edge_groups: list[Any],
    start_executor_id: str,
    output_executors: list[Executor] | None = None,
) -> Workflow:
    """Create a mock Workflow."""
    workflow = MagicMock(spec=Workflow)
    workflow.executors = executors
    workflow.edge_groups = edge_groups
    workflow.start_executor_id = start_executor_id
    workflow.get_output_executors.return_value = output_executors or []
    return workflow


def _make_workflow_agent(
    workflow: Workflow, name: str = "workflow_agent"
) -> WorkflowAgent:
    """Create a mock WorkflowAgent."""
    agent = MagicMock(spec=WorkflowAgent)
    agent.name = name
    agent.workflow = workflow
    return agent


class TestWorkflowGraph:
    """Tests for workflow-based graph building."""

    def test_simple_workflow_has_start_and_end(self):
        """Workflow graph has __start__ and __end__ nodes."""
        executor = _make_executor("agent_a", agent=None)
        workflow = _make_workflow(
            executors={"agent_a": executor},
            edge_groups=[],
            start_executor_id="agent_a",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        node_types = {n.type for n in graph.nodes}
        assert "__start__" in node_types
        assert "__end__" in node_types

    def test_workflow_executor_nodes(self):
        """Each executor becomes a node in the graph."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
            "tech": _make_executor("tech"),
        }
        workflow = _make_workflow(
            executors=executors,
            edge_groups=[],
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        node_ids = {n.id for n in graph.nodes}
        assert "triage" in node_ids
        assert "billing" in node_ids
        assert "tech" in node_ids

    def test_workflow_start_edge(self):
        """__start__ connects to the start executor."""
        executor = _make_executor("triage")
        workflow = _make_workflow(
            executors={"triage": executor},
            edge_groups=[],
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        start_edge = next(e for e in graph.edges if e.source == "__start__")
        assert start_edge.target == "triage"
        assert start_edge.label == "input"

    def test_workflow_edges_from_edge_groups(self):
        """Edge groups are translated into graph edges."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
        }
        edge_groups = [
            _make_edge_group([_make_edge("triage", "billing")], "SingleEdgeGroup"),
            _make_edge_group([_make_edge("billing", "triage")], "SingleEdgeGroup"),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("triage", "billing") in edge_pairs
        assert ("billing", "triage") in edge_pairs

    def test_workflow_fanout_edges(self):
        """FanOutEdgeGroup creates edges to multiple targets."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
            "tech": _make_executor("tech"),
        }
        edge_groups = [
            _make_edge_group(
                [
                    _make_edge("triage", "billing"),
                    _make_edge("triage", "tech"),
                ],
                "FanOutEdgeGroup",
            ),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("triage", "billing") in edge_pairs
        assert ("triage", "tech") in edge_pairs

    def test_workflow_fanin_edges(self):
        """FanInEdgeGroup creates edges from multiple sources."""
        executors = {
            "start": _make_executor("start"),
            "a": _make_executor("a"),
            "b": _make_executor("b"),
            "merge": _make_executor("merge"),
        }
        edge_groups = [
            _make_edge_group(
                [_make_edge("a", "merge"), _make_edge("b", "merge")],
                "FanInEdgeGroup",
            ),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="start",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("a", "merge") in edge_pairs
        assert ("b", "merge") in edge_pairs

    def test_workflow_internal_edges_skipped(self):
        """InternalEdgeGroup edges are excluded from the graph."""
        executors = {
            "a": _make_executor("a"),
            "b": _make_executor("b"),
        }
        edge_groups = [
            _make_edge_group([_make_edge("a", "b")], "InternalEdgeGroup"),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="a",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        # Only __start__→a and a→__end__ (fallback), no a→b internal edge
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("a", "b") not in edge_pairs

    def test_workflow_output_executors_connect_to_end(self):
        """Output executors connect to __end__."""
        out_exec = _make_executor("writer")
        executors = {
            "researcher": _make_executor("researcher"),
            "writer": out_exec,
        }
        workflow = _make_workflow(
            executors=executors,
            edge_groups=[],
            start_executor_id="researcher",
            output_executors=[out_exec],
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        end_edges = [e for e in graph.edges if e.target == "__end__"]
        assert len(end_edges) == 1
        assert end_edges[0].source == "writer"
        assert end_edges[0].label == "output"

    def test_workflow_no_output_executors_fallback(self):
        """Without output executors, start executor connects to __end__."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
        }
        workflow = _make_workflow(
            executors=executors,
            edge_groups=[],
            start_executor_id="triage",
            output_executors=[],
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        end_edges = [e for e in graph.edges if e.target == "__end__"]
        assert len(end_edges) == 1
        assert end_edges[0].source == "triage"

    def test_workflow_executor_with_tools(self):
        """Agent executors with tools get tool nodes."""

        def search_wikipedia():
            pass

        search_wikipedia.__name__ = "search_wikipedia"

        inner_agent = _make_agent(name="researcher", tools=[search_wikipedia])
        executors = {
            "researcher": _make_executor("researcher", agent=inner_agent),
        }
        workflow = _make_workflow(
            executors=executors,
            edge_groups=[],
            start_executor_id="researcher",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        node_ids = {n.id for n in graph.nodes}
        assert "researcher_tools" in node_ids

        tools_node = next(n for n in graph.nodes if n.id == "researcher_tools")
        assert tools_node.type == "tool"
        assert tools_node.metadata is not None
        assert "search_wikipedia" in tools_node.metadata["tool_names"]

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("researcher", "researcher_tools") in edge_pairs
        assert ("researcher_tools", "researcher") in edge_pairs

    def test_workflow_edge_condition_labels(self):
        """Conditional edges include condition_name as label."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
        }
        edge_groups = [
            _make_edge_group(
                [_make_edge("triage", "billing", condition_name="is_billing")],
                "SwitchCaseEdgeGroup",
            ),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        edge = next(
            e for e in graph.edges if e.source == "triage" and e.target == "billing"
        )
        assert edge.label == "is_billing"

    def test_workflow_handoff_pattern(self):
        """Full handoff pattern: triage routes to specialists."""
        executors = {
            "triage": _make_executor("triage"),
            "billing": _make_executor("billing"),
            "tech": _make_executor("tech"),
            "returns": _make_executor("returns"),
        }
        edge_groups = [
            _make_edge_group(
                [
                    _make_edge("triage", "billing"),
                    _make_edge("triage", "tech"),
                    _make_edge("triage", "returns"),
                ],
                "FanOutEdgeGroup",
            ),
            _make_edge_group([_make_edge("billing", "triage")], "SingleEdgeGroup"),
            _make_edge_group([_make_edge("tech", "triage")], "SingleEdgeGroup"),
            _make_edge_group([_make_edge("returns", "triage")], "SingleEdgeGroup"),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="triage",
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        node_ids = {n.id for n in graph.nodes}
        assert node_ids == {
            "__start__",
            "__end__",
            "triage",
            "billing",
            "tech",
            "returns",
        }

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        # Start → triage
        assert ("__start__", "triage") in edge_pairs
        # Triage routes to all specialists
        assert ("triage", "billing") in edge_pairs
        assert ("triage", "tech") in edge_pairs
        assert ("triage", "returns") in edge_pairs
        # Specialists route back to triage
        assert ("billing", "triage") in edge_pairs
        assert ("tech", "triage") in edge_pairs
        assert ("returns", "triage") in edge_pairs

    def test_workflow_concurrent_pattern(self):
        """Full concurrent pattern: fan-out then fan-in."""
        executors = {
            "sentiment": _make_executor("sentiment"),
            "topics": _make_executor("topics"),
            "summary": _make_executor("summary"),
            "merge": _make_executor("merge"),
        }
        merge_exec = executors["merge"]
        edge_groups = [
            _make_edge_group(
                [
                    _make_edge("sentiment", "merge"),
                    _make_edge("topics", "merge"),
                    _make_edge("summary", "merge"),
                ],
                "FanInEdgeGroup",
            ),
        ]
        workflow = _make_workflow(
            executors=executors,
            edge_groups=edge_groups,
            start_executor_id="sentiment",
            output_executors=[merge_exec],
        )
        agent = _make_workflow_agent(workflow)
        graph = get_agent_graph(agent)

        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("sentiment", "merge") in edge_pairs
        assert ("topics", "merge") in edge_pairs
        assert ("summary", "merge") in edge_pairs
        assert ("merge", "__end__") in edge_pairs
