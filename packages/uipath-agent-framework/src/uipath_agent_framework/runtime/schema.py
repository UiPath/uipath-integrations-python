"""Schema extraction utilities for Agent Framework agents."""

from collections.abc import Callable
from typing import Any

from agent_framework import (
    AgentExecutor,
    BaseAgent,
    Edge,
    Executor,
    FunctionTool,
    Workflow,
    WorkflowAgent,
)
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
)


def get_entrypoints_schema(agent: BaseAgent) -> dict[str, Any]:
    """Extract input/output schema from an Agent Framework agent.

    Agent Framework agents are conversational — they always take messages
    as input and return conversation messages as output. Uses the standard
    UiPath conversation message format (matching Google ADK pattern).
    """
    return {
        "input": _default_messages_schema(),
        "output": _default_messages_schema(),
    }


def _conversation_message_item_schema() -> dict[str, Any]:
    """Minimal message schema: role and contentParts required, contentParts items only need data.inline."""
    return {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "contentParts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mimeType": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "inline": {},
                            },
                            "required": ["inline"],
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["data"],
                },
            },
            "toolCalls": {"type": "array", "items": {"type": "object"}},
            "interrupts": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["role", "contentParts"],
    }


def _default_messages_schema() -> dict[str, Any]:
    """Default messages schema using UiPath conversation message format."""
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": _conversation_message_item_schema(),
                "title": "Messages",
                "description": "UiPath conversation messages",
            }
        },
        "required": ["messages"],
    }


def get_agent_graph(agent: BaseAgent) -> UiPathRuntimeGraph:
    """Extract graph structure from an Agent Framework agent.

    Handles two cases:
    1. WorkflowAgent (from orchestrations): extracts the underlying Workflow's
       executors and edge_groups to build a proper multi-agent graph.
    2. Regular BaseAgent: traverses the agent tree, inspecting tools for
       agent-as-tool instances (created via BaseAgent.as_tool()).

    Args:
        agent: An Agent Framework BaseAgent instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure
    """
    if isinstance(agent, WorkflowAgent):
        return _build_workflow_graph(agent.workflow)

    return _build_agent_graph(agent)


def _build_workflow_graph(workflow: Workflow) -> UiPathRuntimeGraph:
    """Build graph from a Workflow's executors and edge groups.

    Traverses the workflow structure to create nodes for each executor
    and edges from the workflow's edge groups. For AgentExecutors that
    wrap agents with tools, also creates tool nodes.
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []

    # Add __start__ and __end__
    nodes.append(
        UiPathRuntimeNode(
            id="__start__",
            name="__start__",
            type="__start__",
            subgraph=None,
            metadata=None,
        )
    )
    nodes.append(
        UiPathRuntimeNode(
            id="__end__",
            name="__end__",
            type="__end__",
            subgraph=None,
            metadata=None,
        )
    )

    executors: dict[str, Executor] = workflow.executors
    start_id: str = workflow.start_executor_id

    # Add a node for each executor
    for exec_id, executor in executors.items():
        nodes.append(
            UiPathRuntimeNode(
                id=exec_id,
                name=exec_id,
                type="node",
                subgraph=None,
                metadata=None,
            )
        )

        # AgentExecutors wrap a BaseAgent that may have tools
        if isinstance(executor, AgentExecutor):
            inner_agent: BaseAgent | None = getattr(executor, "_agent", None)
            if inner_agent is not None:
                _add_executor_tool_nodes(exec_id, inner_agent, nodes, edges)

    # Connect __start__ → start executor
    edges.append(UiPathRuntimeEdge(source="__start__", target=start_id, label="input"))

    # Process edge groups into graph edges
    for edge_group in workflow.edge_groups:
        group_type = type(edge_group).__name__
        if group_type == "InternalEdgeGroup":
            continue

        edge: Edge
        for edge in edge_group.edges:
            label = edge.condition_name
            edges.append(
                UiPathRuntimeEdge(
                    source=edge.source_id, target=edge.target_id, label=label
                )
            )

    # Connect output executors → __end__
    output_executors: list[Executor] = []
    try:
        output_executors = workflow.get_output_executors()
    except Exception:
        pass

    if output_executors:
        for executor in output_executors:
            edges.append(
                UiPathRuntimeEdge(source=executor.id, target="__end__", label="output")
            )
    else:
        # Fallback: connect start executor to __end__
        edges.append(
            UiPathRuntimeEdge(source=start_id, target="__end__", label="output")
        )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def _add_executor_tool_nodes(
    executor_id: str,
    agent: BaseAgent,
    nodes: list[UiPathRuntimeNode],
    edges: list[UiPathRuntimeEdge],
) -> None:
    """Add tool nodes for an executor's wrapped agent's tools."""
    tools = get_agent_tools(agent)
    if not tools:
        return

    regular_tools = [t for t in tools if extract_agent_from_tool(t) is None]
    if not regular_tools:
        return

    tool_names = [_get_tool_name(t) for t in regular_tools]
    tool_names = [n for n in tool_names if n]

    if tool_names:
        tools_node_id = f"{executor_id}_tools"
        nodes.append(
            UiPathRuntimeNode(
                id=tools_node_id,
                name="tools",
                type="tool",
                subgraph=None,
                metadata={
                    "tool_names": tool_names,
                    "tool_count": len(tool_names),
                },
            )
        )
        edges.append(
            UiPathRuntimeEdge(source=executor_id, target=tools_node_id, label=None)
        )
        edges.append(
            UiPathRuntimeEdge(source=tools_node_id, target=executor_id, label=None)
        )


def _build_agent_graph(agent: BaseAgent) -> UiPathRuntimeGraph:
    """Build graph from a regular BaseAgent with tools.

    Traverses the agent tree, inspecting tools for agent-as-tool instances
    (created via BaseAgent.as_tool()). For each agent-as-tool, creates a
    separate node and recursively processes its own tools.
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []
    visited: set[str] = set()

    def _add_agent_and_tools(current_agent: BaseAgent) -> None:
        """Recursively add agent, its tools, and nested agents to the graph."""
        agent_name = current_agent.name or "agent"

        if agent_name in visited:
            return
        visited.add(agent_name)

        # Add agent node
        nodes.append(
            UiPathRuntimeNode(
                id=agent_name,
                name=agent_name,
                type="node",
                subgraph=None,
                metadata=None,
            )
        )

        # Process tools: separate agent-as-tool from regular tools
        _process_tools(current_agent, agent_name, nodes, edges, visited)

    # Add __start__ node
    nodes.append(
        UiPathRuntimeNode(
            id="__start__",
            name="__start__",
            type="__start__",
            subgraph=None,
            metadata=None,
        )
    )

    _add_agent_and_tools(agent)

    agent_name = agent.name or "agent"

    # Add __end__ node
    nodes.append(
        UiPathRuntimeNode(
            id="__end__",
            name="__end__",
            type="__end__",
            subgraph=None,
            metadata=None,
        )
    )

    # Connect start → agent → end
    edges.append(
        UiPathRuntimeEdge(source="__start__", target=agent_name, label="input")
    )
    edges.append(UiPathRuntimeEdge(source=agent_name, target="__end__", label="output"))

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def _process_tools(
    agent: BaseAgent,
    agent_name: str,
    nodes: list[UiPathRuntimeNode],
    edges: list[UiPathRuntimeEdge],
    visited: set[str],
) -> None:
    """Process an agent's tools list, separating agent-as-tools from regular tools."""
    tools = get_agent_tools(agent)

    agent_tools: list[tuple[str, BaseAgent]] = []
    regular_tools: list[Any] = []

    for tool in tools:
        inner_agent = extract_agent_from_tool(tool)
        if inner_agent is not None:
            tool_name = _get_tool_name(tool) or (inner_agent.name or "agent")
            agent_tools.append((tool_name, inner_agent))
        else:
            regular_tools.append(tool)

    # Agent-as-tool: add the wrapped agent as a node and recurse
    for tool_name, tool_agent in agent_tools:
        tool_agent_name = tool_agent.name or "agent"
        if tool_agent_name not in visited:
            # Recursively add the sub-agent and its own tools
            _add_agent_node(tool_agent, nodes, edges, visited)

        edges.append(
            UiPathRuntimeEdge(
                source=agent_name, target=tool_agent_name, label=tool_name
            )
        )
        edges.append(
            UiPathRuntimeEdge(source=tool_agent_name, target=agent_name, label=None)
        )

    # Regular tools — aggregate into single tools node
    if regular_tools:
        tool_names = [_get_tool_name(t) for t in regular_tools]
        tool_names = [n for n in tool_names if n]

        if tool_names:
            tools_node_id = f"{agent_name}_tools"
            nodes.append(
                UiPathRuntimeNode(
                    id=tools_node_id,
                    name="tools",
                    type="tool",
                    subgraph=None,
                    metadata={
                        "tool_names": tool_names,
                        "tool_count": len(tool_names),
                    },
                )
            )
            edges.append(
                UiPathRuntimeEdge(source=agent_name, target=tools_node_id, label=None)
            )
            edges.append(
                UiPathRuntimeEdge(source=tools_node_id, target=agent_name, label=None)
            )


def _add_agent_node(
    agent: BaseAgent,
    nodes: list[UiPathRuntimeNode],
    edges: list[UiPathRuntimeEdge],
    visited: set[str],
) -> None:
    """Add an agent node and recursively process its tools."""
    agent_name = agent.name or "agent"

    if agent_name in visited:
        return
    visited.add(agent_name)

    nodes.append(
        UiPathRuntimeNode(
            id=agent_name,
            name=agent_name,
            type="node",
            subgraph=None,
            metadata=None,
        )
    )

    _process_tools(agent, agent_name, nodes, edges, visited)


_extract_cache: dict[int, BaseAgent | None] = {}


def extract_agent_from_tool(
    tool: FunctionTool | Callable[..., Any],
) -> BaseAgent | None:
    """Extract a BaseAgent from a tool created via BaseAgent.as_tool().

    The as_tool() method creates an async agent_wrapper closure that captures
    `self` (the BaseAgent instance). We inspect the closure cells to find it.
    Results are cached by tool identity to avoid repeated introspection.
    """
    tool_id = id(tool)
    if tool_id in _extract_cache:
        return _extract_cache[tool_id]

    result = _extract_agent_from_closure(tool)
    _extract_cache[tool_id] = result
    return result


def _extract_agent_from_closure(
    tool: FunctionTool | Callable[..., Any],
) -> BaseAgent | None:
    if not isinstance(tool, FunctionTool):
        return None

    func = getattr(tool, "func", None)
    if func is None:
        return None

    closure = getattr(func, "__closure__", None)
    if not closure:
        return None

    for cell in closure:
        try:
            content = cell.cell_contents
            if isinstance(content, BaseAgent):
                return content
        except ValueError:
            continue

    return None


def get_agent_tools(agent: BaseAgent) -> list[Any]:
    """Extract tools list from an Agent Framework agent.

    Tools are stored in agent.default_options["tools"], not on a .tools attribute.
    """
    return getattr(agent, "default_options", {}).get("tools", [])


def _get_tool_name(tool: FunctionTool | Callable[..., Any]) -> str | None:
    """Extract the name of a tool.

    Tools in Agent Framework are either FunctionTool instances or plain callables.
    """
    if isinstance(tool, FunctionTool):
        return tool.name
    if callable(tool) and hasattr(tool, "__name__"):
        return tool.__name__
    return None


__all__ = [
    "get_entrypoints_schema",
    "get_agent_graph",
    "extract_agent_from_tool",
    "get_agent_tools",
]
