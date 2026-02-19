"""Schema extraction utilities for Agent Framework agents."""

from collections.abc import Callable
from typing import Any

from agent_framework import BaseAgent, FunctionTool
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

    Traverses the agent tree, inspecting tools for agent-as-tool instances
    (created via BaseAgent.as_tool()). For each agent-as-tool, creates a
    separate node and recursively processes its own tools.

    Args:
        agent: An Agent Framework BaseAgent instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure
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


def extract_agent_from_tool(
    tool: FunctionTool | Callable[..., Any],
) -> BaseAgent | None:
    """Extract a BaseAgent from a tool created via BaseAgent.as_tool().

    The as_tool() method creates an async agent_wrapper closure that captures
    `self` (the BaseAgent instance). We inspect the closure cells to find it.
    """
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
