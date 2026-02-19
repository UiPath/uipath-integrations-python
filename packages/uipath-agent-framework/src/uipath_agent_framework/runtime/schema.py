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

    Agent Framework agents are simple: one agent node with optional tools.
    No sub-agents or handoffs at the agent level.

    Args:
        agent: An Agent Framework BaseAgent instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []

    agent_name = agent.name or "agent"

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

    # Extract tools from the agent
    tool_names = _get_agent_tool_names(agent)
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
    edges.append(
        UiPathRuntimeEdge(source=agent_name, target="__end__", label="output")
    )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def _get_agent_tool_names(agent: BaseAgent) -> list[str]:
    """Extract tool names from an Agent Framework agent.

    Agent Framework tools are function callables passed as tools=[fn1, fn2].
    """
    tools = agent.tools or []
    tool_names: list[str] = []

    for tool in tools:
        name = _get_tool_name(tool)
        if name:
            tool_names.append(name)

    return tool_names


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
]
