"""Schema extraction utilities for ClaudeAgent definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    transform_nullable_types,
    transform_references,
)

from ..agent import ClaudeAgent


def _extract_schema_from_model(model_type: type[BaseModel]) -> dict[str, Any] | None:
    """Extract a JSON schema from a Pydantic model, resolving $refs and nullable types."""
    try:
        adapter: TypeAdapter[Any] = TypeAdapter(model_type)
        raw_schema = adapter.json_schema()
        unpacked, _ = transform_references(raw_schema)

        result: dict[str, Any] = {
            "type": "object",
            "properties": transform_nullable_types(unpacked.get("properties", {})),
            "required": unpacked.get("required", []),
        }

        if "title" in unpacked:
            result["title"] = unpacked["title"]
        if "description" in unpacked:
            result["description"] = unpacked["description"]

        return result
    except Exception:
        return None


def _default_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "title": "Input",
                "description": "User message for the agent",
            }
        },
        "required": ["input"],
    }


def _default_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "title": "Result",
                "description": "Final agent response",
            }
        },
        "required": ["result"],
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
                    },
                    "required": ["data"],
                },
            },
        },
        "required": ["role", "contentParts"],
    }


def default_messages_schema() -> dict[str, Any]:
    """Default messages schema using UiPath conversation message format.

    Nothing is required. A conversation is fed by its host rather than typed
    into a form, and an exchange reports its reply over the chat bridge rather
    than in the run's output, so requiring ``messages`` on either side would
    reject an empty payload the runtime handles and promise an output the
    runtime never returns.
    """
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
        "required": [],
    }


def _output_schema_from_options(agent: ClaudeAgent) -> dict[str, Any] | None:
    """Derive the output schema from the SDK-native options.output_format."""
    output_format = agent.options.output_format
    if (
        isinstance(output_format, dict)
        and output_format.get("type") == "json_schema"
        and isinstance(output_format.get("schema"), dict)
    ):
        return output_format["schema"]
    return None


def get_entrypoints_schema(
    agent: ClaudeAgent, *, conversational: bool = False
) -> dict[str, Any]:
    """Extract input/output schema from a ClaudeAgent.

    Conversational agents always use the UiPath conversation message format.
    Standard agents use the declared input/output Pydantic models when set,
    then the SDK-native ``options.output_format`` json_schema for output,
    falling back to a generic string input/result.
    """
    if conversational:
        return {
            "input": default_messages_schema(),
            "output": default_messages_schema(),
        }

    input_schema = (
        _extract_schema_from_model(agent.input_schema) if agent.input_schema else None
    )
    output_schema = (
        _extract_schema_from_model(agent.output_schema) if agent.output_schema else None
    ) or _output_schema_from_options(agent)

    return {
        "input": input_schema or _default_input_schema(),
        "output": output_schema or _default_output_schema(),
    }


def get_agent_graph(agent: ClaudeAgent) -> UiPathRuntimeGraph:
    """Build a visualization graph from a ClaudeAgent.

    Claude SDK agents are represented as a single model node. MCP servers are
    aggregated into a single tools node with metadata.
    """
    agent_name = agent.name or "agent"
    model = agent.uipath_llm.model if agent.uipath_llm else agent.options.model

    nodes: list[UiPathRuntimeNode] = [
        UiPathRuntimeNode(
            id="__start__",
            name="__start__",
            type="__start__",
            subgraph=None,
            metadata=None,
        ),
        UiPathRuntimeNode(
            id=agent_name,
            name=agent_name,
            type="model" if model else "node",
            subgraph=None,
            metadata={"model_name": model} if model else None,
        ),
    ]
    edges: list[UiPathRuntimeEdge] = [
        UiPathRuntimeEdge(source="__start__", target=agent_name, label="input"),
    ]

    mcp_servers = agent.options.mcp_servers
    server_names = list(mcp_servers.keys()) if isinstance(mcp_servers, dict) else []
    builtin_tools = agent.options.tools if isinstance(agent.options.tools, list) else []
    tool_names = [*server_names, *builtin_tools]

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
                    "mcp_servers": server_names,
                    "builtin_tools": builtin_tools,
                },
            )
        )
        edges.append(
            UiPathRuntimeEdge(source=agent_name, target=tools_node_id, label=None)
        )
        edges.append(
            UiPathRuntimeEdge(source=tools_node_id, target=agent_name, label=None)
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
    edges.append(UiPathRuntimeEdge(source=agent_name, target="__end__", label="output"))

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


__all__ = [
    "get_entrypoints_schema",
    "get_agent_graph",
    "default_messages_schema",
]
