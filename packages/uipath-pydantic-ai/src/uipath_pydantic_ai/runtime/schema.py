"""Schema extraction utilities for PydanticAI Agents."""

import inspect
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent, FunctionToolset
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    transform_nullable_types,
    transform_references,
)


def _is_pydantic_model(type_hint: Any) -> bool:
    """Check if a type hint is a Pydantic BaseModel."""
    try:
        if inspect.isclass(type_hint) and issubclass(type_hint, BaseModel):
            return True

        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            for arg in args:
                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                    return True
    except TypeError:
        pass

    return False


def _get_agent_name(agent: Agent[Any, Any]) -> str:
    """Get the agent name, falling back to 'agent' if not set."""
    return agent.name or "agent"


def _get_tool_names(agent: Agent[Any, Any]) -> list[str]:
    """Get the list of function tool names from an agent."""
    tool_names: list[str] = []
    for toolset in agent.toolsets:
        if isinstance(toolset, FunctionToolset):
            tool_names.extend(toolset.tools.keys())
    return tool_names


def _get_output_type(agent: Agent[Any, Any]) -> Any:
    """Extract the output type from a PydanticAI Agent.

    Returns None if it's the default str type (no explicit output schema).
    """
    output_type = agent.output_type
    if output_type is str:
        return None
    return output_type


def get_deps_type(agent: Agent[Any, Any]) -> type[BaseModel] | None:
    """Extract the deps_type from a PydanticAI Agent if it's a Pydantic model.

    PydanticAI agents can declare deps_type=MyModel, which represents
    structured input data passed alongside the user prompt.
    Only Pydantic BaseModel subclasses are supported as structured deps.
    """
    deps_type = agent.deps_type
    if deps_type is not None and _is_pydantic_model(deps_type):
        return deps_type
    return None


def parse_input_to_deps(
    input_dict: dict[str, Any], deps_type: type[BaseModel]
) -> BaseModel:
    """Parse the input dict into a Pydantic deps model.

    All fields go to the deps model. If the model doesn't have a particular
    field (like 'messages'), Pydantic validation will ignore or reject it
    depending on the model's config.
    """
    try:
        return deps_type.model_validate(input_dict)
    except Exception as e:
        raise ValueError(f"Failed to parse deps: {e}") from e


def _extract_schema_from_model(model_type: type[BaseModel]) -> dict[str, Any] | None:
    """Extract a JSON schema from a Pydantic model, resolving $refs and nullable types."""
    try:
        adapter = TypeAdapter(model_type)
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


def get_entrypoints_schema(agent: Agent[Any, Any]) -> dict[str, Any]:
    """Extract input/output schema from a PydanticAI Agent.

    Input schema:
    - If agent has deps_type (Pydantic model): input IS that model's schema
    - Otherwise: conversational fallback with 'messages' field

    Output schema:
    - If agent has output_type (Pydantic model): output IS that model's schema
    - Otherwise: generic result fallback
    """
    # --- Input schema ---
    deps_type = get_deps_type(agent)
    if deps_type is not None:
        input_schema = _extract_schema_from_model(deps_type)
    else:
        input_schema = None

    if input_schema is None:
        # Conversational fallback: messages-only input
        input_schema = {
            "type": "object",
            "properties": {
                "messages": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "object"}},
                    ],
                    "title": "Messages",
                    "description": "User messages to send to the agent",
                },
            },
            "required": ["messages"],
        }

    # --- Output schema ---
    output_type = _get_output_type(agent)
    output_schema: dict[str, Any] | None = None

    if output_type is not None and _is_pydantic_model(output_type):
        output_schema = _extract_schema_from_model(output_type)

    if output_schema is None:
        # Generic result fallback
        output_schema = {
            "type": "object",
            "properties": {
                "result": {
                    "title": "Result",
                    "description": "The agent's response",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "object"},
                        {"type": "array", "items": {"type": "object"}},
                    ],
                }
            },
            "required": ["result"],
        }

    return {"input": input_schema, "output": output_schema}


def get_agent_schema(agent: Agent[Any, Any]) -> UiPathRuntimeGraph:
    """Extract graph structure from a PydanticAI Agent.

    PydanticAI Agents are represented as simple nodes. Function tools are
    aggregated into a single tools node per agent with metadata.
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []
    visited: set[str] = set()

    def _add_agent_and_tools(current_agent: Agent[Any, Any]) -> None:
        """Add agent and its tools to the graph."""
        agent_name = _get_agent_name(current_agent)

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

        tool_names = _get_tool_names(current_agent)

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
                UiPathRuntimeEdge(
                    source=agent_name,
                    target=tools_node_id,
                    label=None,
                )
            )
            edges.append(
                UiPathRuntimeEdge(
                    source=tools_node_id,
                    target=agent_name,
                    label=None,
                )
            )

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

    agent_name = _get_agent_name(agent)

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

    edges.append(
        UiPathRuntimeEdge(
            source="__start__",
            target=agent_name,
            label="input",
        )
    )

    edges.append(
        UiPathRuntimeEdge(
            source=agent_name,
            target="__end__",
            label="output",
        )
    )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


__all__ = [
    "get_entrypoints_schema",
    "get_agent_schema",
    "get_deps_type",
    "parse_input_to_deps",
]
