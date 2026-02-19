"""Schema extraction utilities for OpenAI Agents."""

import inspect
from typing import Any, get_args, get_origin

from agents import Agent
from pydantic import BaseModel, TypeAdapter
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    transform_nullable_types,
    transform_references,
)

from .context import get_agent_context_type


def _extract_agent_from_tool(tool: Any) -> Agent | None:
    """
    Extract an Agent from a tool that was created via Agent.as_tool().

    The agent is stored deep in the closure chain of the tool's on_invoke_tool function.
    """
    if not hasattr(tool, "on_invoke_tool"):
        return None

    try:
        func = tool.on_invoke_tool
        if not hasattr(func, "__closure__") or not func.__closure__:
            return None

        # First level: get _on_invoke_tool_impl
        impl = func.__closure__[0].cell_contents
        if (
            not callable(impl)
            or not hasattr(impl, "__closure__")
            or not impl.__closure__
        ):
            return None

        # Second level: get run_agent function
        run_agent = impl.__closure__[1].cell_contents
        if (
            not callable(run_agent)
            or not hasattr(run_agent, "__closure__")
            or not run_agent.__closure__
        ):
            return None

        # Third level: find the Agent in run_agent's closure
        for cell in run_agent.__closure__:
            content = cell.cell_contents
            if isinstance(content, Agent):
                return content

    except (IndexError, AttributeError):
        pass

    return None


def _is_pydantic_model(type_hint: Any) -> bool:
    """
    Check if a type hint is a Pydantic BaseModel.

    Args:
        type_hint: A type hint from type annotations

    Returns:
        True if the type is a Pydantic model, False otherwise
    """
    try:
        # Direct check
        if inspect.isclass(type_hint) and issubclass(type_hint, BaseModel):
            return True

        # Handle generic types (e.g., Optional[Model])
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            for arg in args:
                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                    return True

    except TypeError:
        pass

    return False


def get_entrypoints_schema(agent: Agent) -> dict[str, Any]:
    """
    Extract input/output schema from an OpenAI Agent.

    Input schema always includes 'messages' field, plus context fields if defined.
    Output schema uses agent's output_type attribute.
    """
    # Messages field is always required
    messages_schema: dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "object"}},
        ],
        "title": "Messages",
        "description": "User messages to send to the agent",
    }

    input_properties: dict[str, Any] = {"messages": messages_schema}
    input_required: list[str] = ["messages"]

    schema: dict[str, Any] = {
        "input": {
            "type": "object",
            "properties": input_properties,
            "required": input_required,
        },
        "output": {"type": "object", "properties": {}, "required": []},
    }

    # Add context fields if agent has a context type (Agent[MyContext])
    context_type = get_agent_context_type(agent)
    if context_type is not None and _is_pydantic_model(context_type):
        try:
            adapter = TypeAdapter(context_type)
            context_schema = adapter.json_schema()
            unpacked, _ = transform_references(context_schema)

            # Merge context properties with messages
            context_props = transform_nullable_types(unpacked.get("properties", {}))
            input_properties.update(context_props)

            # Add context required fields (messages is already required)
            for field in unpacked.get("required", []):
                if field not in input_required:
                    input_required.append(field)
        except Exception:
            pass

    # Extract output schema - Agent's output_type (native OpenAI Agents pattern)
    output_type = getattr(agent, "output_type", None)
    output_extracted = False

    # Unwrap AgentOutputSchema if present (OpenAI Agents SDK wrapper)
    # AgentOutputSchema wraps the actual Pydantic model in an 'output_type' attribute
    if (
        output_type is not None
        and hasattr(output_type, "output_type")
        and not isinstance(output_type, type)
    ):
        # This is an AgentOutputSchema wrapper instance, extract the actual model
        output_type = output_type.output_type

    if output_type is not None and _is_pydantic_model(output_type):
        try:
            adapter = TypeAdapter(output_type)
            output_schema = adapter.json_schema()

            # Resolve references and handle nullable types
            unpacked_output, _ = transform_references(output_schema)
            schema["output"]["properties"] = transform_nullable_types(
                unpacked_output.get("properties", {})
            )
            schema["output"]["required"] = unpacked_output.get("required", [])

            # Add title and description if available
            if "title" in unpacked_output:
                schema["output"]["title"] = unpacked_output["title"]
            if "description" in unpacked_output:
                schema["output"]["description"] = unpacked_output["description"]

            output_extracted = True
        except Exception:
            # Continue to fallback if extraction fails
            pass

    # Fallback: Default output schema for agents without explicit output_type
    if not output_extracted:
        schema["output"] = {
            "type": "object",
            "properties": {
                "result": {
                    "title": "Result",
                    "description": "The agent's response",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "object"},
                        {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    ],
                }
            },
            "required": ["result"],
        }

    return schema


def get_agent_schema(agent: Agent) -> UiPathRuntimeGraph:
    """
    Extract graph structure from an OpenAI Agent.

    OpenAI Agents are represented as simple nodes. Regular tools are aggregated
    into a single tools node per agent with metadata. Agent-tools and handoff
    agents are represented as separate agent nodes.

    Args:
        agent: An OpenAI Agent instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []
    visited: set[str] = set()  # Track visited agents to avoid circular references

    def _add_agent_and_tools(current_agent: Agent) -> None:
        """Recursively add agent, its tools, and nested agents to the graph."""
        agent_name = getattr(current_agent, "name", "agent")

        # Prevent circular references using agent name
        if agent_name in visited:
            return
        visited.add(agent_name)

        # Add agent node (first visit always adds the node)
        nodes.append(
            UiPathRuntimeNode(
                id=agent_name,
                name=agent_name,
                type="node",
                subgraph=None,
                metadata=None,
            )
        )

        # Process tools - separate agent-tools from regular function tools
        tools = getattr(current_agent, "tools", None) or []
        agent_tools: list[tuple[str, Agent]] = []  # (tool_name, agent)
        regular_tools: list[Any] = []

        for tool in tools:
            if isinstance(tool, Agent):
                # Direct Agent instance
                agent_name_str: str = getattr(tool, "name", None) or "agent"
                agent_tools.append((agent_name_str, tool))
            else:
                # Check if this is an agent wrapped via .as_tool()
                wrapped_agent = _extract_agent_from_tool(tool)
                if wrapped_agent is not None:
                    # Use the tool's name (e.g., "translate_to_spanish") for the edge
                    tool_name_str: str = (
                        _get_tool_name(tool)
                        or getattr(wrapped_agent, "name", None)
                        or "agent"
                    )
                    agent_tools.append((tool_name_str, wrapped_agent))
                else:
                    regular_tools.append(tool)

        # Process agent-tools (agents used as tools via .as_tool())
        for tool_name, tool_agent in agent_tools:
            tool_agent_name = getattr(tool_agent, "name", "agent")
            if tool_agent_name not in visited:
                # Recursively process agent-tool
                _add_agent_and_tools(tool_agent)

            # Add edges for agent-tool (even if already visited, we need edges)
            edges.append(
                UiPathRuntimeEdge(
                    source=agent_name,
                    target=tool_agent_name,
                    label=tool_name,
                )
            )
            edges.append(
                UiPathRuntimeEdge(
                    source=tool_agent_name,
                    target=agent_name,
                    label=None,
                )
            )

        # Process regular function tools - aggregate into single tools node
        if regular_tools:
            tool_names = [_get_tool_name(tool) for tool in regular_tools]
            tool_names = [name for name in tool_names if name]  # Filter out None values

            if tool_names:
                # Create a single tools node for this agent
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

                # Add bidirectional edges for tools node
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

        # Process handoff agents
        handoffs = getattr(current_agent, "handoffs", None) or []
        for handoff_agent in handoffs:
            handoff_name = getattr(handoff_agent, "name", None)
            if handoff_name and handoff_name not in visited:
                # Recursively process handoff agent
                _add_agent_and_tools(handoff_agent)

                # Add handoff edges without labels
                edges.append(
                    UiPathRuntimeEdge(
                        source=agent_name,
                        target=handoff_name,
                        label=None,
                    )
                )
                edges.append(
                    UiPathRuntimeEdge(
                        source=handoff_name,
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

    # Recursively build graph starting from main agent
    _add_agent_and_tools(agent)

    # Get the main agent name
    agent_name = getattr(agent, "name", "agent")

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

    # Connect start to main agent
    edges.append(
        UiPathRuntimeEdge(
            source="__start__",
            target=agent_name,
            label="input",
        )
    )

    # Connect main agent to end
    edges.append(
        UiPathRuntimeEdge(
            source=agent_name,
            target="__end__",
            label="output",
        )
    )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def _get_tool_name(tool: Any) -> str | None:
    """
    Extract the name of a tool from various tool types.

    Args:
        tool: A tool object (could be a function, class, or tool instance)

    Returns:
        The tool name as a string, or None if it cannot be determined
    """
    # Try common attributes for tool names
    if hasattr(tool, "name"):
        return str(tool.name)
    if hasattr(tool, "__name__"):
        return str(tool.__name__)
    if hasattr(tool, "tool_name"):
        return str(tool.tool_name)

    # For class-based tools, try to get class name
    if hasattr(tool, "__class__"):
        class_name = tool.__class__.__name__
        # Remove common suffixes like "Tool" for cleaner names
        if class_name.endswith("Tool"):
            return class_name[:-4].lower()
        return class_name.lower()

    return None


__all__ = [
    "get_entrypoints_schema",
    "get_agent_schema",
]
