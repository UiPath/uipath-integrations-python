"""Schema extraction utilities for Google ADK agents."""

import inspect
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from pydantic import BaseModel, TypeAdapter
from uipath.core.chat import UiPathConversationMessage
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    transform_nullable_types,
    transform_references,
)


def _is_pydantic_model(type_hint: Any) -> bool:
    """Check if a type hint is a Pydantic BaseModel class."""
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


def _is_llm_agent(agent: BaseAgent) -> bool:
    """Check if agent is an LlmAgent (has model, tools, input/output schema)."""
    return isinstance(agent, LlmAgent)


def resolve_input_schema(agent: BaseAgent) -> type[BaseModel] | None:
    """Resolve input schema from an agent, recursing into composite agents.

    Follows the same convention as google.adk.tools.agent_tool._get_input_schema:
    - LlmAgent: returns input_schema directly
    - Composite agent: recursively checks the FIRST sub_agent

    Note: input_schema is primarily used when the agent is wrapped in AgentTool.
    For root agent execution, ADK doesn't enforce it — the runner just receives
    a Content(text=...) message. However, our runtime serializes typed input as
    JSON text, which the LLM can understand.
    """
    if isinstance(agent, LlmAgent):
        return agent.input_schema
    if agent.sub_agents:
        return resolve_input_schema(agent.sub_agents[0])
    return None


def resolve_output_schema(agent: BaseAgent) -> type[BaseModel] | None:
    """Resolve output schema from an agent, recursing into composite agents.

    Follows the same convention as google.adk.tools.agent_tool._get_output_schema:
    - LlmAgent: returns output_schema directly
    - Composite agent: recursively checks the LAST sub_agent

    IMPORTANT: output_schema is only valid on LlmAgents that have NO tools
    and NO sub_agents. The Gemini API rejects combining response_mime_type
    'application/json' (set by output_schema) with function calling (tools).
    If an LlmAgent has output_schema AND tools/sub_agents, we raise a
    ValueError to catch the misconfiguration early (instead of getting a
    cryptic Gemini API 400 error at runtime).
    """
    if isinstance(agent, LlmAgent):
        if agent.output_schema:
            if agent.tools or agent.sub_agents:
                has = []
                if agent.tools:
                    has.append("tools")
                if agent.sub_agents:
                    has.append("sub_agents")
                raise ValueError(
                    f"LlmAgent '{agent.name}' has output_schema set but also has "
                    f"{' and '.join(has)}. The Gemini API rejects combining "
                    f"response_mime_type 'application/json' (set by output_schema) "
                    f"with function calling. Use the formatter pattern: "
                    f"SequentialAgent → [worker_agent, formatter_agent] where only "
                    f"the formatter has output_schema (no tools or sub_agents)."
                )
            return agent.output_schema
        return None
    if agent.sub_agents:
        return resolve_output_schema(agent.sub_agents[-1])
    return None


def resolve_output_key(agent: BaseAgent) -> str | None:
    """Resolve output_key from an agent, recursing into composite agents.

    output_key tells ADK to store the agent's final response text in
    session.state[output_key]. If output_schema is also set (and valid),
    the stored value is parsed JSON (dict), otherwise it's raw text.

    For composite agents, checks the LAST sub_agent (the final stage
    in a SequentialAgent pipeline produces the output).
    """
    if isinstance(agent, LlmAgent):
        return agent.output_key
    if agent.sub_agents:
        return resolve_output_key(agent.sub_agents[-1])
    return None


def get_entrypoints_schema(agent: BaseAgent) -> dict[str, Any]:
    """Extract input/output schema from a Google ADK Agent.

    Uses recursive resolution for composite agents (SequentialAgent, etc.):
    - Input: from the FIRST sub_agent's input_schema
    - Output: from the LAST sub_agent's output_schema or output_key

    This matches google.adk.tools.agent_tool conventions.
    """
    schema: dict[str, Any] = {
        "input": _default_input_schema(),
        "output": _default_output_schema(),
    }

    # Resolve input schema (recursive for composite agents)
    input_type = resolve_input_schema(agent)
    if input_type is not None and _is_pydantic_model(input_type):
        try:
            adapter = TypeAdapter(input_type)
            input_json_schema = adapter.json_schema()
            unpacked, _ = transform_references(input_json_schema)

            schema["input"] = {
                "type": "object",
                "properties": transform_nullable_types(unpacked.get("properties", {})),
                "required": unpacked.get("required", []),
            }
            if "title" in unpacked:
                schema["input"]["title"] = unpacked["title"]
            if "description" in unpacked:
                schema["input"]["description"] = unpacked["description"]
        except Exception:
            pass

    # Resolve output schema (recursive for composite agents).
    # resolve_output_schema returns None if agent has tools/sub_agents
    # (output_schema + function calling is invalid in Gemini API).
    output_type = resolve_output_schema(agent)
    if output_type is not None and _is_pydantic_model(output_type):
        try:
            adapter = TypeAdapter(output_type)
            output_json_schema = adapter.json_schema()
            unpacked_output, _ = transform_references(output_json_schema)

            schema["output"] = {
                "type": "object",
                "properties": transform_nullable_types(
                    unpacked_output.get("properties", {})
                ),
                "required": unpacked_output.get("required", []),
            }
            if "title" in unpacked_output:
                schema["output"]["title"] = unpacked_output["title"]
            if "description" in unpacked_output:
                schema["output"]["description"] = unpacked_output["description"]
        except Exception:
            pass
    else:
        # Fallback: check output_key (stores text in session state)
        output_key = resolve_output_key(agent)
        if output_key:
            schema["output"] = {
                "type": "object",
                "properties": {
                    output_key: {
                        "title": output_key,
                        "description": (
                            f"Agent output stored in session state key: {output_key}"
                        ),
                        "type": "string",
                    }
                },
                "required": [output_key],
            }

    return schema


def _conversation_messages_schema() -> dict[str, Any]:
    """Generate JSON schema for list[UiPathConversationMessage]."""
    adapter = TypeAdapter(list[UiPathConversationMessage])
    return adapter.json_schema()


def _default_input_schema() -> dict[str, Any]:
    """Default input schema using UiPath conversation message format."""
    messages_schema = _conversation_messages_schema()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": messages_schema["items"],
                "title": "Messages",
                "description": "UiPath conversation messages",
            }
        },
        "required": ["messages"],
    }
    if "$defs" in messages_schema:
        schema["$defs"] = messages_schema["$defs"]
    return schema


def _default_output_schema() -> dict[str, Any]:
    """Default output schema using UiPath conversation message format."""
    messages_schema = _conversation_messages_schema()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": messages_schema["items"],
                "title": "Messages",
                "description": "UiPath conversation messages",
            }
        },
        "required": ["messages"],
    }
    if "$defs" in messages_schema:
        schema["$defs"] = messages_schema["$defs"]
    return schema


def get_agent_graph(agent: BaseAgent) -> UiPathRuntimeGraph:
    """
    Extract graph structure from a Google ADK Agent.

    Traverses the agent tree via BaseAgent.sub_agents (available on all agent
    types). For LlmAgents, also inspects LlmAgent.tools for AgentTool instances
    and regular tools.

    Args:
        agent: A Google ADK BaseAgent instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []
    visited: set[str] = set()

    def _add_agent_and_tools(current_agent: BaseAgent) -> None:
        """Recursively add agent, its tools, and nested agents to the graph."""
        agent_name = current_agent.name

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

        # Only LlmAgent has tools; composite agents don't.
        if _is_llm_agent(current_agent):
            assert isinstance(current_agent, LlmAgent)
            _process_tools(current_agent, agent_name, nodes, edges, visited)

        # All agents have sub_agents (defined on BaseAgent)
        for sub_agent in current_agent.sub_agents:
            sub_name = sub_agent.name
            if sub_name not in visited:
                _add_agent_and_tools(sub_agent)

            edges.append(
                UiPathRuntimeEdge(source=agent_name, target=sub_name, label=None)
            )
            edges.append(
                UiPathRuntimeEdge(source=sub_name, target=agent_name, label=None)
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

    agent_name = agent.name

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
        UiPathRuntimeEdge(source="__start__", target=agent_name, label="input")
    )
    edges.append(UiPathRuntimeEdge(source=agent_name, target="__end__", label="output"))

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def _process_tools(
    agent: LlmAgent,
    agent_name: str,
    nodes: list[UiPathRuntimeNode],
    edges: list[UiPathRuntimeEdge],
    visited: set[str],
) -> None:
    """Process an LlmAgent's tools list, separating AgentTools from regular tools."""
    agent_tools: list[tuple[str, BaseAgent]] = []
    regular_tools: list[Any] = []

    for tool in agent.tools:
        inner_agent = _extract_agent_from_tool(tool)
        if inner_agent is not None:
            tool_name = _get_tool_name(tool) or str(inner_agent.name)
            agent_tools.append((tool_name, inner_agent))
        else:
            regular_tools.append(tool)

    # Agent-as-tool: add the wrapped agent as a node
    for tool_name, tool_agent in agent_tools:
        tool_agent_name = tool_agent.name
        if tool_agent_name not in visited:
            visited.add(tool_agent_name)
            nodes.append(
                UiPathRuntimeNode(
                    id=tool_agent_name,
                    name=tool_agent_name,
                    type="node",
                    subgraph=None,
                    metadata=None,
                )
            )

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


ToolType = Union[Callable[..., Any], BaseTool, BaseToolset]


def _extract_agent_from_tool(tool: ToolType) -> BaseAgent | None:
    """
    Extract a BaseAgent from an AgentTool instance.

    google.adk.tools.AgentTool wraps an agent as a tool, exposing it
    via the 'agent' attribute.
    """
    if isinstance(tool, AgentTool) and isinstance(tool.agent, BaseAgent):
        return tool.agent
    return None


def _get_tool_name(tool: ToolType) -> str | None:
    """Extract the name of a tool.

    Handles the three tool types in LlmAgent.tools:
    - BaseTool: has a 'name' attribute set in __init__
    - Callable: has '__name__' (function name)
    - BaseToolset: may have a 'name' attribute
    """
    if isinstance(tool, BaseTool):
        return str(tool.name)
    if callable(tool) and hasattr(tool, "__name__"):
        return str(tool.__name__)
    if hasattr(tool, "name"):
        return str(tool.name)
    return None


__all__ = [
    "get_entrypoints_schema",
    "get_agent_graph",
    "resolve_input_schema",
    "resolve_output_schema",
    "resolve_output_key",
]
