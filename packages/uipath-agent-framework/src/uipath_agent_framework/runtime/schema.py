"""Schema extraction utilities for Agent Framework agents."""

from collections.abc import Callable
from typing import Any

from agent_framework import (
    Agent,
    AgentExecutor,
    BaseAgent,
    Edge,
    Executor,
    FunctionTool,
    Workflow,
    WorkflowAgent,
)
from pydantic import BaseModel
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

    If the workflow's output executor has a response_format (Pydantic model),
    the output schema reflects that structured type instead of default messages.
    """
    output_schema = _extract_output_schema(agent) or _default_messages_schema()
    return {
        "input": _default_messages_schema(),
        "output": output_schema,
    }


def _extract_output_schema(agent: BaseAgent) -> dict[str, Any] | None:
    """Extract structured output schema from a WorkflowAgent's output executors.

    Checks if the output executor's inner agent has a response_format set to a
    Pydantic BaseModel. If so, returns its JSON schema as the output schema.
    """
    if not isinstance(agent, WorkflowAgent):
        return None

    try:
        output_executors = agent.workflow.get_output_executors()
    except Exception:
        return None

    for executor in output_executors:
        if not isinstance(executor, AgentExecutor):
            continue
        inner_agent = executor._agent
        if not isinstance(inner_agent, Agent):
            continue
        response_format = inner_agent.default_options.get("response_format")
        if response_format is None:
            continue
        try:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                return response_format.model_json_schema()
        except Exception:
            continue

    return None


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

    Only WorkflowAgent is supported. Extracts the underlying Workflow's
    executors and edge_groups to build a proper multi-agent graph.

    Args:
        agent: An Agent Framework BaseAgent instance (must be WorkflowAgent)

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the agent structure

    Raises:
        TypeError: If agent is not a WorkflowAgent
    """
    if isinstance(agent, WorkflowAgent):
        return _build_workflow_graph(agent.workflow)

    raise TypeError(
        f"Only WorkflowAgent is supported for graph extraction, got {type(agent).__name__}"
    )


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
    executor_ids: set[str] = set(executors.keys())

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
            _add_executor_tool_nodes(
                exec_id, executor._agent, nodes, edges, executor_ids
            )

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


def _is_handoff_tool(tool_name: str, executor_ids: set[str]) -> bool:
    """Check if a tool is a handoff tool by matching against executor IDs."""
    if not tool_name.startswith("handoff_to_"):
        return False
    target = tool_name[len("handoff_to_") :]
    return target in executor_ids


def _add_executor_tool_nodes(
    executor_id: str,
    agent: BaseAgent,
    nodes: list[UiPathRuntimeNode],
    edges: list[UiPathRuntimeEdge],
    executor_ids: set[str],
) -> None:
    """Add tool nodes for an executor's wrapped agent's tools.

    Handoff tools (tools that transfer control to another executor) are
    excluded since they are already represented as edges between nodes.
    """
    tools = get_agent_tools(agent)
    if not tools:
        return

    tool_names: list[str] = [
        n
        for t in tools
        if (n := get_tool_name(t)) is not None and not _is_handoff_tool(n, executor_ids)
    ]

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


def get_agent_tools(agent: BaseAgent) -> list[Any]:
    """Extract tools list from an Agent Framework agent.

    Tools are stored in agent.default_options["tools"], not on a .tools attribute.
    """
    return getattr(agent, "default_options", {}).get("tools", [])


def get_tool_name(tool: FunctionTool | Callable[..., Any]) -> str | None:
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
    "get_agent_tools",
    "get_tool_name",
]
