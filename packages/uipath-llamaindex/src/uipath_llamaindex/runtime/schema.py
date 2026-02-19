"""Schema extraction utilities for LlamaIndex workflows."""

from typing import Any

from llama_index.core.agent.workflow import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from pydantic import BaseModel
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    transform_nullable_types,
    transform_references,
)
from workflows import Workflow
from workflows.decorators import StepConfig
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StopEvent,
)


def get_entrypoints_schema(workflow: Workflow) -> dict[str, Any]:
    """
    Extract input/output schema from a LlamaIndex workflow.

    Args:
        workflow: A LlamaIndex Workflow instance

    Returns:
        Dictionary with input and output schemas
    """
    schema = {
        "input": {"type": "object", "properties": {}, "required": []},
        "output": {"type": "object", "properties": {}, "required": []},
    }

    # Find the actual StartEvent and StopEvent classes used in this workflow
    start_event_class = workflow._start_event_class
    stop_event_class = workflow._stop_event_class

    # Generate input schema from StartEvent
    try:
        if isinstance(workflow, BaseWorkflowAgent):
            # For workflow agents, define a simple schema with just user_msg
            schema["input"] = {
                "type": "object",
                "properties": {
                    "user_msg": {
                        "type": "string",
                        "title": "User Message",
                        "description": "The user's question or request",
                    }
                },
                "required": ["user_msg"],
            }
        else:
            input_schema = start_event_class.model_json_schema()
            # Resolve references and handle nullable types
            unpacked_input, _ = transform_references(input_schema)
            schema["input"]["properties"] = transform_nullable_types(
                unpacked_input.get("properties", {})
            )
            schema["input"]["required"] = unpacked_input.get("required", [])
    except (AttributeError, Exception):
        pass

    # Handle output schema - check if it's a workflow agent with output_cls first
    if isinstance(workflow, BaseWorkflowAgent):
        output_cls: type[BaseModel] | None = getattr(workflow, "output_cls", None)
        if output_cls is not None:
            try:
                output_schema = output_cls.model_json_schema()
                # Resolve references and handle nullable types
                unpacked_output, _ = transform_references(output_schema)
                schema["output"]["properties"] = transform_nullable_types(
                    unpacked_output.get("properties", {})
                )
                schema["output"]["required"] = unpacked_output.get("required", [])
            except (AttributeError, Exception):
                pass
    # Check if it's the base StopEvent or a custom subclass
    elif stop_event_class is StopEvent:
        # base StopEvent
        schema["output"] = {
            "type": "object",
            "properties": {
                "result": {
                    "title": "Result",
                    "type": "object",
                }
            },
            "required": ["result"],
        }
    else:
        # For custom StopEvent subclasses, extract their Pydantic schema
        try:
            output_schema = stop_event_class.model_json_schema()
            # Resolve references and handle nullable types
            unpacked_output, _ = transform_references(output_schema)
            schema["output"]["properties"] = transform_nullable_types(
                unpacked_output.get("properties", {})
            )
            schema["output"]["required"] = unpacked_output.get("required", [])
        except (AttributeError, Exception):
            pass

    return schema


def get_workflow_schema(workflow: Workflow) -> UiPathRuntimeGraph:
    """
    Extract graph structure from a LlamaIndex workflow.

    Args:
        workflow: A LlamaIndex Workflow instance

    Returns:
        UiPathRuntimeGraph with nodes and edges representing the workflow structure
    """
    nodes: list[UiPathRuntimeNode] = []
    edges: list[UiPathRuntimeEdge] = []

    nodes.append(
        UiPathRuntimeNode(
            id="__start__",
            name="__start__",
            type="__start__",
            subgraph=None,
        )
    )

    steps = workflow._get_steps()

    # Track if we need external step for human interaction
    has_human_interaction = False
    current_stop_event: type | None = None

    # First pass: find the StopEvent used in this workflow and check for human interaction
    for name, step_func in steps.items():
        step_config: StepConfig | None = get_step_config(name, step_func)
        if step_config is None:
            continue

        for return_type in step_config.return_types:
            if issubclass(return_type, StopEvent):
                current_stop_event = return_type
            if issubclass(return_type, (InputRequiredEvent, HumanResponseEvent)):
                has_human_interaction = True

        for event_type in step_config.accepted_events:
            if issubclass(event_type, (InputRequiredEvent, HumanResponseEvent)):
                has_human_interaction = True

    # Create step nodes (all steps are type "node")
    for step_name, step_func in steps.items():
        step_config = get_step_config(step_name, step_func)
        if step_config is None:
            continue

        node_type: str = "node"

        # Steps that execute tools - accept ToolCall
        for event_type in step_config.accepted_events:
            if issubclass(event_type, (ToolCall, ToolCallResult)):
                node_type = "tool"

        # Steps that produce AgentOutput are LLM steps (they call the model)
        for return_type in step_config.return_types:
            if return_type is not type(None) and issubclass(return_type, AgentOutput):
                node_type = "model"

        nodes.append(
            UiPathRuntimeNode(
                id=step_name,
                name=step_name,
                type=node_type,
                subgraph=None,
            )
        )

    # Add external step node if there are human interactions
    if has_human_interaction:
        nodes.append(
            UiPathRuntimeNode(
                id="external_step",
                name="external_step",
                type="external",
                subgraph=None,
            )
        )

    nodes.append(
        UiPathRuntimeNode(
            id="__end__",
            name="__end__",
            type="__end__",
            subgraph=None,
        )
    )

    # Create edges based on event flow
    start_event_class = workflow._start_event_class
    first_step_found = False

    for step_name, step_func in steps.items():
        step_config = get_step_config(step_name, step_func)
        if step_config is None:
            continue

        # Add edge from __start__ to first step that accepts the start event
        if not first_step_found and step_config.accepted_events:
            if start_event_class in step_config.accepted_events:
                edges.append(
                    UiPathRuntimeEdge(
                        source="__start__",
                        target=step_name,
                        label=start_event_class.__name__,
                    )
                )
                first_step_found = True

        # Add edges between steps based on event flow
        for return_type in step_config.return_types:
            if return_type is type(None):
                continue

            # If this returns StopEvent, connect to __end__
            if issubclass(return_type, StopEvent):
                if current_stop_event and return_type == current_stop_event:
                    edges.append(
                        UiPathRuntimeEdge(
                            source=step_name,
                            target="__end__",
                            label=return_type.__name__,
                        )
                    )
                continue  # Don't look for steps that accept StopEvent

            # Find steps that accept this return type
            for target_step_name, target_step_func in steps.items():
                target_config: StepConfig | None = get_step_config(
                    target_step_name, target_step_func
                )
                if target_config is None:
                    continue

                if return_type in target_config.accepted_events:
                    edges.append(
                        UiPathRuntimeEdge(
                            source=step_name,
                            target=target_step_name,
                            label=return_type.__name__,
                        )
                    )

            # If this returns InputRequiredEvent, add edge to external_step
            if issubclass(return_type, InputRequiredEvent):
                edges.append(
                    UiPathRuntimeEdge(
                        source=step_name,
                        target="external_step",
                        label=return_type.__name__,
                    )
                )

        # If this accepts HumanResponseEvent, add edge from external_step
        for event_type in step_config.accepted_events:
            if issubclass(event_type, HumanResponseEvent):
                edges.append(
                    UiPathRuntimeEdge(
                        source="external_step",
                        target=step_name,
                        label=event_type.__name__,
                    )
                )

    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def get_step_config(step_name: str, step_func: Any) -> StepConfig | None:
    """
    Get the step configuration from a step function.

    Returns None if:
    - The step name starts with underscore (internal method)
    - No step config is found

    Args:
        step_name: Name of the step
        step_func: The step function

    Returns:
        StepConfig if found and valid, None otherwise
    """
    # Skip internal methods
    if step_name.startswith("_"):
        return None

    return getattr(step_func, "_step_config", None) or getattr(
        step_func, "__step_config", None
    )


__all__ = [
    "get_entrypoints_schema",
    "get_workflow_schema",
]
