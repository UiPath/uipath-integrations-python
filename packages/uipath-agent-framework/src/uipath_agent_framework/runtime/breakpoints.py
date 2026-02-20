"""Breakpoint management for the Agent Framework runtime.

Implements breakpoints by wrapping executor.execute() methods so that
execution pauses BEFORE the executor runs. This works regardless of
the inner agent type (RawAgent, Agent, etc.) because interception
happens at the executor level, not via agent middleware.

The debug UI sends graph node IDs which are resolved to executor IDs:
- ``"*"`` → all executors
- Executor IDs (e.g. ``"triage"``) → that executor
- Tools container IDs (e.g. ``"triage_tools"``) → the parent executor
- Tool names (e.g. ``"calculator"``) → the executor that owns that tool
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from agent_framework import AgentExecutor, WorkflowAgent
from uipath.runtime.debug import UiPathBreakpointResult

from .interrupt import AgentInterruptException
from .schema import get_agent_tools, get_tool_name

_ORIGINAL_EXECUTE_ATTR = "_bp_original_execute"


def _build_executor_tool_map(agent: WorkflowAgent) -> dict[str, set[str]]:
    """Build a mapping of executor_id -> set of tool names."""
    tool_map: dict[str, set[str]] = {}
    for exec_id, executor in agent.workflow.executors.items():
        if isinstance(executor, AgentExecutor):
            inner = getattr(executor, "_agent", None)
            if inner is not None:
                tools = get_agent_tools(inner)
                names = {get_tool_name(t) for t in tools if get_tool_name(t)}
                tool_map[exec_id] = names
    return tool_map


def _resolve_to_executor_ids(
    agent: WorkflowAgent,
    breakpoints: list[str] | str,
) -> set[str]:
    """Resolve graph node IDs to executor IDs.

    Maps breakpoint node IDs from the debug UI to the actual executor IDs
    in the workflow so we know which executors to wrap.
    """
    if breakpoints == "*" or (isinstance(breakpoints, list) and "*" in breakpoints):
        return set(agent.workflow.executors.keys())

    all_executors = set(agent.workflow.executors.keys())
    tool_map = _build_executor_tool_map(agent)

    # Reverse map: tool_name -> executor_id
    tool_to_executor: dict[str, str] = {}
    for exec_id, tool_names in tool_map.items():
        for name in tool_names:
            tool_to_executor[name] = exec_id

    resolved: set[str] = set()

    for bp in breakpoints:
        if bp in all_executors:
            # Direct executor ID
            resolved.add(bp)
        elif bp.endswith("_tools"):
            # Tools container node → parent executor
            exec_id = bp[: -len("_tools")]
            if exec_id in all_executors:
                resolved.add(exec_id)
        elif bp in tool_to_executor:
            # Tool name → owning executor
            resolved.add(tool_to_executor[bp])

    return resolved


def inject_breakpoint_middleware(
    agent: WorkflowAgent,
    breakpoints: list[str] | str,
    skip_nodes: set[str] | None = None,
) -> None:
    """Wrap executor.execute() to pause before breakpointed executors run.

    Replaces each matching executor's execute() with a wrapper that raises
    AgentInterruptException(is_breakpoint=True) before the executor runs.

    Args:
        agent: The workflow agent whose executors to wrap.
        breakpoints: ``"*"`` or a list of node IDs from the debug UI.
        skip_nodes: Executor IDs to skip (for resume after breakpoint).
            In concurrent workflows multiple executors may have been
            breakpointed across sequential resumes within the same
            superstep, so all of them must be skipped.
    """
    executor_ids = _resolve_to_executor_ids(agent, breakpoints)

    for exec_id in executor_ids:
        executor = agent.workflow.executors.get(exec_id)
        if executor is None:
            continue

        # Don't double-wrap
        if hasattr(executor, _ORIGINAL_EXECUTE_ATTR):
            continue

        # Skip executors already resumed past
        if skip_nodes and exec_id in skip_nodes:
            continue

        original = executor.execute

        async def wrapped_execute(
            message: Any,
            source_executor_ids: list[str],
            state: Any,
            runner_context: Any,
            trace_contexts: list[dict[str, str]] | None = None,
            source_span_ids: list[str] | None = None,
            *,
            _exec_id: str = exec_id,
        ) -> None:
            raise AgentInterruptException(
                interrupt_id=str(uuid4()),
                suspend_value={
                    "type": "breakpoint",
                    "node_id": _exec_id,
                },
                is_breakpoint=True,
            )

        setattr(executor, _ORIGINAL_EXECUTE_ATTR, original)
        executor.execute = wrapped_execute  # type: ignore[assignment]


def remove_breakpoint_middleware(agent: WorkflowAgent) -> None:
    """Restore original execute methods on all wrapped executors."""
    for executor in agent.workflow.executors.values():
        original = getattr(executor, _ORIGINAL_EXECUTE_ATTR, None)
        if original is not None:
            executor.execute = original  # type: ignore[assignment]
            delattr(executor, _ORIGINAL_EXECUTE_ATTR)


def create_breakpoint_result(
    exc: AgentInterruptException,
) -> UiPathBreakpointResult:
    """Create a UiPathBreakpointResult from a breakpoint interrupt."""
    node_id = ""
    if isinstance(exc.suspend_value, dict):
        node_id = exc.suspend_value.get("node_id", "")

    return UiPathBreakpointResult(
        breakpoint_node=node_id,
        breakpoint_type="before",
        current_state=exc.suspend_value,
        next_nodes=[node_id] if node_id else [],
    )


__all__ = [
    "create_breakpoint_result",
    "inject_breakpoint_middleware",
    "remove_breakpoint_middleware",
]
