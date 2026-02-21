"""Breakpoint management for the Agent Framework runtime.

Provides:
- AgentInterruptException: raised by middleware to suspend agent execution
- BreakpointMiddleware: intercepts tools matching breakpoint configuration

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

from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

from agent_framework import AgentExecutor, WorkflowAgent
from agent_framework._middleware import (
    FunctionInvocationContext,
    FunctionMiddleware,
)
from uipath.runtime.debug import UiPathBreakpointResult

from .schema import get_agent_tools, get_tool_name


class AgentInterruptException(Exception):
    """Raised by middleware to suspend agent execution for HITL.

    Carries an interrupt_id and suspend_value that the runtime uses
    to create a UiPathRuntimeResult with SUSPENDED status.
    When is_breakpoint is True, the runtime returns UiPathBreakpointResult
    instead, which bypasses trigger management and is handled by the
    debug runtime layer.
    """

    def __init__(
        self,
        interrupt_id: str,
        suspend_value: Any,
        *,
        is_breakpoint: bool = False,
    ) -> None:
        self.interrupt_id = interrupt_id
        self.suspend_value = suspend_value
        self.is_breakpoint = is_breakpoint
        super().__init__(f"Agent interrupted: {interrupt_id}")


class BreakpointMiddleware(FunctionMiddleware):
    """Intercepts tools matching breakpoint configuration.

    Breakpoint flow (orchestrated by UiPathDebugRuntime):

    1. UiPathDebugRuntime gets breakpoints from debug bridge and passes
       them via ``options.breakpoints`` to the integration runtime.
    2. The integration runtime injects this middleware into the agent's
       middleware chain with the breakpoint list.
    3. When the agent calls a matching tool, this middleware raises
       ``AgentInterruptException(is_breakpoint=True)`` BEFORE the tool runs.
    4. The runtime catches the exception and returns
       ``UiPathBreakpointResult`` (a SUSPENDED result subclass).
    5. ``UiPathResumableRuntime`` passes the breakpoint result through
       (no trigger management — breakpoints bypass the trigger system).
    6. ``UiPathDebugRuntime`` sees ``UiPathBreakpointResult``, notifies
       the debug bridge, and waits for a resume command.
    7. On resume, ``UiPathDebugRuntime`` re-invokes the runtime with
       ``options.resume=True, input=None``. The runtime re-injects this
       middleware with ``skip_tool`` set to the previously-interrupted
       tool name so the first matching call is let through (one-shot).
    8. After the skipped call completes, subsequent breakpoint-matching
       tool calls will pause again.
    """

    def __init__(
        self,
        breakpoints: list[str] | str,
        skip_tool: str | None = None,
    ) -> None:
        self.breakpoints = breakpoints
        self._skip_tool = skip_tool

    def _matches(self, tool_name: str) -> bool:
        if self.breakpoints == "*":
            return True
        if isinstance(self.breakpoints, list):
            return tool_name in self.breakpoints
        return False

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        tool = context.function
        tool_name = getattr(tool, "name", "")

        if not self._matches(tool_name):
            await call_next()
            return

        # One-shot skip for the tool we just resumed from
        if self._skip_tool and tool_name == self._skip_tool:
            self._skip_tool = None
            await call_next()
            return

        # Legacy metadata-based resume (kept for backward compatibility)
        if context.metadata.get("_breakpoint_continue"):
            await call_next()
            return

        interrupt_id = str(uuid4())

        input_value: Any = None
        if context.arguments is not None:
            try:
                if isinstance(context.arguments, Mapping):
                    input_value = dict(context.arguments)
                else:
                    input_value = context.arguments.model_dump()
            except Exception:
                input_value = str(context.arguments)

        suspend_value = {
            "type": "breakpoint",
            "tool_name": tool_name,
            "input_value": input_value,
        }

        raise AgentInterruptException(
            interrupt_id=interrupt_id,
            suspend_value=suspend_value,
            is_breakpoint=True,
        )


_ORIGINAL_EXECUTE_ATTR = "_bp_original_execute"


def _build_executor_tool_map(agent: WorkflowAgent) -> dict[str, set[str]]:
    """Build a mapping of executor_id -> set of tool names."""
    tool_map: dict[str, set[str]] = {}
    for exec_id, executor in agent.workflow.executors.items():
        if isinstance(executor, AgentExecutor):
            inner = getattr(executor, "_agent", None)
            if inner is not None:
                tools = get_agent_tools(inner)
                names = {n for t in tools if (n := get_tool_name(t)) is not None}
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
    skip_nodes: dict[str, int] | None = None,
) -> None:
    """Wrap executor.execute() to pause before breakpointed executors run.

    Replaces each matching executor's execute() with a wrapper that raises
    AgentInterruptException(is_breakpoint=True) before the executor runs.

    For executors in *skip_nodes*, the wrapper allows *N* pass-through calls
    (running the original execute) before re-arming the breakpoint.  The
    count *N* equals the number of times that executor has previously been
    breakpointed and resumed — this correctly handles both:

    * **GroupChat star topology** where the orchestrator is called multiple
      times per workflow run (initial + once per participant response).
    * **Cyclic graphs** where each executor is visited on every cycle.

    Each resume increments the count so the executor passes through all the
    calls that happened *before* the breakpoint, then breaks on the next new
    call.

    Args:
        agent: The workflow agent whose executors to wrap.
        breakpoints: ``"*"`` or a list of node IDs from the debug UI.
        skip_nodes: Mapping of executor_id → pass-through count.
            Each value is the number of calls to let through before
            re-arming the breakpoint on that executor.
    """
    executor_ids = _resolve_to_executor_ids(agent, breakpoints)
    skip = skip_nodes or {}

    for exec_id in executor_ids:
        executor = agent.workflow.executors.get(exec_id)
        if executor is None:
            continue

        # Don't double-wrap
        if hasattr(executor, _ORIGINAL_EXECUTE_ATTR):
            continue

        original = executor.execute
        pass_count = skip.get(exec_id, 0)

        async def wrapped_execute(
            message: Any,
            source_executor_ids: list[str],
            state: Any,
            runner_context: Any,
            trace_contexts: list[dict[str, str]] | None = None,
            source_span_ids: list[str] | None = None,
            *,
            _original: Any = original,
            _exec_id: str = exec_id,
            _remaining: list[int] = [pass_count],  # noqa: B006
        ) -> None:
            if _remaining[0] > 0:
                _remaining[0] -= 1
                return await _original(
                    message,
                    source_executor_ids,
                    state,
                    runner_context,
                    trace_contexts,
                    source_span_ids,
                )
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
    "AgentInterruptException",
    "BreakpointMiddleware",
    "create_breakpoint_result",
    "inject_breakpoint_middleware",
    "remove_breakpoint_middleware",
]
