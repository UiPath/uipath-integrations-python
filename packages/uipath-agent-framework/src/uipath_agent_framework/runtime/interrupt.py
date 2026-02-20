"""Interrupt infrastructure for human-in-the-loop (HITL) support.

Provides:
- AgentInterruptException: raised by middleware to suspend agent execution
- BreakpointMiddleware: intercepts tools matching breakpoint configuration
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from agent_framework._middleware import (
    FunctionInvocationContext,
    FunctionMiddleware,
)


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

        input_value = None
        if context.arguments is not None:
            try:
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


__all__ = [
    "AgentInterruptException",
    "BreakpointMiddleware",
]
