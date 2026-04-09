"""
Breakpoint-related events for workflow execution.
"""

from __future__ import annotations

import functools
from typing import Any, cast

from workflows import Context, Workflow
from workflows.decorators import StepFunction
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
)

from uipath_llamaindex.runtime.schema import get_step_config


class BreakpointEvent(InputRequiredEvent):
    """Event emitted when a breakpoint is hit (before step execution)."""

    breakpoint_node: str = ""

    def __init__(self, breakpoint_node: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.breakpoint_node = breakpoint_node


class BreakpointResumeEvent(HumanResponseEvent):
    """Event sent to resume execution from a breakpoint."""

    pass


def inject_breakpoints(workflow: Workflow) -> None:
    cls = workflow.__class__

    for name, fn in list(vars(cls).items()):
        step_config = get_step_config(name, fn)
        if step_config is None or not callable(fn):
            continue

        wrapped = make_wrapper(name, fn)
        # ensure `self` is bound correctly
        bound = wrapped.__get__(workflow, cls)  # type: ignore[attr-defined]
        setattr(workflow, name, bound)

        # also patch in _step_functions if present
        if name in cls._step_functions:
            cls._step_functions[name] = bound

    for name, fn in list(cls._step_functions.items()):
        step_config = get_step_config(name, fn)
        if step_config is None:
            continue

        wrapped = make_wrapper(name, fn)

        # If it was originally a bound method, bind it again
        if hasattr(fn, "__get__"):
            cls._step_functions[name] = wrapped.__get__(workflow, cls)  # type: ignore[attr-defined]
        else:
            cls._step_functions[name] = wrapped


def make_wrapper(
    step_name: str,
    original: StepFunction[..., Any],
) -> StepFunction[..., Any]:
    """
    Return a wrapped step function that pauses on breakpoints.

    The wrapper creates an InternalContext via ``Context._create_internal``
    to call ``wait_for_event``. This works because the wrapper executes
    inside the step worker where the framework has already set the
    ``StepWorkerStateContextVar``.
    """

    @functools.wraps(original)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        ctx = Context._create_internal(workflow=self)

        bp_event = BreakpointEvent(
            breakpoint_node=step_name,
            prefix=f"Breakpoint at {step_name}",
        )
        # Suspend until debugger resumes
        await ctx.wait_for_event(
            BreakpointResumeEvent,
            waiter_event=bp_event,
            waiter_id=f"bp_{step_name}",
            timeout=None,
        )

        # Continue original step logic
        return await original(self, *args, **kwargs)

    wrapped = cast(StepFunction[..., Any], wrapper)
    wrapped._step_config = original._step_config

    return wrapped


__all__ = [
    "BreakpointEvent",
    "BreakpointResumeEvent",
    "inject_breakpoints",
]
