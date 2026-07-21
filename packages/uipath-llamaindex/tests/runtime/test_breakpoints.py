"""Tests for breakpoint injection wrappers."""

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from workflows import Context
from workflows.decorators import StepFunction
from workflows.events import HumanResponseEvent

from uipath_llamaindex.runtime.breakpoints import (
    BreakpointEvent,
    BreakpointResumeEvent,
    make_wrapper,
)


def test_breakpoint_event_stores_node_name():
    event = BreakpointEvent(breakpoint_node="my_step")
    assert event.breakpoint_node == "my_step"


@pytest.mark.asyncio
async def test_wrapper_suspends_then_runs_original_step(monkeypatch):
    calls = {}

    async def original(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["ran"] = True
        return "step-result"

    # make_wrapper copies the original step config onto the wrapper.
    step_config = object()
    original._step_config = step_config  # type: ignore[attr-defined]

    wrapped = make_wrapper("my_step", cast(StepFunction[..., Any], original))
    assert wrapped._step_config is step_config

    fake_ctx = AsyncMock()
    monkeypatch.setattr(Context, "get_step_context", lambda: fake_ctx)
    result = await wrapped(object())

    assert result == "step-result"
    assert calls["ran"] is True
    # It waits for the debugger's resume event before running the step.
    fake_ctx.wait_for_event.assert_awaited_once()
    awaited_kwargs = fake_ctx.wait_for_event.await_args.kwargs
    assert awaited_kwargs["waiter_id"] == "bp_my_step"
    assert isinstance(awaited_kwargs["waiter_event"], BreakpointEvent)


def test_resume_event_is_distinct_type():
    assert issubclass(BreakpointResumeEvent, HumanResponseEvent)
