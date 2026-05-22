"""Tests for the resume-vs-start branch selection in `UiPathLlamaIndexRuntime`.

Covers three scenarios:
  1. First turn with no stored context starts via `start_event`.
  2. Subsequent turn with stored context resumes and sends `HumanResponseEvent`,
     even when `options.resume` is False, provided `is_conversational=True`.
  3. Non-conversational agents keep the existing behavior: without
     `options.resume`, a fresh `start_event` is used even when context exists.
"""

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.runtime import UiPathExecuteOptions
from workflows.events import HumanResponseEvent

from uipath_llamaindex.runtime.runtime import UiPathLlamaIndexRuntime


class _FakeStartEvent:
    """Stand-in for the workflow's start event class."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeHandler:
    """Stand-in for WorkflowHandler: empty event stream, awaitable, mockable ctx."""

    def __init__(self) -> None:
        self.ctx = MagicMock()
        self.ctx.send_event = MagicMock()
        self.cancel_run = AsyncMock()

    def stream_events(self, *, expose_internal: bool = False) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            if False:  # pragma: no cover - typing only
                yield None

        return _gen()

    def __await__(self):
        async def _result():
            return {}

        return _result().__await__()


def _make_handler() -> _FakeHandler:
    return _FakeHandler()


def _make_workflow() -> MagicMock:
    workflow = MagicMock()
    workflow._start_event_class = _FakeStartEvent
    workflow.run = MagicMock(return_value=_make_handler())
    return workflow


def _make_runtime(
    workflow: MagicMock,
    *,
    is_conversational: bool,
    stored_context: dict[str, Any] | None,
) -> UiPathLlamaIndexRuntime:
    storage = MagicMock()
    storage.load_context = AsyncMock(return_value=stored_context)
    storage.save_context = AsyncMock()
    storage.get_value = AsyncMock(return_value=None)
    storage.set_value = AsyncMock()

    return UiPathLlamaIndexRuntime(
        workflow=workflow,
        runtime_id="test-runtime",
        storage=storage,
        is_conversational=is_conversational,
    )


async def _drive(
    runtime: UiPathLlamaIndexRuntime,
    input: dict[str, Any] | None,
    options: UiPathExecuteOptions | None,
) -> None:
    """Consume the workflow event stream to completion."""
    async for _ in runtime._run_workflow(input, options, stream_events=False):
        pass


@pytest.mark.asyncio
async def test_first_turn_no_stored_context_starts_fresh():
    """Conversational agent, turn 1: no prior state -> start_event path."""
    workflow = _make_workflow()
    runtime = _make_runtime(workflow, is_conversational=True, stored_context=None)

    with patch.object(
        UiPathLlamaIndexRuntime, "_load_context", autospec=True
    ) as load_ctx:
        # Simulate the real method: no stored context -> fresh Context, flag stays False.
        async def fake_load(self):
            self._has_prior_state = False
            return MagicMock()

        load_ctx.side_effect = fake_load
        await _drive(runtime, input={"messages": "hi"}, options=None)

    # Workflow was started with a start_event (not resumed).
    assert workflow.run.call_count == 1
    call_kwargs = workflow.run.call_args.kwargs
    assert "start_event" in call_kwargs, (
        "first turn must use start_event, got: %s" % call_kwargs
    )

    # No HumanResponseEvent was injected.
    sent_events = [
        c.args[0] for c in workflow.run.return_value.ctx.send_event.call_args_list
    ]
    assert not any(isinstance(e, HumanResponseEvent) for e in sent_events)


@pytest.mark.asyncio
async def test_subsequent_turn_with_stored_context_auto_resumes():
    """Conversational agent, turn 2: prior state -> resume without options.resume."""
    workflow = _make_workflow()
    runtime = _make_runtime(
        workflow, is_conversational=True, stored_context={"some": "state"}
    )

    with patch.object(
        UiPathLlamaIndexRuntime, "_load_context", autospec=True
    ) as load_ctx:

        async def fake_load(self):
            self._has_prior_state = True
            return MagicMock()

        load_ctx.side_effect = fake_load
        await _drive(
            runtime,
            input={"messages": "follow-up"},
            options=None,  # no explicit resume signal
        )

    # Workflow was resumed with ctx only — no start_event.
    assert workflow.run.call_count == 1
    call_kwargs = workflow.run.call_args.kwargs
    assert "start_event" not in call_kwargs, (
        "conversational resume must omit start_event, got: %s" % call_kwargs
    )
    assert "ctx" in call_kwargs

    # A HumanResponseEvent was injected with the new input.
    sent_events = [
        c.args[0] for c in workflow.run.return_value.ctx.send_event.call_args_list
    ]
    assert any(isinstance(e, HumanResponseEvent) for e in sent_events)


@pytest.mark.asyncio
async def test_non_conversational_with_stored_context_still_starts_fresh():
    """Non-conversational agent: stored context alone does not trigger resume."""
    workflow = _make_workflow()
    runtime = _make_runtime(
        workflow, is_conversational=False, stored_context={"some": "state"}
    )

    with patch.object(
        UiPathLlamaIndexRuntime, "_load_context", autospec=True
    ) as load_ctx:

        async def fake_load(self):
            self._has_prior_state = True
            return MagicMock()

        load_ctx.side_effect = fake_load
        await _drive(runtime, input={"messages": "hi"}, options=None)

    # Without an explicit resume signal and without is_conversational, we start fresh.
    assert workflow.run.call_count == 1
    call_kwargs = workflow.run.call_args.kwargs
    assert "start_event" in call_kwargs, (
        "non-conversational agent without options.resume must use start_event"
    )
