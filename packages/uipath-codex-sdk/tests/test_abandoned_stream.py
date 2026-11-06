"""A stream closed from another context detaches its ``Agent run`` quietly."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from opentelemetry import trace
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import UiPathRuntimeEvent, UiPathStreamOptions
from uipath.runtime.events import UiPathRuntimeMessageEvent

from uipath_codex_sdk.runtime.factory import UiPathCodexResumableRuntime
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage


class _EndlessDelegate:
    """Streams messages until it is closed."""

    async def stream(
        self, input: Any = None, options: Any = None
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        while True:
            yield UiPathRuntimeMessageEvent(payload={"text": "tick"})
            await asyncio.sleep(0)

    async def dispose(self) -> None:
        pass


async def test_stream_closed_from_another_task_does_not_log_an_error(tmp_path, caplog):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = UiPathCodexResumableRuntime(
        delegate=_EndlessDelegate(),  # type: ignore[arg-type]
        storage=storage,
        trigger_manager=UiPathResumeTriggerHandler(),
        runtime_id="rt",
    )
    before = trace.get_current_span()
    caplog.set_level(logging.DEBUG)
    try:
        stream = runtime.stream({}, UiPathStreamOptions())

        async def consume() -> None:
            await stream.__anext__()  # the Agent run context is attached here

        # Each task runs in its own copy of the context: the generator is
        # started in one and abandoned (closed) from another.
        await asyncio.create_task(consume())
        await asyncio.create_task(stream.aclose())
    finally:
        await storage.dispose()

    errors = [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= logging.ERROR
    ]
    assert not errors, [r.getMessage() for r in errors]
    assert any(
        "could not detach the agent run context" in r.getMessage()
        for r in caplog.records
    )
    # This task's context was never touched.
    assert trace.get_current_span() is before
