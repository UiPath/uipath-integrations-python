"""Tests for what `uipath debug` needs from this runtime.

The debug command wraps the runtime in ``UiPathDebugRuntime``. These tests
drive that real stack (factory → resumable runtime → bundled codex → mock
model, fake Orchestrator jobs) over a scripted bridge, so the contract is
checked rather than assumed.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

import pytest
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.debug import UiPathBreakpointResult, UiPathDebugRuntime
from uipath.runtime.events import UiPathRuntimeStateEvent

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory

from .contract.mock_responses import function_call_item, input_items, message_item

# The durable suite's fixtures, shared by import (so F401 here, F811 file-wide).
from .durable.conftest import (  # noqa: F401
    _hermetic_env,
    mock_model,
    project,
    upstream,
    world,
)
from .durable.fakes import AddScript, FakeOrchestrator

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
WARNING = "Breakpoints are not supported by the Codex runtime"
RUNTIME_LOGGER = "uipath_codex_sdk.runtime.runtime"

PLAIN_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(model="mock-model", prompt="Say hi to {name}.")
"""


class ShellScript:
    """Mock model: run one shell command, then answer."""

    CALL_ID = "call_shell0000000000000000"

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]]:
        outputs = [
            it
            for it in input_items(body)
            if it.get("type") == "function_call_output"
            and it.get("call_id") == self.CALL_ID
        ]
        if outputs:
            return [message_item("FINAL done", idx)]
        return [
            function_call_item(self.CALL_ID, "exec_command", {"cmd": "echo debug-ran"})
        ]


class ScriptedBridge:
    """A debug bridge that never pauses and records what it is shown.

    ``on_suspended`` runs when the run suspends, so a test can finish the job
    the debug runtime is about to poll.
    """

    def __init__(
        self,
        breakpoints: list[str] | Literal["*"] | None = None,
        on_suspended: Any = None,
    ) -> None:
        self.log: list[str] = []
        self.state_events: list[UiPathRuntimeStateEvent] = []
        self.breakpoints = breakpoints if breakpoints is not None else []
        self.on_suspended = on_suspended

    async def connect(self) -> None:
        self.log.append("connect")

    async def disconnect(self) -> None:
        self.log.append("disconnect")

    async def emit_execution_started(self, **kwargs: Any) -> None:
        self.log.append("started")

    async def emit_state_update(self, state_event: UiPathRuntimeStateEvent) -> None:
        if state_event.node_name != "<polling>":
            self.state_events.append(state_event)

    async def emit_breakpoint_hit(self, breakpoint_result: UiPathBreakpointResult):
        self.log.append("breakpoint")

    async def emit_execution_suspended(self, runtime_result: UiPathRuntimeResult):
        trigger = runtime_result.trigger
        assert trigger is not None
        self.log.append(f"suspended:{trigger.trigger_type.name}")
        if self.on_suspended is not None:
            self.on_suspended(runtime_result)

    async def emit_execution_resumed(self, resume_data: Any) -> None:
        self.log.append("resumed")

    async def emit_execution_completed(self, runtime_result: UiPathRuntimeResult):
        self.log.append(f"completed:{runtime_result.status.name}")

    async def emit_execution_error(self, error: str) -> None:
        self.log.append(f"error:{error}")

    async def wait_for_resume(self) -> Any:
        return None

    async def wait_for_terminate(self) -> None:
        await asyncio.Event().wait()  # never asked to quit

    def get_breakpoints(self) -> list[str] | Literal["*"]:
        return self.breakpoints


async def _debug_run(
    project: Path, bridge: ScriptedBridge, input: dict[str, Any]
) -> list[Any]:
    context = UiPathRuntimeContext(
        runtime_dir=str(project / "__uipath"), state_file="state.db"
    )
    factory = UiPathCodexRuntimeFactory(context)
    runtime = UiPathDebugRuntime(
        delegate=await factory.new_runtime("agent", RUNTIME_ID),
        debug_bridge=bridge,
        trigger_poll_interval=0.05,
    )
    try:
        return [event async for event in runtime.stream(input, UiPathStreamOptions())]
    finally:
        await runtime.dispose()
        await factory.dispose()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if WARNING in r.getMessage()]


@pytest.fixture
def plain_project(project: Path) -> Path:
    (project / "main.py").write_text(PLAIN_MAIN)
    return project


async def test_state_events_reach_the_debug_bridge(plain_project, upstream, caplog):
    upstream.script = ShellScript()
    bridge = ScriptedBridge()
    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
        events = await _debug_run(plain_project, bridge, {"name": "Ada"})

    assert events[-1].status == UiPathRuntimeStatus.SUCCESSFUL
    assert bridge.log == ["connect", "started", "completed:SUCCESSFUL", "disconnect"]
    shell = [
        e
        for e in bridge.state_events
        if (e.metadata or {}).get("item_type") == "commandExecution"
    ]
    assert len(shell) == 1, [e.node_name for e in bridge.state_events]
    assert shell[0].node_name == "tools"
    # Whether the command itself ran depends on the host's sandbox support.
    assert "echo debug-ran" in shell[0].payload["command"]
    assert {e.node_name for e in bridge.state_events} <= {"agent", "tools"}
    assert all(e.execution_id == RUNTIME_ID for e in bridge.state_events)
    assert _warnings(caplog) == []


async def test_breakpoints_are_reported_as_unsupported_once(
    plain_project, upstream, caplog
):
    bridge = ScriptedBridge(breakpoints="*")
    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
        events = await _debug_run(plain_project, bridge, {"name": "Ada"})

    assert events[-1].status == UiPathRuntimeStatus.SUCCESSFUL
    assert "breakpoint" not in bridge.log
    assert len(_warnings(caplog)) == 1


async def test_named_breakpoints_are_reported_too(plain_project, upstream, caplog):
    bridge = ScriptedBridge(breakpoints=["tools"])
    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
        events = await _debug_run(plain_project, bridge, {"name": "Ada"})

    assert events[-1].status == UiPathRuntimeStatus.SUCCESSFUL
    assert len(_warnings(caplog)) == 1


async def test_a_durable_run_suspends_and_resumes_through_the_debug_stack(
    project, upstream, world: FakeOrchestrator
):
    """The loop `uipath debug` runs on a job trigger: suspend, poll, finish."""
    script = AddScript(n_calls=1)
    upstream.script = script

    def finish_job(result: UiPathRuntimeResult) -> None:
        assert result.trigger is not None and result.trigger.item_key is not None
        world.finish(result.trigger.item_key, "Successful", {"sum": 42})

    bridge = ScriptedBridge(on_suspended=finish_job)
    events = await _debug_run(project, bridge, {})

    results = [e for e in events if isinstance(e, UiPathRuntimeResult)]
    assert [r.status for r in results] == [
        UiPathRuntimeStatus.SUSPENDED,
        UiPathRuntimeStatus.SUCCESSFUL,
    ]
    assert bridge.log == [
        "connect",
        "started",
        "suspended:JOB",
        "resumed",
        "completed:SUCCESSFUL",
        "disconnect",
    ]
    resumed = [e for e in bridge.state_events if (e.payload or {}).get("resumed")]
    assert len(resumed) == 1
    assert resumed[0].node_name == "tools"
    assert resumed[0].execution_id == RUNTIME_ID
    assert resumed[0].payload["result"] == {"sum": 42}
    output = results[-1].output
    assert isinstance(output, dict)
    final = output["output"]
    assert final == "FINAL " + json.dumps({script.call_ids[0]: '{"sum": 42}'})
