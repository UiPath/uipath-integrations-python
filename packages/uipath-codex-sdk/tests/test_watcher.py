"""The suspend watcher: when it kills codex, and how it fails.

Driven with fakes (no codex): a host with held calls, a shim, an item tracker
and a codex object whose private process handle records the kill.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

import uipath_codex_sdk.runtime.runtime as runtime_mod
from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.runtime import (
    UiPathCodexRuntime,
    _ItemTracker,
    _SuspendState,
)
from uipath_codex_sdk.runtime.tool_host import SERVER_NAME


@dataclass
class FakeHost:
    held: dict[str, Any] = field(default_factory=dict)
    starting_count: int = 0
    inflight_count: int = 0
    mocking_count: int = 0
    last_activity: float = 0.0
    closing: bool = False

    def close(self) -> None:
        self.closing = True


@dataclass
class FakeShim:
    inflight: int = 0
    last_activity: float = 0.0


class FakeProc:
    pid = -1

    def __init__(self, wait_error: Exception | None = None) -> None:
        self.killed = False
        self.wait_error = wait_error

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_error is not None:
            raise self.wait_error
        return -9


class FakeCodex:
    def __init__(self, proc: FakeProc) -> None:
        self._client = type("C", (), {})()
        self._client._sync = type("S", (), {})()
        self._client._sync._proc = proc
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(runtime_mod, "WATCH_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(runtime_mod, "QUIET_WINDOW_SEC", 0.05)


def _runtime(settle: float = 60.0) -> UiPathCodexRuntime:
    agent = UiPathCodexAgent(
        model="m",
        prompt="p",
        tools=[uipath_process_tool("a_plus_b")],
        durable_settle_timeout=settle,
    )
    return UiPathCodexRuntime(
        agent=agent, runtime_id="rt", entrypoint="agent", file_path="/tmp/main.py"
    )


def _durable_item(call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "mcpToolCall",
        "server": SERVER_NAME,
        "tool": "a_plus_b",
    }


def _immediate_item(call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "mcpToolCall",
        "server": SERVER_NAME,
        "tool": "web_search",
    }


def _tracker() -> _ItemTracker:
    return _ItemTracker(frozenset({"a_plus_b"}))


def _quiet(tracker: _ItemTracker) -> None:
    tracker.last_activity = time.monotonic() - 10


async def _watch(rt, codex, host, tracker, suspend, tmp_path):
    return await rt._watch(
        codex, FakeShim(), host, tracker, suspend, str(tmp_path), "thread"
    )


async def test_waits_for_a_durable_call_that_has_not_arrived(tmp_path):
    """codex started two durable calls; only one reached the host. Killing now
    would let the second start a job the suspension never records."""
    proc = FakeProc()
    host = FakeHost(held={"c1": object()})
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    tracker.on_started(_durable_item("c2"))
    _quiet(tracker)
    suspend = _SuspendState()
    task = asyncio.create_task(
        _watch(_runtime(), FakeCodex(proc), host, tracker, suspend, tmp_path)
    )
    await asyncio.sleep(0.4)
    assert not proc.killed and not task.done() and not host.closing
    host.held = {"c1": object(), "c2": object()}  # the second one arrives
    await asyncio.wait_for(task, 5)
    assert proc.killed and suspend.killed and host.closing


async def test_a_simulated_durable_call_is_not_counted_as_arriving(tmp_path):
    """c2 reached the host and is being answered with a simulated result: it
    is neither held nor starting, but it has arrived."""
    proc = FakeProc()
    host = FakeHost(held={"c1": object()}, mocking_count=1)
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    tracker.on_started(_durable_item("c2"))
    _quiet(tracker)
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(_runtime(), FakeCodex(proc), host, tracker, suspend, tmp_path), 5
    )
    assert proc.killed and suspend.killed and host.closing


async def test_a_call_that_never_arrives_is_bounded_by_the_settle_timeout(tmp_path):
    proc = FakeProc()
    host = FakeHost(held={"c1": object()})
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    tracker.on_started(_durable_item("c2"))
    _quiet(tracker)
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(_runtime(settle=0.3), FakeCodex(proc), host, tracker, suspend, tmp_path),
        5,
    )
    # closed before the kill: a late tools/call is refused, not started
    assert host.closing and proc.killed and suspend.killed


async def test_kill_wait_timeout_still_suspends(tmp_path):
    proc = FakeProc(wait_error=subprocess.TimeoutExpired("codex", 10))
    host = FakeHost(held={"c1": object()})
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    _quiet(tracker)
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(_runtime(), FakeCodex(proc), host, tracker, suspend, tmp_path), 5
    )
    assert proc.killed and suspend.killed and suspend.error is None


async def test_watcher_failure_fails_the_run_and_stops_codex(tmp_path, monkeypatch):
    """An unexpected watcher error must end the stream (by stopping codex) and
    surface as a typed error, not leave the turn blocked for a day."""

    def broken(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runtime_mod, "find_rollout", broken)
    proc = FakeProc()
    host = FakeHost(held={"c1": object()})
    tracker = _tracker()
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(_runtime(), FakeCodex(proc), host, tracker, suspend, tmp_path), 5
    )
    assert proc.killed and host.closing
    assert not suspend.killed
    assert isinstance(suspend.error, UiPathCodexRuntimeError)
    assert "ROLLOUT_CHECKPOINT_FAILED" in str(suspend.error.error_info.code)
    assert "disk on fire" in suspend.error.error_info.detail


async def test_watcher_failure_without_process_handle_closes_codex(
    tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_mod, "find_rollout", broken)
    codex = FakeCodex(FakeProc())
    codex._client._sync._proc = None
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(
            _runtime(),
            codex,
            FakeHost(held={"c1": 1}),
            _tracker(),
            suspend,
            tmp_path,
        ),
        5,
    )
    assert codex.closed and suspend.error is not None


async def test_an_immediate_call_on_the_host_blocks_the_kill(tmp_path):
    """A held durable call plus an immediate call still running on the host
    (it may not have an item yet): no kill until the immediate one is done."""
    proc = FakeProc()
    host = FakeHost(held={"c1": object()}, inflight_count=1)
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    _quiet(tracker)
    suspend = _SuspendState()
    task = asyncio.create_task(
        _watch(_runtime(), FakeCodex(proc), host, tracker, suspend, tmp_path)
    )
    await asyncio.sleep(0.4)
    assert not proc.killed and not task.done() and not host.closing
    host.inflight_count = 0
    await asyncio.wait_for(task, 5)
    assert proc.killed and suspend.killed


async def test_an_immediate_call_is_bounded_by_the_settle_timeout(tmp_path, caplog):
    proc = FakeProc()
    host = FakeHost(held={"c1": object()}, inflight_count=1)
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    _quiet(tracker)
    suspend = _SuspendState()
    await asyncio.wait_for(
        _watch(_runtime(settle=0.3), FakeCodex(proc), host, tracker, suspend, tmp_path),
        5,
    )
    assert proc.killed and suspend.killed and host.closing
    assert "Suspending after" in caplog.text


def test_tracker_counts_hosted_immediate_items_as_non_durable():
    """An immediate uipath_tools call is not a durable one: no phantom
    'durable call has not arrived'."""
    tracker = _tracker()
    tracker.on_started(_durable_item("c1"))
    tracker.on_started(_immediate_item("c2"))
    assert tracker.running_durable() == ["c1"]
    assert tracker.running_non_durable() == ["c2"]
    assert _ItemTracker._is_hosted(_immediate_item("c2"))
    assert not tracker._is_durable(_immediate_item("c2"))
    tracker.on_completed(_immediate_item("c2"))
    assert tracker.running_non_durable() == []
