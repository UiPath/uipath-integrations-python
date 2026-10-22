"""``--resume`` with an input that answers no pending call.

``uipath run agent --file input.json --resume`` passes the agent input again.
``UiPathResumableRuntime`` takes any non-``None`` input as the resume input and
deletes the (single) trigger, so the runtime must read the pending jobs itself
instead of using the input as a job result or waiting another resume cycle.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime

from .._typing import sqlite_storage
from ..contract.mock_responses import input_items, summarize_input
from .fakes import AddScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
AGENT_INPUT = {"topic": "sums"}


async def _run(project: Path, input, *, resume: bool):
    context = UiPathRuntimeContext(
        runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
    )
    factory = UiPathCodexRuntimeFactory(context)
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        result = await runtime.execute(input, UiPathExecuteOptions(resume=resume))
        triggers = await (await sqlite_storage(factory)).get_triggers(RUNTIME_ID)
        return result, triggers
    finally:
        await runtime.dispose()
        await factory.dispose()


def _new_robot(project: Path) -> None:
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _outputs_by_a(upstream, call_ids: list[str]) -> dict[int, str]:
    """``{argument a: injected output}`` of the request answering all calls."""
    for req in upstream.snapshot():
        items = input_items(req.body)
        outs = {
            it["call_id"]: it["output"]
            for it in items
            if it.get("type") == "function_call_output"
        }
        if set(call_ids) <= set(outs):
            assert "turn_aborted" not in json.dumps(items), summarize_input(req.body)
            return {
                json.loads(it["arguments"])["a"]: outs[it["call_id"]]
                for it in items
                if it.get("type") == "function_call"
            }
    raise AssertionError([summarize_input(r.body) for r in upstream.snapshot()])


@pytest.fixture
def prepares(monkeypatch) -> list[object]:
    """Inputs ``_prepare_resume`` was called with (one per resume cycle)."""
    seen: list[object] = []
    original = UiPathCodexRuntime._prepare_resume

    async def counting(self, input):
        seen.append(input)
        return await original(self, input)

    monkeypatch.setattr(UiPathCodexRuntime, "_prepare_resume", counting)
    return seen


async def _suspend(project: Path, upstream, n_calls: int) -> AddScript:
    script = AddScript(n_calls=n_calls)
    upstream.script = script
    result, triggers = await _run(project, AGENT_INPUT, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert triggers is not None and len(triggers) == n_calls
    return script


@pytest.mark.parametrize("n_calls", [1, 2], ids=["one-call", "two-calls"])
async def test_resume_with_agent_input_reads_the_jobs(
    project, upstream, world, prepares, caplog, n_calls
):
    script = await _suspend(project, upstream, n_calls)
    for i, inv in enumerate(world.invocations):
        world.finish(inv.job_key, "Successful", {"sum": 42 + i})

    _new_robot(project)
    caplog.set_level(logging.INFO, logger="uipath_codex_sdk")
    result, triggers = await _run(project, dict(AGENT_INPUT), resume=True)

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    assert prepares == [AGENT_INPUT]  # one resume cycle, no re-suspension
    assert "still waiting" not in caplog.text
    assert _outputs_by_a(upstream, script.call_ids) == {
        17 + i: json.dumps({"sum": 42 + i}) for i in range(n_calls)
    }
    assert len(world.invocations) == n_calls


async def test_resume_with_agent_input_while_job_runs(
    project, upstream, world, prepares
):
    """The input is not taken as the job result; the run keeps waiting on a
    fresh trigger for the job (the wrapper deleted the old one)."""
    script = await _suspend(project, upstream, 1)
    _new_robot(project)
    result, triggers = await _run(project, dict(AGENT_INPUT), resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert triggers is not None
    assert [t.interrupt_id for t in triggers] == script.call_ids
    assert triggers[0].item_key == world.invocations[0].job_key
    assert prepares == [AGENT_INPUT]

    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert _outputs_by_a(upstream, script.call_ids) == {17: json.dumps({"sum": 42})}


async def test_partial_finish_with_agent_input_drops_used_trigger(
    project, upstream, world, prepares
):
    """Two calls, one job done: the finished call's stored trigger is removed,
    so a later resume cannot hand its job to the call still waiting."""
    script = await _suspend(project, upstream, 2)
    by_a = {inv.input_arguments["a"]: inv for inv in world.invocations}
    world.finish(by_a[17].job_key, "Successful", {"sum": 42})

    _new_robot(project)
    result, triggers = await _run(project, dict(AGENT_INPUT), resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert triggers is not None and len(triggers) == 1
    assert triggers[0].item_key == by_a[18].job_key

    world.finish(by_a[18].job_key, "Successful", {"sum": 43})
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert _outputs_by_a(upstream, script.call_ids) == {
        17: json.dumps({"sum": 42}),
        18: json.dumps({"sum": 43}),
    }
