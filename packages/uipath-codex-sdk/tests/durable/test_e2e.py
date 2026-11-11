"""Offline end-to-end tests of durable process tools (suspend → resume).

Drives the real factory → ``UiPathResumableRuntime`` → ``UiPathCodexRuntime``
→ bundled codex 0.161 → gateway shim (mock model) + tool host, with fake
Orchestrator jobs. Each resume runs in a fresh factory with ``CODEX_HOME``
deleted, as on another robot: only ``state.db`` carries over.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from uipath.core.triggers import UiPathResumeTriggerName, UiPathResumeTriggerType
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.runtime import CONTINUE_PROMPT, HERMETIC_OVERRIDES

from .._typing import as_result, sqlite_storage
from ..contract.mock_responses import input_items, summarize_input
from .conftest import AGENT_MAIN
from .fakes import FOLDER_KEY, AddScript, ChainScript, SiblingScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"


def _context(project: Path, *, resume: bool) -> UiPathRuntimeContext:
    return UiPathRuntimeContext(
        runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
    )


async def _run(project: Path, input, *, resume: bool):
    factory = UiPathCodexRuntimeFactory(_context(project, resume=resume))
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        result = await runtime.execute(input, UiPathExecuteOptions(resume=resume))
        storage = await sqlite_storage(factory)
        triggers = await storage.get_triggers(RUNTIME_ID)
        return result, triggers
    finally:
        await runtime.dispose()
        await factory.dispose()


def _new_robot(project: Path) -> None:
    """A resumed job gets a fresh runtime dir except for state.db."""
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _resume_request(mock, call_ids: list[str]):
    """First model request that carries an output for every call id."""
    for req in mock.snapshot():
        outs = {
            it.get("call_id")
            for it in input_items(req.body)
            if it.get("type", "").endswith("_output")
        }
        if set(call_ids) <= outs:
            return req
    return None


def _assert_paired(body: dict, call_id: str, expected_output: str) -> None:
    items = input_items(body)
    calls = [
        (i, it)
        for i, it in enumerate(items)
        if it.get("type") == "function_call" and it.get("call_id") == call_id
    ]
    outs = [
        (i, it)
        for i, it in enumerate(items)
        if it.get("type", "").endswith("_output") and it.get("call_id") == call_id
    ]
    summary = summarize_input(body)
    assert len(calls) == 1, summary
    assert len(outs) == 1, summary
    (ci, call), (oi, out) = calls[0], outs[0]
    assert call.get("name") == "a_plus_b", summary
    assert call.get("namespace") == "mcp__uipath_tools", summary
    assert out["type"] == "function_call_output", summary
    assert out["output"] == expected_output, summary
    assert oi > ci, summary
    blob = json.dumps(items)
    assert "turn_aborted" not in blob, summary
    assert "aborted" not in json.dumps([o for _, o in outs]), summary


async def test_suspend_then_resume_successful(project, upstream, world):
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]

    # --- run: suspends on the job --------------------------------------
    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert script.advertised == ("a_plus_b", "mcp__uipath_tools")
    assert len(world.invocations) == 1
    inv = world.invocations[0]
    assert inv.name == "a_plus_b"
    assert inv.input_arguments == {"a": 17, "b": 25}
    assert inv.folder_path == "Shared/Durable"
    # The printed/output.json output is a slim summary; the trigger holds the job.
    assert isinstance(result.output, dict) and list(result.output) == [cid]
    assert result.output[cid] == {
        "tool": "a_plus_b",
        "arguments": {"a": 17, "b": 25},
        "job_key": inv.job_key,
        "job_state": result.output[cid]["job_state"],
        "folder_key": FOLDER_KEY,
        "type": "WaitJobRaw",
    }
    assert triggers is not None and len(triggers) == 1
    trig = triggers[0]
    assert trig.interrupt_id == cid
    assert trig.trigger_type == UiPathResumeTriggerType.JOB
    assert trig.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert trig.item_key == inv.job_key
    assert trig.folder_key == FOLDER_KEY
    n_requests = len(upstream.snapshot())

    # --- the job is still running: --resume re-suspends -----------------
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert triggers is not None and [t.interrupt_id for t in triggers] == [cid]
    assert len(upstream.snapshot()) == n_requests  # no codex run, no model call

    # --- the job finished: resume completes the run ---------------------
    world.finish(inv.job_key, "Successful", {"sum": 42})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    expected = json.dumps({"sum": 42})
    req = _resume_request(upstream, [cid])
    assert req is not None, [summarize_input(r.body) for r in upstream.snapshot()]
    _assert_paired(req.body, cid, expected)
    assert CONTINUE_PROMPT in json.dumps(input_items(req.body))
    assert result.output == {"output": "FINAL " + json.dumps({cid: expected})}
    assert len(world.invocations) == 1  # the job was never started again


async def test_faulted_job_becomes_error_result(project, upstream, world):
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]

    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    world.finish(world.invocations[0].job_key, "Faulted", info="Division by zero")

    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    req = _resume_request(upstream, [cid])
    assert req is not None
    out = next(
        it
        for it in input_items(req.body)
        if it.get("type") == "function_call_output" and it.get("call_id") == cid
    )
    payload = json.loads(out["output"])
    assert payload["status"] == "faulted"
    assert payload["error"] == "Division by zero"
    assert payload["job_key"] == world.invocations[0].job_key


async def test_two_parallel_calls(project, upstream, world):
    script = AddScript(n_calls=2)
    upstream.script = script

    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert len(world.invocations) == 2
    assert sorted(result.output) == sorted(script.call_ids)
    assert triggers is not None
    assert sorted(t.interrupt_id for t in triggers) == sorted(script.call_ids)
    by_args = {inv.input_arguments["a"]: inv for inv in world.invocations}

    # only the first job is done: resume finishes it and waits for the other
    first = by_args[17]
    world.finish(first.job_key, "Successful", {"sum": 42})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert triggers is not None and len(triggers) == 1

    second = by_args[18]
    world.finish(second.job_key, "Successful", {"sum": 43})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    req = _resume_request(upstream, script.call_ids)
    assert req is not None, [summarize_input(r.body) for r in upstream.snapshot()]
    cid_of = {
        json.loads(it["arguments"])["a"]: it["call_id"]
        for it in input_items(req.body)
        if it.get("type") == "function_call"
    }
    _assert_paired(req.body, cid_of[17], json.dumps({"sum": 42}))
    _assert_paired(req.body, cid_of[18], json.dumps({"sum": 43}))


async def test_fast_path_completes_in_process(project, upstream, world, monkeypatch):
    """A job that is already done when the run suspends is resumed in the same
    process by ``UiPathResumableRuntime`` (re-entrant runtime, same CODEX_HOME)."""
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]

    original = world.client

    def finishing_client():
        for inv in world.invocations:
            if world.jobs[inv.job_key].state == "Pending":
                world.finish(inv.job_key, "Successful", {"sum": 42})
        return original()

    import uipath.platform.resume_triggers._protocol as protocol

    monkeypatch.setattr(protocol, "UiPath", finishing_client)
    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    req = _resume_request(upstream, [cid])
    assert req is not None
    _assert_paired(req.body, cid, json.dumps({"sum": 42}))


async def test_explicit_resume_payload(project, upstream, world):
    """``uipath run agent '{"sum": 7}' --resume`` answers the single call."""
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]

    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    _new_robot(project)
    result, triggers = await _run(project, {"sum": 7}, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    req = _resume_request(upstream, [cid])
    assert req is not None
    _assert_paired(req.body, cid, json.dumps({"sum": 7}))


@pytest.mark.parametrize("sibling_after", [True, False], ids=["after", "before"])
async def test_shell_sibling_is_paired_on_resume(
    project, upstream, world, sibling_after
):
    """A non-durable sibling in the same response as a held call.

    After the held call its output never reaches the rollout (outputs are
    recorded in call order), so it is taken from the item/completed
    notification and injected; before it, codex already persisted it.
    Either way every call is answered exactly once and nothing is "aborted".
    """
    script = SiblingScript(sibling_after=sibling_after)
    upstream.script = script
    cid, sib = script.call_ids[0], SiblingScript.SIBLING_ID

    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    world.finish(world.invocations[0].job_key, "Successful", {"sum": 3})
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result

    req = _resume_request(upstream, [cid, sib])
    assert req is not None, [summarize_input(r.body) for r in upstream.snapshot()]
    _assert_paired(req.body, cid, json.dumps({"sum": 3}))
    items = input_items(req.body)
    sib_outs = [
        it
        for it in items
        if it.get("type") == "function_call_output" and it.get("call_id") == sib
    ]
    assert len(sib_outs) == 1, summarize_input(req.body)
    assert "aborted" not in sib_outs[0]["output"]
    assert "Not executed" not in sib_outs[0]["output"]
    order = [
        it.get("call_id") for it in items if it.get("type") == "function_call_output"
    ]
    assert order == ([cid, sib] if sibling_after else [sib, cid])


async def test_suspends_again_after_resume(project, upstream, world):
    """The resumed turn calls the durable tool again: a second suspension."""
    script = ChainScript()
    upstream.script = script
    first, second = script.call_ids

    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert list(result.output) == [first]
    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})

    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert list(result.output) == [second]
    assert triggers is not None and [t.interrupt_id for t in triggers] == [second]
    assert len(world.invocations) == 2
    world.finish(world.invocations[1].job_key, "Successful", {"sum": 43})

    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    req = _resume_request(upstream, [first, second])
    assert req is not None
    _assert_paired(req.body, first, json.dumps({"sum": 42}))
    _assert_paired(req.body, second, json.dumps({"sum": 43}))
    assert result.output == {
        "output": "FINAL "
        + json.dumps(
            {first: json.dumps({"sum": 42}), second: json.dumps({"sum": 43})},
            sort_keys=True,
        )
    }


PLAIN_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(model="mock-model", prompt="Say hi to {name}.")
"""


async def test_agent_without_durable_tools_is_unchanged(
    project, upstream, world, monkeypatch, tmp_path
):
    """No durable tools: codex keeps its own CODEX_HOME and gets no hermetic
    overrides, exactly as before durable tools existed."""
    import openai_codex

    user_home = tmp_path / "user_codex_home"
    user_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(user_home))
    configs: list = []
    real_config = openai_codex.CodexConfig

    def spy_config(*args, **kwargs):
        configs.append(kwargs)
        return real_config(*args, **kwargs)

    monkeypatch.setattr(openai_codex, "CodexConfig", spy_config)
    (project / "main.py").write_text(PLAIN_MAIN)
    upstream.script = AddScript()
    result, triggers = await _run(project, {"name": "Ada"}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert result.output == {"output": "FINAL {}"}
    assert not triggers
    reqs = upstream.snapshot()
    assert len(reqs) == 1
    body = reqs[0].body
    assert "Say hi to Ada." in json.dumps(input_items(body))
    assert "long-running" not in json.dumps(input_items(body))
    assert all(t.get("name") != "mcp__uipath_tools" for t in body.get("tools") or [])
    assert not world.invocations
    # codex resolved CODEX_HOME itself: the user's, not a run-scoped one
    [config] = configs
    assert "CODEX_HOME" not in config["env"]
    assert not set(HERMETIC_OVERRIDES) & set(config["config_overrides"])
    assert list(user_home.glob("sessions/**/rollout-*.jsonl"))
    assert not (project / "__uipath" / "codex_home").exists()


async def test_missing_process_handle_fails_instead_of_hanging(
    project, upstream, world, monkeypatch
):
    """If the SDK's private process handle disappears (an SDK upgrade), the
    run fails with SUSPEND_KILL_UNAVAILABLE rather than hanging or writing an
    'aborted' history."""
    from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
    from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime

    real = UiPathCodexRuntime._kill_app_server

    def no_handle(codex, suspend):
        return real(object(), suspend)

    monkeypatch.setattr(UiPathCodexRuntime, "_kill_app_server", staticmethod(no_handle))
    upstream.script = AddScript(n_calls=1)
    with pytest.raises(UiPathCodexRuntimeError) as exc:
        await asyncio.wait_for(_run(project, {}, resume=False), 60)
    assert "SUSPEND_KILL_UNAVAILABLE" in str(exc.value.error_info.code)


async def test_code_mode_only_model_still_calls_durable_tools_directly(
    project, upstream, world
):
    """Regression: codex's bundled catalog marks e.g. gpt-5.6-terra as
    ``tool_mode: code_mode_only`` and defers MCP tools behind tool_search, so
    the model reached durable tools only from inside an ``exec`` JavaScript
    cell, as nested calls that never reach the rollout and cannot suspend.
    """
    (project / "main.py").write_text(
        AGENT_MAIN.replace('model="mock-model"', 'model="gpt-5.6-terra"')
    )
    script = AddScript(n_calls=1)
    upstream.script = script

    result, triggers = await _run(project, {}, resume=False)

    first = upstream.snapshot()[0].body
    # code mode stays on for codex's own tools; ours are offered directly
    # (omit_tools_from=["deferred", "code_mode"]) and called as a top-level
    # function_call, so the call can be held and suspended.
    assert "ALL_TOOLS" in json.dumps(first), "expected a code_mode_only model"
    assert script.advertised == ("a_plus_b", "mcp__uipath_tools")
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert triggers is not None and len(triggers) == 1
    assert len(world.invocations) == 1


async def test_resume_streams_the_durable_tool_result(project, upstream, world):
    """On resume the finished durable call shows up as a ``tool`` state event
    with its arguments and the job result, before the model continues."""
    from uipath.runtime import UiPathStreamOptions
    from uipath.runtime.events import UiPathRuntimeStateEvent

    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]
    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    inv = world.invocations[0]
    world.finish(inv.job_key, "Successful", {"sum": 42})
    _new_robot(project)

    factory = UiPathCodexRuntimeFactory(_context(project, resume=True))
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    events = []
    try:
        async for ev in runtime.stream(None, UiPathStreamOptions(resume=True)):
            events.append(ev)
    finally:
        await runtime.dispose()
        await factory.dispose()

    resumed = [
        e
        for e in events
        if isinstance(e, UiPathRuntimeStateEvent)
        and e.node_name == "tools"
        and (e.payload or {}).get("resumed")
    ]
    assert len(resumed) == 1, [getattr(e, "payload", e) for e in events]
    payload = resumed[0].payload
    assert payload is not None
    assert payload["tool"] == "a_plus_b"
    assert payload["arguments"] == {"a": 17, "b": 25}
    assert payload["result"] == {"sum": 42}
    assert payload["status"] == "completed"
    assert payload["job_key"] == inv.job_key
    metadata = resumed[0].metadata
    assert metadata is not None
    assert metadata["call_id"] == cid
    assert as_result(events[-1]).status == UiPathRuntimeStatus.SUCCESSFUL
