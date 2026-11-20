"""The documented local flow through the real ``uipath`` CLI, offline.

``uipath run agent --file input.json`` suspends (``__uipath/output.json`` lists
a JOB_RAW resume trigger), then ``uipath run agent --resume`` completes once
the job is done. The CLI calls ``asyncio.run`` itself, so these tests are sync
and the mock model server runs on a background loop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from uipath.core.triggers import UiPathResumeTriggerName

from ..contract.mock_responses import input_items
from .fakes import AddScript

pytestmark = pytest.mark.contract


def _uipath(*args: str):
    from uipath._cli import cli

    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def _output(proj: Path) -> dict:
    return json.loads((proj / "__uipath" / "output.json").read_text())


def test_run_then_resume_via_cli(cli_env, world, monkeypatch):
    """Deployed shape: a job key is set, so output.json carries the triggers
    and the runtime id is the job key (the same key is resumed)."""
    monkeypatch.setenv("UIPATH_JOB_KEY", "parent-job-key")
    proj, server = cli_env
    script = AddScript(n_calls=1)
    server.script = script
    cid = script.call_ids[0]

    res = _uipath("run", "agent", "--file", "input.json")
    assert res.exit_code == 0, res.output
    out = _output(proj)
    assert out["status"].lower() == "suspended", out
    triggers = out.get("resumeTriggers") or (
        [out["resume"]] if out.get("resume") else []
    )
    assert triggers, out
    assert triggers[0].get("interruptId") == cid, triggers
    assert triggers[0].get("itemKey") == world.invocations[0].job_key

    # still running: --resume suspends again
    res = _uipath("run", "agent", "--resume")
    assert res.exit_code == 0, res.output
    assert _output(proj)["status"].lower() == "suspended"

    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    shutil.rmtree(proj / "__uipath" / "codex_home")  # only state.db carries over
    res = _uipath("run", "agent", "--resume")
    assert res.exit_code == 0, res.output
    out = _output(proj)
    assert out["status"].lower() == "successful", out
    expected = json.dumps({"sum": 42})
    assert out["output"] == {"output": "FINAL " + json.dumps({cid: expected})}
    last = server.snapshot()[-1].body
    outs = [
        it
        for it in input_items(last)
        if it.get("type") == "function_call_output" and it.get("call_id") == cid
    ]
    assert [o["output"] for o in outs] == [expected]


def _db_triggers(proj: Path) -> list[tuple[str, dict]]:
    import sqlite3

    conn = sqlite3.connect(str(proj / "__uipath" / "state.db"))
    try:
        rows = conn.execute(
            "SELECT interrupt_id, data FROM __uipath_resume_triggers"
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], json.loads(r[1])) for r in rows]


def test_local_run_then_resume_via_cli(cli_env, world, monkeypatch):
    """Local shape (README): no job key, the trigger lives in state.db."""
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    proj, server = cli_env
    script = AddScript(n_calls=1)
    server.script = script
    cid = script.call_ids[0]

    res = _uipath("run", "agent", "--file", "input.json")
    assert res.exit_code == 0, res.output
    [(interrupt_id, data)] = _db_triggers(proj)
    assert interrupt_id == cid
    assert data["trigger_name"] == UiPathResumeTriggerName.JOB_RAW.value  # 'JobRaw'
    assert data["item_key"] == world.invocations[0].job_key

    world.finish(world.invocations[0].job_key, "Successful", {"sum": 42})
    res = _uipath("run", "agent", "--resume")
    assert res.exit_code == 0, res.output
    assert _db_triggers(proj) == []
    assert "FINAL" in res.output
    last = server.snapshot()[-1].body
    outs = [
        it["output"]
        for it in input_items(last)
        if it.get("type") == "function_call_output" and it.get("call_id") == cid
    ]
    assert outs == [json.dumps({"sum": 42})]
