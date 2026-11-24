"""Contract tests for durable MCP tools via SIGKILL-suspend + inject-resume (design A-prime).

Each test drives the real bundled codex 0.161 app-server against:
  * ``MockResponsesServer`` - scripted OpenAI Responses SSE "model";
  * ``DurableMcpHost``      - in-process stateless streamable-HTTP MCP server
                              that records ``tools/call`` and never answers.

Claims (see module docstrings on each test): C1..C7. Assertions are strict;
evidence (request input items, rollout lines) is printed for every test
(``-rA`` shows it for passes too).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from .durable_mcp_host import DurableMcpHost, HeldCall
from .harness import (
    TRIGGER,
    CodexApp,
    ContractEnv,
    Evidence,
    compact_line,
    dir_listing,
    find_rollouts,
    pid_alive,
    read_rollout,
    response_items,
    restore_rollout,
    rollout_for_thread,
    rollout_function_calls,
    rollout_function_outputs,
    turn_aborted_lines,
    wait_until,
)
from .mock_responses import advertised_tools, input_items, summarize_input

pytestmark = pytest.mark.contract

OUR_OUTPUT = 'UIPATH_JOB_OUTPUT {"status": "Successful", "result": 42}'
ABORT_RE = re.compile(r"aborted", re.IGNORECASE)


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class Suspended:
    app: CodexApp
    host: DurableMcpHost
    home: Path
    cwd: Path
    thread_id: str
    turn: Any
    held: list[HeldCall]
    call_ids: list[str]


def wait_quiet(
    cenv: ContractEnv,
    home: Path,
    thread_id: str,
    quiet: float = 1.0,
    max_wait: float = 15.0,
) -> None:
    """Wait until neither model requests nor the rollout file change for ``quiet`` s."""

    def state():
        ro = rollout_for_thread(home, thread_id)
        return (
            len(cenv.mock.snapshot()),
            ro.stat().st_size if ro and ro.exists() else -1,
        )

    deadline = time.time() + max_wait
    last, since = state(), time.time()
    while time.time() < deadline:
        time.sleep(0.1)
        cur = state()
        if cur != last:
            last, since = cur, time.time()
        elif time.time() - since >= quiet:
            return


def start_and_hold(
    cenv: ContractEnv,
    ev: Evidence,
    call_ids: list[str],
    *,
    parallel: bool = True,
    on_call=None,
    home_name: str = "home",
    cwd_name: str = "cwdA",
    expect_held: int | None = None,
    hold_timeout: float = 30.0,
) -> Suspended:
    cenv.script.call_ids = call_ids
    home, cwd = cenv.dir(home_name), cenv.dir(cwd_name)
    host = cenv.new_host(f"tok-{uuid.uuid4().hex[:8]}", on_call=on_call)
    app = cenv.new_app(home, cwd, host, parallel=parallel)
    ev.add("initialize metadata", app.metadata.model_dump(mode="json"))
    thread = app.thread_start()
    turn = thread.turn(f"{TRIGGER}: start the durable job(s) now")
    n = expect_held if expect_held is not None else len(call_ids)
    held = host.wait_for_held(n, hold_timeout)
    ev.add("advertised durable tool (name, namespace)", cenv.script.advertised)
    ev.add("MCP host JSON-RPC methods seen", host.methods())
    ev.add("MCP held tools/call params", [c.params for c in held])
    return Suspended(app, host, home, cwd, thread.id, turn, held, call_ids)


def rollout_evidence(ev: Evidence, label: str, lines: list[dict]) -> None:
    ev.add(label, "\n".join(compact_line(ln, 420) for ln in lines))


def first_request_with_trigger(cenv: ContractEnv):
    for r in cenv.mock.snapshot():
        if any(TRIGGER in json.dumps(it) for it in input_items(r.body)):
            return r
    return None


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
def test_c1_tools_call_carries_meta_call_id(cenv: ContractEnv, evidence: Evidence):
    """C1: Codex sends params._meta.callId on MCP tools/call, equal to the
    Responses function_call call_id."""
    cid = new_call_id()
    s = start_and_hold(cenv, evidence, [cid])
    assert len(s.held) == 1, (
        f"durable tools/call never reached the host\n{evidence.dump()}"
    )
    call = s.held[0]
    evidence.add("held._meta", call.meta)
    assert call.params.get("name") == "start_job", evidence.dump()
    assert call.params.get("arguments") == {"job": "job-0"}, evidence.dump()
    assert "_meta" in call.params, f"no _meta on tools/call\n{evidence.dump()}"
    assert call.meta.get("callId") == cid, (
        f"_meta.callId={call.meta.get('callId')!r} != Responses call_id {cid!r}\n{evidence.dump()}"
    )
    # and it is the same id Codex persisted for the function_call
    wait_quiet(cenv, s.home, s.thread_id)
    rollout = rollout_for_thread(s.home, s.thread_id)
    assert rollout is not None, evidence.dump()
    lines = read_rollout(rollout)
    assert [fc.get("call_id") for fc in rollout_function_calls(lines)] == [cid], (
        evidence.dump()
    )


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def test_c2_parallel_durable_calls_held_concurrently(
    cenv: ContractEnv, evidence: Evidence
):
    """C2: with supports_parallel_tool_calls=true, two durable calls in one model
    response are both dispatched to the host and held at the same time."""
    ids = [new_call_id(), new_call_id()]
    s = start_and_hold(cenv, evidence, ids, parallel=True, hold_timeout=20.0)
    req = first_request_with_trigger(cenv)
    evidence.add(
        "model request parallel_tool_calls",
        req.body.get("parallel_tool_calls") if req else None,
    )
    evidence.add("host.max_concurrent", s.host.max_concurrent)
    evidence.add(
        "held call ids / received_at / released_at",
        [(c.call_id, c.received_at, c.released_at) for c in s.held],
    )
    assert sorted(str(c.call_id) for c in s.held) == sorted(ids), (
        f"expected both durable calls dispatched, got {[c.call_id for c in s.held]}\n{evidence.dump()}"
    )
    assert all(c.released_at is None for c in s.held), evidence.dump()
    assert s.host.max_concurrent == 2 and s.host.concurrent_now == 2, evidence.dump()


def test_c2_control_serial_without_parallel_flag(cenv: ContractEnv, evidence: Evidence):
    """C2 control: with supports_parallel_tool_calls=false only the first durable
    call is dispatched while it is held (shows the flag is what enables C2)."""
    ids = [new_call_id(), new_call_id()]
    s = start_and_hold(
        cenv, evidence, ids, parallel=False, expect_held=2, hold_timeout=8.0
    )
    evidence.add("held call ids", [c.call_id for c in s.held])
    assert len(s.held) == 1, (
        f"expected serialized dispatch (1 held) without the flag, got {len(s.held)}\n{evidence.dump()}"
    )


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def test_c3_sigkill_leaves_dangling_function_call_in_rollout(
    cenv: ContractEnv, evidence: Evidence
):
    """C3: after SIGKILL while a call is held, the rollout contains the
    function_call and NO output for that call_id, and no turn_aborted marker."""
    cid = new_call_id()
    s = start_and_hold(cenv, evidence, [cid])
    assert len(s.held) == 1, evidence.dump()
    wait_quiet(cenv, s.home, s.thread_id)
    kids = s.app.sigkill()
    evidence.add(
        "app-server pid / alive after kill / children",
        (s.app.pid, pid_alive(s.app.pid), kids),
    )
    assert not pid_alive(s.app.pid)
    wait_until(lambda: s.held[0].disconnected, 5)
    evidence.add(
        "held HTTP request dropped by codex after SIGKILL (disconnected, released_at)",
        (s.held[0].disconnected, s.held[0].released_at),
    )
    ro = rollout_for_thread(s.home, s.thread_id)
    assert ro is not None, f"no rollout for thread\n{dir_listing(s.home)}"
    lines = read_rollout(ro)
    rollout_evidence(evidence, f"rollout {ro.relative_to(s.home)}", lines)
    fcs = rollout_function_calls(lines, cid)
    outs = rollout_function_outputs(lines, cid)
    aborted = turn_aborted_lines(lines)
    any_abort_text = [
        compact_line(ln) for ln in lines if ABORT_RE.search(json.dumps(ln))
    ]
    assert len(fcs) == 1, f"function_call for {cid} not in rollout\n{evidence.dump()}"
    assert outs == [], f"unexpected output(s) for {cid}: {outs}\n{evidence.dump()}"
    assert aborted == [], f"turn_aborted marker present\n{evidence.dump()}"
    assert any_abort_text == [], f"'aborted' text in rollout\n{evidence.dump()}"
    # the dangling function_call is the last model-visible item
    assert response_items(lines)[-1].get("call_id") == cid, evidence.dump()


# ---------------------------------------------------------------------------
# C4 + C5 (shared suspend -> restore -> resume)
# ---------------------------------------------------------------------------
def suspend_and_restore(cenv: ContractEnv, ev: Evidence, cid: str):
    s = start_and_hold(cenv, ev, [cid])
    assert len(s.held) == 1, ev.dump()
    wait_quiet(cenv, s.home, s.thread_id)
    s.app.sigkill()
    ro = rollout_for_thread(s.home, s.thread_id)
    assert ro is not None, ev.dump()
    # fresh CODEX_HOME containing ONLY the restored rollout; original cwd is gone
    home2, cwd2 = cenv.dir("home2"), cenv.dir("cwdB_other_robot")
    restored = restore_rollout(s.home, ro, home2)
    shutil.rmtree(s.cwd)
    ev.add("fresh CODEX_HOME listing before launch", dir_listing(home2))
    assert [p for p in home2.rglob("*") if p.is_file()] == [restored], ev.dump()
    host2 = cenv.new_host(f"tok2-{uuid.uuid4().hex[:8]}")
    app2 = cenv.new_app(home2, cwd2, host2)
    return s, app2, host2, home2, cwd2


def test_c4_resume_in_fresh_home_from_different_cwd(
    cenv: ContractEnv, evidence: Evidence
):
    """C4: a new app-server with a fresh CODEX_HOME holding only the restored
    rollout, launched from a different cwd (original cwd deleted), can
    thread_resume the thread."""
    cid = new_call_id()
    s, app2, host2, home2, cwd2 = suspend_and_restore(cenv, evidence, cid)
    evidence.add("new process cwd", str(cwd2))
    try:
        th2 = app2.thread_resume(s.thread_id, cwd=cwd2)
    except Exception as exc:
        evidence.add("thread_resume error", repr(exc))
        evidence.add("stderr", app2.stderr_tail())
        pytest.fail(f"thread_resume failed: {exc!r}\n{evidence.dump()}")
    assert th2.id == s.thread_id, evidence.dump()
    read = th2.read(include_turns=True).model_dump(mode="json")
    turns = read["thread"].get("turns") or []
    evidence.add(
        "resumed thread turns (status, item types)",
        [(t.get("status"), [i.get("type") for i in t.get("items", [])]) for t in turns],
    )
    evidence.add(
        "resumed thread cwd/path",
        (read["thread"].get("cwd"), read["thread"].get("path")),
    )
    items = [i for t in turns for i in t.get("items", [])]
    assert any(i.get("type") == "mcpToolCall" for i in items) or any(
        TRIGGER in json.dumps(i) for i in items
    ), f"resumed history missing prior turn\n{evidence.dump()}"


def test_c5_inject_output_then_continue_yields_clean_pair(
    cenv: ContractEnv, evidence: Evidence
):
    """C5: thread/inject_items(function_call_output same call_id) + turn/start
    -> next model request has function_call + our output paired by call_id,
    no 'aborted' output and no <turn_aborted>."""
    cid = new_call_id()
    s, app2, host2, home2, cwd2 = suspend_and_restore(cenv, evidence, cid)
    th2 = app2.thread_resume(s.thread_id, cwd=cwd2)
    n_before = len(cenv.mock.snapshot())
    inj = app2.raw(
        "thread/inject_items",
        {
            "threadId": s.thread_id,
            "items": [
                {"type": "function_call_output", "call_id": cid, "output": OUR_OUTPUT}
            ],
        },
    )
    evidence.add("thread/inject_items result", inj)
    evidence.add(
        "model requests made by inject alone", len(cenv.mock.snapshot()) - n_before
    )
    handle = th2.turn("continue")
    reqs = cenv.mock.wait_for_requests(n_before + 1, 30)
    assert len(reqs) > n_before, (
        f"no model request after turn/start\n{app2.stderr_tail()}\n{evidence.dump()}"
    )
    body = reqs[n_before].body
    items = input_items(body)
    evidence.add("post-resume model request input (summary)", summarize_input(body))
    from .mock_responses import find_tool

    evidence.add(
        "post-resume request re-advertises durable tool (name, namespace)",
        find_tool(advertised_tools(body), "start_job"),
    )
    result = handle.run()
    evidence.add(
        "resumed turn status / final", (str(result.status), result.final_response)
    )
    evidence.add("host2 methods (should contain no tools/call)", host2.methods())

    calls = [
        (i, it)
        for i, it in enumerate(items)
        if it.get("type") == "function_call" and it.get("call_id") == cid
    ]
    outs = [
        (i, it)
        for i, it in enumerate(items)
        if it.get("type", "").endswith("_output") and it.get("call_id") == cid
    ]
    assert len(calls) == 1, (
        f"expected exactly one function_call {cid}\n{evidence.dump()}"
    )
    assert len(outs) == 1, (
        f"expected exactly one output for {cid}, got {len(outs)}\n{evidence.dump()}"
    )
    (ci, call), (oi, out) = calls[0], outs[0]
    assert out["type"] == "function_call_output", evidence.dump()
    assert out.get("output") == OUR_OUTPUT, evidence.dump()
    assert oi > ci, f"output precedes call\n{evidence.dump()}"
    assert (
        call.get("name") == "start_job" and call.get("namespace") == "mcp__uipath_tools"
    ), evidence.dump()
    # nothing between the call and its output besides other tool items
    between = [it.get("type") for it in items[ci + 1 : oi]]
    assert between == [], f"items between call and output: {between}\n{evidence.dump()}"
    blob = json.dumps(items)
    assert "<turn_aborted>" not in blob and "turn_aborted" not in blob, evidence.dump()
    assert not any(
        ABORT_RE.search(json.dumps(it.get("output", "")))
        for it in items
        if it.get("type", "").endswith("_output")
    ), evidence.dump()
    assert result.final_response, evidence.dump()
    assert "tools/call" not in host2.methods(), (
        evidence.dump()
    )  # durable call not re-dispatched


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------
RE_ROLLOUT = re.compile(
    r"^sessions/(\d{4})/(\d{2})/(\d{2})/rollout-(\d{4})-(\d{2})-(\d{2})T\d{2}-\d{2}-\d{2}-([0-9a-f-]{36})\.jsonl$"
)


def test_c6_rollout_path_pattern(cenv: ContractEnv, evidence: Evidence):
    """C6a: rollout lives at CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl."""
    cid = new_call_id()
    s = start_and_hold(cenv, evidence, [cid])
    wait_quiet(cenv, s.home, s.thread_id)
    rels = [str(p.relative_to(s.home)) for p in find_rollouts(s.home)]
    evidence.add("rollout files", rels)
    evidence.add("CODEX_HOME top level", sorted(p.name for p in s.home.iterdir()))
    assert len(rels) == 1, evidence.dump()
    m = RE_ROLLOUT.match(rels[0])
    assert m, f"unexpected rollout path {rels[0]}\n{evidence.dump()}"
    assert m.group(7) == s.thread_id, evidence.dump()
    assert m.group(1, 2, 3) == m.group(4, 5, 6), evidence.dump()


@pytest.mark.parametrize("rep", range(3))
def test_c6_function_call_on_disk_when_tools_call_arrives(
    cenv: ContractEnv, evidence: Evidence, rep: int
):
    """C6b: when the host's tools/call handler runs, the function_call for that
    call_id is already persisted in the rollout (snapshot taken inside the handler)."""
    home_holder: dict[str, Path] = {}

    def on_call(params: dict) -> dict:
        home = home_holder["home"]
        cid = (params.get("_meta") or {}).get("callId")
        snap: dict[str, Any] = {"t": time.time(), "callId": cid, "files": []}
        for ro in find_rollouts(home):
            lines = read_rollout(ro)
            snap["files"].append(
                {
                    "path": str(ro.relative_to(home)),
                    "n_lines": len(lines),
                    "fc_on_disk": bool(rollout_function_calls(lines, cid)),
                    "last_line": compact_line(lines[-1], 300) if lines else None,
                }
            )
        return snap

    home_holder["home"] = cenv.dir("home")
    cid = new_call_id()
    s = start_and_hold(cenv, evidence, [cid], on_call=on_call)
    assert len(s.held) == 1, evidence.dump()
    snap = s.held[0].arrival_snapshot
    evidence.add("rollout snapshot taken inside tools/call handler", snap)
    # when (relative to arrival) does it appear, if not already?
    ro_ok = wait_until(
        lambda: bool(
            rollout_function_calls(
                read_rollout(
                    rollout_for_thread(s.home, s.thread_id) or Path("/nonexistent")
                ),
                cid,
            )
        ),
        10,
    )
    evidence.add("function_call on disk within 10s after arrival", ro_ok)
    assert ro_ok, evidence.dump()
    assert snap and snap["files"], (
        f"no rollout file existed at tools/call arrival\n{evidence.dump()}"
    )
    assert any(f["fc_on_disk"] for f in snap["files"]), (
        f"function_call NOT yet flushed when tools/call arrived\n{evidence.dump()}"
    )


# ---------------------------------------------------------------------------
# C7
# ---------------------------------------------------------------------------
def test_c7_turn_interrupt_pollutes_history(cenv: ContractEnv, evidence: Evidence):
    """C7: using turn/interrupt instead of SIGKILL records an 'aborted' output
    and/or <turn_aborted> marker that the model sees on the next request
    (i.e. SIGKILL is necessary for a clean resume)."""
    cid = new_call_id()
    s = start_and_hold(cenv, evidence, [cid])
    assert len(s.held) == 1, evidence.dump()
    wait_quiet(cenv, s.home, s.thread_id)
    evidence.add("turn/interrupt result", s.turn.interrupt().model_dump(mode="json"))
    done = s.turn.run()
    evidence.add("interrupted turn status", str(done.status))
    wait_quiet(cenv, s.home, s.thread_id)
    wait_until(lambda: s.held[0].released_at is not None, 5)
    evidence.add("host methods after interrupt", s.host.methods())
    evidence.add(
        "held call released/disconnected",
        (s.held[0].released_at, s.held[0].disconnected),
    )
    rollout = rollout_for_thread(s.home, s.thread_id)
    assert rollout is not None, evidence.dump()
    lines = read_rollout(rollout)
    rollout_evidence(evidence, "rollout after interrupt", lines[-8:])
    rollout_outs = rollout_function_outputs(lines, cid)
    rollout_aborted = turn_aborted_lines(lines)

    # what the model sees next: inject our output (as on resume) + continue
    n_before = len(cenv.mock.snapshot())
    s.app.raw(
        "thread/inject_items",
        {
            "threadId": s.thread_id,
            "items": [
                {"type": "function_call_output", "call_id": cid, "output": OUR_OUTPUT}
            ],
        },
    )
    t2 = s.app.raw(
        "turn/start",
        {"threadId": s.thread_id, "input": [{"type": "text", "text": "continue"}]},
    )
    evidence.add(
        "turn/start", t2.get("turn", {}).get("id") if isinstance(t2, dict) else t2
    )
    reqs = cenv.mock.wait_for_requests(n_before + 1, 30)
    assert len(reqs) > n_before, evidence.dump()
    items = input_items(reqs[n_before].body)
    evidence.add(
        "next model request input (summary)", summarize_input(reqs[n_before].body)
    )
    outs_for_call = [
        it
        for it in items
        if it.get("type", "").endswith("_output") and it.get("call_id") == cid
    ]
    aborted_outputs = [
        it for it in outs_for_call if ABORT_RE.search(json.dumps(it.get("output", "")))
    ]
    marker = "turn_aborted" in json.dumps(items)
    evidence.add(
        "rollout outputs for call / turn_aborted lines",
        (rollout_outs, [compact_line(x) for x in rollout_aborted]),
    )
    evidence.add(
        "request: outputs for call / aborted outputs / <turn_aborted> marker",
        (outs_for_call, aborted_outputs, marker),
    )
    assert aborted_outputs or marker, (
        f"turn/interrupt left a clean history -> SIGKILL would NOT be necessary\n{evidence.dump()}"
    )
