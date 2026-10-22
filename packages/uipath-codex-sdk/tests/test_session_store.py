"""CodexSessionStore and rollout helpers."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job

from uipath_codex_sdk.runtime.session_store import (
    STATE_STARTED,
    CodexSessionStore,
    DanglingCall,
    EntrypointRecord,
    PendingCall,
    ThreadRecord,
    cut_torn_tail,
    dangling_calls,
    encode_rollout,
    find_rollout,
    find_rollouts,
    has_call,
    parse_rollout,
    write_rollout,
)
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage

THREAD = "01a11c0c-5728-7780-816f-157c0652c4b0"


def _line(payload: dict) -> str:
    return json.dumps({"type": "response_item", "payload": payload}) + "\n"


def _rollout(*payloads: dict) -> bytes:
    head = json.dumps({"type": "session_meta", "payload": {"id": THREAD}}) + "\n"
    return (head + "".join(_line(p) for p in payloads)).encode()


def test_cut_torn_tail():
    assert cut_torn_tail(b'{"a":1}\n{"b":2}\n') == b'{"a":1}\n{"b":2}\n'
    assert cut_torn_tail(b'{"a":1}\n{"b":') == b'{"a":1}\n'
    assert cut_torn_tail(b'{"b":') == b""
    assert cut_torn_tail(b"") == b""


def test_dangling_calls_in_order_with_types():
    data = _rollout(
        {"type": "message", "role": "user", "content": []},
        {"type": "function_call", "call_id": "c1", "name": "a"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "function_call", "call_id": "c2", "name": "durable"},
        {"type": "custom_tool_call", "call_id": "c3", "name": "apply_patch"},
        {"type": "function_call", "call_id": "c4", "name": "exec_command"},
        {"type": "custom_tool_call_output", "call_id": "c3", "output": "done"},
    )
    lines = parse_rollout(data + b'{"torn')
    assert dangling_calls(lines) == [("c2", "function_call"), ("c4", "function_call")]
    assert has_call(lines, "c2") and not has_call(lines, "nope")
    d = DanglingCall("c3", "custom_tool_call")
    assert d.output_type == "custom_tool_call_output"
    assert DanglingCall("c2", "function_call").output_type == "function_call_output"


def test_rollout_encode_and_restore_in_place(tmp_path: Path):
    home = tmp_path / "home"
    rel = f"sessions/2026/10/08/rollout-2026-10-08T10-00-00-{THREAD}.jsonl"
    data = _rollout({"type": "function_call", "call_id": "c1", "name": "x"})
    record = encode_rollout(rel, data)
    assert record.encoding == "zlib+b64" and record.raw_size == len(data)
    assert record.content() == data

    # fresh CODEX_HOME: written back at its original relative path
    target = write_rollout(home, THREAD, record)
    assert target == home / rel and target.read_bytes() == data
    assert find_rollout(home, THREAD) == target

    # an existing copy elsewhere is replaced in place, never duplicated
    home2 = tmp_path / "home2"
    other = (
        home2 / "sessions/2026/10/09" / f"rollout-2026-10-09T00-00-00-{THREAD}.jsonl"
    )
    other.parent.mkdir(parents=True)
    other.write_bytes(b"old\n")
    target2 = write_rollout(home2, THREAD, record)
    assert target2 == other
    assert other.read_bytes() == data
    assert find_rollouts(home2, THREAD) == [other]


def test_zst_rollouts_are_stored_raw():
    record = encode_rollout("sessions/x/rollout-a.jsonl.zst", b"\x28\xb5\x2f\xfd")
    assert record.encoding == "raw+b64"
    assert record.content() == b"\x28\xb5\x2f\xfd"


def _wait_job_raw(key: str = "job-1") -> WaitJobRaw:
    job = Job.model_validate(
        {"Id": 1, "Key": key, "State": "Running", "FolderKey": "fk"}
    )
    return WaitJobRaw(job=job, process_folder_key="fk")


async def test_store_round_trip(tmp_path: Path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    try:
        store = CodexSessionStore(storage, "rt")
        await store.set_thread(ThreadRecord(THREAD, "sessions/a.jsonl"))
        await store.set_entrypoint(EntrypointRecord("agent"))
        await store.set_rollout(encode_rollout("sessions/a.jsonl", b"x\n"))
        value = _wait_job_raw()
        await store.put_pending(
            PendingCall(
                "c1",
                "a_plus_b",
                {"a": 1},
                STATE_STARTED,
                "WaitJobRaw",
                value,
                1,
                span={"span_id": "00f067aa0ba902b7", "attributes": {"a": 1}},
            )
        )
        await store.set_trace({"span_id": "b7ad6b7169203331", "name": "Agent run"})
        await store.set_completed({"c0": "out"})
        await store.set_dangling([DanglingCall("c1", "function_call", durable=True)])

        # a second store (the resumed process) sees everything
        other = CodexSessionStore(storage, "rt")
        assert await other.get_thread() == ThreadRecord(THREAD, "sessions/a.jsonl")
        assert await other.get_entrypoint() == EntrypointRecord("agent")
        rollout = await other.get_rollout()
        assert rollout is not None and rollout.content() == b"x\n"
        pending = await other.get_pending()
        rebuilt = pending["c1"].suspend_value()
        # a typed model again: a dict would become an API trigger
        assert isinstance(rebuilt, WaitJobRaw)
        assert rebuilt.job.key == "job-1" and rebuilt.process_folder_key == "fk"
        assert pending["c1"].span == {
            "span_id": "00f067aa0ba902b7",
            "attributes": {"a": 1},
        }
        saved_trace = await other.get_trace()
        assert saved_trace is not None
        assert saved_trace["span_id"] == "b7ad6b7169203331"
        assert await other.get_completed() == {"c0": "out"}
        assert [d.call_id for d in await other.get_dangling()] == ["c1"]
        # scoped by runtime id
        assert await CodexSessionStore(storage, "other").get_thread() is None

        await other.clear_suspension()
        assert await other.get_pending() == {}
        assert await other.get_completed() == {}
        assert await other.get_dangling() == []
        assert await other.get_thread() is not None
        # the open Agent run outlives a suspension's bookkeeping
        assert await other.get_trace() is not None
        await other.clear_all()
        assert await other.get_thread() is None and await other.get_rollout() is None
        assert await other.get_trace() is None
    finally:
        await storage.dispose()


async def test_concurrent_pending_writes_do_not_lose_calls(tmp_path: Path):
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    try:
        store = CodexSessionStore(storage, "rt")
        await asyncio.gather(
            *(
                store.put_pending(PendingCall(f"c{i}", "t", {}, order=i))
                for i in range(10)
            )
        )
        assert sorted(await store.get_pending()) == sorted(f"c{i}" for i in range(10))
        await store.remove_pending(["c1", "c2"])
        assert "c1" not in await store.get_pending()
    finally:
        await storage.dispose()


def test_find_rollout_prefers_newest(tmp_path: Path):
    a = tmp_path / "sessions/2026/10/01" / f"rollout-x-{THREAD}.jsonl"
    b = tmp_path / "sessions/2026/10/02" / f"rollout-y-{THREAD}.jsonl"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("{}\n")
    old = time.time() - 100
    os.utime(a, (old, old))
    assert find_rollout(tmp_path, THREAD) == b
    assert find_rollout(tmp_path, "missing") is None


def test_leftover_restore_tmp_is_never_the_rollout(tmp_path: Path):
    """A crash between the temp write and the rename leaves ``.restore-tmp``
    behind; it must not be picked as the live rollout, and the next restore
    removes it."""
    live = tmp_path / "sessions/2026/10/01" / f"rollout-x-{THREAD}.jsonl"
    live.parent.mkdir(parents=True)
    live.write_text("{}\n")
    old = time.time() - 100
    os.utime(live, (old, old))
    tmp = live.with_name(live.name + ".restore-tmp")
    tmp.write_text('{"partial": true}\n')  # newer than the live rollout
    zst = live.with_name(f"rollout-z-{THREAD}.jsonl.zst")
    zst.write_bytes(b"")
    os.utime(zst, (old - 10, old - 10))
    assert find_rollouts(tmp_path, THREAD) == sorted([live, zst])
    assert find_rollout(tmp_path, THREAD) == live
    zst.unlink()

    target = write_rollout(
        tmp_path,
        THREAD,
        encode_rollout(f"sessions/2026/10/01/{live.name}", _rollout({"a": 1})),
    )
    assert target == live
    assert not tmp.exists()
    assert parse_rollout(live.read_bytes()) == parse_rollout(_rollout({"a": 1}))
