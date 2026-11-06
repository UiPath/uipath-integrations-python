"""SqliteResumableStorage: langchain-compatible schema and round-trips."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import BaseModel
from uipath.core.triggers import (
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from uipath_codex_sdk.runtime.storage import SqliteResumableStorage


def _job_raw_trigger(interrupt_id: str, job_key: str) -> UiPathResumeTrigger:
    trigger = UiPathResumeTrigger(
        trigger_type=UiPathResumeTriggerType.JOB,
        trigger_name=UiPathResumeTriggerName.JOB_RAW,
        item_key=job_key,
        folder_key="folder-1",
        payload={"job": {"Key": job_key, "Id": 1}, "process_folder_key": "folder-1"},
    )
    trigger.interrupt_id = interrupt_id
    return trigger


@pytest.fixture
async def storage(tmp_path):
    s = SqliteResumableStorage(str(tmp_path / "rt" / "state.db"))
    yield s
    await s.dispose()


async def test_job_raw_name_and_payload_survive_round_trip(storage):
    await storage.save_triggers("r1", [_job_raw_trigger("call_a", "job-a")])
    [loaded] = await storage.get_triggers("r1")
    assert loaded.interrupt_id == "call_a"
    assert loaded.trigger_type == UiPathResumeTriggerType.JOB
    # both are exclude=True on the model; dropping them would turn a JOB_RAW
    # trigger into a plain JOB one (which raises on a faulted job)
    assert loaded.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert loaded.payload == {
        "job": {"Key": "job-a", "Id": 1},
        "process_folder_key": "folder-1",
    }
    assert loaded.item_key == "job-a"
    assert loaded.folder_key == "folder-1"


async def test_delete_by_interrupt_id(storage):
    a, b = _job_raw_trigger("call_a", "job-a"), _job_raw_trigger("call_b", "job-b")
    await storage.save_triggers("r1", [a, b])
    # a re-read copy differs from what was saved (payload/name re-added), and
    # must still delete: matching is by interrupt id, not by serialized JSON
    [loaded_a, _] = await storage.get_triggers("r1")
    await storage.delete_triggers("r1", [loaded_a])
    remaining = await storage.get_triggers("r1")
    assert [t.interrupt_id for t in remaining] == ["call_b"]
    await storage.delete_trigger("r1", remaining[0])
    assert await storage.get_triggers("r1") is None


async def test_save_replaces_and_is_scoped_by_runtime(storage):
    await storage.save_triggers("r1", [_job_raw_trigger("call_a", "job-a")])
    await storage.save_triggers("r2", [_job_raw_trigger("call_x", "job-x")])
    await storage.save_triggers("r1", [_job_raw_trigger("call_b", "job-b")])
    assert [t.interrupt_id for t in await storage.get_triggers("r1")] == ["call_b"]
    assert [t.interrupt_id for t in await storage.get_triggers("r2")] == ["call_x"]


async def test_order_is_insertion_order(storage):
    ids = [f"call_{i}" for i in range(5)]
    await storage.save_triggers("r1", [_job_raw_trigger(i, f"job-{i}") for i in ids])
    assert [t.interrupt_id for t in await storage.get_triggers("r1")] == ids


class _Model(BaseModel):
    x: int


async def test_kv_round_trip(storage):
    await storage.set_value("r1", "ns", "s", "text")
    await storage.set_value("r1", "ns", "d", {"a": [1, 2]})
    await storage.set_value("r1", "ns", "m", _Model(x=3))
    await storage.set_value("r1", "ns", "n", None)
    assert await storage.get_value("r1", "ns", "s") == "text"
    assert await storage.get_value("r1", "ns", "d") == {"a": [1, 2]}
    assert await storage.get_value("r1", "ns", "m") == {"x": 3}
    assert await storage.get_value("r1", "ns", "n") is None
    assert await storage.get_value("r1", "ns", "missing") is None
    assert await storage.get_value("r2", "ns", "s") is None
    await storage.set_value("r1", "ns", "s", "updated")
    assert await storage.get_value("r1", "ns", "s") == "updated"
    with pytest.raises(TypeError):
        await storage.set_value("r1", "ns", "bad", 42)


async def test_schema_matches_langchain(storage, tmp_path):
    await storage.save_triggers("r1", [_job_raw_trigger("call_a", "job-a")])
    await storage.dispose()
    conn = sqlite3.connect(str(tmp_path / "rt" / "state.db"))
    try:
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(__uipath_resume_triggers)")
        ]
        assert cols == ["id", "runtime_id", "interrupt_id", "data", "timestamp"]
        kv = [r[1] for r in conn.execute("PRAGMA table_info(__uipath_runtime_kv)")]
        assert kv == ["runtime_id", "namespace", "key", "value", "timestamp"]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        conn.close()
