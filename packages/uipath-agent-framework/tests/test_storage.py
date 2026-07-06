"""Tests for SQLite checkpoint storage."""

import os
import tempfile

from agent_framework import WorkflowCheckpoint
from uipath.runtime import (
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from uipath_agent_framework.runtime.resumable_storage import (
    ScopedCheckpointStorage,
    SqliteResumableStorage,
)


def _make_checkpoint(
    workflow_name: str = "test_workflow",
    checkpoint_id: str | None = None,
    timestamp: str | None = None,
) -> WorkflowCheckpoint:
    """Create a WorkflowCheckpoint for testing."""
    return WorkflowCheckpoint(
        workflow_name=workflow_name,
        graph_signature_hash="abc123",
        checkpoint_id=checkpoint_id or f"cp-{id(object())}",
        timestamp=timestamp or "2026-01-01T00:00:00+00:00",
        state={"key": "value"},
        messages={},
        pending_request_info_events={},
        iteration_count=1,
        metadata={"test": True},
    )


class TestSqliteCheckpointStorage:
    """Tests for SqliteCheckpointStorage via SqliteResumableStorage."""

    async def test_setup_creates_db_file(self):
        """Setup creates the SQLite database file with checkpoints table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()

            assert os.path.exists(db_path)
            assert storage.checkpoint_storage is not None
            await storage.dispose()

    async def test_setup_creates_nested_directories(self):
        """Setup creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "dir", "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()

            assert os.path.exists(db_path)
            await storage.dispose()

    async def test_save_and_load_checkpoint(self):
        """Saved checkpoint can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            checkpoint = _make_checkpoint(checkpoint_id="cp-1")
            await cs.save(checkpoint)
            loaded = await cs.load("cp-1")

            assert loaded.checkpoint_id == "cp-1"
            assert loaded.workflow_name == "test_workflow"
            assert loaded.graph_signature_hash == "abc123"
            assert loaded.state == {"key": "value"}
            assert loaded.metadata == {"test": True}
            await storage.dispose()

    async def test_load_nonexistent_raises(self):
        """Loading a non-existent checkpoint raises an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            try:
                await cs.load("nonexistent")
                raise AssertionError("Should have raised")
            except Exception:
                pass
            await storage.dispose()

    async def test_get_latest_returns_most_recent(self):
        """get_latest returns the checkpoint with the most recent timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp1 = _make_checkpoint(
                checkpoint_id="cp-old", timestamp="2026-01-01T00:00:00+00:00"
            )
            cp2 = _make_checkpoint(
                checkpoint_id="cp-new", timestamp="2026-01-02T00:00:00+00:00"
            )
            await cs.save(cp1)
            await cs.save(cp2)

            latest = await cs.get_latest(workflow_name="test_workflow")
            assert latest is not None
            assert latest.checkpoint_id == "cp-new"
            await storage.dispose()

    async def test_get_latest_returns_none_for_empty(self):
        """get_latest returns None when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            latest = await cs.get_latest(workflow_name="nonexistent")
            assert latest is None
            await storage.dispose()

    async def test_delete_checkpoint(self):
        """Deleted checkpoint is no longer loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp = _make_checkpoint(checkpoint_id="cp-del")
            await cs.save(cp)

            result = await cs.delete("cp-del")
            assert result is True

            # Verify it's gone
            result = await cs.delete("cp-del")
            assert result is False
            await storage.dispose()

    async def test_list_checkpoints_filtered_by_workflow_name(self):
        """list_checkpoints only returns checkpoints for the given workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp1 = _make_checkpoint(workflow_name="wf_a", checkpoint_id="cp-a1")
            cp2 = _make_checkpoint(workflow_name="wf_a", checkpoint_id="cp-a2")
            cp3 = _make_checkpoint(workflow_name="wf_b", checkpoint_id="cp-b1")
            await cs.save(cp1)
            await cs.save(cp2)
            await cs.save(cp3)

            wf_a_cps = await cs.list_checkpoints(workflow_name="wf_a")
            assert len(wf_a_cps) == 2
            assert {cp.checkpoint_id for cp in wf_a_cps} == {"cp-a1", "cp-a2"}

            wf_b_cps = await cs.list_checkpoints(workflow_name="wf_b")
            assert len(wf_b_cps) == 1
            assert wf_b_cps[0].checkpoint_id == "cp-b1"
            await storage.dispose()

    async def test_list_checkpoint_ids(self):
        """list_checkpoint_ids returns IDs filtered by workflow_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp1 = _make_checkpoint(workflow_name="wf_x", checkpoint_id="cp-x1")
            cp2 = _make_checkpoint(workflow_name="wf_x", checkpoint_id="cp-x2")
            cp3 = _make_checkpoint(workflow_name="wf_y", checkpoint_id="cp-y1")
            await cs.save(cp1)
            await cs.save(cp2)
            await cs.save(cp3)

            ids = await cs.list_checkpoint_ids(workflow_name="wf_x")
            assert set(ids) == {"cp-x1", "cp-x2"}
            await storage.dispose()

    async def test_save_overwrites_existing_checkpoint(self):
        """Saving with the same checkpoint_id overwrites the previous data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp1 = _make_checkpoint(checkpoint_id="cp-ow")
            cp1.state = {"version": 1}
            await cs.save(cp1)

            cp2 = _make_checkpoint(checkpoint_id="cp-ow")
            cp2.state = {"version": 2}
            await cs.save(cp2)

            loaded = await cs.load("cp-ow")
            assert loaded.state == {"version": 2}
            await storage.dispose()

    async def test_dispose_allows_reconnect(self):
        """After dispose, the store can be set up again and data persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            cp = _make_checkpoint(checkpoint_id="cp-persist")
            await cs.save(cp)
            await storage.dispose()

            # Reconnect to the same DB
            storage2 = SqliteResumableStorage(db_path)
            await storage2.setup()
            cs2 = storage2.checkpoint_storage
            assert cs2 is not None

            loaded = await cs2.load("cp-persist")
            assert loaded.checkpoint_id == "cp-persist"
            await storage2.dispose()

    async def test_delete_triggers(self):
        """delete_triggers removes sibling triggers and preserves runtime isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()

            api_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.API,
                trigger_name=UiPathResumeTriggerName.API,
                item_key="first",
                interrupt_id="interrupt-1",
            )
            timer_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.TIMER,
                trigger_name=UiPathResumeTriggerName.TIMER,
                item_key="third",
                interrupt_id="interrupt-1",
            )
            unrelated_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.TASK,
                trigger_name=UiPathResumeTriggerName.TASK,
                item_key="second",
                interrupt_id="interrupt-2",
            )
            other_runtime_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.API,
                trigger_name=UiPathResumeTriggerName.API,
                item_key="other-runtime",
                interrupt_id="interrupt-1",
            )

            await storage.save_triggers(
                "runtime-a", [api_trigger, timer_trigger, unrelated_trigger]
            )
            await storage.save_triggers("runtime-b", [other_runtime_trigger])

            await storage.delete_triggers("runtime-a", [api_trigger, timer_trigger])

            runtime_a_triggers = await storage.get_triggers("runtime-a")
            assert runtime_a_triggers is not None
            assert [trigger.interrupt_id for trigger in runtime_a_triggers] == [
                "interrupt-2"
            ]
            assert runtime_a_triggers[0].item_key == "second"

            runtime_b_triggers = await storage.get_triggers("runtime-b")
            assert runtime_b_triggers is not None
            assert len(runtime_b_triggers) == 1
            assert runtime_b_triggers[0].item_key == "other-runtime"
            await storage.dispose()

    async def test_delete_trigger_deletes_single_trigger(self):
        """delete_trigger removes one trigger through the compatibility wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()

            trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.API,
                trigger_name=UiPathResumeTriggerName.API,
                item_key="first",
                interrupt_id="interrupt-1",
            )
            unrelated_trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.TASK,
                trigger_name=UiPathResumeTriggerName.TASK,
                item_key="second",
                interrupt_id="interrupt-2",
            )

            await storage.save_triggers("runtime-a", [trigger, unrelated_trigger])

            await storage.delete_trigger("runtime-a", trigger)

            runtime_a_triggers = await storage.get_triggers("runtime-a")
            assert runtime_a_triggers is not None
            assert [trigger.interrupt_id for trigger in runtime_a_triggers] == [
                "interrupt-2"
            ]
            await storage.dispose()

    async def test_delete_triggers_empty_list_is_noop(self):
        """delete_triggers with an empty list leaves stored triggers unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()

            trigger = UiPathResumeTrigger(
                trigger_type=UiPathResumeTriggerType.API,
                trigger_name=UiPathResumeTriggerName.API,
                item_key="first",
                interrupt_id="interrupt-1",
            )
            await storage.save_triggers("runtime-a", [trigger])

            await storage.delete_triggers("runtime-a", [])

            runtime_a_triggers = await storage.get_triggers("runtime-a")
            assert runtime_a_triggers is not None
            assert len(runtime_a_triggers) == 1
            assert runtime_a_triggers[0].item_key == "first"
            await storage.dispose()


class TestScopedCheckpointStorage:
    """Tests for ScopedCheckpointStorage prefix isolation."""

    async def test_scoped_storage_isolates_by_runtime_id(self):
        """Different runtime scopes produce isolated checkpoint namespaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            scoped_a = ScopedCheckpointStorage(cs, "runtime-a")
            scoped_b = ScopedCheckpointStorage(cs, "runtime-b")

            cp_a = _make_checkpoint(workflow_name="my_wf", checkpoint_id="cp-a")
            cp_b = _make_checkpoint(workflow_name="my_wf", checkpoint_id="cp-b")

            await scoped_a.save(cp_a)
            await scoped_b.save(cp_b)

            # Each scope sees only its own checkpoints
            a_cps = await scoped_a.list_checkpoints(workflow_name="my_wf")
            b_cps = await scoped_b.list_checkpoints(workflow_name="my_wf")

            assert len(a_cps) == 1
            assert a_cps[0].checkpoint_id == "cp-a"

            assert len(b_cps) == 1
            assert b_cps[0].checkpoint_id == "cp-b"
            await storage.dispose()

    async def test_scoped_get_latest_respects_scope(self):
        """get_latest only returns checkpoints within the runtime scope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            scoped_a = ScopedCheckpointStorage(cs, "rt-a")
            scoped_b = ScopedCheckpointStorage(cs, "rt-b")

            cp_a = _make_checkpoint(
                workflow_name="wf",
                checkpoint_id="cp-a",
                timestamp="2026-01-01T00:00:00+00:00",
            )
            cp_b = _make_checkpoint(
                workflow_name="wf",
                checkpoint_id="cp-b",
                timestamp="2026-01-02T00:00:00+00:00",
            )

            await scoped_a.save(cp_a)
            await scoped_b.save(cp_b)

            latest_a = await scoped_a.get_latest(workflow_name="wf")
            assert latest_a is not None
            assert latest_a.checkpoint_id == "cp-a"

            latest_b = await scoped_b.get_latest(workflow_name="wf")
            assert latest_b is not None
            assert latest_b.checkpoint_id == "cp-b"
            await storage.dispose()

    async def test_scoped_load_and_delete_are_global(self):
        """load and delete operate on checkpoint_id which is globally unique."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            scoped = ScopedCheckpointStorage(cs, "rt-x")
            cp = _make_checkpoint(workflow_name="wf", checkpoint_id="cp-global")
            await scoped.save(cp)

            # Load from any scope
            loaded = await scoped.load("cp-global")
            assert loaded.checkpoint_id == "cp-global"

            # Delete from any scope
            result = await scoped.delete("cp-global")
            assert result is True
            await storage.dispose()

    async def test_scoped_list_checkpoint_ids(self):
        """list_checkpoint_ids respects scope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteResumableStorage(db_path)
            await storage.setup()
            cs = storage.checkpoint_storage
            assert cs is not None

            scoped_a = ScopedCheckpointStorage(cs, "rt-a")
            scoped_b = ScopedCheckpointStorage(cs, "rt-b")

            cp_a = _make_checkpoint(workflow_name="wf", checkpoint_id="cp-a")
            cp_b = _make_checkpoint(workflow_name="wf", checkpoint_id="cp-b")

            await scoped_a.save(cp_a)
            await scoped_b.save(cp_b)

            ids_a = await scoped_a.list_checkpoint_ids(workflow_name="wf")
            ids_b = await scoped_b.list_checkpoint_ids(workflow_name="wf")

            assert ids_a == ["cp-a"]
            assert ids_b == ["cp-b"]
            await storage.dispose()
