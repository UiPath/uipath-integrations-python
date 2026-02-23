"""Tests for SqliteResumableStorage class."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from pydantic import BaseModel
from uipath.runtime import (
    UiPathApiTrigger,
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from uipath_llamaindex.runtime.storage import SqliteResumableStorage


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestSqliteResumableStorageInitialization:
    """Test storage initialization and setup."""

    @pytest.mark.asyncio
    async def test_setup_creates_database_file(self, tmp_path: Path):
        """Test that setup creates the database file."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            assert db_path.exists()

    @pytest.mark.asyncio
    async def test_setup_creates_directory_if_missing(self, tmp_path: Path):
        """Test that setup creates parent directories if they don't exist."""
        db_path = tmp_path / "subdir" / "another" / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            assert db_path.exists()
            assert db_path.parent.exists()

    @pytest.mark.asyncio
    async def test_setup_creates_workflow_contexts_table(self, tmp_path: Path):
        """Test that setup creates the workflow_contexts table."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            async with aiosqlite.connect(str(db_path)) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_contexts'"
                )
                result = await cursor.fetchone()
                assert result is not None

    @pytest.mark.asyncio
    async def test_setup_creates_resume_triggers_table(self, tmp_path: Path):
        """Test that setup creates the resume_triggers table."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            async with aiosqlite.connect(str(db_path)) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='resume_triggers'"
                )
                result = await cursor.fetchone()
                assert result is not None

    @pytest.mark.asyncio
    async def test_setup_creates_runtime_kv_table(self, tmp_path: Path):
        """Test that setup creates the runtime_kv table."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            async with aiosqlite.connect(str(db_path)) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='runtime_kv'"
                )
                result = await cursor.fetchone()
                assert result is not None

    @pytest.mark.asyncio
    async def test_setup_is_idempotent(self, tmp_path: Path):
        """Test that setup can be called multiple times safely."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            await storage.setup()
            await storage.setup()  # Should not raise
            assert db_path.exists()


class TestTriggerOperations:
    """Test resume trigger save and retrieval operations."""

    @pytest.fixture
    async def storage(self):
        """Create a SqliteResumableStorage instance with temporary database file."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        try:
            async with SqliteResumableStorage(str(temp_db.name)) as storage:
                await storage.setup()
                yield storage
        finally:
            if os.path.exists(temp_db.name):
                os.remove(temp_db.name)

    @pytest.mark.asyncio
    async def test_save_trigger_basic(self, storage: SqliteResumableStorage):
        """Test saving a basic resume trigger."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="queue-123",
            folder_path="/test/folder",
            folder_key="folder-456",
            payload={"data": "test"},
            interrupt_id="interrupt-789",
        )

        await storage.save_triggers("runtime-1", [trigger])

        # Verify it was saved
        triggers = await storage.get_triggers("runtime-1")
        assert triggers is not None
        assert triggers[0] is not None
        assert triggers[0].trigger_type == UiPathResumeTriggerType.QUEUE_ITEM
        assert triggers[0].trigger_name == UiPathResumeTriggerName.QUEUE_ITEM
        assert triggers[0].item_key == "queue-123"

    @pytest.mark.asyncio
    async def test_save_trigger_with_api_type(self, storage: SqliteResumableStorage):
        """Test saving an API type trigger."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API.value,
            item_key="inbox-789",
            folder_path="/api/folder",
            folder_key="folder-abc",
            payload='{"request": "data"}',
            interrupt_id="interrupt-123",
        )
        trigger.api_resume = UiPathApiTrigger(
            inbox_id="inbox-789", request='{"request": "data"}'
        )

        await storage.save_triggers("runtime-2", [trigger])

        retrieved = await storage.get_triggers("runtime-2")
        assert retrieved is not None
        assert retrieved[0] is not None
        assert retrieved[0].trigger_type == UiPathResumeTriggerType.API
        assert retrieved[0].api_resume is not None
        assert retrieved[0].api_resume.inbox_id == "inbox-789"

    @pytest.mark.asyncio
    async def test_save_trigger_with_dict_payload(
        self, storage: SqliteResumableStorage
    ):
        """Test saving trigger with dictionary payload."""
        payload_dict = {"key1": "value1", "key2": 123, "nested": {"a": "b"}}
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="queue-dict",
            payload=payload_dict,
            interrupt_id="interrupt-456",
        )

        await storage.save_triggers("runtime-3", [trigger])

        retrieved = await storage.get_triggers("runtime-3")
        assert retrieved is not None
        assert retrieved[0] is not None
        # Payload should be JSON string after serialization/deserialization
        assert retrieved[0].payload is not None

    @pytest.mark.asyncio
    async def test_save_trigger_with_none_payload(
        self, storage: SqliteResumableStorage
    ):
        """Test saving trigger with None payload."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="queue-none",
            payload=None,
            interrupt_id="interrupt-789",
        )

        await storage.save_triggers("runtime-4", [trigger])

        retrieved = await storage.get_triggers("runtime-4")
        assert retrieved is not None
        assert retrieved[0] is not None
        assert retrieved[0].payload is None

    @pytest.mark.asyncio
    async def test_get_latest_trigger_multiple_triggers(
        self, storage: SqliteResumableStorage
    ):
        """Test that get_latest_trigger returns the most recent trigger."""
        # Save multiple triggers for the same runtime
        trigger1 = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="first",
            interrupt_id="interrupt-1",
        )
        trigger2 = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="second",
            interrupt_id="interrupt-2",
        )
        trigger3 = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="third",
            interrupt_id="interrupt-3",
        )

        await storage.save_triggers("runtime-5", [trigger1])
        await storage.save_triggers("runtime-5", [trigger2])
        await storage.save_triggers("runtime-5", [trigger3])

        retrieved = await storage.get_triggers("runtime-5")
        assert retrieved is not None
        assert retrieved[0] is not None
        assert retrieved[0].item_key == "third"

    @pytest.mark.asyncio
    async def test_get_latest_trigger_nonexistent(
        self, storage: SqliteResumableStorage
    ):
        """Test getting trigger for non-existent runtime_id."""
        result = await storage.get_triggers("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_trigger_different_runtimes(
        self, storage: SqliteResumableStorage
    ):
        """Test that triggers are properly isolated by runtime_id."""
        trigger1 = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="runtime1-trigger",
            interrupt_id="interrupt-1",
        )
        trigger2 = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="runtime2-trigger",
            interrupt_id="interrupt-2",
        )

        await storage.save_triggers("runtime-a", [trigger1])
        await storage.save_triggers("runtime-b", [trigger2])

        retrieved_a = await storage.get_triggers("runtime-a")
        retrieved_b = await storage.get_triggers("runtime-b")
        assert retrieved_a is not None
        assert retrieved_b is not None
        assert retrieved_a[0] is not None
        assert retrieved_b[0] is not None
        assert retrieved_a[0].item_key == "runtime1-trigger"
        assert retrieved_b[0].item_key == "runtime2-trigger"


class TestContextOperations:
    """Test workflow context save and load operations."""

    @pytest.fixture
    async def storage(self):
        """Create a SqliteResumableStorage instance with temporary database file."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        try:
            async with SqliteResumableStorage(str(temp_db.name)) as storage:
                await storage.setup()
                yield storage
        finally:
            if os.path.exists(temp_db.name):
                os.remove(temp_db.name)

    @pytest.mark.asyncio
    async def test_save_and_load_context_basic(self, storage: SqliteResumableStorage):
        """Test saving and loading a basic context."""
        context = {"step": 1, "data": "test data", "flags": {"active": True}}

        await storage.save_context("runtime-1", context)
        loaded = await storage.load_context("runtime-1")

        assert loaded == context

    @pytest.mark.asyncio
    async def test_save_and_load_context_complex(self, storage: SqliteResumableStorage):
        """Test saving and loading complex context with nested structures."""
        context = {
            "variables": {"counter": 42, "name": "test", "items": [1, 2, 3, 4, 5]},
            "state": {
                "current_step": "processing",
                "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
            },
        }

        await storage.save_context("runtime-2", context)
        loaded = await storage.load_context("runtime-2")

        assert loaded == context

    @pytest.mark.asyncio
    async def test_save_context_overwrites_existing(
        self, storage: SqliteResumableStorage
    ):
        """Test that saving context overwrites existing context."""
        context1 = {"step": 1}
        context2 = {"step": 2, "new_field": "value"}

        await storage.save_context("runtime-3", context1)
        await storage.save_context("runtime-3", context2)

        loaded = await storage.load_context("runtime-3")
        assert loaded == context2
        assert loaded != context1

    @pytest.mark.asyncio
    async def test_load_context_nonexistent(self, storage: SqliteResumableStorage):
        """Test loading context for non-existent runtime_id."""
        result = await storage.load_context("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_context_empty_dict(self, storage: SqliteResumableStorage):
        """Test saving empty dictionary as context."""
        context: dict[str, Any] = {}

        await storage.save_context("runtime-4", context)
        loaded = await storage.load_context("runtime-4")

        assert loaded == {}

    @pytest.mark.asyncio
    async def test_contexts_isolated_by_runtime_id(
        self, storage: SqliteResumableStorage
    ):
        """Test that contexts are properly isolated by runtime_id."""
        context_a = {"runtime": "a", "value": 100}
        context_b = {"runtime": "b", "value": 200}

        await storage.save_context("runtime-a", context_a)
        await storage.save_context("runtime-b", context_b)

        loaded_a = await storage.load_context("runtime-a")
        loaded_b = await storage.load_context("runtime-b")

        assert loaded_a == context_a
        assert loaded_b == context_b


class TestKeyValueOperations:
    """Test key-value storage operations."""

    @pytest.fixture
    async def storage(self):
        """Create a SqliteResumableStorage instance with temporary database file."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        try:
            async with SqliteResumableStorage(str(temp_db.name)) as storage:
                await storage.setup()
                yield storage
        finally:
            if os.path.exists(temp_db.name):
                os.remove(temp_db.name)

    @pytest.mark.asyncio
    async def test_set_and_get_string_value(self, storage: SqliteResumableStorage):
        """Test setting and getting a string value."""
        await storage.set_value("runtime-1", "namespace1", "key1", "test_value")

        value = await storage.get_value("runtime-1", "namespace1", "key1")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_set_and_get_dict_value(self, storage: SqliteResumableStorage):
        """Test setting and getting a dictionary value."""
        test_dict = {"name": "John", "age": 30, "active": True}

        await storage.set_value("runtime-2", "namespace2", "key2", test_dict)

        value = await storage.get_value("runtime-2", "namespace2", "key2")
        assert value == test_dict

    @pytest.mark.asyncio
    async def test_set_and_get_pydantic_model(self, storage: SqliteResumableStorage):
        """Test setting and getting a Pydantic model."""
        model = SampleModel(name="test", value=42)

        await storage.set_value("runtime-3", "namespace3", "key3", model)

        value = await storage.get_value("runtime-3", "namespace3", "key3")
        assert value == model.model_dump()

    @pytest.mark.asyncio
    async def test_set_and_get_none_value(self, storage: SqliteResumableStorage):
        """Test setting and getting None value."""
        await storage.set_value("runtime-4", "namespace4", "key4", None)

        value = await storage.get_value("runtime-4", "namespace4", "key4")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_value_invalid_type(self, storage: SqliteResumableStorage):
        """Test that setting invalid type raises TypeError."""
        with pytest.raises(
            TypeError, match="Value must be str, dict, BaseModel or None"
        ):
            await storage.set_value("runtime-5", "namespace5", "key5", 123)

        with pytest.raises(
            TypeError, match="Value must be str, dict, BaseModel or None"
        ):
            await storage.set_value("runtime-5", "namespace5", "key5", [1, 2, 3])

    @pytest.mark.asyncio
    async def test_set_value_overwrites_existing(self, storage: SqliteResumableStorage):
        """Test that setting a value overwrites existing value."""
        await storage.set_value("runtime-6", "namespace6", "key6", "first")
        await storage.set_value("runtime-6", "namespace6", "key6", "second")

        value = await storage.get_value("runtime-6", "namespace6", "key6")
        assert value == "second"

    @pytest.mark.asyncio
    async def test_get_value_nonexistent(self, storage: SqliteResumableStorage):
        """Test getting non-existent value returns None."""
        value = await storage.get_value("nonexistent", "namespace", "key")
        assert value is None

    @pytest.mark.asyncio
    async def test_values_isolated_by_runtime_id(self, storage: SqliteResumableStorage):
        """Test that values are isolated by runtime_id."""
        await storage.set_value("runtime-a", "ns", "key", "value-a")
        await storage.set_value("runtime-b", "ns", "key", "value-b")

        value_a = await storage.get_value("runtime-a", "ns", "key")
        value_b = await storage.get_value("runtime-b", "ns", "key")

        assert value_a == "value-a"
        assert value_b == "value-b"

    @pytest.mark.asyncio
    async def test_values_isolated_by_namespace(self, storage: SqliteResumableStorage):
        """Test that values are isolated by namespace."""
        await storage.set_value("runtime-1", "ns-a", "key", "value-a")
        await storage.set_value("runtime-1", "ns-b", "key", "value-b")

        value_a = await storage.get_value("runtime-1", "ns-a", "key")
        value_b = await storage.get_value("runtime-1", "ns-b", "key")

        assert value_a == "value-a"
        assert value_b == "value-b"

    @pytest.mark.asyncio
    async def test_values_isolated_by_key(self, storage: SqliteResumableStorage):
        """Test that values are isolated by key."""
        await storage.set_value("runtime-1", "ns", "key-a", "value-a")
        await storage.set_value("runtime-1", "ns", "key-b", "value-b")

        value_a = await storage.get_value("runtime-1", "ns", "key-a")
        value_b = await storage.get_value("runtime-1", "ns", "key-b")

        assert value_a == "value-a"
        assert value_b == "value-b"


class TestSerializationMethods:
    """Test internal serialization/deserialization methods."""

    @pytest.fixture
    async def storage(self):
        """Create a SqliteResumableStorage instance with temporary database file."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()

        try:
            async with SqliteResumableStorage(str(temp_db.name)) as storage:
                await storage.setup()
                yield storage
        finally:
            if os.path.exists(temp_db.name):
                os.remove(temp_db.name)

    def test_serialize_trigger_queue_type(self, storage: SqliteResumableStorage):
        """Test serialization of queue type trigger."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
            trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
            item_key="queue-123",
            folder_path="/folder",
            folder_key="folder-key",
            payload={"test": "data"},
            interrupt_id="interrupt-456",
        )

        serialized = storage._serialize_trigger(trigger)

        assert serialized["type"] == UiPathResumeTriggerType.QUEUE_ITEM.value
        assert serialized["key"] == "queue-123"
        assert serialized["name"] == UiPathResumeTriggerName.QUEUE_ITEM.value
        assert serialized["folder_path"] == "/folder"
        assert serialized["folder_key"] == "folder-key"
        assert serialized["interrupt_id"] == "interrupt-456"
        assert json.loads(serialized["payload"]) == {"test": "data"}

    def test_serialize_trigger_api_type(self, storage: SqliteResumableStorage):
        """Test serialization of API type trigger."""
        trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType.API,
            trigger_name=UiPathResumeTriggerName.API.value,
            item_key="inbox-456",
            payload="string payload",
            interrupt_id="interrupt-123",
        )
        trigger.api_resume = UiPathApiTrigger(
            inbox_id="inbox-456", request="string payload"
        )

        serialized = storage._serialize_trigger(trigger)

        assert serialized["type"] == UiPathResumeTriggerType.API.value
        assert serialized["key"] == "inbox-456"
        assert serialized["payload"] == "string payload"
        assert serialized["interrupt_id"] == "interrupt-123"

    def test_deserialize_trigger_queue_type(self, storage: SqliteResumableStorage):
        """Test deserialization of queue type trigger."""
        trigger_data = {
            "type": UiPathResumeTriggerType.QUEUE_ITEM.value,
            "key": "queue-789",
            "name": UiPathResumeTriggerName.QUEUE_ITEM.value,
            "folder_path": "/test",
            "folder_key": "folder-123",
            "payload": '{"key": "value"}',
        }

        trigger = storage._deserialize_trigger(trigger_data)

        assert trigger is not None
        assert trigger.trigger_type == UiPathResumeTriggerType.QUEUE_ITEM
        assert trigger.trigger_name == UiPathResumeTriggerName.QUEUE_ITEM
        assert trigger.item_key == "queue-789"
        assert trigger.folder_path == "/test"
        assert trigger.folder_key == "folder-123"

    def test_deserialize_trigger_api_type(self, storage: SqliteResumableStorage):
        """Test deserialization of API type trigger."""
        trigger_data = {
            "type": UiPathResumeTriggerType.API.value,
            "key": "inbox-abc",
            "name": UiPathResumeTriggerName.API.value,
            "payload": "request data",
        }

        trigger = storage._deserialize_trigger(trigger_data)

        assert trigger.trigger_type == UiPathResumeTriggerType.API
        assert trigger.api_resume is not None
        assert trigger.api_resume.inbox_id == "inbox-abc"
        assert trigger.api_resume.request == "request data"

    def test_dump_value_string(self, storage: SqliteResumableStorage):
        """Test _dump_value with string."""
        result = storage._dump_value("test string")
        assert result == "s:test string"

    def test_dump_value_dict(self, storage: SqliteResumableStorage):
        """Test _dump_value with dictionary."""
        result = storage._dump_value({"key": "value"})
        assert result == 'j:{"key": "value"}'

    def test_dump_value_pydantic_model(self, storage: SqliteResumableStorage):
        """Test _dump_value with Pydantic model."""
        model = SampleModel(name="test", value=42)
        result = storage._dump_value(model)
        assert result == 'j:{"name": "test", "value": 42}'

    def test_dump_value_none(self, storage: SqliteResumableStorage):
        """Test _dump_value with None."""
        result = storage._dump_value(None)
        assert result is None

    def test_load_value_string(self, storage: SqliteResumableStorage):
        """Test _load_value with string."""
        result = storage._load_value("s:test string")
        assert result == "test string"

    def test_load_value_json(self, storage: SqliteResumableStorage):
        """Test _load_value with JSON."""
        result = storage._load_value('j:{"key": "value"}')
        assert result == {"key": "value"}

    def test_load_value_none(self, storage: SqliteResumableStorage):
        """Test _load_value with None."""
        result = storage._load_value(None)
        assert result is None

    def test_load_value_raw_string(self, storage: SqliteResumableStorage):
        """Test _load_value with raw string (no prefix)."""
        result = storage._load_value("raw string")
        assert result == "raw string"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_context_with_non_picklable_object(self, tmp_path: Path):
        """Test handling of non-picklable objects in context."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            # Lambda functions are not picklable
            context = {"func": lambda x: x + 1}

            with pytest.raises(
                (AttributeError, TypeError)
            ):  # pickle.PicklingError or similar
                await storage.save_context("runtime-1", context)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_operations(self, tmp_path: Path):
        """Test that multiple concurrent operations work correctly."""
        import asyncio

        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:

            async def save_trigger(runtime_id: str, key: str):
                trigger = UiPathResumeTrigger(
                    interrupt_id="interrupt-1",
                    trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
                    trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
                    item_key=key,
                )
                await storage.save_triggers(runtime_id, [trigger])

            # Run multiple saves concurrently
            await asyncio.gather(
                save_trigger("runtime-1", "key-1"),
                save_trigger("runtime-2", "key-2"),
                save_trigger("runtime-3", "key-3"),
            )

            # Verify all were saved
            trigger1 = await storage.get_triggers("runtime-1")
            trigger2 = await storage.get_triggers("runtime-2")
            trigger3 = await storage.get_triggers("runtime-3")

            assert trigger1 is not None
            assert trigger2 is not None
            assert trigger3 is not None
            assert trigger1[0] is not None
            assert trigger2[0] is not None
            assert trigger3[0] is not None
            assert trigger1[0].item_key == "key-1"
            assert trigger2[0].item_key == "key-2"
            assert trigger3[0].item_key == "key-3"


class TestDatabaseSchema:
    """Test database schema and constraints."""

    @pytest.mark.asyncio
    async def test_runtime_kv_primary_key_constraint(self, tmp_path: Path):
        """Test that runtime_kv primary key constraint works."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            # First insert
            await storage.set_value("runtime-1", "ns", "key", "value1")

            # Second insert with same primary key should update, not fail
            await storage.set_value("runtime-1", "ns", "key", "value2")

            value = await storage.get_value("runtime-1", "ns", "key")
            assert value == "value2"

    @pytest.mark.asyncio
    async def test_workflow_contexts_primary_key_constraint(self, tmp_path: Path):
        """Test that workflow_contexts primary key constraint works."""
        db_path = tmp_path / "test.db"
        async with SqliteResumableStorage(str(db_path)) as storage:
            # First save
            await storage.save_context("runtime-1", {"step": 1})

            # Second save with same runtime_id should update, not fail
            await storage.save_context("runtime-1", {"step": 2})

            context = await storage.load_context("runtime-1")
            assert context == {"step": 2}

    @pytest.mark.asyncio
    async def test_resume_triggers_autoincrement(self, tmp_path: Path):
        """Test that resume_triggers id autoincrement works."""
        db_path = tmp_path / "test.db"

        async with SqliteResumableStorage(str(db_path)) as storage:
            trigger1 = UiPathResumeTrigger(
                interrupt_id="interrupt-1",
                trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
                trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
                item_key="key-1",
            )
            trigger2 = UiPathResumeTrigger(
                interrupt_id="interrupt-2",
                trigger_type=UiPathResumeTriggerType.QUEUE_ITEM,
                trigger_name=UiPathResumeTriggerName.QUEUE_ITEM.value,
                item_key="key-2",
            )
            await storage.save_triggers("runtime-1", [trigger1, trigger2])

        # Verify both were saved (autoincrement allowed multiple rows).
        # Open a fresh connection after the storage is disposed to avoid locking.
        async with aiosqlite.connect(str(db_path)) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM resume_triggers WHERE runtime_id = ?",
                ("runtime-1",),
            )
            row = await cursor.fetchone()
            assert row is not None
            count = row[0]
            assert count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
