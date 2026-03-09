"""Tests for SqliteResumableStorage."""

import pytest
from pydantic import BaseModel
from uipath.core.triggers import (
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from uipath_openai_agents.runtime.storage import SqliteResumableStorage


class ValueModel(BaseModel):
    value: str


@pytest.fixture
async def storage(tmp_path):
    db_path = tmp_path / "state.db"
    instance = SqliteResumableStorage(str(db_path))
    yield instance
    await instance.close()


@pytest.mark.asyncio
async def test_set_and_get_values(storage: SqliteResumableStorage):
    await storage.set_value("runtime-1", "ns", "str_key", "hello")
    await storage.set_value("runtime-1", "ns", "dict_key", {"x": 1, "y": True})
    await storage.set_value("runtime-1", "ns", "model_key", ValueModel(value="ok"))

    assert await storage.get_value("runtime-1", "ns", "str_key") == "hello"
    assert await storage.get_value("runtime-1", "ns", "dict_key") == {"x": 1, "y": True}
    assert await storage.get_value("runtime-1", "ns", "model_key") == {"value": "ok"}
    assert await storage.get_value("runtime-1", "ns", "missing") is None


@pytest.mark.asyncio
async def test_save_get_and_delete_triggers(storage: SqliteResumableStorage):
    trigger_1 = UiPathResumeTrigger(
        interrupt_id="interrupt-1",
        trigger_type=UiPathResumeTriggerType.API,
        trigger_name=UiPathResumeTriggerName.API,
        payload={"msg": "one"},
    )
    trigger_2 = UiPathResumeTrigger(
        interrupt_id="interrupt-2",
        trigger_type=UiPathResumeTriggerType.JOB,
        trigger_name=UiPathResumeTriggerName.JOB,
        payload={"msg": "two"},
    )

    await storage.save_triggers("runtime-1", [trigger_1, trigger_2])
    loaded = await storage.get_triggers("runtime-1")

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].interrupt_id == "interrupt-1"
    assert loaded[1].interrupt_id == "interrupt-2"

    await storage.delete_trigger("runtime-1", loaded[0])
    remaining = await storage.get_triggers("runtime-1")
    assert remaining is not None
    assert len(remaining) == 1
    assert remaining[0].interrupt_id == "interrupt-2"

    await storage.save_triggers("runtime-1", [])
    assert await storage.get_triggers("runtime-1") is None
