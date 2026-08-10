"""What the store does with state it cannot use.

Everything here comes back out of the state database, which survives a
suspension and therefore outlives the code that wrote it. A record written by an
older version, or truncated, must not take the run down: the reader drops what
it cannot rebuild and says so, and the caller treats that as "nothing stored".
"""

from __future__ import annotations

import logging

import pytest
from uipath.platform.common import CreateTask, InvokeProcess

from uipath_claude_sdk.runtime.session_store import (
    ClaudeSessionStore,
    EntrypointRecord,
    TranscriptRecord,
    _pending_from_stored,
    _rebuild_value,
)


class _Storage:
    """The runtime key-value table, in memory."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], object] = {}

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> object:
        return self.values.get((runtime_id, namespace, key))

    async def set_value(
        self, runtime_id: str, namespace: str, key: str, value: object
    ) -> None:
        self.values[(runtime_id, namespace, key)] = value


@pytest.fixture
def store() -> ClaudeSessionStore:
    return ClaudeSessionStore(_Storage(), "runtime-1")  # type: ignore[arg-type]


class TestTranscript:
    def test_a_round_trip_keeps_every_field(self):
        record = TranscriptRecord(
            session_id="s1", project_dir="-tmp-agent", content='{"a": 1}'
        )
        assert TranscriptRecord.from_stored(record.to_stored()) == record

    @pytest.mark.parametrize("raw", [None, "not a record", 7, ["s1"]])
    def test_anything_that_is_not_a_record_reads_as_nothing(self, raw):
        assert TranscriptRecord.from_stored(raw) is None

    def test_a_record_missing_a_field_reads_as_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert TranscriptRecord.from_stored({"session_id": "s1"}) is None
        assert "incomplete transcript record" in caplog.text


class TestEntrypoint:
    def test_a_round_trip_keeps_the_entrypoint(self):
        record = EntrypointRecord(entrypoint="agent")
        assert EntrypointRecord.from_stored(record.to_stored()) == record

    @pytest.mark.parametrize("raw", [None, {}, {"other": "agent"}, "agent"])
    def test_anything_without_an_entrypoint_reads_as_nothing(self, raw):
        assert EntrypointRecord.from_stored(raw) is None


class TestRebuildValue:
    def test_a_known_model_is_restored_to_its_type(self):
        task = CreateTask(title="Approve?", app_name="generic_escalation_app")
        rebuilt = _rebuild_value("CreateTask", task.model_dump(), "toolu_1")
        assert isinstance(rebuilt, CreateTask)
        assert rebuilt.title == "Approve?"

    def test_siblings_are_restored_one_by_one(self):
        stored = [
            InvokeProcess(name="p1", input_arguments={}).model_dump(),
            InvokeProcess(name="p2", input_arguments={}).model_dump(),
        ]
        rebuilt = _rebuild_value("InvokeProcess", stored, "toolu_1")
        assert [type(item) for item in rebuilt] == [InvokeProcess, InvokeProcess]

    def test_a_plain_value_is_left_alone(self):
        assert _rebuild_value("str", "Which vendor?", "toolu_1") == "Which vendor?"
        assert _rebuild_value(None, "Which vendor?", "toolu_1") == "Which vendor?"

    def test_an_unknown_type_is_reloaded_as_stored_and_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _rebuild_value("SomeFutureModel", {"a": 1}, "toolu_1") == {"a": 1}
        assert "not a known" in caplog.text


class TestPendingSuspend:
    def test_anything_that_is_not_a_record_reads_as_nothing(self, caplog):
        assert _pending_from_stored(None) is None
        with caplog.at_level(logging.WARNING):
            assert _pending_from_stored("not a record") is None
        assert "not an object" in caplog.text

    def test_a_record_missing_a_field_reads_as_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _pending_from_stored({"interrupt_id": "toolu_1"}) is None
        assert "incomplete pending suspend record" in caplog.text

    def test_a_stored_tool_use_id_stays_a_string(self):
        pending = _pending_from_stored(
            {
                "interrupt_id": "toolu_1",
                "tool_name": "mcp__uipath__wait_for_input",
                "value": "Which vendor?",
                "value_type": "str",
                "tool_use_id": "toolu_1",
                "requested_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert pending is not None
        assert pending.tool_use_id == "toolu_1"

    def test_a_record_without_a_tool_use_id_keeps_it_absent(self):
        pending = _pending_from_stored(
            {
                "interrupt_id": "toolu_1",
                "tool_name": "mcp__uipath__wait_for_input",
                "value": "Which vendor?",
                "value_type": "str",
                "tool_use_id": None,
                "requested_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert pending is not None
        assert pending.tool_use_id is None


class TestStore:
    async def test_an_empty_store_reports_nothing_stored(self, store):
        assert await store.get_session_id() is None
        assert await store.get_pending_suspend() is None
        assert await store.get_transcript() is None

    async def test_a_transcript_survives_a_round_trip_and_a_clear(self, store):
        record = TranscriptRecord(session_id="s1", project_dir="-tmp", content="{}")
        await store.set_transcript(record)
        assert await store.get_transcript() == record

        await store.clear_transcript()
        assert await store.get_transcript() is None

    async def test_a_session_id_that_is_not_a_string_reads_as_nothing(self, store):
        await store.set_session_id("s1")
        assert await store.get_session_id() == "s1"

        await store._set("session_id", 7)
        assert await store.get_session_id() is None
