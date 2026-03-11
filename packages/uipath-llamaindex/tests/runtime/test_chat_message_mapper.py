"""Tests for UiPathChatMessagesMapper."""

from typing import Any
from unittest.mock import AsyncMock

from llama_index.core.agent.workflow.workflow_events import (  # type: ignore[attr-defined]
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
    ToolSelection,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolOutput

from uipath_llamaindex.runtime.chat.messages import (
    STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
    STORAGE_NAMESPACE_EVENT_MAPPER,
    UiPathChatMessagesMapper,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def create_mock_storage(initial_map: dict[str, str] | None = None):
    """Create an AsyncMock storage with optional pre-populated tool_id map."""
    storage = AsyncMock()
    storage.get_value = AsyncMock(return_value=initial_map)
    storage.set_value = AsyncMock()
    return storage


def make_agent_stream(delta: str, response: str = "") -> AgentStream:
    return AgentStream(delta=delta, response=response, current_agent_name="agent")


def make_tool_selection(
    tool_id: str, tool_name: str, tool_kwargs: dict[str, Any] | None = None
) -> ToolSelection:
    return ToolSelection(
        tool_id=tool_id, tool_name=tool_name, tool_kwargs=tool_kwargs or {}
    )


def make_agent_output(
    tool_selections: list[ToolSelection] | None = None,
) -> AgentOutput:
    msg = ChatMessage(role="assistant", content="response")
    return AgentOutput(
        response=msg,
        current_agent_name="agent",
        tool_calls=tool_selections or [],
    )


def make_tool_call_event(tool_id: str, tool_name: str = "tool") -> ToolCall:
    return ToolCall(tool_id=tool_id, tool_name=tool_name, tool_kwargs={})


def make_tool_call_result(tool_id: str, content: str) -> ToolCallResult:
    tool_output = ToolOutput(
        content=content,
        tool_name="tool",
        raw_input={},
        raw_output=content,
    )
    return ToolCallResult(
        tool_id=tool_id,
        tool_name="tool",
        tool_kwargs={},
        tool_output=tool_output,
        return_direct=False,
    )


# ── TestMapInput ──────────────────────────────────────────────────────────────


class TestMapInput:
    def test_passthrough_when_user_msg_present(self):
        result = UiPathChatMessagesMapper.map_input({"user_msg": "hello"})
        assert result == {"user_msg": "hello"}

    def test_passthrough_when_no_messages_key(self):
        result = UiPathChatMessagesMapper.map_input({"other_key": "value"})
        assert result == {"other_key": "value"}

    def test_passthrough_on_empty_input(self):
        result = UiPathChatMessagesMapper.map_input({})
        assert result == {}

    def test_extracts_text_from_messages(self):
        input_data = {
            "messages": [
                {
                    "contentParts": [
                        {
                            "mimeType": "text/plain",
                            "data": {"inline": "hello world"},
                        }
                    ]
                }
            ]
        }
        result = UiPathChatMessagesMapper.map_input(input_data)
        assert result == {"user_msg": "hello world"}

    def test_joins_multiple_text_parts(self):
        input_data = {
            "messages": [
                {
                    "contentParts": [
                        {"mimeType": "text/plain", "data": {"inline": "hello"}},
                        {"mimeType": "text/plain", "data": {"inline": "world"}},
                    ]
                }
            ]
        }
        result = UiPathChatMessagesMapper.map_input(input_data)
        assert result == {"user_msg": "hello world"}

    def test_skips_non_text_mime_types(self):
        input_data = {
            "messages": [
                {
                    "contentParts": [
                        {"mimeType": "image/png", "data": {"inline": "base64data"}}
                    ]
                }
            ]
        }
        result = UiPathChatMessagesMapper.map_input(input_data)
        # No text parts → returned unchanged
        assert result == input_data

    def test_passthrough_when_messages_is_empty(self):
        result = UiPathChatMessagesMapper.map_input({"messages": []})
        assert result == {"messages": []}

    def test_uses_first_message_only(self):
        input_data = {
            "messages": [
                {
                    "contentParts": [
                        {"mimeType": "text/plain", "data": {"inline": "first"}}
                    ]
                },
                {
                    "contentParts": [
                        {"mimeType": "text/plain", "data": {"inline": "second"}}
                    ]
                },
            ]
        }
        result = UiPathChatMessagesMapper.map_input(input_data)
        assert result == {"user_msg": "first"}


# ── TestMapEventDispatch ──────────────────────────────────────────────────────


class TestMapEventDispatch:
    async def test_agent_stream_dispatches_to_stream_handler(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        event = make_agent_stream("hello")
        result = await mapper.map_event(event)
        assert result is not None

    async def test_agent_output_dispatches_to_output_handler(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-1"
        event = make_agent_output()
        result = await mapper.map_event(event)
        assert result is not None

    async def test_tool_call_event_returns_none(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        event = make_tool_call_event("tc-1")
        result = await mapper.map_event(event)
        assert result is None

    async def test_tool_call_result_dispatches_to_result_handler(self):
        storage = create_mock_storage(initial_map={"tc-1": "msg-1"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        event = make_tool_call_result("tc-1", "output")
        result = await mapper.map_event(event)
        assert result is not None


# ── TestMapEventAgentStream ───────────────────────────────────────────────────


class TestMapEventAgentStream:
    async def test_first_chunk_emits_message_start_and_content(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        result = await mapper.map_event(make_agent_stream("hello"))

        assert result is not None
        assert len(result) == 2
        start_event = result[0]
        assert start_event.start is not None
        assert start_event.start.role == "assistant"
        assert start_event.content_part is not None
        assert start_event.content_part.start is not None
        chunk_event = result[1]
        assert chunk_event.content_part is not None
        assert chunk_event.content_part.chunk is not None
        assert chunk_event.content_part.chunk.data == "hello"

    async def test_first_chunk_assigns_message_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        assert mapper._current_message_id is None
        await mapper.map_event(make_agent_stream("hello"))
        assert mapper._current_message_id is not None

    async def test_first_chunk_with_empty_delta_emits_only_start(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        result = await mapper.map_event(make_agent_stream(""))

        assert result is not None
        assert len(result) == 1
        assert result[0].start is not None

    async def test_subsequent_chunks_reuse_same_message_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        await mapper.map_event(make_agent_stream("first"))
        first_id = mapper._current_message_id

        result = await mapper.map_event(make_agent_stream("second"))
        assert result is not None
        for event in result:
            assert event.message_id == first_id

    async def test_subsequent_chunk_does_not_emit_message_start(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        await mapper.map_event(make_agent_stream("first"))

        result = await mapper.map_event(make_agent_stream("second"))
        assert result is not None
        # Only a chunk event, no start event
        assert all(e.start is None for e in result)

    async def test_subsequent_empty_delta_returns_none(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        await mapper.map_event(make_agent_stream("first"))

        result = await mapper.map_event(make_agent_stream(""))
        assert not result

    async def test_all_events_share_consistent_content_part_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        result = await mapper.map_event(make_agent_stream("hello"))
        assert result is not None
        message_id = result[0].message_id
        expected_part_id = mapper.get_content_part_id(message_id)
        for event in result:
            if event.content_part is not None:
                assert event.content_part.content_part_id == expected_part_id


# ── TestMapEventAgentOutputNoToolCalls ───────────────────────────────────────


class TestMapEventAgentOutputNoToolCalls:
    async def test_emits_message_end(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-1"
        result = await mapper.map_event(make_agent_output())

        assert result is not None
        assert len(result) == 1
        end_event = result[0]
        assert end_event.message_id == "msg-1"
        assert end_event.end is not None
        assert end_event.content_part is not None
        assert end_event.content_part.end is not None

    async def test_resets_current_message_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-1"
        await mapper.map_event(make_agent_output())
        assert mapper._current_message_id is None

    async def test_returns_none_when_no_current_message_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        result = await mapper.map_event(make_agent_output())
        assert result is None

    async def test_end_event_uses_correct_content_part_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-42"
        result = await mapper.map_event(make_agent_output())

        assert result is not None
        end_event = result[0]
        assert end_event.content_part is not None
        assert end_event.content_part.content_part_id == "chunk-msg-42-0"


# ── TestMapEventAgentOutputWithToolCalls (storage) ───────────────────────────


class TestMapEventAgentOutputWithToolCallsStorage:
    async def test_emits_tool_call_start_events(self):
        storage = create_mock_storage(initial_map={})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool", {"arg": "val"})
        result = await mapper.map_event(make_agent_output([ts]))

        assert result is not None
        tool_events = [e for e in result if e.tool_call is not None]
        assert len(tool_events) == 1
        tc_event = tool_events[0]
        assert tc_event.message_id == "msg-1"
        assert tc_event.tool_call is not None
        assert tc_event.tool_call.tool_call_id == "tc-1"
        assert tc_event.tool_call.start is not None
        assert tc_event.tool_call.start.tool_name == "my_tool"
        assert tc_event.tool_call.start.input == {"arg": "val"}

    async def test_does_not_emit_message_end(self):
        storage = create_mock_storage(initial_map={})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool")
        result = await mapper.map_event(make_agent_output([ts]))

        assert result is not None
        assert all(e.end is None for e in result)

    async def test_writes_tool_id_to_message_id_map_to_storage(self):
        storage = create_mock_storage(initial_map={})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool")
        await mapper.map_event(make_agent_output([ts]))

        storage.set_value.assert_called_once_with(
            "test-runtime",
            STORAGE_NAMESPACE_EVENT_MAPPER,
            STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
            {"tc-1": "msg-1"},
        )

    async def test_merges_with_existing_storage_map(self):
        storage = create_mock_storage(initial_map={"existing-tc": "msg-0"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool")
        await mapper.map_event(make_agent_output([ts]))

        call_args = storage.set_value.call_args[0]
        stored_map = call_args[3]
        assert stored_map["existing-tc"] == "msg-0"
        assert stored_map["tc-1"] == "msg-1"

    async def test_multiple_tool_calls_all_written_to_storage(self):
        storage = create_mock_storage(initial_map={})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts1 = make_tool_selection("tc-1", "tool_a")
        ts2 = make_tool_selection("tc-2", "tool_b")
        result = await mapper.map_event(make_agent_output([ts1, ts2]))

        assert result is not None
        tool_events = [e for e in result if e.tool_call is not None]
        assert len(tool_events) == 2

        call_args = storage.set_value.call_args[0]
        stored_map = call_args[3]
        assert stored_map == {"tc-1": "msg-1", "tc-2": "msg-1"}

    async def test_resets_current_message_id_after_tool_calls(self):
        storage = create_mock_storage(initial_map={})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool")
        await mapper.map_event(make_agent_output([ts]))

        assert mapper._current_message_id is None


# ── TestMapEventAgentOutputWithToolCallsNoStorage (in-memory) ────────────────


class TestMapEventAgentOutputWithToolCallsNoStorage:
    async def test_emits_tool_call_start_events(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool", {"x": 1})
        result = await mapper.map_event(make_agent_output([ts]))

        assert result is not None
        tool_events = [e for e in result if e.tool_call is not None]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call is not None
        assert tool_events[0].tool_call.tool_call_id == "tc-1"
        assert tool_events[0].tool_call.start is not None
        assert tool_events[0].tool_call.start.tool_name == "my_tool"

    async def test_populates_in_memory_state(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._current_message_id = "msg-1"

        ts = make_tool_selection("tc-1", "my_tool")
        await mapper.map_event(make_agent_output([ts]))

        assert mapper._tool_id_to_message_id == {"tc-1": "msg-1"}
        assert "msg-1" in mapper._pending_tool_calls
        assert "tc-1" in mapper._pending_tool_calls["msg-1"]


# ── TestMapEventToolCallResult (storage) ─────────────────────────────────────


class TestMapEventToolCallResultStorage:
    async def test_emits_tool_call_end_and_message_end_for_single_call(self):
        storage = create_mock_storage(initial_map={"tc-1": "msg-1"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        result = await mapper.map_event(make_tool_call_result("tc-1", "the output"))

        assert result is not None
        assert len(result) == 2

        end_event = result[0]
        assert end_event.message_id == "msg-1"
        assert end_event.tool_call is not None
        assert end_event.tool_call.tool_call_id == "tc-1"
        assert end_event.tool_call.end is not None
        assert end_event.tool_call.end.output == "the output"

        msg_end = result[1]
        assert msg_end.message_id == "msg-1"
        assert msg_end.end is not None
        assert msg_end.content_part is not None
        assert msg_end.content_part.end is not None

    async def test_does_not_emit_message_end_when_other_calls_pending(self):
        # Two tool calls for the same message; first result should not close it
        storage = create_mock_storage(initial_map={"tc-1": "msg-1", "tc-2": "msg-1"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        # After consuming tc-1 the map still has tc-2
        storage.get_value.return_value = {"tc-1": "msg-1", "tc-2": "msg-1"}

        result = await mapper.map_event(make_tool_call_result("tc-1", "result1"))

        assert result is not None
        assert len(result) == 1  # Only tool_call_end, no message_end
        assert result[0].tool_call is not None
        assert result[0].end is None

    async def test_emits_message_end_on_last_tool_call(self):
        storage = create_mock_storage()
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        # First result: tc-1 consumed, tc-2 still pending
        storage.get_value.return_value = {"tc-1": "msg-1", "tc-2": "msg-1"}
        result1 = await mapper.map_event(make_tool_call_result("tc-1", "r1"))
        assert result1 is not None
        assert len(result1) == 1  # no message_end

        # Second result: tc-2 consumed, map now empty for msg-1
        storage.get_value.return_value = {"tc-2": "msg-1"}
        result2 = await mapper.map_event(make_tool_call_result("tc-2", "r2"))
        assert result2 is not None
        assert len(result2) == 2  # tool_call_end + message_end
        assert result2[1].end is not None

    async def test_removes_entry_from_storage_after_processing(self):
        storage = create_mock_storage(initial_map={"tc-1": "msg-1"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        await mapper.map_event(make_tool_call_result("tc-1", "output"))

        storage.set_value.assert_called_once()
        call_args = storage.set_value.call_args[0]
        assert "tc-1" not in call_args[3]

    async def test_returns_none_for_unknown_tool_id(self, caplog):
        storage = create_mock_storage(initial_map={"tc-other": "msg-1"})
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        import logging

        with caplog.at_level(logging.ERROR):
            result = await mapper.map_event(make_tool_call_result("tc-unknown", "x"))

        assert result is None

    async def test_returns_none_when_storage_map_is_missing(self, caplog):
        storage = create_mock_storage(initial_map=None)
        mapper = UiPathChatMessagesMapper("test-runtime", storage=storage)

        import logging

        with caplog.at_level(logging.ERROR):
            result = await mapper.map_event(make_tool_call_result("tc-1", "x"))

        assert result is None


# ── TestMapEventToolCallResultNoStorage (in-memory fallback) ─────────────────


class TestMapEventToolCallResultNoStorage:
    async def test_resolves_tool_id_from_in_memory_state(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        # Prime in-memory state as if AgentOutput was already processed
        mapper._tool_id_to_message_id["tc-1"] = "msg-1"
        mapper._pending_tool_calls["msg-1"] = {"tc-1"}

        result = await mapper.map_event(make_tool_call_result("tc-1", "output"))

        assert result is not None
        assert result[0].tool_call is not None
        assert result[0].tool_call.tool_call_id == "tc-1"
        assert result[0].message_id == "msg-1"

    async def test_emits_message_end_when_last_pending_call_resolved(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._tool_id_to_message_id["tc-1"] = "msg-1"
        mapper._pending_tool_calls["msg-1"] = {"tc-1"}

        result = await mapper.map_event(make_tool_call_result("tc-1", "output"))

        assert result is not None
        assert len(result) == 2
        assert result[1].end is not None

    async def test_does_not_emit_message_end_with_pending_calls_remaining(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._tool_id_to_message_id["tc-1"] = "msg-1"
        mapper._tool_id_to_message_id["tc-2"] = "msg-1"
        mapper._pending_tool_calls["msg-1"] = {"tc-1", "tc-2"}

        result = await mapper.map_event(make_tool_call_result("tc-1", "r1"))

        assert result is not None
        assert len(result) == 1  # no message_end
        # tc-2 still pending
        assert "tc-2" in mapper._pending_tool_calls.get("msg-1", set())

    async def test_returns_none_for_unknown_tool_id(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        result = await mapper.map_event(make_tool_call_result("unknown", "x"))
        assert result is None

    async def test_cleans_up_in_memory_state_after_resolution(self):
        mapper = UiPathChatMessagesMapper("test-runtime")
        mapper._tool_id_to_message_id["tc-1"] = "msg-1"
        mapper._pending_tool_calls["msg-1"] = {"tc-1"}

        await mapper.map_event(make_tool_call_result("tc-1", "output"))

        assert "tc-1" not in mapper._tool_id_to_message_id
        assert "msg-1" not in mapper._pending_tool_calls


# ── TestStoragePersistenceAcrossInstances (suspend/resume simulation) ─────────


class TestStoragePersistenceAcrossInstances:
    """Verify that a fresh mapper instance can resolve tool_ids written by a
    previous instance — the key scenario during workflow suspend/resume."""

    async def test_second_mapper_resolves_tool_id_from_storage(self):
        # Shared in-memory store keyed by (runtime_id, namespace, key)
        store: dict[tuple[str, str, str], object] = {}

        async def get_value(runtime_id, namespace, key):
            return store.get((runtime_id, namespace, key))

        async def set_value(runtime_id, namespace, key, value):
            store[(runtime_id, namespace, key)] = value

        storage1 = AsyncMock()
        storage1.get_value.side_effect = get_value
        storage1.set_value.side_effect = set_value

        storage2 = AsyncMock()
        storage2.get_value.side_effect = get_value
        storage2.set_value.side_effect = set_value

        # --- First execution: agent emits a tool call then workflow suspends ---
        mapper1 = UiPathChatMessagesMapper("run-001", storage=storage1)
        mapper1._current_message_id = "msg-abc"

        ts = make_tool_selection("tc-xyz", "search_tool", {"query": "foo"})
        await mapper1.map_event(make_agent_output([ts]))

        # Verify the map was persisted
        saved = await storage1.get_value(
            "run-001",
            STORAGE_NAMESPACE_EVENT_MAPPER,
            STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
        )
        assert saved == {"tc-xyz": "msg-abc"}

        # --- Second execution (resume): fresh mapper processes the tool result ---
        mapper2 = UiPathChatMessagesMapper("run-001", storage=storage2)

        result = await mapper2.map_event(make_tool_call_result("tc-xyz", "found it"))

        assert result is not None
        assert len(result) == 2  # tool_call_end + message_end

        end_event = result[0]
        assert end_event.message_id == "msg-abc"
        assert end_event.tool_call is not None
        assert end_event.tool_call.tool_call_id == "tc-xyz"
        assert end_event.tool_call.end is not None
        assert end_event.tool_call.end.output == "found it"

        msg_end = result[1]
        assert msg_end.message_id == "msg-abc"
        assert msg_end.end is not None

        # Verify map was cleaned up after resolution
        remaining = await storage1.get_value(
            "run-001",
            STORAGE_NAMESPACE_EVENT_MAPPER,
            STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
        )
        assert remaining == {}

    async def test_multiple_tool_calls_resolved_across_instances(self):
        store: dict[tuple[str, str, str], object] = {}

        async def get_value(runtime_id, namespace, key):
            return store.get((runtime_id, namespace, key))

        async def set_value(runtime_id, namespace, key, value):
            store[(runtime_id, namespace, key)] = value

        def make_storage():
            s = AsyncMock()
            s.get_value.side_effect = get_value
            s.set_value.side_effect = set_value
            return s

        # First execution: two tool calls written
        mapper1 = UiPathChatMessagesMapper("run-002", storage=make_storage())
        mapper1._current_message_id = "msg-multi"
        ts1 = make_tool_selection("tc-a", "tool_a")
        ts2 = make_tool_selection("tc-b", "tool_b")
        await mapper1.map_event(make_agent_output([ts1, ts2]))

        # Resume: first tool result — message still open
        mapper2 = UiPathChatMessagesMapper("run-002", storage=make_storage())
        result1 = await mapper2.map_event(make_tool_call_result("tc-a", "result_a"))
        assert result1 is not None
        assert len(result1) == 1  # tool_call_end only
        assert result1[0].end is None

        # Resume: second tool result — message now closed
        mapper3 = UiPathChatMessagesMapper("run-002", storage=make_storage())
        result2 = await mapper3.map_event(make_tool_call_result("tc-b", "result_b"))
        assert result2 is not None
        assert len(result2) == 2  # tool_call_end + message_end
        assert result2[1].end is not None
