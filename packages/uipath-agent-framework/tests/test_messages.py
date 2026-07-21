"""Tests for AgentFrameworkChatMessagesMapper inbound/outbound mapping."""

from agent_framework import Content
from uipath.core.chat import (
    UiPathConversationContentPart,
    UiPathConversationMessage,
    UiPathInlineValue,
)

from uipath_agent_framework.runtime.messages import AgentFrameworkChatMessagesMapper


def _uipath_msg(role: str, text: str) -> UiPathConversationMessage:
    return UiPathConversationMessage(
        message_id=f"m-{role}",
        role=role,
        content_parts=[
            UiPathConversationContentPart(
                content_part_id="c1",
                mime_type="text/plain",
                data=UiPathInlineValue(inline=text),
            )
        ],
    )


def test_map_string_input_returned_as_is():
    mapper = AgentFrameworkChatMessagesMapper()
    assert mapper.map_messages_to_input("hello world") == "hello world"


def test_map_uipath_messages_extracts_last_user_text():
    mapper = AgentFrameworkChatMessagesMapper()
    messages = [_uipath_msg("user", "first"), _uipath_msg("user", "second")]
    assert mapper.map_messages_to_input(messages) == "second"


def test_map_dict_messages_parsed_as_uipath():
    mapper = AgentFrameworkChatMessagesMapper()
    messages = [_uipath_msg("user", "from dict").model_dump()]
    assert mapper.map_messages_to_input(messages) == "from dict"


def test_map_raw_dicts_fallback_uses_content_field():
    mapper = AgentFrameworkChatMessagesMapper()
    # Dicts that are not valid UiPath conversation messages fall back to raw.
    messages = [{"content": "raw one"}, {"content": "raw two"}]
    assert mapper.map_messages_to_input(messages) == "raw one\nraw two"


def test_map_list_of_strings_joined_with_newline():
    mapper = AgentFrameworkChatMessagesMapper()
    assert mapper.map_messages_to_input(["a", "b"]) == "a\nb"


def test_map_empty_input_returns_empty_string():
    mapper = AgentFrameworkChatMessagesMapper()
    assert mapper.map_messages_to_input([]) == ""


def test_extract_falls_back_to_last_message_when_no_user_role():
    mapper = AgentFrameworkChatMessagesMapper()
    messages = [_uipath_msg("assistant", "only assistant text")]
    assert mapper.map_messages_to_input(messages) == "only assistant text"


def test_text_content_emits_message_start_then_chunk():
    mapper = AgentFrameworkChatMessagesMapper()
    events = mapper.map_streaming_content(Content(type="text", text="hi"))
    # First a message start (with content_part start), then the chunk.
    assert events[0].start is not None
    chunk_part = events[1].content_part
    assert chunk_part is not None and chunk_part.chunk is not None
    assert chunk_part.chunk.data == "hi"
    # Second text chunk reuses the same open message, no new start.
    events2 = mapper.map_streaming_content(Content(type="text", text="!"))
    assert events2[0].start is None
    chunk_part2 = events2[0].content_part
    assert chunk_part2 is not None and chunk_part2.chunk is not None
    assert chunk_part2.chunk.data == "!"


def test_function_call_then_result_pairs_tool_events_and_closes_message():
    mapper = AgentFrameworkChatMessagesMapper()
    start_events = mapper.map_streaming_content(
        Content(type="function_call", name="lookup", call_id="c1", arguments={"q": "x"})
    )
    tool_start = [e for e in start_events if e.tool_call and e.tool_call.start][0]
    assert tool_start.tool_call is not None and tool_start.tool_call.start is not None
    assert tool_start.tool_call.start.tool_name == "lookup"
    assert tool_start.tool_call.start.input == {"q": "x"}

    result_events = mapper.map_streaming_content(
        Content(type="function_result", call_id="c1", result="done")
    )
    tool_end = [e for e in result_events if e.tool_call and e.tool_call.end][0]
    assert tool_end.tool_call is not None and tool_end.tool_call.end is not None
    assert tool_end.tool_call.end.output == "done"
    # Message closed once all pending tool calls resolved.
    assert any(e.end is not None for e in result_events)


def test_function_call_partial_chunk_without_name_is_skipped():
    mapper = AgentFrameworkChatMessagesMapper()
    events = mapper.map_streaming_content(
        Content(type="function_call", name="", call_id="c1", arguments={})
    )
    assert events == []


def test_close_message_emits_end_for_unresolved_tool_call():
    mapper = AgentFrameworkChatMessagesMapper()
    mapper.map_streaming_content(
        Content(type="function_call", name="pending", call_id="c9", arguments={})
    )
    events = mapper.close_message()
    tool_end = [e for e in events if e.tool_call and e.tool_call.end][0]
    assert tool_end.tool_call is not None and tool_end.tool_call.end is not None
    assert tool_end.tool_call.end.output == {}
