"""Tests for runtime/messages.py — GoogleADKChatMessagesMapper."""

from google.adk.events.event import Event
from google.genai import types
from uipath.core.chat import (
    UiPathConversationContentPart,
    UiPathConversationMessage,
    UiPathInlineValue,
)
from uipath.core.chat.tool import (
    UiPathConversationToolCall,
    UiPathConversationToolCallResult,
)

from uipath_google_adk.runtime.messages import GoogleADKChatMessagesMapper


def _text_part(text: str) -> UiPathConversationContentPart:
    return UiPathConversationContentPart(
        mime_type="text/plain", data=UiPathInlineValue(inline=text)
    )


def _user_msg(text: str) -> UiPathConversationMessage:
    return UiPathConversationMessage(role="user", content_parts=[_text_part(text)])


class TestMapMessages:
    def test_passthrough_content_objects(self):
        mapper = GoogleADKChatMessagesMapper()
        content = types.Content(role="user", parts=[types.Part(text="hi")])
        assert mapper.map_messages([content]) == [content]

    def test_empty_returns_empty(self):
        assert GoogleADKChatMessagesMapper().map_messages([]) == []

    def test_uipath_conversation_messages(self):
        mapper = GoogleADKChatMessagesMapper()
        contents = mapper.map_messages([_user_msg("hello there")])
        assert len(contents) == 1
        assert contents[0].role == "user"
        parts = contents[0].parts
        assert parts is not None
        assert parts[0].text == "hello there"

    def test_list_of_dicts_parsed(self):
        mapper = GoogleADKChatMessagesMapper()
        raw = _user_msg("from dict").model_dump(by_alias=True)
        contents = mapper.map_messages([raw])
        parts = contents[0].parts
        assert parts is not None
        assert parts[0].text == "from dict"

    def test_fallback_raw_text(self):
        mapper = GoogleADKChatMessagesMapper()
        contents = mapper.map_messages(["plain string"])
        assert contents[0].role == "user"
        parts = contents[0].parts
        assert parts is not None
        assert parts[0].text == "plain string"


class TestMapMessagesInternalAssistant:
    def test_assistant_tool_call_and_response(self):
        mapper = GoogleADKChatMessagesMapper()
        tc = UiPathConversationToolCall(
            name="lookup",
            input={"q": "x"},
            tool_call_id="call-1",
            created_at="t",
            updated_at="t",
            result=UiPathConversationToolCallResult(output={"answer": 1}),
        )
        assistant = UiPathConversationMessage(
            role="assistant",
            content_parts=[_text_part("using tool")],
            tool_calls=[tc],
        )
        contents = mapper.map_messages([assistant])
        # model content (text + function_call) then a user content with the response
        model_content = contents[0]
        assert model_content.role == "model"
        model_parts = model_content.parts
        assert model_parts is not None
        assert model_parts[0].text == "using tool"
        function_call = model_parts[1].function_call
        assert function_call is not None
        assert function_call.name == "lookup"
        response_parts = contents[1].parts
        assert response_parts is not None
        function_response = response_parts[0].function_response
        assert function_response is not None
        assert function_response.response == {"answer": 1}


class TestNormalizeToolOutput:
    def test_none_becomes_empty_dict(self):
        assert GoogleADKChatMessagesMapper._normalize_tool_output(None) == {}

    def test_json_string_parsed_to_dict(self):
        out = GoogleADKChatMessagesMapper._normalize_tool_output('{"a": 1}')
        assert out == {"a": 1}

    def test_plain_string_wrapped_in_result(self):
        out = GoogleADKChatMessagesMapper._normalize_tool_output("hi")
        assert out == {"result": "hi"}

    def test_non_string_non_dict_wrapped(self):
        out = GoogleADKChatMessagesMapper._normalize_tool_output(42)
        assert out == {"result": "42"}


class TestMapEvent:
    def test_partial_text_starts_message_and_emits_chunk(self):
        mapper = GoogleADKChatMessagesMapper()
        event = Event(
            author="asst",
            content=types.Content(role="model", parts=[types.Part(text="Hel")]),
            partial=True,
        )
        events = mapper.map_event(event)
        # first the message start, then the content chunk
        assert events[0].start is not None
        content_part = events[1].content_part
        assert content_part is not None and content_part.chunk is not None
        assert content_part.chunk.data == "Hel"

    def test_user_author_ignored(self):
        mapper = GoogleADKChatMessagesMapper()
        event = Event(
            author="user",
            content=types.Content(role="user", parts=[types.Part(text="hi")]),
        )
        assert mapper.map_event(event) == []

    def test_function_call_emits_tool_call_start(self):
        mapper = GoogleADKChatMessagesMapper()
        part = types.Part.from_function_call(name="search", args={"q": "x"})
        assert part.function_call is not None
        part.function_call.id = "fc-1"
        event = Event(
            author="asst",
            content=types.Content(role="model", parts=[part]),
        )
        events = mapper.map_event(event)
        tool_events = [e for e in events if e.tool_call is not None]
        tool_call = tool_events[0].tool_call
        assert tool_call is not None and tool_call.start is not None
        assert tool_call.start.tool_name == "search"
        assert "fc-1" in mapper._pending_tool_calls

    def test_final_response_closes_message(self):
        mapper = GoogleADKChatMessagesMapper()
        # start a message via a partial chunk first
        mapper.map_event(
            Event(
                author="asst",
                content=types.Content(role="model", parts=[types.Part(text="Hi")]),
                partial=True,
            )
        )
        final = Event(
            author="asst",
            content=types.Content(role="model", parts=[types.Part(text="Hi done")]),
        )
        events = mapper.map_event(final)
        assert any(e.end is not None for e in events)
        assert mapper._message_started is False


class TestCloseMessage:
    def test_close_open_message(self):
        mapper = GoogleADKChatMessagesMapper()
        mapper._message_started = True
        mapper._current_message_id = "m1"
        events = mapper.close_message()
        assert events[0].end is not None
        assert mapper._message_started is False

    def test_close_noop_when_no_open_message(self):
        assert GoogleADKChatMessagesMapper().close_message() == []
