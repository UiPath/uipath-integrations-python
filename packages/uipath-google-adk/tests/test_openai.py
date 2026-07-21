"""Tests for chat/openai.py — request/response conversion and gateway calls."""

import json
from typing import Any

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel

from uipath_google_adk.chat.openai import (
    UiPathOpenAI,
    _content_to_messages,
    _function_declaration_to_tool,
    _message_to_llm_response,
    _parse_complete_response,
    _parts_to_content,
    _safe_json,
    _schema_to_dict,
    _to_openai_role,
    _to_response_format,
)


class TestSmallHelpers:
    def test_to_openai_role_maps_model_to_assistant(self):
        assert _to_openai_role("model") == "assistant"
        assert _to_openai_role("assistant") == "assistant"
        assert _to_openai_role("user") == "user"
        assert _to_openai_role(None) == "user"

    def test_safe_json_passes_through_string_and_serializes_dict(self):
        assert _safe_json("already") == "already"
        assert _safe_json({"a": 1}) == '{"a": 1}'

    def test_safe_json_falls_back_to_str_on_unserializable(self):
        obj = object()
        assert _safe_json(obj) == str(obj)


class TestSchemaToDict:
    def test_lowercases_type_and_recurses_into_items_and_properties(self):
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tags": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                )
            },
        )
        result = _schema_to_dict(schema)
        assert result["type"] == "object"
        assert result["properties"]["tags"]["type"] == "array"
        assert result["properties"]["tags"]["items"]["type"] == "string"


class TestContentToMessages:
    def test_user_text_content(self):
        content = types.Content(role="user", parts=[types.Part.from_text(text="hi")])
        msgs = _content_to_messages(content)
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_assistant_with_tool_call(self):
        content = types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(name="lookup", args={"q": "x"}),
            ],
        )
        msgs = _content_to_messages(content)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "lookup"
        assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {
            "q": "x"
        }

    def test_function_response_becomes_tool_message(self):
        part = types.Part.from_function_response(name="lookup", response={"result": 42})
        assert part.function_response is not None
        part.function_response.id = "call-1"
        content = types.Content(role="user", parts=[part])
        msgs = _content_to_messages(content)
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call-1"
        assert json.loads(msgs[0]["content"]) == {"result": 42}


class TestPartsToContent:
    def test_single_text_returns_string(self):
        parts = [types.Part.from_text(text="solo")]
        assert _parts_to_content(parts) == "solo"

    def test_inline_image_becomes_image_url_object(self):
        part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG")
        )
        result = _parts_to_content([types.Part.from_text(text="see"), part])
        assert isinstance(result, list)
        assert {"type": "text", "text": "see"} in result
        image_obj = [o for o in result if o["type"] == "image_url"][0]
        assert image_obj["image_url"]["url"].startswith("data:image/png;base64,")


class TestFunctionDeclarationToTool:
    def test_converts_declaration_with_required_params(self):
        decl = types.FunctionDeclaration(
            name="search",
            description="Search things",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"q": types.Schema(type=types.Type.STRING)},
                required=["q"],
            ),
        )
        tool = _function_declaration_to_tool(decl)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search"
        assert tool["function"]["parameters"]["properties"]["q"]["type"] == "string"
        assert tool["function"]["parameters"]["required"] == ["q"]


class TestToResponseFormat:
    def test_pydantic_model_produces_strict_json_schema(self):
        class Answer(BaseModel):
            value: int

        fmt = _to_response_format(Answer)
        assert fmt is not None
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Answer"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"]["additionalProperties"] is False

    def test_unsupported_type_returns_none(self):
        assert _to_response_format(123) is None


class TestMessageToLlmResponse:
    def test_maps_reasoning_text_and_tool_calls(self):
        message = {
            "reasoning_content": "thinking...",
            "content": "final answer",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "f", "arguments": '{"x": 1}'},
                }
            ],
        }
        resp = _message_to_llm_response(message, model_version="gpt-x")
        assert resp.content is not None and resp.content.parts is not None
        parts = resp.content.parts
        assert parts[0].thought is True and parts[0].text == "thinking..."
        assert parts[1].text == "final answer"
        assert parts[2].function_call is not None
        assert parts[2].function_call.name == "f"
        assert parts[2].function_call.id == "c1"
        assert resp.model_version == "gpt-x"


class TestParseCompleteResponse:
    def test_parses_message_finish_reason_and_usage(self):
        data = {
            "model": "gpt-4.1",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            },
        }
        resp = _parse_complete_response(data)
        assert resp.content is not None and resp.content.parts is not None
        assert resp.content.parts[0].text == "hello"
        assert resp.finish_reason == types.FinishReason.STOP
        assert resp.usage_metadata is not None
        assert resp.usage_metadata.total_token_count == 8

    def test_no_choices_raises(self):
        with pytest.raises(ValueError, match="No choices"):
            _parse_complete_response({"choices": []})


class TestBuildRequestBody:
    def test_includes_system_tools_response_format_and_params(self):
        llm = UiPathOpenAI(model="gpt-4.1")

        class Out(BaseModel):
            answer: str

        config = types.GenerateContentConfig(
            system_instruction="be brief",
            temperature=0.5,
            max_output_tokens=100,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="t",
                            parameters=types.Schema(type=types.Type.OBJECT),
                        )
                    ]
                )
            ],
            response_schema=Out,
        )
        request = LlmRequest(
            model="gpt-4.1",
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text="hi")])
            ],
            config=config,
        )
        body = llm._build_request_body(request, "gpt-4.1", stream=False)
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1]["content"] == "hi"
        assert body["tools"][0]["function"]["name"] == "t"
        assert body["response_format"]["type"] == "json_schema"
        assert body["temperature"] == 0.5
        # max_output_tokens is remapped to max_completion_tokens
        assert body["max_completion_tokens"] == 100
        assert "stream" not in body

    def test_stream_adds_stream_options(self):
        llm = UiPathOpenAI(model="gpt-4.1")
        request = LlmRequest(model="gpt-4.1", contents=[])
        body = llm._build_request_body(request, "gpt-4.1", stream=True)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}


class TestSupportedModels:
    def test_patterns_include_gpt_and_o_series(self):
        patterns = UiPathOpenAI.supported_models()
        import re

        assert any(re.fullmatch(p, "gpt-4.1-2025-04-14") for p in patterns)
        assert any(re.fullmatch(p, "o3-mini") for p in patterns)


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeAsyncClient:
    def __init__(self, response, captured):
        self._response = response
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["body"] = json
        return self._response


class TestGenerateContentAsync:
    @pytest.mark.asyncio
    async def test_non_streaming_posts_to_gateway_and_parses(self, monkeypatch):
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
        captured: dict[str, Any] = {}
        response = _FakeResponse(
            {"choices": [{"message": {"content": "world"}, "finish_reason": "stop"}]}
        )

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: _FakeAsyncClient(response, captured),
        )

        llm = UiPathOpenAI(model="gpt-4.1")
        request = LlmRequest(
            model="gpt-4.1",
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text="hello")])
            ],
        )
        results = [r async for r in llm.generate_content_async(request, stream=False)]
        assert len(results) == 1
        content = results[0].content
        assert content is not None and content.parts is not None
        assert content.parts[0].text == "world"
        assert captured["headers"]["X-UiPath-LlmGateway-ApiFlavor"] == (
            "OpenAiChatCompletions"
        )
        assert captured["headers"]["Authorization"] == "Bearer tok"


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeStreamingClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, headers=None, json=None):
        return _FakeStreamCtx(_FakeStreamResponse(self._lines))


class TestStreamRequest:
    @pytest.mark.asyncio
    async def test_streams_partial_text_and_aggregates_final(self, monkeypatch):
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")

        lines = [
            'data: {"model": "gpt-4.1", "choices": [{"delta": {"content": "Hel"}}]}',
            'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}',
            'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: _FakeStreamingClient(lines),
        )

        llm = UiPathOpenAI(model="gpt-4.1")
        request = LlmRequest(
            model="gpt-4.1",
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text="hi")])
            ],
        )
        results = [r async for r in llm.generate_content_async(request, stream=True)]

        def _first_text(resp: LlmResponse) -> str:
            assert resp.content is not None and resp.content.parts is not None
            return resp.content.parts[0].text or ""

        # partial chunks (Hel, lo) plus an aggregated final response
        partials = [r for r in results if r.partial]
        assert "".join(_first_text(p) for p in partials) == "Hello"
        final = [r for r in results if not r.partial][-1]
        assert _first_text(final) == "Hello"
        assert final.usage_metadata is not None
        assert final.usage_metadata.total_token_count == 3
