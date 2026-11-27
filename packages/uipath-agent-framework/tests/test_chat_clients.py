"""Tests for UiPath gateway chat client transports and helpers."""

import importlib.util
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from uipath_agent_framework.chat.anthropic import (
    _AsyncUrlRewriteTransport as AnthropicTransport,
)
from uipath_agent_framework.chat.anthropic import _check_anthropic_dependency
from uipath_agent_framework.chat.openai import (
    UiPathOpenAIChatClient,
)
from uipath_agent_framework.chat.openai import (
    _AsyncUrlRewriteTransport as OpenAITransport,
)

_real_find_spec = importlib.util.find_spec

GATEWAY = "https://cloud.uipath.com/org/tenant/llmgateway/openai/chat/completions"


async def _capture(transport, request):
    """Invoke handle_async_request with the parent send mocked out."""
    captured = {}

    async def fake_super(self, req):
        captured["request"] = req
        return httpx.Response(200)

    with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", fake_super):
        await transport.handle_async_request(request)
    return captured["request"]


async def test_openai_transport_rewrites_chat_completions_url():
    transport = OpenAITransport(GATEWAY)
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=b'{"model":"gpt-4"}',
    )
    out = await _capture(transport, request)
    assert str(out.url) == GATEWAY
    assert out.headers["host"] == "cloud.uipath.com"


async def test_openai_transport_sets_streaming_header_when_streaming():
    transport = OpenAITransport(GATEWAY)
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=b'{"stream": true}',
    )
    out = await _capture(transport, request)
    assert out.headers["X-UiPath-Streaming-Enabled"] == "true"


async def test_openai_transport_passes_through_other_urls():
    transport = OpenAITransport(GATEWAY)
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    out = await _capture(transport, request)
    assert str(out.url) == "https://api.openai.com/v1/models"


async def test_anthropic_transport_converts_body_to_bedrock_invoke_format():
    transport = AnthropicTransport(GATEWAY)
    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps({"model": "claude", "stream": True}).encode(),
    )
    out = await _capture(transport, request)
    body = json.loads(out.content)
    assert "model" not in body
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert out.headers["X-UiPath-Streaming-Enabled"] == "true"


def test_check_anthropic_dependency_raises_when_missing(monkeypatch):
    # Force the "not installed" path so the check is deterministic regardless
    # of whether the anthropic extra happens to be present in the environment.
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "anthropic" else _real_find_spec(name),
    )
    with pytest.raises(ImportError, match="anthropic"):
        _check_anthropic_dependency()


def test_openai_client_normalizes_plain_callable_tools(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    client = UiPathOpenAIChatClient(model="gpt-4.1-mini")

    def my_tool(x: str) -> str:
        """Echo a value."""
        return x

    captured = {}

    def fake_base(self, tools):
        captured["tools"] = tools
        return {}

    with patch.object(
        UiPathOpenAIChatClient.__bases__[0],
        "_prepare_tools_for_openai",
        fake_base,
    ):
        client._prepare_tools_for_openai([my_tool])

    # The plain function was wrapped as a FunctionTool before delegation.
    from agent_framework import FunctionTool

    assert isinstance(captured["tools"][0], FunctionTool)


def test_openai_client_uses_chat_completions_client_with_model(monkeypatch):
    """The gateway speaks Chat Completions; the model is passed as ``model``."""
    from agent_framework.openai import OpenAIChatCompletionClient

    from uipath_agent_framework.runtime.schema import _get_model_name

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    client = UiPathOpenAIChatClient(model="gpt-4.1-mini")

    assert isinstance(client, OpenAIChatCompletionClient)
    assert client.model == "gpt-4.1-mini"
    assert _get_model_name(client.as_agent(name="a")) == "gpt-4.1-mini"


async def test_openai_client_sends_chat_completion_to_gateway(monkeypatch):
    """A real request goes out as a Chat Completion to the gateway URL."""
    from agent_framework import Message

    from uipath_agent_framework.chat._common import build_gateway_url

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    client = UiPathOpenAIChatClient(model="gpt-4.1-mini")

    sent: list[httpx.Request] = []

    async def fake_send(self, request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
            request=request,
        )

    with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", fake_send):
        response = await client.get_response([Message(role="user", contents=["hello"])])

    assert response.text == "hi"
    assert len(sent) == 1
    assert str(sent[0].url) == build_gateway_url("openai", "gpt-4.1-mini")
    assert json.loads(sent[0].content)["model"] == "gpt-4.1-mini"
    assert sent[0].headers["X-UiPath-LlmGateway-ApiFlavor"] == "OpenAiChatCompletions"


@pytest.mark.skipif(
    importlib.util.find_spec("agent_framework_anthropic") is None,
    reason="anthropic extra not installed",
)
def test_anthropic_client_passes_model(monkeypatch):
    from uipath_agent_framework.chat.anthropic import UiPathAnthropicClient

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    client: Any = UiPathAnthropicClient(
        model="anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    assert client.model == "anthropic.claude-haiku-4-5-20251001-v1:0"
