"""Tests for UiPathVertex LLM and its gateway request rewriting."""

from unittest.mock import patch

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from uipath_llamaindex.llms.supported_models import GeminiModel
from uipath_llamaindex.llms.vertex import (
    UiPathVertex,
    _rewrite_request_for_gateway,
)


def test_rewrite_request_redirects_streaming_and_sets_flag():
    gateway = "https://cloud.uipath.com/gw/vertexai"
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1/models/x:streamGenerateContent",
        headers={"host": "generativelanguage.googleapis.com"},
        content=b"{}",
    )

    rewritten = _rewrite_request_for_gateway(request, gateway)

    assert str(rewritten.url) == gateway
    assert rewritten.headers["X-UiPath-Streaming-Enabled"] == "true"
    assert rewritten.headers["host"] == "cloud.uipath.com"


def test_rewrite_request_leaves_unrelated_requests_untouched():
    gateway = "https://cloud.uipath.com/gw/vertexai"
    request = httpx.Request("GET", "https://example.com/models")

    assert _rewrite_request_for_gateway(request, gateway) is request


def test_build_headers_includes_job_and_process_keys(monkeypatch):
    monkeypatch.setenv("UIPATH_JOB_KEY", "job-1")
    monkeypatch.setenv("UIPATH_PROCESS_KEY", "proc-1")

    headers = UiPathVertex._build_headers_static("my-token")

    assert headers["Authorization"] == "Bearer my-token"
    assert headers["X-UiPath-JobKey"] == "job-1"
    assert headers["X-UiPath-ProcessKey"] == "proc-1"


def test_build_base_url_raises_without_uipath_url(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    with pytest.raises(ValueError, match="UIPATH_URL environment variable is required"):
        UiPathVertex._build_base_url_static("gemini-2.5-flash")


def test_init_requires_org_id(monkeypatch):
    monkeypatch.delenv("UIPATH_ORGANIZATION_ID", raising=False)
    monkeypatch.setenv("UIPATH_TENANT_ID", "ten")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
    with pytest.raises(ValueError, match="UIPATH_ORGANIZATION_ID"):
        UiPathVertex()


def _make_vertex(monkeypatch):
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.setenv("UIPATH_TENANT_ID", "ten")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
    return UiPathVertex()


def test_stream_chat_yields_single_full_response(monkeypatch):
    llm = _make_vertex(monkeypatch)
    chat_response = ChatResponse(message=ChatMessage(role="assistant", content="hi"))

    with patch.object(UiPathVertex, "chat", return_value=chat_response):
        chunks = list(llm.stream_chat([ChatMessage(role="user", content="q")]))

    assert len(chunks) == 1
    assert chunks[0].delta == "hi"
    assert chunks[0].message.content == "hi"


def test_stream_complete_yields_single_full_response(monkeypatch):
    llm = _make_vertex(monkeypatch)
    chat_response = ChatResponse(message=ChatMessage(role="assistant", content="text"))

    with patch.object(UiPathVertex, "chat", return_value=chat_response):
        chunks = list(llm.stream_complete("prompt"))

    assert len(chunks) == 1
    assert chunks[0].text == "text"
    assert chunks[0].delta == "text"


def test_complete_delegates_to_chat(monkeypatch):
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.setenv("UIPATH_TENANT_ID", "ten")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")

    llm = UiPathVertex(model=GeminiModel.gemini_2_5_pro)
    assert llm.model == "gemini-2.5-pro"

    chat_response = ChatResponse(message=ChatMessage(role="assistant", content="Paris"))
    with patch.object(UiPathVertex, "chat", return_value=chat_response) as mocked_chat:
        result = llm.complete("What is the capital of France?")

    assert result.text == "Paris"
    sent_messages = mocked_chat.call_args.args[0]
    assert sent_messages[0].content == "What is the capital of France?"
