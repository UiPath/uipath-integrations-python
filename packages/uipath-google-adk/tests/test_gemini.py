"""Tests for chat/gemini.py — URL rewriting transports and api_client."""

import httpx
from google.genai import Client

from uipath_google_adk.chat.gemini import (
    UiPathGemini,
    _rewrite_request_for_gateway,
)


class TestRewriteRequestForGateway:
    def test_generate_content_url_rewritten_to_gateway(self):
        gateway = "https://cloud.uipath.com/gw/generateContent"
        original = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1/models/x:generateContent",
            content=b"{}",
        )
        rewritten = _rewrite_request_for_gateway(original, gateway)
        assert rewritten.url.host == "cloud.uipath.com"
        assert str(rewritten.url).startswith(gateway)

    def test_streaming_sets_streaming_header_and_preserves_params(self):
        gateway = "https://cloud.uipath.com/gw/stream"
        original = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1/x:streamGenerateContent?alt=sse",
            content=b"{}",
        )
        rewritten = _rewrite_request_for_gateway(original, gateway)
        assert rewritten.headers["X-UiPath-Streaming-Enabled"] == "true"
        assert "alt=sse" in str(rewritten.url)

    def test_unrelated_request_passed_through_unchanged(self):
        original = httpx.Request("GET", "https://example.com/other")
        assert _rewrite_request_for_gateway(original, "https://gw") is original


class TestApiClient:
    def test_builds_gateway_client(self, monkeypatch):
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
        gemini = UiPathGemini(model="gemini-2.5-flash")
        client = gemini.api_client
        assert isinstance(client, Client)
