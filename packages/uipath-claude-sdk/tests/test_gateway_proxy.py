"""Tests for the gateway proxy request/response transformations."""

from __future__ import annotations

import json

from uipath.platform.constants import HEADER_AGENTHUB_CONFIG, HEADER_JOB_KEY

from uipath_claude_sdk.runtime.gateway_proxy import (
    _build_headers,
    _coerce_to_string,
    _gateway_url,
    _transform_body,
)


def test_gateway_url():
    url = _gateway_url("https://cloud.uipath.com/org/tenant", "anthropic.claude-x")
    assert url == (
        "https://cloud.uipath.com/org/tenant/agenthub_/llm/raw/vendor/"
        "awsbedrock/model/anthropic.claude-x/completions"
    )


def test_build_headers(monkeypatch):
    monkeypatch.setenv("UIPATH_JOB_KEY", "job-123")
    headers = _build_headers("tok", is_streaming=True, agenthub_config="agentsruntime")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X-UiPath-Streaming-Enabled"] == "true"
    assert headers["X-UiPath-LlmGateway-ApiFlavor"] == "invoke"
    assert headers[HEADER_AGENTHUB_CONFIG] == "agentsruntime"
    assert headers[HEADER_JOB_KEY] == "job-123"


def test_transform_body_strips_unknown_fields_and_flattens():
    payload = {
        "model": "anthropic.claude-x",
        "stream": True,
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "sys a"},
            {"type": "text", "text": "sys b"},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": [{"type": "text", "text": "result text"}],
                    }
                ],
            }
        ],
        "unknown_field": 1,
    }
    body = json.loads(_transform_body(payload))
    assert "model" not in body
    assert "stream" not in body
    assert "unknown_field" not in body
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "sys a\n\nsys b"
    assert body["messages"][0]["content"][0]["content"] == "result text"


def test_coerce_to_string_passthrough():
    assert _coerce_to_string("plain") == "plain"
    assert _coerce_to_string(None) is None
