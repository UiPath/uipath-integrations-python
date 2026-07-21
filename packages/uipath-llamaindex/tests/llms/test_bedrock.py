"""Tests for the AWS Bedrock passthrough client and UiPath Bedrock LLMs."""

from types import SimpleNamespace

import pytest

from uipath_llamaindex.llms.bedrock import (
    AwsBedrockCompletionsPassthroughClient,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
)
from uipath_llamaindex.llms.supported_models import BedrockModel


def _make_request(url: str) -> SimpleNamespace:
    return SimpleNamespace(url=url, headers={})


def test_build_base_url_raises_without_uipath_url(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    client = AwsBedrockCompletionsPassthroughClient("model", "token", "invoke")
    with pytest.raises(ValueError, match="UIPATH_URL environment variable is required"):
        client._build_base_url()


def test_build_base_url_appends_vendor_endpoint(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    client = AwsBedrockCompletionsPassthroughClient("my-model", "token", "invoke")

    url = client._build_base_url()

    assert url.startswith("https://cloud.uipath.com/org/tenant/")
    assert "awsbedrock" in url
    assert "my-model" in url


def test_modify_request_marks_streaming_and_sets_headers(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
    monkeypatch.setenv("UIPATH_JOB_KEY", "job-9")
    monkeypatch.delenv("UIPATH_PROCESS_KEY", raising=False)
    client = AwsBedrockCompletionsPassthroughClient("m", "tok", "converse")

    streaming_req = _make_request("https://aws/model/m/converse-stream")
    client._modify_request(streaming_req)

    assert streaming_req.url == client._build_base_url()
    assert streaming_req.headers["X-UiPath-Streaming-Enabled"] == "true"
    assert streaming_req.headers["Authorization"] == "Bearer tok"
    assert streaming_req.headers["X-UiPath-LlmGateway-ApiFlavor"] == "converse"
    assert streaming_req.headers["X-UiPath-JobKey"] == "job-9"
    assert "X-UiPath-ProcessKey" not in streaming_req.headers


def test_modify_request_marks_non_streaming(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
    client = AwsBedrockCompletionsPassthroughClient("m", "tok", "invoke")

    req = _make_request("https://aws/model/m/invoke")
    client._modify_request(req)

    assert req.headers["X-UiPath-Streaming-Enabled"] == "false"


def test_chat_bedrock_requires_token(monkeypatch):
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.setenv("UIPATH_TENANT_ID", "ten")
    monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="UIPATH_ACCESS_TOKEN"):
        UiPathChatBedrock()


def test_chat_bedrock_converse_constructs_with_default_model(monkeypatch):
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.setenv("UIPATH_TENANT_ID", "ten")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")

    llm = UiPathChatBedrockConverse()

    assert llm.model == BedrockModel.anthropic_claude_haiku_4_5
