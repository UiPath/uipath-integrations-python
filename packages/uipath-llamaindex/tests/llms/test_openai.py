"""Tests for UiPathOpenAI LLM and its URL-rewriting transports."""

from unittest.mock import patch

import httpx
import pytest

from uipath_llamaindex.llms._openai import (
    UiPathOpenAI,
    _UiPathAsyncURLRewriteTransport,
    _UiPathSyncURLRewriteTransport,
)
from uipath_llamaindex.llms.supported_models import OpenAIModel


def test_sync_transport_rewrites_deployment_url_and_keeps_query():
    transport = _UiPathSyncURLRewriteTransport()
    request = httpx.Request(
        "POST",
        "https://gw.example.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
    )
    with patch.object(
        httpx.HTTPTransport, "handle_request", return_value=httpx.Response(200)
    ) as mocked:
        transport.handle_request(request)

    forwarded = mocked.call_args.args[0]
    assert (
        str(forwarded.url)
        == "https://gw.example.com/completions?api-version=2024-10-21"
    )


@pytest.mark.asyncio
async def test_async_transport_rewrites_deployment_url_without_query():
    transport = _UiPathAsyncURLRewriteTransport()
    request = httpx.Request(
        "POST",
        "https://gw.example.com/openai/deployments/gpt-4o/chat/completions",
    )

    async def fake_handle(self, req):
        return httpx.Response(200)

    with patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request", new=fake_handle
    ):
        # Re-implement capture since new= replaces the method; assert via request mutation
        await transport.handle_async_request(request)

    assert str(request.url) == "https://gw.example.com/completions"


def test_init_raises_when_uipath_url_missing(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "token")
    with pytest.raises(ValueError, match="UIPATH_URL environment variable is not set"):
        UiPathOpenAI()


def test_init_builds_gateway_endpoint_from_enum_model(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "token")

    llm = UiPathOpenAI(model=OpenAIModel.GPT_4O_2024_08_06)

    assert llm.model == "gpt-4o-2024-08-06"
    # trailing slash stripped and vendor endpoint appended, no duplicate /completions
    assert llm.azure_endpoint.startswith("https://cloud.uipath.com/org/tenant/")
    assert "openai" in llm.azure_endpoint
    assert not llm.azure_endpoint.endswith("/completions")


def test_default_headers_include_agenthub_config_at_design_time(monkeypatch, tmp_path):
    from uipath.platform.common import UiPathConfig

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "token")
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    monkeypatch.delenv("UIPATH_PROJECT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    UiPathConfig.reset()

    llm = UiPathOpenAI(model=OpenAIModel.GPT_4O_2024_08_06)
    assert (
        llm.default_headers.get("x-uipath-agenthub-config") == "codedagentsplayground"
    )
    UiPathConfig.reset()


def test_default_headers_omit_agenthub_config_when_deployed(monkeypatch, tmp_path):
    from uipath.platform.common import UiPathConfig

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "token")
    monkeypatch.setenv("UIPATH_JOB_KEY", "deployed-job")
    monkeypatch.delenv("UIPATH_PROJECT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    UiPathConfig.reset()

    llm = UiPathOpenAI(model=OpenAIModel.GPT_4O_2024_08_06)
    assert "x-uipath-agenthub-config" not in (llm.default_headers or {})
    UiPathConfig.reset()
