"""Tests for the UiPath OpenAI chat client and URL rewriting."""

import httpx
import pytest

from uipath_openai_agents.chat.openai import (
    UiPathChatOpenAI,
    _rewrite_openai_url,
)


def test_rewrite_responses_url_to_completions() -> None:
    """A /responses URL is rewritten to the gateway /completions endpoint."""
    rewritten = _rewrite_openai_url(
        "https://host/llm/openai/gpt-4o/responses", httpx.QueryParams()
    )

    assert str(rewritten) == "https://host/llm/openai/gpt-4o/completions"


def test_rewrite_preserves_query_params() -> None:
    """Query params (e.g. api-version) are preserved on the rewritten URL."""
    params = httpx.QueryParams({"api-version": "2024-12-01-preview"})
    rewritten = _rewrite_openai_url("https://host/base?api-version=x", params)

    assert rewritten is not None
    assert rewritten.path == "/base/completions"
    assert rewritten.params["api-version"] == "2024-12-01-preview"


@pytest.fixture
def chat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    for var in ("UIPATH_ORGANIZATION_ID", "UIPATH_TENANT_ID", "UIPATH_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_missing_org_id_raises_value_error(chat_env: None) -> None:
    """Construction fails clearly when no organization id is available."""
    with pytest.raises(ValueError, match="UIPATH_ORGANIZATION_ID"):
        UiPathChatOpenAI(token="t", tenant_id="tn")


def test_build_headers_includes_optional_and_respects_overrides(
    chat_env: None,
) -> None:
    """Headers carry flavor/auth plus optional agenthub/byo, with extra_headers winning."""
    client = UiPathChatOpenAI(
        token="my-token",
        org_id="org",
        tenant_id="tn",
        api_flavor="responses",
        agenthub_config="cfg",
        byo_connection_id="conn-1",
        extra_headers={"X-UiPath-LlmGateway-ApiFlavor": "chat-completions"},
    )

    headers = client._build_headers()

    assert headers["Authorization"] == "Bearer my-token"
    assert headers["X-UiPath-AgentHub-Config"] == "cfg"
    assert headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "conn-1"
    # extra_headers override the default flavor
    assert headers["X-UiPath-LlmGateway-ApiFlavor"] == "chat-completions"


def test_missing_uipath_url_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base URL construction requires the UIPATH_URL environment variable."""
    monkeypatch.delenv("UIPATH_URL", raising=False)

    with pytest.raises(ValueError, match="UIPATH_URL"):
        UiPathChatOpenAI(token="t", org_id="org", tenant_id="tn")
