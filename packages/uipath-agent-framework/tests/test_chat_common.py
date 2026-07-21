"""Tests for chat gateway shared utilities."""

import pytest

from uipath_agent_framework.chat._common import (
    build_gateway_url,
    get_uipath_config,
    get_uipath_headers,
)


def test_get_uipath_config_returns_url_and_token(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "tok-123")
    assert get_uipath_config() == ("https://cloud.uipath.com/org/tenant", "tok-123")


def test_get_uipath_config_missing_url_raises(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    with pytest.raises(ValueError, match="UIPATH_URL"):
        get_uipath_config()


def test_get_uipath_config_missing_token_raises(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
    monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="UIPATH_ACCESS_TOKEN"):
        get_uipath_config()


def test_get_uipath_headers_includes_job_and_process_keys(monkeypatch):
    monkeypatch.setenv("UIPATH_JOB_KEY", "job-1")
    monkeypatch.setenv("UIPATH_PROCESS_KEY", "proc-1")
    headers = get_uipath_headers("tok-abc")
    assert headers["Authorization"] == "Bearer tok-abc"
    assert headers["X-UiPath-JobKey"] == "job-1"
    assert headers["X-UiPath-ProcessKey"] == "proc-1"


def test_get_uipath_headers_omits_optional_keys_when_unset(monkeypatch):
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    monkeypatch.delenv("UIPATH_PROCESS_KEY", raising=False)
    headers = get_uipath_headers("tok-abc")
    assert "X-UiPath-JobKey" not in headers
    assert "X-UiPath-ProcessKey" not in headers


def test_build_gateway_url_uses_explicit_url_and_strips_trailing_slash():
    url = build_gateway_url("openai", "gpt-4.1-mini", "https://cloud.uipath.com/org/")
    assert url.startswith("https://cloud.uipath.com/org/")
    assert "//" not in url[len("https://") :]


def test_build_gateway_url_missing_url_raises(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    with pytest.raises(ValueError, match="UIPATH_URL"):
        build_gateway_url("openai", "gpt-4.1-mini")
