"""Tests for chat/_common.py UiPath configuration helpers."""

import pytest

from uipath_google_adk.chat._common import (
    build_gateway_url,
    get_uipath_config,
    get_uipath_headers,
)


class TestGetUiPathConfig:
    def test_returns_url_and_token(self, monkeypatch):
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "secret-token")
        assert get_uipath_config() == (
            "https://cloud.uipath.com/org/tenant",
            "secret-token",
        )

    def test_missing_url_raises(self, monkeypatch):
        monkeypatch.delenv("UIPATH_URL", raising=False)
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "t")
        with pytest.raises(ValueError, match="UIPATH_URL"):
            get_uipath_config()

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.setenv("UIPATH_URL", "https://x")
        monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)
        with pytest.raises(ValueError, match="UIPATH_ACCESS_TOKEN"):
            get_uipath_config()


class TestGetUiPathHeaders:
    def test_authorization_only_without_optional_env(self, monkeypatch):
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
        monkeypatch.delenv("UIPATH_PROCESS_KEY", raising=False)
        headers = get_uipath_headers("abc")
        assert headers == {"Authorization": "Bearer abc"}

    def test_includes_job_and_process_keys_when_set(self, monkeypatch):
        monkeypatch.setenv("UIPATH_JOB_KEY", "job-1")
        monkeypatch.setenv("UIPATH_PROCESS_KEY", "proc-1")
        headers = get_uipath_headers("abc")
        assert headers["X-UiPath-JobKey"] == "job-1"
        assert headers["X-UiPath-ProcessKey"] == "proc-1"


class TestBuildGatewayUrl:
    def test_builds_url_with_explicit_uipath_url(self):
        url = build_gateway_url("openai", "gpt-4.1", "https://cloud.uipath.com/org/")
        # vendor/model are substituted into the endpoint template, trailing slash stripped
        assert url.startswith("https://cloud.uipath.com/org/")
        assert "openai" in url
        assert "gpt-4.1" in url
        assert "//org//" not in url

    def test_missing_url_raises(self, monkeypatch):
        monkeypatch.delenv("UIPATH_URL", raising=False)
        with pytest.raises(ValueError, match="UIPATH_URL"):
            build_gateway_url("openai", "gpt-4.1")
