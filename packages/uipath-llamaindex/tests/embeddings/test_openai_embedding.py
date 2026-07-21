"""Tests for UiPathOpenAIEmbedding gateway configuration."""

from unittest.mock import patch

import pytest

from uipath_llamaindex.embeddings._openai import (
    OpenAIEmbeddingModel,
    UiPathOpenAIEmbedding,
)


def test_init_raises_when_uipath_url_missing(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    with pytest.raises(ValueError, match="UIPATH_URL environment variable is not set"):
        UiPathOpenAIEmbedding()


def test_init_maps_enum_model_and_builds_gateway_defaults(monkeypatch):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "secret-token")

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    # Avoid the real Azure init which performs a network capability lookup.
    with patch(
        "uipath_llamaindex.embeddings._openai.AzureOpenAIEmbedding.__init__",
        fake_init,
    ):
        UiPathOpenAIEmbedding(model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE)

    assert captured["model"] == "text-embedding-3-large"
    assert captured["deployment_name"] == "text-embedding-3-large"
    assert (
        captured["azure_endpoint"] == "https://cloud.uipath.com/org/tenant/llmgateway_/"
    )
    assert captured["api_key"] == "secret-token"
    assert captured["default_headers"]["X-UIPATH-STREAMING-ENABLED"] == "false"
