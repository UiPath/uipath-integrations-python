"""Upstream HTTP clients for the UiPath LLM Gateway.

``UiPathHttpxAsyncClient`` already supplies endpoint selection, the refreshing
auth pipeline, org and tenant routing headers, retries and the UiPath SSL
defaults, so the shim configures one per model and adds nothing.

Codex speaks the OpenAI Responses API, and the tenant's OpenAI models advertise
the ``OpenAiResponses`` flavor, so every upstream client is a passthrough
OpenAI/responses client. The gateway reads the model from the URL path and
selects responses-vs-chat from the ``x-uipath-llmgateway-apiflavor`` header.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from uipath.llm_client import UiPathHttpxAsyncClient
from uipath.llm_client.settings import (
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    UiPathBaseSettings,
)
from uipath.llm_client.settings.constants import ApiFlavor, VendorType

logger = logging.getLogger(__name__)

# The full gateway endpoint (incl. model + org/tenant) is baked into the
# frozen base_url, so the request path is just "/".
UPSTREAM_PATH = "/"
UPSTREAM_TIMEOUT = httpx.Timeout(timeout=900.0, connect=30.0)
API_VERSION = "2025-04-01-preview"


def build_api_config() -> UiPathAPIConfig:
    """Passthrough routing config for an OpenAI Responses-API call."""
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=VendorType.OPENAI,
        api_flavor=ApiFlavor.RESPONSES,
        api_version=API_VERSION,
        freeze_base_url=True,
    )


class UpstreamRegistry:
    """Lazily built, per-model upstream clients with a single shutdown point."""

    def __init__(self, settings: UiPathBaseSettings) -> None:
        self._settings = settings
        self._clients: dict[str, UiPathHttpxAsyncClient] = {}
        self._lock = asyncio.Lock()

    async def get(self, model: str) -> UiPathHttpxAsyncClient:
        """Return the client for a model, building it on first use."""
        client = self._clients.get(model)
        if client is not None:
            return client
        async with self._lock:
            client = self._clients.get(model)
            if client is None:
                client = UiPathHttpxAsyncClient(
                    model_name=model,
                    client_settings=self._settings,
                    api_config=build_api_config(),
                    timeout=UPSTREAM_TIMEOUT,
                    logger=logger,
                )
                self._clients[model] = client
                logger.debug(
                    "upstream client ready for %s at %s", model, client.base_url
                )
            return client

    async def aclose(self) -> None:
        """Close every client this registry created."""
        async with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                await client.aclose()
            except Exception:
                logger.debug("upstream client close failed", exc_info=True)


__all__ = ["UPSTREAM_PATH", "UPSTREAM_TIMEOUT", "UpstreamRegistry", "build_api_config"]
