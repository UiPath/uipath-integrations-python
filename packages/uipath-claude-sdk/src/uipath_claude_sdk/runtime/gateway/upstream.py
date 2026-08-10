"""Upstream HTTP clients for the UiPath LLM Gateway.

``UiPathHttpxAsyncClient`` already supplies endpoint selection, the refreshing
auth pipeline, org and tenant routing headers, retries and the UiPath SSL
defaults, so the shim configures one per resolved model and adds nothing.
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

from .catalog import ResolvedModel

logger = logging.getLogger(__name__)

UPSTREAM_PATH = "/"
UPSTREAM_TIMEOUT = httpx.Timeout(timeout=900.0, connect=30.0)


def build_api_config(resolved: ResolvedModel) -> UiPathAPIConfig:
    """Build the passthrough routing configuration for a resolved model."""
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=resolved.vendor_type,
        api_flavor=resolved.api_flavor,
        api_version=resolved.api_version,
        freeze_base_url=True,
    )


class UpstreamRegistry:
    """Lazily built, per-model upstream clients with a single shutdown point."""

    def __init__(self, settings: UiPathBaseSettings) -> None:
        self._settings = settings
        self._clients: dict[str, UiPathHttpxAsyncClient] = {}
        self._lock = asyncio.Lock()

    async def get(self, resolved: ResolvedModel) -> UiPathHttpxAsyncClient:
        """Return the client for a resolved model, building it on first use."""
        client = self._clients.get(resolved.model_id)
        if client is not None:
            return client
        async with self._lock:
            client = self._clients.get(resolved.model_id)
            if client is None:
                client = self._build(resolved)
                self._clients[resolved.model_id] = client
                logger.debug(
                    "upstream client ready for %s (%s/%s) at %s",
                    resolved.model_id,
                    resolved.vendor_type,
                    resolved.api_flavor,
                    client.base_url,
                )
            return client

    def _build(self, resolved: ResolvedModel) -> UiPathHttpxAsyncClient:
        return UiPathHttpxAsyncClient(
            model_name=resolved.wire_name,
            client_settings=self._settings,
            api_config=build_api_config(resolved),
            timeout=UPSTREAM_TIMEOUT,
            logger=logger,
        )

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
