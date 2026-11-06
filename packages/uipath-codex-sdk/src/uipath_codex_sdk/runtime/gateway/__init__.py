"""Routing of Codex's LLM traffic through the UiPath tenant.

The runtime starts a :class:`CodexGatewayShim`, points the bundled ``codex``
binary at it via a custom ``model_provider`` (base_url = loopback, wire_api =
responses, env_key = a per-run secret), and the shim relays the Responses-API
calls to the UiPath LLM Gateway.
"""

from __future__ import annotations

import asyncio
import logging
import secrets

from opentelemetry.context import Context
from uipath.llm_client.settings import (
    UiPathBaseSettings,
    get_default_client_settings,
)

from .proxy import MCP_PATH, CodexGatewayProxy, MountedHandler
from .upstream import UpstreamRegistry

logger = logging.getLogger(__name__)

STOP_TIMEOUT = 10.0
_API_KEY_BYTES = 32

#: The model_provider id injected into Codex config.
PROVIDER_ID = "uipath"
#: Env var name Codex reads for the provider bearer token (its ``env_key``).
ENV_KEY_NAME = "UIPATH_CODEX_GATEWAY_KEY"


class CodexGatewayShim:
    """Local listener that routes Codex's LLM calls to the UiPath tenant.

    Args:
        default_model: Model used when a request omits one; also the model the
            thread is started with.
        settings: Optional pre-built LLM client settings (mainly for tests).
        mcp_handler: Optional endpoint mounted on ``/mcp`` next to the model
            routes (the durable tool host).
        trace_context: Context the ``LLM call`` spans are opened in.
    """

    def __init__(
        self,
        default_model: str | None = None,
        *,
        settings: UiPathBaseSettings | None = None,
        mcp_handler: MountedHandler | None = None,
        trace_context: Context | None = None,
    ) -> None:
        self._default_model = default_model
        self._settings = settings
        self._mcp_handler = mcp_handler
        self._trace_context = trace_context
        self._api_key = secrets.token_urlsafe(_API_KEY_BYTES)
        self._lock = asyncio.Lock()
        self._upstream: UpstreamRegistry | None = None
        self._proxy: CodexGatewayProxy | None = None

    async def start(self) -> None:
        """Build settings and bind the loopback listener. Idempotent."""
        async with self._lock:
            if self._proxy is not None:
                return
            settings = self._settings
            if settings is None:
                settings = await asyncio.to_thread(get_default_client_settings)
            upstream = UpstreamRegistry(settings)
            proxy = CodexGatewayProxy(
                upstream=upstream,
                api_key=self._api_key,
                default_model=self._default_model,
                mcp_handler=self._mcp_handler,
                trace_context=self._trace_context,
            )
            try:
                await proxy.start()
            except Exception:
                await upstream.aclose()
                raise
            self._settings = settings
            self._upstream = upstream
            self._proxy = proxy
            logger.info("codex gateway shim ready at %s", self.base_url)

    async def stop(self) -> None:
        async with self._lock:
            proxy, upstream = self._proxy, self._upstream
            self._proxy = None
            self._upstream = None
        if proxy is not None:
            try:
                await asyncio.wait_for(proxy.stop(), timeout=STOP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("codex gateway did not drain within %ss", STOP_TIMEOUT)
        if upstream is not None:
            await upstream.aclose()

    async def __aenter__(self) -> "CodexGatewayShim":
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    # --- surface handed to Codex -----------------------------------------

    @property
    def base_url(self) -> str:
        """Loopback base URL for the Codex model_provider (Codex adds /responses)."""
        return f"http://127.0.0.1:{self._require_proxy().port}/v1"

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def mcp_url(self) -> str:
        """URL of the mounted ``/mcp`` endpoint."""
        return f"http://127.0.0.1:{self._require_proxy().port}{MCP_PATH}"

    @property
    def inflight(self) -> int:
        """Model requests currently being relayed (0 when stopped)."""
        return self._proxy.inflight if self._proxy is not None else 0

    @property
    def last_activity(self) -> float:
        """``time.monotonic()`` of the last model request start or end."""
        return self._proxy.last_activity if self._proxy is not None else 0.0

    def config_overrides(self) -> tuple[str, ...]:
        """``--config`` overrides that define the UiPath model_provider.

        Retry posture (intentional): Codex's provider-side retries are disabled
        here (``request_max_retries``/``stream_max_retries`` = 0) because the
        loopback proxy forwards through ``UiPathHttpxAsyncClient``, whose own
        retry pipeline already handles transient upstream/gateway failures
        (e.g. 503s). Retrying at both layers would compound; the gateway client
        owns retries.
        """
        return (
            f'model_providers.{PROVIDER_ID}.name="UiPath LLM Gateway"',
            f'model_providers.{PROVIDER_ID}.base_url="{self.base_url}"',
            f'model_providers.{PROVIDER_ID}.wire_api="responses"',
            f'model_providers.{PROVIDER_ID}.env_key="{ENV_KEY_NAME}"',
            f"model_providers.{PROVIDER_ID}.request_max_retries=0",
            f"model_providers.{PROVIDER_ID}.stream_max_retries=0",
        )

    def env(self) -> dict[str, str]:
        """Env vars for the Codex subprocess (the provider bearer token)."""
        return {ENV_KEY_NAME: self._api_key}

    def _require_proxy(self) -> CodexGatewayProxy:
        if self._proxy is None:
            raise RuntimeError("CodexGatewayShim.start() has not completed")
        return self._proxy


__all__ = ["CodexGatewayShim", "CodexGatewayProxy", "PROVIDER_ID", "ENV_KEY_NAME"]
