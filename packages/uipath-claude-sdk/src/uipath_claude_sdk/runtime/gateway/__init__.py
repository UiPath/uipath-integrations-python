"""Explicit opt-in routing of Claude Code LLM traffic through the UiPath tenant.

The runtime starts a ``GatewayShim`` only when the developer passes
``uipath_llm=UiPathModel(...)``. Without it nothing is injected and the Claude
Agent SDK talks to Anthropic on its own terms.
"""

from __future__ import annotations

import asyncio
import logging
import secrets

from uipath.llm_client.settings import (
    UiPathBaseSettings,
    get_default_client_settings,
)

from .catalog import (
    ModelCatalog,
    ResolvedModel,
    ResolvedModelSet,
    UiPathModelSpec,
)
from .env import build_llm_env
from .errors import (
    GatewayShimError,
    ModelNotInCatalogError,
    UnsupportedApiFlavorError,
    to_anthropic_error,
)
from .records import GatewayCallRecord, GatewayCallSink, TraceHeaderSource
from .router import GatewayRouter
from .strategies import (
    AnthropicMessagesStrategy,
    BedrockInvokeStrategy,
    GatewayStrategy,
    select_strategy,
)
from .upstream import UpstreamRegistry

logger = logging.getLogger(__name__)

STOP_TIMEOUT = 10.0

_API_KEY_BYTES = 32


class GatewayShim:
    """Local listener that routes Claude Code's LLM calls to the UiPath tenant.

    Args:
        llm: The routing descriptor naming the model the developer asked for.
        on_call: Optional sink invoked once per upstream call.
        trace_headers: Optional source of trace context headers, letting the
            gateway record the model span under the turn that caused it.
        settings: Optional pre-built LLM client settings, mainly for tests.
    """

    def __init__(
        self,
        llm: UiPathModelSpec,
        *,
        on_call: GatewayCallSink | None = None,
        trace_headers: TraceHeaderSource | None = None,
        settings: UiPathBaseSettings | None = None,
    ) -> None:
        self._llm = llm
        self._on_call = on_call
        self._trace_headers = trace_headers
        self._settings = settings
        self._api_key = secrets.token_urlsafe(_API_KEY_BYTES)
        self._lock = asyncio.Lock()
        self._catalog: ModelCatalog | None = None
        self._models: ResolvedModelSet | None = None
        self._upstream: UpstreamRegistry | None = None
        self._router: GatewayRouter | None = None

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Resolve the model against discovery and bind the loopback listener.

        Idempotent: a second call on a running shim does nothing.
        """
        async with self._lock:
            if self._router is not None:
                return
            configured = self._settings
            settings = (
                configured
                if configured is not None
                else await asyncio.to_thread(get_default_client_settings)
            )
            catalog = await asyncio.to_thread(ModelCatalog.from_settings, settings)
            models = catalog.resolve_set(self._llm.model)
            select_strategy(models.primary)

            upstream = UpstreamRegistry(settings)
            router = GatewayRouter(
                catalog=catalog,
                upstream=upstream,
                models=models,
                api_key=self._api_key,
                on_call=self._on_call,
                trace_headers=self._trace_headers,
            )
            try:
                await router.start()
            except Exception:
                await upstream.aclose()
                raise

            self._settings = settings
            self._catalog = catalog
            self._models = models
            self._upstream = upstream
            self._router = router
            logger.info(
                "gateway shim routing %s to %s (%s/%s)",
                self._llm.model,
                models.primary.model_id,
                models.primary.vendor_type,
                models.primary.api_flavor,
            )

    async def stop(self) -> None:
        """Drain and release the listener and every upstream client."""
        async with self._lock:
            router, upstream = self._router, self._upstream
            self._router = None
            self._upstream = None
        if router is not None:
            try:
                await asyncio.wait_for(router.stop(), timeout=STOP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("gateway shim did not drain within %ss", STOP_TIMEOUT)
        if upstream is not None:
            await upstream.aclose()

    async def __aenter__(self) -> GatewayShim:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    # --- surface ----------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The loopback base URL to hand to the CLI."""
        return f"http://127.0.0.1:{self._require_router().port}"

    @property
    def api_key(self) -> str:
        """The per-run secret the CLI must present on every shim request."""
        return self._api_key

    @property
    def resolved_model(self) -> str:
        """The gateway model id to put in ``options.model``."""
        return self._require_models().primary.model_id

    @property
    def resolved_models(self) -> ResolvedModelSet:
        """The primary model plus the family models pinned for auxiliary traffic."""
        return self._require_models()

    @property
    def catalog(self) -> ModelCatalog:
        """The tenant catalog this shim resolved against."""
        if self._catalog is None:
            raise RuntimeError("GatewayShim.start() has not completed")
        return self._catalog

    def build_env(self) -> dict[str, str]:
        """Build the CLI subprocess environment for this shim."""
        return build_llm_env(self, self._require_models())

    def _require_router(self) -> GatewayRouter:
        if self._router is None:
            raise RuntimeError("GatewayShim.start() has not completed")
        return self._router

    def _require_models(self) -> ResolvedModelSet:
        if self._models is None:
            raise RuntimeError("GatewayShim.start() has not completed")
        return self._models


__all__ = [
    "AnthropicMessagesStrategy",
    "BedrockInvokeStrategy",
    "GatewayCallRecord",
    "GatewayCallSink",
    "GatewayRouter",
    "GatewayShim",
    "GatewayShimError",
    "GatewayStrategy",
    "ModelCatalog",
    "ModelNotInCatalogError",
    "ResolvedModel",
    "ResolvedModelSet",
    "UiPathModelSpec",
    "UnsupportedApiFlavorError",
    "UpstreamRegistry",
    "build_llm_env",
    "select_strategy",
    "to_anthropic_error",
]
