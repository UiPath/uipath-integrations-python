"""UiPath Gemini LLM for Google ADK agents.

Wraps the native Google ADK ``Gemini`` class, overriding the ``api_client``
cached property to inject httpx transports that rewrite URLs to the
UiPath LLM Gateway. All native Gemini features (streaming, caching, tool
calling) work automatically.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from functools import cached_property

import httpx
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import Client, types
from typing_extensions import override
from uipath._utils._ssl_context import get_httpx_client_kwargs

from ._common import build_gateway_url, get_uipath_config, get_uipath_headers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL-rewriting transports (same pattern as uipath-llamaindex vertex.py)
# ---------------------------------------------------------------------------


def _rewrite_request_for_gateway(
    request: httpx.Request, gateway_url: str
) -> httpx.Request:
    """Rewrite a Gemini generateContent request to the UiPath gateway."""
    url_str = str(request.url)
    if "generateContent" in url_str or "streamGenerateContent" in url_str:
        is_streaming = "streamGenerateContent" in url_str
        headers = dict(request.headers)
        if is_streaming:
            headers["X-UiPath-Streaming-Enabled"] = "true"

        # Preserve query parameters (e.g. ?alt=sse for streaming)
        original_url = httpx.URL(url_str)
        target_url = httpx.URL(gateway_url)
        if original_url.params:
            target_url = target_url.copy_merge_params(original_url.params)

        headers["host"] = target_url.host

        return httpx.Request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )
    return request


class _UrlRewriteTransport(httpx.BaseTransport):
    """Sync transport that rewrites URLs to the UiPath gateway."""

    def __init__(self, gateway_url: str, **kwargs):
        self.gateway_url = gateway_url
        self._transport = httpx.HTTPTransport(**kwargs)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request = _rewrite_request_for_gateway(request, self.gateway_url)
        return self._transport.handle_request(request)

    def close(self) -> None:
        self._transport.close()


class _AsyncUrlRewriteTransport(httpx.AsyncBaseTransport):
    """Async transport that rewrites URLs to the UiPath gateway."""

    def __init__(self, gateway_url: str, **kwargs):
        self.gateway_url = gateway_url
        self._transport = httpx.AsyncHTTPTransport(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = _rewrite_request_for_gateway(request, self.gateway_url)
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


# ---------------------------------------------------------------------------
# UiPathGemini
# ---------------------------------------------------------------------------


class UiPathGemini(Gemini):
    """Gemini LLM that routes requests through UiPath's LLM Gateway.

    Subclasses Google ADK's native ``Gemini`` class and overrides the
    ``api_client`` cached property to inject URL-rewriting transports.

    Example::

        from uipath_google_adk.chat import UiPathGemini
        from google.adk.agents import Agent

        agent = Agent(
            name="assistant",
            model=UiPathGemini(model="gemini-2.5-flash"),
            instruction="You are a helpful assistant.",
        )
    """

    @cached_property
    @override
    def api_client(self) -> Client:
        """Create a google.genai Client that routes through UiPath gateway."""
        uipath_url, token = get_uipath_config()
        effective_model = self.model
        gateway_url = build_gateway_url("vertexai", effective_model, uipath_url)
        auth_headers = get_uipath_headers(token)
        auth_headers["X-UiPath-LlmGateway-ApiFlavor"] = "GeminiGenerateContent"
        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        http_options = types.HttpOptions(
            headers=self._tracking_headers(),
            retry_options=self.retry_options,
            base_url=self.base_url,
            httpx_client=httpx.Client(
                transport=_UrlRewriteTransport(gateway_url, verify=verify),
                headers=auth_headers,
                **client_kwargs,
            ),
            httpx_async_client=httpx.AsyncClient(
                transport=_AsyncUrlRewriteTransport(gateway_url, verify=verify),
                headers=auth_headers,
                **client_kwargs,
            ),
        )

        return Client(
            api_key="uipath-gateway",
            http_options=http_options,
        )

    @override
    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Force non-streaming to work around gateway streaming duplication."""
        async for response in super().generate_content_async(llm_request, stream=False):
            yield response
