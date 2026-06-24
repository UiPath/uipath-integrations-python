"""UiPath Anthropic Client for Agent Framework agents.

Routes Anthropic requests through UiPath's LLM Gateway using the
Bedrock invoke format, then passes the pre-configured
``AsyncAnthropic`` client to the Agent Framework ``AnthropicClient``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

from ._common import build_gateway_url, get_uipath_config, get_uipath_headers

logger = logging.getLogger(__name__)


def _check_anthropic_dependency() -> None:
    """Check that the anthropic SDK is installed."""
    import importlib.util

    if importlib.util.find_spec("anthropic") is None:
        raise ImportError(
            "The 'anthropic' package is required to use UiPathAnthropicClient.\n"
            "Install it with:\n\n"
            "  pip install 'uipath-agent-framework[anthropic]'\n\n"
            "  # or with uv:\n"
            "  uv add 'uipath-agent-framework[anthropic]'\n"
        )


class _AsyncUrlRewriteTransport(httpx.AsyncHTTPTransport):
    """Async transport that rewrites Anthropic SDK requests to UiPath gateway.

    Intercepts the Anthropic SDK's ``/v1/messages`` endpoint and:
    1. Rewrites the URL to the UiPath gateway
    2. Transforms the body to Bedrock invoke format (removes ``model``,
       adds ``anthropic_version``)
    """

    def __init__(self, gateway_url: str, **kwargs: Any):
        self.gateway_url = gateway_url
        super().__init__(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/messages" in url_str:
            gateway_parsed = httpx.URL(self.gateway_url)

            # Transform body to Bedrock invoke format
            body = json.loads(request.content)
            body.pop("model", None)
            body.setdefault("anthropic_version", "bedrock-2023-05-31")
            is_streaming = body.get("stream", False)
            new_content = json.dumps(body).encode()

            headers = dict(request.headers)
            headers["host"] = gateway_parsed.host
            headers["content-length"] = str(len(new_content))
            if is_streaming:
                headers["X-UiPath-Streaming-Enabled"] = "true"

            request = httpx.Request(
                method=request.method,
                url=gateway_parsed,
                headers=headers,
                content=new_content,
                extensions=request.extensions,
            )

        return await super().handle_async_request(request)


class UiPathAnthropicClient:
    """Anthropic Client that routes requests through UiPath's LLM Gateway.

    Subclasses Agent Framework's ``AnthropicClient`` and configures a
    pre-built ``AsyncAnthropic`` client with URL-rewriting + Bedrock
    body transform.

    Uses Bedrock model names (``anthropic.claude-*``) with the ``awsbedrock``
    vendor and ``invoke`` API flavor.

    Example::

        from uipath_agent_framework.chat import UiPathAnthropicClient

        client = UiPathAnthropicClient(model="anthropic.claude-haiku-4-5-20251001-v1:0")
        agent = client.as_agent(
            name="assistant",
            instructions="You are a helpful assistant.",
            tools=[my_tool],
        )
    """

    def __new__(
        cls, model: str = "anthropic.claude-haiku-4-5-20251001-v1:0", **kwargs: Any
    ):
        _check_anthropic_dependency()

        from agent_framework_anthropic import AnthropicClient
        from anthropic import AsyncAnthropic  # type: ignore[import-untyped]

        uipath_url, token = get_uipath_config()
        gateway_url = build_gateway_url("awsbedrock", model, uipath_url)

        auth_headers = get_uipath_headers(token)
        auth_headers["X-UiPath-LlmGateway-ApiFlavor"] = "invoke"

        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        http_client = httpx.AsyncClient(
            transport=_AsyncUrlRewriteTransport(gateway_url, verify=verify),
            **client_kwargs,
        )

        anthropic_client = AsyncAnthropic(
            api_key="uipath-gateway",
            default_headers=auth_headers,
            http_client=http_client,
        )

        return AnthropicClient(
            model=model,
            anthropic_client=anthropic_client,
            **kwargs,
        )
