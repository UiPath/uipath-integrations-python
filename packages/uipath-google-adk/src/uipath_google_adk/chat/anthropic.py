"""UiPath Anthropic LLM for Google ADK agents.

Wraps the Google ADK ``AnthropicLlm`` class, overriding the
``_anthropic_client`` cached property to create an ``AsyncAnthropic``
client that routes all requests through UiPath's LLM Gateway using
the Bedrock invoke format.
"""

from __future__ import annotations

import json
import logging
from functools import cached_property

import httpx
from google.adk.models.anthropic_llm import AnthropicLlm
from typing_extensions import override
from uipath._utils._ssl_context import get_httpx_client_kwargs

from ._common import build_gateway_url, get_uipath_config, get_uipath_headers

logger = logging.getLogger(__name__)


def _check_anthropic_dependency() -> None:
    """Check that the anthropic SDK is installed."""
    import importlib.util

    if importlib.util.find_spec("anthropic") is None:
        raise ImportError(
            "The 'anthropic' package is required to use UiPathAnthropic.\n"
            "Install it with:\n\n"
            "  pip install 'uipath-google-adk[anthropic]'\n\n"
            "  # or with uv:\n"
            "  uv add 'uipath-google-adk[anthropic]'\n"
        )


class _AsyncUrlRewriteTransport(httpx.AsyncHTTPTransport):
    """Async transport that rewrites Anthropic SDK requests to UiPath gateway.

    Intercepts the Anthropic SDK's ``/v1/messages`` endpoint and:
    1. Rewrites the URL to the UiPath gateway
    2. Transforms the body to Bedrock invoke format (removes ``model``,
       adds ``anthropic_version``)

    Extends ``httpx.AsyncHTTPTransport`` directly (proven pattern from
    ``uipath-openai-agents``) and mutates ``request.url`` in-place.
    """

    def __init__(self, gateway_url: str, **kwargs):
        self.gateway_url = gateway_url
        super().__init__(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        # Intercept Anthropic SDK's /v1/messages endpoint
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

            # Create new request with modified URL and body
            request = httpx.Request(
                method=request.method,
                url=gateway_parsed,
                headers=headers,
                content=new_content,
                extensions=request.extensions,
            )

        return await super().handle_async_request(request)


class UiPathAnthropic(AnthropicLlm):
    """Anthropic LLM that routes requests through UiPath's LLM Gateway.

    Subclasses Google ADK's ``AnthropicLlm`` and overrides
    ``_anthropic_client`` to use the Bedrock invoke format through
    the UiPath gateway (matching the ``uipath-llamaindex`` implementation).

    Uses Bedrock model names (``anthropic.claude-*``) with the ``awsbedrock``
    vendor and ``invoke`` API flavor.

    Example::

        from uipath_google_adk.chat import UiPathAnthropic
        from google.adk.agents import Agent

        agent = Agent(
            name="assistant",
            model=UiPathAnthropic(model="anthropic.claude-haiku-4-5-20251001-v1:0"),
            instruction="You are a helpful assistant.",
        )
    """

    @cached_property
    @override
    def _anthropic_client(self):
        _check_anthropic_dependency()

        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        uipath_url, token = get_uipath_config()
        effective_model = self.model
        gateway_url = build_gateway_url("awsbedrock", effective_model, uipath_url)
        auth_headers = get_uipath_headers(token)
        auth_headers["X-UiPath-LlmGateway-ApiFlavor"] = "invoke"
        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        http_client = httpx.AsyncClient(
            transport=_AsyncUrlRewriteTransport(gateway_url, verify=verify),
            **client_kwargs,
        )

        return AsyncAnthropic(
            api_key="uipath-gateway",
            default_headers=auth_headers,
            http_client=http_client,
        )
