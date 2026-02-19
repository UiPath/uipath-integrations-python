"""UiPath OpenAI Chat Client for Agent Framework agents.

Routes OpenAI requests through UiPath's LLM Gateway using a
URL-rewriting httpx transport, then passes the pre-configured
``AsyncOpenAI`` client to the Agent Framework ``OpenAIChatClient``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx
from agent_framework import FunctionTool
from agent_framework import tool as tool_decorator
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
from uipath._utils._ssl_context import get_httpx_client_kwargs

from ._common import build_gateway_url, get_uipath_config, get_uipath_headers

logger = logging.getLogger(__name__)


class _AsyncUrlRewriteTransport(httpx.AsyncHTTPTransport):
    """Async transport that rewrites OpenAI SDK requests to UiPath gateway.

    Intercepts ``/chat/completions`` requests and rewrites the URL
    to the UiPath LLM Gateway endpoint.
    """

    def __init__(self, gateway_url: str, **kwargs: Any):
        self.gateway_url = gateway_url
        super().__init__(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/chat/completions" in url_str:
            gateway_parsed = httpx.URL(self.gateway_url)
            headers = dict(request.headers)
            headers["host"] = gateway_parsed.host

            is_streaming = b'"stream": true' in (
                request.content or b""
            ) or b'"stream":true' in (request.content or b"")
            if is_streaming:
                headers["X-UiPath-Streaming-Enabled"] = "true"

            request = httpx.Request(
                method=request.method,
                url=gateway_parsed,
                headers=headers,
                content=request.content,
                extensions=request.extensions,
            )

        return await super().handle_async_request(request)


class UiPathOpenAIChatClient(OpenAIChatClient):
    """OpenAI Chat Client that routes requests through UiPath's LLM Gateway.

    Subclasses Agent Framework's ``OpenAIChatClient`` and configures a
    pre-built ``AsyncOpenAI`` client with URL-rewriting transport.

    Example::

        from uipath_agent_framework.chat import UiPathOpenAIChatClient

        client = UiPathOpenAIChatClient(model="gpt-4o-mini")
        agent = client.as_agent(
            name="assistant",
            instructions="You are a helpful assistant.",
            tools=[my_tool],
        )
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any):
        uipath_url, token = get_uipath_config()
        gateway_url = build_gateway_url("openai", model, uipath_url)

        auth_headers = get_uipath_headers(token)
        auth_headers["X-UiPath-LlmGateway-ApiFlavor"] = "OpenAiChatCompletions"

        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        http_client = httpx.AsyncClient(
            transport=_AsyncUrlRewriteTransport(gateway_url, verify=verify),
            headers=auth_headers,
            **client_kwargs,
        )

        async_client = AsyncOpenAI(
            api_key=token,
            http_client=http_client,
        )

        super().__init__(
            model_id=model,
            async_client=async_client,
            **kwargs,
        )

    def _prepare_tools_for_openai(self, tools: Sequence[Any]) -> dict[str, Any]:
        """Prepare tools for the OpenAI Chat Completions API.

        Extends the base implementation to convert plain callables to
        FunctionTool instances before processing. The base class only
        handles FunctionTool and dict types — plain functions fall through
        as-is and cause JSON serialization errors.
        """
        normalized: list[Any] = []
        for t in tools:
            if callable(t) and not isinstance(t, FunctionTool):
                normalized.append(tool_decorator(t))
            else:
                normalized.append(t)
        return super()._prepare_tools_for_openai(normalized)
