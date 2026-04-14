import os
from typing import Any

import httpx
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.tools import ToolSelection
from llama_index.llms.azure_openai import AzureOpenAI  # type: ignore
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from .supported_models import OpenAIModel


class _UiPathSyncURLRewriteTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)

        if "/openai/deployments/" in original_url:
            base_url = original_url.split("/openai/deployments/")[0]
            query_string = request.url.params
            new_url_str = f"{base_url}/completions"
            if query_string:
                request.url = httpx.URL(new_url_str, params=query_string)
            else:
                request.url = httpx.URL(new_url_str)

        return super().handle_request(request)


class _UiPathAsyncURLRewriteTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)

        if "/openai/deployments/" in original_url:
            base_url = original_url.split("/openai/deployments/")[0]
            query_string = request.url.params
            new_url_str = f"{base_url}/completions"
            if query_string:
                request.url = httpx.URL(new_url_str, params=query_string)
            else:
                request.url = httpx.URL(new_url_str)

        return await super().handle_async_request(request)


class UiPathOpenAI(AzureOpenAI):
    def __init__(
        self,
        model: str | OpenAIModel = OpenAIModel.GPT_4_1_2025_04_14,
        api_version: str = "2024-10-21",
        **kwargs: Any,
    ):
        default_headers_dict = {
            "X-UiPath-LlmGateway-ApiFlavor": "auto",
            "X-UiPath-LlmGateway-RequestingProduct": "uipath-python-sdk",
            "X-UiPath-LlmGateway-RequestingFeature": "llama-index-agent",
        }
        model_value = model.value if isinstance(model, OpenAIModel) else model

        base_url = os.environ.get("UIPATH_URL", "EMPTY").rstrip("/")

        if base_url == "EMPTY":
            raise ValueError(
                "UIPATH_URL environment variable is not set. Please run uipath auth."
            )

        vendor_endpoint = (
            EndpointManager.get_vendor_endpoint()
            .format(vendor="openai", model=model_value)
            .replace("/completions", "")
        )

        defaults = {
            "model": model_value,
            "deployment_name": model_value,
            "azure_endpoint": f"{base_url}/{vendor_endpoint}",
            "api_key": os.environ.get("UIPATH_ACCESS_TOKEN"),
            "api_version": api_version,
            "is_chat_model": True,
            "default_headers": default_headers_dict,
            "http_client": httpx.Client(
                transport=_UiPathSyncURLRewriteTransport(),
                **get_httpx_client_kwargs(),
            ),
            "async_http_client": httpx.AsyncClient(
                transport=_UiPathAsyncURLRewriteTransport(),
                **get_httpx_client_kwargs(),
            ),
        }
        final_kwargs = {**defaults, **kwargs}
        super().__init__(**final_kwargs)

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = False,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        return super().get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call, **kwargs
        )
