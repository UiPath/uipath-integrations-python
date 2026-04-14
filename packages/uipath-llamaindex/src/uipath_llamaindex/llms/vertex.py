import logging
import os
from typing import Any, Generator, Optional, Sequence

import httpx
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from .supported_models import GeminiModel

logger = logging.getLogger(__name__)


def _check_vertex_dependencies() -> None:
    """Check if required dependencies for UiPath Vertex LLMs are installed."""
    import importlib.util

    missing_packages = []

    if importlib.util.find_spec("llama_index.llms.google_genai") is None:
        missing_packages.append("llama-index-llms-google-genai")

    if missing_packages:
        packages_str = ", ".join(missing_packages)
        raise ImportError(
            f"The following packages are required to use UiPath Vertex LLMs: {packages_str}\n"
            "Please install them using one of the following methods:\n\n"
            "  # Using pip:\n"
            f"  pip install uipath-llamaindex[vertex]\n\n"
            "  # Using uv:\n"
            f"  uv add 'uipath-llamaindex[vertex]'\n\n"
        )


_check_vertex_dependencies()

import google.genai  # noqa: E402
import google.genai.types as genai_types  # noqa: E402
from llama_index.core.base.llms.types import (  # noqa: E402
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import PrivateAttr  # noqa: E402
from llama_index.core.llms.callbacks import (  # noqa: E402
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.tools import ToolSelection  # noqa: E402
from llama_index.llms.google_genai import GoogleGenAI  # noqa: E402


def _rewrite_request_for_gateway(
    request: httpx.Request, gateway_url: str
) -> httpx.Request:
    """Rewrite a request to redirect to the UiPath gateway."""
    url_str = str(request.url)
    if "generateContent" in url_str or "streamGenerateContent" in url_str:
        is_streaming = "streamGenerateContent" in url_str
        # Build headers with streaming flag and correct host
        headers = dict(request.headers)
        if is_streaming:
            headers["X-UiPath-Streaming-Enabled"] = "true"
        # Update host header to match the gateway URL
        gateway_url_parsed = httpx.URL(gateway_url)
        headers["host"] = gateway_url_parsed.host
        # Create new request with rewritten URL
        return httpx.Request(
            method=request.method,
            url=gateway_url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )
    return request


class _UrlRewriteTransport(httpx.BaseTransport):
    """Transport that rewrites URLs to redirect to UiPath gateway."""

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url
        self._transport = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request = _rewrite_request_for_gateway(request, self.gateway_url)
        return self._transport.handle_request(request)

    def close(self) -> None:
        self._transport.close()


class _AsyncUrlRewriteTransport(httpx.AsyncBaseTransport):
    """Async transport that rewrites URLs to redirect to UiPath gateway."""

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url
        self._transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = _rewrite_request_for_gateway(request, self.gateway_url)
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


class UiPathVertex(GoogleGenAI):
    """
    UiPath Vertex AI LLM that routes requests through UiPath's LLM Gateway.

    This class wraps LlamaIndex's GoogleGenAI class and redirects all API calls
    to UiPath's LLM Gateway for authentication and routing.

    Args:
        org_id: UiPath organization ID. Falls back to UIPATH_ORGANIZATION_ID env var.
        tenant_id: UiPath tenant ID. Falls back to UIPATH_TENANT_ID env var.
        token: UiPath access token. Falls back to UIPATH_ACCESS_TOKEN env var.
        model: Model identifier. Defaults to gemini-2.5-flash.
        **kwargs: Additional arguments passed to the GoogleGenAI base class.
    Example:
        ```python
        from uipath_llamaindex.llms.vertex import UiPathVertex
        from uipath_llamaindex.llms import GeminiModels
        llm = UiPathVertex(model=GeminiModels.gemini_2_5_pro)
        response = llm.complete("What is the capital of France?")
        ```
    """

    # Private attributes for UiPath-specific state
    _uipath_vendor: str = PrivateAttr(default="vertexai")
    _uipath_model_name: str = PrivateAttr()
    _uipath_url: Optional[str] = PrivateAttr(default=None)
    _uipath_token: str = PrivateAttr()

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model: str = GeminiModel.gemini_2_5_flash,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
        max_retries: int = 3,
        generation_config: Optional[genai_types.GenerateContentConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ):
        org_id = org_id or os.getenv("UIPATH_ORGANIZATION_ID")
        tenant_id = tenant_id or os.getenv("UIPATH_TENANT_ID")
        token = token or os.getenv("UIPATH_ACCESS_TOKEN")

        if not org_id:
            raise ValueError(
                "UIPATH_ORGANIZATION_ID environment variable or org_id parameter is required"
            )
        if not tenant_id:
            raise ValueError(
                "UIPATH_TENANT_ID environment variable or tenant_id parameter is required"
            )
        if not token:
            raise ValueError(
                "UIPATH_ACCESS_TOKEN environment variable or token parameter is required"
            )

        # Build UiPath gateway URL and headers
        uipath_url = self._build_base_url_static(model)
        headers = self._build_headers_static(token)

        # Create httpx clients with URL rewriting transport
        http_options = genai_types.HttpOptions(
            httpx_client=httpx.Client(
                transport=_UrlRewriteTransport(uipath_url),
                headers=headers,
                **get_httpx_client_kwargs(),
            ),
            httpx_async_client=httpx.AsyncClient(
                transport=_AsyncUrlRewriteTransport(uipath_url),
                headers=headers,
                **get_httpx_client_kwargs(),
            ),
        )

        # Create google.genai client with custom httpx client
        # We pass a dummy api_key since auth is handled by UiPath headers
        client = google.genai.Client(
            api_key="uipath-gateway",
            http_options=http_options,
        )

        # Skip calling GoogleGenAI.__init__ which tries to fetch model metadata
        # Instead, initialize the grandparent (FunctionCallingLLM) directly
        # and set up the attributes ourselves
        from llama_index.core.llms.function_calling import FunctionCallingLLM

        FunctionCallingLLM.__init__(
            self,
            callback_manager=callback_manager,
            **kwargs,
        )

        # Set GoogleGenAI public attributes
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.max_retries = max_retries
        self.is_function_calling_model = is_function_calling_model
        self.cached_content = None
        self.built_in_tool = None
        self.file_mode = "hybrid"

        # Set GoogleGenAI private attributes
        self._client = client
        # Create a minimal model metadata object to avoid None errors
        # when methods try to access input_token_limit, etc.
        self._model_meta = genai_types.Model(
            name=model,
            input_token_limit=1000000,  # Gemini 2.5 supports 1M tokens
            output_token_limit=65536,
        )
        self._max_tokens = max_tokens or DEFAULT_NUM_OUTPUTS

        # Set up generation config
        if generation_config:
            self._generation_config = generation_config.model_dump()  # type: ignore[assignment]
        else:
            self._generation_config = genai_types.GenerateContentConfig(  # type: ignore[assignment]
                temperature=temperature,
                max_output_tokens=max_tokens,
            ).model_dump()

        # Set UiPath private attributes
        self._uipath_vendor = "vertexai"
        self._uipath_model_name = model
        self._uipath_url = uipath_url
        self._uipath_token = token

    @staticmethod
    def _build_headers_static(token: str) -> dict[str, str]:
        """Build HTTP headers for UiPath Gateway requests."""
        headers = {
            "Authorization": f"Bearer {token}",
        }
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key
        return headers

    @staticmethod
    def _build_base_url_static(model: str) -> str:
        """Build the full URL for the UiPath LLM Gateway."""
        env_uipath_url = os.getenv("UIPATH_URL")

        if not env_uipath_url:
            raise ValueError("UIPATH_URL environment variable is required")

        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor="vertexai",
            model=model,
        )
        return f"{env_uipath_url.rstrip('/')}/{formatted_endpoint}"

    # Streaming fallback methods - call non-streaming and yield single response
    # This works around backend streaming bugs in UiPath Gateway

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint - delegates to chat."""
        response = self.chat([ChatMessage(role="user", content=prompt)], **kwargs)
        return CompletionResponse(
            text=response.message.content or "",
            raw=response.raw,
            additional_kwargs=response.additional_kwargs,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion fallback - calls complete and yields single response."""

        def gen() -> Generator[CompletionResponse, None, None]:
            response = self.complete(prompt, formatted=formatted, **kwargs)
            # Yield the full response as a single "chunk" with delta = full text
            yield CompletionResponse(
                text=response.text,
                raw=response.raw,
                delta=response.text,
                additional_kwargs=response.additional_kwargs,
            )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion fallback - calls acomplete and yields single response."""

        async def gen() -> CompletionResponseAsyncGen:
            response = await self.acomplete(prompt, formatted=formatted, **kwargs)
            # Yield the full response as a single "chunk" with delta = full text
            yield CompletionResponse(
                text=response.text,
                raw=response.raw,
                delta=response.text,
                additional_kwargs=response.additional_kwargs,
            )

        return gen()

    def _create_sync_client(self) -> google.genai.Client:
        """Create a fresh google.genai client for sync operations."""
        uipath_url = self._uipath_url
        if uipath_url is None:
            raise ValueError("UiPath URL is not set")
        headers = self._build_headers_static(self._uipath_token)

        http_options = genai_types.HttpOptions(
            httpx_client=httpx.Client(
                transport=_UrlRewriteTransport(uipath_url),
                headers=headers,
                **get_httpx_client_kwargs(),
            ),
        )

        return google.genai.Client(
            api_key="uipath-gateway",
            http_options=http_options,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint using sync client."""
        from llama_index.llms.google_genai.utils import chat_from_gemini_response

        generation_config = genai_types.GenerateContentConfig(
            **{
                **(self._generation_config or {}),
                **kwargs.pop("generation_config", {}),
            }
        )

        # Create a fresh sync client to avoid event loop issues
        sync_client = self._create_sync_client()

        # Convert messages to Gemini format (sync version - no file uploads)
        contents: list[genai_types.Content] = []
        for message in messages:
            content = genai_types.Content(
                role="user" if message.role.value == "user" else "model",
                parts=[genai_types.Part(text=message.content or "")],
            )
            contents.append(content)

        # Use sync client to send message
        chat = sync_client.chats.create(
            model=self.model,
            config=generation_config,
            history=contents[:-1] if len(contents) > 1 else None,  # type: ignore[arg-type]
        )
        last_parts = contents[-1].parts if contents else []
        response = chat.send_message(last_parts if last_parts else "")  # type: ignore[arg-type]

        return chat_from_gemini_response(response, [])

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat fallback - calls chat and yields single response."""

        def gen() -> Generator[ChatResponse, None, None]:
            response = self.chat(messages, **kwargs)
            # Yield the full response as a single "chunk" with delta = full content
            yield ChatResponse(
                message=response.message,
                raw=response.raw,
                delta=response.message.content or "",
                additional_kwargs=response.additional_kwargs,
            )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat fallback - calls achat and yields single response."""

        async def gen() -> ChatResponseAsyncGen:
            response = await self.achat(messages, **kwargs)
            # Yield the full response as a single "chunk" with delta = full content
            yield ChatResponse(
                message=response.message,
                raw=response.raw,
                delta=response.message.content or "",
                additional_kwargs=response.additional_kwargs,
            )

        return gen()

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = False,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        return super().get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call, **kwargs
        )
