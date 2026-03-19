"""UiPath OpenAI chat client for PydanticAI with custom endpoint integration."""

import os
from typing import Optional

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.platform.common import apply_debug_licensing_override
from uipath.utils import EndpointManager

from .supported_models import OpenAIModels


def _rewrite_openai_url(
    original_url: str, params: httpx.QueryParams
) -> httpx.URL | None:
    """Rewrite OpenAI URLs to UiPath gateway completions endpoint.

    Handles URL patterns from OpenAI SDK and rewrites to /completions.
    The X-UiPath-LlmGateway-ApiFlavor header determines API behavior.

    Args:
        original_url: Original URL from OpenAI SDK
        params: Query parameters to preserve

    Returns:
        Rewritten URL pointing to UiPath completions endpoint
    """
    # Extract base URL before endpoint path
    if "/responses" in original_url:
        base_url = original_url.split("/responses")[0]
    elif "/chat/completions" in original_url:
        base_url = original_url.split("/chat/completions")[0]
    elif "/completions" in original_url:
        base_url = original_url.split("/completions")[0]
    else:
        # Handle base URL case - strip query string
        base_url = original_url.split("?")[0]

    new_url_str = f"{base_url}/completions"
    if params:
        return httpx.URL(new_url_str, params=params)
    return httpx.URL(new_url_str)


class UiPathURLRewriteTransport(httpx.AsyncHTTPTransport):
    """Custom async transport that rewrites URLs to UiPath endpoints."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle async request with URL rewriting."""
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url

        return await super().handle_async_request(request)


class UiPathSyncURLRewriteTransport(httpx.HTTPTransport):
    """Custom sync transport that rewrites URLs to UiPath endpoints."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle sync request with URL rewriting."""
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url

        return super().handle_request(request)


class UiPathChatOpenAI:
    """UiPath OpenAI client for PydanticAI chat completions.

    This client wraps the OpenAI SDK and configures it to use UiPath's
    LLM Gateway endpoints with proper authentication and headers.
    Returns PydanticAI-compatible OpenAIChatModel instances.

    Example:
        ```python
        from uipath_pydantic_ai.chat import UiPathChatOpenAI

        client = UiPathChatOpenAI(model_name="gpt-4o")

        # Use with PydanticAI Agent
        from pydantic_ai import Agent
        agent = Agent(client.model)
        ```
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model_name: str = OpenAIModels.gpt_4o_2024_11_20,
        api_version: str = "2024-12-01-preview",
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        agenthub_config: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        byo_connection_id: Optional[str] = None,
        api_flavor: str = "chat-completions",
    ):
        """Initialize UiPath OpenAI client for PydanticAI.

        Args:
            token: UiPath access token (defaults to UIPATH_ACCESS_TOKEN env var)
            model_name: Model to use (e.g., "gpt-4o-2024-11-20")
            api_version: OpenAI API version
            org_id: UiPath organization ID (defaults to UIPATH_ORGANIZATION_ID env var)
            tenant_id: UiPath tenant ID (defaults to UIPATH_TENANT_ID env var)
            agenthub_config: Optional AgentHub configuration
            extra_headers: Additional headers to include in requests
            byo_connection_id: Bring-your-own connection ID
            api_flavor: API flavor to use - "chat-completions" (default for PydanticAI),
                       "responses", or "auto" (let UiPath decide)
        """
        # Get credentials from env vars if not provided
        self._org_id = org_id or os.getenv("UIPATH_ORGANIZATION_ID")
        self._tenant_id = tenant_id or os.getenv("UIPATH_TENANT_ID")
        self._token = token or os.getenv("UIPATH_ACCESS_TOKEN")

        # Validate required credentials
        if not self._org_id:
            raise ValueError(
                "UIPATH_ORGANIZATION_ID environment variable or org_id parameter is required"
            )
        if not self._tenant_id:
            raise ValueError(
                "UIPATH_TENANT_ID environment variable or tenant_id parameter is required"
            )
        if not self._token:
            raise ValueError(
                "UIPATH_ACCESS_TOKEN environment variable or token parameter is required"
            )

        # Store configuration
        self._model_name = model_name
        self._api_version = api_version
        self._vendor = "openai"
        self._agenthub_config = agenthub_config
        self._byo_connection_id = byo_connection_id
        self._api_flavor = api_flavor
        self._extra_headers = extra_headers or {}

        # Build base URL and headers
        base_url = self._build_base_url()
        headers = self._build_headers()

        # Get SSL configuration
        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        # Create sync client
        self._client = OpenAI(
            base_url=base_url,
            api_key=self._token,
            default_headers=headers,
            http_client=httpx.Client(
                transport=UiPathSyncURLRewriteTransport(verify=verify),
                **client_kwargs,
            ),
        )

        # Create async client
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=self._token,
            default_headers=headers,
            http_client=httpx.AsyncClient(
                transport=UiPathURLRewriteTransport(verify=verify),
                **client_kwargs,
            ),
        )

        # Create PydanticAI-compatible model
        self._model = OpenAIChatModel(
            self._model_name,
            provider=OpenAIProvider(openai_client=self._async_client),
        )

    def _build_headers(self) -> dict[str, str]:
        """Build headers for UiPath LLM Gateway."""
        headers = {
            "X-UiPath-LlmGateway-ApiFlavor": self._api_flavor,
            "Authorization": f"Bearer {self._token}",
        }

        # Add optional headers
        if self._agenthub_config:
            headers["X-UiPath-AgentHub-Config"] = apply_debug_licensing_override(
                self._agenthub_config
            )
        if self._byo_connection_id:
            headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = self._byo_connection_id
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key

        # Allow extra_headers to override defaults
        headers.update(self._extra_headers)
        return headers

    @property
    def endpoint(self) -> str:
        """Get the UiPath endpoint for this model (without query parameters)."""
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self._model_name,
        )
        # Remove /completions suffix - will be added by URL rewriting
        base_endpoint = formatted_endpoint.replace("/completions", "")
        return base_endpoint

    def _build_base_url(self) -> str:
        """Build the base URL for OpenAI client.

        Note: Query parameters like api-version are added by the URL rewriting logic,
        not in the base URL, to allow the SDK to append paths properly.
        """
        env_uipath_url = os.getenv("UIPATH_URL")

        if env_uipath_url:
            return f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
        else:
            raise ValueError("UIPATH_URL environment variable is required")

    @property
    def model(self) -> OpenAIChatModel:
        """Get the PydanticAI-compatible OpenAIChatModel."""
        return self._model

    @property
    def client(self) -> OpenAI:
        """Get the synchronous OpenAI client."""
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get the asynchronous OpenAI client."""
        return self._async_client

    @property
    def model_name(self) -> str:
        """Get the configured model name."""
        return self._model_name
