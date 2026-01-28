import logging
import os
from typing import Any, Optional, Sequence

from uipath.utils import EndpointManager

from .supported_models import BedrockModel

logger = logging.getLogger(__name__)


def _check_bedrock_dependencies() -> None:
    """Check if required dependencies for UiPath Bedrock LLMs are installed."""
    import importlib.util

    missing_packages = []

    if importlib.util.find_spec("llama_index.llms.bedrock") is None:
        missing_packages.append("llama-index-llms-bedrock")

    if importlib.util.find_spec("llama_index.llms.bedrock_converse") is None:
        missing_packages.append("llama-index-llms-bedrock-converse")

    if importlib.util.find_spec("boto3") is None:
        missing_packages.append("boto3")

    if importlib.util.find_spec("aiobotocore") is None:
        missing_packages.append("aiobotocore")

    if missing_packages:
        packages_str = ", ".join(missing_packages)
        raise ImportError(
            f"The following packages are required to use UiPath Bedrock LLMs: {packages_str}\n"
            "Please install them using one of the following methods:\n\n"
            "  # Using pip:\n"
            f"  pip install uipath-llamaindex[bedrock]\n\n"
            "  # Using uv:\n"
            f"  uv add 'uipath-llamaindex[bedrock]'\n\n"
        )


_check_bedrock_dependencies()

import boto3  # type: ignore[import-untyped]
from botocore import UNSIGNED
from botocore.config import Config
from llama_index.core.base.llms.types import (  # noqa: E402
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
)
from llama_index.core.llms.callbacks import (  # noqa: E402
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.bedrock import Bedrock  # type: ignore[import-untyped]
from llama_index.llms.bedrock_converse import (  # type: ignore[import-untyped]
    BedrockConverse,
)


class AwsBedrockCompletionsPassthroughClient:
    def __init__(
        self,
        model: str,
        token: str,
        api_flavor: str,
    ):
        self.model = model
        self.token = token
        self.api_flavor = api_flavor
        self._vendor = "awsbedrock"
        self._url: Optional[str] = None

    @property
    def endpoint(self) -> str:
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self.model,
        )
        return formatted_endpoint

    def _build_base_url(self) -> str:
        if not self._url:
            env_uipath_url = os.getenv("UIPATH_URL")

            if env_uipath_url:
                self._url = f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
            else:
                raise ValueError("UIPATH_URL environment variable is required")

        return self._url

    def get_client(self):
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="none",
            aws_secret_access_key="none",
            config=Config(
                signature_version=UNSIGNED,
                retries={
                    "total_max_attempts": 1,
                },
            ),
            verify=True,
        )
        client.meta.events.register(
            "before-send.bedrock-runtime.*", self._modify_request
        )
        return client

    def get_session(self):
        """Get aiobotocore session for async operations with custom event handlers."""
        from aiobotocore.session import get_session  # type: ignore[import-untyped]

        session = get_session()
        session.get_component("event_emitter").register(
            "before-send.bedrock-runtime.*", self._modify_request
        )
        return session

    def _modify_request(self, request, **kwargs):
        """Intercept boto3 request and redirect to LLM Gateway"""
        # Detect streaming based on URL suffix:
        # - converse-stream / invoke-with-response-stream -> streaming
        # - converse / invoke -> non-streaming
        streaming = "true" if request.url.endswith("-stream") else "false"
        request.url = self._build_base_url()

        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-UiPath-LlmGateway-ApiFlavor": self.api_flavor,
            "X-UiPath-Streaming-Enabled": streaming,
        }

        job_key = os.getenv("UIPATH_JOB_KEY")
        process_key = os.getenv("UIPATH_PROCESS_KEY")
        if job_key:
            headers["X-UiPath-JobKey"] = job_key
        if process_key:
            headers["X-UiPath-ProcessKey"] = process_key

        request.headers.update(headers)


class UiPathChatBedrockConverse(BedrockConverse):
    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model: str = BedrockModel.anthropic_claude_haiku_4_5,
        **kwargs,
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

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model,
            token=token,
            api_flavor="converse",
        )

        client = passthrough_client.get_client()
        botocore_session = passthrough_client.get_session()

        super().__init__(
            model=model,
            client=client,
            botocore_session=botocore_session,
            region_name="us-east-1",
            aws_access_key_id="none",
            aws_secret_access_key="none",
            **kwargs,
        )


class UiPathChatBedrock(Bedrock):
    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model: str = BedrockModel.anthropic_claude_haiku_4_5,
        context_size: int = 200000,
        **kwargs,
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

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model,
            token=token,
            api_flavor="invoke",
        )

        client = passthrough_client.get_client()

        super().__init__(
            model=model,
            client=client,
            context_size=context_size,
            aws_access_key_id="none",
            aws_secret_access_key="none",
            region_name="us-east-1",
            **kwargs,
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint - delegates to sync complete."""
        return self.complete(prompt, formatted=formatted, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat fallback - calls achat and yields single response."""

        async def gen() -> ChatResponseAsyncGen:
            response = await self.achat(messages, **kwargs)
            yield ChatResponse(
                message=response.message,
                raw=response.raw,
                delta=response.message.content or "",
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
            yield CompletionResponse(
                text=response.text,
                raw=response.raw,
                delta=response.text,
                additional_kwargs=response.additional_kwargs,
            )

        return gen()
