"""Local HTTP proxy bridging the Claude Agent SDK to the UiPath LLM Gateway.

The Claude SDK speaks the Anthropic API (SSE). The UiPath gateway speaks AWS
Bedrock (binary Event Stream). This proxy translates between the two: strips
non-Bedrock fields and converts the binary response stream to SSE.

Callers are expected to pass Bedrock ARN-style model IDs (e.g.
"anthropic.claude-sonnet-4-5-20250929-v1:0") — the proxy forwards them
unchanged to the gateway URL.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any
from urllib.parse import quote

import httpx
from aiohttp import web
from uipath.platform.common._config import UiPathConfig
from uipath.platform.constants import (
    ENV_FOLDER_KEY,
    ENV_JOB_KEY,
    ENV_PROCESS_KEY,
    ENV_UIPATH_TRACE_ID,
    HEADER_AGENTHUB_CONFIG,
    HEADER_FOLDER_KEY,
    HEADER_JOB_KEY,
    HEADER_LICENSING_CONTEXT,
    HEADER_PROCESS_KEY,
    HEADER_TRACE_ID,
)

logger = logging.getLogger(__name__)

# --- Constants ----------------------------------------------------------------

_UPSTREAM_TIMEOUT = httpx.Timeout(timeout=300.0, connect=30.0)

# Bedrock rejects unknown fields with 400 "Extra inputs are not permitted".
_ALLOWED_BEDROCK_FIELDS: frozenset[str] = frozenset(
    {
        "anthropic_version",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "tools",
        "tool_choice",
        "thinking",
        "metadata",
    }
)

# --- Pure functions -----------------------------------------------------------


def _gateway_url(base_url: str, model_id: str) -> str:
    return (
        f"{base_url}/agenthub_/llm/raw/vendor/awsbedrock/model/{model_id}/completions"
    )


def _build_uipath_context_headers(agenthub_config: str | None) -> dict[str, str]:
    """Build UiPath job/process/licensing context headers from the environment."""
    headers: dict[str, str] = {}
    if agenthub_config:
        headers[HEADER_AGENTHUB_CONFIG] = agenthub_config
    if process_key := os.getenv(ENV_PROCESS_KEY):
        headers[HEADER_PROCESS_KEY] = quote(process_key, safe="")
    if job_key := os.getenv(ENV_JOB_KEY):
        headers[HEADER_JOB_KEY] = job_key
    if folder_key := os.getenv(ENV_FOLDER_KEY):
        headers[HEADER_FOLDER_KEY] = folder_key
    if trace_id := os.getenv(ENV_UIPATH_TRACE_ID):
        headers[HEADER_TRACE_ID] = trace_id
    if licensing_context := UiPathConfig.licensing_context:
        headers[HEADER_LICENSING_CONTEXT] = licensing_context
    return headers


def _build_headers(
    access_token: str, *, is_streaming: bool, agenthub_config: str
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-UiPath-Streaming-Enabled": "true" if is_streaming else "false",
        "X-UiPath-LlmGateway-ApiFlavor": "invoke",
    }
    headers.update(_build_uipath_context_headers(agenthub_config))
    return headers


def _coerce_to_string(value: Any) -> Any:
    """Flatten Anthropic array-form content blocks to a newline-joined string.

    Pass-through for non-list values (string, None).
    """
    if not isinstance(value, list):
        return value
    return "\n\n".join(b.get("text", "") for b in value if b.get("type") == "text")


def _transform_body(payload: dict[str, Any]) -> bytes:
    """Return a Bedrock-safe request body.

    Strips unknown fields, injects anthropic_version, and flattens
    Anthropic array-form content to string at the two positions where the
    UiPath Bedrock InvokeModel passthrough requires strings: system
    and tool_result.content.
    """
    body = {k: v for k, v in payload.items() if k in _ALLOWED_BEDROCK_FIELDS}
    body["anthropic_version"] = "bedrock-2023-05-31"
    body["system"] = _coerce_to_string(body.get("system"))
    for msg in body.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result":
                block["content"] = _coerce_to_string(block.get("content"))
    return json.dumps(body).encode()


async def _stream_bedrock(
    upstream: httpx.Response,
    response: web.StreamResponse,
) -> int:
    """Convert AWS Bedrock Event Stream binary frames to Anthropic SSE.

    The gateway returns application/vnd.amazon.eventstream — a binary framing
    format where each frame carries a base64-encoded Anthropic event JSON.
    Decodes those frames and re-emits them as text/event-stream SSE.

    Returns:
        Total SSE bytes written to the client.
    """
    from botocore.eventstream import EventStreamBuffer

    buf = EventStreamBuffer()
    bytes_sent = 0

    async for chunk in upstream.aiter_bytes():
        buf.add_data(chunk)
        for msg in buf:
            try:
                envelope = json.loads(msg.payload)
                inner = base64.b64decode(envelope["bytes"])
                obj = json.loads(inner)
                sse = f"event: {obj.get('type', 'message')}\ndata: {inner.decode()}\n\n"
                encoded = sse.encode()
                await response.write(encoded)
                bytes_sent += len(encoded)
            except (KeyError, ValueError, json.JSONDecodeError):
                # AWS control/metadata frames have no "bytes" field — skip them.
                continue

    return bytes_sent


async def _stream_passthrough(
    upstream: httpx.Response,
    response: web.StreamResponse,
) -> int:
    bytes_sent = 0
    async for chunk in upstream.aiter_bytes():
        await response.write(chunk)
        bytes_sent += len(chunk)
    return bytes_sent


# --- GatewayProxy -------------------------------------------------------------


class GatewayProxy:
    """Local aiohttp server that proxies Claude SDK LLM calls to the UiPath LLM Gateway.

    Binds to 127.0.0.1 on a random port. Pass the port to the Claude SDK via
    ANTHROPIC_BASE_URL=http://127.0.0.1:{port}.

    Args:
        uipath_base_url: Base URL from UIPATH_URL env var.
        access_token: Bearer token from UIPATH_ACCESS_TOKEN env var.
        agenthub_config: AgentHub billing/consumption config header value.
    """

    def __init__(
        self,
        *,
        uipath_base_url: str,
        access_token: str,
        agenthub_config: str = "agentsruntime",
    ) -> None:
        self._base_url = uipath_base_url.rstrip("/")
        self._access_token = access_token
        self._agenthub_config = agenthub_config
        self._http: httpx.AsyncClient | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._port: int = 0

    @property
    def port(self) -> int:
        """The port the proxy is listening on (0 if not started)."""
        return self._port

    # --- Route handlers ---

    async def _handle_head(self, request: web.Request) -> web.Response:
        # The bundled Bun CLI does a HEAD / connectivity check against ANTHROPIC_BASE_URL.
        return web.Response(status=200)

    async def _handle_count_tokens(self, request: web.Request) -> web.Response:
        # The UiPath gateway does not expose a count_tokens endpoint; return 0 so the
        # Claude SDK does not fail when it probes for token counts.
        logger.debug("count_tokens short-circuited, returning 0")
        return web.json_response({"input_tokens": 0})

    async def _handle_messages(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        raw = await request.read()
        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("invalid JSON body")
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        model_id = payload.get("model", "")
        is_streaming = bool(payload.get("stream", False))

        logger.debug(
            f"Forwarding request: model={model_id}, stream={is_streaming}, "
            f"tools={len(payload.get('tools', []))}, thinking={payload.get('thinking')}"
        )

        url = _gateway_url(self._base_url, model_id)
        headers = _build_headers(
            self._access_token,
            is_streaming=is_streaming,
            agenthub_config=self._agenthub_config,
        )
        body = _transform_body(payload)

        try:
            async with self._http.stream(  # type: ignore[union-attr]
                "POST", url, headers=headers, content=body
            ) as upstream:
                content_type = upstream.headers.get("content-type", "unknown")

                logger.debug(
                    f"Gateway responded with status {upstream.status_code}, content-type {content_type}"
                )

                if upstream.status_code != 200:
                    logger.error(
                        f"Gateway returned error status {upstream.status_code}"
                    )
                    error_body = await upstream.aread()
                    ct = (
                        upstream.headers.get("content-type", "application/json")
                        .split(";")[0]
                        .strip()
                    )
                    return web.Response(
                        status=upstream.status_code, body=error_body, content_type=ct
                    )

                is_bedrock_stream = "amazon.eventstream" in content_type
                resp_ct = "text/event-stream" if is_bedrock_stream else content_type
                response = web.StreamResponse(
                    status=200,
                    headers={"Content-Type": resp_ct, "Cache-Control": "no-cache"},
                )
                await response.prepare(request)

                if is_bedrock_stream:
                    bytes_sent = await _stream_bedrock(upstream, response)
                else:
                    bytes_sent = await _stream_passthrough(upstream, response)

                await response.write_eof()
                logger.debug(f"Forwarded {bytes_sent} bytes ({content_type}→{resp_ct})")
                return response

        except (
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ) as exc:
            logger.error(f"Stream error: {exc}")
            return web.json_response({"error": f"Stream error: {exc}"}, status=502)

    # --- Lifecycle ---

    async def start(self) -> int:
        """Start the proxy server and return the port it is listening on."""
        try:
            self._http = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)

            self._app = web.Application(client_max_size=100 * 1024 * 1024)  # 100 MiB
            self._app.router.add_route("HEAD", "/", self._handle_head)
            self._app.router.add_post(
                "/v1/messages/count_tokens", self._handle_count_tokens
            )
            self._app.router.add_post("/v1/messages", self._handle_messages)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
            await self._site.start()

            # _server.sockets is a private aiohttp attribute — hence the type: ignore.
            sockets = self._site._server.sockets  # type: ignore[union-attr]
            if not sockets:
                raise RuntimeError("Proxy failed to bind — no sockets available")
            self._port = sockets[0].getsockname()[1]
            logger.info(f"gateway proxy started on 127.0.0.1:{self._port}")
            return self._port
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the proxy and release all resources."""
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http:
            await self._http.aclose()
            self._http = None
        self._app = None
        self._port = 0
        logger.info("gateway proxy stopped")

    async def __aenter__(self) -> "GatewayProxy":
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
