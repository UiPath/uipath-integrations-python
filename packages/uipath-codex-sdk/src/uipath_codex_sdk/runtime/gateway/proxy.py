"""Loopback listener that routes Codex's LLM calls to the UiPath gateway.

The bundled ``codex`` binary (not the Python package) makes the HTTP calls, and
with a custom ``model_provider`` it POSTs the OpenAI **Responses API** to a
configured ``base_url``. A loopback listener is therefore the only way to
intercept them and attach the UiPath auth + routing headers.

Because the UiPath gateway speaks the Responses API natively (proven: OpenAI
models advertise the ``OpenAiResponses`` flavor), this is a near pass-through
relay — the request/response bodies are forwarded verbatim; only the model is
read from the body to pick the per-model upstream client.
"""

from __future__ import annotations

import gzip
import hmac
import json
import logging
import socket
import time
from typing import Any, Protocol

import httpx
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError
from opentelemetry.context import Context

from ..telemetry import (
    ModelCallSpan,
    SseCompletedReader,
)
from .upstream import UPSTREAM_PATH, UpstreamRegistry

# Codex closes the connection the moment it has the final SSE event, which can
# land mid-write on our side. These are benign client disconnects, not errors.
_CLIENT_GONE = (ConnectionResetError, ClientConnectionResetError, RuntimeError)

logger = logging.getLogger(__name__)

CLIENT_MAX_SIZE = 100 * 1024 * 1024
SHUTDOWN_TIMEOUT = 5.0
GZIP_MAGIC = b"\x1f\x8b"
#: Path the durable tool host (an MCP endpoint) is mounted on.
MCP_PATH = "/mcp"


class MountedHandler(Protocol):
    """An extra endpoint served next to the model routes (the tool host)."""

    async def handle(self, request: web.Request) -> web.StreamResponse: ...


class CodexGatewayProxy:
    """aiohttp application bound to 127.0.0.1 on an OS-assigned port.

    Args:
        upstream: Per-model upstream client registry.
        api_key: Per-run secret Codex must present on every request.
        default_model: Model to use when a request body omits ``model``.
        mcp_handler: Optional endpoint mounted on ``/mcp`` (the durable tool
            host). It does its own authentication, with its own token.
        trace_context: Context the ``LLM call`` spans are opened in (the
            run's ``Agent run`` span). The handlers run on aiohttp's tasks, so
            the parent is passed rather than taken from the current context.
    """

    def __init__(
        self,
        *,
        upstream: UpstreamRegistry,
        api_key: str,
        default_model: str | None = None,
        mcp_handler: MountedHandler | None = None,
        trace_context: Context | None = None,
    ) -> None:
        self._upstream = upstream
        self._trace_context = trace_context
        self._api_key = api_key.encode()
        self._default_model = default_model
        self._mcp_handler = mcp_handler
        self._socket: socket.socket | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.SockSite | None = None
        self._port = 0
        self._inflight = 0
        self._last_activity = time.monotonic()

    @property
    def port(self) -> int:
        return self._port

    @property
    def inflight(self) -> int:
        """Model (``/responses``) requests currently being relayed."""
        return self._inflight

    @property
    def last_activity(self) -> float:
        """``time.monotonic()`` of the last model request start or end."""
        return self._last_activity

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            self._port = sock.getsockname()[1]
            self._socket = sock

            app = web.Application(
                client_max_size=CLIENT_MAX_SIZE,
                middlewares=[self._authenticate],
            )
            # Codex base_url is "http://127.0.0.1:<port>/v1"; it appends
            # "/responses". Accept both with and without the /v1 prefix.
            app.router.add_post("/v1/responses", self._handle_responses)
            app.router.add_post("/responses", self._handle_responses)
            app.router.add_get("/v1/models", self._handle_models)
            app.router.add_get("/models", self._handle_models)
            if self._mcp_handler is not None:
                app.router.add_route("*", MCP_PATH, self._mcp_handler.handle)

            self._runner = web.AppRunner(app, shutdown_timeout=SHUTDOWN_TIMEOUT)
            await self._runner.setup()
            self._site = web.SockSite(self._runner, sock)
            await self._site.start()
        except Exception:
            sock.close()
            self._socket = None
            self._port = 0
            await self.stop()
            raise
        logger.info("codex gateway proxy listening on 127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._port = 0

    # --- authentication ---------------------------------------------------

    @web.middleware
    async def _authenticate(
        self, request: web.Request, handler: Any
    ) -> web.StreamResponse:
        if self._mcp_handler is not None and request.path == MCP_PATH:
            return await handler(request)  # the tool host checks its own token
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        presented = token if scheme.lower() == "bearer" else authorization
        if not presented:
            presented = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(presented.encode(), self._api_key):
            return _error_response(401, "Invalid API key for the UiPath codex gateway.")
        return await handler(request)

    # --- handlers ---------------------------------------------------------

    async def _handle_models(self, request: web.Request) -> web.Response:
        data = []
        if self._default_model:
            data.append({"id": self._default_model, "object": "model"})
        return web.json_response({"object": "list", "data": data})

    async def _handle_responses(self, request: web.Request) -> web.StreamResponse:
        self._inflight += 1
        self._last_activity = time.monotonic()
        try:
            return await self._relay_responses(request)
        finally:
            self._inflight -= 1
            self._last_activity = time.monotonic()

    async def _relay_responses(self, request: web.Request) -> web.StreamResponse:
        raw = await _read_body(request)
        if raw is None:
            return _error_response(400, "Request body is not valid gzip.")
        try:
            payload = json.loads(raw)
        except ValueError:
            return _error_response(400, "Request body is not valid JSON.")
        if not isinstance(payload, dict):
            return _error_response(400, "Request body must be a JSON object.")

        model = str(payload.get("model") or self._default_model or "")
        if not model:
            return _error_response(
                400, "No model in request and no default configured."
            )

        span = ModelCallSpan(self._trace_context, payload, model)

        client = await self._upstream.get(model)
        body = json.dumps(payload).encode()
        headers = {"content-type": "application/json", **span.trace_headers()}
        try:
            if payload.get("stream"):
                response, completed = await self._forward_stream(
                    request, client, body, headers
                )
            else:
                response, completed = await self._forward_unary(client, body, headers)
        except BaseException as exc:
            span.finish(error=str(exc) or type(exc).__name__)
            raise
        status = getattr(response, "status", None)
        span.finish(
            response=completed,
            status=status,
            error=None if status == 200 else f"HTTP {status}",
        )
        return response

    # --- forwarding -------------------------------------------------------

    async def _forward_unary(
        self, client: httpx.AsyncClient, body: bytes, headers: dict[str, str]
    ) -> tuple[web.StreamResponse, dict[str, Any] | None]:
        try:
            upstream = await client.request(
                "POST", UPSTREAM_PATH, headers=headers, content=body
            )
        except httpx.HTTPError as exc:
            return (
                _error_response(502, f"UiPath LLM Gateway is unreachable: {exc}"),
                None,
            )

        completed: dict[str, Any] | None = None
        if upstream.status_code == 200:
            try:
                parsed = upstream.json()
                completed = parsed if isinstance(parsed, dict) else None
            except ValueError:
                completed = None
        return (
            web.Response(
                status=upstream.status_code,
                body=upstream.content,
                content_type="application/json",
            ),
            completed,
        )

    async def _forward_stream(
        self,
        request: web.Request,
        client: httpx.AsyncClient,
        body: bytes,
        headers: dict[str, str],
    ) -> tuple[web.StreamResponse, dict[str, Any] | None]:
        reader = SseCompletedReader()
        try:
            async with client.stream(
                "POST", UPSTREAM_PATH, headers=headers, content=body
            ) as upstream:
                if upstream.status_code != 200:
                    raw = await upstream.aread()
                    return (
                        web.Response(
                            status=upstream.status_code,
                            body=raw,
                            content_type="application/json",
                        ),
                        None,
                    )
                response = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                await response.prepare(request)
                try:
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            reader.feed(chunk)
                            await response.write(chunk)
                    await response.write_eof()
                except _CLIENT_GONE as exc:
                    logger.debug("codex client disconnected during stream: %s", exc)
                return response, reader.response
        except httpx.HTTPError as exc:
            return (
                _error_response(502, f"UiPath LLM Gateway is unreachable: {exc}"),
                None,
            )


# --- helpers --------------------------------------------------------------


async def _read_body(request: web.Request) -> bytes | None:
    raw = await request.read()
    if not raw.startswith(GZIP_MAGIC):
        return raw
    try:
        return gzip.decompress(raw)
    except (OSError, EOFError):
        return None


def _error_response(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "uipath_codex_gateway_error"}},
        status=status,
    )


__all__ = [
    "CLIENT_MAX_SIZE",
    "MCP_PATH",
    "CodexGatewayProxy",
    "MountedHandler",
]
