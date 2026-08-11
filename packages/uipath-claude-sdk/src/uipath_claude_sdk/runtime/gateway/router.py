"""Local listener that routes Claude Code's HTTP calls to the UiPath gateway.

The bundled CLI, not the Python package, makes the HTTP calls, and it POSTs a
fixed path. A loopback listener is therefore the only way to intercept them,
and ``ANTHROPIC_BASE_URL`` is the only routing channel available.
"""

from __future__ import annotations

import gzip
import hmac
import json
import logging
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from aiohttp import web
from uipath.llm_client.utils.headers import extract_matching_headers

from .catalog import ModelCatalog, ResolvedModel, ResolvedModelSet
from .errors import (
    GatewayShimError,
    error_payload,
    to_anthropic_error,
)
from .records import (
    GatewayCallRecord,
    GatewayCallSink,
    TraceHeaderSource,
    UsageTee,
)
from .strategies import GatewayStrategy, UpstreamRequest, select_strategy
from .upstream import UPSTREAM_PATH, UpstreamRegistry

logger = logging.getLogger(__name__)

CLIENT_MAX_SIZE = 100 * 1024 * 1024
SHUTDOWN_TIMEOUT = 5.0
GZIP_MAGIC = b"\x1f\x8b"
CAPTURED_HEADER_PREFIXES = ("x-uipath-", "request-id", "anthropic-ratelimit-")

_COUNT_TOKENS_MESSAGE = (
    "The UiPath LLM Gateway does not expose a token counting endpoint, so the "
    "UiPath gateway shim cannot answer /v1/messages/count_tokens."
)


@dataclass
class _Call:
    """One in-flight request, from parse through to the emitted record."""

    request: web.Request
    client: httpx.AsyncClient
    upstream: UpstreamRequest
    strategy: GatewayStrategy
    resolved: ResolvedModel
    tee: UsageTee = field(default_factory=UsageTee)
    started: float = field(default_factory=time.monotonic)
    traced_upstream: bool = False


class GatewayRouter:
    """aiohttp application bound to 127.0.0.1 on an OS-assigned port."""

    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        upstream: UpstreamRegistry,
        models: ResolvedModelSet,
        api_key: str,
        on_call: GatewayCallSink | None = None,
        trace_headers: TraceHeaderSource | None = None,
    ) -> None:
        self._catalog = catalog
        self._upstream = upstream
        self._models = models
        self._api_key = api_key.encode()
        self._on_call = on_call
        self._trace_headers = trace_headers
        self._socket: socket.socket | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.SockSite | None = None
        self._port = 0

    @property
    def port(self) -> int:
        """The bound port, or 0 before the router is started."""
        return self._port

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Bind the loopback socket and serve. The port is known before serving."""
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
            app.router.add_get("/api/hello", self._handle_hello, allow_head=True)
            app.router.add_post("/v1/messages/count_tokens", self._handle_count_tokens)
            app.router.add_post("/v1/messages", self._handle_messages)
            app.router.add_get("/v1/models", self._handle_models)

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
        logger.info("gateway shim listening on 127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        """Drain in-flight requests within the shutdown timeout and release the port."""
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
        presented = request.headers.get("x-api-key")
        if not presented:
            authorization = request.headers.get("Authorization", "")
            scheme, _, token = authorization.partition(" ")
            presented = token if scheme.lower() == "bearer" else authorization
        if not hmac.compare_digest(presented.encode(), self._api_key):
            return _error_response(
                401, "Invalid API key for the local UiPath gateway shim."
            )
        return await handler(request)

    # --- handlers ---------------------------------------------------------

    async def _handle_hello(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _handle_count_tokens(self, request: web.Request) -> web.Response:
        return _error_response(404, _COUNT_TOKENS_MESSAGE)

    async def _handle_models(self, request: web.Request) -> web.Response:
        data = [
            {"type": "model", "id": name, "display_name": name}
            for name in self._catalog.available
        ]
        return web.json_response({"data": data, "has_more": False})

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        raw = await _read_body(request)
        if raw is None:
            return _error_response(400, "Request body is not valid gzip.")
        try:
            payload = json.loads(raw)
        except ValueError:
            return _error_response(400, "Request body is not valid JSON.")
        if not isinstance(payload, dict):
            return _error_response(400, "Request body must be a JSON object.")

        try:
            resolved = (
                self._catalog.resolve(str(payload["model"]))
                if payload.get("model")
                else self._models.primary
            )
            strategy = select_strategy(resolved)
        except GatewayShimError as exc:
            return _error_response(exc.status, str(exc), error_type=exc.error_type)

        upstream_request = strategy.build_request(payload, request.headers, resolved)
        call = _Call(
            request=request,
            client=await self._upstream.get(resolved),
            upstream=upstream_request,
            strategy=strategy,
            resolved=resolved,
            traced_upstream=self._attach_trace_context(upstream_request),
        )
        if call.upstream.stream:
            return await self._forward_stream(call)
        return await self._forward_unary(call)

    # --- forwarding -------------------------------------------------------

    async def _forward_unary(self, call: _Call) -> web.StreamResponse:
        try:
            upstream = await call.client.request(
                "POST",
                UPSTREAM_PATH,
                headers=call.upstream.headers,
                content=call.upstream.body,
            )
        except httpx.HTTPError as exc:
            self._emit(call, 502, error=str(exc))
            return _error_response(502, f"UiPath LLM Gateway is unreachable: {exc}")

        captured = _captured_headers(upstream.headers)
        if upstream.status_code != 200:
            self._emit(
                call, upstream.status_code, error="upstream error", headers=captured
            )
            return _upstream_error_response(
                upstream.status_code, upstream.content, captured
            )

        try:
            call.tee.feed_message(json.loads(upstream.content))
        except ValueError:
            logger.debug("non-JSON body on a non-streaming gateway response")
        self._emit(call, 200, headers=captured)
        return web.Response(
            status=200, body=upstream.content, content_type="application/json"
        )

    async def _forward_stream(self, call: _Call) -> web.StreamResponse:
        try:
            async with call.client.stream(
                "POST",
                UPSTREAM_PATH,
                headers=call.upstream.headers,
                content=call.upstream.body,
            ) as upstream:
                captured = _captured_headers(upstream.headers)
                if upstream.status_code != 200:
                    body = await upstream.aread()
                    self._emit(
                        call,
                        upstream.status_code,
                        error="upstream error",
                        headers=captured,
                    )
                    return _upstream_error_response(
                        upstream.status_code, body, captured
                    )
                return await self._pump(call, upstream, captured)
        except httpx.HTTPError as exc:
            self._emit(call, 502, error=str(exc))
            return _error_response(502, f"UiPath LLM Gateway is unreachable: {exc}")

    async def _pump(
        self,
        call: _Call,
        upstream: httpx.Response,
        captured: dict[str, str],
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(call.request)
        try:
            async for chunk in call.strategy.stream_response(upstream):
                call.tee.feed(chunk)
                await response.write(chunk)
        except Exception as exc:
            logger.error("gateway stream failed mid-response: %s", exc)
            await _write_stream_error(response, exc)
            self._emit(call, 502, error=str(exc), headers=captured)
        else:
            self._emit(call, 200, headers=captured)
        await response.write_eof()
        return response

    # --- telemetry --------------------------------------------------------

    def _attach_trace_context(self, upstream: UpstreamRequest) -> bool:
        """Let the gateway record the model span under the turn that caused it.

        The gateway builds its own span from these headers, so a call that
        carries them must not also be given one here.
        """
        if self._trace_headers is None:
            return False
        try:
            headers = self._trace_headers()
        except Exception:
            logger.debug("trace context headers unavailable", exc_info=True)
            return False
        if not headers:
            return False
        upstream.headers.update(headers)
        return True

    def _emit(
        self,
        call: _Call,
        status: int,
        *,
        error: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if self._on_call is None:
            return
        record = GatewayCallRecord(
            requested_model=call.resolved.requested,
            resolved_model=call.resolved.model_id,
            vendor_type=call.resolved.vendor_type,
            api_flavor=call.resolved.api_flavor,
            streaming=call.upstream.stream,
            status=status,
            duration_ms=(time.monotonic() - call.started) * 1000,
            response_headers=headers or {},
            error=error,
            request_body=_decode_request(call.upstream.body),
            traced_upstream=call.traced_upstream,
        )
        call.tee.apply(record)
        try:
            self._on_call(record)
        except Exception:
            logger.debug("gateway call sink raised", exc_info=True)


def _decode_request(body: bytes) -> dict[str, Any] | None:
    """The request as sent upstream, for the tracing consumer."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


# --- helpers --------------------------------------------------------------


async def _read_body(request: web.Request) -> bytes | None:
    """Read the request body, decompressing gzip aiohttp left encoded."""
    raw = await request.read()
    if not raw.startswith(GZIP_MAGIC):
        return raw
    try:
        return gzip.decompress(raw)
    except (OSError, EOFError):
        return None


def _captured_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return extract_matching_headers(httpx.Headers(headers), CAPTURED_HEADER_PREFIXES)


def _correlation_id(headers: Mapping[str, str]) -> str | None:
    for name, value in headers.items():
        if name.lower().endswith(("correlation-id", "request-id")):
            return value
    return None


def _error_response(
    status: int, message: str, *, error_type: str | None = None
) -> web.Response:
    return web.json_response(
        error_payload(status, message, error_type=error_type), status=status
    )


def _upstream_error_response(
    status: int, body: bytes, headers: dict[str, str]
) -> web.Response:
    payload = to_anthropic_error(status, body, correlation_id=_correlation_id(headers))
    return web.json_response(payload, status=status)


async def _write_stream_error(response: web.StreamResponse, exc: Exception) -> None:
    """Report a mid-stream failure inside the already committed SSE stream.

    Once the response is prepared the status is on the wire, so the only way to
    reach the CLI's error path is an ``error`` frame.
    """
    payload = error_payload(502, f"UiPath LLM Gateway stream failed: {exc}")
    frame = b"event: error\ndata: " + json.dumps(payload).encode() + b"\n\n"
    try:
        await response.write(frame)
    except (ConnectionResetError, RuntimeError):
        logger.debug("client gone before the stream error frame was written")


__all__ = ["CAPTURED_HEADER_PREFIXES", "CLIENT_MAX_SIZE", "GatewayRouter"]
