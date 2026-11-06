"""In-process "durable tools" MCP endpoint (design A-prime).

Stateless streamable-HTTP MCP server answering with plain JSON (no SSE),
guarded by a bearer token. ``tools/call`` requests for durable tools are
recorded (including ``params._meta``) and then *held forever*: the handler
never answers, so the Codex turn stays blocked on the call until the
app-server is killed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from aiohttp import web

JsonObj = dict[str, Any]

DURABLE_TOOL = "start_job"
SERVER_NAME = "uipath_tools"

TOOLS: list[JsonObj] = [
    {
        "name": DURABLE_TOOL,
        "description": "Start a long-running UiPath Orchestrator job and return its output.",
        "inputSchema": {
            "type": "object",
            "properties": {"job": {"type": "string"}},
            "required": ["job"],
        },
    }
]


@dataclass
class HeldCall:
    jsonrpc_id: Any
    params: JsonObj
    received_at: float
    # Snapshot taken *inside the handler* (synchronously, on arrival) by the
    # optional on_call hook, e.g. what the rollout file looked like then.
    arrival_snapshot: Any = None
    released_at: float | None = None
    disconnected: bool = False

    @property
    def meta(self) -> JsonObj:
        return self.params.get("_meta") or {}

    @property
    def call_id(self) -> str | None:
        return self.meta.get("callId")


@dataclass
class RpcLogEntry:
    method: str | None
    body: Any
    auth: str | None
    at: float
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


class DurableMcpHost:
    def __init__(
        self, token: str, on_call: Callable[[JsonObj], Any] | None = None
    ) -> None:
        self.token = token
        self.on_call = on_call
        self.held: list[HeldCall] = []
        self.log: list[RpcLogEntry] = []
        self.concurrent_now = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self._runner: web.AppRunner | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stopping = False
        self.port: int | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/mcp", self._mcp)
        # handler_cancellation: a held handler is cancelled when the client
        # (codex) drops the HTTP connection -> lets tests observe disconnects.
        self._runner = web.AppRunner(
            app, access_log=None, shutdown_timeout=1.0, handler_cancellation=True
        )
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def stop(self) -> None:
        # Held handlers never finish on their own; cancel them so the
        # server can shut down promptly.
        self._stopping = True
        for t in list(self._tasks):
            t.cancel()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def _record(self, entry: RpcLogEntry) -> None:
        with self._lock:
            self.log.append(entry)

    async def _mcp(self, request: web.Request) -> web.StreamResponse:
        auth = request.headers.get("Authorization")
        hdrs = {k: v for k, v in request.headers.items()}
        if request.method != "POST":
            # Stateless server: no standalone SSE stream, no sessions.
            self._record(
                RpcLogEntry(
                    f"HTTP {request.method}", None, auth, time.time(), 405, hdrs
                )
            )
            return web.Response(status=405, headers={"Allow": "POST"})
        if auth != f"Bearer {self.token}":
            self._record(
                RpcLogEntry("UNAUTHORIZED", None, auth, time.time(), 401, hdrs)
            )
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            msg = json.loads(await request.read())
        except Exception:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "parse error"},
                },
                status=400,
            )
        if isinstance(msg, list):  # batches are not expected from Codex
            self._record(RpcLogEntry("BATCH", msg, auth, time.time(), 400, hdrs))
            return web.Response(status=400)

        method = msg.get("method")
        mid = msg.get("id")
        self._record(RpcLogEntry(method, msg, auth, time.time(), 200, hdrs))

        if mid is None:  # notification / response from client
            return web.Response(status=202)

        if method == "initialize":
            pv = (msg.get("params") or {}).get("protocolVersion") or "2025-06-18"
            return self._ok(
                mid,
                {
                    "protocolVersion": pv,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "uipath-durable-tools", "version": "0.0.1"},
                },
            )
        if method == "ping":
            return self._ok(mid, {})
        if method == "tools/list":
            return self._ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            return await self._hold(request, mid, msg.get("params") or {})
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        )

    def _ok(self, mid: Any, result: JsonObj) -> web.Response:
        return web.json_response({"jsonrpc": "2.0", "id": mid, "result": result})

    async def _hold(
        self, request: web.Request, mid: Any, params: JsonObj
    ) -> web.StreamResponse:
        call = HeldCall(mid, params, time.time())
        if self.on_call is not None:
            try:
                call.arrival_snapshot = self.on_call(params)
            except Exception as exc:  # pragma: no cover - evidence only
                call.arrival_snapshot = f"on_call error: {exc!r}"
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        with self._lock:
            self.held.append(call)
            self.concurrent_now += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_now)
        try:
            # Never answer. Durable result is delivered later via thread/inject_items.
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        except asyncio.CancelledError:
            # cancelled by aiohttp because codex closed the connection
            # (unless we are shutting the host down ourselves)
            call.disconnected = not self._stopping
            raise
        finally:
            call.released_at = time.time()
            if task is not None:
                self._tasks.discard(task)
            with self._lock:
                self.concurrent_now -= 1

    # -- thread-safe helpers ------------------------------------------------
    def held_snapshot(self) -> list[HeldCall]:
        with self._lock:
            return list(self.held)

    def wait_for_held(self, n: int, timeout: float = 30.0) -> list[HeldCall]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.held_snapshot()
            if len(snap) >= n:
                return snap
            time.sleep(0.05)
        return self.held_snapshot()

    def methods(self) -> list[str | None]:
        with self._lock:
            return [e.method for e in self.log]
