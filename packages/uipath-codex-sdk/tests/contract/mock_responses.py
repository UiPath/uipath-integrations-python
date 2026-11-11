"""Local mock of the OpenAI Responses API (SSE) used as Codex's model provider.

Codex is registered against it via ``model_providers.mock.*`` config overrides.
Every request body is recorded so tests can assert on exactly what the model
would have seen. Responses are scripted by a callable supplied by the test.

With ``deltas=True`` an assistant message streams as ``response.output_text.delta``
events (between its ``output_item.added`` and ``output_item.done``), as the LLM
Gateway streams it; Codex turns them into ``item/agentMessage/delta``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from aiohttp import web

JsonObj = dict[str, Any]
# script(request_index, body) -> list of Responses output items, or None for a
# default final assistant message.
Script = Callable[[int, JsonObj], "list[JsonObj] | None"]


def function_call_item(
    call_id: str,
    name: str,
    arguments: JsonObj | None = None,
    namespace: str | None = None,
) -> JsonObj:
    item: JsonObj = {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments or {}),
        "status": "completed",
    }
    if namespace:
        item["namespace"] = namespace
    return item


def message_item(text: str, idx: int = 0) -> JsonObj:
    return {
        "type": "message",
        "id": f"msg_{idx}_{int(time.time() * 1000)}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def advertised_tools(body: JsonObj) -> list[JsonObj]:
    """Tools Codex offered the model in a request.

    Older models get them in the top-level ``tools`` field; newer catalog
    entries (e.g. gpt-5.6) carry them inside the input, as ``tools`` on a
    developer message.
    """
    tools = list(body.get("tools") or [])
    for item in body.get("input") or []:
        if isinstance(item, dict) and isinstance(item.get("tools"), list):
            tools.extend(item["tools"])
    return tools


def find_tool(tools: list[JsonObj], needle: str) -> tuple[str | None, str | None]:
    """Find an advertised tool whose name contains ``needle``.

    Returns (name, namespace). Codex may advertise MCP tools flat
    (``mcp__server__tool``) or grouped under a ``namespace`` entry.
    """
    for t in tools or []:
        name = t.get("name")
        if t.get("type") == "namespace" or isinstance(t.get("tools"), list):
            for sub in t.get("tools") or []:
                sname = sub.get("name") or ""
                if needle in sname:
                    return sname, name
        if name and needle in name:
            return name, None
    return None, None


def input_items(body: JsonObj) -> list[JsonObj]:
    inp = body.get("input")
    return inp if isinstance(inp, list) else []


def last_user_text(body: JsonObj) -> str:
    for item in reversed(input_items(body)):
        if item.get("type") == "message" and item.get("role") == "user":
            parts = item.get("content") or []
            if isinstance(parts, str):
                return parts
            return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return ""


@dataclass
class RecordedRequest:
    index: int
    path: str
    method: str
    body: JsonObj
    received_at: float
    headers: dict[str, str] = field(default_factory=dict)


def split_text(text: str, size: int = 3) -> list[str]:
    """``text`` in word-boundary chunks of about ``size`` words (as deltas)."""
    words = text.split(" ")
    return [
        " ".join(words[i : i + size]) + (" " if i + size < len(words) else "")
        for i in range(0, len(words), size)
    ] or [""]


class MockResponsesServer:
    def __init__(self, script: Script | None = None, deltas: bool = False) -> None:
        self.script: Script | None = script
        #: Stream assistant messages as text deltas.
        self.deltas = deltas
        #: The deltas sent, one list per streamed message.
        self.sent_deltas: list[list[str]] = []
        self.requests: list[RecordedRequest] = []
        self.other_requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._runner: web.AppRunner | None = None
        self.port: int | None = None

    # -- lifecycle (called on the background loop) -------------------------
    async def start(self) -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/responses", self._responses)
        app.router.add_route("*", "/{tail:.*}", self._other)
        self._runner = web.AppRunner(app, access_log=None, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # -- handlers ----------------------------------------------------------
    async def _other(self, request: web.Request) -> web.Response:
        with self._lock:
            self.other_requests.append((request.method, request.path))
        return web.json_response({"error": "not mocked"}, status=404)

    async def _responses(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        body = json.loads(raw) if raw else {}
        with self._lock:
            idx = len(self.requests)
            self.requests.append(
                RecordedRequest(
                    idx,
                    request.path,
                    request.method,
                    body,
                    time.time(),
                    {k: v for k, v in request.headers.items()},
                )
            )
        items: list[JsonObj] | None = None
        if self.script is not None:
            try:
                items = self.script(idx, body)
            except Exception as exc:  # surface script bugs to the model text
                items = [message_item(f"MOCK_SCRIPT_ERROR {exc!r}", idx)]
        if not items:
            items = [message_item(f"final answer {idx}", idx)]

        rid = f"resp_{idx}"
        events: list[JsonObj] = [
            {"type": "response.created", "response": {"id": rid}},
        ]
        for i, item in enumerate(items):
            if self.deltas and item.get("type") == "message":
                events.extend(self._streamed_message(i, item))
                continue
            events.append(
                {"type": "response.output_item.added", "output_index": i, "item": item}
            )
            events.append(
                {"type": "response.output_item.done", "output_index": i, "item": item}
            )
        events.append(
            {
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            }
        )
        data = b"".join(
            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
        )
        resp = web.StreamResponse(
            status=200, headers={"content-type": "text/event-stream"}
        )
        resp.content_length = len(data)
        await resp.prepare(request)
        await resp.write(data)
        await resp.write_eof()
        return resp

    def _streamed_message(self, index: int, item: JsonObj) -> list[JsonObj]:
        text = "".join(
            p.get("text", "") for p in item.get("content") or [] if isinstance(p, dict)
        )
        chunks = split_text(text)
        with self._lock:
            self.sent_deltas.append(chunks)
        empty = {**item, "status": "in_progress", "content": []}
        events: list[JsonObj] = [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": empty,
            },
            {
                "type": "response.content_part.added",
                "item_id": item["id"],
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]
        for chunk in chunks:
            events.append(
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": chunk,
                }
            )
        events.append(
            {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": index,
                "content_index": 0,
                "text": text,
            }
        )
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
        return events

    # -- helpers for tests (thread-safe) -----------------------------------
    def snapshot(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self.requests)

    def wait_for_requests(self, n: int, timeout: float = 30.0) -> list[RecordedRequest]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.snapshot()
            if len(snap) >= n:
                return snap
            time.sleep(0.05)
        return self.snapshot()


def summarize_input(body: JsonObj, max_text: int = 160) -> list[JsonObj]:
    """Compact view of request input items for evidence printing."""
    out = []
    for it in input_items(body):
        d: JsonObj = {
            k: it[k]
            for k in ("type", "role", "call_id", "name", "namespace")
            if k in it
        }
        if "output" in it:
            d["output"] = (
                it["output"]
                if len(json.dumps(it["output"])) < 400
                else str(it["output"])[:400]
            )
        if "arguments" in it:
            d["arguments"] = it["arguments"]
        if "content" in it:
            c = it["content"]
            txt = (
                c
                if isinstance(c, str)
                else " | ".join(p.get("text", "") for p in c if isinstance(p, dict))
            )
            d["text"] = txt[:max_text] + ("..." if len(txt) > max_text else "")
        out.append(d)
    return out


__all__ = [
    "MockResponsesServer",
    "RecordedRequest",
    "function_call_item",
    "message_item",
    "find_tool",
    "split_text",
    "input_items",
    "last_user_text",
    "summarize_input",
]
