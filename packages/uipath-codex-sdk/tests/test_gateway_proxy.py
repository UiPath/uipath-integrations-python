"""Gateway proxy: LLM Gateway sticky-session headers on the upstream calls.

uipath-llm-client puts the headers on the upstream client for a job or a
conversation. The proxy sends them itself for a run that has neither.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from aiohttp import ClientSession
from uipath.llm_client.settings.constants import (
    HEADER_LLMGATEWAY_STICKY_SESSION as STICKY_SESSION,
)
from uipath.llm_client.settings.constants import (
    HEADER_LLMGATEWAY_STICKY_SESSION_TRANSACTION_ID as TRANSACTION_ID,
)

from uipath_codex_sdk.runtime.gateway.proxy import CodexGatewayProxy

API_KEY = "secret"
COMPLETED = {"id": "resp_1", "object": "response", "status": "completed"}
SSE = (
    b"event: response.completed\n"
    b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
)


class RecordingUpstream:
    """Stands in for ``UpstreamRegistry``; records the headers the gateway gets.

    Args:
        client_headers: Default headers of the upstream client, as
            uipath-llm-client sets them (the sticky-session ones for a job).
    """

    def __init__(self, client_headers: dict[str, str] | None = None) -> None:
        self.headers: list[httpx.Headers] = []
        self._client = httpx.AsyncClient(
            base_url="http://gateway.test",
            headers=client_headers,
            transport=httpx.MockTransport(self._respond),
        )

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=SSE, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=COMPLETED)

    async def get(self, model: str) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    def transaction_ids(self) -> list[str | None]:
        return [h.get(TRANSACTION_ID) for h in self.headers]


async def _relay(
    upstream: RecordingUpstream,
    bodies: list[dict[str, Any]],
    headers: dict[str, str] | None = None,
) -> None:
    """Post ``bodies`` to a proxy as Codex would, with Codex's ``headers``."""
    proxy = CodexGatewayProxy(
        upstream=upstream,  # type: ignore[arg-type]
        api_key=API_KEY,
        default_model="m",
    )
    await proxy.start()
    try:
        async with ClientSession() as session:
            for body in bodies:
                async with session.post(
                    f"http://127.0.0.1:{proxy.port}/v1/responses",
                    json=body,
                    headers={"Authorization": f"Bearer {API_KEY}", **(headers or {})},
                ) as response:
                    assert response.status == 200
                    await response.read()
    finally:
        await proxy.stop()
        await upstream.aclose()


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
async def test_unpinned_run_is_pinned_with_the_codex_thread_id(stream: bool) -> None:
    upstream = RecordingUpstream()
    await _relay(
        upstream,
        [{"input": "hi", "stream": stream}] * 2,
        headers={"session-id": "thread-1", "thread-id": "thread-1"},
    )
    assert len(upstream.headers) == 2
    for headers in upstream.headers:
        assert headers[STICKY_SESSION] == "true"
        assert headers[TRANSACTION_ID] == "thread-1"


@pytest.mark.parametrize(
    ("codex_headers", "expected"),
    [
        ({"session-id": " thread-1 "}, "thread-1"),
        ({"thread-id": "thread-2"}, "thread-2"),
        ({"session-id": " ", "thread-id": "thread-2"}, "thread-2"),
        ({"session-id": "x" * 200}, "x" * 128),
    ],
    ids=["trimmed", "thread-id", "blank-session-id", "capped"],
)
async def test_codex_thread_id_header_forms(
    codex_headers: dict[str, str], expected: str
) -> None:
    upstream = RecordingUpstream()
    await _relay(upstream, [{"input": "hi"}], headers=codex_headers)
    assert upstream.transaction_ids() == [expected]


async def test_llm_client_sticky_headers_are_left_alone() -> None:
    # A job: uipath-llm-client put the headers, with the job key, on the client.
    upstream = RecordingUpstream({STICKY_SESSION: "true", TRANSACTION_ID: "job-1"})
    await _relay(
        upstream,
        [{"input": "a"}, {"input": "b", "stream": True}],
        headers={"session-id": "thread-1"},
    )
    assert upstream.transaction_ids() == ["job-1", "job-1"]
    assert [h[STICKY_SESSION] for h in upstream.headers] == ["true", "true"]


async def test_without_a_codex_thread_id_the_proxy_uses_one_id_of_its_own() -> None:
    upstream = RecordingUpstream()
    await _relay(upstream, [{"input": "a"}, {"input": "b", "stream": True}])
    first, second = upstream.transaction_ids()
    assert first and first == second
    assert [h[STICKY_SESSION] for h in upstream.headers] == ["true", "true"]

    other = RecordingUpstream()
    await _relay(other, [{"input": "a"}])
    assert other.transaction_ids() != [first]
