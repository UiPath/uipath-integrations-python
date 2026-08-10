"""Tests for the gateway routing shim.

The router is exercised end to end against a fake UiPath gateway, over a real
``UiPathHttpxAsyncClient`` upstream leg, so header injection and URL freezing
are covered rather than mocked away.
"""

from __future__ import annotations

import base64
import binascii
import functools
import gzip
import json
import socket
import struct
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from aiohttp import web
from uipath.llm_client import UiPathHttpxAsyncClient

from uipath_claude_sdk.runtime.gateway import GatewayShim, upstream
from uipath_claude_sdk.runtime.gateway.catalog import (
    ModelCatalog,
    alias_candidates,
)
from uipath_claude_sdk.runtime.gateway.errors import (
    ModelNotInCatalogError,
    UnsupportedApiFlavorError,
    to_anthropic_error,
)
from uipath_claude_sdk.runtime.gateway.records import UsageTee
from uipath_claude_sdk.runtime.gateway.strategies import (
    AnthropicMessagesStrategy,
    BedrockInvokeStrategy,
    select_strategy,
)

SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
SONNET_NAME = "claude-sonnet-4-5"
"""What a developer writes. Discovery picks the route, never the name, so this
is what goes upstream, exactly as uipath-langchain sends it."""
HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"

DISCOVERY: list[dict[str, Any]] = [
    {"modelName": "gpt-5.2-2025-12-11", "vendor": "OpenAi", "apiFlavor": None},
    {"modelName": SONNET, "vendor": "AwsBedrock", "apiFlavor": "AnthropicMessages"},
    {"modelName": HAIKU, "vendor": "AwsBedrock", "apiFlavor": "AnthropicMessages"},
    {"modelName": OPUS, "vendor": "AwsBedrock", "apiFlavor": None},
]

SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
    b'{"input_tokens":11,"cache_read_input_tokens":3}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":'
    b'{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}\n\n'
)


def eventstream_frame(event: dict[str, Any]) -> bytes:
    """Encode one AWS Event Stream frame carrying a base64 Anthropic event."""
    inner = json.dumps(event).encode()
    payload = json.dumps({"bytes": base64.b64encode(inner).decode()}).encode()
    prelude = struct.pack(">II", 16 + len(payload), 0)
    frame = prelude + struct.pack(">I", binascii.crc32(prelude)) + payload
    return frame + struct.pack(">I", binascii.crc32(frame))


# --- fixtures --------------------------------------------------------------


@pytest.fixture
async def gateway() -> AsyncIterator[dict[str, Any]]:
    """A fake UiPath gateway that records what the shim forwarded."""
    seen: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        body = json.loads(raw)
        seen.append({"headers": dict(request.headers), "body": body, "raw": raw})
        match body.get("mode"):
            case "error":
                return web.json_response(
                    {"error": {"message": "denied", "correlationId": "corr-9"}},
                    status=403,
                )
            case "ratelimit":
                return web.json_response(
                    {"error": {"message": "too many requests"}},
                    status=429,
                    headers={"anthropic-ratelimit-requests-remaining": "0"},
                )
            case "unary":
                return web.json_response(
                    {"usage": {"input_tokens": 5, "output_tokens": 6}}
                )
            case "midstream":
                partial = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await partial.prepare(request)
                await partial.write(b"event: ping\ndata: {}\n\n")
                request.transport.abort()  # type: ignore[union-attr]
                return partial
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "x-uipath-correlation-id": "c1",
            }
        )
        await response.prepare(request)
        await response.write(SSE)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield {"base_url": f"http://127.0.0.1:{port}", "seen": seen}
    await runner.cleanup()


@pytest.fixture
async def shim(
    gateway: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[GatewayShim]:
    """A started shim whose upstream leg does not retry.

    The LLM client retries 429 and 5xx with a multi-second backoff of its own,
    which no unit test should sit through.
    """
    monkeypatch.setattr(
        upstream,
        "UiPathHttpxAsyncClient",
        functools.partial(UiPathHttpxAsyncClient, max_retries=0),
    )

    settings = MagicMock()
    settings.get_available_models.return_value = DISCOVERY
    settings.build_base_url.return_value = gateway["base_url"]
    settings.build_auth_headers.return_value = {}
    settings.build_auth_pipeline.return_value = None

    records: list[Any] = []
    started = GatewayShim(
        MagicMock(model="claude-sonnet-4-5"),
        settings=settings,
        on_call=records.append,
    )
    started.records = records  # type: ignore[attr-defined]
    await started.start()
    yield started
    await started.stop()


@pytest.fixture
def client(shim: GatewayShim) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=shim.base_url, headers={"x-api-key": shim.api_key}
    )


# --- catalog ---------------------------------------------------------------


class TestCatalog:
    def test_exact_id(self):
        assert ModelCatalog(DISCOVERY).resolve(SONNET).model_id == SONNET

    @pytest.mark.parametrize(
        "requested",
        [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929-v1:0",
            "Claude-Sonnet-4-5",
            "anthropic.claude-sonnet-4-5",
        ],
    )
    def test_aliases_reach_the_tenant_id(self, requested):
        assert ModelCatalog(DISCOVERY).resolve(requested).model_id == SONNET

    def test_family_word_picks_the_newest(self):
        catalog = ModelCatalog(DISCOVERY)
        assert catalog.resolve("haiku").model_id == HAIKU
        assert catalog.resolve("opus").model_id == OPUS

    def test_vertex_style_alias(self):
        assert "claude-sonnet-4-5" in alias_candidates("claude-sonnet-4-5@20250929")

    def test_discovered_flavor_wins(self):
        assert ModelCatalog(DISCOVERY).resolve("claude-sonnet-4-5").api_flavor == (
            "AnthropicMessages"
        )

    def test_bedrock_without_a_flavor_defaults_to_invoke(self):
        resolved = ModelCatalog(DISCOVERY).resolve("opus")
        assert (resolved.vendor_type, resolved.api_flavor) == ("awsbedrock", "invoke")

    def test_unknown_model_lists_what_the_tenant_has(self):
        with pytest.raises(ModelNotInCatalogError) as exc:
            ModelCatalog(DISCOVERY).resolve("claude-nonesuch")
        assert SONNET in str(exc.value)

    def test_family_falls_back_when_absent(self):
        catalog = ModelCatalog([DISCOVERY[1]])
        assert catalog.resolve_set("claude-sonnet-4-5").opus.model_id == SONNET


# --- strategies ------------------------------------------------------------


class TestStrategies:
    def test_selection_follows_discovery(self):
        catalog = ModelCatalog(DISCOVERY)
        assert isinstance(
            select_strategy(catalog.resolve("claude-sonnet-4-5")),
            AnthropicMessagesStrategy,
        )
        assert isinstance(
            select_strategy(catalog.resolve("opus")), BedrockInvokeStrategy
        )

    def test_unroutable_flavor_is_rejected(self):
        catalog = ModelCatalog(DISCOVERY)
        with pytest.raises(UnsupportedApiFlavorError):
            select_strategy(catalog.resolve("gpt-5.2-2025-12-11"))

    def test_anthropic_messages_sends_the_name_the_developer_wrote(self):
        resolved = ModelCatalog(DISCOVERY).resolve("claude-sonnet-4-5")
        payload = {
            "model": "claude-sonnet-4-5",
            "context_management": {"edits": []},
            "system": [{"type": "text", "text": "hi"}],
        }
        body = json.loads(
            AnthropicMessagesStrategy().build_request(payload, {}, resolved).body
        )
        assert body["model"] == SONNET_NAME
        assert body["system"] == [{"type": "text", "text": "hi"}]
        assert "context_management" not in body

    def test_no_phantom_system_key(self):
        resolved = ModelCatalog(DISCOVERY).resolve("claude-sonnet-4-5")
        body = json.loads(
            AnthropicMessagesStrategy().build_request({}, {}, resolved).body
        )
        assert "system" not in body

    def test_invoke_moves_transport_fields_out_of_the_body(self):
        resolved = ModelCatalog(DISCOVERY).resolve("opus")
        payload = {"model": "x", "stream": True, "system": [{"type": "text"}]}
        body = json.loads(
            BedrockInvokeStrategy().build_request(payload, {}, resolved).body
        )
        assert "model" not in body
        assert "stream" not in body
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["system"] == [{"type": "text"}]

    async def test_invoke_reemits_eventstream_frames_as_sse(self):
        upstream = httpx.Response(
            200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            content=eventstream_frame({"type": "message_start", "message": {}}),
        )
        out = b"".join(
            [c async for c in BedrockInvokeStrategy().stream_response(upstream)]
        )
        assert out.startswith(b"event: message_start\ndata: ")


# --- errors ----------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (429, "rate_limit_error"),
            (503, "overloaded_error"),
            (529, "overloaded_error"),
            (500, "api_error"),
        ],
    )
    def test_status_mapping(self, status, expected):
        assert to_anthropic_error(status, b"{}")["error"]["type"] == expected

    def test_gateway_message_and_correlation_survive(self):
        payload = to_anthropic_error(
            429, b'{"error":{"message":"slow down","correlationId":"abc"}}'
        )
        assert payload["error"]["message"] == "slow down"
        assert payload["request_id"] == "abc"

    def test_non_json_body_is_kept(self):
        payload = to_anthropic_error(502, b"<html>gateway down</html>")
        assert "gateway down" in payload["error"]["message"]


# --- usage tee -------------------------------------------------------------


class TestUsageTee:
    def test_counts_survive_arbitrary_chunk_boundaries(self):
        tee = UsageTee()
        for start in range(0, len(SSE), 7):
            tee.feed(SSE[start : start + 7])
        assert (tee.input_tokens, tee.output_tokens) == (11, 42)
        assert tee.cache_read_input_tokens == 3
        assert tee.stop_reason == "end_turn"

    def test_non_streaming_body(self):
        tee = UsageTee()
        tee.feed_message({"usage": {"input_tokens": 1, "output_tokens": 2}})
        assert (tee.input_tokens, tee.output_tokens) == (1, 2)


# --- environment -----------------------------------------------------------


class TestEnv:
    def test_every_auxiliary_model_is_pinned_to_a_tenant_id(self, shim):
        env = shim.build_env()
        assert env["ANTHROPIC_BASE_URL"] == shim.base_url
        assert env["ANTHROPIC_API_KEY"] == shim.api_key
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == HAIKU
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == SONNET
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == OPUS
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == HAIKU
        assert env["CLAUDE_CODE_BG_CLASSIFIER_MODEL"] == HAIKU
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == SONNET_NAME
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CODE_GZIP_REQUEST_BODIES"] == "0"

    async def test_start_is_idempotent(self, shim):
        router, port = shim._router, shim._router.port
        await shim.start()
        assert shim._router is router
        assert shim._router.port == port
        assert shim.base_url == f"http://127.0.0.1:{port}"

    async def test_a_repeated_start_leaves_exactly_one_runner_to_stop(self, shim):
        """A second bind would leak a listener that a single stop cannot release."""
        runner = shim._router._runner
        await shim.start()
        await shim.start()
        assert shim._router._runner is runner
        port = shim._router.port

        await shim.stop()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2.0)
            assert probe.connect_ex(("127.0.0.1", port)) != 0


# --- router ----------------------------------------------------------------


class TestRouter:
    async def test_probe_requires_the_run_secret(self, shim, client):
        async with httpx.AsyncClient(base_url=shim.base_url) as anonymous:
            assert (await anonymous.request("HEAD", "/api/hello")).status_code == 401
        async with client:
            assert (await client.request("HEAD", "/api/hello")).status_code == 200

    async def test_bearer_is_accepted(self, shim, client):
        async with client:
            response = await client.request(
                "HEAD",
                "/api/hello",
                headers={"x-api-key": "", "Authorization": f"Bearer {shim.api_key}"},
            )
        assert response.status_code == 200

    async def test_models_lists_the_tenant_catalog(self, shim, client):
        async with client:
            body = (await client.get("/v1/models")).json()
        assert {m["id"] for m in body["data"]} == {m["modelName"] for m in DISCOVERY}

    async def test_count_tokens_is_honest_about_being_unsupported(self, shim, client):
        async with client:
            response = await client.post("/v1/messages/count_tokens", json={})
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found_error"

    async def test_stream_is_copied_verbatim(self, shim, client, gateway):
        async with client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "anthropic-beta": "context-management-2025-06-27",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5",
                    "stream": True,
                    "output_config": {"x": 1},
                },
            )
        assert response.status_code == 200
        assert response.content == SSE
        sent = gateway["seen"][-1]
        assert sent["body"]["model"] == SONNET_NAME
        assert sent["body"]["output_config"] == {"x": 1}
        assert sent["headers"]["anthropic-beta"] == "context-management-2025-06-27"
        assert sent["headers"]["anthropic-version"] == "2023-06-01"
        assert sent["headers"]["X-UiPath-Streaming-Enabled"] == "true"

    async def test_forwarded_body_is_verbatim_apart_from_pruned_fields(
        self, shim, client, gateway
    ):
        """Everything the CLI sends must survive byte for byte, model included."""
        payload = {
            "model": "claude-sonnet-4-5",
            "stream": True,
            "max_tokens": 8192,
            "system": [
                {
                    "type": "text",
                    "text": "be brief",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "metadata": {"user_id": "u-1"},
            "tool_choice": {"type": "auto"},
        }
        async with client:
            response = await client.post("/v1/messages", json=payload)

        assert response.status_code == 200
        expected = {k: v for k, v in payload.items() if k != "context_management"}
        assert gateway["seen"][-1]["raw"] == json.dumps(expected).encode()

    async def test_a_wrong_run_secret_is_rejected(self, shim):
        async with httpx.AsyncClient(base_url=shim.base_url) as wrong:
            response = await wrong.post(
                "/v1/messages",
                headers={"x-api-key": "x" * len(shim.api_key)},
                json={"model": "claude-sonnet-4-5", "stream": True},
            )
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    async def test_rate_limited_upstream_becomes_an_anthropic_rate_limit_error(
        self, shim, client
    ):
        async with client:
            response = await client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-5", "mode": "ratelimit"},
            )
        assert response.status_code == 429
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["message"] == "too many requests"
        record = shim.records[-1]
        assert record.status == 429
        assert record.response_headers["anthropic-ratelimit-requests-remaining"] == "0"

    async def test_gzipped_request_body_is_accepted(self, shim, client, gateway):
        payload = json.dumps({"model": "claude-sonnet-4-5", "stream": True}).encode()
        async with client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "content-encoding": "gzip",
                    "content-type": "application/json",
                },
                content=gzip.compress(payload),
            )
        assert response.status_code == 200
        assert gateway["seen"][-1]["body"]["model"] == SONNET_NAME

    async def test_record_carries_tokens_and_uipath_headers(self, shim, client):
        async with client:
            await client.post(
                "/v1/messages", json={"model": "claude-sonnet-4-5", "stream": True}
            )
        record = shim.records[-1]
        assert (record.input_tokens, record.output_tokens) == (11, 42)
        assert record.cache_read_input_tokens == 3
        assert record.stop_reason == "end_turn"
        assert record.resolved_model == SONNET
        assert record.requested_model == "claude-sonnet-4-5"
        assert record.response_headers["x-uipath-correlation-id"] == "c1"

    async def test_non_streaming_call(self, shim, client, gateway):
        async with client:
            response = await client.post(
                "/v1/messages", json={"model": "claude-sonnet-4-5", "mode": "unary"}
            )
        assert response.status_code == 200
        assert gateway["seen"][-1]["headers"]["X-UiPath-Streaming-Enabled"] == "false"
        record = shim.records[-1]
        assert (record.input_tokens, record.output_tokens) == (5, 6)

    async def test_upstream_error_is_reshaped(self, shim, client):
        async with client:
            response = await client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-5", "stream": True, "mode": "error"},
            )
        assert response.status_code == 403
        body = response.json()
        assert body["error"]["type"] == "permission_error"
        assert body["error"]["message"] == "denied"
        assert body["request_id"] == "corr-9"

    async def test_mid_stream_failure_becomes_an_sse_error_frame(self, shim, client):
        """A clean read proves the stream was terminated, not truncated.

        Whatever the upstream got out before it died is forwarded verbatim, but
        the abort resets the connection and an unread frame can be discarded
        with it, so only the trailing error frame is guaranteed.
        """
        async with client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5",
                    "stream": True,
                    "mode": "midstream",
                },
            )
        assert response.status_code == 200

        frames = [frame for frame in response.content.split(b"\n\n") if frame]
        assert set(frames[:-1]) <= {b"event: ping\ndata: {}"}
        assert frames[-1].startswith(b"event: error\ndata: ")
        payload = json.loads(frames[-1].split(b"data: ", 1)[1])
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "api_error"
        assert shim.records[-1].status == 502

    async def test_unknown_model_404s_with_the_tenant_list(self, shim, client):
        async with client:
            response = await client.post("/v1/messages", json={"model": "gpt-9"})
        assert response.status_code == 404
        assert SONNET in response.json()["error"]["message"]

    async def test_unroutable_model_is_rejected_before_forwarding(self, shim, client):
        async with client:
            response = await client.post(
                "/v1/messages", json={"model": "gpt-5.2-2025-12-11"}
            )
        assert response.status_code == 400

    async def test_malformed_body(self, shim, client):
        async with client:
            response = await client.post("/v1/messages", content=b"not json")
        assert response.status_code == 400
