"""Wire-format strategies for forwarding a Claude Code request upstream.

The shim is a router, not a protocol translator. Which strategy applies is
decided by the api flavor discovery reported for the resolved model, never by
a hardcoded assumption about how a tenant hosts Claude.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from uipath.llm_client.settings.constants import ApiFlavor, VendorType

from .catalog import ResolvedModel
from .errors import UnsupportedApiFlavorError

logger = logging.getLogger(__name__)

BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"
BEDROCK_EVENTSTREAM_CONTENT_TYPE = "application/vnd.amazon.eventstream"

FORWARDED_REQUEST_HEADERS: tuple[str, ...] = (
    "anthropic-version",
    "anthropic-beta",
    "content-type",
)

_BEDROCK_TRANSPORT_FIELDS: frozenset[str] = frozenset({"model", "stream"})

UNSUPPORTED_UPSTREAM_FIELDS: tuple[str, ...] = ("context_management",)
"""Request fields the CLI sends that the gateway rejects outright.

The gateway validates the body strictly, so an unknown field fails the whole
call with ``400 Extra inputs are not permitted`` rather than being ignored.
These are dropped and the drop is logged once per run, because silently
reshaping a request is how the proxy this shim replaced lost prompt caching
without anyone noticing.

Keep this list as short as the evidence justifies. A field belongs here only
after a gateway has actually rejected it.
"""


_pruned_once: set[str] = set()


def prune_unsupported(body: dict[str, Any]) -> dict[str, Any]:
    """Drop request fields the gateway cannot accept, warning the first time.

    A strategy is built per request, so the warning is tracked per process
    rather than per instance.
    """
    dropped = [name for name in UNSUPPORTED_UPSTREAM_FIELDS if name in body]
    if not dropped:
        return body
    for name in dropped:
        del body[name]
    fresh = [name for name in dropped if name not in _pruned_once]
    if fresh:
        _pruned_once.update(fresh)
        logger.warning(
            "Dropped %s from the request because the UiPath LLM Gateway rejects "
            "unknown fields. The Claude Code CLI sends this and the feature it "
            "drives is unavailable through the gateway.",
            ", ".join(sorted(fresh)),
        )
    return body


@dataclass(frozen=True)
class UpstreamRequest:
    """Everything the router needs to issue the upstream call."""

    body: bytes
    headers: dict[str, str]
    stream: bool


class GatewayStrategy(Protocol):
    """How one wire format is forwarded to and streamed back from the gateway."""

    name: str

    def build_request(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str],
        resolved: ResolvedModel,
    ) -> UpstreamRequest:
        """Turn the CLI's request into the upstream request."""
        ...

    def stream_response(self, upstream: httpx.Response) -> AsyncIterator[bytes]:
        """Yield Anthropic SSE bytes for the client."""
        ...


class AnthropicMessagesStrategy:
    """Near pass-through for gateways that speak the Anthropic Messages format.

    The body is forwarded apart from the model id and the handful of fields in
    ``UNSUPPORTED_UPSTREAM_FIELDS``, and the response stream is copied byte for
    byte, so beta features, prompt caching and any field the CLI adds in a
    future release keep working without a change here.
    """

    name = "AnthropicMessages"

    def build_request(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str],
        resolved: ResolvedModel,
    ) -> UpstreamRequest:
        body = prune_unsupported(dict(payload))
        body["model"] = resolved.wire_name
        return UpstreamRequest(
            body=json.dumps(body).encode(),
            headers=forwarded_headers(headers),
            stream=bool(payload.get("stream", False)),
        )

    async def stream_response(self, upstream: httpx.Response) -> AsyncIterator[bytes]:
        async for chunk in upstream.aiter_bytes():
            yield chunk


class BedrockInvokeStrategy:
    """Translation for tenants whose discovery still reports the Bedrock invoke flavor.

    Bedrock's InvokeModel carries the model in the URL and the api version in
    the body, and answers a streaming call with binary Event Stream frames that
    have to be re-emitted as SSE.
    """

    name = "invoke"

    def build_request(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str],
        resolved: ResolvedModel,
    ) -> UpstreamRequest:
        body = prune_unsupported(
            {k: v for k, v in payload.items() if k not in _BEDROCK_TRANSPORT_FIELDS}
        )
        body["anthropic_version"] = BEDROCK_ANTHROPIC_VERSION
        return UpstreamRequest(
            body=json.dumps(body).encode(),
            headers=forwarded_headers(headers),
            stream=bool(payload.get("stream", False)),
        )

    async def stream_response(self, upstream: httpx.Response) -> AsyncIterator[bytes]:
        content_type = upstream.headers.get("content-type", "")
        if "amazon.eventstream" not in content_type:
            async for chunk in upstream.aiter_bytes():
                yield chunk
            return

        from botocore.eventstream import EventStreamBuffer

        buffer = EventStreamBuffer()
        async for chunk in upstream.aiter_bytes():
            buffer.add_data(chunk)
            for message in buffer:
                frame = _decode_eventstream_payload(message.payload)
                if frame is not None:
                    yield frame


def forwarded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Pick the inbound headers that belong on the upstream request."""
    forwarded = {
        name: headers[name] for name in FORWARDED_REQUEST_HEADERS if name in headers
    }
    forwarded.setdefault("content-type", "application/json")
    return forwarded


def select_strategy(resolved: ResolvedModel) -> GatewayStrategy:
    """Pick the strategy for a resolved model from its discovered api flavor."""
    if resolved.api_flavor == ApiFlavor.ANTHROPIC_MESSAGES:
        return AnthropicMessagesStrategy()
    if resolved.api_flavor == ApiFlavor.INVOKE:
        return BedrockInvokeStrategy()
    if resolved.api_flavor is None and resolved.vendor_type in (
        VendorType.ANTHROPIC,
        VendorType.AZURE,
    ):
        return AnthropicMessagesStrategy()
    raise UnsupportedApiFlavorError(
        resolved.model_id, resolved.vendor_type, resolved.api_flavor
    )


def _decode_eventstream_payload(payload: bytes) -> bytes | None:
    try:
        envelope = json.loads(payload)
        inner = base64.b64decode(envelope["bytes"])
        event = json.loads(inner)
    except (KeyError, TypeError, ValueError):
        return None
    event_type = event.get("type", "message") if isinstance(event, dict) else "message"
    return b"event: " + str(event_type).encode() + b"\ndata: " + inner + b"\n\n"


__all__ = [
    "BEDROCK_ANTHROPIC_VERSION",
    "FORWARDED_REQUEST_HEADERS",
    "UNSUPPORTED_UPSTREAM_FIELDS",
    "AnthropicMessagesStrategy",
    "BedrockInvokeStrategy",
    "GatewayStrategy",
    "UpstreamRequest",
    "forwarded_headers",
    "prune_unsupported",
    "select_strategy",
]
