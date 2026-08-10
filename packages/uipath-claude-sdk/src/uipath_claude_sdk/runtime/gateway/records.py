"""Per-call telemetry emitted by the gateway shim.

Token counts are read by teeing the SSE frames the shim is already streaming
to the client, so nothing is buffered, re-encoded or delayed on their account.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_SSE_DELIMITER = b"\n\n"
_DATA_PREFIX = b"data:"
_MAX_PENDING_BYTES = 1_048_576


@dataclass
class GatewayCallRecord:
    """One upstream call through the shim, ready for a tracing consumer."""

    requested_model: str
    resolved_model: str
    vendor_type: str
    api_flavor: str | None
    streaming: bool
    status: int
    duration_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    stop_reason: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    request_body: dict[str, Any] | None = None
    output_content: list[dict[str, Any]] = field(default_factory=list)
    traced_upstream: bool = False
    """The call carried trace context, so the gateway owns the model span."""


GatewayCallSink = Callable[[GatewayCallRecord], None]
TraceHeaderSource = Callable[[], dict[str, str]]


class UsageTee:
    """Accumulate Anthropic usage counters from a stream passing through.

    Feed the exact bytes written to the client. Frames that are not usage
    bearing are ignored, and a partial frame is carried over to the next feed.
    """

    def __init__(self) -> None:
        self._pending = bytearray()
        self._blocks: dict[int, dict[str, Any]] = {}
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_creation_input_tokens: int | None = None
        self.cache_read_input_tokens: int | None = None
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        """Consume a slice of the SSE stream."""
        self._pending.extend(chunk)
        while _SSE_DELIMITER in self._pending:
            frame, _, rest = bytes(self._pending).partition(_SSE_DELIMITER)
            self._pending = bytearray(rest)
            self._consume_frame(frame)
        if len(self._pending) > _MAX_PENDING_BYTES:
            self._pending.clear()

    def feed_message(self, message: Any) -> None:
        """Consume a whole non-streaming Anthropic message body."""
        if not isinstance(message, dict):
            return
        self._absorb_usage(message.get("usage"))
        stop_reason = message.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        content = message.get("content")
        if isinstance(content, list) and content:
            self._blocks = dict(enumerate(b for b in content if isinstance(b, dict)))

    @property
    def content(self) -> list[dict[str, Any]]:
        """The assistant blocks the model produced, in the order it sent them."""
        return [self._blocks[i] for i in sorted(self._blocks)]

    def _consume_frame(self, frame: bytes) -> None:
        for line in frame.split(b"\n"):
            if not line.startswith(_DATA_PREFIX):
                continue
            try:
                event = json.loads(line[len(_DATA_PREFIX) :].strip())
            except ValueError:
                continue
            if isinstance(event, dict):
                self._consume_event(event)

    def _consume_event(self, event: dict[str, Any]) -> None:
        match event.get("type"):
            case "message_start":
                self.feed_message(event.get("message"))
            case "content_block_start":
                block = event.get("content_block")
                if isinstance(block, dict):
                    self._blocks[int(event.get("index", 0))] = dict(block)
            case "content_block_delta":
                self._append_delta(event)
            case "message_delta":
                self._absorb_usage(event.get("usage"))
                delta = event.get("delta")
                if isinstance(delta, dict) and isinstance(
                    delta.get("stop_reason"), str
                ):
                    self.stop_reason = delta["stop_reason"]

    def _append_delta(self, event: dict[str, Any]) -> None:
        """Grow the block this delta belongs to, whatever kind it is."""
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return
        block = self._blocks.setdefault(int(event.get("index", 0)), {})
        for source, target in (
            ("text", "text"),
            ("thinking", "thinking"),
            ("partial_json", "partial_json"),
        ):
            piece = delta.get(source)
            if isinstance(piece, str):
                block[target] = str(block.get(target, "")) + piece

    def _absorb_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.input_tokens = _pick(usage, "input_tokens", self.input_tokens)
        self.output_tokens = _pick(usage, "output_tokens", self.output_tokens)
        self.cache_creation_input_tokens = _pick(
            usage, "cache_creation_input_tokens", self.cache_creation_input_tokens
        )
        self.cache_read_input_tokens = _pick(
            usage, "cache_read_input_tokens", self.cache_read_input_tokens
        )

    def apply(self, record: GatewayCallRecord) -> None:
        """Copy the accumulated counters onto a record."""
        record.input_tokens = self.input_tokens
        record.output_tokens = self.output_tokens
        record.cache_creation_input_tokens = self.cache_creation_input_tokens
        record.cache_read_input_tokens = self.cache_read_input_tokens
        record.stop_reason = self.stop_reason
        record.output_content = self.content


def _pick(usage: dict[str, Any], key: str, current: int | None) -> int | None:
    value = usage.get(key)
    return value if isinstance(value, int) else current


__all__ = ["GatewayCallRecord", "GatewayCallSink", "UsageTee"]
