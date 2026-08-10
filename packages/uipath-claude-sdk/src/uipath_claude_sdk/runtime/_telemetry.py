"""The telemetry this package owns on top of the Claude Agent SDK instrumentor.

``openinference-instrumentation-claude-agent-sdk`` wraps the SDK's own entry
points and emits the AGENT span per turn and the TOOL span per tool call. Three
things are left over, and they live here.

LLM spans and token counts. The instrumentor emits none, and its README's advice
to add ``openinference-instrumentation-anthropic`` does not apply: the bundled
Claude Code CLI issues every model HTTP call from its own subprocess, so
patching this process's ``anthropic`` client would instrument nothing. The
gateway shim is the only place that observes those calls, so
:class:`GatewayCallTelemetry` turns each one it served into an LLM span.

Cost. The instrumentor copies ``ResultMessage.total_cost_usd`` onto the AGENT
span as ``llm.cost.total``. The CLI computes that number client-side from
Anthropic's public catalogue, which says nothing about what a gateway-routed
model costs.

Tool identity. The instrumentor writes the ``tool_use_id`` of one invocation,
a value minted fresh by the model on every call, to ``tool.id``. UiPath reads
that attribute as the tool's stable identifier, taken from its resource
definition, and prefers it over ``tool.name`` when scoring a tool-call
evaluation, so a per-invocation value there leaves every criterion unmatchable.

The last two are attributes whose meaning does not survive the trip to UiPath,
and :class:`AttributeStripper` removes both on the way out.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from uipath.core.serialization import serialize_json
from uipath.platform.chat.llm_trace_context import build_trace_context_headers

from .gateway.records import GatewayCallRecord

logger = logging.getLogger(__name__)

TRACER_NAME = "uipath-claude-sdk"

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_SPAN_KIND = OpenInferenceSpanKindValues.LLM.value
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
PROMPT_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
COMPLETION_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
TOTAL_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_COST_TOTAL = SpanAttributes.LLM_COST_TOTAL
TOOL_ID = SpanAttributes.TOOL_ID
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE

LLM_SPAN_NAME = "AnthropicMessages"
_TRACE_SOURCE = "source=claude-sdk"
_ANTHROPIC = "anthropic"
_JSON = "application/json"
_INVOCATION_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "stream")

STRIPPED_ATTRIBUTES = frozenset({LLM_COST_TOTAL, TOOL_ID})
"""Attributes the instrumentor sets that must not reach UiPath."""

_MAX_BUFFERED_CALLS = 256
_PROMPT_TOKEN_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class _BufferedCall:
    """One upstream call and the moment the shim finished reporting it."""

    record: GatewayCallRecord
    arrived_ns: int


class GatewayCallTelemetry:
    """Emits the LLM span for every upstream call the gateway shim served.

    Wired into ``GatewayShim(on_call=...)``. Records are buffered as they
    arrive, on whichever task served the request, and turned into spans when the
    run flushes them. The buffer belongs to a single run and must not be shared
    with another run executing in the same process.

    Args:
        tracer: Tracer to build spans with. Defaults to the global provider's.
    """

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer if tracer is not None else trace.get_tracer(TRACER_NAME)
        self._gateway_calls: deque[_BufferedCall] = deque(maxlen=_MAX_BUFFERED_CALLS)
        self._parent: Context | None = None

    def adopt_parent(self, parent: Context) -> None:
        """Nest these spans under the turn they belong to.

        The shim serves the CLI's calls on its own task and the spans are built
        after the turn has finished, so neither moment has the instrumentor's
        agent span in context. The runtime captures it while iterating the
        response, where it is current, and hands it over.
        """
        self._parent = parent

    def trace_context_headers(self) -> dict[str, str]:
        """Headers that let the LLM Gateway record the model span itself.

        Every UiPath agent framework hands the gateway the caller's trace and
        span through these, and the gateway answers with a model span of its
        own, so agents built on different frameworks produce the same trace.
        Returns nothing when the platform has the feature turned off, which is
        the signal to fall back to recording the span here.
        """
        if self._parent is None:
            return {}
        token = context_api.attach(self._parent)
        try:
            return build_trace_context_headers(extra_baggage=[_TRACE_SOURCE])
        except Exception:
            logger.debug("could not build trace context headers", exc_info=True)
            return {}
        finally:
            context_api.detach(token)

    def on_gateway_call(self, record: GatewayCallRecord) -> None:
        """Buffer one upstream call."""
        self._gateway_calls.append(_BufferedCall(record, time.time_ns()))

    def flush_gateway_calls(self) -> None:
        """Emit a span per buffered call, including background and subagent traffic."""
        buffered = list(self._gateway_calls)
        self._gateway_calls.clear()
        for call in buffered:
            try:
                self._emit_gateway_span(call)
            except Exception:
                logger.exception("Error emitting a span for a gateway call")

    def _emit_gateway_span(self, call: _BufferedCall) -> None:
        record = call.record
        if record.traced_upstream:
            return
        start_ns = call.arrived_ns - int(record.duration_ms * 1_000_000)
        span = self._tracer.start_span(
            LLM_SPAN_NAME, context=self._parent, start_time=start_ns
        )
        span.set_attribute(SPAN_KIND, LLM_SPAN_KIND)
        span.set_attribute(LLM_MODEL_NAME, record.resolved_model)
        span.set_attribute(LLM_PROVIDER, record.vendor_type)
        span.set_attribute(LLM_SYSTEM, _ANTHROPIC)
        _set_token_counts(span, *_record_tokens(record))
        _set_prompt(span, record.request_body)
        _set_completion(span, record.output_content, record.stop_reason)
        if record.error is not None:
            span.set_status(Status(StatusCode.ERROR, record.error))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=call.arrived_ns)


def _set_prompt(span: Span, request: dict[str, Any] | None) -> None:
    """Record what the model was asked, as openinference input messages."""
    if not request:
        return
    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": _flatten(system)})
    for message in request.get("messages", []):
        if isinstance(message, dict):
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": _flatten(message.get("content")),
                }
            )
    for index, message in enumerate(messages):
        prefix = f"{LLM_INPUT_MESSAGES}.{index}.message"
        span.set_attribute(f"{prefix}.role", message["role"])
        span.set_attribute(f"{prefix}.content", message["content"])
    span.set_attribute(INPUT_VALUE, serialize_json(messages))
    span.set_attribute(INPUT_MIME_TYPE, _JSON)
    params = {k: request[k] for k in _INVOCATION_KEYS if k in request}
    if params:
        span.set_attribute(LLM_INVOCATION_PARAMETERS, serialize_json(params))


def _set_completion(
    span: Span, content: list[dict[str, Any]], stop_reason: str | None
) -> None:
    """Record what the model answered, tool calls included."""
    if not content:
        return
    prefix = f"{LLM_OUTPUT_MESSAGES}.0.message"
    span.set_attribute(f"{prefix}.role", "assistant")
    text = " ".join(
        str(b.get("text", "")) for b in content if b.get("type") == "text"
    ).strip()
    if text:
        span.set_attribute(f"{prefix}.content", text)
    thinking = " ".join(
        str(b.get("thinking", "")) for b in content if b.get("type") == "thinking"
    ).strip()
    if thinking:
        span.set_attribute(f"{prefix}.reasoning", thinking)
    for index, block in enumerate(b for b in content if b.get("type") == "tool_use"):
        call_prefix = f"{prefix}.tool_calls.{index}.tool_call"
        span.set_attribute(f"{call_prefix}.function.name", str(block.get("name", "")))
        span.set_attribute(
            f"{call_prefix}.function.arguments",
            str(block.get("partial_json") or serialize_json(block.get("input") or {})),
        )
    if stop_reason:
        span.set_attribute("llm.stop_reason", stop_reason)
    span.set_attribute(OUTPUT_VALUE, serialize_json(content))
    span.set_attribute(OUTPUT_MIME_TYPE, _JSON)


def _flatten(content: Any) -> str:
    """Anthropic content, which may be a string or a block list, as text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return serialize_json(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        match block.get("type"):
            case "text":
                parts.append(str(block.get("text", "")))
            case "tool_result":
                parts.append(f"[tool_result] {_flatten(block.get('content'))}")
            case "tool_use":
                parts.append(f"[tool_use {block.get('name')}] {block.get('input')}")
            case other:
                parts.append(f"[{other}]")
    return "\n".join(p for p in parts if p)


class AttributeStripper(SpanProcessor):
    """Removes every :data:`STRIPPED_ATTRIBUTES` entry before a span is exported.

    A span's attributes are sealed by the time ``on_end`` runs, so the filtered
    copy is swapped onto the ``ReadableSpan`` the processors share rather than
    written through it. Processors registered after this one, which is where the
    UiPath exporters are added, therefore see the stripped span.
    """

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        """Nothing to do: the attributes are only complete once the span ends."""

    def on_end(self, span: ReadableSpan) -> None:
        """Drop the stripped attributes, leaving the rest of the span untouched."""
        attributes = span.attributes
        if not attributes or STRIPPED_ATTRIBUTES.isdisjoint(attributes):
            return
        try:
            span._attributes = {
                key: value
                for key, value in attributes.items()
                if key not in STRIPPED_ATTRIBUTES
            }
        except Exception:
            logger.debug("Failed to strip attributes from span '%s'", span.name)

    def shutdown(self) -> None:
        """Nothing to release."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is held back, so there is nothing to flush."""
        return True


def _usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """Prompt and completion counts from an Anthropic usage mapping.

    Cache reads and cache writes are billed input, so they join the prompt
    count rather than disappearing from the total.
    """
    if not isinstance(usage, Mapping):
        return None, None
    prompt: int | None = None
    for key in _PROMPT_TOKEN_KEYS:
        value = usage.get(key)
        if isinstance(value, int):
            prompt = value if prompt is None else prompt + value
    completion = usage.get("output_tokens")
    return prompt, completion if isinstance(completion, int) else None


def _record_tokens(record: GatewayCallRecord) -> tuple[int | None, int | None]:
    """Prompt and completion counts from a gateway call record."""
    return _usage_tokens(
        {
            "input_tokens": record.input_tokens,
            "cache_read_input_tokens": record.cache_read_input_tokens,
            "cache_creation_input_tokens": record.cache_creation_input_tokens,
            "output_tokens": record.output_tokens,
        }
    )


def _set_token_counts(span: Span, prompt: int | None, completion: int | None) -> None:
    if prompt is not None:
        span.set_attribute(PROMPT_TOKENS, prompt)
    if completion is not None:
        span.set_attribute(COMPLETION_TOKENS, completion)
    if prompt is not None and completion is not None:
        span.set_attribute(TOTAL_TOKENS, prompt + completion)


__all__ = [
    "LLM_SPAN_NAME",
    "STRIPPED_ATTRIBUTES",
    "TRACER_NAME",
    "AttributeStripper",
    "GatewayCallTelemetry",
]
