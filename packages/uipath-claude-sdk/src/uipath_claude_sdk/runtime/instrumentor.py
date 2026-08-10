"""Claude SDK LLMOps span instrumentor.

Created inside the runtime's ``stream()`` from an injected span factory.
Called once per assistant message to emit per-turn spans.

Span hierarchy emitted per assistant turn::

    llmCall
    └── modelRun
    toolCall  (one per ToolUseBlock, closed when ToolResultBlock arrives)

The span factory is injected by the host (e.g. UiPath Agents' LlmOpsSpanFactory
behind an adapter) via the runtime's ``span_factory`` attribute; this package
has no dependency on any concrete factory implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from opentelemetry.trace import Span
from uipath.core.serialization import serialize_json

logger = logging.getLogger(__name__)


@runtime_checkable
class ClaudeSdkSpanFactory(Protocol):
    """Protocol for the span factory injected into the Claude SDK runtime."""

    def start_llm_call(self) -> Span:
        """Start an llmCall span."""
        ...

    def start_model_run(
        self, *, model_name: str, parent_span: Span
    ) -> tuple[Span, Any]:
        """Start a modelRun span under the llmCall span.

        Returns:
            The span and an opaque context token passed back to end_model_run().
        """
        ...

    def end_model_run(self, span: Span, token: Any) -> None:
        """End a modelRun span and release its context token."""
        ...

    def start_tool_call(self, *, tool_name: str, arguments: Any, call_id: str) -> Span:
        """Start a toolCall span."""
        ...

    def end_span_ok(self, span: Span) -> None:
        """End a span with OK status."""
        ...

    def end_span_error(self, span: Span, error: Exception) -> None:
        """End a span with error status."""
        ...


class ClaudeSdkInstrumentor:
    """Emits per-turn LLMOps spans from Claude SDK messages.

    Args:
        span_factory: Injected span factory (see ClaudeSdkSpanFactory).
        model: Model ID forwarded to modelRun span attributes.
    """

    def __init__(self, span_factory: ClaudeSdkSpanFactory, model: str) -> None:
        self._span_factory = span_factory
        self._model = model
        # tool_use_id → open toolCall span; closed on matching ToolResultBlock
        self._open_tool_spans: dict[str, Span] = {}

    def on_assistant_message(self, message: AssistantMessage) -> None:
        """Emit llmCall + modelRun spans; open one toolCall span per ToolUseBlock."""
        try:
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_use_blocks: list[ToolUseBlock] = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    thinking_parts.append(block.thinking)
                elif isinstance(block, ToolUseBlock):
                    tool_use_blocks.append(block)

            content = " ".join(text_parts) if text_parts else None
            thinking = " ".join(thinking_parts) if thinking_parts else None
            tool_calls = [
                {"id": b.id, "name": b.name, "arguments": b.input}
                for b in tool_use_blocks
            ]

            llm_span = self._span_factory.start_llm_call()
            model_span, token = self._span_factory.start_model_run(
                model_name=self._model,
                parent_span=llm_span,
            )
            if content:
                model_span.set_attribute("explanation", content)
            if thinking:
                model_span.set_attribute("thinking", thinking)
            if tool_calls:
                model_span.set_attribute("toolCalls", serialize_json(tool_calls))
            usage = getattr(message, "usage", None)
            if usage:
                model_span.set_attribute(
                    "usage",
                    json.dumps(
                        {
                            "input_tokens": usage.get("input_tokens"),
                            "output_tokens": usage.get("output_tokens"),
                        }
                    ),
                )
            # Close inner before outer — OTEL requires child spans to end first.
            self._span_factory.end_model_run(model_span, token)
            self._span_factory.end_span_ok(llm_span)

            for block in tool_use_blocks:
                tool_span = self._span_factory.start_tool_call(
                    tool_name=block.name,
                    arguments=block.input,
                    call_id=block.id,
                )
                self._open_tool_spans[block.id] = tool_span
        except Exception:
            logger.exception("Error emitting LLM spans for AssistantMessage")

    def on_user_message(self, message: UserMessage) -> None:
        """Close open toolCall spans with results from ToolResultBlocks."""
        try:
            if not isinstance(message.content, list):
                return
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                span = self._open_tool_spans.pop(block.tool_use_id, None)
                if span is None:
                    continue
                result = block.content
                if result is not None:
                    if isinstance(result, list):
                        span.set_attribute("result", serialize_json(result))
                    else:
                        span.set_attribute("result", str(result))
                if block.is_error:
                    error_msg = str(result) if result is not None else "tool error"
                    self._span_factory.end_span_error(span, Exception(error_msg))
                else:
                    self._span_factory.end_span_ok(span)
        except Exception:
            logger.exception("Error emitting tool result spans for UserMessage")

    def cleanup(self) -> None:
        """End any toolCall spans that were opened but never matched to a result."""
        for span in self._open_tool_spans.values():
            try:
                self._span_factory.end_span_error(
                    span, Exception("tool result not received")
                )
            except Exception:
                logger.debug("Failed to close orphaned tool span during cleanup")
        self._open_tool_spans.clear()
