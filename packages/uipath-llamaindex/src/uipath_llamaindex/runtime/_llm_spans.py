"""Leave the LLM span to the LLM Gateway.

UiPath models send the trace-context headers, and the LLM Gateway records the
model call ("Model run") under the span they name. OpenInference's client-side
LLM span would duplicate it, so it isn't created, unless
``UIPATH_USE_CLIENT_LLM_SPANS`` is true (checked on every call).

LlamaIndex nests spans by its own span ids, not by OTel context: each call gets
an id, and OpenInference looks its parent up by id. So an LLM call without a
span still needs three things, which the handlers below provide:

* whatever it calls nests under its nearest ancestor that has a span;
* its exit and events are ignored, rather than logged as a missing span;
* the traceparent names that ancestor (``current_span``), so the Gateway's
  span lands under the workflow step or agent that made the call.
"""

from typing import Any, Optional

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.instrumentation.events.exception import ExceptionEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.core.instrumentation.span import active_span_id
from openinference.instrumentation.llama_index import (
    LlamaIndexInstrumentor,
    get_current_span,
)
from openinference.instrumentation.llama_index._handler import (
    EventHandler,
    _SpanHandler,
)
from opentelemetry.trace import Span

from uipath_llamaindex._trace_context import use_client_llm_spans

# llama-index's chat / completion wrapper emits a start event when the call
# begins and exactly one of these when it's done: at once for a plain call,
# and for a stream when it's exhausted, fails or is closed (GeneratorExit).
_STARTS = (LLMChatStartEvent, LLMCompletionStartEvent)
_ENDS = (LLMChatEndEvent, LLMCompletionEndEvent, ExceptionEvent)


class _SkippedLlmCalls:
    """LLM calls that have no span, and where they nest instead.

    A call is forgotten once nothing can refer to it any more: when it returns
    or fails, or, if it returned a stream that still emits events with its span
    id, when that stream's last event arrives.
    """

    def __init__(self) -> None:
        self._nearest_created: dict[str, Optional[str]] = {}
        self._running: set[str] = set()
        self._awaiting_end: set[str] = set()

    def __contains__(self, span_id: object) -> bool:
        return span_id in self._nearest_created

    def __len__(self) -> int:
        return len(self._nearest_created)

    def skip(self, span_id: str, parent_id: Optional[str]) -> None:
        self._nearest_created[span_id] = parent_id
        self._running.add(span_id)

    def nearest_created(self, span_id: Optional[str]) -> Optional[str]:
        """The span itself if it was created, otherwise its nearest ancestor."""
        if span_id in self._nearest_created:
            return self._nearest_created[span_id]
        return span_id

    def returned(self, span_id: str) -> None:
        self._running.discard(span_id)
        if span_id not in self._awaiting_end:
            self.forget(span_id)

    def saw_event(self, span_id: str, event: Any) -> None:
        if isinstance(event, _STARTS):
            self._awaiting_end.add(span_id)
        elif isinstance(event, _ENDS):
            self._awaiting_end.discard(span_id)
            if span_id not in self._running:
                self.forget(span_id)

    def forget(self, span_id: str) -> None:
        self._nearest_created.pop(span_id, None)
        self._running.discard(span_id)
        self._awaiting_end.discard(span_id)

    def clear(self) -> None:
        self._nearest_created.clear()
        self._running.clear()
        self._awaiting_end.clear()


_skipped = _SkippedLlmCalls()


class _SkipLlmSpanHandler(_SpanHandler):
    """OpenInference's span handler, without the spans of LLM calls."""

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        parent_span_id = _skipped.nearest_created(parent_span_id)
        if isinstance(instance, BaseLLM) and not use_client_llm_spans():
            _skipped.skip(id_, parent_span_id)
            return None
        return super().new_span(
            id_, bound_args, instance, parent_span_id, tags, **kwargs
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if id_ in _skipped:
            _skipped.returned(id_)
            return None
        return super().prepare_to_exit_span(id_, bound_args, instance, result, **kwargs)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        if id_ in _skipped:
            _skipped.forget(id_)
            return None
        return super().prepare_to_drop_span(id_, bound_args, instance, err, **kwargs)


class _SkipLlmEventHandler(EventHandler):
    """OpenInference's event handler, ignoring events of LLM calls with no span."""

    def handle(self, event: Any, **kwargs: Any) -> Any:
        if event.span_id in _skipped:
            _skipped.saw_event(event.span_id, event)
            return event
        return super().handle(event, **kwargs)


def skip_client_llm_spans(instrumentor: LlamaIndexInstrumentor) -> None:
    """Make an instrumented ``LlamaIndexInstrumentor`` skip LLM spans. Idempotent.

    OpenInference builds its handlers inside ``instrument()``, so we can't hand
    it ours; instead the instances it built, already registered with
    llama-index's dispatcher, become the subclasses above.
    """
    span_handler = getattr(instrumentor, "_span_handler", None)
    event_handler = getattr(instrumentor, "_event_handler", None)
    if isinstance(span_handler, _SpanHandler):
        object.__setattr__(span_handler, "__class__", _SkipLlmSpanHandler)
    if isinstance(event_handler, EventHandler):
        object.__setattr__(event_handler, "__class__", _SkipLlmEventHandler)


def current_span() -> Optional[Span]:
    """The innermost span that was created: what the traceparent names.

    Inside an LLM call with no span, that's the workflow step or agent span
    that made the call.
    """
    span_id = active_span_id.get()
    if span_id not in _skipped:
        return get_current_span()
    span_handler = getattr(LlamaIndexInstrumentor(), "_span_handler", None)
    ancestor = _skipped.nearest_created(span_id)
    if span_handler is None or ancestor is None:
        return None
    open_span = span_handler.open_spans.get(ancestor)
    return open_span._otel_span if open_span is not None else None
