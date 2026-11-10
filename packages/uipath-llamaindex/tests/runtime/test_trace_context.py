"""LLM Gateway trace context and client-side LLM spans.

* every UiPath model sends the trace-context headers on each Gateway request;
* OpenInference no longer opens the client-side LLM span the Gateway's
  "Model run" replaces, and the traceparent names the span that made the call;
* with ``UIPATH_USE_CLIENT_LLM_SPANS`` true, it's the other way around: the
  client-side LLM span is kept and the traceparent isn't sent (the baggage is).
"""

import asyncio
import gc
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from llama_index.core.llms import CompletionResponse, MockLLM
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from uipath.core.tracing import UiPathSpanUtils
from uipath.runtime import UiPathRuntimeContext
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from uipath_llamaindex._trace_context import (
    TRACEPARENT_HEADER,
    USE_CLIENT_LLM_SPANS_ENV,
    trace_context_headers,
)
from uipath_llamaindex.llms._openai import _UiPathSyncURLRewriteTransport
from uipath_llamaindex.llms.bedrock import AwsBedrockCompletionsPassthroughClient
from uipath_llamaindex.llms.vertex import _rewrite_request_for_gateway
from uipath_llamaindex.runtime import _llm_spans
from uipath_llamaindex.runtime._llm_spans import (
    _SkipLlmEventHandler,
    _SkipLlmSpanHandler,
    current_span,
    skip_client_llm_spans,
)
from uipath_llamaindex.runtime.factory import UiPathLlamaIndexRuntimeFactory

_TRACE_ID = "aabbccddeeff00112233445566778899"
_HEADERS = {
    TRACEPARENT_HEADER: f"00-{_TRACE_ID}-1122334455667788-01",
    "x-uipath-tracebaggage": "source=llamaindex",
}

# What the model saw while generating: the headers a request would carry.
_seen: list[dict[str, str]] = []


class _RecordingLLM(MockLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        _seen.append(trace_context_headers())
        return CompletionResponse(text="ok")


class _FailingLLM(MockLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise RuntimeError("gateway said no")


class _AskWorkflow(Workflow):
    def __init__(self, llm: MockLLM, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    @step
    async def ask(self, ctx: Context, ev: StartEvent) -> StopEvent:
        reply = await self._llm.acomplete("hi")
        return StopEvent(result=reply.text)


class _Recorder(SpanProcessor):
    """Captures spans at start (what live tracking sees) and at end."""

    def __init__(self) -> None:
        self.started: list[Span] = []
        self.ended: list[ReadableSpan] = []

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        self.started.append(span)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span)

    def kinds(self) -> list[Any]:
        return [
            (s.attributes or {}).get("openinference.span.kind") for s in self.started
        ]


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(USE_CLIENT_LLM_SPANS_ENV, raising=False)
    monkeypatch.setenv("UIPATH_TRACE_ID", _TRACE_ID)
    _seen.clear()
    _llm_spans._skipped.clear()
    yield
    _llm_spans._skipped.clear()


@pytest.fixture
def recorder() -> Iterator[_Recorder]:
    """Instrument against a local provider, as the factory does."""
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.uninstrument()
    provider = TracerProvider()
    rec = _Recorder()
    provider.add_span_processor(rec)
    instrumentor.instrument(tracer_provider=provider)
    skip_client_llm_spans(instrumentor)
    UiPathSpanUtils.register_current_span_provider(current_span)
    yield rec
    instrumentor.uninstrument()


async def _run(llm: MockLLM) -> Any:
    return await _AskWorkflow(llm=llm, timeout=10).run()


class TestSkipClientLlmSpans:
    async def test_never_starts_client_llm_span(self, recorder: _Recorder) -> None:
        assert await _run(_RecordingLLM(max_tokens=None)) == "ok"

        # Not just dropped at export: never started, so live tracking can't
        # leave it behind as "running".
        assert "LLM" not in recorder.kinds()
        assert "_AskWorkflow.ask" in [s.name for s in recorder.started]

    async def test_traceparent_names_the_step_that_made_the_call(
        self, recorder: _Recorder
    ) -> None:
        await _run(_RecordingLLM(max_tokens=None))

        step_span = next(s for s in recorder.ended if s.name == "_AskWorkflow.ask")
        span_id = format(step_span.context.span_id, "016x")
        assert _seen[0][TRACEPARENT_HEADER] == f"00-{_TRACE_ID}-{span_id}-01"
        assert "source=llamaindex" in _seen[0]["x-uipath-tracebaggage"]

    async def test_skipped_spans_log_no_missing_span_warnings(
        self, recorder: _Recorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            await _run(_RecordingLLM(max_tokens=None))

        assert not [r for r in caplog.records if "Open span is missing" in r.message]

    async def test_model_error_still_propagates(self, recorder: _Recorder) -> None:
        llm = _FailingLLM(max_tokens=None)
        with pytest.raises(Exception, match="gateway said no"):
            await _run(llm)

        step_span = next(s for s in recorder.ended if s.name == "_AskWorkflow.ask")
        assert not step_span.status.is_ok
        assert "LLM" not in recorder.kinds()

    async def test_skipped_spans_are_forgotten_when_the_call_ends(
        self, recorder: _Recorder
    ) -> None:
        await _run(_RecordingLLM(max_tokens=None))
        assert not _llm_spans._skipped

        llm = _FailingLLM(max_tokens=None)
        with pytest.raises(Exception, match="gateway said no"):
            await _run(llm)
        assert not _llm_spans._skipped

    async def test_streamed_call_is_forgotten_once_the_stream_ends(
        self, recorder: _Recorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            stream = await MockLLM(max_tokens=None).astream_complete("a b c")
            # The call has returned, but iterating still emits its events.
            assert _llm_spans._skipped
            chunks = [chunk.delta async for chunk in stream]

        assert "".join(c or "" for c in chunks) == "a b c"
        assert "LLM" not in recorder.kinds()
        assert not [r for r in caplog.records if "Open span is missing" in r.message]
        assert not _llm_spans._skipped

    async def test_abandoned_async_stream_closes_without_warnings(
        self, recorder: _Recorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            stream = await MockLLM(max_tokens=None).astream_complete("a b c")
            await anext(stream)
            # Dropped half-way: the event loop closes it later, after it's
            # collected, and closing emits an ExceptionEvent for its span id.
            del stream
            gc.collect()
            for _ in range(5):
                await asyncio.sleep(0)

        assert not [r for r in caplog.records if "Open span is missing" in r.message]
        assert not _llm_spans._skipped

    def test_abandoned_sync_stream_closes_without_warnings(
        self, recorder: _Recorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            stream = MockLLM(max_tokens=None).stream_complete("a b c")
            next(stream)
            del stream
            gc.collect()

        assert not [r for r in caplog.records if "Open span is missing" in r.message]
        assert not _llm_spans._skipped

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
    async def test_client_llm_spans_env_keeps_span_and_drops_traceparent(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(USE_CLIENT_LLM_SPANS_ENV, value)
        await _run(_RecordingLLM(max_tokens=None))

        assert "LLM" in recorder.kinds()
        assert TRACEPARENT_HEADER not in _seen[0]
        assert "source=llamaindex" in _seen[0]["x-uipath-tracebaggage"]

    @pytest.mark.parametrize("value", ["false", "0", ""])
    async def test_client_llm_spans_env_not_true_skips_span(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(USE_CLIENT_LLM_SPANS_ENV, value)
        await _run(_RecordingLLM(max_tokens=None))

        assert "LLM" not in recorder.kinds()
        assert TRACEPARENT_HEADER in _seen[0]

    async def test_skipping_twice_is_harmless(self, recorder: _Recorder) -> None:
        skip_client_llm_spans(LlamaIndexInstrumentor())
        await _run(_RecordingLLM(max_tokens=None))

        assert type(LlamaIndexInstrumentor()._span_handler) is _SkipLlmSpanHandler
        assert "LLM" not in recorder.kinds()

    def test_factory_installs_the_skip(self, tmp_path: Any) -> None:
        # Guards the private OpenInference handlers we subclass: if an upgrade
        # renames them, this fails instead of client LLM spans silently coming back.
        instrumentor = LlamaIndexInstrumentor()
        instrumentor.uninstrument()
        try:
            UiPathLlamaIndexRuntimeFactory(
                UiPathRuntimeContext(
                    runtime_dir=str(tmp_path), state_file=str(tmp_path / "state.db")
                )
            )
            assert isinstance(instrumentor._span_handler, _SkipLlmSpanHandler)
            assert isinstance(instrumentor._event_handler, _SkipLlmEventHandler)
        finally:
            instrumentor.uninstrument()


class TestModelsSendTraceHeaders:
    def test_openai_transport(self) -> None:
        request = httpx.Request(
            "POST", "https://gw/openai/deployments/gpt/chat/completions"
        )
        with (
            patch(
                "uipath_llamaindex._trace_context.build_trace_context_headers",
                return_value=dict(_HEADERS),
            ),
            patch.object(
                httpx.HTTPTransport, "handle_request", return_value=httpx.Response(200)
            ) as sent,
        ):
            _UiPathSyncURLRewriteTransport().handle_request(request)

        headers = sent.call_args.args[0].headers
        assert headers[TRACEPARENT_HEADER] == _HEADERS[TRACEPARENT_HEADER]
        assert headers["x-uipath-tracebaggage"] == "source=llamaindex"

    def test_bedrock_request_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com")
        request = SimpleNamespace(url="https://aws/model/m/converse", headers={})
        with patch(
            "uipath_llamaindex._trace_context.build_trace_context_headers",
            return_value=dict(_HEADERS),
        ):
            AwsBedrockCompletionsPassthroughClient(
                "m", "tok", "converse"
            )._modify_request(request)

        assert _HEADERS.items() <= request.headers.items()

    def test_vertex_request_rewrite(self) -> None:
        request = httpx.Request(
            "POST", "https://vertex/models/gemini:generateContent", content=b"{}"
        )
        with patch(
            "uipath_llamaindex._trace_context.build_trace_context_headers",
            return_value=dict(_HEADERS),
        ):
            rewritten = _rewrite_request_for_gateway(request, "https://gw/vertex")

        assert rewritten.headers[TRACEPARENT_HEADER] == _HEADERS[TRACEPARENT_HEADER]

    def test_header_errors_never_break_the_request(self) -> None:
        with patch(
            "uipath_llamaindex._trace_context.build_trace_context_headers",
            side_effect=RuntimeError("boom"),
        ):
            assert trace_context_headers() == {}
