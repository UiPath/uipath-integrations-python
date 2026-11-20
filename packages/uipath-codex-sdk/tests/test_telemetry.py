"""Unit tests of the span helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath_codex_sdk.runtime.telemetry import (
    ItemSpans,
    SseCompletedReader,
    complete_span,
    complete_tool_call,
    has_output_schema,
    item_tool,
    item_tool_result,
    item_tool_text,
    latest_input_text,
    response_summary,
    restored_span,
    span_attributes,
    span_record,
    start_tool_call,
)

from ._typing import attrs, end_ns, parent_span_id, start_ns


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(
        f"event: {e['type']}\r\ndata: {json.dumps(e)}\r\n\r\n".encode() for e in events
    )


def test_sse_reader_takes_the_completed_response_across_chunk_boundaries():
    final = {"id": "r1", "output": [{"type": "message", "content": []}]}
    data = _sse(
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_text.delta", "delta": "hel"},
        {"type": "response.completed", "response": final},
    )
    reader = SseCompletedReader()
    for i in range(0, len(data), 7):
        reader.feed(data[i : i + 7])
    assert reader.response == final


def test_sse_reader_fills_output_from_item_events():
    call = {"type": "function_call", "name": "add", "arguments": "{}", "call_id": "c1"}
    reader = SseCompletedReader()
    reader.feed(
        _sse(
            {"type": "response.output_item.done", "item": call},
            {"type": "response.completed", "response": {"id": "r1", "usage": {}}},
        )
    )
    assert reader.response == {"id": "r1", "usage": {}, "output": [call]}
    assert SseCompletedReader().response is None


def test_response_summary():
    summary = response_summary(
        {
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "a_plus_b",
                    "arguments": '{"a": 1}',
                },
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }
    )
    assert summary == {
        "text": "hi",
        "toolCalls": [{"id": "c1", "name": "a_plus_b", "arguments": '{"a": 1}'}],
        "usage": {"promptTokens": 5, "completionTokens": 2, "totalTokens": 7},
    }


def test_response_summary_drops_schema_placeholders_next_to_tool_calls():
    placeholder = {
        "type": "message",
        "content": [{"type": "output_text", "text": '{"issue_key": "pending"}'}],
    }
    call = {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}
    payload = {"text": {"format": {"type": "json_schema", "schema": {}}}}
    assert has_output_schema(payload)
    assert not has_output_schema({"text": {"verbosity": "low"}})
    assert not has_output_schema({})

    with_calls = {"output": [placeholder, call]}
    assert response_summary(with_calls, schema_output=True)["text"] == ""
    assert response_summary(with_calls)["text"] == '{"issue_key": "pending"}'
    final = {"output": [placeholder]}
    assert response_summary(final, schema_output=True)["text"] == (
        '{"issue_key": "pending"}'
    )


def test_latest_input_text_prefers_the_newest_item_with_text():
    payload = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "q"}],
            },
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]
    }
    assert latest_input_text(payload) == "42"
    assert latest_input_text({"input": "plain"}) == "plain"
    assert latest_input_text({}) == ""


def test_item_tool_names_codex_items():
    mcp = {
        "type": "mcpToolCall",
        "server": "orders",
        "tool": "get",
        "arguments": {"id": 1},
    }
    assert item_tool(mcp) == ("get", "Mcp", {"id": 1})
    assert item_tool({"type": "commandExecution", "command": "ls"}) == (
        "shell",
        "Internal",
        {"command": "ls"},
    )
    assert item_tool({"type": "agentMessage"}) is None
    failed = {**mcp, "status": "failed", "error": {"message": "boom"}}
    assert item_tool_result(failed) == (None, "boom")


def test_span_attributes_json_encodes_structures_and_drops_none(monkeypatch):
    monkeypatch.setenv("UIPATH_PROCESS_VERSION", "1.2.3")
    attributes = span_attributes("toolCall", input={"a": 1}, missing=None, flag=True)
    assert attributes["span_type"] == attributes["type"] == "toolCall"
    assert attributes["uipath.custom_instrumentation"] is True
    assert attributes["agentVersion"] == "1.2.3"
    assert attributes["input"] == '{"a": 1}' and attributes["flag"] is True
    assert "missing" not in attributes


def test_saved_span_is_completed_with_its_identity():
    """``span_record`` → JSON → ``complete_span``: same ids, start and parent,
    through the provider's processors like an ended span."""
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]
    try:
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("parent") as parent:
            open_span = tracer.start_span(
                "Tool call - t", attributes=span_attributes("toolCall", tags=["x"])
            )
        record = json.loads(json.dumps(span_record(open_span)))
        assert record["parent_span_id"] == format(
            parent.get_span_context().span_id, "016x"
        )
        assert record["attributes"]["tags"] == '["x"]'

        # A child of the restored span joins the saved trace (sampled).
        with tracer.start_as_current_span(
            "child", context=trace.set_span_in_context(restored_span(record))
        ) as child:
            assert child.is_recording()

        complete_span(record, attributes={"output": "42"}, error="job faulted")
        finished = {s.name: s for s in exporter.get_finished_spans()}
        done = finished["Tool call - t"]
        assert done.context.span_id == open_span.get_span_context().span_id
        assert done.context.trace_id == open_span.get_span_context().trace_id
        assert parent_span_id(done) == parent.get_span_context().span_id
        assert isinstance(open_span, ReadableSpan)
        assert start_ns(done) == open_span.start_time
        assert end_ns(done) >= start_ns(done)
        assert attrs(done)["output"] == "42"
        assert attrs(done)["span_type"] == "toolCall"
        assert attrs(done)["error"] == "job faulted"
        assert done.status.status_code.name == "ERROR"
        assert parent_span_id(finished["child"]) == done.context.span_id
        assert done.to_json()  # what the --trace-file exporter writes
    finally:
        exporter.shutdown()


@pytest.mark.parametrize("error", [RuntimeError(), asyncio.CancelledError()])
async def test_a_messageless_error_still_faults_the_agent_run(error):
    """``CancelledError()`` has no text: the run still completes with ERROR,
    and without the output saved at suspension."""
    from uipath_codex_sdk.runtime.factory import _AgentRun

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]
    try:
        tracer = trace.get_tracer("test")
        live = tracer.start_span(
            "Agent run - a",
            attributes=span_attributes("agentRun", output={"x": 1}, status="suspended"),
        )
        record = json.loads(json.dumps(span_record(live)))
        run = _AgentRun(restored_span(record), store=None, delegate=None, saved=record)
        await run.fail(error)
        run.detach()
        [done] = exporter.get_finished_spans()
        assert done.context.span_id == live.get_span_context().span_id
        assert done.status.status_code.name == "ERROR"
        assert attrs(done)["status"] == "faulted"
        assert attrs(done)["error"] == type(error).__name__
        assert "output" not in attrs(done)
    finally:
        exporter.shutdown()


def _exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]
    return exporter


def _mcp_item(**extra: Any) -> dict[str, Any]:
    return {
        "id": "item-1",
        "type": "mcpToolCall",
        "server": "orders",
        "tool": "get_order_status",
        "arguments": {"order_id": "A-1"},
        "status": "completed",
        **extra,
    }


def test_item_tool_text_of_codex_items():
    result = {
        "content": [
            {"type": "text", "text": "shipped"},
            {"type": "image", "data": "..."},
            {"type": "text", "text": "eta 2d"},
        ],
        "structured_content": None,
    }
    assert item_tool_text(_mcp_item(result=result)) == "shipped\neta 2d"
    assert item_tool_text(_mcp_item(result={"content": []})) is None
    shell = {"type": "commandExecution", "aggregated_output": "a.txt\n"}
    assert item_tool_text(shell) == "a.txt\n"
    changes = [
        {"path": "a.py", "kind": {"type": "add"}, "diff": "+x"},
        {"path": "b.py", "kind": {"type": "update", "move_path": "c.py"}},
    ]
    assert (
        item_tool_text({"type": "fileChange", "changes": changes})
        == "add a.py\nupdate b.py -> c.py"
    )
    assert item_tool_text({"type": "webSearch", "query": "uipath"}) == "uipath"
    dynamic = {"type": "dynamicToolCall", "content_items": [{"text": "42"}]}
    assert item_tool_text(dynamic) == "42"


def test_tool_spans_feed_the_eval_helpers_and_the_llmops_mapping():
    """Spans built as the runtime builds them read back through ``uipath.eval``
    (names, args, ``output.value``'s content) and the LLMOps exporter."""
    from uipath.eval._helpers.evaluators_helpers import (
        extract_tool_calls,
        extract_tool_calls_names,
        extract_tool_calls_outputs,
    )
    from uipath.tracing._otel_exporters import LlmOpsHttpExporter

    exporter = _exporter()
    try:
        items = ItemSpans()
        item = _mcp_item()
        items.start(item)
        result = {"content": [{"type": "text", "text": "shipped"}]}
        items.finish({**item, "result": result})
        shell = {"id": "item-2", "type": "commandExecution", "command": "ls"}
        items.start(shell)
        items.finish({**shell, "aggregated_output": "a.txt", "exit_code": 0})
        durable = start_tool_call(
            "a_plus_b", tool_type="Process", call_id="c1", arguments={"a": 1}
        )
        record = span_record(durable)
        assert record is not None
        complete_tool_call(record, output={"sum": 42})
        spans = list(exporter.get_finished_spans())
    finally:
        exporter.shutdown()

    names = ["get_order_status", "shell", "a_plus_b"]
    assert [s.name for s in spans] == [f"Tool call - {n}" for n in names]
    assert extract_tool_calls_names(spans) == names
    calls = extract_tool_calls(spans)
    assert [c.args for c in calls] == [{"order_id": "A-1"}, {"command": "ls"}, {"a": 1}]
    outputs = extract_tool_calls_outputs(spans)
    assert [o.output for o in outputs] == ["shipped", "a.txt", '{"sum": 42}']
    mcp = attrs(spans[0])
    assert mcp["tool.server"] == mcp["mcp.server"] == "orders"
    assert json.loads(mcp["output"]) == result

    mapped = LlmOpsHttpExporter._map_tool_call_attributes(
        object.__new__(LlmOpsHttpExporter), dict(mcp)
    )
    assert mapped["toolName"] == "get_order_status"
    assert mapped["arguments"] == {"order_id": "A-1"}
    assert mapped["result"] == result
    durable_attrs = dict(attrs(spans[2]))
    assert LlmOpsHttpExporter._map_tool_call_attributes(
        object.__new__(LlmOpsHttpExporter), durable_attrs
    )["result"] == {"sum": 42}
