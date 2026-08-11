"""Tests for the telemetry this package owns around the Claude SDK instrumentor.

The agent and tool spans come from
``openinference-instrumentation-claude-agent-sdk``. What is tested here is
everything that meets it: the interrupt hook it merges alongside, the LLM spans
the gateway shim is the only source of, and the attributes it writes whose
meaning does not survive the trip to UiPath.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLIConnectionError,
    HookContext,
    HookJSONOutput,
    HookMatcher,
    PreToolUseHookInput,
    tool,
)
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from uipath.runtime import UiPathRuntimeContext

from tests.conftest import make_session_paths
from uipath_claude_sdk import (
    ClaudeAgent,
    UiPathModel,
    interrupt,
    uipath_tool_server,
)
from uipath_claude_sdk.runtime._telemetry import (
    LLM_SPAN_NAME,
    AttributeStripper,
    GatewayCallTelemetry,
)
from uipath_claude_sdk.runtime.conversational_runtime import (
    UiPathClaudeSDKConversationalRuntime,
)
from uipath_claude_sdk.runtime.factory import UiPathClaudeSDKRuntimeFactory
from uipath_claude_sdk.runtime.gateway.records import GatewayCallRecord
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime

SERVING_MODEL = "claude-haiku-4-5"
"""Deliberately not the agent's configured model: a gateway call is attributed
to the model that actually served it."""

SERVER_KEY = "tickets"
TOOL_NAME = "ask"
INTERRUPT_TOOL = f"mcp__{SERVER_KEY}__{TOOL_NAME}"
TOOL_USE_ID = "toolu_1"
QUESTION = "Should I ship it?"

LLM_COST_TOTAL = SpanAttributes.LLM_COST_TOTAL
SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
TOOL_ID = SpanAttributes.TOOL_ID
TOOL_NAME_ATTR = SpanAttributes.TOOL_NAME


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


@pytest.fixture
def telemetry(provider: TracerProvider) -> GatewayCallTelemetry:
    return GatewayCallTelemetry(provider.get_tracer("test"))


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [span.name for span in exporter.get_finished_spans()]


def make_gateway_record(
    *,
    resolved_model: str = SERVING_MODEL,
    input_tokens: int | None = 11,
    output_tokens: int | None = 7,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    error: str | None = None,
    traced_upstream: bool = False,
) -> GatewayCallRecord:
    return GatewayCallRecord(
        requested_model="claude-sonnet-4-5",
        resolved_model=resolved_model,
        vendor_type="anthropic",
        api_flavor="messages",
        streaming=True,
        status=200 if error is None else 502,
        duration_ms=12.5,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        error=error,
        traced_upstream=traced_upstream,
    )


def _make_runtime(agent, tmp_path, session_store, telemetry=None):
    runtime = UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path, "rt-1"),
        runtime_id="rt-1",
    )
    runtime.telemetry = telemetry
    return runtime


# --- The interrupt hook and the instrumentor's hooks -------------------------


@tool(TOOL_NAME, "Ask a human and wait for the answer.", {"question": str})
async def ask(args: dict[str, Any]) -> dict[str, Any]:
    answer = await interrupt(args["question"])
    return {"content": [{"type": "text", "text": str(answer)}]}


@pytest.fixture
def hitl_agent() -> ClaudeAgent:
    return ClaudeAgent(
        options=ClaudeAgentOptions(
            mcp_servers={SERVER_KEY: uipath_tool_server(SERVER_KEY, tools=[ask])}
        )
    )


async def merged_hooks(
    options: ClaudeAgentOptions,
) -> dict[str, list[HookMatcher]]:
    """The hooks the CLI is really handed, once the instrumentor has merged its own.

    The instrumentor injects them into ``ClaudeSDKClient.options`` on the way
    into ``query()``, before the call reaches the transport, so a client that
    was never connected still shows exactly what a connected one would send.
    """
    ClaudeAgentSDKInstrumentor().instrument()
    client = ClaudeSDKClient(options=options)
    with pytest.raises(CLIConnectionError):
        await client.query("anything")
    hooks = client.options.hooks or {}
    return {str(event): matchers for event, matchers in hooks.items()}


async def run_pre_tool_use(
    matchers: list[HookMatcher], tool_name: str
) -> list[HookJSONOutput]:
    """Run every ``PreToolUse`` hook in order, the way the CLI does."""
    hook_input: PreToolUseHookInput = {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"question": QUESTION},
        "tool_use_id": TOOL_USE_ID,
        "session_id": "session-1",
        "transcript_path": "",
        "cwd": "",
    }
    return [
        await hook(hook_input, TOOL_USE_ID, HookContext(signal=None))
        for matcher in matchers
        for hook in matcher.hooks
    ]


def permission_decision(output: HookJSONOutput) -> str:
    payload: dict[str, Any] = dict(output)
    if not payload:
        return "none"
    decision: str = payload["hookSpecificOutput"]["permissionDecision"]
    return decision


async def test_the_interrupt_hook_still_decides_once_the_instrumentor_merges_its_own(
    hitl_agent, fake_client, tmp_path, session_store
):
    """The whole suspend feature rides on this hook reaching the CLI.

    The instrumentor injects tracing hooks into the same ``PreToolUse`` event.
    If it ever replaced the list instead of merging into it, the interrupt hook
    would vanish, every ``interrupt()`` would stop parking its call, and the run
    would report success having skipped the human step.
    """
    runtime = _make_runtime(hitl_agent, tmp_path, session_store)
    await runtime.execute({"input": "ask me"})
    ours = fake_client.last_options

    merged = await merged_hooks(ours)

    assert merged["PreToolUse"][0] is ours.hooks["PreToolUse"][0], (
        "The interrupt hook must stay, and stay first, so its decision is the "
        "one the CLI reads."
    )
    decisions = [
        permission_decision(output)
        for output in await run_pre_tool_use(merged["PreToolUse"], INTERRUPT_TOOL)
    ]
    assert decisions == ["defer", "none"]


async def test_an_agent_without_uipath_tools_contributes_no_hook_of_its_own(
    plain_agent, fake_client, tmp_path, session_store
):
    """Every hook the CLI ends up with belongs to the instrumentor, none to us.

    ``test_an_agent_without_uipath_tools_gets_the_options_it_wrote`` pins that
    this runtime injects nothing. This pins the other half: what the tracing
    instrumentor adds on top is the instrumentor's, so an agent still behaves
    identically wherever it runs with the same instrumentation installed.
    """
    plain_agent.options = ClaudeAgentOptions()
    runtime = _make_runtime(plain_agent, tmp_path, session_store)
    await runtime.execute({"input": "hi"})
    ours = fake_client.last_options
    assert ours.hooks is None

    merged = await merged_hooks(ours)

    assert set(merged) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    owners = {
        hook.__module__
        for matchers in merged.values()
        for matcher in matchers
        for hook in matcher.hooks
    }
    assert all(owner.startswith("openinference.") for owner in owners), owners
    assert [
        permission_decision(output)
        for output in await run_pre_tool_use(merged["PreToolUse"], "Bash")
    ] == ["none"]


# --- LLM spans come from the gateway and nowhere else ------------------------


def test_every_gateway_call_becomes_one_llm_span(telemetry, exporter):
    telemetry.on_gateway_call(make_gateway_record(input_tokens=11, output_tokens=7))
    telemetry.on_gateway_call(
        make_gateway_record(
            resolved_model="claude-opus-4-1", input_tokens=3, output_tokens=5
        )
    )
    telemetry.flush_gateway_calls()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [LLM_SPAN_NAME, LLM_SPAN_NAME]

    first, second = (attrs(span) for span in spans)
    assert first[SPAN_KIND] == "LLM"
    assert first["llm.model_name"] == SERVING_MODEL
    assert first["llm.token_count.prompt"] == 11
    assert first["llm.token_count.completion"] == 7
    assert first["llm.token_count.total"] == 18

    assert second[SPAN_KIND] == "LLM"
    assert second["llm.model_name"] == "claude-opus-4-1"
    assert second["llm.token_count.prompt"] == 3
    assert second["llm.token_count.completion"] == 5
    assert second["llm.token_count.total"] == 8

    for span in spans:
        assert span.start_time is not None and span.end_time is not None
        assert span.end_time > span.start_time
        assert span.status.status_code is StatusCode.OK


def test_a_call_the_gateway_traced_gets_no_span_from_us(telemetry, exporter):
    """The gateway records the model span itself when it was handed the trace.

    Both sides emitting produces the same call twice, which is why every other
    UiPath framework drops its client-side span once the headers go out.
    """
    telemetry.on_gateway_call(make_gateway_record(traced_upstream=True))
    telemetry.flush_gateway_calls()

    assert span_names(exporter) == []


def test_a_call_the_gateway_did_not_trace_still_gets_our_span(telemetry, exporter):
    """The fallback, for a platform with the trace-context feature turned off."""
    telemetry.on_gateway_call(make_gateway_record(traced_upstream=False))
    telemetry.flush_gateway_calls()

    assert span_names(exporter) == [LLM_SPAN_NAME]


def test_trace_context_headers_need_a_turn_to_hang_from(telemetry):
    """No adopted parent means no span to name, so nothing is claimed."""
    assert telemetry.trace_context_headers() == {}


def test_cached_prompt_tokens_join_the_prompt_count(telemetry, exporter):
    telemetry.on_gateway_call(
        make_gateway_record(
            input_tokens=10,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=5,
            output_tokens=2,
        )
    )
    telemetry.flush_gateway_calls()

    span = attrs(exporter.get_finished_spans()[0])
    assert span["llm.token_count.prompt"] == 115
    assert span["llm.token_count.completion"] == 2


def test_failed_gateway_call_span_carries_the_error(telemetry, exporter):
    telemetry.on_gateway_call(make_gateway_record(error="upstream refused"))
    telemetry.flush_gateway_calls()

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "upstream refused"


async def test_a_run_emits_a_span_for_every_call_the_gateway_served(
    structured_agent, fake_client, tmp_path, session_store, telemetry, exporter
):
    """No assistant turn claims a record any more, so nothing is ever dropped.

    Background and subagent traffic never reaches the SDK message stream, and
    the CLI's own per-message counts describe a fragment of a call rather than
    the call. The shim saw all of it, so all of it is reported.
    """
    telemetry.on_gateway_call(make_gateway_record())
    telemetry.on_gateway_call(make_gateway_record(resolved_model="claude-opus-4-1"))
    runtime = _make_runtime(structured_agent, tmp_path, session_store, telemetry)

    await runtime.execute({"topic": "penguins"})

    assert span_names(exporter) == [LLM_SPAN_NAME, LLM_SPAN_NAME]


async def test_a_run_without_telemetry_emits_nothing(
    structured_agent, fake_client, tmp_path, session_store, exporter
):
    runtime = _make_runtime(structured_agent, tmp_path, session_store)

    await runtime.execute({"topic": "penguins"})

    assert exporter.get_finished_spans() == ()


async def test_gateway_shim_is_given_the_telemetry_sink(
    plain_agent, fake_client, tmp_path, session_store, telemetry, monkeypatch
):
    captured: dict[str, object] = {}

    class RecordingShim:
        def __init__(self, llm, **kwargs):
            captured.update(kwargs)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        @property
        def base_url(self) -> str:
            return "http://127.0.0.1:1234"

        @property
        def resolved_model(self) -> str:
            return "resolved-sonnet"

        def build_env(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.runtime.GatewayShim", RecordingShim, raising=True
    )
    plain_agent.uipath_llm = UiPathModel("claude-sonnet-4-5")
    runtime = _make_runtime(plain_agent, tmp_path, session_store, telemetry)
    await runtime.execute({"input": "hi"})

    assert captured["on_call"] == telemetry.on_gateway_call


# --- Attributes that must not reach UiPath -----------------------------------


def emit_agent_span(provider: TracerProvider, cost: float | None = 0.0731) -> None:
    """One span shaped like the instrumentor's, cost attribute included."""
    attributes: dict[str, Any] = {
        SPAN_KIND: "AGENT",
        "llm.model_name": SERVING_MODEL,
        "llm.token_count.prompt": 945,
    }
    if cost is not None:
        attributes[LLM_COST_TOTAL] = cost
    provider.get_tracer("test").start_span(
        "ClaudeAgentSDK.ClaudeSDKClient.receive_response", attributes=attributes
    ).end()


def emit_tool_span(provider: TracerProvider, tool_use_id: str = TOOL_USE_ID) -> None:
    """One span shaped like the instrumentor's TOOL span.

    ``tool.id`` carries the ``tool_use_id`` the model minted for this one
    invocation, which is what the instrumentor writes there.
    """
    provider.get_tracer("test").start_span(
        f"Tool.{INTERRUPT_TOOL}",
        attributes={
            SPAN_KIND: "TOOL",
            TOOL_ID: tool_use_id,
            TOOL_NAME_ATTR: INTERRUPT_TOOL,
            "input.value": '{"question": "Should I ship it?"}',
        },
    ).end()


def test_no_exported_span_carries_a_cost(exporter):
    """total_cost_usd is the CLI's own estimate, priced off Anthropic's public
    catalogue, and says nothing about what a gateway-routed model costs."""
    provider = TracerProvider()
    provider.add_span_processor(AttributeStripper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    emit_agent_span(provider)

    exported = attrs(exporter.get_finished_spans()[0])
    assert [key for key in exported if "cost" in key.lower()] == []
    # Only the cost goes: the rest of the span is untouched.
    assert exported[SPAN_KIND] == "AGENT"
    assert exported["llm.model_name"] == SERVING_MODEL
    assert exported["llm.token_count.prompt"] == 945


def test_no_exported_tool_span_carries_a_tool_id(exporter):
    """A per-invocation tool_use_id is not the stable tool identifier UiPath
    reads out of tool.id, so it must not be exported under that name."""
    provider = TracerProvider()
    provider.add_span_processor(AttributeStripper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    emit_tool_span(provider)

    exported = attrs(exporter.get_finished_spans()[0])
    assert TOOL_ID not in exported
    # Everything the evaluators do read is left in place.
    assert exported[TOOL_NAME_ATTR] == INTERRUPT_TOOL
    assert exported["input.value"] == '{"question": "Should I ship it?"}'


async def test_tool_call_evaluators_match_our_spans_by_name(exporter):
    """The contract the stripper exists for.

    ``uipath eval`` buckets a tool call under its ``tool.id`` when the span has
    one and only falls back to ``tool.name``. A criterion is authored against
    the name, so leaving the tool_use_id in place scores every tool-call
    evaluator 0.
    """
    from uipath.eval.evaluators import ToolCallCountEvaluator, ToolCallOrderEvaluator
    from uipath.eval.evaluators.tool_call_count_evaluator import (
        ToolCallCountEvaluationCriteria,
        ToolCallCountEvaluatorConfig,
    )
    from uipath.eval.evaluators.tool_call_order_evaluator import (
        ToolCallOrderEvaluationCriteria,
        ToolCallOrderEvaluatorConfig,
    )
    from uipath.eval.models import WorkloadExecution

    provider = TracerProvider()
    provider.add_span_processor(AttributeStripper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    emit_tool_span(provider)

    execution = WorkloadExecution(
        agent_input={},
        workload_output={},
        workload_trace=list(exporter.get_finished_spans()),
    )

    # The three generic type fields are filled in by a model validator, so they
    # are not arguments even though mypy's pydantic plugin reads them as such.
    count = await ToolCallCountEvaluator(  # type: ignore[call-arg]
        id="count", evaluatorConfig=ToolCallCountEvaluatorConfig(name="count")
    ).evaluate(
        execution,
        ToolCallCountEvaluationCriteria(tool_calls_count={INTERRUPT_TOOL: ("=", 1)}),
    )
    order = await ToolCallOrderEvaluator(  # type: ignore[call-arg]
        id="order", evaluatorConfig=ToolCallOrderEvaluatorConfig(name="order")
    ).evaluate(
        execution,
        ToolCallOrderEvaluationCriteria(tool_calls_order=[INTERRUPT_TOOL]),
    )

    assert count.score == 1.0
    assert order.score == 1.0


def test_a_span_without_a_cost_is_left_alone(exporter):
    provider = TracerProvider()
    provider.add_span_processor(AttributeStripper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    emit_agent_span(provider, cost=None)

    assert attrs(exporter.get_finished_spans()[0])["llm.token_count.prompt"] == 945


def test_the_factory_strips_the_attributes_before_the_exporters_run(
    tmp_path, monkeypatch
):
    """The stripper is registered through the trace manager, so it lands among
    the exporters and runs before the ones the CLI adds once a job is known."""
    monkeypatch.chdir(tmp_path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    trace_manager = FakeTraceManager(provider)

    context = UiPathRuntimeContext(runtime_dir=str(tmp_path / "__uipath"))
    context.trace_manager = trace_manager  # type: ignore[assignment]
    UiPathClaudeSDKRuntimeFactory(context=context)
    # Registered after the factory, exactly as the CLI registers LLM Ops.
    trace_manager.add_span_processor(SimpleSpanProcessor(exporter))

    emit_agent_span(provider)
    emit_tool_span(provider)

    exported = [attrs(span) for span in exporter.get_finished_spans()]
    assert LLM_COST_TOTAL not in exported[0]
    assert TOOL_ID not in exported[1]


# --- Factory wiring ----------------------------------------------------------


class _DelegatingProcessor(SpanProcessor):
    """One processor on the provider, fronting a list that can grow later."""

    def __init__(self) -> None:
        self.processors: list[SpanProcessor] = []

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        for processor in self.processors:
            processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        for processor in self.processors:
            processor.on_end(span)


class FakeTraceManager:
    """The parts of ``UiPathTraceManager`` the factory touches.

    Processors sit behind a single delegate on the provider, as the real one
    arranges them, which is what makes registration order decide who sees a
    span first.
    """

    def __init__(self, provider: TracerProvider) -> None:
        self.tracer_provider = provider
        self._delegate = _DelegatingProcessor()
        provider.add_span_processor(self._delegate)

    def add_span_processor(self, span_processor: SpanProcessor) -> "FakeTraceManager":
        self._delegate.processors.append(span_processor)
        return self


def _make_delegate(factory: UiPathClaudeSDKRuntimeFactory, runtime_id: str):
    return factory._create_delegate(
        agent=None,  # type: ignore[arg-type]
        runtime_id=runtime_id,
        entrypoint="agent",
        storage=None,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("conversational", [False, True])
def test_factory_assigns_telemetry_per_runtime(tmp_path, monkeypatch, conversational):
    monkeypatch.chdir(tmp_path)
    context = UiPathRuntimeContext(
        runtime_dir=str(tmp_path / "__uipath"),
        conversation_id="conv-1" if conversational else None,
    )
    factory = UiPathClaudeSDKRuntimeFactory(context=context)
    runtime = _make_delegate(factory, "rt-1")
    other = _make_delegate(factory, "rt-2")

    expected = (
        UiPathClaudeSDKConversationalRuntime
        if conversational
        else UiPathClaudeSDKRuntime
    )
    assert isinstance(runtime, expected)
    assert isinstance(runtime.telemetry, GatewayCallTelemetry)
    # Concurrent runs must not share a gateway call buffer.
    assert runtime.telemetry is not other.telemetry


def test_factory_instruments_the_claude_agent_sdk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = UiPathRuntimeContext(runtime_dir=str(tmp_path / "__uipath"))

    UiPathClaudeSDKRuntimeFactory(context=context)

    assert ClaudeAgentSDKInstrumentor().is_instrumented_by_opentelemetry


def test_factory_uses_the_context_trace_manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    context = UiPathRuntimeContext(runtime_dir=str(tmp_path / "__uipath"))
    trace_manager = FakeTraceManager(provider)
    trace_manager.add_span_processor(SimpleSpanProcessor(exporter))
    context.trace_manager = trace_manager  # type: ignore[assignment]
    factory = UiPathClaudeSDKRuntimeFactory(context=context)
    telemetry = _make_delegate(factory, "rt-1").telemetry
    assert isinstance(telemetry, GatewayCallTelemetry)

    telemetry.on_gateway_call(make_gateway_record())
    telemetry.flush_gateway_calls()

    assert span_names(exporter) == [LLM_SPAN_NAME]


# --- What the LLM span actually carries -------------------------------------

_SSE_TURN = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"m",'
    b'"type":"message","role":"assistant","model":"m","content":[],'
    b'"stop_reason":null,"usage":{"input_tokens":11,"output_tokens":2}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Looking that up"}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":1,'
    b'"content_block":{"type":"tool_use","id":"toolu_9","name":"get_rate",'
    b'"input":{}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
    b'"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Paris\\"}"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta",'
    b'"delta":{"stop_reason":"tool_use"}}\n\n'
)

_REQUEST = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 4096,
    "stream": True,
    "system": [{"type": "text", "text": "You are terse."}],
    "messages": [{"role": "user", "content": "What is the rate in Paris?"}],
}


def _served_span(telemetry, exporter):
    """One gateway call, teed exactly as the router tees a live stream."""
    from uipath_claude_sdk.runtime.gateway.records import UsageTee

    tee = UsageTee()
    tee.feed(_SSE_TURN)
    record = make_gateway_record(input_tokens=None, output_tokens=None)
    record.request_body = _REQUEST
    tee.apply(record)
    telemetry.on_gateway_call(record)
    telemetry.flush_gateway_calls()
    return exporter.get_finished_spans()[-1]


def test_the_llm_span_carries_the_prompt(telemetry, exporter):
    """A trace with no prompt cannot answer what the model was asked."""
    a = attrs(_served_span(telemetry, exporter))
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.0.message.content"] == "You are terse."
    assert a["llm.input_messages.1.message.role"] == "user"
    assert a["llm.input_messages.1.message.content"] == "What is the rate in Paris?"
    assert json.loads(a["llm.invocation_parameters"])["max_tokens"] == 4096


def test_the_llm_span_carries_the_answer_and_its_tool_calls(telemetry, exporter):
    a = attrs(_served_span(telemetry, exporter))
    assert a["llm.output_messages.0.message.role"] == "assistant"
    assert a["llm.output_messages.0.message.content"] == "Looking that up"
    assert (
        a["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]
        == "get_rate"
    )
    arguments = a[
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
    ]
    assert json.loads(arguments) == {"city": "Paris"}
    assert a["llm.stop_reason"] == "tool_use"


def test_usage_still_comes_from_the_stream(telemetry, exporter):
    """Accumulating content must not disturb the counters sharing that stream."""
    a = attrs(_served_span(telemetry, exporter))
    assert a["llm.token_count.prompt"] == 11
    assert a["llm.token_count.completion"] == 2


def test_the_llm_span_nests_under_the_turn_it_served(telemetry, exporter, provider):
    """An orphan LLM span sorts by start time and reads as unrelated work.

    The shim serves calls on its own task and the spans are built after the
    turn ends, so the agent span has to be carried over deliberately.
    """
    from opentelemetry import context as context_api

    parent_tracer = provider.get_tracer("test-agent")
    with parent_tracer.start_as_current_span("AgentTurn") as agent_span:
        telemetry.adopt_parent(context_api.get_current())
        expected = agent_span.get_span_context().span_id
    telemetry.on_gateway_call(make_gateway_record())
    telemetry.flush_gateway_calls()

    llm = next(s for s in exporter.get_finished_spans() if s.name == LLM_SPAN_NAME)
    assert llm.parent is not None, "The LLM span was emitted without a parent."
    assert llm.parent.span_id == expected
