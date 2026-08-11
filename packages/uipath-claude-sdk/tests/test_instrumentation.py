"""The run loop has to read a turn the way the tracing instrumentor expects.

``openinference-instrumentation-claude-agent-sdk`` wraps ``connect``, ``query``
and ``receive_response``. It does not wrap ``receive_messages``, and its tool
tracker only records while a ``receive_response`` span is open, so a loop that
reads the underlying stream directly produces no agent span, no tool spans, and
no parent for the gateway's model spans to nest under.

Nothing else in the suite notices that. The telemetry tests assert what the
attribute stripper does to spans shaped like the instrumentor's, which they build
by hand, so they keep passing when the instrumentor stops being reached at all.
These two facts compose into "the spans arrive", and each is cheap to hold:
upstream wraps that one method, and this runtime calls it once per turn.
"""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
)
from claude_agent_sdk.client import ClaudeSDKClient
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.conftest import (
    FakeClaudeSDKClient,
    make_result_message,
    make_session_paths,
)
from uipath_claude_sdk.interrupts import (
    InterruptToolBinding,
    SuspendChannel,
    _identifying_inputs_only,
    run_tool_body,
)
from uipath_claude_sdk.runtime.runtime import (
    UiPathClaudeSDKRuntime,
    _resume_shape_only,
)

WRAPPED = ("connect", "query", "receive_response")
NOT_WRAPPED = ("receive_messages",)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def test_the_instrumentor_wraps_receive_response_and_not_receive_messages():
    """A version bump that moves this surface has to fail loudly here.

    The whole reason the run loop reads a turn through the convenience method is
    that this is where the instrumentation lives. If a release starts wrapping
    ``receive_messages`` too, the constraint is gone and the loop is free again.
    """
    for name in WRAPPED + NOT_WRAPPED:
        assert not hasattr(getattr(ClaudeSDKClient, name), "__wrapped__")

    ClaudeAgentSDKInstrumentor().instrument()

    for name in WRAPPED:
        assert hasattr(getattr(ClaudeSDKClient, name), "__wrapped__"), (
            f"{name} is no longer instrumented"
        )
    for name in NOT_WRAPPED:
        assert not hasattr(getattr(ClaudeSDKClient, name), "__wrapped__"), (
            f"{name} is now instrumented, so the run loop no longer has to avoid it"
        )


class _RecordingClient(FakeClaudeSDKClient):
    """Records which stream method each turn was read through."""

    reads: list[str] = []

    async def receive_response(self):
        type(self).reads.append("receive_response")
        async for message in super().receive_response():
            yield message

    async def receive_messages(self):
        type(self).reads.append("receive_messages")
        async for message in super().receive_messages():
            yield message


@pytest.fixture
def recording_client(fake_client, monkeypatch: pytest.MonkeyPatch):
    _RecordingClient.reads = []
    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.runtime.ClaudeSDKClient", _RecordingClient
    )
    return _RecordingClient


async def _run(agent: Any, tmp_path: Any, session_store: Any) -> None:
    runtime = UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path, "rt-1"),
        runtime_id="rt-1",
    )
    await runtime.execute({"input": "which accounts are open?"})


async def test_a_turn_is_read_through_receive_response(
    recording_client, plain_agent, tmp_path, session_store
):
    recording_client.scripted_messages = [make_result_message(result="done")]

    await _run(plain_agent, tmp_path, session_store)

    assert recording_client.reads[0] == "receive_response"
    assert recording_client.reads.count("receive_response") == 1


class TestTracedInternalsKeepPayloadsOut:
    """``@traced`` records inputs and outputs on the span by default.

    The internals it is applied to handle a suspend value, a resume payload and
    a transcript, which are the customer's data and have no business on a span.
    Each of those spans therefore carries a processor, and these tests are what
    hold the processors to naming the work without copying it.
    """

    def test_a_suspension_is_named_without_its_value(self):
        kept = _identifying_inputs_only(
            {
                "tool_name": "review",
                "tool_use_id": "toolu_1",
                "value": {"data": {"AgentOutput": "customer's quote"}},
            }
        )

        assert kept == {"tool_name": "review", "tool_use_id": "toolu_1"}

    def test_a_resume_reports_only_that_it_is_resuming(self):
        kept = _resume_shape_only(
            {"resuming": True, "input": {"answer": "Acme Corporation"}}
        )

        assert kept == {"resuming": True}


class TestToolBodyTracing:
    """Where a span opened by a developer's tool body ends up.

    The client's tasks inherit their context from before the run began, so a
    connection lookup or an HTTP call inside a tool would otherwise be parented
    to whatever was current then and appear beside the turn instead of within
    it. The runtime hands the turn to the channel, and the body runs under it.
    """

    @staticmethod
    def _turn(provider: TracerProvider):
        turn = provider.get_tracer("test").start_span("turn")
        return turn, trace_api.set_span_in_context(turn)

    async def _run_body(self, channel: SuspendChannel, body: Any) -> None:
        await run_tool_body(
            channel,
            InterruptToolBinding(token="tok", body=body),
            {},
            "retrieve",
            "toolu_1",
        )

    async def test_a_span_from_a_tool_body_belongs_to_the_turn(self, exporter):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        turn, turn_context = self._turn(provider)
        channel = SuspendChannel()
        channel.adopt_turn_parent(turn_context)

        async def body(args: dict[str, Any]) -> dict[str, Any]:
            provider.get_tracer("test").start_span("connections_retrieve").end()
            return {"content": []}

        await self._run_body(channel, body)
        turn.end()

        emitted = {span.name: span for span in exporter.get_finished_spans()}
        parent = emitted["connections_retrieve"].parent
        assert parent is not None
        assert parent.span_id == turn.get_span_context().span_id

    async def test_a_tool_body_still_runs_before_any_turn_is_recorded(self):
        channel = SuspendChannel()
        ran = []

        async def body(args: dict[str, Any]) -> dict[str, Any]:
            ran.append(True)
            return {"content": []}

        await self._run_body(channel, body)

        assert ran == [True]


async def test_a_background_turn_is_read_through_receive_response_too(
    recording_client, plain_agent, tmp_path, session_store
):
    """A turn held open for background work still gets its own agent span."""
    recording_client.scripted_messages = [
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="agent-1",
            description="research",
            uuid="task-start-1",
            session_id="session-1",
            task_type="local_agent",
        ),
        make_result_message(result="Task launched."),
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="agent-1",
            status="completed",
            output_file="agent-1.output",
            summary="Research complete",
            uuid="task-end-1",
            session_id="session-1",
        ),
        AssistantMessage(
            content=[TextBlock(text="Final answer.")],
            model="claude-sonnet-4-5",
        ),
        make_result_message(result="Final answer."),
    ]

    await _run(plain_agent, tmp_path, session_store)

    assert recording_client.reads.count("receive_response") == 2
