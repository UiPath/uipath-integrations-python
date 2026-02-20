"""Tests for streaming event pairing (STARTED/COMPLETED).

Uses real Agent Framework agents and workflows with mocked execution
to verify that the runtime emits matched STARTED/COMPLETED state events
for every node in the graph.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework import (
    AgentExecutor,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    RawAgent,
    WorkflowAgent,
    WorkflowBuilder,
)
from uipath.runtime import UiPathRuntimeResult
from uipath.runtime.events import (
    UiPathRuntimeStateEvent,
    UiPathRuntimeStatePhase,
)

from uipath_agent_framework.runtime.factory import UiPathAgentFrameworkRuntimeFactory
from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime

STARTED = UiPathRuntimeStatePhase.STARTED
COMPLETED = UiPathRuntimeStatePhase.COMPLETED


# ---------------------------------------------------------------------------
# Async stream mock
# ---------------------------------------------------------------------------


class _MockAsyncStream:
    """Async iterable with get_final_response() support."""

    def __init__(self, items: list[Any], final: Any = None):
        self._items = list(items)
        self._final = final or MagicMock(text="done")

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def get_final_response(self):
        return self._final


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _wf_event(event_type: str, executor_id: str, data: Any = None) -> MagicMock:
    evt = MagicMock()
    evt.type = event_type
    evt.executor_id = executor_id
    evt.data = data
    return evt


# ---------------------------------------------------------------------------
# Real tools (no LLM needed)
# ---------------------------------------------------------------------------


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a topic."""
    return f"Result for: {query}"


def run_python(code: str) -> str:
    """Execute a Python code snippet."""
    return f"Output: {code}"


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


# ---------------------------------------------------------------------------
# Agent / runtime setup helpers
# ---------------------------------------------------------------------------

_mock_client = MagicMock()


def _make_runtime(agent: BaseAgent) -> UiPathAgentFrameworkRuntime:
    """Create a runtime with mocked chat mapper."""
    runtime = UiPathAgentFrameworkRuntime(agent=agent)
    runtime.chat = MagicMock()
    runtime.chat.map_messages_to_input.return_value = "test"
    runtime.chat.map_streaming_content.return_value = []
    runtime.chat.close_message.return_value = []
    return runtime


async def _collect_events(runtime: UiPathAgentFrameworkRuntime) -> list[Any]:
    events: list[Any] = []
    async for event in runtime.stream(input=None):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _state_events(events: list[Any]) -> list[tuple[str, UiPathRuntimeStatePhase]]:
    return [
        (e.node_name, e.phase)
        for e in events
        if isinstance(e, UiPathRuntimeStateEvent) and e.node_name is not None
    ]


def _assert_all_completed(se: list[tuple[str, UiPathRuntimeStatePhase]]) -> None:
    """Every node that was STARTED must also have a COMPLETED event."""
    started = {n for n, p in se if p == STARTED}
    completed = {n for n, p in se if p == COMPLETED}
    missing = started - completed
    assert not missing, f"STARTED but never COMPLETED: {missing}"


def _assert_started_before_completed(
    se: list[tuple[str, UiPathRuntimeStatePhase]], node: str
) -> None:
    first_started = next(
        (i for i, (n, p) in enumerate(se) if n == node and p == STARTED), None
    )
    first_completed = next(
        (i for i, (n, p) in enumerate(se) if n == node and p == COMPLETED), None
    )
    assert first_started is not None, f"{node} never STARTED"
    assert first_completed is not None, f"{node} never COMPLETED"
    assert first_started < first_completed, f"{node}: COMPLETED before STARTED"


# ===========================================================================
# Workflow streaming tests
# ===========================================================================


class TestWorkflowStreamingEvents:
    """Verify STARTED/COMPLETED pairing for workflow streaming."""

    async def test_simple_workflow(self):
        """Workflow with two executors: all nodes paired."""
        triage = RawAgent(_mock_client, name="triage")
        billing = RawAgent(_mock_client, name="billing", tools=[calculator])

        workflow = (
            WorkflowBuilder(start_executor=triage).add_edge(triage, billing).build()  # type: ignore[arg-type]
        )
        agent = WorkflowAgent(workflow=workflow, name="my_workflow")

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _wf_event("executor_invoked", "triage"),
                    _wf_event("executor_completed", "triage"),
                    _wf_event("executor_invoked", "billing"),
                    _wf_event("executor_completed", "billing"),
                ],
                final,
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "my_workflow")
        _assert_started_before_completed(se, "triage")
        _assert_started_before_completed(se, "billing")
        assert isinstance(events[-1], UiPathRuntimeResult)

    async def test_multi_executor_workflow(self):
        """Workflow with three executors in sequence."""
        a = RawAgent(_mock_client, name="step_a")
        b = RawAgent(_mock_client, name="step_b", tools=[search_wikipedia])
        c = RawAgent(_mock_client, name="step_c", tools=[run_python])

        workflow = (
            WorkflowBuilder(start_executor=a).add_edge(a, b).add_edge(b, c).build()  # type: ignore[arg-type]
        )
        agent = WorkflowAgent(workflow=workflow, name="pipeline")

        wf_events: list[Any] = []
        for name in ["step_a", "step_b", "step_c"]:
            wf_events.append(_wf_event("executor_invoked", name))
            wf_events.append(_wf_event("executor_completed", name))

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(return_value=_MockAsyncStream(wf_events, final))  # type: ignore[method-assign]
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        for name in ["pipeline", "step_a", "step_b", "step_c"]:
            _assert_started_before_completed(se, name)

    async def test_workflow_root_wraps_executors(self):
        """Root workflow STARTED is first, COMPLETED is last state event."""
        a = RawAgent(_mock_client, name="worker")
        workflow = WorkflowBuilder(start_executor=a).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="wf")

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _wf_event("executor_invoked", "worker"),
                    _wf_event("executor_completed", "worker"),
                ],
                final,
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        assert se[0] == ("wf", STARTED)
        assert se[-1] == ("wf", COMPLETED)


# ===========================================================================
# Factory tests — no agent caching
# ===========================================================================


class TestFactoryNoCaching:
    """Verify factory creates fresh agent instances per runtime."""

    async def test_concurrent_new_runtime_gets_separate_agents(self):
        """Multiple concurrent new_runtime calls must each get their own agent."""
        context = MagicMock()
        context.resolved_state_file_path = ":memory:"
        context.resume = False
        context.job_id = None
        context.keep_state_file = False

        factory = UiPathAgentFrameworkRuntimeFactory(context)

        # Track every agent instance returned by _load_agent
        loaded_agents: list[BaseAgent] = []

        async def _fake_load_agent(entrypoint: str) -> BaseAgent:
            agent = MagicMock(spec=BaseAgent)
            agent.name = f"agent_{len(loaded_agents)}"
            loaded_agents.append(agent)
            return agent

        with patch.object(factory, "_load_agent", side_effect=_fake_load_agent):
            with patch.object(factory, "_get_storage", new_callable=AsyncMock):
                runtimes = await asyncio.gather(
                    factory.new_runtime("agent", "runtime_1"),
                    factory.new_runtime("agent", "runtime_2"),
                    factory.new_runtime("agent", "runtime_3"),
                )

        # Each runtime must have gotten a separate agent instance
        assert len(loaded_agents) == 3
        # Factory wraps in UiPathResumableRuntime; access delegate.agent
        agents = [r.delegate.agent for r in runtimes]  # type: ignore[attr-defined]
        assert len(set(id(a) for a in agents)) == 3, "Runtimes share agent instances!"


# ===========================================================================
# Tool state event tests
# ===========================================================================


class TestToolStateEvents:
    """Verify that output events with function_call/function_result Content
    emit STARTED/COMPLETED state events for tool nodes."""

    async def test_tool_call_emits_state_events(self):
        """Output event with function_call + function_result Content should
        produce STARTED and COMPLETED state events for '{executor}_tools'."""
        worker = RawAgent(_mock_client, name="weather_agent", tools=[calculator])
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="wf")

        # Simulate a tool call cycle: function_call then function_result
        call_content = Content(
            type="function_call",
            name="calculator",
            call_id="call_1",
            arguments='{"expression": "2+2"}',
        )
        result_content = Content(
            type="function_result",
            call_id="call_1",
            result="4",
        )

        call_update = AgentResponseUpdate(contents=[call_content])
        result_update = AgentResponseUpdate(contents=[result_content])

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _wf_event("executor_invoked", "weather_agent"),
                    _wf_event("output", "weather_agent", data=call_update),
                    _wf_event("output", "weather_agent", data=result_update),
                    _wf_event("executor_completed", "weather_agent"),
                ],
                final,
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)

        # Tool node should have STARTED and COMPLETED
        tool_events = [(n, p) for n, p in se if n == "weather_agent_tools"]
        assert ("weather_agent_tools", STARTED) in tool_events
        assert ("weather_agent_tools", COMPLETED) in tool_events
        _assert_started_before_completed(se, "weather_agent_tools")

        # The STARTED event should carry the tool_name payload
        tool_started = [
            e
            for e in events
            if isinstance(e, UiPathRuntimeStateEvent)
            and e.node_name == "weather_agent_tools"
            and e.phase == STARTED
        ]
        assert len(tool_started) == 1
        assert tool_started[0].payload == {"tool_name": "calculator"}

    async def test_multiple_tool_calls_emit_paired_events(self):
        """Multiple tool call cycles should each produce a STARTED/COMPLETED pair."""
        worker = RawAgent(
            _mock_client,
            name="multi_tool_agent",
            tools=[calculator, search_wikipedia],
        )
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="wf")

        updates = [
            AgentResponseUpdate(
                contents=[
                    Content(
                        type="function_call",
                        name="calculator",
                        call_id="c1",
                        arguments="{}",
                    )
                ]
            ),
            AgentResponseUpdate(
                contents=[
                    Content(type="function_result", call_id="c1", result="42")
                ]
            ),
            AgentResponseUpdate(
                contents=[
                    Content(
                        type="function_call",
                        name="search_wikipedia",
                        call_id="c2",
                        arguments="{}",
                    )
                ]
            ),
            AgentResponseUpdate(
                contents=[
                    Content(type="function_result", call_id="c2", result="found")
                ]
            ),
        ]

        wf_events = [_wf_event("executor_invoked", "multi_tool_agent")]
        for upd in updates:
            wf_events.append(_wf_event("output", "multi_tool_agent", data=upd))
        wf_events.append(_wf_event("executor_completed", "multi_tool_agent"))

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(return_value=_MockAsyncStream(wf_events, final))  # type: ignore[method-assign]
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        tool_events = [(n, p) for n, p in se if n == "multi_tool_agent_tools"]

        # Two STARTED + two COMPLETED
        assert tool_events.count(("multi_tool_agent_tools", STARTED)) == 2
        assert tool_events.count(("multi_tool_agent_tools", COMPLETED)) == 2

    async def test_no_tool_events_for_text_content(self):
        """Output events with only text Content should not emit tool state events."""
        worker = RawAgent(_mock_client, name="text_agent")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="wf")

        text_update = AgentResponseUpdate(
            contents=[Content(type="text", text="Hello world")]
        )

        final = MagicMock()
        final.get_outputs.return_value = []
        workflow.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _wf_event("executor_invoked", "text_agent"),
                    _wf_event("output", "text_agent", data=text_update),
                    _wf_event("executor_completed", "text_agent"),
                ],
                final,
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        tool_events = [(n, p) for n, p in se if "_tools" in n]
        assert tool_events == [], "Text-only output should not emit tool state events"

    async def test_executor_completed_payload_excludes_streaming_updates(self):
        """executor_completed StateEvent payload must NOT contain
        AgentResponseUpdate streaming chunks — only the summary data."""
        worker = RawAgent(_mock_client, name="agent_x", tools=[calculator])
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="wf")

        # Simulate streaming: many AgentResponseUpdate chunks arrive as output
        # events, then executor_completed carries sent_messages + yielded_outputs
        text_chunks = [
            AgentResponseUpdate(contents=[Content(type="text", text=tok)])
            for tok in ["Hello", " ", "world"]
        ]
        call_chunk = AgentResponseUpdate(
            contents=[
                Content(
                    type="function_call",
                    name="calculator",
                    call_id="c1",
                    arguments="{}",
                )
            ]
        )
        result_chunk = AgentResponseUpdate(
            contents=[
                Content(type="function_result", call_id="c1", result="42")
            ]
        )

        # The framework packs sent_messages + yielded_outputs into completed data
        summary = MagicMock()  # represents AgentExecutorResponse (not an update)
        completed_data = [summary] + text_chunks + [call_chunk, result_chunk]

        final = MagicMock()
        final.get_outputs.return_value = []

        stream_events = [
            _wf_event("executor_invoked", "agent_x"),
        ]
        for chunk in text_chunks + [call_chunk, result_chunk]:
            stream_events.append(_wf_event("output", "agent_x", data=chunk))
        stream_events.append(
            _wf_event("executor_completed", "agent_x", data=completed_data)
        )

        workflow.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(stream_events, final)
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        # Find the executor_completed state event for agent_x
        completed_events = [
            e
            for e in events
            if isinstance(e, UiPathRuntimeStateEvent)
            and e.node_name == "agent_x"
            and e.phase == COMPLETED
        ]
        assert len(completed_events) == 1

        # The payload must NOT contain any AgentResponseUpdate data
        payload = completed_events[0].payload
        payload_str = str(payload)
        assert "agent_response_update" not in payload_str.lower()
        # Text token chunks should not appear in the completed payload
        assert "Hello" not in payload_str or "world" not in payload_str


# ===========================================================================
# Checkpoint propagation tests
# ===========================================================================


class TestCheckpointPropagation:
    """Verify that checkpoint_storage is passed to workflow.run()."""

    async def test_checkpoint_storage_passed_to_workflow_run_stream(self):
        """Streaming: workflow.run() should receive checkpoint_storage parameter."""
        worker = RawAgent(_mock_client, name="assistant")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="chat_wf")

        mock_checkpoint_storage = MagicMock()
        captured_kwargs: list[dict[str, Any]] = []

        def mock_run(**kwargs):
            captured_kwargs.append(kwargs)
            final = MagicMock()
            final.get_outputs.return_value = []
            return _MockAsyncStream(
                [
                    _wf_event("executor_invoked", "assistant"),
                    _wf_event("executor_completed", "assistant"),
                ],
                final,
            )

        workflow.run = mock_run  # type: ignore[method-assign]

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="test-session",
            checkpoint_storage=mock_checkpoint_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "hello"
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        await _collect_events(runtime)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["checkpoint_storage"] is mock_checkpoint_storage
        assert captured_kwargs[0]["stream"] is True

    async def test_checkpoint_storage_passed_to_workflow_run_execute(self):
        """Non-streaming execute() should pass checkpoint_storage to workflow.run()."""
        worker = RawAgent(_mock_client, name="assistant")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="exec_wf")

        mock_checkpoint_storage = MagicMock()
        captured_kwargs: list[dict[str, Any]] = []

        async def mock_run(**kwargs):
            captured_kwargs.append(kwargs)
            result = MagicMock()
            result.get_outputs.return_value = ["done"]
            return result

        workflow.run = mock_run  # type: ignore[method-assign]

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="exec-session",
            checkpoint_storage=mock_checkpoint_storage,
        )

        await runtime.execute(input={"messages": []})

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["checkpoint_storage"] is mock_checkpoint_storage

    async def test_hitl_resume_passes_checkpoint_id_and_responses(self):
        """HITL resume: workflow.run() should receive responses, checkpoint_id, and checkpoint_storage."""
        worker = RawAgent(_mock_client, name="assistant")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="resume_wf")
        workflow.name = "resume_wf"

        mock_checkpoint_storage = MagicMock()
        mock_checkpoint_storage.get_latest = AsyncMock(return_value=MagicMock(checkpoint_id="cp-123"))
        captured_kwargs: list[dict[str, Any]] = []

        def mock_run(**kwargs):
            captured_kwargs.append(kwargs)
            final = MagicMock()
            final.get_outputs.return_value = []
            return _MockAsyncStream(
                [
                    _wf_event("executor_invoked", "assistant"),
                    _wf_event("executor_completed", "assistant"),
                ],
                final,
            )

        workflow.run = mock_run  # type: ignore[method-assign]

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="resume-session",
            checkpoint_storage=mock_checkpoint_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = ""
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        options = MagicMock()
        options.resume = True
        options.breakpoints = None

        responses = {"req-1": "approved"}
        await _collect_events(runtime)

        # Trigger a resume via stream with responses
        runtime._resume_responses = responses
        events = []
        async for event in runtime._stream_workflow("", "resume_wf"):
            events.append(event)

        # The second call (resume) should have responses and checkpoint_id
        assert len(captured_kwargs) == 2
        resume_call = captured_kwargs[1]
        assert resume_call["responses"] == responses
        assert resume_call["checkpoint_id"] == "cp-123"
        assert resume_call["checkpoint_storage"] is mock_checkpoint_storage


# ===========================================================================
# Session propagation tests (multi-turn conversation history)
# ===========================================================================


class TestSessionPropagation:
    """Verify that sessions are loaded from storage and propagated to executors."""

    async def test_session_propagated_to_executors_in_stream(self):
        """Session loaded from resumable_storage should be set on each
        AgentExecutor before workflow.run(), so inner agents see conversation history."""
        worker = RawAgent(_mock_client, name="assistant")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="chat_wf")

        # Create a session with pre-existing state (simulating a prior turn)
        session = agent.create_session(session_id="test-session")
        session.state["prior_turn_data"] = "previous conversation"

        # Mock resumable_storage that returns our pre-populated session via KV
        mock_storage = AsyncMock()
        mock_storage.get_value = AsyncMock(return_value=session.to_dict())
        mock_storage.set_value = AsyncMock()

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="test-session",
            resumable_storage=mock_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "hello"
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        # Track which session gets set on the executor
        captured_sessions: list[AgentSession] = []

        def mock_run(**kwargs):
            # At the point workflow.run() is called, capture the executor's session
            executor = workflow.executors["assistant"]
            if isinstance(executor, AgentExecutor):
                captured_sessions.append(executor._session)
            final = MagicMock()
            final.get_outputs.return_value = []
            return _MockAsyncStream(
                [
                    _wf_event("executor_invoked", "assistant"),
                    _wf_event("executor_completed", "assistant"),
                ],
                final,
            )

        workflow.run = mock_run  # type: ignore[method-assign]

        await _collect_events(runtime)

        # Session should have been captured with prior turn data
        assert len(captured_sessions) == 1
        assert captured_sessions[0].state.get("prior_turn_data") == "previous conversation"

        # Session should have been loaded from KV storage
        mock_storage.get_value.assert_called_once_with("test-session", "session", "data")

        # Session should have been saved after execution
        mock_storage.set_value.assert_called_once()

    async def test_session_propagated_in_execute_path(self):
        """Non-streaming execute() should also propagate session to executors."""
        worker = RawAgent(_mock_client, name="assistant")
        workflow = WorkflowBuilder(start_executor=worker).build()  # type: ignore[arg-type]
        agent = WorkflowAgent(workflow=workflow, name="exec_wf")

        session = agent.create_session(session_id="exec-session")
        session.state["history_key"] = "turn1_data"

        mock_storage = AsyncMock()
        mock_storage.get_value = AsyncMock(return_value=session.to_dict())
        mock_storage.set_value = AsyncMock()

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="exec-session",
            resumable_storage=mock_storage,
        )

        captured_sessions: list[AgentSession] = []

        async def mock_run(**kwargs):
            executor = workflow.executors["assistant"]
            if isinstance(executor, AgentExecutor):
                captured_sessions.append(executor._session)
            result = MagicMock()
            result.get_outputs.return_value = ["done"]
            return result

        workflow.run = mock_run  # type: ignore[method-assign]

        await runtime.execute(input={"messages": []})

        assert len(captured_sessions) == 1
        assert captured_sessions[0].state.get("history_key") == "turn1_data"
        mock_storage.set_value.assert_called_once()
