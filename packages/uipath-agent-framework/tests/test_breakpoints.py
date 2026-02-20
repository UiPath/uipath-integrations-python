"""Tests for executor-level breakpoints.

Verifies that breakpoints pause execution before the targeted executor runs
and that resume correctly skips the breakpointed executor.

Includes integration tests with UiPathDebugRuntime to simulate the full
debug flow: breakpoints → pause → resume → continue.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

from agent_framework import RawAgent, WorkflowAgent, WorkflowBuilder
from uipath.runtime.debug import (
    UiPathBreakpointResult,
    UiPathDebugProtocol,
    UiPathDebugQuitError,
    UiPathDebugRuntime,
)
from uipath.runtime.events import UiPathRuntimeStateEvent
from uipath.runtime.result import UiPathRuntimeResult, UiPathRuntimeStatus

from uipath_agent_framework.runtime.breakpoints import (
    _resolve_to_executor_ids,
    create_breakpoint_result,
    inject_breakpoint_middleware,
    remove_breakpoint_middleware,
)
from uipath_agent_framework.runtime.interrupt import AgentInterruptException
from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime


_mock_client = MagicMock()


class _MockWorkflowStream:
    """Fake workflow stream for testing runtime orchestration without LLM.

    Simulates the async iterable returned by ``workflow.run(stream=True)``
    so we can control when breakpoint exceptions fire and what the final
    response looks like.
    """

    def __init__(
        self,
        events: list[Any] | None = None,
        exception: Exception | None = None,
        final_output: str = "done",
    ):
        self._events = events or []
        self._exception = exception
        self._final_output = final_output

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for event in self._events:
            yield event
        if self._exception:
            raise self._exception

    async def get_final_response(self):
        mock_result = MagicMock()
        mock_result.get_outputs.return_value = [self._final_output]
        return mock_result


def _make_debug_bridge(**overrides: Any) -> UiPathDebugProtocol:
    """Create a mock debug bridge with sensible defaults."""
    bridge: Mock = Mock(spec=UiPathDebugProtocol)
    bridge.connect = AsyncMock()
    bridge.disconnect = AsyncMock()
    bridge.emit_execution_started = AsyncMock()
    bridge.emit_execution_completed = AsyncMock()
    bridge.emit_execution_error = AsyncMock()
    bridge.emit_execution_suspended = AsyncMock()
    bridge.emit_breakpoint_hit = AsyncMock()
    bridge.emit_state_update = AsyncMock()
    bridge.emit_execution_resumed = AsyncMock()
    bridge.wait_for_resume = AsyncMock(return_value=None)
    bridge.wait_for_terminate = AsyncMock()
    bridge.get_breakpoints = Mock(return_value=[])
    for k, v in overrides.items():
        setattr(bridge, k, v)
    return cast(UiPathDebugProtocol, bridge)


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------


class TestResolveBreakpoints:
    """Verify graph node IDs are correctly resolved to executor IDs."""

    def test_wildcard_resolves_to_all_executors(self):
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, "*")
        assert result == set(workflow.executors.keys())

    def test_executor_id_resolves_directly(self):
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["agent_a"])
        assert result == {"agent_a"}

    def test_tools_suffix_resolves_to_parent_executor(self):
        a = RawAgent(_mock_client, name="agent_a", tools=[calculator])
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["agent_a_tools"])
        assert result == {"agent_a"}

    def test_tool_name_resolves_to_owning_executor(self):
        a = RawAgent(_mock_client, name="agent_a", tools=[calculator])
        b = RawAgent(_mock_client, name="agent_b", tools=[search_web])
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["calculator"])
        assert result == {"agent_a"}

        result = _resolve_to_executor_ids(agent, ["search_web"])
        assert result == {"agent_b"}

    def test_wildcard_in_list_resolves_to_all(self):
        """Wildcard passed as ["*"] (list) also resolves to all executors."""
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["*"])
        assert result == set(workflow.executors.keys())

    def test_unknown_node_id_ignored(self):
        a = RawAgent(_mock_client, name="agent_a")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["nonexistent"])
        assert result == set()

    def test_mixed_breakpoints(self):
        a = RawAgent(_mock_client, name="agent_a", tools=[calculator])
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        result = _resolve_to_executor_ids(agent, ["agent_b", "calculator"])
        assert result == {"agent_a", "agent_b"}


# ---------------------------------------------------------------------------
# Injection tests
# ---------------------------------------------------------------------------


class TestInjectBreakpoints:
    """Verify executor wrapping for breakpoint injection."""

    def test_inject_wraps_executor_execute(self):
        """Injecting breakpoints replaces executor.execute with a wrapper."""
        a = RawAgent(_mock_client, name="agent_a")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        original = workflow.executors["agent_a"].execute
        inject_breakpoint_middleware(agent, ["agent_a"])

        assert workflow.executors["agent_a"].execute is not original
        assert hasattr(workflow.executors["agent_a"], "_bp_original_execute")

        remove_breakpoint_middleware(agent)

    def test_remove_restores_original_execute(self):
        """Removing breakpoints restores the original execute method."""
        a = RawAgent(_mock_client, name="agent_a")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(agent, ["agent_a"])
        assert hasattr(workflow.executors["agent_a"], "_bp_original_execute")

        remove_breakpoint_middleware(agent)
        assert not hasattr(workflow.executors["agent_a"], "_bp_original_execute")

    async def test_wrapped_execute_raises_interrupt(self):
        """Wrapped executor raises AgentInterruptException on execute."""
        a = RawAgent(_mock_client, name="agent_a")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(agent, ["agent_a"])

        executor = workflow.executors["agent_a"]
        try:
            await executor.execute("msg", [], MagicMock(), MagicMock())
            assert False, "Should have raised AgentInterruptException"
        except AgentInterruptException as e:
            assert e.is_breakpoint is True
            assert e.suspend_value["type"] == "breakpoint"
            assert e.suspend_value["node_id"] == "agent_a"
        finally:
            remove_breakpoint_middleware(agent)

    def test_skip_nodes_not_wrapped(self):
        """Executors in skip_nodes should not be wrapped (resume scenario)."""
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(
            agent, ["agent_a", "agent_b"], skip_nodes={"agent_a"}
        )

        # agent_a should NOT be wrapped (it's in skip_nodes)
        assert not hasattr(workflow.executors["agent_a"], "_bp_original_execute")
        # agent_b SHOULD be wrapped
        assert hasattr(workflow.executors["agent_b"], "_bp_original_execute")

        remove_breakpoint_middleware(agent)

    def test_skip_nodes_multiple(self):
        """Multiple skip_nodes are all excluded from wrapping."""
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        c = RawAgent(_mock_client, name="agent_c")
        workflow = (
            WorkflowBuilder(start_executor=a)
            .add_edge(a, b)
            .add_edge(a, c)
            .build()
        )
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(
            agent, "*", skip_nodes={"agent_a", "agent_b"}
        )

        assert not hasattr(workflow.executors["agent_a"], "_bp_original_execute")
        assert not hasattr(workflow.executors["agent_b"], "_bp_original_execute")
        assert hasattr(workflow.executors["agent_c"], "_bp_original_execute")

        remove_breakpoint_middleware(agent)

    def test_no_double_wrap(self):
        """Calling inject twice doesn't double-wrap executors."""
        a = RawAgent(_mock_client, name="agent_a")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(agent, ["agent_a"])
        first_wrapped = workflow.executors["agent_a"].execute

        inject_breakpoint_middleware(agent, ["agent_a"])
        second_wrapped = workflow.executors["agent_a"].execute

        # Should be the same wrapper, not double-wrapped
        assert first_wrapped is second_wrapped

        remove_breakpoint_middleware(agent)

    def test_wildcard_wraps_all_executors(self):
        """Wildcard breakpoint wraps every executor."""
        a = RawAgent(_mock_client, name="agent_a")
        b = RawAgent(_mock_client, name="agent_b")
        workflow = WorkflowBuilder(start_executor=a).add_edge(a, b).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(agent, "*")

        for exec_id in workflow.executors:
            assert hasattr(workflow.executors[exec_id], "_bp_original_execute")

        remove_breakpoint_middleware(agent)

    def test_agents_without_tools_can_be_breakpointed(self):
        """Executors with no tools (pure chat agents) can be breakpointed."""
        # This was the original bug: pure chat agents had no tools,
        # so FunctionMiddleware never fired.
        a = RawAgent(_mock_client, name="chat_agent")
        workflow = WorkflowBuilder(start_executor=a).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        inject_breakpoint_middleware(agent, ["chat_agent"])
        assert hasattr(workflow.executors["chat_agent"], "_bp_original_execute")

        remove_breakpoint_middleware(agent)


# ---------------------------------------------------------------------------
# Result creation tests
# ---------------------------------------------------------------------------


class TestBreakpointResult:
    """Verify breakpoint result creation."""

    def test_create_breakpoint_result_with_node_id(self):
        exc = AgentInterruptException(
            interrupt_id="int-1",
            suspend_value={"type": "breakpoint", "node_id": "my_agent"},
            is_breakpoint=True,
        )
        result = create_breakpoint_result(exc)
        assert isinstance(result, UiPathBreakpointResult)
        assert result.breakpoint_node == "my_agent"
        assert result.breakpoint_type == "before"
        assert result.next_nodes == ["my_agent"]

    def test_create_breakpoint_result_empty_suspend_value(self):
        exc = AgentInterruptException(
            interrupt_id="int-2",
            suspend_value="unexpected",
            is_breakpoint=True,
        )
        result = create_breakpoint_result(exc)
        assert result.breakpoint_node == ""
        assert result.next_nodes == []


# ---------------------------------------------------------------------------
# Integration tests: UiPathDebugRuntime ← UiPathAgentFrameworkRuntime
# ---------------------------------------------------------------------------


def _make_agent_runtime(agent: WorkflowAgent) -> UiPathAgentFrameworkRuntime:
    """Create a runtime with mocked chat mapper (no LLM needed)."""
    runtime = UiPathAgentFrameworkRuntime(agent=agent)
    runtime.chat = MagicMock()
    runtime.chat.map_messages_to_input.return_value = "hello"
    runtime.chat.map_streaming_content.return_value = []
    runtime.chat.close_message.return_value = []
    return runtime


class TestDebugRuntimeBreakpointIntegration:
    """Integration tests: UiPathDebugRuntime wrapping our runtime with real workflows.

    These tests verify the full breakpoint flow that the debug UI exercises:
    debug bridge sends breakpoint node IDs → UiPathDebugRuntime passes them
    as options.breakpoints → our runtime wraps executor.execute() → workflow
    runs → wrapped executor raises → our runtime yields UiPathBreakpointResult
    → UiPathDebugRuntime sees it and notifies the bridge.
    """

    async def test_breakpoint_fires_on_start_executor(self):
        """Breakpoint on the start executor pauses before it runs."""
        worker = RawAgent(_mock_client, name="worker")
        workflow = WorkflowBuilder(start_executor=worker).build()
        agent = WorkflowAgent(workflow=workflow, name="test_wf")

        runtime = _make_agent_runtime(agent)

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = ["worker"]
        # Initial resume + quit after breakpoint (don't try to actually resume)
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,
            UiPathDebugQuitError("quit"),
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )

        result = await debug_runtime.execute({"messages": []})

        # Breakpoint should have been hit
        cast(AsyncMock, bridge.emit_breakpoint_hit).assert_awaited_once()
        bp_result = cast(AsyncMock, bridge.emit_breakpoint_hit).call_args[0][0]
        assert isinstance(bp_result, UiPathBreakpointResult)
        assert bp_result.breakpoint_node == "worker"

        # Quit produces a successful result
        assert result.status == UiPathRuntimeStatus.SUCCESSFUL

    async def test_breakpoint_fires_on_toolless_agent(self):
        """Breakpoint works on agents with no tools (the original bug)."""
        # This is the concurrent sample scenario: pure chat agents, no tools
        chat_agent = RawAgent(_mock_client, name="sentiment")
        workflow = WorkflowBuilder(start_executor=chat_agent).build()
        agent = WorkflowAgent(workflow=workflow, name="concurrent_wf")

        runtime = _make_agent_runtime(agent)

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = ["sentiment"]
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,
            UiPathDebugQuitError("quit"),
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )

        result = await debug_runtime.execute({"messages": []})

        cast(AsyncMock, bridge.emit_breakpoint_hit).assert_awaited_once()
        bp_result = cast(AsyncMock, bridge.emit_breakpoint_hit).call_args[0][0]
        assert bp_result.breakpoint_node == "sentiment"

    async def test_state_events_emitted_before_breakpoint(self):
        """Debug bridge should receive state events (STARTED) before the breakpoint."""
        worker = RawAgent(_mock_client, name="agent_x")
        workflow = WorkflowBuilder(start_executor=worker).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        runtime = _make_agent_runtime(agent)

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = ["agent_x"]
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,
            UiPathDebugQuitError("quit"),
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )

        # Collect all events from the stream
        events: list[Any] = []
        async for event in debug_runtime.stream({"messages": []}):
            events.append(event)

        # Should have state events (STARTED for wf and agent_x) before the breakpoint
        state_events = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]
        bp_events = [e for e in events if isinstance(e, UiPathBreakpointResult)]

        assert len(state_events) >= 1, "Should have at least one state event"
        assert len(bp_events) >= 1, "Should have a breakpoint result"

        # The workflow STARTED should come before the breakpoint
        wf_started = [e for e in state_events if e.node_name == "wf"]
        assert len(wf_started) >= 1, "Workflow STARTED event should be emitted"

    async def test_no_breakpoints_runs_to_completion(self):
        """With no breakpoints set, the workflow should run normally (or fail
        trying to call LLM, but not hit any breakpoint)."""
        worker = RawAgent(_mock_client, name="worker")
        workflow = WorkflowBuilder(start_executor=worker).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        runtime = _make_agent_runtime(agent)

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = []  # no breakpoints
        cast(AsyncMock, bridge.wait_for_resume).return_value = None

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )

        # Without breakpoints, the workflow tries to actually execute the agent.
        # Since we have a mock client, this will fail — but NOT as a breakpoint.
        try:
            await debug_runtime.execute({"messages": []})
        except Exception:
            pass  # Expected — no LLM to call

        # No breakpoint should have been hit
        cast(AsyncMock, bridge.emit_breakpoint_hit).assert_not_awaited()

    async def test_breakpoint_resume_preserves_original_input_and_session(self):
        """After breakpoint → continue, the original user input and session
        must be restored so the agent doesn't lose context.

        This was a real bug: on resume, stream() was passing message="" to
        workflow.run() and not loading the session, so the agent acted like
        it never received the user's message.
        """
        worker = RawAgent(_mock_client, name="weather_agent")
        workflow = WorkflowBuilder(start_executor=worker).build()
        agent = WorkflowAgent(workflow=workflow, name="wf")

        # Track what workflow.run() receives on each call
        captured_run_kwargs: list[dict[str, Any]] = []
        original_run = workflow.run

        def tracking_run(**kwargs: Any) -> Any:
            captured_run_kwargs.append(kwargs)
            return original_run(**kwargs)

        workflow.run = tracking_run  # type: ignore[assignment]

        # Mock resumable storage for session + breakpoint state persistence
        kv_store: dict[str, Any] = {}

        mock_storage = AsyncMock()

        async def mock_set_value(
            runtime_id: str, namespace: str, key: str, value: Any
        ) -> None:
            kv_store[f"{runtime_id}:{namespace}:{key}"] = value

        async def mock_get_value(
            runtime_id: str, namespace: str, key: str
        ) -> Any:
            return kv_store.get(f"{runtime_id}:{namespace}:{key}")

        mock_storage.set_value = mock_set_value
        mock_storage.get_value = mock_get_value

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="test-bp-resume",
            resumable_storage=mock_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = (
            "what's the weather in San Francisco"
        )
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = ["weather_agent"]
        # Initial resume → None; after breakpoint → resume (None = continue)
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,  # initial resume
            None,  # continue after breakpoint
            UiPathDebugQuitError("quit"),  # quit after second run fails (no LLM)
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )

        # Execute — first run hits breakpoint, resume continues
        try:
            await debug_runtime.execute({"messages": []})
        except Exception:
            pass  # Expected: no real LLM on the resume run

        # First call: fresh run that hit the breakpoint
        assert len(captured_run_kwargs) >= 1
        first_call = captured_run_kwargs[0]
        assert first_call.get("message") == "what's the weather in San Francisco"
        assert first_call.get("stream") is True

        # Second call: breakpoint resume — must use original input, NOT ""
        assert len(captured_run_kwargs) >= 2
        resume_call = captured_run_kwargs[1]
        assert resume_call.get("message") == "what's the weather in San Francisco"
        assert resume_call.get("stream") is True

        # Breakpoint state should have been persisted with original input
        bp_state = kv_store.get("test-bp-resume:breakpoint:state")
        assert bp_state is not None
        assert bp_state["original_input"] == "what's the weather in San Francisco"
        assert "weather_agent" in bp_state["skip_nodes"]

        # Session should have been saved
        session_data = kv_store.get("test-bp-resume:session:data")
        assert session_data is not None

    async def test_two_sequential_breakpoints_with_resumes(self):
        """Two agents (agent_a → agent_b), both breakpointed. Verifies:
        - Fresh run: BP fires on agent_a, both executors wrapped
        - Resume 1: agent_a skipped, BP fires on agent_b
        - Resume 2: both skipped, completes normally
        """
        agent_a = RawAgent(_mock_client, name="agent_a")
        agent_b = RawAgent(_mock_client, name="agent_b")
        workflow = (
            WorkflowBuilder(start_executor=agent_a)
            .add_edge(agent_a, agent_b)
            .build()
        )
        agent = WorkflowAgent(workflow=workflow, name="wf")

        call_log: list[dict[str, Any]] = []
        call_count = 0
        checkpoint_counter = [0]

        def mock_run(**kwargs: Any) -> _MockWorkflowStream:
            nonlocal call_count
            call_count += 1
            call_log.append({
                "call_number": call_count,
                "kwargs": dict(kwargs),
                "agent_a_wrapped": hasattr(
                    workflow.executors["agent_a"], "_bp_original_execute"
                ),
                "agent_b_wrapped": hasattr(
                    workflow.executors["agent_b"], "_bp_original_execute"
                ),
            })
            if call_count == 1:
                # Fresh run → breakpoint on agent_a
                checkpoint_counter[0] = 1
                return _MockWorkflowStream(
                    exception=AgentInterruptException(
                        interrupt_id="bp-1",
                        suspend_value={
                            "type": "breakpoint",
                            "node_id": "agent_a",
                        },
                        is_breakpoint=True,
                    )
                )
            elif call_count == 2:
                # Resume 1 → breakpoint on agent_b
                checkpoint_counter[0] = 2
                return _MockWorkflowStream(
                    exception=AgentInterruptException(
                        interrupt_id="bp-2",
                        suspend_value={
                            "type": "breakpoint",
                            "node_id": "agent_b",
                        },
                        is_breakpoint=True,
                    )
                )
            else:
                return _MockWorkflowStream(final_output="completed")

        workflow.run = mock_run  # type: ignore[assignment]

        # KV store
        kv_store: dict[str, Any] = {}
        mock_storage = AsyncMock()

        async def mock_set_value(
            runtime_id: str, namespace: str, key: str, value: Any
        ) -> None:
            kv_store[f"{runtime_id}:{namespace}:{key}"] = value

        async def mock_get_value(
            runtime_id: str, namespace: str, key: str
        ) -> Any:
            return kv_store.get(f"{runtime_id}:{namespace}:{key}")

        mock_storage.set_value = mock_set_value
        mock_storage.get_value = mock_get_value

        # Mock checkpoint storage
        mock_cs = AsyncMock()

        async def mock_get_latest(**kwargs: Any) -> Any:
            if checkpoint_counter[0] == 0:
                return None
            cp = MagicMock()
            cp.checkpoint_id = f"cp-{checkpoint_counter[0]}"
            return cp

        mock_cs.get_latest = mock_get_latest

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="test-2bp",
            checkpoint_storage=mock_cs,
            resumable_storage=mock_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "hello world"
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = ["agent_a", "agent_b"]
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,  # initial resume
            None,  # continue after BP on agent_a
            None,  # continue after BP on agent_b
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )
        result = await debug_runtime.execute({"messages": []})

        # --- 3 workflow.run() calls ---
        assert len(call_log) == 3

        # Call 1: Fresh run — both executors wrapped
        c1 = call_log[0]
        assert c1["kwargs"].get("message") == "hello world"
        assert c1["kwargs"].get("stream") is True
        assert c1["agent_a_wrapped"] is True
        assert c1["agent_b_wrapped"] is True

        # Call 2: Resume after agent_a breakpoint — checkpoint-based
        c2 = call_log[1]
        assert c2["kwargs"].get("checkpoint_id") == "cp-1"
        assert c2["kwargs"].get("stream") is True
        assert "message" not in c2["kwargs"]
        assert c2["agent_a_wrapped"] is False  # Skipped
        assert c2["agent_b_wrapped"] is True  # Still breakpointed

        # Call 3: Resume after agent_b breakpoint — checkpoint-based
        c3 = call_log[2]
        assert c3["kwargs"].get("checkpoint_id") == "cp-2"
        assert c3["kwargs"].get("stream") is True
        assert "message" not in c3["kwargs"]
        assert c3["agent_a_wrapped"] is False  # Still skipped (accumulated)
        assert c3["agent_b_wrapped"] is False  # Now also skipped

        # --- Debug bridge interactions ---
        assert cast(AsyncMock, bridge.emit_breakpoint_hit).await_count == 2
        bp_calls = cast(
            AsyncMock, bridge.emit_breakpoint_hit
        ).call_args_list
        assert bp_calls[0].args[0].breakpoint_node == "agent_a"
        assert bp_calls[1].args[0].breakpoint_node == "agent_b"

        assert result.status == UiPathRuntimeStatus.SUCCESSFUL

        # --- KV state ---
        bp_state = kv_store.get("test-2bp:breakpoint:state")
        assert bp_state is not None
        assert set(bp_state["skip_nodes"]) == {"agent_a", "agent_b"}
        assert bp_state["checkpoint_id"] == "cp-2"
        assert bp_state["original_input"] == "hello world"

    async def test_concurrent_breakpoints_accumulate_skip_nodes(self):
        """Concurrent executors: breakpoints fire one-at-a-time, skip_nodes
        accumulate so previously-resumed executors aren't re-breakpointed.

        Graph: dispatcher → [worker_a, worker_b] (fan-out)
        With breakpoints on all nodes:
        - Fresh: BP on dispatcher
        - Resume 1: dispatcher skipped, BP on worker_a
        - Resume 2: dispatcher+worker_a skipped, BP on worker_b
        - Resume 3: all skipped, completes

        This was the infinite loop bug: without accumulating skip_nodes,
        worker_a and worker_b kept trading breakpoints forever.
        """
        dispatcher = RawAgent(_mock_client, name="dispatcher")
        worker_a = RawAgent(_mock_client, name="worker_a")
        worker_b = RawAgent(_mock_client, name="worker_b")
        workflow = (
            WorkflowBuilder(start_executor=dispatcher)
            .add_edge(dispatcher, worker_a)
            .add_edge(dispatcher, worker_b)
            .build()
        )
        agent = WorkflowAgent(workflow=workflow, name="concurrent_wf")

        call_log: list[dict[str, Any]] = []
        call_count = 0
        checkpoint_counter = [0]

        def mock_run(**kwargs: Any) -> _MockWorkflowStream:
            nonlocal call_count
            call_count += 1
            call_log.append({
                "call_number": call_count,
                "kwargs": dict(kwargs),
                "dispatcher_wrapped": hasattr(
                    workflow.executors["dispatcher"], "_bp_original_execute"
                ),
                "worker_a_wrapped": hasattr(
                    workflow.executors["worker_a"], "_bp_original_execute"
                ),
                "worker_b_wrapped": hasattr(
                    workflow.executors["worker_b"], "_bp_original_execute"
                ),
            })
            if call_count == 1:
                checkpoint_counter[0] = 1
                return _MockWorkflowStream(
                    exception=AgentInterruptException(
                        interrupt_id="bp-1",
                        suspend_value={
                            "type": "breakpoint",
                            "node_id": "dispatcher",
                        },
                        is_breakpoint=True,
                    )
                )
            elif call_count == 2:
                # After dispatcher skipped, worker_a fires
                checkpoint_counter[0] = 2
                return _MockWorkflowStream(
                    exception=AgentInterruptException(
                        interrupt_id="bp-2",
                        suspend_value={
                            "type": "breakpoint",
                            "node_id": "worker_a",
                        },
                        is_breakpoint=True,
                    )
                )
            elif call_count == 3:
                # After dispatcher+worker_a skipped, worker_b fires
                checkpoint_counter[0] = 3
                return _MockWorkflowStream(
                    exception=AgentInterruptException(
                        interrupt_id="bp-3",
                        suspend_value={
                            "type": "breakpoint",
                            "node_id": "worker_b",
                        },
                        is_breakpoint=True,
                    )
                )
            else:
                # All skipped, completes
                return _MockWorkflowStream(final_output="done")

        workflow.run = mock_run  # type: ignore[assignment]

        kv_store: dict[str, Any] = {}
        mock_storage = AsyncMock()

        async def mock_set_value(
            runtime_id: str, namespace: str, key: str, value: Any
        ) -> None:
            kv_store[f"{runtime_id}:{namespace}:{key}"] = value

        async def mock_get_value(
            runtime_id: str, namespace: str, key: str
        ) -> Any:
            return kv_store.get(f"{runtime_id}:{namespace}:{key}")

        mock_storage.set_value = mock_set_value
        mock_storage.get_value = mock_get_value

        mock_cs = AsyncMock()

        async def mock_get_latest(**kwargs: Any) -> Any:
            if checkpoint_counter[0] == 0:
                return None
            cp = MagicMock()
            cp.checkpoint_id = f"cp-{checkpoint_counter[0]}"
            return cp

        mock_cs.get_latest = mock_get_latest

        runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id="test-concurrent-bp",
            checkpoint_storage=mock_cs,
            resumable_storage=mock_storage,
        )
        runtime.chat = MagicMock()
        runtime.chat.map_messages_to_input.return_value = "analyze this text"
        runtime.chat.map_streaming_content.return_value = []
        runtime.chat.close_message.return_value = []

        bridge = _make_debug_bridge()
        cast(Mock, bridge.get_breakpoints).return_value = [
            "dispatcher", "worker_a", "worker_b",
        ]
        cast(AsyncMock, bridge.wait_for_resume).side_effect = [
            None,  # initial
            None,  # continue after dispatcher BP
            None,  # continue after worker_a BP
            None,  # continue after worker_b BP
        ]

        debug_runtime = UiPathDebugRuntime(
            delegate=runtime, debug_bridge=bridge
        )
        result = await debug_runtime.execute({"messages": []})

        # --- 4 workflow.run() calls ---
        assert len(call_log) == 4

        # Call 1: Fresh — all wrapped
        c1 = call_log[0]
        assert c1["dispatcher_wrapped"] is True
        assert c1["worker_a_wrapped"] is True
        assert c1["worker_b_wrapped"] is True

        # Call 2: dispatcher skipped, workers wrapped
        c2 = call_log[1]
        assert c2["dispatcher_wrapped"] is False
        assert c2["worker_a_wrapped"] is True
        assert c2["worker_b_wrapped"] is True

        # Call 3: dispatcher+worker_a skipped, worker_b wrapped
        c3 = call_log[2]
        assert c3["dispatcher_wrapped"] is False
        assert c3["worker_a_wrapped"] is False
        assert c3["worker_b_wrapped"] is True

        # Call 4: all skipped — completes
        c4 = call_log[3]
        assert c4["dispatcher_wrapped"] is False
        assert c4["worker_a_wrapped"] is False
        assert c4["worker_b_wrapped"] is False

        # 3 breakpoints hit
        assert cast(AsyncMock, bridge.emit_breakpoint_hit).await_count == 3
        bp_nodes = [
            call.args[0].breakpoint_node
            for call in cast(
                AsyncMock, bridge.emit_breakpoint_hit
            ).call_args_list
        ]
        assert bp_nodes == ["dispatcher", "worker_a", "worker_b"]

        assert result.status == UiPathRuntimeStatus.SUCCESSFUL

        # skip_nodes accumulated all three
        bp_state = kv_store.get("test-concurrent-bp:breakpoint:state")
        assert bp_state is not None
        assert set(bp_state["skip_nodes"]) == {
            "dispatcher", "worker_a", "worker_b",
        }
