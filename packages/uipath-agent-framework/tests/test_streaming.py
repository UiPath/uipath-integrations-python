"""Tests for streaming event pairing (STARTED/COMPLETED).

Uses real Agent Framework agents and workflows with mocked execution
to verify that the runtime emits matched STARTED/COMPLETED state events
for every node in the graph.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework import (
    AgentResponseUpdate,
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


def _update(*contents: Content) -> AgentResponseUpdate:
    return AgentResponseUpdate(contents=list(contents))


def _fc(name: str, call_id: str = "c1") -> Content:
    return Content(type="function_call", name=name, call_id=call_id)


def _fr(name: str = "", call_id: str = "c1", result: Any = "ok") -> Content:
    return Content(type="function_result", name=name, call_id=call_id, result=result)


def _text(text: str = "hi") -> Content:
    return Content(type="text", text=text)


def _wf_event(event_type: str, executor_id: str) -> MagicMock:
    evt = MagicMock()
    evt.type = event_type
    evt.executor_id = executor_id
    evt.data = None
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
# Agent streaming tests
# ===========================================================================


class TestAgentStreamingEvents:
    """Verify STARTED/COMPLETED pairing for agent streaming."""

    async def test_simple_agent_no_tools(self):
        """Agent with no tools: root STARTED then COMPLETED."""
        agent = RawAgent(_mock_client, name="root")
        agent.run = MagicMock(return_value=_MockAsyncStream([_update(_text())]))  # type: ignore[method-assign]
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "root")
        assert isinstance(events[-1], UiPathRuntimeResult)

    async def test_agent_with_regular_tools(self):
        """Agent with regular tools: tools node gets STARTED/COMPLETED."""
        agent = RawAgent(_mock_client, name="researcher", tools=[search_wikipedia])
        agent.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _update(_fc("search_wikipedia", "c1")),
                    _update(_fr("search_wikipedia", "c1", "wiki result")),
                    _update(_text("here's what I found")),
                ]
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "researcher")
        _assert_started_before_completed(se, "researcher_tools")

    async def test_multi_agent_with_sub_agents(self):
        """Coordinator with sub-agents via as_tool(): all nodes paired."""
        research = RawAgent(
            _mock_client,
            name="research_agent",
            tools=[search_wikipedia],
        )
        coder = RawAgent(
            _mock_client,
            name="code_agent",
            tools=[run_python],
        )
        coordinator = RawAgent(
            _mock_client,
            name="coordinator",
            tools=[research.as_tool(), coder.as_tool()],
        )

        # Get actual tool names assigned by as_tool()
        tools = coordinator.default_options.get("tools", [])
        research_tool_name = tools[0].name
        code_tool_name = tools[1].name

        coordinator.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _update(_fc(research_tool_name, "c1")),
                    _update(_fr("", "c1", "research done")),
                    _update(_fc(code_tool_name, "c2")),
                    _update(_fr("", "c2", "code done")),
                    _update(_text("final answer")),
                ]
            )
        )
        coordinator.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(coordinator)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "coordinator")
        _assert_started_before_completed(se, "research_agent")
        _assert_started_before_completed(se, "research_agent_tools")
        _assert_started_before_completed(se, "code_agent")
        _assert_started_before_completed(se, "code_agent_tools")

    async def test_sub_agent_completed_via_call_id(self):
        """Sub-agent COMPLETED even when function_result has empty name.

        The original bug: as_tool() wrappers produce function_result with
        empty content.name. We match by call_id instead.
        """
        inner = RawAgent(_mock_client, name="inner_agent", tools=[calculator])
        outer = RawAgent(
            _mock_client,
            name="outer",
            tools=[inner.as_tool()],
        )

        tool_name = outer.default_options["tools"][0].name

        outer.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _update(_fc(tool_name, "call_xyz")),
                    # empty name on result — must still complete inner_agent
                    _update(_fr("", "call_xyz", "42")),
                    _update(_text("done")),
                ]
            )
        )
        outer.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(outer)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "inner_agent")
        _assert_started_before_completed(se, "inner_agent_tools")

    async def test_mixed_regular_tools_and_sub_agents(self):
        """Agent with both regular tools and agent-as-tool."""
        inner = RawAgent(_mock_client, name="helper")
        agent = RawAgent(
            _mock_client,
            name="main",
            tools=[search_wikipedia, inner.as_tool()],
        )

        agent_tool_name = next(
            t.name for t in agent.default_options["tools"] if hasattr(t, "func")
        )

        agent.run = MagicMock(  # type: ignore[method-assign]
            return_value=_MockAsyncStream(
                [
                    _update(_fc("search_wikipedia", "c1")),
                    _update(_fr("search_wikipedia", "c1", "wiki")),
                    _update(_fc(agent_tool_name, "c2")),
                    _update(_fr("", "c2", "helped")),
                    _update(_text("done")),
                ]
            )
        )
        agent.create_session = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        runtime = _make_runtime(agent)
        events = await _collect_events(runtime)

        se = _state_events(events)
        _assert_all_completed(se)
        _assert_started_before_completed(se, "main")
        _assert_started_before_completed(se, "main_tools")
        _assert_started_before_completed(se, "helper")


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
            with patch.object(factory, "_get_session_store", new_callable=AsyncMock):
                runtimes = await asyncio.gather(
                    factory.new_runtime("agent", "runtime_1"),
                    factory.new_runtime("agent", "runtime_2"),
                    factory.new_runtime("agent", "runtime_3"),
                )

        # Each runtime must have gotten a separate agent instance
        assert len(loaded_agents) == 3
        agents = [r.agent for r in runtimes]  # type: ignore[attr-defined]
        assert len(set(id(a) for a in agents)) == 3, "Runtimes share agent instances!"
