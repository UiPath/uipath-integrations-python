"""Integration tests: sample topologies with breakpoints on all nodes.

Reproduces the infinite loop bug when running samples with breakpoints
enabled on all nodes via UiPathDebugRuntime.

Uses REAL builders, real checkpoint storage, real breakpoint injection —
only LLM calls are mocked.

Covers all sample topologies:
- group-chat (GroupChatBuilder): orchestrator ↔ participants
- quickstart-workflow (WorkflowBuilder): single agent
- concurrent (ConcurrentBuilder): dispatcher → parallel agents → aggregator
- handoff (HandoffBuilder): triage → specialists
- hitl-workflow (HandoffBuilder): triage → specialists with HITL tools
- sequential (SequentialBuilder): researcher → editor chain
"""

import json
import os
import tempfile
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

from agent_framework import WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    ConcurrentBuilder,
    GroupChatBuilder,
    HandoffBuilder,
    SequentialBuilder,
)
from conftest import (
    extract_system_text,
    make_mock_response,
    make_tool_call_response,
)
from uipath.runtime.debug import (
    UiPathDebugProtocol,
    UiPathDebugRuntime,
)
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.events.state import UiPathRuntimeMessageEvent
from uipath.runtime.result import UiPathRuntimeResult, UiPathRuntimeStatus

from uipath_agent_framework.runtime.resumable_storage import (
    ScopedCheckpointStorage,
    SqliteResumableStorage,
)
from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime

# Safety limit: if the debug loop exceeds this many resume calls,
# the test fails — this means breakpoints are stuck in a loop.
MAX_RESUME_CALLS = 50


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


class TestGroupChatBreakpoints:
    """Integration test: GroupChat sample with breakpoints on all nodes.

    Uses the exact same topology as the group-chat sample:
    - 3 participants: researcher, critic, writer
    - 1 orchestrator agent (agent-based)
    - max_rounds=6
    - breakpoints="*" (all nodes)

    Only LLM calls are mocked via a fake AsyncOpenAI client.
    Everything else (GroupChatBuilder, checkpoint storage, breakpoint
    injection, UiPathDebugRuntime) uses real code.
    """

    async def test_group_chat_breakall_completes_without_loop(self):
        """GroupChat with breakpoints on ALL nodes must eventually complete.

        This is the exact scenario that causes an infinite loop in production:
        the debug UI sets breakpoints="*", and the runtime should pause at
        each executor, resume, and eventually finish the workflow.

        The test fails if we exceed MAX_RESUME_CALLS, proving the loop exists.
        """
        # --- LLM mock ---
        # Track which participant the orchestrator selects on each call.
        # The orchestrator cycles: researcher → critic → writer, then repeats.
        participant_names = ["researcher", "critic", "writer"]
        orchestrator_call_count = [0]
        llm_call_log: list[dict[str, str]] = []

        async def mock_chat_completions_create(**kwargs: Any):
            """Mock LLM: orchestrator returns structured JSON, participants return text."""
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if (
                "coordinate" in system_msg.lower()
                or "next speaker" in system_msg.lower()
            ):
                # Orchestrator: select next participant (structured JSON output)
                idx = orchestrator_call_count[0] % len(participant_names)
                orchestrator_call_count[0] += 1
                selected = participant_names[idx]
                response_text = json.dumps(
                    {
                        "terminate": False,
                        "reason": f"Selecting {selected} for the discussion.",
                        "next_speaker": selected,
                        "final_message": None,
                    }
                )
                llm_call_log.append({"agent": "orchestrator", "response": selected})
            elif "research" in system_msg.lower():
                response_text = (
                    "Based on my research, AI safety involves alignment, "
                    "interpretability, and robustness."
                )
                llm_call_log.append(
                    {"agent": "researcher", "response": response_text[:50]}
                )
            elif "critical" in system_msg.lower():
                response_text = (
                    "I challenge the assumption that current approaches "
                    "are sufficient for AI safety."
                )
                llm_call_log.append({"agent": "critic", "response": response_text[:50]})
            elif "writer" in system_msg.lower() or "synthesize" in system_msg.lower():
                response_text = (
                    "In summary, the discussion revealed important nuances "
                    "in AI safety research."
                )
                llm_call_log.append({"agent": "writer", "response": response_text[:50]})
            else:
                response_text = "OK"
                llm_call_log.append({"agent": "unknown", "response": response_text})

            return make_mock_response(response_text, stream=is_stream)

        # --- Build agents exactly like the group-chat sample ---
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_chat_completions_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        researcher = client.as_agent(
            name="researcher",
            description="Expert at finding facts and data using Wikipedia.",
            instructions=(
                "You are a research specialist. Use the search_wikipedia tool "
                "to find factual information. Provide concise, well-sourced "
                "responses."
            ),
        )

        critic = client.as_agent(
            name="critic",
            description="Challenges assumptions and evaluates claims critically.",
            instructions=(
                "You are a critical thinker. Evaluate the claims made by other "
                "participants. Point out gaps, biases, or missing context. "
                "Ask probing questions to deepen the discussion."
            ),
        )

        writer = client.as_agent(
            name="writer",
            description="Synthesizes group discussion into clear, structured prose.",
            instructions=(
                "You are a skilled writer. Synthesize the group discussion into "
                "a clear, well-organized summary. Incorporate the researcher's "
                "facts and address the critic's concerns."
            ),
        )

        orchestrator = client.as_agent(
            name="orchestrator",
            description="Coordinates the group discussion by selecting the next speaker.",
            instructions=(
                "You coordinate a team of researcher, critic, and writer. "
                "Select the next speaker based on the conversation flow:\n"
                "- Pick 'researcher' when facts or data are needed.\n"
                "- Pick 'critic' to challenge or evaluate claims.\n"
                "- Pick 'writer' to synthesize when enough discussion has happened.\n"
                "Respond with ONLY the agent name, nothing else."
            ),
        )

        # --- Build workflow exactly like the sample ---
        workflow = GroupChatBuilder(
            participants=[researcher, critic, writer],
            orchestrator_agent=orchestrator,
            max_rounds=6,
        ).build()

        agent = workflow.as_agent(name="group_chat")

        # --- Real SQLite storage (temp file) ---
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-gc-bp"
            )

            # --- Create runtime ---
            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-gc-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = "Discuss AI safety"
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            # --- Debug bridge: breakpoints on ALL nodes ---
            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"INFINITE LOOP DETECTED: exceeded {MAX_RESUME_CALLS} "
                        f"resume calls.\n"
                        f"LLM calls: {len(llm_call_log)}\n"
                        f"Orchestrator selections: {orchestrator_call_count[0]}\n"
                        f"Call log: {llm_call_log[-10:]}"
                    )
                return None  # Continue (resume after breakpoint)

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            # --- Execute ---
            result = await debug_runtime.execute({"messages": []})

            # --- Assertions ---
            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            # Must complete successfully
            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL but got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}, "
                f"LLM calls: {len(llm_call_log)}"
            )

            # Must have hit at least one breakpoint
            assert bp_count > 0, "Should have hit at least one breakpoint"

            # Must have made LLM calls (agents actually ran)
            assert len(llm_call_log) > 0, "LLM should have been called"

            # Verify breakpoint nodes were reported
            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            assert all(
                n in ("orchestrator", "researcher", "critic", "writer")
                for n in bp_nodes
            ), f"Unexpected breakpoint nodes: {bp_nodes}"

        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_group_chat_single_breakpoint_completes(self):
        """GroupChat with a breakpoint on only the orchestrator completes."""
        orchestrator_call_count = [0]
        participant_names = ["researcher", "critic", "writer"]

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if (
                "coordinate" in system_msg.lower()
                or "next speaker" in system_msg.lower()
            ):
                idx = orchestrator_call_count[0] % len(participant_names)
                orchestrator_call_count[0] += 1
                text = json.dumps(
                    {
                        "terminate": False,
                        "reason": f"Selecting {participant_names[idx]}.",
                        "next_speaker": participant_names[idx],
                        "final_message": None,
                    }
                )
                return make_mock_response(text, stream=is_stream)
            return make_mock_response("Some response text.", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        researcher = client.as_agent(
            name="researcher",
            description="Research expert.",
            instructions="You are a research specialist.",
        )
        critic = client.as_agent(
            name="critic",
            description="Critical thinker.",
            instructions="You are a critical thinker.",
        )
        writer = client.as_agent(
            name="writer",
            description="Writer.",
            instructions="You are a skilled writer.",
        )
        orchestrator = client.as_agent(
            name="orchestrator",
            description="Coordinator.",
            instructions=(
                "You coordinate a team of researcher, critic, and writer. "
                "Select the next speaker. Respond with ONLY the agent name."
            ),
        )

        workflow = GroupChatBuilder(
            participants=[researcher, critic, writer],
            orchestrator_agent=orchestrator,
            max_rounds=3,
        ).build()
        agent = workflow.as_agent(name="group_chat")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-gc-single-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-gc-single-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = "Discuss AI"
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(f"Loop detected: {resume_count[0]} resumes")
                return None

            bridge = _make_debug_bridge()
            # Breakpoint ONLY on orchestrator
            cast(Mock, bridge.get_breakpoints).return_value = ["orchestrator"]
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}"
            )
            assert bp_count > 0

            # All BPs should be on orchestrator
            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            assert all(n == "orchestrator" for n in bp_nodes)

        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_group_chat_no_breakpoints_completes(self):
        """GroupChat without breakpoints runs to completion (baseline)."""
        orchestrator_call_count = [0]
        participant_names = ["researcher", "critic", "writer"]

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if (
                "coordinate" in system_msg.lower()
                or "next speaker" in system_msg.lower()
            ):
                idx = orchestrator_call_count[0] % len(participant_names)
                orchestrator_call_count[0] += 1
                text = json.dumps(
                    {
                        "terminate": False,
                        "reason": f"Selecting {participant_names[idx]}.",
                        "next_speaker": participant_names[idx],
                        "final_message": None,
                    }
                )
                return make_mock_response(text, stream=is_stream)
            return make_mock_response("Some response text.", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        researcher = client.as_agent(
            name="researcher",
            description="Research expert.",
            instructions="You are a research specialist.",
        )
        critic = client.as_agent(
            name="critic",
            description="Critical thinker.",
            instructions="You are a critical thinker.",
        )
        writer = client.as_agent(
            name="writer",
            description="Writer.",
            instructions="You are a skilled writer.",
        )
        orchestrator = client.as_agent(
            name="orchestrator",
            description="Coordinator.",
            instructions=(
                "You coordinate a team of researcher, critic, and writer. "
                "Select the next speaker. Respond with ONLY the agent name."
            ),
        )

        workflow = GroupChatBuilder(
            participants=[researcher, critic, writer],
            orchestrator_agent=orchestrator,
            max_rounds=3,
        ).build()
        agent = workflow.as_agent(name="group_chat")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-gc-no-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-gc-no-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = "Discuss AI"
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            bridge = _make_debug_bridge()
            # No breakpoints
            cast(Mock, bridge.get_breakpoints).return_value = []
            cast(AsyncMock, bridge.wait_for_resume).return_value = None

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL
            cast(AsyncMock, bridge.emit_breakpoint_hit).assert_not_awaited()

        finally:
            await storage.dispose()
            os.unlink(tmp_path)


class TestQuickstartWorkflowBreakpoints:
    """Integration test: quickstart-workflow sample with breakpoints.

    Topology: single weather_agent in a WorkflowBuilder.
    Breakpoints="*" → breaks on the single agent, then completes.
    """

    async def test_quickstart_breakall_completes_without_loop(self):
        """Single-agent workflow with breakpoints="*" completes."""
        llm_call_log: list[str] = []

        async def mock_create(**kwargs: Any):
            is_stream = kwargs.get("stream", False)
            llm_call_log.append("weather_agent")
            return make_mock_response(
                "The weather in New York is 72°F and sunny.",
                stream=is_stream,
            )

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        weather_agent = client.as_agent(
            name="weather_agent",
            description="Provides weather information.",
            instructions="You are a weather assistant.",
        )

        workflow = WorkflowBuilder(start_executor=weather_agent).build()
        agent = workflow.as_agent(name="weather_assistant")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-qs-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-qs-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = (
                "What is the weather in New York?"
            )
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(f"Loop detected: {resume_count[0]} resumes")
                return None

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}"
            )

            assert bp_count >= 1, "Should have hit at least one breakpoint"
            assert len(llm_call_log) > 0, "LLM should have been called"

            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            assert "weather_agent" in bp_nodes

        finally:
            await storage.dispose()
            os.unlink(tmp_path)


class TestConcurrentBreakpoints:
    """Integration test: concurrent sample with breakpoints.

    Topology: dispatcher → [sentiment, topic, summarizer] → aggregator
    (ConcurrentBuilder fan-out / fan-in).
    Breakpoints="*" → breaks on each executor, then completes.
    """

    async def test_concurrent_breakall_completes_without_loop(self):
        """Concurrent workflow with breakpoints="*" completes."""
        llm_call_log: list[str] = []

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "sentiment" in system_msg.lower():
                llm_call_log.append("sentiment")
                return make_mock_response(
                    "Sentiment: positive (0.85)", stream=is_stream
                )
            elif "topic" in system_msg.lower() or "entit" in system_msg.lower():
                llm_call_log.append("topic")
                return make_mock_response(
                    "Topics: AI, safety, alignment", stream=is_stream
                )
            elif "summar" in system_msg.lower():
                llm_call_log.append("summarizer")
                return make_mock_response(
                    "Summary: A discussion about AI safety.",
                    stream=is_stream,
                )
            else:
                llm_call_log.append("unknown")
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        sentiment_agent = client.as_agent(
            name="sentiment",
            description="Analyzes text sentiment.",
            instructions="You analyze sentiment of the given text.",
        )
        topic_agent = client.as_agent(
            name="topic",
            description="Extracts topics and entities.",
            instructions="You extract topics and entities from text.",
        )
        summarizer = client.as_agent(
            name="summarizer",
            description="Creates concise summaries.",
            instructions="You summarize the given text concisely.",
        )

        workflow = ConcurrentBuilder(
            participants=[sentiment_agent, topic_agent, summarizer],
        ).build()
        agent = workflow.as_agent(name="multi_analyzer")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-conc-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-conc-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = (
                "Analyze this text about AI safety"
            )
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(f"Loop detected: {resume_count[0]} resumes")
                return None

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}"
            )

            # At least the 3 agents + dispatcher + aggregator
            assert bp_count >= 3, f"Expected at least 3 breakpoints, got {bp_count}"

            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            # All 3 agent executors should appear in breakpoint nodes
            for agent_name in ("sentiment", "topic", "summarizer"):
                assert agent_name in bp_nodes, (
                    f"{agent_name} not in breakpoint nodes: {bp_nodes}"
                )

        finally:
            await storage.dispose()
            os.unlink(tmp_path)


class TestHandoffBreakpoints:
    """Integration test: handoff sample with breakpoints.

    Topology: triage → [billing_agent, tech_agent, returns_agent]
    (HandoffBuilder with handoff tools).
    Breakpoints="*" → breaks on triage, then on the specialist it
    hands off to, then completes.
    """

    async def test_handoff_breakall_completes_without_loop(self):
        """Handoff workflow with breakpoints="*" completes."""
        llm_call_log: list[str] = []

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "route" in system_msg.lower() or "triage" in system_msg.lower():
                llm_call_log.append("triage")
                # Triage hands off to billing_agent via tool call
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            elif "billing" in system_msg.lower():
                llm_call_log.append("billing_agent")
                return make_mock_response(
                    "I've resolved your billing issue. Your account "
                    "has been credited $50.",
                    stream=is_stream,
                )
            elif "tech" in system_msg.lower():
                llm_call_log.append("tech_agent")
                return make_mock_response(
                    "I can help with your technical issue.",
                    stream=is_stream,
                )
            elif "return" in system_msg.lower() or "refund" in system_msg.lower():
                llm_call_log.append("returns_agent")
                return make_mock_response(
                    "I'll process your return right away.",
                    stream=is_stream,
                )
            else:
                llm_call_log.append("unknown")
                return make_mock_response("How can I help you?", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        triage = client.as_agent(
            name="triage",
            description="Routes customers to the right specialist.",
            instructions=(
                "You are a triage agent. Route the customer to the "
                "appropriate specialist based on their issue."
            ),
        )
        billing_agent = client.as_agent(
            name="billing_agent",
            description="Handles billing and payment issues.",
            instructions=(
                "You are a billing specialist. Help customers with "
                "billing inquiries and payment issues."
            ),
        )
        tech_agent = client.as_agent(
            name="tech_agent",
            description="Handles technical support.",
            instructions=(
                "You are a tech support specialist. Help customers "
                "with technical issues."
            ),
        )
        returns_agent = client.as_agent(
            name="returns_agent",
            description="Handles returns and refunds.",
            instructions=(
                "You are a returns specialist. Help customers with "
                "returns and refund requests."
            ),
        )

        workflow = (
            HandoffBuilder(
                name="customer_support",
                participants=[triage, billing_agent, tech_agent, returns_agent],
            )
            .with_start_agent(triage)
            .add_handoff(triage, [billing_agent, tech_agent, returns_agent])
            .add_handoff(billing_agent, [triage])
            .add_handoff(tech_agent, [triage])
            .add_handoff(returns_agent, [triage])
            .build()
        )
        agent = workflow.as_agent(name="customer_support")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-handoff-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-handoff-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = "I have a billing issue"
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Loop detected: {resume_count[0]} resumes. "
                        f"LLM calls: {llm_call_log}"
                    )
                return None

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}, "
                f"LLM calls: {llm_call_log}"
            )

            # At least triage and billing_agent should have been breakpointed
            assert bp_count >= 2, f"Expected at least 2 breakpoints, got {bp_count}"

            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            assert "triage" in bp_nodes, f"triage not in breakpoint nodes: {bp_nodes}"
            assert "billing_agent" in bp_nodes, (
                f"billing_agent not in breakpoint nodes: {bp_nodes}"
            )

        finally:
            await storage.dispose()
            os.unlink(tmp_path)


class TestHitlWorkflowBreakpoints:
    """Integration test: hitl-workflow sample with breakpoints.

    Topology: triage → [billing_agent, returns_agent]
    (HandoffBuilder, same as handoff but specialists have HITL tools).
    Breakpoints="*" → breaks on each executor independently of HITL.
    """

    async def test_hitl_breakall_completes_without_loop(self):
        """HITL handoff workflow with breakpoints="*" completes.

        The HITL tools (@requires_approval) are present on specialist agents
        but the mock LLM does not trigger them.  This test verifies that
        breakpoints work correctly on the HITL topology.
        """
        llm_call_log: list[str] = []

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "route" in system_msg.lower() or "triage" in system_msg.lower():
                llm_call_log.append("triage")
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            elif "billing" in system_msg.lower():
                llm_call_log.append("billing_agent")
                return make_mock_response(
                    "Your billing issue has been resolved.",
                    stream=is_stream,
                )
            elif "return" in system_msg.lower() or "refund" in system_msg.lower():
                llm_call_log.append("returns_agent")
                return make_mock_response(
                    "Your refund has been processed.",
                    stream=is_stream,
                )
            else:
                llm_call_log.append("unknown")
                return make_mock_response("How can I help you?", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

        triage = client.as_agent(
            name="triage",
            description="Routes customers to the right specialist.",
            instructions=(
                "You are a triage agent. Route the customer to the "
                "appropriate specialist."
            ),
        )
        billing_agent = client.as_agent(
            name="billing_agent",
            description="Handles billing with approval-required tools.",
            instructions=(
                "You are a billing specialist. Use transfer_funds "
                "when needed (requires approval)."
            ),
        )
        returns_agent = client.as_agent(
            name="returns_agent",
            description="Handles returns with approval-required tools.",
            instructions=(
                "You are a returns specialist. Use issue_refund "
                "when needed (requires approval)."
            ),
        )

        workflow = (
            HandoffBuilder(
                name="hitl_support",
                participants=[triage, billing_agent, returns_agent],
            )
            .with_start_agent(triage)
            .add_handoff(triage, [billing_agent, returns_agent])
            .add_handoff(billing_agent, [triage])
            .add_handoff(returns_agent, [triage])
            .build()
        )
        agent = workflow.as_agent(name="hitl_support")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-hitl-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-hitl-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            runtime.chat = MagicMock()
            runtime.chat.map_messages_to_input.return_value = "I need to transfer funds"
            runtime.chat.map_streaming_content.return_value = []
            runtime.chat.close_message.return_value = []

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Loop detected: {resume_count[0]} resumes. "
                        f"LLM calls: {llm_call_log}"
                    )
                return None

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, BPs: {bp_count}, "
                f"LLM calls: {llm_call_log}"
            )

            assert bp_count >= 2, f"Expected at least 2 breakpoints, got {bp_count}"

            bp_nodes = [
                call.args[0].breakpoint_node
                for call in cast(AsyncMock, bridge.emit_breakpoint_hit).call_args_list
            ]
            assert "triage" in bp_nodes, (
                f"triage not in breakpoint nodes (hitl): {bp_nodes}"
            )
            assert "billing_agent" in bp_nodes, (
                f"billing_agent not in breakpoint nodes (hitl): {bp_nodes}"
            )

        finally:
            await storage.dispose()
            os.unlink(tmp_path)


class TestSequentialBreakpoints:
    """Integration test: sequential workflow with breakpoints on all nodes.

    Topology: input-conversation → researcher → editor → end
    (SequentialBuilder chain).
    Breakpoints="*" → breaks on each executor, then completes.

    Verifies that chat message events are NOT duplicated across breakpoint
    resume cycles.  This reproduces the bug where the "end" executor's
    output event re-emitted the full conversation because the per-executor
    message tracking was reset on each resume.
    """

    async def test_sequential_breakall_no_duplicate_messages(self):
        """Sequential workflow with breakpoints="*" must not duplicate messages.

        Each agent's text should appear exactly once in the streamed message
        events.  Before the fix, the "end" executor's output event contained
        the full conversation and was emitted because executors_with_messages
        was a local variable reset on each resume cycle.
        """
        researcher_text = "Tokyo is the capital of Japan."
        editor_text = '{"city": "Tokyo", "country": "Japan"}'

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "researcher" in system_msg.lower():
                return make_mock_response(researcher_text, stream=is_stream)
            elif "editor" in system_msg.lower():
                return make_mock_response(editor_text, stream=is_stream)
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)
        researcher = client.as_agent(
            name="researcher",
            description="Researches factual information about a city.",
            instructions="You are a researcher.",
        )
        editor = client.as_agent(
            name="editor",
            description="Edits research into a structured profile.",
            instructions="You are an editor.",
        )

        workflow = SequentialBuilder(
            participants=[researcher, editor],
        ).build()
        agent = workflow.as_agent(name="sequential_test")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            scoped_cs = ScopedCheckpointStorage(
                storage.checkpoint_storage, "test-seq-bp"
            )

            runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id="test-seq-bp",
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            # Use the REAL chat mapper so message events are generated.
            # Only mock map_messages_to_input to provide the user prompt.
            # Use object.__setattr__ to avoid mypy [method-assign] error.
            object.__setattr__(
                runtime.chat,
                "map_messages_to_input",
                MagicMock(return_value="Tell me about Tokyo"),
            )

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(f"Loop detected: {resume_count[0]} resumes")
                return None

            bridge = _make_debug_bridge()
            cast(Mock, bridge.get_breakpoints).return_value = "*"
            cast(AsyncMock, bridge.wait_for_resume).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(delegate=runtime, debug_bridge=bridge)

            # ---- Stream and collect all events ----
            all_events: list[UiPathRuntimeEvent] = []
            async for event in debug_runtime.stream({"messages": []}):
                all_events.append(event)

            # ---- Verify execution completed successfully ----
            results = [e for e in all_events if isinstance(e, UiPathRuntimeResult)]
            assert len(results) >= 1, "No UiPathRuntimeResult events found"
            final_result = results[-1]
            assert final_result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {final_result.status}. "
                f"Resumes: {resume_count[0]}"
            )

            # ---- Verify breakpoints fired ----
            bp_count = cast(AsyncMock, bridge.emit_breakpoint_hit).await_count
            assert bp_count >= 2, (
                f"Expected at least 2 breakpoints (researcher + editor), got {bp_count}"
            )

            # ---- Collect all text chunks from message events ----
            text_chunks: list[str] = []
            for event in all_events:
                if not isinstance(event, UiPathRuntimeMessageEvent):
                    continue
                payload = event.payload
                cp = getattr(payload, "content_part", None)
                if cp is None:
                    continue
                chunk = getattr(cp, "chunk", None)
                if chunk is None:
                    continue
                data = getattr(chunk, "data", None)
                if data:
                    text_chunks.append(data)

            all_text = "".join(text_chunks)

            # ---- Assert no duplicate messages ----
            researcher_count = all_text.count(researcher_text)
            editor_count = all_text.count(editor_text)

            assert researcher_count == 1, (
                f"Researcher text should appear exactly once, "
                f"but appeared {researcher_count} times. "
                f"This means breakpoint resume duplicated messages."
            )
            assert editor_count == 1, (
                f"Editor text should appear exactly once, "
                f"but appeared {editor_count} times. "
                f"This means breakpoint resume duplicated messages."
            )

        finally:
            await storage.dispose()
            os.unlink(tmp_path)
