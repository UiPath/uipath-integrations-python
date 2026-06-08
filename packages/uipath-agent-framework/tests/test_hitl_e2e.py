"""End-to-end tests for the HITL tool approval flow.

Tests the full runtime stack:
  UiPathChatRuntime -> UiPathResumableRuntime -> UiPathAgentFrameworkRuntime

Only LLM calls are mocked. The chat bridge is a mock implementing UiPathChatProtocol.
All other components (builders, checkpoint storage, trigger management) use real code.

Covers:
- Tool approval (approved) completes successfully
- Tool approval (rejected) completes successfully
- Multiple tools across agents
- Streaming tool approval
- Interrupt payload format validation
"""

import asyncio
import json
import os
import tempfile
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder
from conftest import (
    extract_system_text,
    make_mock_response,
    make_tool_call_response,
)
from uipath.core.chat import UiPathConversationToolCallConfirmationValue
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import UiPathResumableRuntime
from uipath.runtime.chat.runtime import UiPathChatRuntime
from uipath.runtime.debug import (
    UiPathDebugProtocol,
    UiPathDebugRuntime,
)
from uipath.runtime.events import (
    UiPathRuntimeEvent,
    UiPathRuntimeStateEvent,
    UiPathRuntimeStatePhase,
)
from uipath.runtime.events.state import UiPathRuntimeMessageEvent
from uipath.runtime.result import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.resumable.trigger import (
    UiPathResumeTrigger,
    UiPathResumeTriggerType,
)

from uipath_agent_framework.chat.tools import requires_approval
from uipath_agent_framework.runtime.resumable_storage import (
    ScopedCheckpointStorage,
    SqliteResumableStorage,
)
from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime

# ---------------------------------------------------------------------------
# Mock chat bridge
# ---------------------------------------------------------------------------


class MockChatBridge:
    """Mock UiPathChatProtocol for testing HITL flows.

    Captures emitted interrupt events and auto-responds with approval/rejection.
    """

    def __init__(self, auto_approve: bool = True):
        self.auto_approve = auto_approve
        self.interrupts: list[UiPathResumeTrigger] = []
        self.messages: list[Any] = []
        self.executing_tool_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def emit_message_event(self, message_event: Any) -> None:
        self.messages.append(message_event)

    async def emit_interrupt_event(self, resume_trigger: UiPathResumeTrigger) -> None:
        self.interrupts.append(resume_trigger)

    async def emit_executing_tool_call_event(
        self,
        tool_call_id: str,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        self.executing_tool_calls.append((tool_call_id, tool_input))

    async def emit_exchange_end_event(self) -> None:
        pass

    async def emit_exchange_error_event(self, error: Exception) -> None:
        pass

    async def wait_for_resume(self) -> dict[str, Any]:
        """Return CAS-format approval/rejection response."""
        return {
            "type": "uipath_cas_tool_call_confirmation",
            "value": {"approved": self.auto_approve},
        }


# ---------------------------------------------------------------------------
# Runtime stack helper
# ---------------------------------------------------------------------------


async def _create_hitl_runtime_stack(
    agent: Any,
    runtime_id: str,
    tmp_path: str,
    auto_approve: bool = True,
) -> tuple[UiPathChatRuntime, MockChatBridge, SqliteResumableStorage]:
    """Create the full runtime stack: ChatRuntime -> ResumableRuntime -> AgentFrameworkRuntime."""
    storage = SqliteResumableStorage(tmp_path)
    await storage.setup()
    assert storage.checkpoint_storage is not None

    scoped_cs = ScopedCheckpointStorage(storage.checkpoint_storage, runtime_id)

    base_runtime = UiPathAgentFrameworkRuntime(
        agent=agent,
        runtime_id=runtime_id,
        checkpoint_storage=scoped_cs,
        resumable_storage=storage,
    )
    base_runtime.chat = MagicMock()
    base_runtime.chat.map_messages_to_input.return_value = "I need help with billing"
    base_runtime.chat.map_streaming_content.return_value = []
    base_runtime.chat.close_message.return_value = []

    resumable_runtime = UiPathResumableRuntime(
        delegate=base_runtime,
        storage=storage,
        trigger_manager=UiPathResumeTriggerHandler(),
        runtime_id=runtime_id,
    )

    chat_bridge = MockChatBridge(auto_approve=auto_approve)
    chat_runtime = UiPathChatRuntime(
        delegate=resumable_runtime, chat_bridge=chat_bridge
    )

    return chat_runtime, chat_bridge, storage


# ---------------------------------------------------------------------------
# Agent builder helper
# ---------------------------------------------------------------------------


def _build_hitl_agents(mock_openai: AsyncMock) -> Any:
    """Build the HITL handoff workflow with real @requires_approval tools.

    Tool functions are defined inside this function so each call creates
    fresh FunctionTool instances, avoiding cross-test state pollution.
    """

    @requires_approval
    def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
        """Transfer funds between accounts. Requires human approval."""
        return f"Transferred ${amount:.2f} from {from_account} to {to_account}"

    @requires_approval
    def get_customer_order(order_id: str) -> str:
        """Retrieve customer order details. Requires human approval."""
        return f"Order {order_id}: Widget x2, $49.99"

    @requires_approval
    def issue_refund(order_id: str, amount: float, reason: str) -> str:
        """Issue a refund for an order. Requires human approval."""
        return f"Refund of ${amount:.2f} issued for order {order_id}: {reason}"

    client = OpenAIChatClient(model_id="mock-model", async_client=mock_openai)

    # IMPORTANT: Agent instructions must use unique keywords for LLM mock routing.
    # The triage instructions must NOT contain keywords used by other agents
    # (e.g., "billing", "order", "returns") or the mock will misroute calls.
    triage = client.as_agent(
        name="triage",
        description="Routes customer requests to the right specialist.",
        instructions=(
            "You are a triage agent. Determine the customer issue and "
            "hand off to the appropriate specialist using handoff tools."
        ),
    )

    billing = client.as_agent(
        name="billing_agent",
        description="Handles billing and fund transfers.",
        instructions=("You are a billing specialist. Use transfer_funds when needed."),
        tools=[transfer_funds],
    )

    orders = client.as_agent(
        name="orders_agent",
        description="Looks up customer orders.",
        instructions=(
            "You are an order lookup specialist. Use get_customer_order to look up orders."
        ),
        tools=[get_customer_order],
    )

    returns = client.as_agent(
        name="returns_agent",
        description="Handles returns and refunds.",
        instructions=("You are a refund specialist. Use issue_refund when needed."),
        tools=[issue_refund],
    )

    workflow = (
        HandoffBuilder(
            name="customer_support",
            participants=[triage, billing, orders, returns],
        )
        .with_start_agent(triage)
        .add_handoff(triage, [billing, orders, returns])
        .add_handoff(billing, [triage])
        .add_handoff(orders, [triage])
        .add_handoff(returns, [triage])
        .build()
    )

    return workflow.as_agent(name="customer_support")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="class")
class TestHitlToolApprovalE2E:
    """End-to-end tests for HITL tool approval flow through the full runtime stack.

    Uses class-scoped event loop to avoid cross-test interference from
    the agent_framework's internal async checkpoint operations.
    """

    @pytest.fixture(autouse=True)
    async def _settle_framework(self):
        """Allow framework background tasks to complete between tests."""
        yield
        await asyncio.sleep(0.2)

    async def test_tool_approval_approved_completes(self):
        """Tool approval with auto_approve=True should complete successfully."""
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-approve", tmp_path, auto_approve=True
            )

            result = await chat_runtime.execute({"messages": []})

            # Interrupt was emitted
            assert len(chat_bridge.interrupts) == 1, (
                f"Expected 1 interrupt, got {len(chat_bridge.interrupts)}"
            )

            # Trigger has correct type and API resume data
            trigger = chat_bridge.interrupts[0]
            assert trigger.trigger_type == UiPathResumeTriggerType.API
            assert trigger.api_resume is not None
            request = trigger.api_resume.request
            assert isinstance(request, dict)
            assert request["toolName"] == "transfer_funds"
            assert request["inputValue"]["from_account"] == "A"
            assert request["inputValue"]["amount"] == 100.0

            # Completed successfully (tool was executed after approval)
            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_tool_approval_rejected_completes(self):
        """Tool approval with auto_approve=False should complete (tool not executed)."""
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 50.0}',
                        stream=is_stream,
                    )
                else:
                    # After rejection, agent responds with text
                    return make_mock_response(
                        "The transfer was not approved.", stream=is_stream
                    )
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-reject", tmp_path, auto_approve=False
            )

            result = await chat_runtime.execute({"messages": []})

            # Interrupt was emitted
            assert len(chat_bridge.interrupts) == 1

            # Trigger type is API
            assert chat_bridge.interrupts[0].trigger_type == UiPathResumeTriggerType.API

            # Completed (framework handles rejection gracefully)
            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_streaming_tool_approval(self):
        """Streaming HITL flow should emit events and complete successfully."""
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "X", "to_account": "Y", "amount": 200.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer done.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-stream", tmp_path, auto_approve=True
            )

            events: list[UiPathRuntimeEvent] = []
            async for event in chat_runtime.stream({"messages": []}):
                events.append(event)

            # Should have collected some events
            assert len(events) > 0, "Should have emitted at least one event"

            # Interrupt was captured by bridge
            assert len(chat_bridge.interrupts) >= 1
            assert chat_bridge.interrupts[0].trigger_type == UiPathResumeTriggerType.API

            # Last event should be a successful result
            results = [e for e in events if isinstance(e, UiPathRuntimeResult)]
            assert len(results) >= 1
            assert results[-1].status == UiPathRuntimeStatus.SUCCESSFUL
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_interrupt_payload_format(self):
        """Verify the exact shape of the interrupt trigger payload."""
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "ACC1", "to_account": "ACC2", "amount": 99.99}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Done.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-format", tmp_path, auto_approve=True
            )

            await chat_runtime.execute({"messages": []})

            assert len(chat_bridge.interrupts) >= 1
            trigger = chat_bridge.interrupts[0]

            # Verify trigger structure uses correct enum type
            assert trigger.trigger_type == UiPathResumeTriggerType.API
            assert trigger.api_resume is not None, "Should have api_resume"
            request = trigger.api_resume.request
            assert isinstance(request, dict), (
                f"request should be dict, got {type(request)}"
            )

            # Verify expected keys (camelCase from Pydantic alias serialization)
            assert "toolCallId" in request, f"Missing toolCallId in {request.keys()}"
            assert "toolName" in request, f"Missing toolName in {request.keys()}"
            assert "inputSchema" in request, f"Missing inputSchema in {request.keys()}"
            assert "inputValue" in request, f"Missing inputValue in {request.keys()}"

            # Verify values
            assert request["toolName"] == "transfer_funds"
            assert request["inputValue"]["from_account"] == "ACC1"
            assert request["inputValue"]["to_account"] == "ACC2"
            assert request["inputValue"]["amount"] == 99.99

            # Verify the request can be reconstructed as UiPathConversationToolCallConfirmationValue
            confirmation = UiPathConversationToolCallConfirmationValue(**request)
            assert confirmation.tool_name == "transfer_funds"
            assert confirmation.input_value is not None
            assert confirmation.input_value["amount"] == 99.99
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_multiple_tools_across_agents(self):
        """Multiple tools across different agents should each trigger an interrupt.

        Flow: triage -> orders -> get_customer_order (HITL) -> orders hands off
        to triage -> triage -> billing -> transfer_funds (HITL) -> billing responds.
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "order" in system_msg.lower():
                count = call_count.get("orders", 0)
                call_count["orders"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "get_customer_order",
                        arguments='{"order_id": "ORD-123"}',
                        stream=is_stream,
                    )
                else:
                    # After tool approval, hand back to triage
                    return make_tool_call_response(
                        "handoff_to_triage", stream=is_stream
                    )
            elif "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 50.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("All done.", stream=is_stream)
            elif "triage" in system_msg.lower():
                count = call_count.get("triage", 0)
                call_count["triage"] = count + 1
                if count == 0:
                    # First triage call: route to orders
                    return make_tool_call_response(
                        "handoff_to_orders_agent", stream=is_stream
                    )
                else:
                    # Second triage call (after orders hands back): route to billing
                    return make_tool_call_response(
                        "handoff_to_billing_agent", stream=is_stream
                    )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-multi", tmp_path, auto_approve=True
            )

            result = await chat_runtime.execute({"messages": []})

            # Two interrupts: one for get_customer_order, one for transfer_funds
            assert len(chat_bridge.interrupts) == 2, (
                f"Expected 2 interrupts, got {len(chat_bridge.interrupts)}. "
                f"Tool names: {[t.api_resume.request.get('toolName') if t.api_resume else None for t in chat_bridge.interrupts]}"
            )

            # Both triggers should be API type
            for trigger in chat_bridge.interrupts:
                assert trigger.trigger_type == UiPathResumeTriggerType.API

            tool_names = [
                t.api_resume.request["toolName"]
                for t in chat_bridge.interrupts
                if t.api_resume
            ]
            assert "get_customer_order" in tool_names
            assert "transfer_funds" in tool_names

            # Final result should be successful
            assert result.status == UiPathRuntimeStatus.SUCCESSFUL
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_multi_turn_after_hitl_resume(self):
        """Second fresh turn after HITL resume should not fail with stale session.

        Reproduces the bug where the session saved after HITL resume was stale
        (missing tool results), causing OpenAI to reject the next turn with:
        "An assistant message with 'tool_calls' must be followed by tool messages"

        Flow:
          Turn 1: triage -> billing -> transfer_funds (HITL approve) -> complete
          Turn 2: triage -> text response -> complete (must not fail)
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            elif "triage" in system_msg.lower():
                count = call_count.get("triage", 0)
                call_count["triage"] = count + 1
                if count == 0:
                    # Turn 1: route to billing
                    return make_tool_call_response(
                        "handoff_to_billing_agent", stream=is_stream
                    )
                else:
                    # Turn 2: respond directly (no tool call)
                    return make_mock_response("How else can I help?", stream=is_stream)
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-multi-turn", tmp_path, auto_approve=True
            )

            # Turn 1: HITL approval flow
            result1 = await chat_runtime.execute({"messages": []})
            assert result1.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 1 failed: {result1.status}"
            )

            # Verify session was saved with complete history (including tool results).
            # The bug was that only the HITL-suspension session was persisted
            # (missing tool results), causing the next turn to fail.
            session_data = await storage.get_value(
                "test-hitl-multi-turn", "session", "data"
            )
            assert session_data is not None, "Session was not saved after HITL resume"

            session_state = session_data.get("state", {})
            # Find all messages in any provider key
            all_messages = []
            for _provider_key, provider_data in session_state.items():
                if not isinstance(provider_data, dict):
                    continue
                msgs = provider_data.get("messages", [])
                if not isinstance(msgs, list):
                    continue
                all_messages = msgs
                break

            assert len(all_messages) > 0, (
                f"No messages found in session state. "
                f"State keys: {list(session_state.keys())}. "
                f"Full state: {json.dumps(session_data, indent=2, default=str)[:3000]}"
            )

            # Verify: every assistant message with a function_call must
            # eventually be followed by a tool message with function_result
            # for the same call_id. This is the exact invariant that OpenAI
            # enforces and that breaks with stale sessions.
            def get_function_call_ids(msg_data: dict[str, Any]) -> list[str]:
                """Get call_ids of function_call contents in a message."""
                ids = []
                for c in msg_data.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_call":
                        cid = c.get("call_id")
                        if cid:
                            ids.append(cid)
                return ids

            def get_function_result_ids(msg_data: dict[str, Any]) -> list[str]:
                """Get call_ids of function_result contents in a message."""
                ids = []
                for c in msg_data.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_result":
                        cid = c.get("call_id")
                        if cid:
                            ids.append(cid)
                return ids

            pending_call_ids: set[str] = set()
            for msg in all_messages:
                if not isinstance(msg, dict):
                    continue
                pending_call_ids.update(get_function_call_ids(msg))
                for rid in get_function_result_ids(msg):
                    pending_call_ids.discard(rid)

            assert len(pending_call_ids) == 0, (
                f"Session has orphaned function_calls (no function_result): "
                f"{pending_call_ids}. This will cause OpenAI to reject the next "
                f"turn. Messages:\n"
                + json.dumps(all_messages, indent=2, default=str)[:5000]
            )

            # Turn 2: fresh turn after HITL — must not fail with stale session
            chat_bridge.interrupts.clear()
            result2 = await chat_runtime.execute({"messages": []})
            assert result2.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 2 failed: {result2.status}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_multi_turn_after_hitl_resume_streaming(self):
        """Streaming: second fresh turn after HITL resume should not fail.

        Same as test_multi_turn_after_hitl_resume but using the streaming path.
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "X", "to_account": "Y", "amount": 200.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer done.", stream=is_stream)
            elif "triage" in system_msg.lower():
                count = call_count.get("triage", 0)
                call_count["triage"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "handoff_to_billing_agent", stream=is_stream
                    )
                else:
                    return make_mock_response("Anything else?", stream=is_stream)
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-stream-multi", tmp_path, auto_approve=True
            )

            # Turn 1: streaming HITL flow
            events1: list[UiPathRuntimeEvent] = []
            async for event in chat_runtime.stream({"messages": []}):
                events1.append(event)
            results1 = [e for e in events1 if isinstance(e, UiPathRuntimeResult)]
            assert len(results1) >= 1
            assert results1[-1].status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 1 failed: {results1[-1].status}"
            )

            # Turn 2: fresh streaming turn after HITL
            chat_bridge.interrupts.clear()
            events2: list[UiPathRuntimeEvent] = []
            async for event in chat_runtime.stream({"messages": []}):
                events2.append(event)
            results2 = [e for e in events2 if isinstance(e, UiPathRuntimeResult)]
            assert len(results2) >= 1
            assert results2[-1].status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 2 failed: {results2[-1].status}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_tool_node_completed_after_hitl_resume(self):
        """Tool node should emit both STARTED and COMPLETED state events across HITL.

        Reproduces the bug where billing_agent_tools emitted STARTED (before HITL
        suspension) but never COMPLETED (after HITL resume), because the framework
        doesn't surface function_result in output/executor_completed events for
        handoff workflows. The fix synthesizes COMPLETED at the start of resume.

        Expected event sequence:
          customer_support STARTED
          triage STARTED
          triage COMPLETED
          billing_agent STARTED
          billing_agent_tools STARTED    <-- before HITL suspension
          [HITL interrupt + resume]
          billing_agent_tools COMPLETED  <-- synthesized on resume
          billing_agent STARTED          <-- resume executor
          billing_agent COMPLETED
          customer_support COMPLETED
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            chat_runtime, chat_bridge, storage = await _create_hitl_runtime_stack(
                agent, "test-hitl-tool-events", tmp_path, auto_approve=True
            )

            # Collect all events from streaming
            events: list[UiPathRuntimeEvent] = []
            async for event in chat_runtime.stream({"messages": []}):
                events.append(event)

            # Extract state events for analysis
            state_events = [e for e in events if isinstance(e, UiPathRuntimeStateEvent)]

            # Build a summary: node_name -> list of phases
            node_phases: dict[str, list[str]] = {}
            for se in state_events:
                if se.node_name:
                    node_phases.setdefault(se.node_name, []).append(se.phase.value)

            # billing_agent_tools MUST have both STARTED and COMPLETED
            tools_node = "billing_agent_tools"
            assert tools_node in node_phases, (
                f"{tools_node} not found in state events. "
                f"Nodes seen: {list(node_phases.keys())}"
            )
            tools_phases = node_phases[tools_node]
            assert UiPathRuntimeStatePhase.STARTED.value in tools_phases, (
                f"{tools_node} missing STARTED. Phases: {tools_phases}"
            )
            assert UiPathRuntimeStatePhase.COMPLETED.value in tools_phases, (
                f"{tools_node} missing COMPLETED after HITL resume. "
                f"Phases: {tools_phases}. "
                f"All state events: "
                + ", ".join(
                    f"{e.node_name}:{e.phase.value}"
                    for e in state_events
                    if e.node_name
                )
            )

            # Verify COMPLETED comes after STARTED
            started_idx = tools_phases.index(UiPathRuntimeStatePhase.STARTED.value)
            completed_idx = tools_phases.index(UiPathRuntimeStatePhase.COMPLETED.value)
            assert completed_idx > started_idx, (
                f"{tools_node} COMPLETED ({completed_idx}) should come after "
                f"STARTED ({started_idx})"
            )

            # Final result should be successful
            results = [e for e in events if isinstance(e, UiPathRuntimeResult)]
            assert len(results) >= 1
            assert results[-1].status == UiPathRuntimeStatus.SUCCESSFUL
        finally:
            await storage.dispose()
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Debug bridge mock
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Breakpoint + HITL combined tests
# ---------------------------------------------------------------------------

# Safety limit: if the debug loop exceeds this many resume calls,
# the test fails — this means breakpoints are stuck in a loop.
MAX_RESUME_CALLS = 20


@pytest.mark.asyncio(loop_scope="class")
class TestBreakpointAndHitlCombined:
    """Tests for the interaction between breakpoints and HITL tool approval.

    Uses the full runtime stack:
      UiPathDebugRuntime -> UiPathChatRuntime -> UiPathResumableRuntime
                         -> UiPathAgentFrameworkRuntime

    Reproduces the bug where breakpoint + @requires_approval on the same
    node causes an infinite loop: breakpoint → continue → HITL → approve →
    breakpoint (again!) → continue → HITL → ...
    """

    @pytest.fixture(autouse=True)
    async def _settle_framework(self):
        """Allow framework background tasks to complete between tests."""
        yield
        await asyncio.sleep(0.2)

    async def test_breakpoint_then_hitl_does_not_loop(self):
        """Breakpoint + HITL on same node must complete without looping.

        Flow:
          1. triage → billing_agent (breakpoint fires before billing_agent)
          2. User continues breakpoint
          3. billing_agent runs, calls transfer_funds (HITL fires)
          4. User approves tool via chat bridge
          5. Execution completes — NO more breakpoints

        Without the fix, step 5 would trigger another breakpoint on
        billing_agent, creating an infinite breakpoint→HITL→breakpoint loop.
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            runtime_id = "test-bp-hitl-loop"
            scoped_cs = ScopedCheckpointStorage(storage.checkpoint_storage, runtime_id)

            base_runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id=runtime_id,
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            base_runtime.chat = MagicMock()
            base_runtime.chat.map_messages_to_input.return_value = (
                "Transfer $100 from A to B"
            )
            base_runtime.chat.map_streaming_content.return_value = []
            base_runtime.chat.close_message.return_value = []

            resumable_runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=UiPathResumeTriggerHandler(),
                runtime_id=runtime_id,
            )

            chat_bridge = MockChatBridge(auto_approve=True)
            chat_runtime = UiPathChatRuntime(
                delegate=resumable_runtime, chat_bridge=chat_bridge
            )

            # Debug bridge: breakpoint on billing_agent
            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Infinite loop detected: {resume_count[0]} resumes. "
                        f"Breakpoint hits: "
                        f"{cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count}"
                    )
                return None

            debug_bridge = _make_debug_bridge()
            cast(Mock, debug_bridge.get_breakpoints).return_value = ["billing_agent"]
            cast(
                AsyncMock, debug_bridge.wait_for_resume
            ).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(
                delegate=chat_runtime, debug_bridge=debug_bridge
            )

            # Execute the full flow
            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count

            # Must complete successfully (not loop forever)
            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, Breakpoint hits: {bp_count}"
            )

            # Breakpoint should have been hit at least once.
            # The exact count depends on how many times the HandoffBuilder
            # calls execute() during checkpoint restore, but it MUST NOT
            # loop forever (which is caught by MAX_RESUME_CALLS).
            assert bp_count >= 1, f"Expected at least 1 breakpoint hit, got {bp_count}"

            # HITL should have been handled by chat bridge
            assert len(chat_bridge.interrupts) >= 1, (
                f"Expected at least 1 HITL interrupt, got {len(chat_bridge.interrupts)}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_breakpoint_on_all_nodes_with_hitl(self):
        """Breakpoints on ALL nodes + HITL on billing_agent must complete.

        Same as above but with breakpoints="*", which means triage also
        gets a breakpoint. Verifies the full sequence:
          1. Breakpoint on triage → continue
          2. triage hands off to billing_agent
          3. Breakpoint on billing_agent → continue
          4. billing_agent calls transfer_funds → HITL → approve
          5. Execution completes
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "X", "to_account": "Y", "amount": 50.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer done.", stream=is_stream)
            elif "triage" in system_msg.lower():
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            runtime_id = "test-bp-all-hitl"
            scoped_cs = ScopedCheckpointStorage(storage.checkpoint_storage, runtime_id)

            base_runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id=runtime_id,
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            base_runtime.chat = MagicMock()
            base_runtime.chat.map_messages_to_input.return_value = (
                "Transfer $50 from X to Y"
            )
            base_runtime.chat.map_streaming_content.return_value = []
            base_runtime.chat.close_message.return_value = []

            resumable_runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=UiPathResumeTriggerHandler(),
                runtime_id=runtime_id,
            )

            chat_bridge = MockChatBridge(auto_approve=True)
            chat_runtime = UiPathChatRuntime(
                delegate=resumable_runtime, chat_bridge=chat_bridge
            )

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Infinite loop detected: {resume_count[0]} resumes. "
                        f"Breakpoint hits: "
                        f"{cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count}"
                    )
                return None

            debug_bridge = _make_debug_bridge()
            cast(Mock, debug_bridge.get_breakpoints).return_value = "*"
            cast(
                AsyncMock, debug_bridge.wait_for_resume
            ).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(
                delegate=chat_runtime, debug_bridge=debug_bridge
            )

            result = await debug_runtime.execute({"messages": []})

            bp_count = cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count

            assert result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {result.status}. "
                f"Resumes: {resume_count[0]}, Breakpoint hits: {bp_count}"
            )

            # At least 1 breakpoint should have been hit
            assert bp_count >= 1, f"Expected at least 1 breakpoint hit, got {bp_count}"

            # HITL should have been handled (1 interrupt for transfer_funds)
            assert len(chat_bridge.interrupts) == 1, (
                f"Expected 1 HITL interrupt, got {len(chat_bridge.interrupts)}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_second_turn_after_breakpoint_and_hitl(self):
        """Second turn after breakpoint+HITL must actually run the workflow.

        Reproduces the bug where a breakpoint on the first executor of
        turn 2 saved a stale checkpoint_id from turn 1 (because no new
        checkpoint existed yet).  On resume, workflow.run(checkpoint_id=stale)
        restored turn 1's completed state and returned immediately with
        no work done — the OTEL trace showed empty workflow.run spans.

        Flow:
          Turn 1: breakpoints=* → triage → billing → HITL → approve → complete
          Turn 2: breakpoints=* → triage breakpoint → continue → triage
                  responds "How else can I help?" → complete

        The key assertion: triage's LLM must have been called during turn 2
        (call_count["triage"] increases).  Without the fix, the resume
        restores turn 1's state and triage is never called.
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            if "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            elif "triage" in system_msg.lower():
                count = call_count.get("triage", 0)
                call_count["triage"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "handoff_to_billing_agent", stream=is_stream
                    )
                else:
                    # Turn 2: triage responds directly
                    return make_mock_response("How else can I help?", stream=is_stream)
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            runtime_id = "test-bp-hitl-turn2"
            scoped_cs = ScopedCheckpointStorage(storage.checkpoint_storage, runtime_id)

            base_runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id=runtime_id,
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )
            base_runtime.chat = MagicMock()
            base_runtime.chat.map_messages_to_input.return_value = (
                "Transfer $100 from A to B"
            )
            base_runtime.chat.map_streaming_content.return_value = []
            base_runtime.chat.close_message.return_value = []

            resumable_runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=UiPathResumeTriggerHandler(),
                runtime_id=runtime_id,
            )

            chat_bridge = MockChatBridge(auto_approve=True)
            chat_runtime = UiPathChatRuntime(
                delegate=resumable_runtime, chat_bridge=chat_bridge
            )

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Infinite loop detected: {resume_count[0]} resumes"
                    )
                return None

            debug_bridge = _make_debug_bridge()
            cast(Mock, debug_bridge.get_breakpoints).return_value = "*"
            cast(
                AsyncMock, debug_bridge.wait_for_resume
            ).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(
                delegate=chat_runtime, debug_bridge=debug_bridge
            )

            # Turn 1: breakpoint + HITL
            result1 = await debug_runtime.execute({"messages": []})
            assert result1.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 1 failed: {result1.status}"
            )

            turn1_bp_count = cast(
                AsyncMock, debug_bridge.emit_breakpoint_hit
            ).await_count
            assert turn1_bp_count >= 1, "Turn 1 should have hit at least 1 breakpoint"

            # Record LLM call counts after turn 1
            triage_calls_after_turn1 = call_count.get("triage", 0)
            assert triage_calls_after_turn1 >= 1, (
                "Triage should have been called at least once during turn 1"
            )

            # Turn 2: fresh turn — should complete normally with breakpoints
            chat_bridge.interrupts.clear()
            resume_count[0] = 0
            cast(AsyncMock, debug_bridge.emit_breakpoint_hit).reset_mock()

            result2 = await debug_runtime.execute({"messages": []})
            assert result2.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Turn 2 failed: {result2.status}. "
                f"Resumes: {resume_count[0]}, "
                f"BPs: {cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count}"
            )

            turn2_bp_count = cast(
                AsyncMock, debug_bridge.emit_breakpoint_hit
            ).await_count
            # Turn 2 should also hit breakpoints (fresh run with breakpoints)
            assert turn2_bp_count >= 1, (
                f"Turn 2 should have hit at least 1 breakpoint, got {turn2_bp_count}"
            )

            # KEY ASSERTION: triage's LLM was actually called during turn 2.
            # Without the stale-checkpoint fix, the resume restores turn 1's
            # completed state and triage is never invoked again.
            triage_calls_after_turn2 = call_count.get("triage", 0)
            assert triage_calls_after_turn2 > triage_calls_after_turn1, (
                f"Turn 2 must invoke triage LLM. "
                f"Calls after turn 1: {triage_calls_after_turn1}, "
                f"after turn 2: {triage_calls_after_turn2}. "
                f"This means the resume used a stale checkpoint from turn 1."
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)

    async def test_no_duplicate_tool_calls_on_breakpoint_resume(self):
        """Breakpoint resumes must not emit duplicate tool call message events.

        Full E2E streaming test with BOTH breakpoints=* AND HITL approval.
        Reproduces the bug where checkpoint restore replays output events for
        already-completed executors.  The chat mapper processed each replayed
        Content(type='function_call') as new, emitting duplicate ToolCallStart
        events — visible as repeated 'handoff_to_billing_agent' in the UI.

        Full runtime stack:
          UiPathDebugRuntime -> UiPathChatRuntime -> UiPathResumableRuntime
                             -> UiPathAgentFrameworkRuntime

        Expected flow (breakpoints="*"):
          1. Breakpoint on triage → debug bridge continue
          2. triage runs, calls handoff_to_billing_agent → handoff to billing
          3. Breakpoint on billing_agent → debug bridge continue
          4. billing_agent runs, calls transfer_funds → HITL suspends
          5. Chat bridge auto-approves → billing completes
          6. More breakpoints may fire on fan-out agents → continues
          7. Workflow completes SUCCESSFUL

        Assertions at every step:
          - Breakpoints fired (emit_breakpoint_hit called)
          - HITL handled (chat_bridge.interrupts)
          - handoff_to_billing_agent ToolCallStart appears exactly ONCE
          - transfer_funds ToolCallStart appears exactly ONCE
          - Final result is SUCCESSFUL
        """
        call_count: dict[str, int] = {}

        async def mock_create(**kwargs: Any):
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            system_msg = extract_system_text(messages)

            # Check triage BEFORE billing — triage's system prompt contains
            # "billing_agent" in the handoff instructions, so a naive
            # "billing" check would match both agents.
            if "triage" in system_msg.lower():
                call_count["triage"] = call_count.get("triage", 0) + 1
                return make_tool_call_response(
                    "handoff_to_billing_agent", stream=is_stream
                )
            elif "billing" in system_msg.lower():
                count = call_count.get("billing", 0)
                call_count["billing"] = count + 1
                if count == 0:
                    return make_tool_call_response(
                        "transfer_funds",
                        arguments='{"from_account": "A", "to_account": "B", "amount": 100.0}',
                        stream=is_stream,
                    )
                else:
                    return make_mock_response("Transfer complete.", stream=is_stream)
            else:
                return make_mock_response("OK", stream=is_stream)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = mock_create

        agent = _build_hitl_agents(mock_openai)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        try:
            storage = SqliteResumableStorage(tmp_path)
            await storage.setup()
            assert storage.checkpoint_storage is not None

            runtime_id = "test-no-dup-tool-calls"
            scoped_cs = ScopedCheckpointStorage(storage.checkpoint_storage, runtime_id)

            base_runtime = UiPathAgentFrameworkRuntime(
                agent=agent,
                runtime_id=runtime_id,
                checkpoint_storage=scoped_cs,
                resumable_storage=storage,
            )

            resumable_runtime = UiPathResumableRuntime(
                delegate=base_runtime,
                storage=storage,
                trigger_manager=UiPathResumeTriggerHandler(),
                runtime_id=runtime_id,
            )

            chat_bridge = MockChatBridge(auto_approve=True)
            chat_runtime = UiPathChatRuntime(
                delegate=resumable_runtime, chat_bridge=chat_bridge
            )

            resume_count = [0]

            async def mock_wait_for_resume(*args: Any, **kwargs: Any) -> None:
                resume_count[0] += 1
                if resume_count[0] > MAX_RESUME_CALLS:
                    raise AssertionError(
                        f"Infinite loop detected: {resume_count[0]} resumes"
                    )
                return None

            debug_bridge = _make_debug_bridge()
            # Breakpoints on ALL nodes — maximizes checkpoint replays
            cast(Mock, debug_bridge.get_breakpoints).return_value = "*"
            cast(
                AsyncMock, debug_bridge.wait_for_resume
            ).side_effect = mock_wait_for_resume

            debug_runtime = UiPathDebugRuntime(
                delegate=chat_runtime, debug_bridge=debug_bridge
            )

            # ---- Stream and collect all events ----
            all_events: list[UiPathRuntimeEvent] = []
            async for event in debug_runtime.stream({"messages": []}):
                all_events.append(event)

            # ---- Step 1: Verify execution completed successfully ----
            results = [e for e in all_events if isinstance(e, UiPathRuntimeResult)]
            assert len(results) >= 1, "No UiPathRuntimeResult events found"
            final_result = results[-1]
            assert final_result.status == UiPathRuntimeStatus.SUCCESSFUL, (
                f"Expected SUCCESSFUL, got {final_result.status}. "
                f"Total results: {len(results)}, "
                f"Resumes: {resume_count[0]}"
            )

            # ---- Step 2: Verify breakpoints fired ----
            bp_hit_count = cast(AsyncMock, debug_bridge.emit_breakpoint_hit).await_count
            assert bp_hit_count >= 1, (
                f"Expected at least 1 breakpoint hit with breakpoints='*', "
                f"got {bp_hit_count}"
            )

            # ---- Step 3: Verify HITL was handled ----
            assert len(chat_bridge.interrupts) >= 1, (
                f"Expected at least 1 HITL interrupt for transfer_funds, "
                f"got {len(chat_bridge.interrupts)}"
            )

            # ---- Step 4: Verify LLM calls ----
            assert call_count.get("triage", 0) >= 1, (
                "Triage LLM should have been called at least once"
            )
            assert call_count.get("billing", 0) >= 1, (
                "Billing LLM should have been called at least once"
            )

            # ---- Step 5: Count ToolCallStart events per tool name ----
            tool_call_starts: dict[str, int] = {}
            tool_call_ends: dict[str, int] = {}
            for event in all_events:
                if not isinstance(event, UiPathRuntimeMessageEvent):
                    continue
                payload = event.payload
                if not hasattr(payload, "tool_call") or not payload.tool_call:
                    continue
                tc = payload.tool_call
                if hasattr(tc, "start") and tc.start:
                    name = tc.start.tool_name
                    tool_call_starts[name] = tool_call_starts.get(name, 0) + 1
                if hasattr(tc, "end") and tc.end:
                    tc_id = tc.tool_call_id or "unknown"
                    tool_call_ends[tc_id] = tool_call_ends.get(tc_id, 0) + 1

            # ---- Step 6: Assert no duplicate tool calls ----
            assert "handoff_to_billing_agent" in tool_call_starts, (
                f"handoff_to_billing_agent not found in tool calls. "
                f"Tool calls seen: {tool_call_starts}"
            )
            assert tool_call_starts["handoff_to_billing_agent"] == 1, (
                f"handoff_to_billing_agent should appear exactly once, "
                f"but appeared {tool_call_starts['handoff_to_billing_agent']} "
                f"times. All tool calls: {tool_call_starts}. "
                f"This means checkpoint replay emitted duplicate events."
            )

            assert "transfer_funds" in tool_call_starts, (
                f"transfer_funds not found in tool calls. "
                f"Tool calls seen: {tool_call_starts}"
            )
            assert tool_call_starts["transfer_funds"] == 1, (
                f"transfer_funds should appear exactly once, "
                f"but appeared {tool_call_starts['transfer_funds']} times. "
                f"All tool calls: {tool_call_starts}."
            )

            # ---- Step 7: Verify state events are well-formed ----
            state_events = [
                e for e in all_events if isinstance(e, UiPathRuntimeStateEvent)
            ]
            node_phases: dict[str, list[str]] = {}
            for se in state_events:
                if se.node_name:
                    node_phases.setdefault(se.node_name, []).append(se.phase.value)
            # The top-level agent node must have started and completed
            agent_node = "customer_support"
            assert agent_node in node_phases, (
                f"Top-level agent '{agent_node}' not found in state events. "
                f"Nodes: {list(node_phases.keys())}"
            )
            assert "started" in node_phases[agent_node], (
                "Agent node missing STARTED phase"
            )
            assert "completed" in node_phases[agent_node], (
                f"Agent node missing COMPLETED phase. Phases: {node_phases[agent_node]}"
            )
        finally:
            await storage.dispose()
            os.unlink(tmp_path)
