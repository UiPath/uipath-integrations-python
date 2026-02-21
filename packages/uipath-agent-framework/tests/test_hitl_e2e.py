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
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
from uipath.runtime.events import UiPathRuntimeEvent
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

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def emit_message_event(self, message_event: Any) -> None:
        self.messages.append(message_event)

    async def emit_interrupt_event(self, resume_trigger: UiPathResumeTrigger) -> None:
        self.interrupts.append(resume_trigger)

    async def emit_exchange_end_event(self) -> None:
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
