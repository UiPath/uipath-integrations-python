"""Refund agent that asks a human for approval before it moves any money."""

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from pydantic import BaseModel, Field

from uipath_claude_sdk import (
    UiPathClaudeAgent,
    UiPathModel,
    interrupt,
    uipath_tool_server,
)

LEDGER: dict[str, float] = {}


@tool(
    "ask_approver",
    "Ask a human a question and wait for their answer.",
    {"question": str},
)
async def ask_approver(args: dict[str, Any]) -> dict[str, Any]:
    answer = await interrupt(args["question"])
    return {"content": [{"type": "text", "text": json.dumps(answer, default=str)}]}


@tool(
    "issue_refund",
    "Refund an order. Never call this before a human has approved the refund.",
    {"order_id": str, "amount": float},
)
async def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
    LEDGER[args["order_id"]] = args["amount"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"Refunded {args['amount']} on order {args['order_id']}.",
            }
        ]
    }


approvals_server = uipath_tool_server("approvals", tools=[ask_approver])
refunds_server = create_sdk_mcp_server(name="refunds", tools=[issue_refund])


class RefundRequest(BaseModel):
    order_id: str = Field(description="Order the customer wants refunded.")
    amount: float = Field(description="Amount the customer is asking back.")
    reason: str = Field(description="Why the customer is asking for a refund.")


class RefundOutcome(BaseModel):
    approved: bool = Field(description="Whether the human approved the refund.")
    refunded_amount: float = Field(description="Amount refunded, 0 when refused.")
    summary: str = Field(description="One sentence describing what happened.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You process customer refund requests. You never refund anything on "
            "your own authority. Call ask_approver first, with a question naming "
            "the order, the amount and the reason. Make that call on its own, "
            "never together with another tool call. Call issue_refund only if the "
            "answer approves the refund, and leave the order untouched otherwise."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"approvals": approvals_server, "refunds": refunds_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=RefundRequest,
    output_schema=RefundOutcome,
    prompt="Refund request for order {order_id}: {amount} back, because {reason}.",
)
