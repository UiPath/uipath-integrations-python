"""Expense agent that raises an Action Center action before it approves anything."""

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from pydantic import BaseModel, Field
from uipath.platform.common import CreateTask

from uipath_claude_sdk import (
    UiPathClaudeAgent,
    UiPathModel,
    interrupt,
    uipath_tool_server,
)

ESCALATION_APP = "generic_escalation_app"
ESCALATION_FOLDER = "Shared"

LEDGER: dict[str, float] = {}


@tool(
    "request_approval",
    "Ask a human to approve or refuse an expense, and wait for their decision.",
    {"employee": str, "amount": float, "question": str},
)
async def request_approval(args: dict[str, Any]) -> dict[str, Any]:
    action_data = await interrupt(
        CreateTask(
            title=f"Expense approval for {args['employee']}: {args['amount']}",
            data={"AgentName": "Expense approver", "AgentOutput": args["question"]},
            app_name=ESCALATION_APP,
            app_folder_path=ESCALATION_FOLDER,
        )
    )
    decision = {
        "answer": action_data.get("Answer"),
        "comment": action_data.get("Comment", ""),
    }
    return {"content": [{"type": "text", "text": json.dumps(decision, default=str)}]}


@tool(
    "record_reimbursement",
    "Record an approved reimbursement. Never call this before a human approved it.",
    {"employee": str, "amount": float},
)
async def record_reimbursement(args: dict[str, Any]) -> dict[str, Any]:
    LEDGER[args["employee"]] = args["amount"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"Recorded {args['amount']} for {args['employee']}.",
            }
        ]
    }


approvals_server = uipath_tool_server("approvals", tools=[request_approval])
expenses_server = create_sdk_mcp_server(name="expenses", tools=[record_reimbursement])


class ExpenseReport(BaseModel):
    employee: str = Field(description="Who filed the expense report.")
    amount: float = Field(description="Amount the employee is claiming back.")
    category: str = Field(description="What the expense was for.")


class ApprovalOutcome(BaseModel):
    approved: bool = Field(description="Whether the human approved the expense.")
    approved_amount: float = Field(description="Amount approved, 0 when refused.")
    summary: str = Field(description="One sentence describing what happened.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You review employee expense reports. You never decide one yourself. "
            "Before you record anything you MUST call request_approval, passing "
            "the employee, the amount and a question naming what they are "
            "claiming and why. Make that call on its own, never together with "
            "another tool call. It comes back with the human's decision. Call "
            "record_reimbursement only if the decision approves the report, for "
            "the amount they approved, and leave it untouched otherwise."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"approvals": approvals_server, "expenses": expenses_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=ExpenseReport,
    output_schema=ApprovalOutcome,
    prompt="Expense report from {employee}: {amount} claimed back for {category}.",
)
