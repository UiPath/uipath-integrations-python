"""Expense agent that escalates to Action Center and waits for a human decision."""

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

BOOKED: dict[str, float] = {}


@tool(
    "request_approval",
    "Ask a human to approve or refuse an expense, and wait for their decision.",
    {"report_id": str, "question": str},
)
async def request_approval(args: dict[str, Any]) -> dict[str, Any]:
    action_data = await interrupt(
        CreateTask(
            title=f"Expense approval {args['report_id']}",
            data={"AgentName": "Expense agent", "AgentOutput": args["question"]},
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
    "book_expense",
    "Book an approved expense against the cost centre. Never call this before a "
    "human has approved the expense.",
    {"report_id": str, "amount": float},
)
async def book_expense(args: dict[str, Any]) -> dict[str, Any]:
    BOOKED[args["report_id"]] = args["amount"]
    return {
        "content": [
            {
                "type": "text",
                "text": f"Booked {args['amount']} for report {args['report_id']}.",
            }
        ]
    }


approvals_server = uipath_tool_server("approvals", tools=[request_approval])
expenses_server = create_sdk_mcp_server(name="expenses", tools=[book_expense])


class ExpenseReport(BaseModel):
    report_id: str = Field(description="Identifier of the expense report.")
    employee: str = Field(description="Person who filed the report.")
    amount: float = Field(description="Total amount claimed.")
    category: str = Field(description="Expense category, e.g. 'Travel'.")
    justification: str = Field(description="Why the employee claims the expense.")


class ExpenseDecision(BaseModel):
    approved: bool = Field(description="Whether the human approved the expense.")
    booked_amount: float = Field(description="Amount booked, 0 when refused.")
    summary: str = Field(description="One sentence describing what happened.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You review employee expense reports. You never decide yourself. "
            "For every report call request_approval exactly once, on its own and "
            "never alongside another tool call, passing the report id and a "
            "question naming the employee, the amount, the category and the "
            "justification. It comes back with the human's decision. Call "
            "book_expense with the full amount when the decision approves the "
            "report, and leave it unbooked otherwise."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"approvals": approvals_server, "expenses": expenses_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=ExpenseReport,
    output_schema=ExpenseDecision,
    prompt=(
        "Expense report {report_id} filed by {employee}: {amount} for "
        "{category}. Justification: {justification}"
    ),
)
