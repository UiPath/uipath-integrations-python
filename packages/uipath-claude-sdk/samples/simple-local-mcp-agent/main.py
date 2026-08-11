"""Agent whose tools come from a local stdio MCP server run as a subprocess."""

import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel, Field

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

LEDGER_SERVER = Path(__file__).parent / "ledger_server.py"


class ExpenseReport(BaseModel):
    expenses: list[str] = Field(
        description="Expense lines, each naming a label and an amount."
    )
    currency: str = Field(default="EUR", description="Currency the amounts are in.")


class ExpenseSummary(BaseModel):
    total: float = Field(description="Sum of every expense, as the ledger computed it.")
    largest: str = Field(description="Label of the most expensive entry.")
    explanation: str = Field(description="One sentence describing the report.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You summarize expense reports. Record every expense line with "
            "add_entry, one call per line, then read the numbers back with "
            "total and largest_entry. Never add the amounts up yourself: the "
            "ledger is the only source of truth for the total."
        ),
        max_turns=15,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={
            "ledger": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(LEDGER_SERVER)],
            }
        },
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=ExpenseReport,
    output_schema=ExpenseSummary,
    prompt="Summarize these expenses, in {currency}: {expenses}",
)
