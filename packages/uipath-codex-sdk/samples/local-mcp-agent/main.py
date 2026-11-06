"""Codex agent that calls a local stdio MCP server tool."""

import os
import sys

from pydantic import BaseModel, Field

from uipath_codex_sdk import StdioMcpServer, UiPathCodexAgent

_HERE = os.path.dirname(os.path.abspath(__file__))


class OrderQuery(BaseModel):
    order_id: str = Field(description="The order id to look up, e.g. 'A-1001'.")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Look up the status of order '{order_id}' using the orders tool and report it verbatim.",
    input_schema=OrderQuery,
    sandbox="full-access",
    approval_mode="deny_all",
    base_instructions=(
        "You have an 'orders' MCP tool. Always call it to look up order "
        "status; never guess. Report exactly what the tool returns."
    ),
    mcp_servers={
        "orders": StdioMcpServer(
            command=sys.executable,
            args=[os.path.join(_HERE, "orders_server.py")],
        )
    },
)
