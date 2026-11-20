"""Codex agent that calls a remote (streamable-HTTP) MCP server tool.

Start orders_http_server.py first (see its docstring). The bearer token Codex
sends is read from the ORDERS_TOKEN env var; for a UiPath AgentHub-hosted MCP
server you would instead point bearer_token_env_var at "UIPATH_ACCESS_TOKEN".
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import RemoteMcpServer, UiPathCodexAgent


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
        "orders": RemoteMcpServer(
            url="http://127.0.0.1:8931/mcp",
            # Codex sends this as `Authorization: Bearer`. Providing a literal
            # token makes the sample self-contained: the runtime injects it into
            # the codex env, so the server is reached regardless of ambient env.
            # (A real UiPath AgentHub MCP server would instead use
            # bearer_token_env_var="UIPATH_ACCESS_TOKEN".)
            bearer_token_env_var="ORDERS_TOKEN",
            bearer_token="demo-secret-token",
        )
    },
)
