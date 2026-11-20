"""Codex agent that uses a UiPath-hosted (AgentHub) MCP server.

The MCP URL comes from UIPATH_MCP_URL (Orchestrator -> folder -> MCP Servers ->
Copy URL). Codex authenticates with the same token `uipath auth` wrote to .env.
"""

import os

from pydantic import BaseModel, Field

from uipath_codex_sdk import RemoteMcpServer, UiPathCodexAgent

MCP_URL = os.environ.get(
    "UIPATH_MCP_URL",
    "https://alpha.uipath.com/AdminStudioTest/al3x/agenthub_/mcp/"
    "cbdabce1-ae2b-4dd6-99f9-632795c88ad9/arithmetic-operations-mcp",
)


class Task(BaseModel):
    task: str = Field(description="What to ask the agent, e.g. 'What is 17 * 23?'")


agent = UiPathCodexAgent(
    model=os.environ.get("CODEX_MODEL", "gpt-5.6-terra"),
    prompt="{task}",
    input_schema=Task,
    # read-only: if the MCP tool isn't available the agent must say so,
    # instead of improvising with shell commands.
    sandbox="read-only",
    approval_mode="deny_all",
    base_instructions=(
        "You have a 'uipath_mcp' MCP server with arithmetic tools. Always use its "
        "tools to compute results; never compute them yourself. Report which "
        "tool you called and its exact result."
    ),
    mcp_servers={
        "uipath_mcp": RemoteMcpServer(
            url=MCP_URL,
            bearer_token_env_var="UIPATH_ACCESS_TOKEN",
            startup_timeout_sec=60,
            tool_timeout_sec=900,
            # UiPath MCP tools carry no readOnlyHint annotation, so Codex's
            # default ("auto") would ask for approval, which deny_all rejects.
            # This is a trusted tenant server: pre-approve its tools.
            default_tools_approval_mode="approve",
        )
    },
)
