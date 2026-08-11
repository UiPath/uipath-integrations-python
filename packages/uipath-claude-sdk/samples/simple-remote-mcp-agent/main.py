"""Agent whose tools come from a remote MCP server over streamable HTTP."""

import os

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel, Field

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel
from uipath_claude_sdk.mcp import remote_mcp_server

MCP_SERVER_URL = os.getenv(
    "UIPATH_MCP_SERVER_URL",
    "https://REPLACE-ME.example.com/mcp",
)


class ServerRequest(BaseModel):
    request: str = Field(description="What the user wants the server to do.")


class ServerAnswer(BaseModel):
    answer: str = Field(description="Answer written from the tool results.")
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of the MCP tools that were called, in call order.",
    )


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You answer requests using only the tools exposed by the 'remote' "
            "MCP server. Inspect the tools you have, pick the ones that fit the "
            "request, and never guess a value a tool could have told you. "
            "List every tool you called in tools_used."
        ),
        max_turns=12,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"remote": remote_mcp_server(MCP_SERVER_URL)},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=ServerRequest,
    output_schema=ServerAnswer,
    prompt="{request}",
)
