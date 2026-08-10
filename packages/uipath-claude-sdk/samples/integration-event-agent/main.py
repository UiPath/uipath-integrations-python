"""On-call agent that suspends until Integration Services delivers a reply."""

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, tool
from pydantic import BaseModel, Field
from uipath.platform.common import WaitIntegrationEvent

from uipath_claude_sdk import (
    UiPathClaudeAgent,
    UiPathModel,
    interrupt,
    uipath_tool_server,
)

CONNECTOR = "uipath-slack"
CONNECTION_NAME = "Slack-OnCall"
OPERATION = "OnMessage"
OBJECT_NAME = "Message"
FILTER_EXPRESSION = "channel == 'oncall'"


@tool(
    "await_oncall_reply",
    "Wait for the on-call engineer to reply, and return their message.",
    {},
)
async def await_oncall_reply(args: dict[str, Any]) -> dict[str, Any]:
    event = await interrupt(
        WaitIntegrationEvent(
            connector=CONNECTOR,
            connection_name=CONNECTION_NAME,
            operation=OPERATION,
            object_name=OBJECT_NAME,
            filter_expression=FILTER_EXPRESSION,
        )
    )
    return {"content": [{"type": "text", "text": json.dumps(event, default=str)}]}


oncall_server = uipath_tool_server("oncall", tools=[await_oncall_reply])


class EscalationRequest(BaseModel):
    incident: str = Field(description="Incident the on-call engineer must answer for.")


class EscalationOutcome(BaseModel):
    acknowledged: bool = Field(
        description="Whether the delivered event acknowledges the incident."
    )
    responder: str = Field(
        description="Who replied, empty when the event does not name anyone."
    )
    next_step: str = Field(description="What the reply says should happen next.")
    summary: str = Field(description="One sentence describing the reply.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You escalate incidents to the on-call engineer and wait for their "
            "reply.\n"
            "Call await_oncall_reply exactly once, on its own and never together "
            "with another tool call. It comes back with the engineer's message.\n"
            "Read the reply and fill the output. Treat the incident as "
            "acknowledged only when the reply actually says someone is taking it, "
            "and leave responder empty when the message names nobody."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"oncall": oncall_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=EscalationRequest,
    output_schema=EscalationOutcome,
    prompt="Incident to escalate: {incident}",
)
