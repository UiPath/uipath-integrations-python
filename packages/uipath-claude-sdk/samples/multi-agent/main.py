"""Briefing agent that delegates the research to another Orchestrator process."""

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, tool
from pydantic import BaseModel, Field
from uipath.platform.common import InvokeProcess

from uipath_claude_sdk import (
    UiPathClaudeAgent,
    UiPathModel,
    interrupt,
    uipath_tool_server,
)

RESEARCH_PROCESS = "company-research-agent"
RESEARCH_FOLDER = "Shared"


@tool(
    "run_research",
    "Research a topic and return the findings. Takes as long as it takes.",
    {"topic": str},
)
async def run_research(args: dict[str, Any]) -> dict[str, Any]:
    output = await interrupt(
        InvokeProcess(
            name=RESEARCH_PROCESS,
            input_arguments={"topic": args["topic"]},
            process_folder_path=RESEARCH_FOLDER,
        )
    )
    return {"content": [{"type": "text", "text": json.dumps(output, default=str)}]}


research_server = uipath_tool_server("research", tools=[run_research])


class BriefingRequest(BaseModel):
    topic: str = Field(description="Company or subject to brief on.")


class Briefing(BaseModel):
    report: str = Field(description="The briefing, written from the research output.")
    sources: list[str] = Field(
        default_factory=list,
        description="Sources the research named, empty when it named none.",
    )


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You write short company briefings, but you never research anything "
            "yourself. Call run_research once with the topic, on its own and "
            "never together with another tool call. It comes back with the "
            "findings. Write the briefing from them, and do not invent anything "
            "the findings do not support."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"research": research_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=BriefingRequest,
    output_schema=Briefing,
    prompt="Write a briefing on {topic}.",
)
