"""Google ADK multi-agent-remote example: same pipeline as multi-agent but with sub-agents hosted remotely via A2A.

Demonstrates how to mix local orchestration with remote agent implementations:
- Coordinator and formatter run locally (they hold the orchestration logic)
- Specialist sub-agents (research, code) are RemoteA2aAgent instances hosted elsewhere
- The remote services don't need to know about each other — coordination stays local

Compare with the multi-agent sample:
  multi-agent:        Agent(tools=[search_web])   Agent(tools=[run_python])
  multi-agent-remote: RemoteA2aAgent(agent_card=...) for each specialist

The key insight: RemoteA2aAgent cannot have sub_agents (it's not an LlmAgent),
but a local Agent CAN have RemoteA2aAgent instances as its sub_agents. This lets
you keep orchestration logic local while moving implementations to remote services.
"""

import os

import httpx
from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import TransportProtocol as A2ATransport
from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from pydantic import BaseModel, Field


class ReportInput(BaseModel):
    """Structured input for the report generation pipeline."""

    topic: str = Field(
        default="Natural Language Processing fundamentals",
        description="The topic to research and analyze",
    )
    depth: str = Field(
        default="standard",
        description="How deep the analysis should be: 'brief', 'standard', or 'detailed'",
    )


class ReportOutput(BaseModel):
    """Structured output from the report generation pipeline."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary of findings")
    key_findings: list[str] = Field(description="Key findings as bullet points")
    code_snippet: str = Field(description="A relevant Python code example")


# UIPATH_ACCESS_TOKEN is set automatically by `uipath auth`
_access_token = os.environ.get("UIPATH_ACCESS_TOKEN", "")

_http_client = httpx.AsyncClient(
    headers={"Authorization": f"Bearer {_access_token}"},
    timeout=httpx.Timeout(300.0),
)

_a2a_client_factory = A2AClientFactory(
    config=A2AClientConfig(
        httpx_client=_http_client,
        supported_transports=[A2ATransport.jsonrpc],
        streaming=False,
        polling=False,
        accepted_output_modes=["text"],
    ),
)

# --- Remote Sub-agents ---
# Replace the URLs with your actual deployed agent endpoints.
ORG_NAME = "YourOrgName"
TENANT_NAME = "YourTenantName"
RESEARCH_AGENT_FOLDER_KEY = "a11f72b1-90fd-4b30-b733-f0285cbf4a19"
RESEARCH_AGENT_RELEASE_ID = "1234"
CODE_AGENT_FOLDER_KEY = "b22f83c2-91fe-5c41-c844-g1396dcg5b2a"
CODE_AGENT_RELEASE_ID = "5678"

research_agent = RemoteA2aAgent(
    name="research_agent",
    agent_card=f"https://cloud.uipath.com/{ORG_NAME}/{TENANT_NAME}/agenthub_/a2a/{RESEARCH_AGENT_FOLDER_KEY}/{RESEARCH_AGENT_RELEASE_ID}/.well-known/agent-card.json",
    description="Remote research specialist that searches the web and summarizes findings",
    a2a_client_factory=_a2a_client_factory,
)

code_agent = RemoteA2aAgent(
    name="code_agent",
    agent_card=f"https://cloud.uipath.com/{ORG_NAME}/{TENANT_NAME}/agenthub_/a2a/{CODE_AGENT_FOLDER_KEY}/{CODE_AGENT_RELEASE_ID}/.well-known/agent-card.json",
    description="Remote Python developer that writes and executes code examples",
    a2a_client_factory=_a2a_client_factory,
)


# --- Coordinator (local Agent, sub_agents are remote) ---
coordinator = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "You are a report coordinator. Given a topic:\n"
        "1. Delegate research to research_agent to gather information\n"
        "2. Delegate to code_agent to write a relevant Python code example\n"
        "3. Compile all findings into a comprehensive text report\n"
        "Include the research findings and the code example in your response."
    ),
    sub_agents=[research_agent, code_agent],
    input_schema=ReportInput,
    output_key="research_results",
)


# --- Formatter (local Agent with output_schema) ---
formatter = Agent(
    name="formatter",
    model="gemini-2.5-flash",
    instruction=(
        "You are a report formatter. Take the research results from the previous "
        "step and format them into a structured report with a title, summary, "
        "key findings, and a code snippet. Output valid JSON matching the schema."
    ),
    output_schema=ReportOutput,
    output_key="report",
)


# --- Root: SequentialAgent pipeline ---
agent = SequentialAgent(
    name="pipeline",
    sub_agents=[coordinator, formatter],
)
