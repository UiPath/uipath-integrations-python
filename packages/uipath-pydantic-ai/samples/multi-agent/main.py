"""PydanticAI multi-agent example with agent delegation.

Demonstrates the "agent delegation" pattern where a coordinator agent
delegates to specialist agents via tool calls:

  coordinator_agent
    ├── research_agent (called via tool)
    └── code_agent (called via tool)

PydanticAI multi-agent pattern:
- Each agent is a standalone Agent instance
- The coordinator uses @agent.tool to call child agents
- Child agents run to completion and return results to the coordinator
- Usage is tracked across all agents via shared RunContext.usage

NOTE: The entrypoint in pydantic_ai.json points to the coordinator agent.
The child agents are internal — they don't need separate entrypoints.
"""

import io
import sys
import traceback

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from uipath_pydantic_ai.chat import UiPathChatOpenAI


# --- Structured Output ---
class ReportOutput(BaseModel):
    """Structured output from the report generation pipeline."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary of findings")
    key_findings: list[str] = Field(description="Key findings as bullet points")
    code_snippet: str = Field(description="A relevant Python code example")


MODEL = "gpt-4.1-mini-2025-04-14"
uipath_client = UiPathChatOpenAI(model_name=MODEL)


# --- Child Agent: Researcher ---
research_agent = Agent(
    uipath_client.model,
    name="research_agent",
    instructions=(
        "You are a research specialist. Use the search_wikipedia tool to find "
        "comprehensive information about the given topic. Provide a thorough "
        "summary of your findings."
    ),
)


@research_agent.tool
async def search_wikipedia(ctx: RunContext[None], topic: str) -> str:
    """Search Wikipedia for encyclopedic information on a topic.

    Args:
        ctx: The agent context.
        topic: The topic to look up on Wikipedia.

    Returns:
        Summary text from Wikipedia, or an error message if not found.
    """
    try:
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
            headers={"User-Agent": "UiPathPydanticAISample/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        return f"Wikipedia — {title}: {extract}"
    except Exception as e:
        return f"Wikipedia search failed for '{topic}': {e}"


# --- Child Agent: Code Generator ---
code_agent = Agent(
    uipath_client.model,
    name="code_agent",
    instructions=(
        "You are a Python developer. Given a topic, write a short, practical "
        "Python code example that demonstrates or relates to the topic. "
        "Use the run_python tool to execute your code and verify it works. "
        "Return both the code and its output."
    ),
)


@code_agent.tool
async def run_python(ctx: RunContext[None], code: str) -> str:
    """Execute Python code and return its stdout output.

    Args:
        ctx: The agent context.
        code: Python source code to execute.

    Returns:
        The captured stdout output, or the error traceback if execution fails.
    """
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_capture
        exec(code, {"__builtins__": __builtins__})  # noqa: S102
        output = stdout_capture.getvalue()
        return output if output else "(no output)"
    except Exception:
        return f"Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


# --- Coordinator Agent ---
# The coordinator delegates to research_agent and code_agent via tools,
# then combines their outputs into a structured report.
agent = Agent(
    uipath_client.model,
    name="coordinator",
    output_type=ReportOutput,
    instructions=(
        "You are a report coordinator. Given a topic:\n"
        "1. Use delegate_to_researcher to gather information about the topic\n"
        "2. Use delegate_to_coder to generate a relevant Python code example\n"
        "3. Combine the results into a structured report\n"
        "Always delegate to both agents before producing your final output."
    ),
)


@agent.tool
async def delegate_to_researcher(ctx: RunContext[None], topic: str) -> str:
    """Delegate research to the research specialist agent.

    Args:
        ctx: The agent context.
        topic: The topic to research.

    Returns:
        Research findings from the research agent.
    """
    result = await research_agent.run(
        f"Research the following topic thoroughly: {topic}",
        usage=ctx.usage,
    )
    return str(result.output)


@agent.tool
async def delegate_to_coder(ctx: RunContext[None], topic: str) -> str:
    """Delegate code generation to the Python developer agent.

    Args:
        ctx: The agent context.
        topic: The topic to write code about.

    Returns:
        Python code example and its output from the code agent.
    """
    result = await code_agent.run(
        f"Write a Python code example related to: {topic}",
        usage=ctx.usage,
    )
    return str(result.output)
