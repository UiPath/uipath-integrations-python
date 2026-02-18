"""Google ADK multi-agent example with SequentialAgent, sub-agents, and typed I/O.

Demonstrates the "formatter pattern" for structured output:
- SequentialAgent root → [coordinator, formatter]
- Coordinator delegates to specialist sub-agents (research + code generation)
- Formatter produces structured JSON output (output_schema)

IMPORTANT — Google ADK / Gemini API constraint:
  output_schema on an LlmAgent sets response_mime_type to 'application/json',
  which is INCOMPATIBLE with function calling (tools or sub_agents). The Gemini
  API will reject the request with a 400 error:
    "Function calling with a response mime type: 'application/json' is unsupported"

  To get structured output from an agent that uses tools/sub_agents, use the
  formatter pattern:
    SequentialAgent → [worker_agent (tools/sub_agents, output_key), formatter_agent (output_schema)]

  The worker does the heavy lifting (tools, sub_agents, function calling), stores
  its result via output_key, and the formatter reads it and produces structured
  JSON output constrained by output_schema.

Graph structure:
  __start__ → pipeline → coordinator → research_agent (with search_wikipedia + search_tavily tools)
                                      → code_agent (with run_python tool)
                       → formatter (output_schema=ReportOutput)
           → __end__

Schema resolution (handled by the runtime recursively):
  - input_schema:  from FIRST sub_agent chain → coordinator.input_schema (ReportInput)
  - output_schema: from LAST sub_agent chain  → formatter.output_schema (ReportOutput)
  - output_key:    from LAST sub_agent chain  → formatter.output_key ("report")

Environment variables:
  - TAVILY_API_KEY: API key for Tavily web search (get one at https://tavily.com)
"""

import io
import os
import sys
import traceback

import httpx
from google.adk.agents import Agent, SequentialAgent
from pydantic import BaseModel, Field


class ReportInput(BaseModel):
    """Structured input for the report generation pipeline."""

    topic: str = Field(description="The topic to research and analyze")
    depth: str = Field(
        default="brief",
        description="How deep the analysis should be: 'brief', 'standard', or 'detailed'",
    )


class ReportOutput(BaseModel):
    """Structured output from the report generation pipeline."""

    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary of findings")
    key_findings: list[str] = Field(description="Key findings as bullet points")
    code_snippet: str = Field(description="A relevant Python code example")


def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for encyclopedic information on a topic.

    Best for well-known concepts, historical facts, and established knowledge.

    Args:
        topic: The topic to look up on Wikipedia

    Returns:
        Summary text from Wikipedia, or an error message if not found
    """
    try:
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
            headers={"User-Agent": "UiPathGoogleADKSample/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        return f"Wikipedia — {title}: {extract}"
    except Exception as e:
        return f"Wikipedia search failed for '{topic}': {e}"


def search_tavily(query: str) -> str:
    """Search the web using Tavily for recent and real-time information.

    Best for current events, recent developments, and up-to-date information
    that may not yet be on Wikipedia.

    Args:
        query: The search query

    Returns:
        Search results with snippets from relevant web pages
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "Tavily search unavailable: TAVILY_API_KEY not set"

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": 3, "include_answer": True},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = []
        answer = data.get("answer")
        if answer:
            parts.append(f"Summary: {answer}")
        for result in data.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            source = result.get("url", "")
            parts.append(f"- {title}: {content} ({source})")
        return "\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Tavily search failed for '{query}': {e}"


def run_python(code: str) -> str:
    """Execute Python code and return its stdout output.

    Args:
        code: Python source code to execute

    Returns:
        The captured stdout output, or the error traceback if execution fails
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


# --- Sub-agents ---

# Research agent: has two search tools for comprehensive research.
# - search_wikipedia: encyclopedic/established knowledge
# - search_tavily: real-time web search for recent information
research_agent = Agent(
    name="research_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a research specialist with two search tools:\n"
        "- search_wikipedia: for encyclopedic facts and established knowledge\n"
        "- search_tavily: for recent developments and real-time web information\n\n"
        "Use both tools to gather comprehensive information about the given topic. "
        "Start with Wikipedia for foundational knowledge, then use Tavily for "
        "recent developments. Provide a thorough summary of your findings."
    ),
    tools=[search_wikipedia, search_tavily],
)

# Code agent: writes and executes Python code related to the topic.
# Has the run_python tool to actually run the code it generates.
code_agent = Agent(
    name="code_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Python developer. Given a topic, write a short, practical "
        "Python code example that demonstrates or relates to the topic. "
        "Use the run_python tool to execute your code and verify it works. "
        "Return both the code and its output."
    ),
    tools=[run_python],
)


# --- Coordinator (has sub_agents + output_key, NO output_schema) ---
#
# output_key stores the coordinator's final text response in
# session.state["research_results"]. This agent delegates to research_agent
# and code_agent, then compiles results.
#
# NOTE: Do NOT set output_schema here — it's incompatible with sub_agents
# in the Gemini API.
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


# --- Formatter (has output_schema, NO tools or sub_agents) ---
#
# This is the "formatter pattern": a dedicated LlmAgent that takes unstructured
# text (from coordinator via output_key in session state) and produces
# structured JSON output.
#
# output_schema sets response_mime_type='application/json' in the Gemini API,
# which constrains the LLM to produce valid JSON matching the Pydantic model.
# This ONLY works when the agent has NO tools and NO sub_agents.
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
#
# SequentialAgent runs sub_agents in order: coordinator first, then formatter.
# The runtime resolves schemas recursively:
#   - input_schema:  from FIRST sub_agent chain → coordinator.input_schema (ReportInput)
#   - output_schema: from LAST sub_agent chain  → formatter.output_schema (ReportOutput)
#   - output_key:    from LAST sub_agent chain  → formatter.output_key ("report")
agent = SequentialAgent(
    name="pipeline",
    sub_agents=[coordinator, formatter],
)
