import io
import sys

import httpx

from uipath_agent_framework.chat import UiPathOpenAIChatClient


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a topic and return a summary.

    Args:
        query: The search query, e.g. "Python programming language"

    Returns:
        A summary of the Wikipedia article, or an error message.
    """
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + query.replace(" ", "_"),
            headers={"User-Agent": "UiPathMultiAgent/1.0"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("extract", "No summary available.")
    except Exception as e:
        return f"Wikipedia search failed for '{query}': {e}"


def run_python(code: str) -> str:
    """Execute a Python code snippet and return its output.

    Args:
        code: The Python code to execute.

    Returns:
        The captured stdout output, or an error message.
    """
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})  # noqa: B023
        return captured.getvalue() or "(no output)"
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        sys.stdout = old_stdout


client = UiPathOpenAIChatClient(model="gpt-4o-mini")

research_agent = client.as_agent(
    name="research_agent",
    description="Searches Wikipedia for information on any topic.",
    instructions="You are a research assistant. Use the search_wikipedia tool to find information. Provide concise, factual summaries.",
    tools=[search_wikipedia],
)

code_agent = client.as_agent(
    name="code_agent",
    description="Executes Python code snippets and returns the output.",
    instructions="You are a coding assistant. Use the run_python tool to execute Python code. Always validate code before running.",
    tools=[run_python],
)

# Coordinator delegates to specialists via as_tool()
agent = client.as_agent(
    name="coordinator",
    instructions=(
        "You are a coordinator that delegates tasks to specialist agents.\n"
        "- Use 'research_agent' for any research or factual questions.\n"
        "- Use 'code_agent' for any coding, calculation, or data processing tasks.\n"
        "Combine their results to give comprehensive answers."
    ),
    tools=[research_agent.as_tool(), code_agent.as_tool()],
)
