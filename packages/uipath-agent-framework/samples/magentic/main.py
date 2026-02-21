import httpx
from agent_framework.orchestrations import MagenticBuilder

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
            headers={"User-Agent": "UiPathMagentic/1.0"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("extract", "No summary available.")
    except Exception as e:
        return f"Wikipedia search failed for '{query}': {e}"


client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

researcher = client.as_agent(
    name="researcher",
    description="Specialist in research and information gathering using Wikipedia.",
    instructions=(
        "You are a Researcher. Use the search_wikipedia tool to find factual "
        "information. Provide concise, well-sourced responses without additional "
        "computation or quantitative analysis."
    ),
    tools=[search_wikipedia],
)

analyst = client.as_agent(
    name="analyst",
    description="Specialist in analyzing data and drawing conclusions.",
    instructions=(
        "You are an Analyst. Analyze the information gathered by other agents. "
        "Draw conclusions, identify patterns, and provide actionable insights. "
        "Be precise and support your analysis with evidence from the discussion."
    ),
)

manager = client.as_agent(
    name="magentic_manager",
    description="Orchestrator that coordinates the research and analysis workflow.",
    instructions=(
        "You coordinate a team of researcher and analyst to complete complex "
        "tasks efficiently. Break down problems, delegate subtasks, and "
        "synthesize results into a comprehensive final answer."
    ),
)

workflow = MagenticBuilder(
    participants=[researcher, analyst],
    manager_agent=manager,
    max_round_count=6,
    max_stall_count=2,
    max_reset_count=1,
).build()

agent = workflow.as_agent(name="magentic_workflow")
