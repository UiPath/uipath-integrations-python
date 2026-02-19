import httpx
from agent_framework.orchestrations import GroupChatBuilder

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
            headers={"User-Agent": "UiPathGroupChat/1.0"},
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
    description="Expert at finding facts and data using Wikipedia.",
    instructions=(
        "You are a research specialist. Use the search_wikipedia tool "
        "to find factual information. Provide concise, well-sourced responses."
    ),
    tools=[search_wikipedia],
)

critic = client.as_agent(
    name="critic",
    description="Challenges assumptions and evaluates claims critically.",
    instructions=(
        "You are a critical thinker. Evaluate the claims made by other "
        "participants. Point out gaps, biases, or missing context. "
        "Ask probing questions to deepen the discussion."
    ),
)

writer = client.as_agent(
    name="writer",
    description="Synthesizes group discussion into clear, structured prose.",
    instructions=(
        "You are a skilled writer. Synthesize the group discussion into "
        "a clear, well-organized summary. Incorporate the researcher's "
        "facts and address the critic's concerns."
    ),
)

orchestrator = client.as_agent(
    name="orchestrator",
    description="Coordinates the group discussion by selecting the next speaker.",
    instructions=(
        "You coordinate a team of researcher, critic, and writer. "
        "Select the next speaker based on the conversation flow:\n"
        "- Pick 'researcher' when facts or data are needed.\n"
        "- Pick 'critic' to challenge or evaluate claims.\n"
        "- Pick 'writer' to synthesize when enough discussion has happened.\n"
        "Respond with ONLY the agent name, nothing else."
    ),
)

workflow = GroupChatBuilder(
    participants=[researcher, critic, writer],
    orchestrator_agent=orchestrator,
    max_rounds=6,
).build()

agent = workflow.as_agent(name="group_chat")
