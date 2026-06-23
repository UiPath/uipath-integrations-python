"""PydanticAI agent with strongly-typed input and output.

Demonstrates:
- deps_type: Structured input as a Pydantic model (becomes the input schema)
- output_type: Structured output as a Pydantic model (validated by PydanticAI)
- Tools accessing deps via RunContext for typed dependency injection
"""

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from uipath_pydantic_ai.chat import UiPathChatOpenAI


# --- Structured Input ---
class ResearchInput(BaseModel):
    """Structured input for the research agent."""

    topic: str = Field(description="The topic to research")
    max_sources: int = Field(
        default=3, description="Maximum number of sources to include"
    )
    language: str = Field(default="en", description="Language for the research output")


# --- Structured Output ---
class ResearchOutput(BaseModel):
    """Structured output from the research agent."""

    summary: str = Field(description="A concise summary of the research findings")
    key_points: list[str] = Field(description="Key findings as bullet points")
    sources: list[str] = Field(description="Sources used for the research")
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0
    )


uipath_client = UiPathChatOpenAI(model_name="gpt-4.1-mini-2025-04-14")

agent = Agent(
    uipath_client.model,
    name="research_agent",
    deps_type=ResearchInput,
    output_type=ResearchOutput,
    instructions=(
        "You are a research specialist. Use the search_wikipedia tool to find "
        "information about the topic provided in your dependencies. "
        "Respect the max_sources limit from the input. "
        "Return a structured research summary with key points and sources."
    ),
)


@agent.tool
async def search_wikipedia(ctx: RunContext[ResearchInput], query: str) -> str:
    """Search Wikipedia for information on a topic.

    Args:
        ctx: The agent context with research input dependencies.
        query: The search query to look up on Wikipedia.

    Returns:
        Summary text from Wikipedia, or an error message if not found.
    """
    try:
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
            headers={"User-Agent": "UiPathPydanticAISample/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", query)
        extract = data.get("extract", "No summary available.")
        return f"Wikipedia — {title}: {extract}"
    except Exception as e:
        return f"Wikipedia search failed for '{query}': {e}"
