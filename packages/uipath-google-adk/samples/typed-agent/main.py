"""Google ADK agent with strongly-typed input/output and the formatter pattern.

Demonstrates the "formatter pattern" required when combining structured output
with tool usage:

  SequentialAgent → [researcher (tools + output_key), formatter (output_schema)]

IMPORTANT — Google ADK / Gemini API constraint:
  output_schema sets response_mime_type='application/json', which is INCOMPATIBLE
  with function calling (tools or sub_agents). To get structured output from an
  agent that uses tools, use the formatter pattern:
  - The researcher does the heavy lifting with tools, stores results via output_key
  - The formatter reads the results and produces structured JSON via output_schema
"""

import httpx
from google.adk.agents import Agent, SequentialAgent
from pydantic import BaseModel, Field


class ResearchInput(BaseModel):
    """Structured input for the research agent."""

    topic: str = Field(description="The topic to research")
    max_sources: int = Field(default=5, description="Maximum number of sources to use")
    language: str = Field(default="en", description="Language for the research output")


class ResearchOutput(BaseModel):
    """Structured output from the research agent."""

    summary: str = Field(description="A concise summary of the research findings")
    key_points: list[str] = Field(description="Key findings as bullet points")
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0
    )


def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for encyclopedic information on a topic.

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


# --- Researcher: has tools + output_key, NO output_schema ---
researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=(
        "You are a research specialist. Use the search_wikipedia tool to find "
        "information about the given topic. Provide a thorough summary of "
        "your findings."
    ),
    tools=[search_wikipedia],
    input_schema=ResearchInput,
    output_key="research_results",
)

# --- Formatter: has output_schema, NO tools ---
formatter = Agent(
    name="formatter",
    model="gemini-2.5-flash",
    instruction=(
        "You are a research formatter. Take the research results from the "
        "previous step and format them into a structured research summary "
        "with key points and a confidence score (0.0 to 1.0 based on source "
        "quality and coverage). Output valid JSON matching the schema."
    ),
    output_schema=ResearchOutput,
    output_key="output",
)

# --- Root: SequentialAgent pipeline ---
#
# Schema resolution (handled by the runtime recursively):
#   - input_schema:  from FIRST sub_agent → researcher.input_schema (ResearchInput)
#   - output_schema: from LAST sub_agent  → formatter.output_schema (ResearchOutput)
#   - output_key:    from LAST sub_agent  → formatter.output_key ("output")
agent = SequentialAgent(
    name="pipeline",
    sub_agents=[researcher, formatter],
)
