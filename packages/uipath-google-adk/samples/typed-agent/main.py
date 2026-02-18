"""Google ADK agent with strongly-typed input and output schemas.

The input_schema and output_schema on LlmAgent define the structured
contract for this agent. UiPath uses them to infer JSON schemas
automatically — no messages fallback is needed.
"""

from google.adk.agents import Agent
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


agent = Agent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a research assistant. Given a topic, provide a structured "
        "research summary with key points and a confidence score."
    ),
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
)
