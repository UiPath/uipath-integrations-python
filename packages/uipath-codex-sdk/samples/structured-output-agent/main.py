"""Codex agent returning structured output (validated against a Pydantic model)."""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent


class JokeInput(BaseModel):
    topic: str = Field(description="What the joke should be about.")


class Joke(BaseModel):
    setup: str = Field(description="The setup line of the joke.")
    punchline: str = Field(description="The punchline.")
    rating: int = Field(description="How funny, 1-10.")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
    input_schema=JokeInput,
    output_schema=Joke,
)
