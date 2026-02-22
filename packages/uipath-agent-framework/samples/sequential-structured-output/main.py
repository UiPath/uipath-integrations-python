from agent_framework.orchestrations import SequentialBuilder
from pydantic import BaseModel

from uipath_agent_framework.chat import UiPathOpenAIChatClient


class CityInfo(BaseModel):
    """Structured output for city information."""

    city: str
    country: str
    description: str
    population_estimate: str
    famous_for: list[str]


client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

researcher = client.as_agent(
    name="researcher",
    description="Researches factual information about a city.",
    instructions=(
        "You are a thorough researcher. Given a city name, gather key facts "
        "including its country, population, notable landmarks, cultural "
        "significance, and what it is famous for. Present your findings clearly."
    ),
)

editor = client.as_agent(
    name="editor",
    description="Edits research into a structured city profile.",
    instructions=(
        "You are a precise editor. Take the researcher's findings and organize "
        "them into a well-structured city profile. Ensure all facts are accurate "
        "and the description is concise and informative."
    ),
    default_options={"response_format": CityInfo},
)

workflow = SequentialBuilder(
    participants=[researcher, editor],
).build()

agent = workflow.as_agent(name="sequential_structured_output_workflow")
