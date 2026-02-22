from agent_framework import WorkflowBuilder
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
city_agent = client.as_agent(
    name="city_agent",
    instructions=(
        "You are a helpful agent that describes cities in a structured format. "
        "Given a city name, provide the city name, country, a brief description, "
        "an estimated population, and a list of things the city is famous for."
    ),
    default_options={"response_format": CityInfo},
)

workflow = WorkflowBuilder(start_executor=city_agent).build()
agent = workflow.as_agent(name="structured_output_workflow")
