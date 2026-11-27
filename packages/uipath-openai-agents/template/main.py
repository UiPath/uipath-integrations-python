from datetime import datetime, timezone

from agents import Agent, function_tool
from agents.models import _openai_shared
from pydantic import BaseModel

from uipath_openai_agents.chat import UiPathChatOpenAI
from uipath_openai_agents.chat.supported_models import OpenAIModels

MODEL = OpenAIModels.gpt_4_1_mini_2025_04_14

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's query using the available tools when needed. "
    "Be concise and informative."
)

MAX_RESPONSE_LENGTH = 5000

WEATHER_DATA = {
    "paris": "Weather in Paris, France: 18°C, wind 12 km/h, partly cloudy",
    "london": "Weather in London, UK: 14°C, wind 20 km/h, overcast",
    "new york": "Weather in New York, USA: 22°C, wind 8 km/h, clear sky",
    "tokyo": "Weather in Tokyo, Japan: 26°C, wind 5 km/h, sunny",
    "sydney": "Weather in Sydney, Australia: 19°C, wind 15 km/h, light rain",
}


class Output(BaseModel):
    response: str


@function_tool
def get_current_time() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


@function_tool
def get_weather(city: str, utc_time: str) -> str:
    """Get the current weather for a city. Requires the current UTC time from get_current_time.

    Args:
        city: The city name, e.g. 'Paris' or 'Tokyo'.
        utc_time: The current UTC time.
    """
    weather = WEATHER_DATA.get(city.lower().strip())
    if weather:
        return f"{weather} (as of {utc_time})"
    return f"Weather data not available for {city}"


def main() -> Agent:
    """Configure UiPath OpenAI client and return the agent."""
    uipath_openai_client = UiPathChatOpenAI(model_name=MODEL)
    _openai_shared.set_default_openai_client(uipath_openai_client.async_client)

    agent = Agent(
        name="assistant",
        instructions=SYSTEM_PROMPT,
        model=MODEL,
        tools=[get_current_time, get_weather],
        output_type=Output,
    )

    return agent
