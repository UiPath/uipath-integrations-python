"""Claude Agent SDK weather agent with a custom in-process tool."""

import json

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from pydantic import BaseModel, Field

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

READINGS = {
    "london": {"temperature_celsius": 14.0, "conditions": "overcast"},
    "paris": {"temperature_celsius": 18.0, "conditions": "partly cloudy"},
    "new york": {"temperature_celsius": 22.0, "conditions": "clear sky"},
    "tokyo": {"temperature_celsius": 26.0, "conditions": "sunny"},
    "sydney": {"temperature_celsius": 19.0, "conditions": "light rain"},
}


@tool(
    "get_weather",
    "Get the current weather reading for a city, e.g. London.",
    {"city": str},
)
async def get_weather(args: dict) -> dict:
    reading = READINGS.get(args["city"].lower().strip())
    if reading is None:
        text = f"No weather reading available for {args['city']}."
    else:
        text = json.dumps(reading)
    return {"content": [{"type": "text", "text": text}]}


weather_server = create_sdk_mcp_server(
    name="weather",
    tools=[get_weather],
)


class WeatherRequest(BaseModel):
    city: str = Field(description="City to report the weather for, e.g. 'London'.")


class WeatherReport(BaseModel):
    city: str = Field(description="City the reading refers to.")
    temperature_celsius: float = Field(description="Temperature returned by the tool.")
    summary: str = Field(description="One sentence description of the weather.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You are a weather assistant. "
            "Always call the get_weather tool before answering, and report only "
            "the values it returns."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"weather": weather_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=WeatherRequest,
    output_schema=WeatherReport,
    prompt="Report the current weather in {city}.",
)
