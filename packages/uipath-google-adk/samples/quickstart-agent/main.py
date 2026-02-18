import httpx
from google.adk.agents import Agent

from uipath_google_adk.chat import UiPathGemini

# from uipath_google_adk.chat import UiPathOpenAI
# from uipath_google_adk.chat import UiPathAnthropic


def get_weather(location: str) -> str:
    """Get the current weather for a location using the Open-Meteo API.

    Args:
        location: The city name, e.g. "San Francisco"

    Returns:
        JSON string with temperature, wind speed, and conditions
    """
    # Geocode the location name to lat/lon via Open-Meteo
    try:
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()
        if not geo.get("results"):
            return f"Could not find location: {location}"
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0].get("name", location)
        country = geo["results"][0].get("country", "")
    except Exception as e:
        return f"Geocoding failed for '{location}': {e}"

    # Fetch current weather
    try:
        weather_resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "temperature_unit": "celsius",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})
        return (
            f"Weather in {name}, {country}: "
            f"{current.get('temperature_2m')}°C, "
            f"wind {current.get('wind_speed_10m')} km/h, "
            f"code {current.get('weather_code')}"
        )
    except Exception as e:
        return f"Weather fetch failed: {e}"


agent = Agent(
    name="weather_agent",
    model=UiPathGemini(model="gemini-2.5-flash"),
    # model=UiPathOpenAI(model="gpt-4o-mini-2024-07-18"),
    # model=UiPathAnthropic(model="anthropic.claude-haiku-4-5-20251001-v1:0"),
    instruction="You are a helpful weather assistant. Use the get_weather tool to provide weather information.",
    tools=[get_weather],
)
