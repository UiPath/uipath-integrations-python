import json
import urllib.parse
import urllib.request

from google.adk.agents import Agent


def get_weather(location: str) -> str:
    """Get the current weather for a location using the Open-Meteo API.

    Args:
        location: The city name, e.g. "San Francisco"

    Returns:
        JSON string with temperature, wind speed, and conditions
    """
    # Geocode the location name to lat/lon via Open-Meteo
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": location, "count": 1})
    )
    try:
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read().decode())
        if not geo.get("results"):
            return f"Could not find location: {location}"
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0].get("name", location)
        country = geo["results"][0].get("country", "")
    except Exception as e:
        return f"Geocoding failed for '{location}': {e}"

    # Fetch current weather
    weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,weather_code",
            "temperature_unit": "celsius",
        }
    )
    try:
        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        current = data.get("current", {})
        return json.dumps(
            {
                "location": f"{name}, {country}",
                "temperature_celsius": current.get("temperature_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
            }
        )
    except Exception as e:
        return f"Weather fetch failed: {e}"


agent = Agent(
    name="weather_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful weather assistant. Use the get_weather tool to provide weather information.",
    tools=[get_weather],
)
