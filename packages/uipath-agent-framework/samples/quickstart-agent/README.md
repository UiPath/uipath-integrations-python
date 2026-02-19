# Quickstart Agent

A simple weather assistant that fetches current weather data for any location using the Open-Meteo API. Uses LLMs through the UiPath LLM Gateway.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  weather_agent(weather_agent)
  weather_agent_tools(tools)
  __end__(__end__)
  weather_agent --> weather_agent_tools
  weather_agent_tools --> weather_agent
  __start__ --> |input|weather_agent
  weather_agent --> |output|__end__
```

## Run

```
uipath run agent '{"messages": [{"contentParts": [{"data": {"inline": "What is the weather in San Francisco?"}}], "role": "user"}]}'
```

## Debug

```
uipath dev web
```
