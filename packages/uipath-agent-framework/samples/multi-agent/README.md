# Multi-Agent

A coordinator agent that delegates to specialist agents (research + code execution) using the Agent Framework's `as_tool()` pattern.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  coordinator(coordinator)
  research_agent(research_agent)
  research_agent_tools(tools)
  code_agent(code_agent)
  code_agent_tools(tools)
  __end__(__end__)
  research_agent --> research_agent_tools
  research_agent_tools --> research_agent
  coordinator --> research_agent
  research_agent --> coordinator
  code_agent --> code_agent_tools
  code_agent_tools --> code_agent
  coordinator --> code_agent
  code_agent --> coordinator
  __start__ --> |input|coordinator
  coordinator --> |output|__end__
```

## Prerequisites

Authenticate with UiPath to configure your `.env` file:

```bash
uipath auth
```

## Run

```
uipath run agent '{"messages": [{"contentParts": [{"data": {"inline": "What is the population of France and calculate its square root?"}}], "role": "user"}]}'
```

## Debug

```
uipath dev web
```
