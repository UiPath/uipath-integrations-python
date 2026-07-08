# Quickstart Agent

A currency conversion assistant built with the Claude Agent SDK. Demonstrates a custom in-process tool (SDK MCP server) and structured output (`output_format` json_schema). LLM calls route through the UiPath LLM Gateway.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  agent(agent)
  agent_tools(tools)
  __end__(__end__)
  agent --> agent_tools
  agent_tools --> agent
  __start__ --> |input|agent
  agent --> |output|__end__
```

## Run

```
uipath auth
uipath init
uipath run agent '{"input": "Convert 100 EUR to RON"}'
```

## Debug

```
uipath dev
```
