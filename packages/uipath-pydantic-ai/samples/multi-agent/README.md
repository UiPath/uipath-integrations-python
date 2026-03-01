# Multi-Agent

A coordinator agent that delegates to specialist agents (researcher + code generator) via tool calls, then combines results into a structured report.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  coordinator(coordinator)
  coordinator_tools(tools)
  __end__(__end__)
  coordinator --> coordinator_tools
  coordinator_tools --> coordinator
  __start__ --> |input|coordinator
  coordinator --> |output|__end__
```

## Run

```
uipath run agent '{"messages": [{"contentParts": [{"data": {"inline": "Write a report about machine learning"}}], "role": "user"}]}'
```

## Debug

```
uipath dev web
```
