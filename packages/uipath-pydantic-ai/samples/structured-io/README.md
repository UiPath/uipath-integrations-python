# Structured I/O

A research agent with strongly-typed input and output using Pydantic models. Demonstrates `deps_type` for structured input and `output_type` for structured output, with tools that access the input via `RunContext`.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  research_agent(research_agent)
  research_agent_tools(tools)
  __end__(__end__)
  research_agent --> research_agent_tools
  research_agent_tools --> research_agent
  __start__ --> |input|research_agent
  research_agent --> |output|__end__
```

## Run

```
uipath run agent '{"topic": "artificial intelligence", "max_sources": 3, "language": "en"}'
```

## Debug

```
uipath dev web
```
