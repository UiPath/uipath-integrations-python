# Multi-Agent

A three-stage sequential agent that coordinates research (Wikipedia + Tavily), code generation, and structured report formatting.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  pipeline(pipeline)
  coordinator(coordinator)
  research_agent(research_agent)
  research_agent_tools(tools)
  code_agent(code_agent)
  code_agent_tools(tools)
  formatter(formatter)
  __end__(__end__)
  research_agent --> research_agent_tools
  research_agent_tools --> research_agent
  coordinator --> research_agent
  research_agent --> coordinator
  code_agent --> code_agent_tools
  code_agent_tools --> code_agent
  coordinator --> code_agent
  code_agent --> coordinator
  pipeline --> coordinator
  coordinator --> pipeline
  pipeline --> formatter
  formatter --> pipeline
  __start__ --> |input|pipeline
  pipeline --> |output|__end__
```

## Run

```
uipath run agent '{"topic": "machine learning", "depth": "detailed"}'
```

## Debug

```
uipath dev web
```
