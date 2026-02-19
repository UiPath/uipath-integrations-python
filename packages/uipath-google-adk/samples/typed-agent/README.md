# Typed Agent

A two-stage sequential agent that researches a topic via Wikipedia and produces structured JSON output using the formatter pattern.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  pipeline(pipeline)
  researcher(researcher)
  researcher_tools(tools)
  formatter(formatter)
  __end__(__end__)
  researcher --> researcher_tools
  researcher_tools --> researcher
  pipeline --> researcher
  researcher --> pipeline
  pipeline --> formatter
  formatter --> pipeline
  __start__ --> |input|pipeline
  pipeline --> |output|__end__
```

## Run

```
uipath run agent '{"topic": "artificial intelligence", "max_sources": 3, "language": "en"}'
```

## Debug

```
uipath dev web
```
