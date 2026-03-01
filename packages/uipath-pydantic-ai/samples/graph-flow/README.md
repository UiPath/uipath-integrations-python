# Graph Flow

A blog post production pipeline using `pydantic-graph` for typed state machine control flow. A writer agent drafts a post, a reviewer agent evaluates it, and the graph loops until the post is approved or max revisions are reached.

Demonstrates:
- `BaseNode[StateT]` with return-type-driven edges
- `End[T]` for typed graph termination
- Agent message history in graph state for multi-turn refinement
- `Graph.run()` for complete graph execution

## Graph Structure

```mermaid
flowchart TB
  WriteDraft --> Review
  Review -->|needs revision| WriteDraft
  Review -->|approved| End
```

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  blog_pipeline(blog_pipeline)
  blog_pipeline_tools(tools)
  __end__(__end__)
  blog_pipeline --> blog_pipeline_tools
  blog_pipeline_tools --> blog_pipeline
  __start__ --> |input|blog_pipeline
  blog_pipeline --> |output|__end__
```

## Run

```
uipath run agent '{"messages": [{"contentParts": [{"data": {"inline": "Write a blog post about the future of AI agents"}}], "role": "user"}]}'
```

## Debug

```
uipath dev web
```
