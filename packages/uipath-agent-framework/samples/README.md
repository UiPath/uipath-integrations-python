# UiPath Agent Framework Samples

Sample agents built with [Agent Framework](https://github.com/microsoft/agent-framework) and [UiPath LLM providers](../src/uipath_agent_framework/chat/).

## Samples

| Sample | Description |
|--------|-------------|
| [quickstart-workflow](./quickstart-workflow/) | Single workflow agent with tool calling: fetches live weather data for any location |
| [structured-output](./structured-output/) | Structured output workflow: extracts city information and returns it as a typed Pydantic model |
| [sequential-structured-output](./sequential-structured-output/) | Sequential pipeline with structured output: researcher and editor agents produce a typed Pydantic city profile |
| [hitl-workflow](./hitl-workflow/) | Human-in-the-loop workflow: customer support with approval-gated billing and refund operations |
| [sequential](./sequential/) | Sequential pipeline: writer, reviewer, and editor agents process a task one after another |
| [concurrent](./concurrent/) | Concurrent orchestration: sentiment, topic extraction, and summarization agents analyze text in parallel |
| [handoff](./handoff/) | Handoff orchestration: customer support agents transfer control to specialists with explicit routing rules |
| [group-chat](./group-chat/) | Group chat orchestration: researcher, critic, and writer discuss a topic with an orchestrator picking speakers |
| [magentic](./magentic/) | Magentic-One orchestration: a manager dynamically coordinates researcher and analyst agents based on task progress |
