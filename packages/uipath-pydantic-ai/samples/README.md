# UiPath PydanticAI Samples

Sample agents built with [PydanticAI](https://ai.pydantic.dev/) and [UiPath LLM providers](../src/uipath_pydantic_ai/chat/).

## Samples

| Sample | Description |
|--------|-------------|
| [quickstart-agent](./quickstart-agent/) | Single agent with tool calling: fetches live weather data for any location |
| [structured-io](./structured-io/) | Structured input/output: research agent with typed `deps_type` input and `output_type` Pydantic model |
| [multi-agent](./multi-agent/) | Agent delegation: coordinator delegates to researcher and code generator agents via tool calls |
| [programmatic-handoff](./programmatic-handoff/) | Programmatic hand-off: app code classifies requests and routes to billing or technical specialists |
| [graph-flow](./graph-flow/) | Graph-based control flow: write-review loop using `pydantic-graph` typed state machine |
