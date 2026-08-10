# UiPath Claude Agent SDK

A Python SDK that enables developers to build and deploy [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) agents to the UiPath Cloud Platform.

You define an agent in Python, point a `claude.json` file at it, and the UiPath runtime discovers it, generates its entry points, and executes it. Routing model calls through the UiPath LLM Gateway is an explicit opt-in, described under [Model routing](#model-routing).

This package is an extension to the [UiPath Python SDK](https://github.com/UiPath/uipath-python) and implements the [UiPath Runtime Protocol](https://github.com/UiPath/uipath-runtime-python).

Check out these [sample projects](https://github.com/UiPath/uipath-integrations-python/tree/main/packages/uipath-claude-sdk/samples) to see the SDK in action.

## Requirements

- Python 3.11 or higher
- UiPath Automation Cloud account

## Installation

```bash
pip install uipath-claude-sdk
```

using `uv`:

```bash
uv add uipath-claude-sdk
```

## Quickstart

```bash
uipath auth                 # authenticate against UiPath
uipath new my-agent         # scaffold main.py, claude.json and pyproject.toml
uipath init                 # generate uipath.json, entry-points.json, bindings.json and agent.mermaid
uipath run agent '{"city": "London"}'
```

`uipath new` writes into the current directory. The scaffolded agent is a weather agent with typed input and output, and its only input field is `city`, which is why the run command above passes `{"city": "London"}`:

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from pydantic import BaseModel, Field

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel


@tool(
    "get_weather",
    "Get the current weather reading for a city, e.g. London.",
    {"city": str},
)
async def get_weather(args: dict) -> dict:
    ...


weather_server = create_sdk_mcp_server(
    name="weather",
    tools=[get_weather],
)


class WeatherRequest(BaseModel):
    city: str = Field(description="City to report the weather for, e.g. 'London'.")


class WeatherReport(BaseModel):
    city: str = Field(description="City the reading refers to.")
    temperature_celsius: float = Field(description="Temperature returned by the tool.")
    summary: str = Field(description="One sentence description of the weather.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You are a weather assistant. "
            "Always call the get_weather tool before answering, and report only "
            "the values it returns."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"weather": weather_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=WeatherRequest,
    output_schema=WeatherReport,
    prompt="Report the current weather in {city}.",
)
```

The `get_weather` tool is defined with `@tool` and served in process through `create_sdk_mcp_server`, which is how you attach custom Python tools to an agent without running an external MCP process.

Anything the Claude Agent SDK accepts on `ClaudeAgentOptions` (system prompt, allowed tools, MCP servers, `output_format`, ...) is yours to set. The runtime only injects execution-scoped fields (`cwd`, `env`, `resume`, `model` when `uipath_llm` is set, and `output_format` when an `output_schema` is declared) and leaves the rest of your configuration untouched.

An agent is what its code says. No tools, no MCP servers and no prompt text are added on your behalf, so an agent written for the Claude Agent SDK runs on UiPath unchanged, and an agent written here runs anywhere else. Everything UiPath-specific, model routing and human in the loop alike, is an explicit opt-in.

## The agent definition

`UiPathClaudeAgent` wraps the SDK options with the pieces the UiPath runtime needs:

| Field | Purpose |
|-------|---------|
| `options` | The `ClaudeAgentOptions` passed to the Claude Agent SDK |
| `uipath_llm` | Optional `UiPathModel`. Set it to route model calls through the UiPath LLM Gateway. Left unset, the Claude Agent SDK reaches Anthropic on its own credentials |
| `input_schema` | Optional Pydantic model. Input is validated against it before the run |
| `output_schema` | Optional Pydantic model. The runtime requests native structured output and validates the result against it |
| `prompt` | Optional user-message template rendered with `str.format` from the validated input, for example `"Report the current weather in {city}"` |
| `name` | Display name used in the runtime graph |

`ClaudeAgent` is kept as an alias of `UiPathClaudeAgent`, so existing agents keep working unchanged.

The schemas are what `uipath init` writes into `entry-points.json`, so they also drive the input and output contract shown in the UiPath platform.

Exporting a bare `ClaudeAgentOptions` also works. The loader wraps it in a `UiPathClaudeAgent` with defaults, in which case the entry point falls back to a single `input` string in and a single `result` string out, unless you set `options.output_format` yourself. An agent exported this way has no `uipath_llm`, so it runs on the direct Anthropic path.

## Model routing

`uipath_llm` decides where model traffic goes. There is no automatic fallback between the two paths.

### With `uipath_llm` set

```python
from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(system_prompt="..."),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
)
```

`UiPathModel` is a routing descriptor, not a client. It holds no connection, opens no socket and is not callable. It records which model to use and how to route to it, and the runtime reads it when it builds the environment for the Claude Agent SDK subprocess.

On this path the runtime starts a local gateway shim, binds a loopback listener, and injects `ANTHROPIC_BASE_URL` and a per-run secret so the SDK talks to the shim instead of Anthropic. The shim forwards every call to your tenant, which is why this path needs `UIPATH_URL` and `UIPATH_ACCESS_TOKEN` (see [Environment variables](#environment-variables)).

Write a plain model id. The shim resolves it against the tenant's model discovery, so you do not need to know the id the tenant actually serves. It matches, in order:

- the exact name discovery reported
- its lowercase form
- the same name with an `anthropic.` prefix stripped
- the same name with a `-YYYYMMDD-vN:N` version suffix or an `@version` suffix stripped
- the family words `opus`, `sonnet` and `haiku`, which resolve to the newest matching model

`options.model` is overwritten with the resolved id on this path, so setting it alongside `uipath_llm` has no effect. The auxiliary models the Claude Code CLI picks for background work are pinned to catalog-resolved ids too, so no traffic escapes to a default model name the tenant does not host.

`UiPathModel` takes the model name and nothing else. How to reach that model is discovery's to report, not yours to declare, which is what keeps a name that works in `uipath-langchain` working here.

### Without `uipath_llm`

No shim starts and no LLM environment is injected. The Claude Agent SDK reaches Anthropic on whatever credentials it finds for itself, so set `ANTHROPIC_API_KEY` and put a plain Anthropic model id on the options:

```python
agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(model="claude-sonnet-4-5", system_prompt="..."),
)
```

Usage on this path is billed to your Anthropic account.

## Human in the loop

Suspending a run is opt-in and lives in your own tools. Nothing is injected: an agent that never calls `interrupt()` has no UiPath surface at all.

Write an ordinary SDK tool, call `interrupt()` inside it with a model imported from `uipath.platform.common`, and register the tool through `uipath_tool_server`:

```python
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, tool
from uipath.platform.common import CreateTask

from uipath_claude_sdk import UiPathClaudeAgent, interrupt, uipath_tool_server


@tool(
    "review_classification",
    "Get a human to review the classification.",
    {"ticket_id": str, "label": str},
)
async def review_classification(args: dict[str, Any]) -> dict[str, Any]:
    action_data = await interrupt(
        CreateTask(
            app_name="escalation_agent_app",
            title="Action Required: Review classification",
            data={"AgentOutput": f"Classified {args['ticket_id']} as {args['label']}"},
            app_folder_path="Shared",
        )
    )
    approved = action_data.get("Answer") is True
    return {"content": [{"type": "text", "text": f"Approved: {approved}"}]}


tickets_server = uipath_tool_server("tickets", tools=[review_classification])

agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(mcp_servers={"tickets": tickets_server}),
)
```

`uipath_tool_server` returns a plain `McpSdkServerConfig`. Register it under whatever key you like, and the model sees the tools as `mcp__<key>__<tool-name>`. Tools that never suspend belong in a normal `create_sdk_mcp_server`.

### The value decides the trigger

Whatever you pass to `interrupt()` reaches the platform untouched, and the platform picks the resume trigger from its type:

| Value | Resume trigger |
|-------|----------------|
| `CreateTask`, `WaitTask`, `CreateEscalation`, `WaitEscalation` | Action Center task |
| `InvokeProcess`, `WaitJob` | Orchestrator job |
| `WaitIntegrationEvent` | Integration Services event |
| `WaitUntil` | Timer |
| Anything else, including a plain string | API trigger, answered by the resume input |

A list or tuple becomes sibling triggers under one interrupt, resolved by whichever fires first. Every current and future model in `uipath.platform.common` works without any change in this package.

`interrupt()` returns whatever the fired trigger carried back: the completed action's data, the finished job's output, the delivered event's payload, or the JSON passed to `uipath run agent ... --resume`. Mapping that onto the result the model reads is your tool's job.

### The tool body runs twice

A tool handler cannot pause across a process boundary, so the body is executed by a `PreToolUse` hook instead. On the suspend pass `interrupt()` raises, the hook parks the call and the run reports SUSPENDED. On resume the same body runs again in a fresh process, `interrupt()` returns the resolved payload, and the body's return value is delivered as the parked call's tool result.

This is the same replay a LangGraph node does, with the same caveat: **work performed before the `interrupt()` call happens twice.** Keep side effects after the call, or make them idempotent.

Two further constraints, both measured against the Claude Code CLI:

- Only one interrupt can be in flight per run. A second suspending call while one is pending is denied and the model is told to retry it on its own.
- A suspending tool call must be the last tool call of its turn. Say so in the system prompt.

Outside a UiPath run the tools simply execute their bodies, and `interrupt()` raises `InterruptOutsideRunError`.

## Configuration

### claude.json

`claude.json` maps an entry-point name to the module attribute holding the agent definition, in `file.py:variable` form:

```json
{
  "agents": {
    "agent": "main.py:agent"
  }
}
```

Every key under `agents` becomes an entry point, so a project can expose several agents from one package:

```json
{
  "agents": {
    "researcher": "research.py:agent",
    "reviewer": "review.py:agent"
  }
}
```

`uipath init` reads this file and produces the `uipath.json`, `entry-points.json` and `bindings.json` that packaging and deployment need. For more details on the configuration format, see the [UiPath configuration specifications](https://github.com/UiPath/uipath-python/blob/main/specs/README.md).

### Environment variables

`uipath auth` writes the UiPath credentials into a `.env` file in your project root:

```
UIPATH_URL=https://cloud.uipath.com/ACCOUNT_NAME/TENANT_NAME
UIPATH_ACCESS_TOKEN=YOUR_TOKEN_HERE
UIPATH_TENANT_ID=YOUR_TENANT_ID
UIPATH_ORGANIZATION_ID=YOUR_ORGANIZATION_ID
```

An agent that declares `uipath_llm` uses those credentials to resolve its model and to route every call. An agent without `uipath_llm` needs `ANTHROPIC_API_KEY` instead for model access, and still needs the UiPath credentials to publish.

### Conversational agents

Set `runtimeOptions.isConversational` in `uipath.json` to run the agent as a multi-turn chat agent:

```json
{
  "runtimeOptions": {
    "isConversational": true
  }
}
```

The conversational runtime handles one exchange per invocation. It streams the assistant response as UiPath conversation message events, then suspends with an API resume trigger to wait for the next user message. The Claude session id is persisted so the SDK re-attaches to the same session, and a stable per-conversation workspace directory keeps files written by the agent across exchanges. Input and output then use the UiPath conversation message format rather than the agent's own input and output schemas.

## Command line interface

The `uipath` CLI commands all work against a Claude SDK project:

| Command | Purpose |
|---------|---------|
| `uipath auth` | Authenticate and populate `.env` |
| `uipath new NAME` | Scaffold a new Claude SDK agent project in the current directory |
| `uipath init` | Generate `uipath.json`, `entry-points.json`, `bindings.json` and the mermaid graph from `claude.json` |
| `uipath run AGENT [INPUT]` | Execute an agent locally with JSON input, inline or with `-f/--file` |
| `uipath pack` | Package the project into a `.nupkg` |
| `uipath publish` | Publish the package to Orchestrator |

`uipath pack` requires a `description` field and author information in `pyproject.toml`.

## Project structure

To use the CLI for packaging and publishing, your project should include:

- A `pyproject.toml` file with project metadata
- A `claude.json` file with your agent definitions (for example `"agents": {"agent": "main.py:agent"}`)
- A `uipath.json` file (generated by `uipath init`)
- An `entry-points.json` file (generated by `uipath init`)
- A `bindings.json` file (generated by `uipath init`) to configure resource overrides
- Any Python files needed for your agent

## Not yet supported

These are not implemented in this package today. Do not rely on them yet.

- Tool approval. A tool call is never routed to a human for permission. What is supported is the developer-driven suspension described under [Human in the loop](#human-in-the-loop)
- Evaluations
- Interactive debugging

## Development

### Developer tools

Check out [uipath-dev](https://github.com/uipath/uipath-dev-python), an interactive application for building, testing, and debugging UiPath Python runtimes, agents, and automation scripts.

### Setting up a development environment

Please read our [contribution guidelines](https://github.com/UiPath/uipath-integrations-python/blob/main/packages/uipath-claude-sdk/CONTRIBUTING.md) before submitting a pull request.

### Special thanks

A huge thank-you to the open-source community and the maintainers of the libraries that make this project possible:

- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) for the agent harness this package builds on.
- [Pydantic](https://github.com/pydantic/pydantic) for reliable, typed configuration and validation.
