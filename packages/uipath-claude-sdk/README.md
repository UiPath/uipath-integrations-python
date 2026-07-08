# UiPath Claude Agent SDK Integration

UiPath runtime integration for the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

Build agents with Anthropic's `claude-agent-sdk` and run them on the UiPath Platform, with LLM calls routed through the UiPath LLM Gateway by default (no Anthropic key needed) and support for both standard (job) and conversational (multi-turn chat) execution.

## Installation

```bash
pip install uipath-claude-sdk
```

## Quick Start

Export your `ClaudeAgentOptions` (system prompt, tools, MCP servers, hooks, subagents, `output_format`, ...) and the UiPath runtime executes it.

`main.py`:

```python
from claude_agent_sdk import ClaudeAgentOptions

agent = ClaudeAgentOptions(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    system_prompt="You are a helpful assistant.",
    max_turns=25,
    permission_mode="bypassPermissions",
    tools=[],
)
```

`claude.json`:

```json
{
  "agents": {
    "agent": "main.py:agent"
  }
}
```

Then:

```bash
uipath auth        # authenticate against UiPath (LLM Gateway access)
uipath init        # generate schemas
uipath run agent '{"input": "What is the capital of France?"}'
```

## Structured output (SDK-native)

Set the SDK's own `output_format` — the runtime picks it up for the agent's output schema and returns the parsed `structured_output` as the job result:

```python
agent = ClaudeAgentOptions(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    output_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
)
```

## Structured input (optional helper)

The Claude SDK takes a prompt string, so structured input needs a mapping. If you want typed input with validation and prompt templating, wrap your options in the optional `ClaudeAgent` helper:

```python
from pydantic import BaseModel
from uipath_claude_sdk import ClaudeAgent

class Input(BaseModel):
    topic: str

class Output(BaseModel):
    summary: str
    sources: list[str]

agent = ClaudeAgent(
    options=ClaudeAgentOptions(model="anthropic.claude-sonnet-4-5-20250929-v1:0"),
    input_schema=Input,
    output_schema=Output,           # sets output_format for you
    prompt="Summarize {topic}",     # rendered from validated input
)
```

## Conversational agents

Mark the project as conversational in `uipath.json`, like every other integration:

```json
{
  "runtimeOptions": {
    "isConversational": true
  }
}
```

Each exchange streams the assistant reply as UiPath conversation events and suspends with an API resume trigger. The Claude SDK session id is persisted, so the next user message resumes the same conversation (`ClaudeAgentOptions(resume=session_id)`) with a stable per-conversation workspace directory.

## LLM access

- Default: LLM calls route through the UiPath LLM Gateway via a local proxy (Anthropic SSE ↔ AWS Bedrock event-stream translation). Use Bedrock ARN-style model IDs (e.g. `anthropic.claude-sonnet-4-5-20250929-v1:0`).
- BYO key: set `ANTHROPIC_API_KEY` to bypass the proxy and call Anthropic directly. Use direct model IDs (e.g. `claude-sonnet-4-5`).

## Features

| Feature | Supported |
|---------|-----------|
| Execute | Yes |
| Streaming | Yes |
| Structured input/output | Yes |
| Conversational (multi-turn) | Yes |
| HITL tool approval | Planned (SDK `can_use_tool` defer) |
| LLM Gateway | Yes |
