# Quickstart

The smallest Codex agent: one prompt template, no tools. The model call goes
through the UiPath LLM Gateway with your tenant's models; no OpenAI API key is
needed.

```python
agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
)
```

## Run locally

```bash
cd samples/quickstart-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json   # {"topic": "robots"}
```

The output is `{"output": "<the joke>"}`.

## Run on Orchestrator

```bash
uv run uipath pack
uv run uipath publish --my-workspace
```

Start a job with `{"topic": "robots"}`.

## Trace

```
Agent run - agent
├─ LLM call → Model run
└─ Agent output
```
