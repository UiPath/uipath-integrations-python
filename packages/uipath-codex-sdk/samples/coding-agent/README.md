# Coding agent (built-in tools)

A Codex agent that writes a file and runs it with Codex's own tools (file
edits and shell commands); no UiPath tools are involved.

```python
agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="{task}",
    input_schema=CodingTask,
    sandbox="full-access",
    approval_mode="deny_all",
    base_instructions=(
        "You are a coding assistant. Keep every file you create in the "
        "./workspace folder (create it if needed). Create and run code as "
        "needed, then report what you did and the result."
    ),
)
```

`sandbox="full-access"` lets the built-in tools write files and run commands.
It is the only mode that works on a serverless robot, which does not have the
Linux features Codex's `"workspace-write"` sandbox needs. On a robot each job
runs in its own working directory, and the files it writes are discarded with
the job. Locally the agent writes into `./workspace` (git-ignored).

> **Security:** commands the agent runs inherit the job's environment,
> including `UIPATH_ACCESS_TOKEN`. Only give an agent `"full-access"` if you
> trust its inputs.

## Run locally

```bash
cd samples/coding-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json
```

`input.json` asks for `workspace/primes.py` printing the first 10 primes, run
with `python3`; the output reports the file and its exact output
(`2 3 5 7 11 13 17 19 23 29`).

## Run on Orchestrator

```bash
uv run uipath pack
uv run uipath publish --my-workspace
```

Start a job with `input.json`. The robot's Python runs the generated script.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - shell         toolType Internal, input {"command": "...cat > workspace/primes.py ... && python3 primes.py"}
├─ LLM call
└─ Agent output
```

The model decides how to split the work: one shell command that writes and
runs the file, or a `Tool call - apply_patch` (the file change) followed by a
`Tool call - shell`.
