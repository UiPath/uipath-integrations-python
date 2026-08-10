# Samples

Each sample is a standalone project demonstrating one capability. They resolve
`uipath-claude-sdk` from this checkout through `[tool.uv.sources]`, so what you run is the code in
this worktree, not a published release.

## Two ways to run

**With a UiPath tenant.** The sample as written declares `uipath_llm=UiPathModel("claude-sonnet-4-5")`,
so model calls route through the UiPath LLM Gateway and the model id is resolved against your tenant's
catalogue.

```bash
cd samples/quickstart-agent
uv sync
uv run uipath auth
uv run uipath init
uv run uipath run agent --file input.json
```

**Without a tenant, using your own Anthropic key.** Delete the `uipath_llm=` line from `main.py` and
export a key. Nothing UiPath-specific is injected in that case, so the Claude Agent SDK reaches
Anthropic directly. The samples set no `model` on `ClaudeAgentOptions`, so the CLI picks its own
default. Add `model="claude-sonnet-4-5"` to the options if you want to choose.

```bash
cd samples/quickstart-agent
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run uipath init
uv run uipath run agent --file input.json
```

The second path is the quickest way to see an agent actually run. Suspension still works on it, because
API and timer triggers are created locally without calling the platform.

## Nothing is injected

An agent is what its code says. The runtime adds no tools, no MCP server and no system prompt of its
own, so an agent written for the Claude Agent SDK runs on UiPath unchanged.

Human in the loop is something you add. Write an ordinary SDK tool, call `interrupt()` inside it, and
register the tool through `uipath_tool_server`:

```python
from uipath.platform.common import CreateTask
from uipath_claude_sdk import interrupt, uipath_tool_server

@tool("request_approval", "Ask a human to approve it.", {"question": str})
async def request_approval(args: dict[str, Any]) -> dict[str, Any]:
    action_data = await interrupt(
        CreateTask(
            title="Action Required",
            data={"AgentOutput": args["question"]},
            app_name="generic_escalation_app",
            app_folder_path="Shared",
        )
    )
    approved = action_data.get("Answer") is True
    return {"content": [{"type": "text", "text": f"Approved: {approved}"}]}


server = uipath_tool_server("approvals", tools=[request_approval])
options = ClaudeAgentOptions(mcp_servers={"approvals": server})
```

The value passed to `interrupt()` decides the resume trigger: `CreateTask` means an Action Center
task, `InvokeProcess` or `WaitJob` mean a job, `WaitIntegrationEvent` means an Integration Services
event, `WaitUntil` means a timer, and anything else, including a plain string, becomes an API trigger
answered by the resume input. A list becomes sibling triggers resolved by whichever fires first.

An agent that never calls `interrupt()` gets no UiPath surface at all.

## What each sample needs

| Sample | Shows | Anthropic key only | Needs a tenant |
| --- | --- | --- | --- |
| `quickstart-agent` | Typed input and output, one in-process tool | Yes | |
| `autonomous-agent` | Built-in file tools, multi-step work | Yes | |
| `chat-agent` | `isConversational`, a multi-turn conversation across invocations | Yes | |
| `simple-local-mcp-agent` | A stdio MCP server as a subprocess | Yes | |
| `simple-remote-mcp-agent` | A remote MCP server over HTTP | | An MCP server endpoint |
| `simple-hitl-agent` | `interrupt(question)`, an API trigger, resume in a new process | Yes | |
| `action-center-hitl-agent` | `interrupt(CreateTask(...))` | | Action Center app |
| `multi-agent` | `interrupt(InvokeProcess(...))`, delegating to another process | | A published process |
| `integration-event-agent` | `interrupt(WaitIntegrationEvent(...))` | | An Integration Services connection |

The last three suspend on triggers the platform has to create, so they fail at suspend time without
credentials. `simple-hitl-agent` does not, because its plain-string interrupt becomes an API trigger
the runtime creates locally.

## Seeing a suspension

`simple-hitl-agent` is the one to try, and it needs nothing but an Anthropic key:

```bash
cd samples/simple-hitl-agent
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run uipath init
uv run uipath run agent --file input.json          # suspends, prints the question
uv run uipath run agent '{"answer": "yes"}' --resume
```

The first command ends with the job suspended: the model called the sample's own
`mcp__approvals__ask_approver`, a `PreToolUse` hook ran its body, `interrupt()` raised out of it, the
CLI parked the call, and the runtime stored the pending record in `__uipath/state.db`. The second
command starts a **new process**, re-attaches to the Claude session, runs the same body again with
`interrupt()` returning your answer, and hands the body's return value back as that parked call's own
tool result, so the agent continues mid-turn rather than being told about it afterwards.

Because the body reruns, work performed **before** the `interrupt()` call happens twice. That is the
same replay LangGraph does in a node. Keep side effects after the call, or make them idempotent.

Look at `__uipath/state.db` and `__uipath/claude_home/projects/*/*.jsonl` between the two commands to
see what actually persists.

## Seeing a conversation

`chat-agent` is the same suspension mechanism used for a different purpose. `isConversational` in its
`uipath.json` selects a runtime that runs one exchange per invocation and then suspends on an API
trigger waiting for the next user message, so **every turn after the first is a `--resume`**:

```bash
cd samples/chat-agent
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run uipath init
uv run uipath run agent --file input.json
uv run uipath run agent --file next_message.json --resume
```

The second message asks the agent to recall something only the first turn established. Getting it right
means the message reached the model and the Claude session was resumed, which are two separate things
that both have to work.

Nothing in that agent's code opts into any of this, and it registers no `uipath_tool_server`, so no tool
call is ever parked. The waiting between turns and the waiting for a human are the same primitive with
different triggers, and an agent can use both.

## Poking at it

The agents are plain Claude Agent SDK objects, so everything you already know applies. Things worth
trying:

- Change `system_prompt` and watch the tool-calling behaviour change.
- Add a tool with `@tool` and `create_sdk_mcp_server`, then register it in `mcp_servers`. Use
  `uipath_tool_server` instead when the tool needs to suspend.
- Swap `input_schema` and `output_schema` for your own pydantic models, re-run `uipath init`, and look
  at the regenerated `entry-points.json` and `agent.mermaid`.
- Drop `output_schema` entirely and see the run fall back to untyped output.
- In `simple-hitl-agent`, ask the model to call the suspending tool alongside another tool in the same
  turn. A suspending call has to be the last one of its turn, so this is the one thing to be careful with.
- Change what `simple-hitl-agent` passes to `interrupt()`, from the plain question string to a
  `WaitUntil(resume_time=...)`, and watch the trigger kind change with no other edit.

## Packing and deploying before the package is published

`uipath-claude-sdk` is not on PyPI yet, so a packed sample has no way to install it. The editable
`[tool.uv.sources]` entry that makes `uipath run` work locally points **outside** the project
directory, and `uipath pack` only walks the project directory, so the dependency is simply absent from
the `.nupkg` and the run fails on the executor at install time.

Vendor the library into the package instead:

```bash
./scripts/pack_sample.sh samples/quickstart-agent
```

That builds a wheel into the sample's gitignored `wheels/`, repoints `[tool.uv.sources]` at it, adds
`.whl` to `packOptions.fileExtensionsIncluded` so the packer picks binaries up, locks, packs, and puts
every file it touched back the way it was. The resulting `.nupkg` carries
`content/wheels/uipath_claude_sdk-<version>-py3-none-any.whl`, and the packed `uv.lock` refers to it by
a **relative** path, so it resolves wherever the executor extracts the package.

Doing it by hand is four edits, if you would rather see them:

```toml
# pyproject.toml
[tool.uv.sources]
uipath-claude-sdk = { path = "wheels/uipath_claude_sdk-0.1.0-py3-none-any.whl" }
```

```json
// uipath.json
{ "packOptions": { "fileExtensionsIncluded": [".whl"] } }
```

then `uv build --wheel --project ../.. --out-dir wheels`, `uv lock` and `uipath pack`.

The wheel is a snapshot. Re-run the script after every change to `src/uipath_claude_sdk`, or you will
deploy stale code. Once the package is published, delete all of this and depend on a version.

## Notes

`uipath init` writes `entry-points.json`, `bindings.json` and `agent.mermaid`, and `uipath pack` builds
a `.nupkg` under `.uipath/`. Neither needs credentials.

Tracing, evaluations and interactive debugging are not wired up yet.
