# Process tool from an agent.json resource

A Codex agent whose process tool is built from a low-code `agent.json` tool
resource, `argumentProperties` included. `resources/add_shipping.json` is the
resource as a low-code agent declares it; `process_tool_from_resource` turns
it into a durable process tool, as LangChain does for the same resource:

```python
tool = process_tool_from_resource(RESOURCE, folder_path="Coded/a_plus_b")
```

The process is `a_plus_b`: input arguments `a` and `b` (integers), output `c`
(the sum). The resource pre-fills both inputs:

| Input | `argumentProperties` | What the model sees | What the job gets |
| --- | --- | --- | --- |
| `a` | `argument`, `argumentPath: "$['order']['items']"` | optional, with a note that it is set from the agent input | `order.items` from the agent input (the model's value only when the input has none) |
| `b` | `static`, `value: 1000` | hidden | `1000` |

With the input `{"order": {"items": 7}}` the model calls `Add_Shipping` with
`{}`, the job gets `{"a": 7, "b": 1000}` and returns `{"c": 1007}`.

## How it works

1. `process_tool_from_resource` validates the resource with the SDK's
   `AgentProcessToolResourceConfig`: `type` Process, Agent, Api,
   ProcessOrchestration, Flow or Function (any case). The tool name is the
   resource `name` sanitized as in LangChain (`Add Shipping` →
   `Add_Shipping`), the process is `properties.processName`.
2. The model-facing input schema hides the static field and marks the
   argument field optional. Every `argumentProperties` entry needs
   `isSensitive` (the SDK model requires it; `agent.json` always has it).
3. On a call, the pre-filled values are merged into the model's arguments and
   the job starts with the merged input. The run suspends until the job
   finishes, then resumes with its output, as for any durable process tool
   (see `samples/durable-process-tool`).
4. `properties.folderPath` is ignored, as in low-code: the job starts in the
   folder the agent runs in (`UIPATH_FOLDER_PATH`). This sample passes
   `folder_path="Coded/a_plus_b"` so it also works from a Personal Workspace;
   omit it when the agent is deployed next to the process.

## Prerequisites

* An `a_plus_b` process deployed in an Orchestrator folder: input arguments
  `a`, `b` (Int32) and an output argument `c` (Int32) with the sum.
* `uipath auth` against the tenant. Set `PROCESS_FOLDER` in `main.py` (and
  `folderPath` in `bindings.json`) to the folder of your `a_plus_b` release.

## Run locally

```bash
cd samples/process-resource-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json   # suspends on the a_plus_b job
# ... the job finishes ...
uv run uipath run agent --resume             # → "c is 1007"
```

If the job is still running, `--resume` suspends again on it; resume once
more later.

## Run on Orchestrator

Publish the agent (`uipath pack` and `uipath publish`) and start a job with
`input.json`: the job suspends on `a_plus_b`, Orchestrator resumes it when the
child job finishes, and the job ends Successful with `c is 1007`.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with
`a_plus_b` mocked, so no Orchestrator job is started and the run never
suspends.

The mockito behavior is named after the resource (`Add Shipping`, not the
model-facing `Add_Shipping`) and matches on the merged arguments
`{"a": 7, "b": 1000}`: the mocker sees the pre-filled values, as in
LangChain.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - Add_Shipping     toolType Process, input {"a": 7, "b": 1000} (the merged arguments)
├─ LLM call
└─ Agent output
```

When mocked, `Tool call - Add_Shipping` contains a `processTool` span named
`Simulated result: a_plus_b`.
