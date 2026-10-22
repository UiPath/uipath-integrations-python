# Durable process tool

A Codex agent that calls an Orchestrator **process** as a tool and **suspends**
while the job runs. When the job finishes, the run resumes (possibly on another
robot) and the model receives the job's output as the result of its original
tool call, then finishes the task.

The process is `a_plus_b`: input arguments `a` and `b` (integers), output `c`
(the sum). The agent asks for two sums in parallel, so one run starts two jobs and
waits for both.

```python
uipath_process_tool(
    "a_plus_b",
    folder_path="Coded/a_plus_b",   # omit → the folder the agent runs in
    description="Adds two integers a and b in a UiPath process.",
    input_schema=AddInput,  # a: int, b: int
)
```

## How it works

1. The model calls `a_plus_b`. The runtime serves durable tools to Codex from
   its own MCP endpoint (`uipath_tools`), so the call reaches the runtime. It
   validates the arguments and starts the job with
   `processes.invoke_async`.
2. The call is never answered, so the Codex turn stays blocked. When nothing
   else is running, the runtime stops Codex, saves the conversation (the Codex
   rollout) in `__uipath/state.db`, and the run ends **SUSPENDED** with one
   `JOB_RAW` resume trigger per job.
3. On resume, each finished job becomes the result of its call:
   * a successful job: its output arguments, as JSON;
   * a faulted job: `{"status": "faulted", "error": ..., "job_key": ...}`. The
     model sees the error and the run continues;
   * a stopped job: `{"status": "stopped", ...}`.

   The runtime then restores the conversation, attaches the results to the
   original call ids, and lets the model continue.
4. If a job is still running on resume, the run suspends again on it.

## Prerequisites

* An `a_plus_b` process deployed in an Orchestrator folder: input arguments
  `a`, `b` (Int32) and an output argument `c` (Int32) with the sum. If yours
  is named differently, change `AddOutput` in `main.py` to match.
* `uipath auth` against the tenant. Set `folder_path` in `main.py` (and
  `folderPath` in `bindings.json`) to the folder of your `a_plus_b` release.
  How the folder is chosen, like the other UiPath integrations:
  1. `folder_path` on the tool, when given;
  2. otherwise the folder the agent runs in (`UIPATH_FOLDER_PATH`, set by
     Orchestrator for a deployed job; set it in `.env` only for local runs);
  3. at deploy time, a resource overwrite on the `bindings.json` entry can
     point the tool at another folder.

```bash
cd samples/durable-process-tool
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
# .env is written by `uipath auth`; nothing else is required
```

## Local run against a real tenant

```bash
# 1. Run: starts two a_plus_b jobs and suspends.
uv run uipath run agent --file input.json
```

Expect two `a_plus_b` jobs in Orchestrator and a SUSPENDED result. The triggers
are in the state database:

```bash
sqlite3 __uipath/state.db "select interrupt_id, data from __uipath_resume_triggers"
```

There is one row per job: `trigger_name` is `JobRaw`, `item_key` is the child
job key and `interrupt_id` is the model's tool `call_id`. If both jobs finish
while the run is still polling, the run may resume in-process and complete
straight away. That's also a pass.

```bash
# 2. Once the jobs have finished: resume. The triggers are read from
#    state.db, the job outputs are handed to the model, and it reports both sums.
uv run uipath run agent --resume

# Resuming while a job is still running suspends again on the same jobs.
uv run uipath run agent --resume
```

Don't delete `__uipath/` between the run and the resume. A fresh run (without
`--resume`) resets `__uipath/state.db` and `__uipath/codex_home`.

### Local run without Orchestrator jobs finishing

You can answer a single pending call by hand. The payload becomes the tool
result:

```bash
uv run uipath run agent '{"sum": 42}' --resume
```

With several pending calls, key the payload by call id, e.g.
`'{"call_abc...": {"sum": 42}}'`. The call ids are the `interrupt_id` column.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

No Orchestrator job is started: `a_plus_b` is mocked by its process name. The
mockito case matches each call on its arguments (`{"a": 17, "b": 25}` and
`{"a": 42, "b": 8}`); the LLM case answers from `AddOutput`.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Deployed

```bash
uv run uipath pack
uv run uipath publish        # to the folder that holds a_plus_b
```

Start the agent job with `input.json` as input. Expect:

* the parent job goes to **Suspended**, with the child `a_plus_b` jobs as its
  resume triggers;
* when the children finish, Orchestrator resumes the parent with the same job
  key, possibly on another robot;
* the job completes with both sums.

The traces should link each child job to the parent's span.

## Notes and limits

* Only tools declared in `tools=[...]` suspend. Shell commands and other MCP
  servers run as usual. If one of them is called in the same model response as
  a durable tool, the suspend waits for it (up to `durable_settle_timeout`,
  120 s by default) and its result is handed back on resume.
* A run waits for **all** pending jobs before the model continues.
* Errors that stop a job from starting (process not found, folder missing,
  409...) are returned to the model as a tool error, not raised.
* The Codex CLI is pinned to 0.161.x: the suspend relies on its rollout and
  `thread/inject_items` behaviour.
