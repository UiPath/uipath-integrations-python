# Expense approval (Action Center escalation, durable)

A Codex agent that approves small expenses itself and asks a manager for
the rest, through an Action Center app:

| Tool | App | Model provides | Returns |
| --- | --- | --- | --- |
| `escalate_SimpleApprovalApp` | `SimpleApprovalApp` (folder `Coded/SimpleApprovalAgent`), assigned to `APPROVER` | `Content` (the app's input: the expense) | `{"status": "completed", "outcome": "approve"/"reject", "output": {"Comment": ...}, "task_id": ..., "assigned_to": ...}` |

The tool is **durable**: the call creates the task and the run **suspends**
until the manager completes it. A `reject` outcome ends the run
(`outcome_mapping={"reject": "end"}`). See the main README's "Action Center
escalations" section.

## Prerequisites

* An Action Center app deployed in the tenant, with:
  * an input `Content` (text);
  * an in/out field `Comment` (text);
  * outcome buttons `approve` and `reject`.

  This sample uses `SimpleApprovalApp` in the `Coded/SimpleApprovalAgent`
  folder.

  Set `APP_NAME` / `APP_FOLDER` in `main.py` and the binding in
  `bindings.json` (key `<app>.<folder>`) to your app.
* `APPROVER` in `main.py`: a user (name or email) who can see tasks in that
  folder.

## Run locally

```bash
cd samples/escalation-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json     # suspends on the task
# ... complete the task in Action Center ...
uv run uipath run agent --resume              # resumes with the decision
```

To answer without Action Center (the real task then stays open):

```bash
uv run uipath run agent --resume '{"action": "approve", "data": {"Comment": "ok"}}'
```

The output is structured: `{approved, decided_by, comment}`.

## Evaluate with a mocked escalation

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent without
Action Center: the escalation is mocked, so no task is created and the run
never suspends.

| Case | Mocking | Checks |
| --- | --- | --- |
| Manager approves (mockito) | `escalate_simpleapprovalapp` returns `{"action": "approve", "data": {"Comment": "Budget OK for Q3"}}` | the comment is in the output; the agent escalated |
| Manager approves (LLM) | an LLM writes the manager's answer from the case's `prompt` | the output mentions the conference; the agent escalated |
| Under the limit | the escalation raises, so escalating would fail the case | the agent decided itself; no tool was called |
| Generated input | an LLM writes the agent's input from `inputMockingStrategy.prompt` (a large expense); the escalation is mocked | the manager's comment is in the output; the agent escalated |

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

The mock name is the tool name in lower case (`escalate_simpleapprovalapp`),
as in LangChain. A mocked answer goes through `outcome_mapping` like a real
one: a mocked `reject` ends the run with `TERMINATION_ESCALATION_REJECTED`,
and `uipath eval` does not score a run that ends that way. In the trace, the
`SimpleApprovalApp` span is renamed `Simulated result: SimpleApprovalApp`.
See the main README's "Mocking tools in evals" section.

## Simulate without an eval

`uipath run` and `uipath debug` mock tools too, with `--simulation` (or a
`simulation.json` file in the folder, for `uipath debug`):

```bash
# an LLM writes the manager's answer
uv run uipath run agent --file input.json --simulation '{"enabled": true, "toolsToSimulate": [{"name": "escalate_simpleapprovalapp"}], "instructions": "The manager approves with a short comment."}'

# the simulate-component API returns a fixed answer (simulationStrategy 2 = static)
uv run uipath run agent --file input.json --simulation '{"enabled": true, "components": [{"componentId": "escalate_simpleapprovalapp", "componentType": "tool", "simulationStrategy": 2, "mockValue": {"action": "approve", "data": {"Comment": "ok"}}}]}'
```

No task is created and the run does not suspend. In the `components` format,
mockito rules with a `when` condition do not match yet: the SDK sends the
tool input as `{"args": [...], "kwargs": {...}}` while the API matches
conditions on top-level fields. Rules without `when` work.

## Run on Orchestrator

Publish the agent and start a job: the job suspends on the task, and
Orchestrator resumes it when the task is completed.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - escalate_SimpleApprovalApp toolType Escalation, open while the run is suspended
│  └─ SimpleApprovalApp                   escalationTool: arguments, channelType, taskId, taskUrl, assignedTo, result {outcome, output}
│     └─ tasks_create
├─ LLM call
└─ Agent output
```
