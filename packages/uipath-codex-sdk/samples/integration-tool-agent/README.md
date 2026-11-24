# Integration Service tool

A Codex agent that researches a topic with the **Web Search** activity of the
*UiPath GenAI Activities* connector (`uipath-uipath-airdk`), called as an
Integration Service (IS) tool.

```python
web_search = uipath_integration_tool(
    "web_search",
    connection_id=CONNECTION_ID,          # also the bindings.json key
    connector_key="uipath-uipath-airdk",
    tool_path="/v2/webSearch",
    method="POST",
    input_schema=WebSearchIn,             # query, num → JSON body
    static_args={"provider": "GoogleCustomSearch"},   # hidden from the model
    parameters=[{"name": "X-Elements-Service-Profile", "fieldLocation": "header",
                 "fieldVariant": "static", "value": "high-capacity"}],
)
```

The activity is read-only and has no side effects.

## How it works

1. The runtime serves `web_search` to Codex from its own MCP endpoint
   (`uipath_tools`), the same one durable tools use.
2. When the model calls it, the runtime validates the arguments, merges the
   static values (`provider`, the header), and calls
   `UiPath().connections.invoke_activity_async(...)`. That is the same SDK
   call low-code agents make.
3. The JSON result goes back to the model as the tool result straight away.
   The run never suspends on an IS tool. An IS error (for example a 400 for
   a bad argument) goes back as a tool error the model can react to. It
   does not fail the run.

## Prerequisites

* A **UiPath GenAI Activities** connection in a folder you can use, such as
  your personal workspace or `Shared`. Create it once in Integration Service
  → Connections → *UiPath GenAI Activities* → Connect, or with
  `uip is connections create`. The connector uses first-party auth, so no
  third-party sign-in is needed.
* Its id (the GUID in the connection's URL) in `CONNECTION_ID` in `main.py`,
  and as `key` and `defaultValue` in `bindings.json`.

```bash
cd samples/integration-tool-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json
```

## Copying a tool from a low-code agent

A low-code agent's `agent.json` lists its IS tools under `resources` with
`"type": "Integration"`. Such a resource carries the connection, tool path,
parameter locations and schemas. Pass it to `integration_tool_from_resource`
and you get the same tool:

```python
import json
from uipath_codex_sdk import integration_tool_from_resource

resources = json.load(open("agent.json"))["resources"]
tools = [
    integration_tool_from_resource(r)
    for r in resources
    if r.get("type", "").lower() == "integration" and r.get("isEnabled", True)
]
```

As in low-code, `static` parameters are fixed and hidden from the model, and
`argument` parameters (`{{name}}`) are read from the agent input (the model
sees them as optional fields and fills them only when the input has no
value). `dynamic` parameters are filled in by the model. The tool name is the
resource name sanitized the way low-code does it, so evaluators and prompts
copied from the low-code agent still match.

## Rebinding the connection

`bindings.json` declares the connection as a `connection` resource keyed by
its id. At deploy time a resource overwrite can point the tool at another
connection, for example in another folder or tenant. The runtime always hands
the SDK the id from `main.py`, and the SDK applies the overwrite when it
resolves the connection. No code change is needed. The CLI does not generate
this binding for coded agents, so keep it in sync with `CONNECTION_ID` by
hand.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

No Integration Service call is made: `web_search` is answered by an LLM. The
mocker sees the activity input with the static `provider` merged in, as in
LangChain.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## The trace

```
Agent run - agent
├── LLM call
├── Tool call - web_search      (toolType Integration, output.value = {"content": <result JSON>})
│   └── web_search              (integrationTool: toolName, arguments, result)
│       └── activity_invoke     (the SDK's own spans: connections_retrieve, the HTTP call)
├── LLM call
└── Agent output
```

The SDK's `activity_invoke` / `connections_retrieve` spans are not filtered
out (low-code agents hide them).

## Limitations

* The input schema comes from you or from the copied resource. It is not
  discovered from the connector at runtime.
* File and multipart inputs are passed through as given. A job attachment is
  not converted into an upload.
* Every IS error, including a missing connection (404) or missing access
  (401/403), goes back to the model as text. Argument errors (400/422) and
  timeouts are logged at WARNING, other HTTP errors at ERROR. The run does
  not fail.
