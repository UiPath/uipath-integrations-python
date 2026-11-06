# uipath-codex-sdk

Build and run [OpenAI Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) agents on the
UiPath Cloud Platform. LLM traffic from the bundled `codex` binary is routed through the
UiPath LLM Gateway (OpenAI Responses API) via a local loopback proxy, so agents run with
your tenant's models and metering — no OpenAI API key required.

## Quickstart

```bash
uipath new my-agent          # scaffold main.py + codex.json + pyproject.toml
uipath auth                  # authenticate against your tenant
uipath run agent '{"topic": "robots"}'
```

`main.py` exposes a module-level `UiPathCodexAgent`, and `codex.json` maps the entrypoint
to it:

```python
from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
)
```

```json
{ "agents": { "agent": "main.py:agent" } }
```

See [`samples/quickstart-agent`](samples/quickstart-agent).

## The agent

`UiPathCodexAgent` is plain configuration; the runtime builds and runs the
`codex` process from it.

| Field | Default | What it does |
| --- | --- | --- |
| `model` | (required) | The tenant model the LLM Gateway serves, e.g. `gpt-5.6-terra` (see [Models](#models)) |
| `prompt` | `None` | The first user message, a `str.format` template over the input fields (`"Tell me a joke about {topic}."`). Without it, the input's `prompt`, `message`, `input` or `query` string is used, else the whole input as JSON |
| `input_schema` | `None` | Pydantic model: the input is validated against it and it becomes the entry point's input schema |
| `output_schema` | `None` | Pydantic model: Codex is asked for native structured output in that shape, and the result is validated against it |
| `base_instructions` | `None` | System/base instructions for the thread |
| `tools` | `None` | UiPath tools (process, escalation, Integration Service, context grounding, ...; the sections below) |
| `mcp_servers` | `None` | Your own MCP servers, by name (see [MCP servers](#mcp-servers)) |
| `sandbox` | `"read-only"` | Codex's sandbox for its built-in tools (see [Built-in tools and the sandbox](#built-in-tools-and-the-sandbox)) |
| `approval_mode` | `"deny_all"` | How Codex handles escalated permission requests; `"deny_all"` never asks |
| `codex_bin` | `None` | A specific `codex` binary instead of the bundled one (also `UIPATH_CODEX_BIN`) |
| `name` | `"agent"` | Display name in the agent graph |

The remaining fields (`codex_home`, `hermetic`, `durable_tool_instructions`,
`durable_settle_timeout`) tune durable tools; their defaults suit almost every
agent.

### Structured output

```python
class Joke(BaseModel):
    setup: str
    punchline: str
    rating: int = Field(description="How funny, 1-10.")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
    input_schema=JokeInput,
    output_schema=Joke,
)
```

The run's output is the validated `Joke` as JSON. Without an `output_schema`
it is `{"output": "<the final message>"}`. See
[`samples/structured-output-agent`](samples/structured-output-agent).

## Built-in tools and the sandbox

Codex's own tools (shell commands, file edits, `apply_patch`) are always
available to the model; `sandbox` decides what they may do:

| `sandbox` | Built-in tools can | Use it for |
| --- | --- | --- |
| `"read-only"` (default) | read files and run read-only commands | agents that only answer, or only use UiPath / MCP tools |
| `"workspace-write"` | write in the working directory | local runs; it needs Codex's Linux sandbox (namespaces, landlock), which serverless robots do not have |
| `"full-access"` | write files and run anything | coding agents, and local stdio MCP servers on a robot |

On a serverless robot each job runs in its own working directory, and the
files it writes are discarded with the job. See
[`samples/coding-agent`](samples/coding-agent), which writes a Python file
and runs it.

> **Security:** commands Codex runs inherit the job's environment, including
> `UIPATH_ACCESS_TOKEN` (the job's UiPath token) and the runtime's gateway
> key. Give `"full-access"` only to agents whose inputs you trust: a prompt
> injection in a document the agent reads could make it run a command that
> sends the token out.

## MCP servers

`mcp_servers` adds your own MCP servers, by name. Codex calls them directly,
so their tools run for real in evals too (they cannot be mocked).

```python
from uipath_codex_sdk import RemoteMcpServer, StdioMcpServer, UiPathCodexAgent

agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Look up the status of order '{order_id}'.",
    sandbox="full-access",  # a local stdio server is a process Codex starts
    mcp_servers={
        "orders": StdioMcpServer(command="python", args=["orders_server.py"]),
        "uipath_mcp": RemoteMcpServer(
            url="https://cloud.uipath.com/<org>/<tenant>/agenthub_/mcp/<folder-key>/<slug>",
            bearer_token_env_var="UIPATH_ACCESS_TOKEN",
            default_tools_approval_mode="approve",
        ),
    },
)
```

* `StdioMcpServer(command, args, env)` starts a local server process.
* `RemoteMcpServer(url, bearer_token_env_var | bearer_token, http_headers,
  startup_timeout_sec, tool_timeout_sec)` reaches a streamable-HTTP server.
  Codex silently skips a remote server whose bearer token variable is not
  set; the runtime warns when that would happen.
* A UiPath-hosted MCP server (Orchestrator → folder → MCP Servers → Copy URL)
  authenticates with the job's own token: `bearer_token_env_var=
  "UIPATH_ACCESS_TOKEN"`. Its tools carry no read-only annotation, so set
  `default_tools_approval_mode="approve"`, or `"deny_all"` rejects every
  call.
* `tool_approval_modes`, `supports_parallel_tool_calls` and
  `omit_tools_from` pass through to Codex.

The model sees a server's tools as `mcp__<server>__<tool>`; trace spans and
tool-call evaluators use the plain tool name. See
[`samples/local-mcp-agent`](samples/local-mcp-agent),
[`samples/remote-mcp-agent`](samples/remote-mcp-agent) and
[`samples/uipath-mcp-agent`](samples/uipath-mcp-agent).

## Models

Codex speaks only the OpenAI Responses API, and the runtime routes it to the
tenant's LLM Gateway as an OpenAI passthrough, so **only OpenAI models** work.
The tenant's Claude, Gemini and Llama models do not. `uipath list-models`
lists what the tenant serves; pass the name as the gateway reports it
(`gpt-5-mini-2025-08-07`, not `gpt-5-mini`).

Verified with every sample: `gpt-5.6-terra`. Verified with the quickstart and
durable process samples: `gpt-6.1-sol` and `gpt-5-mini-2025-08-07`. Smaller
models follow tool instructions less reliably and can spend extra turns (for
example on shell commands) before calling the right tool.

## Durable process and agent tools

A durable tool runs an Orchestrator process or a deployed agent as a job, and
**suspends** the run until the job finishes. The run then resumes, possibly on
another robot, and the job's output becomes the result of the model's original
tool call.

```python
from pydantic import BaseModel
from uipath_codex_sdk import UiPathCodexAgent, uipath_agent_tool, uipath_process_tool


class AddIn(BaseModel):
    a: int
    b: int


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Use a_plus_b to add {a} and {b}.",
    tools=[
        uipath_process_tool(
            "a_plus_b",
            folder_path="Shared",
            input_schema=AddIn,
            description="Adds a and b.",
        ),
        uipath_agent_tool("Research_Agent", folder_path="Agents"),
    ],
)
```

* `uipath run agent ...` suspends. `uipath run agent --resume` continues once the
  jobs are done, and suspends again if one is still running.
* A successful job returns its output arguments. A faulted, stopped, stopping
  or terminating job returns a `{"status": ...}` JSON result that the model can
  act on. A job that fails to start comes back as a tool error.
* Several durable calls in one model response run in parallel. The run waits
  for all of them.
* State lives in `__uipath/state.db`, including the Codex conversation.
  An agent with durable tools runs on a run-scoped `CODEX_HOME`
  (`__uipath/codex_home`, so `~/.codex` is not read) and with codex's ChatGPT
  backend calls turned off (`hermetic`), as a conversational agent does.
  Other agents without durable tools are not affected.
* The child job's `ParentSpanId` is the tool span. The job attachments in the
  arguments' attachment fields are linked to the child job (see "Job
  attachments"); BTS correlation (`parent_operation_id`) is not passed.
* Custom durable tools implement the `UiPathDurableTool` protocol: `name`,
  `description`, `input_schema`, `validate_args`, `start` and `finish`.
  `start` can return any suspend value the resume trigger handler knows
  (`WaitJobRaw`, `WaitDeepRagRaw`, `WaitEscalation`). DeepRAG tasks have their own tool
  (`uipath_deep_rag_tool`, under "Context grounding tools").

See [`samples/durable-process-tool`](samples/durable-process-tool) for a full
walkthrough, including local and tenant test steps.

### Pre-filled arguments

Some process inputs do not need to come from the model. `static_args` fixes
them, and `args_from_input` reads them from the agent input:

```python
invoice = uipath_process_tool(
    "CreateInvoice",
    input_schema=InvoiceIn,  # customerId, currency, amount, ...
    static_args={"currency": "EUR", "lines[*].taxed": True},
    args_from_input={"customerId": "{{customer.id}}"},
)
```

* Keys are paths into the process input: `a.b`, `items[*].id` (every
  element), `items[*]` (every element), `items[0]`. LangChain's JSONPath
  (`$['a']['b']`, `$.a.b`) works too. A field the input schema does not have
  is refused.
* Static fields are hidden from the model, at any depth. `items[*]` hides the
  array `items`. An index key such as `items[0]` stays visible: only its
  value is merged.
* `args_from_input` fields stay visible to the model, but optional, with a
  note saying where they come from. The agent input's value wins; when the
  agent input has none, the model's value is sent, as in LangChain. A
  required field neither of them fills is a tool error asking the model to
  fill it.
* Pre-filled values overwrite the model's when the job starts. A static
  `items[*]` on an empty or missing `items` becomes `[value]`. A pydantic
  `input_schema` validates the merged input, so a pre-filled field can be
  required.
* The `Tool call` span's input, the simulated `processTool` span's
  `arguments` and the eval mocker see the merged input, as in low-code.

### From an agent.json

`process_tool_from_resource(resource)` builds the tool from a low-code
`agent.json` `tool` resource of `type` Process, Agent, Api,
ProcessOrchestration, Flow or Function (any case). Every one of them starts
an Orchestrator job, as in LangChain:

| Tool field | From the resource |
| --- | --- |
| process name (and `process.<name>` binding key) | `properties.processName` (required) |
| tool name | `name`, sanitized like low-code: `Basic RPA Process` becomes `Basic_RPA_Process` (`tool_name=` overrides it) |
| description | `description`, plus the output schema hint |
| `kind` | `agent` for `Agent`, `process` otherwise |
| `toolType` of the `Tool call` span | low-code's: `Process`, `Agent`, `Api`, `agenticProcess` (ProcessOrchestration), `Function`, and `Integration` for a Flow |
| input / output schema | `inputSchema` / `outputSchema` |
| pre-filled inputs | `argumentProperties` (below) |
| mock name, example calls | `name`, `properties.exampleCalls` |

* The job runs in the run's folder (`UIPATH_FOLDER_PATH`), as in low-code:
  `properties.folderPath` is ignored, and bindings redirect the process.
  `folder_path=` sets the folder explicitly. `run_as_me=` is the caller's
  choice (low-code sets it for conversational agents only).
* `settings` (timeout, retries), `requireConversationalConfirmation` and the
  legacy `arguments` are ignored. A disabled resource is refused: filter it
  out first.

`argumentProperties` keys are paths into the process input, usually
JSONPath (`$['customer']['id']`):

| `variant` | Value |
| --- | --- |
| `static` | `value`, hidden from the model |
| `argument` | the agent input at `argumentPath` (`$.customer.id`); visible but optional, the model's value is used when the agent input has none |
| `textBuilder` | `tokens` rendered against the agent input (`input.customer.name`), hidden |
| `arrayBuilder` | a list of its `name[0]`, `name[1]`, ... items (each one static, argument, text builder or array builder); a gap, or an argument item the agent input lacks, is `null`; a text-builder item keeps the text of a variable the agent input lacks; hidden |
| `objectBuilder` | refused, as in LangChain ("unsupported argument property type"), unless a parent path covers it |
| `dynamic` | refused (the resource does not validate); leave the key out to let the model fill it |

A parent path wins over the paths under it (`customer` over
`customer.id`). A path the input schema does not have is ignored with a
warning. `isSensitive` values are hidden from the model like any pre-filled
value, and show in clear text in the spans, as in low-code.

Escalation, Analyze Files, DeepRAG and Batch Transform resources read
`argumentProperties` the same way, as in LangChain: JSONPath keys, array
builders, parent wins, `static` / `textBuilder` / `arrayBuilder` fields
hidden at any depth, and an `argument` field visible but optional, with the
note. The agent input's value wins; when it has none, the model's value is
used, and a required field neither of them fills is a tool error asking the
model to fill it (nothing is started). Job attachments are checked once the
pre-filled values are merged in, as LangChain resolves them after its static
arguments: an attachment the agent input (or a static value) fills is the
one checked, and an unknown ID the model passed for that field is ignored;
an attachment field only the model fills (Analyze Files `attachments`,
DeepRAG / Batch Transform `attachment`, a process input, an app field) is
resolved against the run's known attachments like any attachment argument,
and an unknown ID there is still a tool error. An escalation's pre-filled
app fields are never checked (LangChain does not resolve an escalation's
attachments); a model value they replace is not checked either. One taken
from the agent input is matched to the known one by its `ID`. The `Tool call` span's input
and `attachments`, and the eval mocker, see the merged arguments; the
model's own arguments are what is saved for resume.

See [`samples/process-resource-agent`](samples/process-resource-agent) for a
process tool built from a resource file, with an `argument` and a `static`
input.

## Integration Service tools

An Integration Service (IS) tool runs one activity of a connector, such as
GenAI Activities *Web Search* or Slack *Send Message*, on one of your
connections. It is an **immediate** tool: the call is answered as soon as
Integration Service returns, and the run never suspends on it. It is served
on the same `uipath_tools` MCP server as durable tools, so the two kinds can
be mixed, even in one model response.

```python
from pydantic import BaseModel
from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool


class WebSearchIn(BaseModel):
    query: str


web_search = uipath_integration_tool(
    "web_search",
    connection_id="<connection-id>",  # also the bindings.json key
    connector_key="uipath-uipath-airdk",
    tool_path="/v2/webSearch",
    method="POST",
    input_schema=WebSearchIn,  # POST: properties go to the body
    static_args={"provider": "GoogleCustomSearch"},  # hidden from the model
)

agent = UiPathCodexAgent(
    model="gpt-5.6-terra", prompt="Research {topic}.", tools=[web_search]
)
```

* The tool calls `UiPath().connections.invoke_activity_async(...)`, as
  low-code agents do. The result goes back to the model as JSON text.
* Parameter locations: each `parameters` entry (low-code shape: `name`,
  `fieldLocation` query/path/header/body/multipart/file, optional
  `fieldVariant` and `value`) sets its own location. A `{x}` in `tool_path`
  is a path parameter. The remaining input properties go to the body for
  POST/PUT/PATCH and to the query string for GET/GETBYID/DELETE.
* `static` values (`static_args`, or `fieldVariant: "static"`) are hidden from
  the model, at any depth (`provider`, `attachment.title`), and merged into
  every call. A static `items[*]` sets every element of `items` and turns an
  empty or missing `items` into `[value]`, as low-code does.
* `argument` parameters (`{{name}}`) take the agent input's value when it has
  one. The model still sees the field, marked optional, and its value is sent
  only when the agent input has none.
* A pydantic `input_schema` is validated and sent by alias, so
  `filter: str = Field(alias="$filter")` goes out as `$filter`.
* To reuse a tool from a low-code agent, pass its `agent.json` resource
  (`"type": "Integration"`) to `integration_tool_from_resource(resource)`.
  You get the same tool name (sanitized like low-code: `Get User (v2)` becomes
  `Get_User_v2`), activity metadata, enum hints and static parameters.
* Errors go back to the model as tool errors and never fail the run. A 400 or
  422 from the connector and timeouts are logged at WARNING, other HTTP errors
  at ERROR. The host enforces `timeout_sec`, 300 s by default; `None` also
  means 300 s, and there is no way to turn the timeout off.
* Connection bindings: the connection id is handed to the SDK unchanged on
  every call, so a `connection` resource overwrite (bindings.json, keyed by
  the id) rebinds it. Bindings for coded agents are written by hand.
* If a run suspends on a durable call while an IS call is still running, it
  waits up to `durable_settle_timeout` for the IS call. If the call is still
  running after that, it is cancelled and the model is told its outcome is
  unknown.
* An agent with only immediate tools keeps its normal Codex configuration and
  `CODEX_HOME`.
* Custom immediate tools implement the `UiPathTool` protocol: `name`,
  `description`, `input_schema`, `validate_args`, and
  `async call(args, ToolContext) -> str`, raising `UiPathToolError` for errors
  the model should see.

Not supported yet: discovering the input schema from the connector at
runtime, and converting job attachments into multipart file uploads. See
[`samples/integration-tool-agent`](samples/integration-tool-agent).

## Context grounding tools

A context tool runs a semantic search on one Context Grounding index and
returns the best matching chunks. It is an **immediate** tool on the same
`uipath_tools` MCP server: the call is answered when the search returns, and
the run never suspends on it.

```python
from uipath_codex_sdk import UiPathCodexAgent, uipath_context_tool

policies = uipath_context_tool(
    "Company Policies",  # display name; the model sees Company_Policies
    index_name="company-policies",  # a bindable `index` resource
    description="HR and travel policies of the company.",
    number_of_results=5,
)

agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Answer from the company policies: {question}",
    tools=[policies],
)
```

* The tool calls `UiPath().context_grounding.unified_search_async(...)`, as
  low-code agents do. The model gets langchain's result shape:
  `{"documents": [{"metadata": {"source", "search_id", "reference",
  "page_number", "score"}, "page_content": "..."}]}`. Zero hits is
  `{"documents": []}`, not an error. Results over 64 000 characters lose
  whole documents from the end and get `"truncated": true`.
* Where the query comes from:
  * the model, by default (a required `query` argument);
  * a fixed value, `query="..."`: the model calls the tool with no arguments;
  * the agent input, `query_from_input="topic"` (or `"{{topic}}"`, `"a.b"`):
    the input's value wins, and the model's `query`, which stays visible but
    optional, is used only when the input has none.

  `folder_path_prefix` / `folder_path_prefix_from_input` restrict the search
  to one folder of the index's data source, and `file_extension` to one file
  type (`All` means no filter).
* The index is looked up in `folder_path`, by default the job's folder
  (`UIPATH_FOLDER_PATH`). The folder key is resolved asynchronously (the
  SDK's own lookup is a blocking call), once per folder.
* Bindings: the index name and folder go through the SDK's `index` resource
  override on every call, so an `index.<name>[.<folder>]` overwrite
  (bindings.json) rebinds them.
* Errors go back to the model as tool errors: a missing index, an index still
  being ingested, a refused search (HTTP 400) or a timeout.
* To reuse a low-code agent's context, pass its `agent.json` resource
  (`"$resourceType": "context"`) to `context_tool_from_resource(resource)`.
  It maps `resultCount`, `threshold`, the query and folder prefix variants and
  `fileExtension`. Like langchain, it does not use the resource's
  `folderPath`, and it answers retrieval mode `Structured` with a semantic
  search.
* Tracing: `Tool call - <tool>` (`toolType` `ContextGrounding`, with
  `index_name` and `context_retrieval_mode`) has a `contextGroundingTool`
  child named after the display name, with the low-code attributes
  (`retrieval_mode`, `query`, `threshold`, `number_of_results`, ...) and the
  documents as `output`. Under it, a `ContextGroundingRetriever` span (an
  OpenInference `RETRIEVER` span, as langchain's retriever records) carries
  the query and the documents for the context precision and faithfulness
  evaluators, and the SDK's own `contextgrounding_*` spans sit under that.
  The retriever is a span of its own because LLMOps takes a span's type from
  `openinference.span.kind` when it is set.
* In evals, the search is `mockable`, as in langchain: the search runs mocked
  under the `contextGroundingTool` span (renamed `Simulated result: <name>`),
  with no `ContextGroundingRetriever` span. See "Mocking tools in evals".

### DeepRAG (durable)

`uipath_deep_rag_tool` runs a DeepRAG task on an index: the service
researches the documents and writes a synthesized answer with citations. A
task takes minutes, so the tool is **durable**: the call starts the task and
**suspends** the run, and the run resumes when the task ends, as for a
durable process tool.

```python
from uipath_codex_sdk import uipath_deep_rag_tool

research = uipath_deep_rag_tool(
    "Invoice Research",
    index_name="invoices",
    folder_path="Coded",
    citation_mode="Inline",  # or "Skip"
    description="Researches the company's invoices across documents.",
)
```

* The model gets langchain's DeepRAG result:
  `{"text": "...", "citations": [{"ordinal", "page_number", "source",
  "reference"}], "deepRagId": "..."}`. A failed task comes back as
  `{"status": "failed", "error": "<reason>", "deepRagId": "..."}`, a result
  the model can act on, not a faulted run. A start error (missing index,
  index still ingesting, a refused task) is a tool error.
* The query, folder prefix and file extension work as for the search tool
  (`query=`, `query_from_input=`, `folder_path_prefix`, `file_extension`);
  the prefix and extension become the task's glob pattern
  (`hr/*.pdf`). The model's `query` is the task prompt (at most 250 000
  characters).
* The tool starts the task itself (`start_deep_rag_async`) and suspends on a
  `WaitDeepRagRaw`; the resume trigger is a DeepRAG trigger, which
  Orchestrator fires when the task ends.
* `context_tool_from_resource` builds one from an `agent.json` context
  resource with retrieval mode `DeepRAG`; `settings.citationMode` is
  required, as in langchain.
* Tracing: the `Tool call` and its `contextGroundingTool` child
  (`retrieval_mode` `DeepRAG`, `query`, `citation_mode`, `deep_rag_id`) stay
  open across the suspension and close with the result in the execution that
  uses it.
* In evals it is simulated as `Invoice Research` (the display / resource
  name) with langchain's DeepRAG output schema, `{text, citations,
  deepRagId}`; see "Mocking tools in evals".

### DeepRAG on a job attachment (durable)

The same DeepRAG task, on a file the run was given instead of an index:

```python
from uipath_codex_sdk import uipath_deep_rag_attachments_tool

research = uipath_deep_rag_attachments_tool("DeepRAG")  # citation_mode="Inline"
```

* The model passes an `attachment` (a JobAttachment from the input or a tool
  result; see "Job attachments") and a `query`. `query=` fixes the prompt
  (hidden from the model). `query_from_input=` reads it from the agent input,
  and `attachment_from_input=` takes the file from the agent input: the
  agent input's value wins, and the model's is used when it has none (the
  field stays visible, with a note; `attachment` becomes optional, and a
  call with no file from either is a tool error).
* One call, `start_deep_rag_from_attachments_async`: the service builds a
  temporary index of the file and starts the task, and the run suspends on it
  as for an index. Same result and failure shape. The task runs in the job's
  folder (`folder_path=` to change it); the service needs one, so a local run
  sets `UIPATH_FOLDER_PATH`. DeepRAG reads PDF and TXT files;
  another type is refused when the task starts, and the model gets the error.
* `deep_rag_tool_from_resource(resource)` builds one from an `agent.json`
  internal tool with `toolType` `deep-rag` (Agent Builder's "DeepRAG"): its
  query (static, dynamic or agent input), citation mode and the static,
  argument and text-builder `argumentProperties`. `fileExtension` and
  `folderPathPrefix` are ignored, as in langchain. A static query without a
  value is refused when the tool is built.
* Differences from langchain: a start error is a tool error the model can act
  on (langchain faults the run), and the `DisableDeepRagFromAttachments` kill
  switch (langchain's older two-step path) is not read.
* In evals it is simulated by the name passed to the factory (or the resource
  name), with the resource's `outputSchema` when it declares fields, else
  DeepRAG's `{text, citations, deepRagId}`.
* Tracing: `Tool call - DeepRAG` (`toolType` `Internal`, `attachments`) and
  its internal tool child `DeepRAG` (`input`, `deep_rag_id`,
  `citation_mode`) stay open across the suspension and close with the
  result. See [`samples/deep-rag-attachments-agent`](samples/deep-rag-attachments-agent).

### Batch Transform on a job attachment (durable)

Adds columns to a CSV the run was given, row by row:

```python
from uipath_codex_sdk import uipath_batch_transform_attachments_tool

enrich = uipath_batch_transform_attachments_tool(
    "Enrich Reviews",
    output_columns=[("sentiment", "positive, neutral or negative")],
    web_search_grounding=False,
)
```

* The model passes an `attachment` (the CSV, a JobAttachment) and a `query`
  (`query=` fixes it; `attachment_from_input=` takes the file from the agent
  input, and the model's `attachment`, now optional, is used when the agent
  input has none; a call with no file from either is a tool error).
* One call, `start_batch_transform_from_attachment_async` (the service
  indexes the file and starts the task), and the run suspends on it. When it
  ends, the tool downloads the result (through the service's authenticated
  download endpoint, which every robot reaches; its storage URL is the
  fallback) and uploads it as a job attachment (`output.csv`, `result_name=`), links it to the job,
  and the model gets langchain's `{"result": {"ID", "FullName", "MimeType"}}`.
  The result is then known to the run: the model can pass it to another
  tool or return it in an attachment output field. A failed task fails the
  run, as in langchain.
* The task runs in the job's folder (`folder_path=` to change it; a local run
  sets `UIPATH_FOLDER_PATH`), and the result is uploaded to the job's folder.
  The upload is retried; if linking it to the job fails, the uploaded file
  is still returned.
* `batch_transform_tool_from_resource(resource)` builds one from an
  `agent.json` internal tool with `toolType` `batch-transform`: its query
  (static, or written by the model for any other variant, as in langchain),
  output columns (at least one, each with a description), web search setting
  and `argumentProperties`. `fileExtension` and the resource's
  `outputSchema` are ignored, as in langchain; `folderPathPrefix` is ignored
  too (langchain sends it; the attachment API has none).
* Tracing: `Tool call - <tool>` (`toolType` `ContextGrounding`, the source
  `in` and the result `out` as `attachments`) and its `contextGroundingTool`
  child (`retrieval_mode` `BatchTransform`, `query`, `output_columns`,
  `web_search_grounding`, `batch_transform_id`, and low-code's `results`)
  stay open across the suspension. See
  [`samples/batch-transform-attachments-agent`](samples/batch-transform-attachments-agent).
* In evals it is simulated by the name passed to the factory (or the resource
  name) with langchain's `{"result": <JobAttachment>}` schema, not wrapped in
  a second `result` as langchain's attachment tool does. The simulated result
  attachment becomes known to the run but does not exist: a later real tool
  that downloads it fails.
* Differences from langchain: a start error is a tool error, the trigger is
  read by codex's own reader (status only;
  the platform's downloads unauthenticated and unchecked), and the result
  file is downloaded to `.uipath/batch_transform/<task id>.csv` (then
  deleted) instead of `output.csv` in the working directory.

Not supported yet: the BatchTransform retrieval mode on an index and Data
Fabric contexts. `context_tool_from_resource` refuses those resources, and
`attachments` context resources (use the internal `deep-rag` or
`batch-transform` tools above).

## Action Center escalations (human in the loop)

An escalation tool asks a person through an Action Center app. It is
**durable**: the call creates the task and **suspends** the run, and the run
resumes when the reviewer completes it, as for a durable process tool.

```python
from pydantic import BaseModel
from uipath_codex_sdk import UiPathCodexAgent, uipath_escalation_tool


class Review(BaseModel):  # the app's inputs, filled by the model
    amount: float
    reason: str


class Decision(BaseModel):  # the app's outputs, returned to the model
    Comment: str


approval = uipath_escalation_tool(
    "ExpenseApproval",  # the deployed app (a bindable `app` resource)
    folder_path="Shared",
    input_schema=Review,
    output_schema=Decision,
    title="Expense approval",
    assignee="manager@example.com",
    outcome_mapping={"Reject": "end"},
)
```

* The model calls `escalate_<app>` with the app's inputs. The tool creates
  the task (`tasks.create_async`), assigns it, and suspends on a
  `WaitEscalation`. Its resume trigger is a task trigger: Orchestrator
  resumes the job when the task is completed.
* On resume the model gets langchain's escalation result:
  `{"status": "completed", "outcome": "<the button>", "output": {<the app's
  output fields>}, "task_id": ..., "assigned_to": "<reviewer email>"}`.
  `output` keeps the output schema's fields (without one, every field that
  is not an input).
* `outcome_mapping` maps an outcome to `continue` (the default) or `end`. An
  `end` outcome ends the run with `TERMINATION_ESCALATION_REJECTED`, as
  langchain does, and so does a deleted task.
* The task: `title` (or `title_from_input`, a path in the agent input),
  `priority`, `labels`, and the actionable-message settings. Pre-filled
  inputs, as for process tools: `static_args` (fixed values, hidden from the
  model) and `args_from_input` (field → path in the agent input: visible but
  optional, with a note; the agent input's value wins, the model's is used
  when it has none, and a required field neither fills is a tool error, no
  task is created). Both are merged into the task data.
* Who gets it: `assignee` (user name or email), or a `TaskRecipient` (user or
  group by id, email or name, Workload, RoundRobin). The task is created and
  then assigned. If the assignment fails, the task stays unassigned in Action
  Center, the run still waits on it (no duplicate on a retry), and the span
  records `assignmentError`.
* `escalation_tool_from_resource(resource)` builds one from an `agent.json`
  `escalation` resource: its first channel (an Action Center app), the app
  and folder, the input and output schemas, the task title (text, text
  builder or agent-input path), the first recipient (including assets and
  agent-input emails or groups), priority, labels, outcome mapping and the
  static, argument, text-builder and array-builder `argumentProperties`
  (read as for process tools; see "From an agent.json" there).
* Tracing: `Tool call - escalate_<app>` (`toolType` `Escalation`) and its
  `escalationTool` child, named after the app (`arguments`, `channelType`,
  `taskId`, `taskUrl`, `assignedTo`), stay open across the suspension. Both
  close with low-code's `{outcome, output}`; the `Tool call`'s `output.value`
  keeps the full result the model received.
* Testing locally: `uipath run agent --resume` reads the real task (and
  suspends again while it is open). To answer without Action Center, pass
  the answer: `uipath run agent --resume '{"action": "Approve", "data":
  {"Comment": "ok"}}'`. The real task then stays open in Action Center.
* In evals it is simulated by its model-facing name lowercased
  (`escalate_expenseapproval`, langchain's) with langchain's
  `{"action": "approve"|"reject", "data": {<output fields>}, "is_deleted"}`;
  the answer then goes through the same output filter and `outcome_mapping`
  as a real task (the action is matched as given, so map `reject`, not
  `Reject`, for a simulated `end`). No task is created.

Not supported yet: QuickForm channels, escalation memory and a recipient chosen
by the model.
When a run ends, other open tasks stay in Action Center (there is no cancel
API).

## IXP extraction and Document Validation

Two durable tools that work as a pair, as in low-code agents: an IXP
extraction of a document, then a human check of it in Action Center's
Validation Station.

```python
from uipath_codex_sdk import uipath_document_validation_tool, uipath_ixp_extraction_tool

extract = uipath_ixp_extraction_tool(
    "Extract Invoice", project_name="Invoices", version_tag="live"
)
validate = uipath_document_validation_tool(
    "Validate Invoice",
    extraction_tool=extract,
    storage_bucket_name="invoice-validation",
)
```

* `Extract_Invoice`: the model passes a document (a JobAttachment, see "Job
  attachments"). The tool downloads it, starts the extraction on the
  project's tagged model version and suspends the run; on resume the model
  gets the extracted fields (`dataProjection`). The full extraction is kept
  with the run for the validation tool.
* `Validate_Invoice` takes no arguments: it creates a Validation Station task
  for its extraction tool's last result (`storage_bucket_name`,
  `storage_bucket_directory_path`, `action_folder`, `priority`, `title`) and
  suspends; on resume the model gets the validated fields, `{"data": ...}`.
  Called before the extraction has returned, it answers with an error the
  model can act on.
* As in langchain: a document marked as an exception, or a deleted task, ends
  the run with `TERMINATION_ESCALATION_REJECTED`; a call without a document,
  a download or start error, and a failed extraction or validation operation
  fail the run.
* From an `agent.json`: `ixp_extraction_tool_from_resource` (a `tool` with
  `type: "Ixp"`) and `vs_escalation_tool_from_resource` (an `escalation` with
  `escalationType: 1`, Agent Builder's "Document Validation Task";
  `escalation_tool_from_resource` routes those there too). The validation
  tool names its IXP tool by the resource name (`ixpToolId`). The bucket is
  looked up in `action_folder` only (the job's folder by default); from a
  resource that is its `storageBucketFolderPath` (langchain sends it as a
  directory inside the bucket instead, where the bucket is not found).
* Tracing: `Tool call - <tool>` (`toolType` `IxpExtraction` / `Escalation`)
  with low-code's `ixpTool` (`projectName`, `versionTag`, `extractionId`,
  `documentId`, `result`) and `vsEscalationTool` (`ixpToolId`,
  `storageBucketName`, `operationId`, `taskUrl`, `result`) children, held
  across the suspension. See [`samples/ixp-validation-agent`](samples/ixp-validation-agent).

* In evals the extraction is simulated by its display (resource) name with
  langchain's `ExtractionResponseIXP` schema, and the validation by its
  display name with `{"data": [<field groups>] | null}`. A simulated
  extraction is kept for the validation tool like a real one, so a real
  validation task then shows the simulated fields; a simulated validation
  still needs an extraction (real or simulated) first, as in langchain.

Not supported yet: the `DocumentValidationStation` telemetry event and
classification projects.

## Job attachments

Orchestrator files a run receives, hands to tools and returns. As in
LangChain ReAct agents, the model never opens a file itself: it sees each
attachment as a JobAttachment object and passes it to the tools that accept
attachments.

```python
from pydantic import BaseModel
from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_analyze_files_tool,
)


class Request(BaseModel):
    document: UiPathAttachment  # the platform's Attachment model
    question: str


class Output(BaseModel):
    answer: str
    source: UiPathAttachment | None = None


agent = UiPathCodexAgent(
    input_schema=Request,
    output_schema=Output,
    prompt="Answer {question} about the attached document.",
    tools=[uipath_analyze_files_tool()],
)
```

* Schemas: a `UiPathAttachment` field is `$ref: #/definitions/job-attachment`
  (the `JobAttachment` resource kind) in the entrypoint's input and output
  schemas, as in LangChain, so Orchestrator shows a file picker.
* The prompt: an attachment renders as JSON, `{"ID", "FullName",
  "MimeType"}` (`{document}`, or `{document[FullName]}`). Attachments the
  template does not show are listed in a block after it. A short developer
  instruction tells the model how to use them.
* Known attachments: those of the input, and those tools return. A tool
  argument at an attachment field (`$ref` to `job-attachment`, the
  `JobAttachment` marker, or a `UiPathAttachment` model field) may be just
  `{"ID": ...}`; the tool host fills in the rest before the tool runs. An ID
  the run does not know is answered with LangChain's "Could not find
  JobAttachment with ID=..." error, and the tool does not run. As in
  LangChain, the IDs checked are those of the arguments with the tool's
  pre-filled values merged in (see "Pre-filled arguments" above): a value
  the agent input or a static value replaces is not checked, the
  pre-filled one is. A custom tool with pre-filled fields gets the same
  through two optional members, `prefilled_args(args, agent_input)` (the
  merged arguments) and `prefilled_schema` (the schema they follow), and
  `checks_prefilled_attachments = False` to have only the model's own
  values the merge keeps checked (as an escalation does). The
  attachments at the attachment fields of a tool's output schema (e.g. a
  process tool's `output_schema`) become known. They are saved with a
  suspended run.
* Process and agent tools link the attachments in their arguments to the
  child job (`processes.invoke_async(attachments=...)`).
* Output: the model writes an attachment field as `{"ID": ...}` (the strict
  output schema asks for nothing else) and the runtime fills in the rest. An
  ID the run does not know fails the run with `OUTPUT_INVALID`. Files the
  agent writes are not uploaded.
* `uipath_analyze_files_tool()` (or `analyze_files_tool_from_resource(...)`
  for an `agent.json` internal tool with `toolType` `analyze-attachments`,
  including its `argumentProperties`): the model passes `attachments` and an
  `analysisTask` (an `argument` `argumentProperties` reads them from the
  agent input; the model's are used when it has none). The tool downloads
  each file, sends it with the task to the agent's model (or `model=`) in
  one LLM Gateway call, with LangChain's system message, and answers `{"analysisResult": ...}`. Images (PNG, JPEG,
  GIF, WebP) go as images, other files (PDF, ...) as files; TIFF is not
  supported yet. A file over 30 MB fails the run (`FILE_ERROR`), as in
  LangChain.
* Tracing: `attachments` (LLMOps' `SpanAttachment` list) on `Agent run`
  (the input's), `Agent output` (the output's) and each `Tool call` (its
  arguments' and result's; the arguments as merged with the pre-filled
  values, so an attachment taken from the agent input is listed too). Analyze Files nests like low-code's internal
  tools: `Tool call - Analyze_Files` (`toolType` `Internal`) → `Analyze
  Files` → `LLM call` with the files sent.
* Testing locally: upload a file with `UiPath().attachments.upload(...)` and
  pass its `{ID, FullName, MimeType}` in the input, as
  [`samples/analyze-files-agent`](samples/analyze-files-agent)'s `upload.py`
  does.

* DeepRAG on an attachment: see "DeepRAG on a job attachment" under
  "Context grounding tools".

* Batch Transform on an attachment: see "Batch Transform on a job
  attachment" under "Context grounding tools".

* Files uploaded in a chat: see "Conversational agents" below.

Not supported yet: attachments in Integration Service file uploads, and TIFF
files in Analyze Files.

## Conversational agents

A Codex agent becomes a chat agent with one flag in `uipath.json`; the
agent code does not change:

```json
{ "runtimeOptions": { "isConversational": true } }
```

`uipath pack` copies the flag into the package, which makes Orchestrator
and the Agents / Autopilot chat treat the process as a conversational agent.
At runtime a chat host's conversation id switches it on too.

```python
agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    base_instructions="You help colleagues pick a meeting room. ...",
    tools=[FindRoomsTool(), uipath_analyze_files_tool()],
)
```

* Input and output: the UiPath conversation message format
  (`{"messages": [{"role": "user", "contentParts": [...]}]}`) on both sides.
  `base_instructions` is the system prompt. `prompt`, `input_schema` and
  `output_schema` do not apply and are refused (`CONFIG_INVALID`).
* One exchange per run: the runtime takes the newest user message, runs one
  Codex turn with it and ends **suspended with no trigger**. The job stays
  alive for the next message, and the chat host closes the exchange
  (`endExchange`).
* Memory: the conversation is one Codex thread. Every exchange resumes it,
  so only the newest message is sent and the earlier ones come from the
  thread; it does not matter whether the host replays the history. The
  thread's rollout is saved in `state.db` after every exchange and restored
  before the next one, so a new job (another robot) continues the same
  conversation. The state is keyed by the conversation id (the runtime id
  of a chat job). Codex's own auto-compaction keeps a long thread's context
  bounded; a rollout over 8 MiB compressed logs a warning.
* Streaming: assistant text streams as it arrives (one `chunk` per text
  delta, `text/markdown`). Commentary (Codex's progress notes) is shown
  too. MCP tool calls (UiPath tools and your own servers) and shell commands
  (`shell`) are tool cards on the message that called them:
  `startToolCall`, then `endToolCall` with the result. Reasoning, file edits
  and web searches stay state events (debugger, traces).
* Durable tools and escalations work inside an exchange: the call's card
  stays open, the job suspends on the call's trigger and the exchange stays
  open (no `endExchange`). The resumed job ends the card with the result,
  streams the rest of the reply and then ends the exchange. A user message
  that arrives while a call is still pending is not processed: the exchange
  suspends again on the same call(s), with their triggers.
* Files uploaded in the chat (`urn:uipath:cas:file:orchestrator:<id>`)
  become job attachments the run knows, listed in the prompt as for any
  input attachment, so `uipath_analyze_files_tool()` and the other
  attachment tools can read them in this exchange and the later ones. A
  message with only a file is fine.
* Tracing: one `Conversational agent run - <agent name>` span per exchange
  (as low-code names it), with `isConversational`, `conversationId`,
  `exchangeId`, `threadId` and the exchange `input`. An exchange that
  suspends on a durable call keeps its span open until the resumed job
  finishes the exchange.

Locally:

```bash
uv run uipath dev                                          # chat in the browser
uv run uipath run agent --file input.json                  # first message
uv run uipath run agent --file next_message.json --resume  # the next ones
```

`uipath run` prints the state events (tool calls) but not the reply, which
is a chat event; to see the envelopes a chat host gets, run as a deployed
chat job does: set `UIPATH_JOB_KEY` to any GUID,
`CAS_WEBSOCKET_DISABLED=true`, and the conversation ids in `uipath.json`
(`"fpsProperties": {"conversationalService.conversationId": "...",
"conversationalService.exchangeId": "..."}`); each envelope is then logged
to `__uipath/execution.log`. See [`samples/chat-agent`](samples/chat-agent).

Not supported yet: in-chat tool confirmation (`requireConfirmation`) and
client-side tools; images sent to the model as images (a chat file is an
attachment for tools).

## Tracing

A job is one span tree with one root, however many times it suspends. It
uses the same span types as low-code agents:

* `Agent run - <entrypoint>` (`agentRun`): prompt, input, schemas, output,
  and `status` (successful or faulted). It opens in the first execution and
  closes in the last one. A conversational agent has one `Conversational
  agent run - <agent name>` per chat exchange instead (see "Conversational
  agents").
  * `LLM call` (`llmCall`): one per model request, with the output text, tool
    calls and token usage. The shim sends the trace context to the LLM
    Gateway, which adds its `Model run` span underneath.
  * `Tool call - <tool>` (`toolCall`): MCP tools, shell commands, file edits
    and web searches that Codex runs, plus durable tool calls. A durable
    call's span opens when its job starts and stays open until a resumed
    execution reads the job's result. The span then closes with that result
    as its output. It is the child job's `ParentSpanId`, so the child job's
    trace sits under it. For process, agent and escalation tools its `input`
    is the arguments with the pre-filled fields merged in, as in low-code.
    An Integration Service call is a `Tool call` with
    `toolType` `Integration` and an `integrationTool` child span (`toolName`,
    `arguments`, `result`). The SDK's own `activity_invoke` spans sit under
    that child.
  * `Agent output` (`agentOutput`): on success.

When an execution suspends, it does not end the `Agent run` span or the
`Tool call` spans of jobs still running. On a deployed job, it sends them to
LLMOps as still running. It saves their identity (trace id, span id, start
time, attributes) in the state database. Each resumed execution parents its
spans (resume-trigger reads, LLM calls, tool calls) directly under the saved
`Agent run`, as low-code agents do. The execution that finishes the job sends
the `Agent run` with its original id and start time and the final status and
output. The span goes through every span processor the CLI set up, so
LLMOps and a local `--trace-file` both get it. With `--trace-file`, a span
appears only when it ends. A file shared across `uipath run` and
`uipath run --resume` therefore holds the complete tree once the job ends.
If the job faults, the `Agent run` and any durable `Tool call` that is still
open close with an error.

A fresh (non-resume) run discards any saved trace context. When a job is
resumed in the same process (its job had already finished at suspend time),
everything stays inside one live `Agent run`. Agents
without durable tools never suspend, so each run is one `Agent run`.

## Evaluations

Codex agents run under `uipath eval` like any other coded agent. Eval sets
live in `evaluations/eval-sets/`, and the evaluators they reference live in
`evaluations/evaluators/`. Run one from the agent's directory:

```bash
uipath eval agent evaluations/eval-sets/evaluation-set-default.json
```

Two samples ship with eval sets:

* [`samples/structured-output-agent`](samples/structured-output-agent): three
  topics, each scored with `uipath-json-similarity` against a reference joke
  and with `uipath-contains` on the topic word. Jokes vary between runs, so
  there is no exact-match evaluator: no field of the output is deterministic.
* [`samples/local-mcp-agent`](samples/local-mcp-agent): the three orders known
  to `orders_server.py` (`A-1001` shipped, `A-1002` processing, `A-1003`
  delivered). Each case checks that `get_order_status` was called once, with
  the requested `order_id`, and that the reply contains a detail only the tool
  returns.

```bash
cd samples/local-mcp-agent
uv sync
uv run uipath auth
uv run uipath eval agent evaluations/eval-sets/evaluation-set-default.json
```

### Tool names in tool-call evaluators

The tool-call evaluators (`uipath-tool-call-order`, `-count`, `-args`,
`-output`) match the tool spans the runtime emits. Use these names:

| Tool | Name in the evaluator |
| --- | --- |
| Shell command | `shell` |
| File edit | `apply_patch` |
| Web search | `web_search` |
| MCP tool | the plain tool name, e.g. `get_order_status` (the server is in the span's `tool.server` attribute, not in the name) |
| Durable process or agent tool | the name the model sees, e.g. `a_plus_b` |
| Integration Service tool | the name the model sees, e.g. `web_search` |
| Context grounding tool | the name the model sees, e.g. `Company_Policies` |
| Escalation tool | the name the model sees, e.g. `escalate_ExpenseApproval` |
| Analyze Files | the name the model sees, e.g. `Analyze_Files` |
| DeepRAG on an attachment | the name the model sees, e.g. `DeepRAG` |
| Batch Transform on an attachment | the name the model sees, e.g. `Batch_Transform` |
| IXP extraction / Document Validation | the name the model sees, e.g. `Extract_Invoice` |

A tool span's output is `{"content": "<text the tool returned>"}`. The
`uipath-tool-call-output` evaluator unwraps it, so a criterion's `output` is
the plain text the tool returned, for example
`{"name": "get_order_status", "output": "Shipped, arriving 2026-09-10 via the Bucharest hub."}`.

### Durable tools in evals

Mocked durable tools answer in the same turn and never suspend; only the
unmocked ones need the resume steps below. A case that calls an unmocked
durable tool starts a real job (task, extraction, ...) and suspends like
`uipath run` does. Resume works on one
evaluation at a time (`--resume` fails on a set with several evaluations), so
select the case with `--eval-ids`. Resume the eval once the jobs finish, and
pass the **same** `--eval-ids` and `--eval-set-run-id` on both calls:

```bash
RUN_ID=$(uuidgen)
uipath eval agent evaluations/eval-sets/<file>.json --eval-ids '["<case-id>"]' --eval-set-run-id "$RUN_ID"
# ... jobs finish ...
uipath eval agent evaluations/eval-sets/<file>.json --eval-ids '["<case-id>"]' --eval-set-run-id "$RUN_ID" --resume
```

Without the same id, the CLI makes a new execution id on the second call and
does not reload the spans recorded before the suspend, so tool-call evaluators
miss the calls made before the suspend.

### Mocking tools in evals

The UiPath tools the runtime serves can be simulated in `uipath eval` (and in
`uipath run` / `uipath debug` with a `simulation.json`), with the same
mechanism, names and output schemas as LangChain and low-code agents. A
simulated call starts nothing on the tenant (no job, task, extraction,
search or activity): the mocker's value is the tool's result, in the same
turn, and the run does not suspend on a simulated durable tool. The LLM
strategy still calls the tenant's LLM Gateway to generate the value (see
below); mockito and simulated components need no gateway.

What is honored (core's own parsing, no Codex-specific schema):

| Source | Fields | Mocker |
| --- | --- | --- |
| Eval set `version: "1.0"` | `mockingStrategy: {"type": "llm", "prompt", "toolsToSimulate": [{"name"}], "model"?}` | LLM |
| Eval set `version: "1.0"` | `mockingStrategy: {"type": "mockito", "config": [{"function", "arguments"?, "then": [{"type": "return"\|"raise", "value"}]}]}` | mockito |
| Legacy (Studio Web) eval set | `simulateTools`, `simulationInstructions`, `toolsToSimulate` | LLM |
| `simulation.json` / `--simulation` | `{enabled, toolsToSimulate, instructions, model}` or `{components: [...]}` | LLM / simulate-component |

Names match as the mockers match them: exactly for `toolsToSimulate` and a
mockito `function`; exactly or with `_` read as a space for `components`.
The name a tool is simulated by is LangChain's `resource.name`, not always
the name the model sees (which the tool-call evaluators use, see above):

| Tool | Simulated as | Mocked value |
| --- | --- | --- |
| Process / agent | the process name passed to the factory, e.g. `a_plus_b`; from a resource, its `name`, e.g. `Create Invoice` (the span is named after `properties.processName`, see Tracing below) | the job output (`output_schema`) |
| Escalation | the model-facing name lowercased, e.g. `escalate_expenseapproval` | `{"action": "approve"\|"reject", "data": {...}, "is_deleted": false}` |
| IXP extraction | the display / resource name, e.g. `Extract Invoice` | an `ExtractionResponseIXP` (`extractionResult`, `projectId`, ..., `dataProjection`) |
| Document Validation | the display / resource name, e.g. `Validate Invoice` | `{"data": [<field groups>] \| null}` |
| DeepRAG (index) | the display / resource name, e.g. `Invoice Research` | `{"text", "citations", "deepRagId"}` |
| DeepRAG / Batch Transform on an attachment | the name passed to the factory / resource name, e.g. `Batch Transform` | the resource's `outputSchema`, else DeepRAG's; `{"result": <JobAttachment>}` |
| Integration Service | the display / resource name, e.g. `Get User (v2)` (not `Get_User_v2`) | the activity result (`output_schema`) |
| Context grounding search | the display / resource name, e.g. `Company Policies` | `{"documents": [{"metadata", "page_content"}]}` |
| Analyze Files | the display / resource name, e.g. `Analyze Files` | the resource's `outputSchema`, else `{"analysisResult"}` |
| Your own `UiPathTool` / `UiPathDurableTool` | `mock_name`, else `display_name`, else `name` | `output_json_schema`; a string is the text, anything else JSON |

A mockito behavior matches keyword arguments, and every one must match:

| Tool | Keyword arguments the mocker sees |
| --- | --- |
| Escalation, process / agent, DeepRAG / Batch Transform on an attachment | the model's validated arguments with the pre-filled fields (`static_args`, `args_from_input`, `attachment_from_input`, `argumentProperties`) merged in, as in LangChain; a required agent-input field neither the agent input nor the model filled is a tool error |
| IXP, Document Validation, DeepRAG on an index, your own tools | the model's validated arguments |
| Context grounding search | `{"query": ...}`: the query searched for (the agent input's when it is set from it); `{}` for a static query |
| Integration Service | the activity input: the model's arguments with the static and agent-input values merged in |
| Analyze Files | the model's arguments with the pre-filled values merged in; a required agent-input field neither the agent input nor the model filled is a tool error |

An attachment the model passes as `{"ID": ...}` is the full JobAttachment
`{"ID", "FullName", "MimeType"}` by then, as in LangChain.

For example:

```json
{
  "mockingStrategy": {
    "type": "mockito",
    "config": [
      {
        "function": "a_plus_b",
        "arguments": {"args": [], "kwargs": {"a": 2, "b": 3}},
        "then": [{"type": "return", "value": {"sum": 5}}]
      }
    ]
  }
}
```

* A tool with no behavior (or not in `toolsToSimulate`) runs for real. Mockito
  arguments no behavior matches, a `raise` answer, or a failed LLM
  simulation come back to the model as a tool error.
* The mocked value goes through the tool's own result handling: an escalation
  `outcome_mapping` to `end`, or `is_deleted`, ends the run
  (`TERMINATION_ESCALATION_REJECTED`); a simulated IXP extraction is kept for
  the validation tool; attachments in a simulated result become known to the
  run.
* `uipath_process_tool(..., example_calls=[...])` /
  `uipath_agent_tool(..., example_calls=[...])` give the LLM mocker examples;
  tools from an `agent.json` use the resource's `exampleCalls` (Integration
  Service, IXP and escalation channels).
* Tracing, as low-code: `Tool call - <tool>` keeps its output; the tool's
  child span (`processTool` / `agentTool`, `escalationTool`, `ixpTool`,
  `vsEscalationTool`, `contextGroundingTool`, `integrationTool`, the internal
  tool span) keeps its type, gets `__uipath_response_mocked: true` and is
  renamed `Simulated result: <name>`. A custom tool has no child span: the
  attribute sits on its `Tool call` span, which is never renamed. A process
  or agent tool from a resource names its child after
  `properties.processName` (`Simulated result: CreateInvoice`; the mocker's
  function name is the resource `name`, `Create Invoice`). A Flow resource
  has no `processTool` child, as in low-code: the attribute sits on its
  `Tool call` span, as for a custom tool.
* The LLM mocker calls the LLM Gateway over chat completions with
  `mockingStrategy.model`, else the agent's model (`metadata.settings.model`
  of the schema). If the agent's model only answers on the Responses API, set
  `mockingStrategy.model` to one that answers chat completions.
* The LLM mocker answers in the tool's output schema. Give tools you declare
  in code an `output_schema` (process, agent and Integration Service tools):
  without one the schema is an empty object, and the mocker may return `{}`
  or copy the `{"content": ...}` shape of earlier tool spans. Tools from an
  `agent.json` use the resource's `outputSchema`.
* Cannot be mocked: tools of your own MCP servers (`mcp_servers`; the Codex
  binary calls them directly, with no Python in between) and Codex built-ins
  (shell, `apply_patch`, web search). They run for real in every eval case.

## Not supported

* **Tool confirmation in a chat** (`requireConfirmation`) and client-side
  tools (see "Conversational agents").
* **Breakpoints in `uipath debug`.** Codex offers no hook to stop between
  steps, so the runtime logs a warning and the run continues to completion.
* **Models other than OpenAI** (see [Models](#models)).
* **Mocking your own MCP servers' tools in evals** (see "Mocking tools in
  evals").
* **Hiding secrets from built-in tool commands** (see
  [Built-in tools and the sandbox](#built-in-tools-and-the-sandbox)).
