# Finance research (DeepRAG tool, durable)

A Codex agent that answers finance research questions with a DeepRAG task on
a Context Grounding index:

| Tool | Index | Model provides | Returns |
| --- | --- | --- | --- |
| `research_finance` | `finance-resources` (folder `Coded`), citation mode `Inline` | `query` (the research task) | `{"text": ..., "citations": [{ordinal, page_number, source, reference}], "deepRagId": ...}` |

The tool is **durable**: the call starts the DeepRAG task and the run
**suspends**. The run resumes when the task ends; a failed task comes back to
the model as `{"status": "failed", "error": ...}`. See the main README's
"DeepRAG (durable)" section.

## Prerequisites

* A Context Grounding index with PDF or TXT documents, ingested (DeepRAG
  refuses an index without them: "No valid files to use for Summarization").
  This sample uses the `finance-resources` index in the `Coded` folder, with
  finance PDFs such as the Federal Reserve's "The Fed Explained". For another
  index,
  change `INDEX_NAME` / `INDEX_FOLDER` in `main.py` and the binding in
  `bindings.json` (key `<name>.<folder>`).
* Read access to the index's folder for whoever runs the agent.

## Run locally

```bash
cd samples/deep-rag-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json     # suspends on the DeepRAG task
# ... wait a minute or two for the task to finish ...
uv run uipath run agent --resume              # resumes; suspends again if it is still running
```

The output is structured: `{answer, sources}`.

## Run on Orchestrator

Publish the agent and start a job: the job suspends while the task runs, and
Orchestrator resumes it when the task ends (the DeepRAG resume trigger).

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

No DeepRAG task is started and the run never suspends: `Research Finance
Resources` (the tool's name) returns a fixed `{text, citations, deepRagId}`.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - research_finance       toolType ContextGrounding, open while the run is suspended
│  └─ Research Finance Resources      contextGroundingTool: retrieval_mode DeepRAG, query, citation_mode, deep_rag_id, output
│     └─ folder_retrieve_key, contextgrounding_start_deep_rag_async → contextgrounding_retrieve
├─ contextgrounding_retrieve_deep_rag_async   (resume: reading the task)
├─ LLM call
└─ Agent output
```
