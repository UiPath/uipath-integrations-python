# Invoice questions (Context Grounding tool)

A Codex agent that answers questions about invoices from a Context Grounding
index, through one context tool:

| Tool | Index | Model provides | Returns |
| --- | --- | --- | --- |
| `search_invoices` | `invoices` (folder `Coded`), semantic search, 5 results | `query` | `{"documents": [{"metadata": {source, search_id, reference, page_number, score}, "page_content": ...}]}` |

The tool is *immediate*: the call is answered when the search returns and the
run never suspends. See the main README's "Context grounding tools" section
for the query variants (fixed, from the agent input) and the result shape.

## Prerequisites

* A Context Grounding index with some documents, ingested. This sample uses
  an index named `invoices` in the `Coded` folder, on a storage bucket with an
  invoice. For another index, change `INDEX_NAME` / `INDEX_FOLDER` in
  `main.py` and the binding in `bindings.json` (its key is
  `<name>.<folder>`).
* Read access to the index's folder for whoever runs the agent.

```bash
cd samples/context-grounding-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json
```

The output is structured: `{answer, invoice_numbers, sources}`. For example,
for "How much does Acme Corp owe us, and by when?": `{"answer": "Acme Corp
owes $4,200, due by 2026-03-15.", "invoice_numbers": ["INV-7421"], "sources":
["invoice.jpg"]}`.

## Rebinding the index

`bindings.json` declares the index as a bindable `index` resource. When the
agent is deployed, Orchestrator can point the binding at another index or
folder. The tool passes the index name and folder through the SDK's resource
override on every call, so no code change is needed.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

The index is not searched: `Search Invoices` (the tool's name, not the
model-facing `search_invoices`) is answered by an LLM in the `{"documents":
[...]}` shape of a real search.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - search_invoices        toolType ContextGrounding, index_name, context_retrieval_mode
│  └─ Search Invoices                 contextGroundingTool: retrieval_mode, query, number_of_results, output (documents)
│     └─ ContextGroundingRetriever    RETRIEVER: input.value (query), output.value (documents), read by the evaluators
│        ├─ folder_retrieve_key       (when the index is not in the job's folder)
│        └─ contextgrounding_unified_search → contextgrounding_retrieve
├─ LLM call
└─ Agent output
```
