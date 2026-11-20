# Document research (DeepRAG on a job attachment, durable)

A Codex agent that researches an attached document with DeepRAG:

| Input | Tool | Output |
| --- | --- | --- |
| `document` (a job attachment, PDF or TXT), `question` | `DeepRAG`: the model passes `attachment` and `query` | `{answer, sources}` |

The tool is **durable**: the call starts a DeepRAG task on the file (the
service builds a temporary index of it, there is no separate indexing step)
and the run **suspends**. It resumes when the task ends; a failed task comes
back to the model as `{"status": "failed", "error": ...}`. See the main
README's "DeepRAG on a job attachment" section.

## Run locally

```bash
cd samples/deep-rag-attachments-agent
uv sync
uv run uipath auth                                    # pick the tenant, e.g. --alpha
uv run python upload.py report.pdf "What are the main findings?"   # writes input.json
export UIPATH_FOLDER_PATH=Shared                       # DeepRAG needs a folder to run in
uv run uipath run agent --file input.json             # suspends on the DeepRAG task
# ... wait a minute or two ...
uv run uipath run agent --resume                      # resumes; suspends again if still running
```

DeepRAG reads PDF and TXT files; another file type is refused when the task
starts, and the model gets the error.

## Run on Orchestrator

Publish the agent and start a job, picking the file in the `document` input.
The job suspends while the task runs and Orchestrator resumes it when the
task ends.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

The document is not uploaded and the run never suspends: `DeepRAG` returns a
fixed `{text, citations, deepRagId}`. The input's attachment id is a
placeholder.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent                  attachments: the document (in)
├─ LLM call
├─ Tool call - DeepRAG             toolType Internal, attachments: the document (in); open while suspended
│  └─ DeepRAG                      the internal tool: input {attachment, query}, deep_rag_id, output
│     └─ contextgrounding_start_deep_rag_from_attachments_async
├─ contextgrounding_retrieve_deep_rag_async   (resume: reading the task)
├─ LLM call
└─ Agent output
```
