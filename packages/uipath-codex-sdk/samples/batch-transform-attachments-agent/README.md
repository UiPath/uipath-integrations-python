# Review enrichment (Batch Transform on a job attachment, durable)

A Codex agent that adds columns to a CSV attachment with Batch Transform:

| Input | Tool | Output |
| --- | --- | --- |
| `reviews` (a CSV job attachment), `task` | `Enrich_Reviews`: the model passes `attachment` and `query`; columns `sentiment`, `topic` | `{summary, enriched}`, `enriched` being the result CSV |

The tool is **durable**: the call starts a Batch Transform task on the file
(the service indexes it on the fly) and the run **suspends**. When the task
ends the run resumes: the result is downloaded, uploaded as a job attachment
(`output.csv`, linked to the job) and returned to the model as a
JobAttachment, which the agent puts in its output. A failed task fails the
run. See the main README's "Batch Transform on a job attachment" section.

## Run locally

```bash
cd samples/batch-transform-attachments-agent
uv sync
uv run uipath auth                                     # pick the tenant, e.g. --alpha
uv run python upload.py reviews.csv "Classify each review by sentiment and topic."
export UIPATH_FOLDER_PATH=Shared                        # the task needs a folder to run in
uv run uipath run agent --file input.json              # suspends on the task
# ... wait a minute or two ...
uv run uipath run agent --resume                       # resumes; suspends again if still running
```

`enriched.ID` in the output is the uploaded result; download it with
`UiPath().attachments.download(key=..., destination_path=...)`.

## Run on Orchestrator

Publish the agent and start a job, picking `reviews.csv` in the `reviews`
input. The job suspends while the task runs and Orchestrator resumes it when
the task ends; the result is attached to the job.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

No Batch Transform task is started and the run never suspends: `Enrich
Reviews` returns a `result` attachment, which the agent returns as `enriched`.
The attachment ids are placeholders; the simulated result file does not exist.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent                          attachments: reviews.csv (in)
├─ LLM call
├─ Tool call - Enrich_Reviews              toolType ContextGrounding; attachments: reviews.csv (in), output.csv (out)
│  └─ Enrich_Reviews                       contextGroundingTool: retrieval_mode BatchTransform, query, output_columns,
│     │                                    batch_transform_id, results (output.csv); open while suspended
│     └─ contextgrounding_start_batch_transform_from_attachment_async
├─ contextgrounding_retrieve_batch_transform_async, attachments_upload   (resume)
├─ LLM call
└─ Agent output                            attachments: output.csv (out)
```
