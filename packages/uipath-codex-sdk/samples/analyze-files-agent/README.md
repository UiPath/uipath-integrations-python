# Document Q&A (Analyze Files, job attachments)

A Codex agent that answers a question about an attached document:

| Input | Tool | Output |
| --- | --- | --- |
| `document` (a job attachment), `question` | `Analyze_Files`: the file and an analysis task go to the agent's model | `{answer, source}`, `source` being the input document |

As in LangChain agents, the model never opens the file itself: it sees the
attachment as a JobAttachment object (`{"ID", "FullName", "MimeType"}`) and
passes it to the tool. See the main README's "Job attachments" section.

## Run locally

```bash
cd samples/analyze-files-agent
uv sync
uv run uipath auth                                  # pick the tenant, e.g. --alpha
uv run python upload.py invoice.pdf "What is the total amount due?"   # writes input.json
uv run uipath run agent --file input.json
```

`upload.py` uploads the file as an Orchestrator attachment and writes its
`{ID, FullName, MimeType}` into `input.json`. Images (PNG, JPEG, GIF, WebP)
and documents such as PDFs work; a file over 30 MB fails the run.

## Run on Orchestrator

Publish the agent and start a job: the `document` input is a file picker
(the schema marks it as a `JobAttachment`).

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

The document is not downloaded or sent to a model: `Analyze Files` returns a
fixed `analysisResult`. The input's attachment id is a placeholder, which is
fine while the tool is mocked.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent                    attachments: the document (in)
├─ LLM call
├─ Tool call - Analyze_Files         toolType Internal, attachments: the document (in)
│  └─ Analyze Files                  input {attachments, analysisTask}, output {analysisResult}
│     ├─ attachments_get_blob_uri
│     └─ LLM call                    the task and the file, attachments: the document (in)
├─ LLM call
└─ Agent output                      attachments: the document (out)
```
