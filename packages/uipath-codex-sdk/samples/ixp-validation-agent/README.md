# Invoice extraction with human validation (IXP + Document Validation, durable)

A Codex agent that extracts an invoice's fields with IXP and has a person
validate them in Action Center:

| Step | Tool | The run |
| --- | --- | --- |
| 1 | `Extract_Invoice`: the model passes the invoice attachment | suspends until the IXP extraction ends; the model gets the fields |
| 2 | `Validate_Invoice`: no arguments | creates a Validation Station task for the last extraction and suspends until it is completed; the model gets the validated fields |

A document marked as an exception in Validation Station ends the run
(`TERMINATION_ESCALATION_REJECTED`), as in low-code agents. See the main
README's "IXP extraction and Document Validation" section.

## Prerequisites

* An IXP project with a published model version and its tag (e.g. `live`):
  set `PROJECT_NAME` / `VERSION_TAG` in `main.py`.
* An Orchestrator storage bucket: `STORAGE_BUCKET`, in folder `BUCKET_FOLDER`
  (the Validation Station task is created in that folder too).
* Someone who can open the task in Action Center.

## Run locally

```bash
cd samples/ixp-validation-agent
uv sync
uv run uipath auth                                 # pick the tenant, e.g. --alpha
uv run python upload.py invoice.pdf                # writes input.json
export UIPATH_FOLDER_PATH=Coded                    # the folder of the bucket and the task
uv run uipath run agent --file input.json          # suspends on the extraction
uv run uipath run agent --resume                   # once extracted: suspends on the validation task
# ... validate the document in Action Center ...
uv run uipath run agent --resume                   # the validated fields
```

## Run on Orchestrator

Publish the agent and start a job, picking the invoice in the `invoice`
input. The job suspends twice; Orchestrator resumes it when the extraction
ends and when the task is completed.

## Evaluate with mocked tools

`evaluations/eval-sets/evaluation-set-mocked.json` runs the agent with its
UiPath tools mocked, so it needs no tenant resources besides the LLM.

No extraction or Validation Station task is created and the run never
suspends. `Extract Invoice` returns a full `ExtractionResponseIXP` (kept for
the validation tool, as a real one would be) and `Validate Invoice` returns
the validated fields, with the total corrected from 4,200.00 to 4,250.00 USD.
The input's attachment id is a placeholder.

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-mocked.json --no-report
```

See the main README's "Mocking tools in evals" section.

## Trace

```
Agent run - agent                       attachments: invoice.pdf (in)
├─ LLM call
├─ Tool call - Extract_Invoice          toolType IxpExtraction; open while suspended
│  └─ Extract Invoice                   ixpTool: projectName, versionTag, extractionId, documentId, result
├─ LLM call
├─ Tool call - Validate_Invoice         toolType Escalation; open while suspended
│  └─ Validate Invoice                  vsEscalationTool: ixpToolId, storageBucketName, operationId, taskUrl, result
├─ LLM call
└─ Agent output
```
