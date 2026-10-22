"""IXP extraction + Document Validation: an invoice agent with a human check.

The input is an invoice (a job attachment). The agent extracts its fields with
an IXP project (``Extract_Invoice``, durable: the run suspends until the
extraction ends), then sends the extraction to a person in Action Center
(``Validate_Invoice``, a Validation Station task; the run suspends until it is
completed) and returns the validated fields. A document the reviewer marks as
an exception ends the run.

Set PROJECT_NAME / VERSION_TAG to your IXP project and its published model's
tag, STORAGE_BUCKET to an Orchestrator storage bucket and BUCKET_FOLDER to its
folder.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_document_validation_tool,
    uipath_ixp_extraction_tool,
)

PROJECT_NAME = "invoices"
VERSION_TAG = "staging"
STORAGE_BUCKET = "ixp-escalations"
#: Folder of the bucket (the task is created there too); None means the job's.
BUCKET_FOLDER = "Coded"


class Request(BaseModel):
    invoice: UiPathAttachment = Field(description="The invoice to process")


class Output(BaseModel):
    vendor: str | None = Field(None, description="The vendor, as validated")
    total: str | None = Field(None, description="The total amount, as validated")
    due_date: str | None = Field(None, description="The due date, as validated")
    changed: list[str] = Field(
        description="Fields the reviewer corrected (empty when none were)"
    )


extract = uipath_ixp_extraction_tool(
    "Extract Invoice",
    project_name=PROJECT_NAME,
    version_tag=VERSION_TAG,
    description="Extracts the fields of an invoice document. Takes a minute.",
)

agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Process the attached invoice: extract its fields with Extract_Invoice, "
        "then send the extraction for human validation with Validate_Invoice "
        "(only after the extraction has returned). Answer from the validated "
        "fields, and list the fields the reviewer changed."
    ),
    tools=[
        extract,
        uipath_document_validation_tool(
            "Validate Invoice",
            extraction_tool=extract,
            storage_bucket_name=STORAGE_BUCKET,
            action_folder=BUCKET_FOLDER,
            title="Validate the extracted invoice",
        ),
    ],
)
