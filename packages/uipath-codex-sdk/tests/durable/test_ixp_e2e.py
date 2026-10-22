"""Offline end-to-end tests of IXP extraction and its Document Validation task.

The real factory, runtime, bundled codex, gateway shim, tool host and trigger
handler (whose IXP readers run against a fake documents service), with a
mock model that extracts a document, then sends it to validation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.core.triggers import UiPathResumeTriggerName, UiPathResumeTriggerType
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory

from .._typing import sqlite_storage
from ..contract.mock_responses import (
    advertised_tools,
    find_tool,
    function_call_item,
    input_items,
    message_item,
)

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
DOC_ID = "abcdabcd-0000-0000-0000-000000000001"
DOC = {"ID": DOC_ID, "FullName": "invoice.pdf", "MimeType": "application/pdf"}
EXTRACT_ID = "call_ixpextract0000000000001"
VALIDATE_ID = "call_ixpvalidate000000000001"

MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_document_validation_tool,
    uipath_ixp_extraction_tool,
)


class Input(BaseModel):
    invoice: UiPathAttachment


extract = uipath_ixp_extraction_tool(
    "Invoice Extraction", project_name="Invoices", version_tag="live"
)

agent = UiPathCodexAgent(
    model="mock-model",
    input_schema=Input,
    prompt="PLEASE_CALL_DURABLE: extract {invoice}, then have it validated.",
    tools=[
        extract,
        uipath_document_validation_tool(
            "Validate Invoice",
            extraction_tool=extract,
            storage_bucket_name="invoices-bucket",
            title="Check the invoice",
        ),
    ],
    durable_settle_timeout=20.0,
)
"""


def _text(output: Any) -> Any:
    if isinstance(output, list):
        texts = [p.get("text", "") for p in output if isinstance(p, dict)]
        return texts[-1] if texts else ""
    return output


class Script:
    """Extract the document; once its fields are back, validate; then answer."""

    def __init__(self, validate_first: bool = False) -> None:
        self.validate_first = validate_first

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        outputs = {
            it["call_id"]: _text(it.get("output"))
            for it in items
            if it.get("type") == "function_call_output"
        }
        tools = advertised_tools(body)
        extract, ns = find_tool(tools, "Invoice_Extraction")
        validate, vns = find_tool(tools, "Validate_Invoice")
        if self.validate_first and VALIDATE_ID not in seen and validate:
            return [function_call_item(VALIDATE_ID, validate, {}, vns)]
        if EXTRACT_ID not in seen and extract:
            args = {"attachment": {"ID": DOC_ID}}
            return [function_call_item(EXTRACT_ID, extract, args, ns)]
        if (
            not self.validate_first
            and EXTRACT_ID in outputs
            and VALIDATE_ID not in seen
            and validate
        ):
            return [function_call_item(VALIDATE_ID, validate, {}, vns)]
        return [message_item("FINAL " + json.dumps(outputs, sort_keys=True), idx)]


@pytest.fixture
def project(project: Path, monkeypatch) -> Path:
    (project / "main.py").write_text(MAIN)
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Coded")
    return project


@pytest.fixture
def spans():
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]
    yield exporter
    exporter.shutdown()


async def _run(project: Path, input, *, resume: bool):
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
        )
    )
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        result = await runtime.execute(input, UiPathExecuteOptions(resume=resume))
        storage = await sqlite_storage(factory)
        return result, await storage.get_triggers(RUNTIME_ID)
    finally:
        await runtime.dispose()
        await factory.dispose()


def _new_robot(project: Path) -> None:
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _final_outputs(result) -> dict:
    text = result.output["output"]
    assert text.startswith("FINAL "), text
    return json.loads(text[len("FINAL ") :])


async def test_extract_then_validate(project, upstream, world, spans):
    upstream.script = Script()
    docs = world.documents

    # --- run: download + start the extraction, suspend ------------------
    result, triggers = await _run(project, {"invoice": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    [start] = docs.starts
    assert (start["project_name"], start["tag"]) == ("Invoices", "live")
    assert start["content"] == f"document {DOC_ID}"  # the downloaded attachment
    assert Path(start["file_path"]).name == "invoice.pdf"
    assert not Path(start["file_path"]).exists()  # the temporary copy is gone
    assert result.output[EXTRACT_ID]["extraction_id"] == start["operation_id"]
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.IXP_EXTRACTION
    assert trig.trigger_name == UiPathResumeTriggerName.EXTRACTION
    assert trig.item_key == start["operation_id"]
    assert (trig.payload["project_id"], trig.payload["tag"]) == (
        docs.PROJECT_ID,
        "live",
    )

    # --- still running: suspends again ----------------------------------
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED

    # --- extracted: the model gets the fields and validates -------------
    docs.finish_extraction(start["operation_id"])
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    [validation] = docs.validation_starts
    assert validation["storage_bucket_name"] == "invoices-bucket"
    assert (
        validation["action_folder"] == "Coded"
    )  # the job's: DU finds the bucket there
    assert validation["action_title"] == "Check the invoice"
    # The full extraction response, kept from the first suspension.
    assert validation["extraction_response"].extraction_result.document_id == (
        f"doc-{start['operation_id']}"
    )
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.IXP_VS_ESCALATION
    assert trig.item_key == validation["operation_id"]

    # --- validated: the model gets the corrected fields -----------------
    docs.complete_validation(validation["operation_id"], total="4250")
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    outputs = _final_outputs(result)
    extracted = json.loads(outputs[EXTRACT_ID])
    assert extracted[0]["fieldValues"][0]["value"] == "4200"
    validated = json.loads(outputs[VALIDATE_ID])
    assert validated["data"][0]["fieldValues"][0]["value"] == "4250"

    finished = spans.get_finished_spans()
    [ext_call] = [s for s in finished if s.name == "Tool call - Invoice_Extraction"]
    [ext] = [s for s in finished if s.name == "Invoice Extraction"]
    assert ext.parent.span_id == ext_call.context.span_id
    assert ext_call.attributes["toolType"] == "IxpExtraction"
    assert ext.attributes["span_type"] == "ixpTool"
    assert ext.attributes["projectName"] == "Invoices"
    assert ext.attributes["versionTag"] == "live"
    assert ext.attributes["extractionId"] == start["operation_id"]
    assert "result" in ext.attributes
    [val_call] = [s for s in finished if s.name == "Tool call - Validate_Invoice"]
    [val] = [s for s in finished if s.name == "Validate Invoice"]
    assert val.parent.span_id == val_call.context.span_id
    assert val_call.attributes["toolType"] == "Escalation"
    assert val.attributes["span_type"] == "vsEscalationTool"
    assert val.attributes["ixpToolId"] == "Invoice Extraction"
    assert val.attributes["storageBucketName"] == "invoices-bucket"
    assert val.attributes["operationId"] == validation["operation_id"]
    assert val.attributes["taskUrl"].endswith(validation["operation_id"])


async def _to_validation(project, upstream, world) -> dict[str, Any]:
    upstream.script = Script()
    docs = world.documents
    result, _ = await _run(project, {"invoice": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    docs.finish_extraction(docs.starts[0]["operation_id"])
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    return docs.validation_starts[0]


async def test_document_marked_as_exception_ends_the_run(project, upstream, world):
    validation = await _to_validation(project, upstream, world)
    world.documents.complete_validation(
        validation["operation_id"], rejection="Not an invoice"
    )
    _new_robot(project)
    with pytest.raises(Exception) as info:
        await _run(project, None, resume=True)
    text = str(getattr(info.value, "error_info", None) or info.value)
    assert "TERMINATION_ESCALATION_REJECTED" in text
    assert "Not an invoice" in text


async def test_deleted_validation_task_fails_the_run(project, upstream, world):
    validation = await _to_validation(project, upstream, world)
    world.documents.validations[validation["operation_id"]]["status"] = "Deleted"
    _new_robot(project)
    with pytest.raises(Exception) as info:
        await _run(project, None, resume=True)
    assert "deleted" in str(info.value).lower()


async def test_failed_extraction_fails_the_run(project, upstream, world):
    upstream.script = Script()
    result, _ = await _run(project, {"invoice": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    world.documents.finish_extraction(
        world.documents.starts[0]["operation_id"], "Failed"
    )
    _new_robot(project)
    with pytest.raises(Exception) as info:
        await _run(project, None, resume=True)
    assert "failed" in str(info.value).lower()
    assert world.documents.validation_starts == []


async def test_start_error_fails_the_run(project, upstream, world):
    world.documents.fail_start = ValueError("Project 'Invoices' not found.")
    upstream.script = Script()
    with pytest.raises(Exception) as info:
        await _run(project, {"invoice": DOC}, resume=False)
    error = info.value
    assert "IXP_FAILED" in str(getattr(error, "error_info", None) or error)
    assert "Project 'Invoices' not found." in str(error)


async def test_validation_before_extraction_is_a_tool_error(project, upstream, world):
    upstream.script = Script(validate_first=True)
    result, _ = await _run(project, {"invoice": DOC}, resume=False)
    # The validation call was refused; the extraction then suspended the run.
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert world.documents.validation_starts == []
    assert len(world.documents.starts) == 1
