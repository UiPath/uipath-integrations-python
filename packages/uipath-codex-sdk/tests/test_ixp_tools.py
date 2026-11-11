"""IXP extraction and Document Validation tools: builders, routing, finish."""

from __future__ import annotations

import json
from typing import Any

import pytest

from uipath_codex_sdk import (
    UiPathDocumentValidationTool,
    UiPathDurableToolPending,
    escalation_tool_from_resource,
    ixp_extraction_tool_from_resource,
    uipath_document_validation_tool,
    uipath_ixp_extraction_tool,
    vs_escalation_tool_from_resource,
)
from uipath_codex_sdk.attachments import attachment_paths
from uipath_codex_sdk.runtime.errors import UiPathRunTerminatedError
from uipath_codex_sdk.tool_results import ToolResultStore

from .test_escalation_tools import HITL_RESOURCE

IXP_RESOURCE: dict[str, Any] = {
    "$resourceType": "tool",
    "type": "ixp",
    "name": "Invoice Extraction",
    "description": "Extracts invoice fields.",
    "inputSchema": {"type": "object", "properties": {}},
    "outputSchema": {"type": "object", "properties": {}},
    "properties": {"projectName": "Invoices", "versionTag": "live"},
}


def _vs_resource(**props: Any) -> dict[str, Any]:
    resource = json.loads(json.dumps(HITL_RESOURCE))
    resource["name"] = "Validate Invoice"
    resource["escalationType"] = 1
    resource["vsEscalationProperties"] = {
        "ixpToolId": "Invoice Extraction",
        "storageBucketName": "invoices-bucket",
        "storageBucketFolderPath": "incoming",
        **props,
    }
    return resource


def test_extraction_factory_and_resource():
    tool = uipath_ixp_extraction_tool(
        "Invoice Extraction", project_name="Invoices", version_tag="live"
    )
    assert tool.name == "Invoice_Extraction" and tool.result_key == "Invoice Extraction"
    assert tool.tool_type == "IxpExtraction"
    assert attachment_paths(tool.input_schema) == [("attachment",)]
    built = ixp_extraction_tool_from_resource(IXP_RESOURCE)
    assert (built.project_name, built.version_tag) == ("Invoices", "live")
    # An input schema without an attachment field falls back to langchain's.
    assert attachment_paths(built.input_schema) == [("attachment",)]
    no_tag = json.loads(json.dumps(IXP_RESOURCE))
    no_tag["properties"]["versionTag"] = None
    with pytest.raises(ValueError, match="Model is required"):
        ixp_extraction_tool_from_resource(no_tag)
    with pytest.raises(ValueError, match="disabled"):
        ixp_extraction_tool_from_resource({**IXP_RESOURCE, "isEnabled": False})


def test_validation_resource_and_routing():
    tool = vs_escalation_tool_from_resource(_vs_resource())
    assert isinstance(tool, UiPathDocumentValidationTool)
    assert tool.extraction_tool == "Invoice Extraction"
    assert tool.storage_bucket_name == "invoices-bucket"
    # The bucket's Orchestrator folder: where DU looks the bucket up.
    assert tool.action_folder == "incoming"
    assert tool.storage_bucket_directory_path is None
    assert tool.input_schema["properties"] == {}
    assert tool.tool_type == "Escalation"
    # escalationType 1 is routed by the general helper, as langchain dispatches.
    routed = escalation_tool_from_resource(_vs_resource())
    assert isinstance(routed, UiPathDocumentValidationTool)
    with pytest.raises(ValueError, match="IXP tool is required"):
        vs_escalation_tool_from_resource(_vs_resource(ixpToolId=""))
    with pytest.raises(ValueError, match="Storage bucket is required"):
        vs_escalation_tool_from_resource(_vs_resource(storageBucketName=None))


def test_validation_factory_takes_the_extraction_tool():
    extract = uipath_ixp_extraction_tool(
        "Invoice Extraction", project_name="Invoices", version_tag="live"
    )
    tool = uipath_document_validation_tool("Validate", extraction_tool=extract)
    assert tool.extraction_tool == "Invoice Extraction"
    assert "'Invoice Extraction'" in tool.description
    assert tool.resolve_title(None) == "VS Escalation Task"


def _action(status: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "actionData": {"status": status, "data": data or {}},
        "actionStatus": status,
        "projectId": "p",
        "projectType": "IXP",
        "tag": "live",
        "operationId": "val-1",
        "extractorId": None,
        "documentTypeId": "00000000-0000-0000-0000-000000000000",
        "dataProjection": [{"fieldGroupName": "Invoice", "fieldValues": []}],
    }


async def test_validation_finish():
    tool = uipath_document_validation_tool("Validate", extraction_tool="X")
    assert json.loads(await tool.finish({}, _action("Completed"))) == {
        "data": [{"fieldGroupName": "Invoice", "fieldValues": []}]
    }
    with pytest.raises(UiPathDurableToolPending):
        await tool.finish({}, _action("Unassigned"))
    with pytest.raises(UiPathRunTerminatedError, match="Not an invoice"):
        await tool.finish(
            {},
            _action(
                "Completed", {"documentRejectionDetails": {"reason": "Not an invoice"}}
            ),
        )
    with pytest.raises(UiPathRunTerminatedError, match="deleted"):
        await tool.finish({}, _action("Deleted"))


def test_result_store_round_trip():
    store = ToolResultStore()
    store.mark_pending("Invoice Extraction")
    assert store.is_pending("Invoice Extraction") and store.changed
    again = ToolResultStore.from_json(json.loads(json.dumps(store.to_json())))
    assert again.is_pending("Invoice Extraction")
    again.put("Invoice Extraction", {"projectId": "p"})
    assert not again.is_pending("Invoice Extraction")
    assert again.get("Invoice Extraction") == {"projectId": "p"}
