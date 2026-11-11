"""DeepRAG on a job attachment: schema, resource builder, start and finish."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from uipath.platform.common import WaitDeepRagRaw

import uipath_codex_sdk.context_tools as ct
import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    AttachmentStore,
    DurableToolContext,
    UiPathAttachment,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    analyze_files_tool_from_resource,
    context_tool_from_resource,
    deep_rag_tool_from_resource,
    uipath_deep_rag_attachments_tool,
)
from uipath_codex_sdk.attachments import attachment_paths, not_found_message

DOC_ID = "cccccccc-0000-0000-0000-000000000001"
DOC = {"ID": DOC_ID, "FullName": "contract.pdf", "MimeType": "application/pdf"}

#: Agent Builder's default resource (``type`` lowercase, as it writes it).
RESOURCE: dict[str, Any] = {
    "$resourceType": "tool",
    "type": "internal",
    "name": "Deep RAG (v2)",
    "description": "Research the attached document.",
    "inputSchema": ct.deep_rag_attachments_input_schema(),
    "outputSchema": {"type": "object", "properties": {}},
    "properties": {
        "toolType": "deep-rag",
        "settings": {
            "contextType": "attachments",
            "query": {"variant": "dynamic", "description": "What to research"},
            "folderPathPrefix": {},
            "citationMode": {"value": "Inline"},
            "fileExtension": {"value": "pdf"},
        },
    },
    "settings": {},
}


class _Grounding:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def start_deep_rag_from_attachments_async(self, **kwargs: Any) -> Any:
        from uipath.platform.context_grounding import DeepRagCreationResponse

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return DeepRagCreationResponse.model_validate(
            {
                "id": "dr-1",
                "name": kwargs["name"],
                "createdDate": "2026-10-21T00:00:00Z",
                "lastDeepRagStatus": "Queued",
            }
        )


class _Folders:
    def __init__(self, key: str | None = "fk-looked-up") -> None:
        self.key = key
        self.lookups: list[str] = []

    async def retrieve_key_async(self, *, folder_path: str) -> str | None:
        self.lookups.append(folder_path)
        return self.key

    def retrieve_key(self, **kwargs: Any) -> Any:
        raise AssertionError("blocking folder lookup")


@pytest.fixture
def client(monkeypatch):
    holder = SimpleNamespace(context_grounding=_Grounding(), folders=_Folders())
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: holder)
    monkeypatch.setattr(ct, "_folder_keys", {})
    for var in ("UIPATH_FOLDER_KEY", "UIPATH_FOLDER_PATH"):
        monkeypatch.delenv(var, raising=False)
    return holder


def _ctx(agent_input: dict[str, Any] | None = None) -> DurableToolContext:
    store = AttachmentStore([UiPathAttachment.model_validate(DOC)])
    return DurableToolContext("call_1", "rt", "DeepRAG", agent_input, store, "m")


# --- schema and builders ---------------------------------------------------


def test_default_schema_and_names():
    tool = uipath_deep_rag_attachments_tool("Deep RAG (v2)")
    assert tool.name == "Deep_RAG_v2" and tool.display_name == "Deep_RAG_v2"
    assert tool.tool_type == "Internal" and tool.kind == "internal"
    schema = tool.input_schema
    assert schema["required"] == ["attachment", "query"]
    assert schema["properties"]["query"]["description"] == (
        ct.DEEP_RAG_ATTACHMENTS_QUERY_DESCRIPTION
    )
    assert attachment_paths(schema) == [("attachment",)]
    fixed = uipath_deep_rag_attachments_tool(
        query="Summarize it.", attachment_from_input="{{document}}"
    )
    # The attachment stays visible, optional, for when the input has none.
    assert list(fixed.input_schema["properties"]) == ["attachment"]
    assert fixed.input_schema["required"] == []
    assert (
        "Set from the agent input 'document'"
        in (fixed.input_schema["properties"]["attachment"]["description"])
    )
    assert attachment_paths(fixed.input_schema) == [("attachment",)]


def test_from_resource_registry_default():
    tool = deep_rag_tool_from_resource(RESOURCE)
    assert tool.name == "Deep_RAG_v2"
    assert tool.description == "Research the attached document."
    assert tool.citation_mode.value == "Inline"
    assert tool.input_schema["properties"]["query"]["description"] == (
        "What to research"
    )
    assert RESOURCE["inputSchema"]["required"] == ["attachment"]  # not mutated


def test_from_resource_refusals():
    static_empty = json.loads(json.dumps(RESOURCE))
    static_empty["properties"]["settings"]["query"] = {"variant": "static"}
    with pytest.raises(ValueError, match="make the query dynamic"):
        deep_rag_tool_from_resource(static_empty)
    no_citation = json.loads(json.dumps(RESOURCE))
    del no_citation["properties"]["settings"]["citationMode"]
    with pytest.raises(ValueError, match="Not a valid internal tool resource"):
        deep_rag_tool_from_resource(no_citation)
    with pytest.raises(ValueError, match="disabled"):
        deep_rag_tool_from_resource({**RESOURCE, "isEnabled": False})
    with pytest.raises(ValueError, match="internal tool"):
        analyze_files_tool_from_resource(RESOURCE)  # a deep-rag resource
    with pytest.raises(ValueError, match="use an internal 'deep-rag' tool"):
        context_tool_from_resource(
            {
                "name": "Files",
                "description": "d",
                "contextType": "attachments",
                "indexName": "Files",
                "settings": {
                    "resultCount": 3,
                    "retrievalMode": "DeepRAG",
                    "threshold": 0,
                },
            }
        )


def test_from_resource_static_query_and_argument_properties():
    data = json.loads(json.dumps(RESOURCE))
    data["properties"]["settings"]["query"] = {
        "variant": "static",
        "value": "Summarize the termination clauses.",
    }
    data["argumentProperties"] = {
        "attachment": {
            "variant": "argument",
            "argumentPath": "$.document",
            "isSensitive": False,
        }
    }
    tool = deep_rag_tool_from_resource(data)
    assert tool.query == "Summarize the termination clauses."
    assert list(tool.input_schema["properties"]) == ["attachment"]
    assert tool.input_schema["required"] == []
    static = json.loads(json.dumps(data))
    static["argumentProperties"]["attachment"] = {
        "variant": "static",
        "value": DOC,
        "isSensitive": False,
    }
    assert deep_rag_tool_from_resource(static).input_schema["properties"] == {}


async def test_from_resource_jsonpath_argument_property(client, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    data = json.loads(json.dumps(RESOURCE))
    data["argumentProperties"] = {
        "$['attachment']": {
            "variant": "argument",
            "argumentPath": "$['document']",
            "isSensitive": False,
        }
    }
    tool = deep_rag_tool_from_resource(data)
    assert set(tool.fixed_args) == {"attachment"}
    assert "attachment" in tool.input_schema["properties"]
    assert tool.input_schema["required"] == ["query"]
    await tool.start({"query": "Find the fees."}, _ctx({"document": DOC}))
    [call] = client.context_grounding.calls
    assert call["attachments"] == [DOC_ID]


# --- start / finish ----------------------------------------------------------


async def test_start_passes_the_attachment_and_the_job_folder(client, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    tool = uipath_deep_rag_attachments_tool(citation_mode="Skip")
    args = tool.validate_args({"attachment": DOC, "query": "Find the fees."})
    value = await tool.start(args, _ctx())
    assert isinstance(value, WaitDeepRagRaw)
    assert value.deep_rag.id == "dr-1" and value.index_folder_key == "fk-job"
    [call] = client.context_grounding.calls
    assert call["attachments"] == [DOC_ID]
    assert call["prompt"] == "Find the fees."
    assert call["citation_mode"].value == "Skip"
    assert (call["folder_key"], call["folder_path"]) == ("fk-job", None)
    assert args == {"attachment": DOC, "query": "Find the fees."}  # not mutated
    assert tool.started_output(value)["deep_rag_id"] == "dr-1"


async def test_folder_resolution(client, monkeypatch):
    tool = uipath_deep_rag_attachments_tool()
    args = {"attachment": DOC, "query": "q"}
    with pytest.raises(UiPathDurableToolError, match="needs a folder"):
        await tool.start(args, _ctx())  # a local run without a folder
    assert client.context_grounding.calls == []
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Shared/Docs")
    await tool.start(args, _ctx())
    assert client.context_grounding.calls[-1]["folder_key"] == "fk-looked-up"
    assert client.folders.lookups == ["Shared/Docs"]  # async, never blocking
    client.folders.key = None
    explicit = uipath_deep_rag_attachments_tool(folder_path="Missing")
    with pytest.raises(UiPathDurableToolError, match="Folder 'Missing'"):
        await explicit.start(args, _ctx())


async def test_start_errors(client, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    from uipath.platform.errors import EnrichedException

    tool = uipath_deep_rag_attachments_tool()
    with pytest.raises(UiPathDurableToolError, match="Query is required"):
        await tool.start({"attachment": DOC, "query": "  "}, _ctx())
    with pytest.raises(UiPathDurableToolError, match="Attachment is required"):
        await tool.start({"attachment": None, "query": "q"}, _ctx())
    with pytest.raises(UiPathDurableToolError, match="Attachment ID is required"):
        await tool.start({"attachment": {"FullName": "x"}, "query": "q"}, _ctx())
    with pytest.raises(ValueError, match="missing required argument"):
        tool.validate_args({"attachment": DOC})
    import httpx

    request = httpx.Request("POST", "https://t/ecs_/v2/deeprag/create")
    response = httpx.Response(400, json={"message": "No valid files"}, request=request)
    client.context_grounding.error = EnrichedException(
        httpx.HTTPStatusError("400", request=request, response=response)
    )
    with pytest.raises(UiPathDurableToolError) as info:
        await tool.start({"attachment": DOC, "query": "q"}, _ctx())
    assert str(info.value).startswith(
        "Could not start DeepRAG on attachment 'contract.pdf' (HTTP 400)"
    )


async def test_attachment_from_the_agent_input(client, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    tool = uipath_deep_rag_attachments_tool(
        query="Summarize.", attachment_from_input="document"
    )
    await tool.start(tool.validate_args({}), _ctx({"document": {"ID": DOC_ID}}))
    assert client.context_grounding.calls[-1]["attachments"] == [DOC_ID]
    unknown = "cccccccc-0000-0000-0000-00000000ffff"
    with pytest.raises(UiPathDurableToolError) as info:
        await tool.start({}, _ctx({"document": {"ID": unknown}}))
    assert str(info.value) == not_found_message(unknown)


async def test_unresolved_attachment_argument(client, monkeypatch):
    # langchain's StaticArgsHandler: an argument the agent input lacks keeps
    # the model's value; a required one neither filled is a tool error.
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    tool = uipath_deep_rag_attachments_tool(attachment_from_input="document")
    other = "cccccccc-0000-0000-0000-00000000ffff"
    # resolved: the agent input wins over the model's attachment
    await tool.start(
        {"attachment": {"ID": other}, "query": "q"}, _ctx({"document": DOC})
    )
    assert client.context_grounding.calls[-1]["attachments"] == [DOC_ID]
    # unresolved: the model's (host-resolved) attachment
    await tool.start({"attachment": DOC, "query": "q"}, _ctx({}))
    assert client.context_grounding.calls[-1]["attachments"] == [DOC_ID]
    # a partial attachment is still matched to a known one
    await tool.start({"attachment": {"ID": DOC_ID}, "query": "q"}, _ctx({}))
    assert client.context_grounding.calls[-1]["attachments"] == [DOC_ID]
    calls = len(client.context_grounding.calls)
    with pytest.raises(UiPathDurableToolError) as info:
        await tool.start(tool.validate_args({"query": "q"}), _ctx({}))
    assert str(info.value) == (
        "Invalid arguments for tool 'DeepRAG': missing required argument(s): "
        "attachment. The agent input has no value for them; fill them in and "
        "call the tool again."
    )
    assert len(client.context_grounding.calls) == calls
    ctx = _ctx({"document": DOC})
    assert tool.span_args({"query": "q"}, ctx) == {"query": "q", "attachment": DOC}
    assert tool.mock_args({"query": "q"}, ctx) == {"query": "q", "attachment": DOC}
    with pytest.raises(UiPathDurableToolError, match="missing required"):
        tool.mock_args({"query": "q"}, _ctx({}))


async def test_finish_maps_the_task():
    tool = uipath_deep_rag_attachments_tool()
    done = {
        "id": "dr-1",
        "name": "t",
        "createdDate": "2026-10-21T00:00:00Z",
        "lastDeepRagStatus": "Successful",
        "content": {"text": "Ends with 30 days notice [1].", "citations": []},
    }
    assert json.loads(await tool.finish({}, done)) == {
        "text": "Ends with 30 days notice [1].",
        "citations": [],
        "deepRagId": "dr-1",
    }
    failed = {**done, "lastDeepRagStatus": "Failed", "failureReason": "bad file"}
    assert json.loads(await tool.finish({}, failed))["error"] == "bad file"
    with pytest.raises(UiPathDurableToolPending):
        await tool.finish({}, {**done, "lastDeepRagStatus": "InProgress"})
