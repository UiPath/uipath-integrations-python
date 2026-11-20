"""Batch Transform on a job attachment: builders, start and finish."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import uipath_codex_sdk.batch_transform_tools as bt
import uipath_codex_sdk.context_tools as ct
import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    AttachmentStore,
    DurableToolContext,
    UiPathAttachment,
    UiPathDurableToolError,
    UiPathToolError,
    batch_transform_tool_from_resource,
    uipath_batch_transform_attachments_tool,
)
from uipath_codex_sdk.attachments import attachment_paths
from uipath_codex_sdk.runtime.session_store import rebuild_suspend_value
from uipath_codex_sdk.runtime.tool_host import output_paths

CSV_ID = "dededede-0000-0000-0000-000000000001"
CSV = {"ID": CSV_ID, "FullName": "reviews.csv", "MimeType": "text/csv"}
COLUMNS = [("sentiment", "positive, neutral or negative")]

#: Agent Builder's default resource, with a query and a column filled in.
RESOURCE: dict[str, Any] = {
    "$resourceType": "tool",
    "type": "internal",
    "name": "Batch Transform",
    "description": "Process and transform CSV files.",
    "inputSchema": bt.batch_transform_attachments_input_schema(),
    "outputSchema": {"type": "array", "items": {}},
    "properties": {
        "toolType": "batch-transform",
        "settings": {
            "contextType": "attachments",
            "query": {"variant": "dynamic", "description": "What to add"},
            "folderPathPrefix": {},
            "fileExtension": {"value": "csv"},
            "outputColumns": [{"name": "topic", "description": "The product area"}],
            "webSearchGrounding": {"value": "Enabled"},
        },
    },
    "settings": {},
}


class _Grounding:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_batch_transform_from_attachment_async(self, **kwargs: Any) -> Any:
        from uipath.platform.context_grounding import BatchTransformCreationResponse

        self.calls.append(kwargs)
        return BatchTransformCreationResponse.model_validate(
            {"id": "bt-1", "lastBatchRagStatus": "Queued"}
        )


class _Attachments:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.uploads: list[dict[str, Any]] = []

    async def upload_async(self, **kwargs: Any) -> Any:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("blob store unavailable")
        self.uploads.append(kwargs)
        return "11111111-0000-0000-0000-00000000aaaa"


class _Jobs:
    def __init__(self, error: Exception | None = None) -> None:
        self.links: list[dict[str, Any]] = []
        self.error = error

    async def link_attachment_async(self, **kwargs: Any) -> None:
        if self.error is not None:
            raise self.error
        self.links.append(kwargs)


@pytest.fixture
def client(monkeypatch, tmp_path):
    holder = SimpleNamespace(
        context_grounding=_Grounding(), attachments=_Attachments(), jobs=_Jobs()
    )
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: holder)
    monkeypatch.setattr(ct, "_folder_keys", {})
    monkeypatch.setattr(bt.asyncio, "sleep", _no_sleep)
    monkeypatch.chdir(tmp_path)
    for var in ("UIPATH_FOLDER_PATH", "UIPATH_JOB_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-job")
    return holder


async def _no_sleep(_: float) -> None:
    return None


def _ctx() -> DurableToolContext:
    store = AttachmentStore([UiPathAttachment.model_validate(CSV)])
    return DurableToolContext("call_1", "rt", "Batch_Transform", None, store, "m")


def _result(task_id: str = "bt-1", text: str = "a,b\n") -> str:
    path = Path(bt.RESULT_DIR) / f"{task_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f"Batch transform completed. Modified file available at {path.resolve()}"


# --- builders ---------------------------------------------------------------


def test_factory_schema_and_columns():
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    assert tool.name == "Batch_Transform" and tool.tool_type == "ContextGrounding"
    assert tool.input_schema["required"] == ["attachment", "query"]
    assert tool.input_schema["properties"]["query"]["description"] == (
        bt.BATCH_TRANSFORM_QUERY_DESCRIPTION
    )
    assert attachment_paths(tool.input_schema) == [("attachment",)]
    assert output_paths(tool) == [("result",)]  # the result becomes known
    with pytest.raises(ValueError, match="at least one output column"):
        uipath_batch_transform_attachments_tool(output_columns=[])
    with pytest.raises(ValueError, match="output column 'bad/name'"):
        uipath_batch_transform_attachments_tool(output_columns=[("bad/name", "d")])
    with pytest.raises(ValueError, match="output column 'x'"):
        uipath_batch_transform_attachments_tool(output_columns=[("x", "")])


def test_from_resource():
    tool = batch_transform_tool_from_resource(RESOURCE)
    assert tool.name == "Batch_Transform" and tool.query is None
    assert tool.web_search_grounding is True
    assert [c.name for c in tool.output_columns] == ["topic"]
    assert tool.input_schema["properties"]["query"]["description"] == "What to add"
    argument = json.loads(json.dumps(RESOURCE))
    argument["properties"]["settings"]["query"] = {
        "variant": "argument",
        "value": "{{task}}",
    }
    # As langchain: any non-static query is written by the model.
    assert (
        "query" in batch_transform_tool_from_resource(argument).input_schema["required"]
    )
    static_empty = json.loads(json.dumps(RESOURCE))
    static_empty["properties"]["settings"]["query"] = {"variant": "static"}
    with pytest.raises(ValueError, match="make the query dynamic"):
        batch_transform_tool_from_resource(static_empty)
    no_columns = json.loads(json.dumps(RESOURCE))
    no_columns["properties"]["settings"]["outputColumns"] = []
    with pytest.raises(ValueError, match="at least one output column"):
        batch_transform_tool_from_resource(no_columns)


def test_suspend_value_round_trip():
    from uipath.core.serialization import serialize_object
    from uipath.platform.context_grounding import BatchTransformCreationResponse

    value = bt.WaitBatchTransformFile(
        batch_transform=BatchTransformCreationResponse.model_validate(
            {"id": "bt-1", "lastBatchRagStatus": "Queued"}
        ),
        index_folder_key="fk",
        destination_path=".uipath/batch_transform/bt-1.csv",
    )
    again = rebuild_suspend_value(
        "WaitBatchTransformFile", json.loads(json.dumps(serialize_object(value)))
    )
    assert isinstance(again, bt.WaitBatchTransformFile)
    assert again.batch_transform.id == "bt-1" and again.destination_path.endswith(
        "bt-1.csv"
    )


# --- start ------------------------------------------------------------------


async def test_start(client):
    tool = uipath_batch_transform_attachments_tool(
        output_columns=COLUMNS, web_search_grounding=True
    )
    args = tool.validate_args({"attachment": CSV, "query": "Classify."})
    value = await tool.start(args, _ctx())
    assert isinstance(value, bt.WaitBatchTransformFile)
    assert value.index_folder_key == "fk-job"
    assert value.destination_path == os.path.join(bt.RESULT_DIR, "bt-1.csv")
    [call] = client.context_grounding.calls
    assert call["attachment"] == CSV_ID and call["prompt"] == "Classify."
    assert call["enable_web_search_grounding"] is True
    assert (call["folder_key"], call["folder_path"]) == ("fk-job", None)
    assert call["name"].startswith("task-")
    assert tool.started_output(value) == {
        "status": "started",
        "batch_transform_id": "bt-1",
        "batch_transform_status": "Queued",
    }


async def test_from_resource_jsonpath_argument_property(client):
    data = json.loads(json.dumps(RESOURCE))
    data["argumentProperties"] = {
        "$['attachment']": {
            "variant": "argument",
            "argumentPath": "$['file']",
            "isSensitive": False,
        }
    }
    tool = batch_transform_tool_from_resource(data)
    assert set(tool.fixed_args) == {"attachment"}
    assert "attachment" in tool.input_schema["properties"]  # visible, optional
    assert tool.input_schema["required"] == ["query"]
    ctx = dataclasses.replace(_ctx(), agent_input={"file": CSV})
    await tool.start({"query": "Classify."}, ctx)
    [call] = client.context_grounding.calls
    assert call["attachment"] == CSV_ID


async def test_unresolved_attachment_argument(client):
    # langchain's StaticArgsHandler: an argument the agent input lacks keeps
    # the model's value; a required one neither filled is a tool error.
    tool = uipath_batch_transform_attachments_tool(
        output_columns=COLUMNS, attachment_from_input="{{file}}"
    )
    schema = tool.input_schema
    assert schema["required"] == ["query"]
    assert (
        "Set from the agent input 'file'"
        in (schema["properties"]["attachment"]["description"])
    )
    assert attachment_paths(schema) == [("attachment",)]
    other = "dededede-0000-0000-0000-00000000ffff"
    with_input = dataclasses.replace(_ctx(), agent_input={"file": CSV})
    await tool.start({"attachment": {"ID": other}, "query": "q"}, with_input)
    assert client.context_grounding.calls[-1]["attachment"] == CSV_ID
    without = dataclasses.replace(_ctx(), agent_input={})
    await tool.start({"attachment": {"ID": CSV_ID}, "query": "q"}, without)
    assert client.context_grounding.calls[-1]["attachment"] == CSV_ID
    calls = len(client.context_grounding.calls)
    with pytest.raises(UiPathDurableToolError) as info:
        await tool.start(tool.validate_args({"query": "q"}), without)
    assert str(info.value) == (
        "Invalid arguments for tool 'Batch_Transform': missing required "
        "argument(s): attachment. The agent input has no value for them; fill "
        "them in and call the tool again."
    )
    assert len(client.context_grounding.calls) == calls
    merged = {"query": "q", "attachment": CSV}
    assert tool.span_args({"query": "q"}, with_input) == merged
    assert tool.mock_args({"query": "q"}, with_input) == merged
    with pytest.raises(UiPathDurableToolError, match="missing required"):
        tool.mock_args({"query": "q"}, without)


async def test_start_errors(client, monkeypatch):
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    with pytest.raises(UiPathDurableToolError, match="Query is required"):
        await tool.start({"attachment": CSV, "query": " "}, _ctx())
    with pytest.raises(UiPathDurableToolError, match="Attachment is required"):
        await tool.start({"attachment": None, "query": "q"}, _ctx())
    with pytest.raises(ValueError, match="longer than"):
        tool.validate_args({"attachment": CSV, "query": "x" * 250_001})
    monkeypatch.delenv("UIPATH_FOLDER_KEY")
    with pytest.raises(UiPathDurableToolError, match="needs a folder"):
        await tool.start({"attachment": CSV, "query": "q"}, _ctx())
    assert client.context_grounding.calls == []


# --- finish -----------------------------------------------------------------


async def test_finish_uploads_and_links(client, monkeypatch):
    monkeypatch.setenv("UIPATH_JOB_KEY", "abababab-0000-0000-0000-000000000001")
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    client.attachments.failures = 1  # retried
    text = await tool.finish({}, _result())
    assert json.loads(text) == {
        "result": {
            "ID": "11111111-0000-0000-0000-00000000aaaa",
            "FullName": "output.csv",
            "MimeType": "text/csv",
        }
    }
    [upload] = client.attachments.uploads
    assert upload["name"] == "output.csv" and upload["folder_key"] == "fk-job"
    [link] = client.jobs.links
    assert str(link["job_key"]) == "abababab-0000-0000-0000-000000000001"
    assert not (Path(bt.RESULT_DIR) / "bt-1.csv").exists()


async def test_finish_keeps_the_upload_when_the_link_fails(client, monkeypatch):
    monkeypatch.setenv("UIPATH_JOB_KEY", "abababab-0000-0000-0000-000000000001")
    client.jobs.error = RuntimeError("link refused")
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    result = json.loads(await tool.finish({}, _result()))
    assert result["result"]["ID"] == "11111111-0000-0000-0000-00000000aaaa"


async def test_finish_local_run_and_bad_values(client, tmp_path):
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    await tool.finish({}, _result())  # no job key: uploaded, not linked
    assert len(client.attachments.uploads) == 1 and client.jobs.links == []
    with pytest.raises(UiPathToolError, match="Unexpected"):
        await tool.finish({}, {"path": "/etc/passwd"})
    outside = tmp_path / "secret.csv"
    outside.write_text("x")
    with pytest.raises(UiPathToolError, match="not available"):
        await tool.finish(
            {}, f"Batch transform completed. Modified file available at {outside}"
        )
    client.attachments.failures = 3
    with pytest.raises(RuntimeError, match="blob store"):
        await tool.finish({}, _result())


# --- reading the trigger and downloading the result -------------------------


class _ResultService:
    """retrieve / read-URI / authenticated download of one task."""

    def __init__(self, status: str = "Successful", encrypted: bool = False) -> None:
        self.status, self.encrypted = status, encrypted
        self.requests: list[str] = []

    async def retrieve_batch_transform_async(self, id: str) -> Any:
        return SimpleNamespace(id=id, last_batch_rag_status=self.status)

    def _batch_transform_get_read_uri_spec(self, id: str) -> Any:
        return SimpleNamespace(method="GET", endpoint="read-uri", headers={})

    def _batch_transform_download_blob_spec(self, id: str) -> Any:
        return SimpleNamespace(method="GET", endpoint="download-blob", headers={})

    async def request_async(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        import httpx

        self.requests.append(endpoint)
        request = httpx.Request(method, f"https://t/{endpoint}")
        if endpoint == "read-uri":
            body = {"uri": "https://blob.test/r.csv", "isEncrypted": self.encrypted}
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(200, content=b"authenticated", request=request)


def _trigger(reader: str | None = "codex") -> Any:
    from uipath.core.triggers import UiPathResumeTrigger, UiPathResumeTriggerType

    trigger = UiPathResumeTrigger.model_validate(
        {"triggerType": UiPathResumeTriggerType.BATCH_RAG, "itemKey": "bt-1"}
    )
    trigger.payload = {
        "destination_path": "x.csv",
        **({"reader": reader} if reader else {}),
    }
    return trigger


async def test_codex_trigger_reader(monkeypatch):
    from uipath.core.errors import UiPathFaultedTriggerError, UiPathPendingTriggerError

    from uipath_codex_sdk.runtime.triggers import CodexResumeTriggerHandler

    service = _ResultService()
    monkeypatch.setattr(
        tools_mod, "_uipath_client", lambda: SimpleNamespace(context_grounding=service)
    )
    handler = CodexResumeTriggerHandler()
    assert await handler.read_trigger(_trigger()) == {
        "batch_transform_id": "bt-1",
        "status": "Successful",
    }
    assert service.requests == []  # nothing downloaded by the reader
    service.status = "InProgress"
    with pytest.raises(UiPathPendingTriggerError):
        await handler.read_trigger(_trigger())
    service.status = "Failed"
    with pytest.raises(UiPathFaultedTriggerError, match="failed"):
        await handler.read_trigger(_trigger())
    assert not bt.is_codex_batch_transform_trigger(_trigger(reader=None))


@pytest.mark.parametrize("authenticated_ok", [True, False])
async def test_download_uses_the_authenticated_endpoint_then_the_storage_url(
    monkeypatch, tmp_path, authenticated_ok
):
    import httpx
    import uipath._utils._ssl_context as ssl_context

    def blob(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"from storage")

    monkeypatch.setattr(
        ssl_context,
        "get_httpx_client_kwargs",
        lambda: {"transport": httpx.MockTransport(blob)},
    )
    service = _ResultService()
    if not authenticated_ok:
        original = service.request_async

        async def failing(method: str, endpoint: str, **kwargs: Any) -> Any:
            if endpoint == "download-blob":
                service.requests.append(endpoint)
                raise RuntimeError("download endpoint unavailable")
            return await original(method, endpoint, **kwargs)

        service.request_async = failing  # type: ignore[method-assign]
    path = tmp_path / "r.csv"
    await bt._download_result(SimpleNamespace(context_grounding=service), "bt-1", path)
    if authenticated_ok:
        assert path.read_bytes() == b"authenticated"
        assert service.requests == ["download-blob"]  # no storage URL needed
    else:
        assert path.read_bytes() == b"from storage"
        assert service.requests == ["download-blob", "read-uri"]


async def test_download_reports_the_authenticated_error_when_both_fail(
    monkeypatch, tmp_path
):
    import httpx
    import uipath._utils._ssl_context as ssl_context

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Name or service not known", request=request)

    monkeypatch.setattr(
        ssl_context,
        "get_httpx_client_kwargs",
        lambda: {"transport": httpx.MockTransport(unreachable)},
    )
    service = _ResultService()
    original = service.request_async

    async def failing(method: str, endpoint: str, **kwargs: Any) -> Any:
        if endpoint == "download-blob":
            raise RuntimeError("download endpoint unavailable")
        return await original(method, endpoint, **kwargs)

    service.request_async = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="download endpoint unavailable"):
        await bt._download_result(
            SimpleNamespace(context_grounding=service), "bt-1", tmp_path / "r.csv"
        )


async def test_finish_downloads_a_codex_resume_value(client):
    client.context_grounding = _ResultService(encrypted=True)
    tool = uipath_batch_transform_attachments_tool(output_columns=COLUMNS)
    text = await tool.finish({}, {"batch_transform_id": "bt-1", "status": "Successful"})
    assert json.loads(text)["result"]["FullName"] == "output.csv"
    assert client.context_grounding.requests == ["download-blob"]
    with pytest.raises(UiPathToolError, match="Unexpected Batch Transform id"):
        await tool.finish({}, {"batch_transform_id": "../../etc"})
