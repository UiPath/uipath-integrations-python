"""Offline end-to-end tests of Batch Transform on a job attachment.

The real factory, runtime, bundled codex, gateway shim, tool host and
``UiPathResumeTriggerHandler`` (its BatchRag reader downloads the result),
with a mock model and fake context grounding, attachments and jobs services.
"""

from __future__ import annotations

import json
import shutil
import uuid
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
JOB_KEY = "abababab-0000-0000-0000-000000000001"
FOLDER_KEY = "cdcdcdcd-0000-0000-0000-000000000001"
CSV_ID = "efefefef-0000-0000-0000-000000000001"
CSV = {"ID": CSV_ID, "FullName": "reviews.csv", "MimeType": "text/csv"}
CALL_ID = "call_batchtransform00000000001"
QUERY = "Classify the sentiment of each review."

MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_batch_transform_attachments_tool,
)


class Input(BaseModel):
    reviews: UiPathAttachment


class Output(BaseModel):
    enriched: UiPathAttachment


agent = UiPathCodexAgent(
    model="mock-model",
    input_schema=Input,
    output_schema=Output,
    prompt="PLEASE_CALL_DURABLE: enrich {reviews}.",
    tools=[
        uipath_batch_transform_attachments_tool(
            "Enrich Reviews",
            output_columns=[("sentiment", "positive, neutral or negative")],
        )
    ],
    durable_settle_timeout=20.0,
)
"""


class Script:
    """Call the tool once; then return the result attachment's ID as output."""

    def __init__(self, attachment_id: str = CSV_ID) -> None:
        self.attachment_id = attachment_id
        self.outputs: dict[str, Any] = {}

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if CALL_ID not in seen:
            name, ns = find_tool(advertised_tools(body), "Enrich_Reviews")
            if name:
                args = {"query": QUERY, "attachment": {"ID": self.attachment_id}}
                return [function_call_item(CALL_ID, name, args, ns)]
        for it in items:
            if it.get("type") == "function_call_output":
                self.outputs[it["call_id"]] = it.get("output")
        text: Any = self.outputs.get(CALL_ID)  # None: no output yet
        try:
            result = json.loads(text)["result"]
            answer = {"enriched": {"ID": result["ID"]}}
        except (TypeError, ValueError, KeyError):
            answer = {"enriched": {"ID": "00000000-0000-0000-0000-000000000000"}}
        return [message_item(json.dumps(answer), idx)]


@pytest.fixture
def project(project: Path, monkeypatch) -> Path:
    (project / "main.py").write_text(MAIN)
    monkeypatch.setenv("UIPATH_FOLDER_KEY", FOLDER_KEY)
    monkeypatch.setenv("UIPATH_JOB_KEY", JOB_KEY)
    monkeypatch.delenv("UIPATH_FOLDER_PATH", raising=False)
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


def _external_triggers(world) -> list[dict[str, Any]]:
    return [r for r in world.api_client.requests if "SaveExternalTrigger" in r["url"]]


async def test_batch_transform_suspends_uploads_and_returns_the_result(
    project, upstream, world, spans
):
    upstream.script = Script()
    cg = world.context_grounding

    # --- run: one start on the attachment, then suspend -----------------
    result, triggers = await _run(project, {"reviews": CSV}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    [start] = cg.batch_starts
    assert start["attachment"] == CSV_ID and start["prompt"] == QUERY
    assert [c.name for c in start["output_columns"]] == ["sentiment"]
    assert (start["folder_key"], start["folder_path"]) == (FOLDER_KEY, None)
    assert start["enable_web_search_grounding"] is False
    summary = result.output[CALL_ID]
    assert summary["type"] == "WaitBatchTransformFile"
    assert summary["batch_transform_id"] == start["id"]
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.BATCH_RAG
    assert trig.trigger_name == UiPathResumeTriggerName.BATCH_RAG
    assert (trig.item_key, trig.folder_key) == (start["id"], FOLDER_KEY)
    assert trig.payload["destination_path"].endswith(f"{start['id']}.csv")
    assert trig.payload["reader"] == "codex"  # read by codex, not the platform
    assert len(_external_triggers(world)) == 1

    # --- still running: suspends again ---------------------------------
    cg.batch_tasks[start["id"]]["status"] = "InProgress"
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED

    # --- finished: downloaded, uploaded, linked, returned ----------------
    cg.finish_batch(start["id"], csv="review,sentiment\ngreat,positive\n")
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers and len(cg.batch_starts) == 1
    # The authenticated download: the storage URL is never needed.
    assert cg.result_requests == ["download-blob"]
    assert cg.downloads == []  # the platform's downloader was never used
    [upload] = world.attachments.uploads
    assert upload["name"] == "output.csv"
    assert upload["content"] == "review,sentiment\ngreat,positive\n"
    assert upload["folder_key"] == FOLDER_KEY
    [link] = world.attachments.links
    assert link["attachment_key"] == upload["key"]
    assert link["job_key"] == uuid.UUID(JOB_KEY)
    assert not Path(upload["source_path"]).exists()  # cleaned up
    # The result is known to the run: the model returned it by ID.
    assert result.output == {
        "enriched": {
            "ID": str(upload["key"]),
            "FullName": "output.csv",
            "MimeType": "text/csv",
            "Metadata": None,
        }
    }

    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - Enrich_Reviews"]
    [child] = [s for s in finished if s.name == "Enrich_Reviews"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert call.attributes["toolType"] == "ContextGrounding"
    directions = [a["direction"] for a in json.loads(call.attributes["attachments"])]
    assert directions == [1, 2]  # the source in, the result out
    attrs = child.attributes
    assert attrs["span_type"] == "contextGroundingTool"
    assert attrs["retrieval_mode"] == "BatchTransform"
    assert attrs["query"] == QUERY
    assert attrs["batch_transform_id"] == start["id"]
    assert json.loads(attrs["output_columns"])[0]["name"] == "sentiment"
    assert json.loads(attrs["results"]) == {
        "id": str(upload["key"]),
        "fileName": "output.csv",
        "mimeType": "text/csv",
    }
    child_attachments = json.loads(attrs["attachments"])
    assert [(a["id"], a["direction"]) for a in child_attachments] == [
        (CSV_ID, 1),
        (str(upload["key"]), 2),
    ]


async def test_failed_batch_transform_faults_the_run(project, upstream, world):
    upstream.script = Script()
    cg = world.context_grounding
    result, _ = await _run(project, {"reviews": CSV}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    [start] = cg.batch_starts
    cg.finish_batch(start["id"], status="Failed")
    _new_robot(project)
    with pytest.raises(Exception) as info:
        await _run(project, None, resume=True)
    assert start["id"] in str(getattr(info.value, "error_info", None) or info.value)
    assert world.attachments.uploads == []


async def test_unknown_attachment_starts_nothing(project, upstream, world):
    unknown = "efefefef-0000-0000-0000-00000000ffff"
    upstream.script = Script(attachment_id=unknown)
    # The output then names an unknown ID too: the run fails at the end.
    with pytest.raises(Exception, match="OUTPUT_INVALID|does not know"):
        await _run(project, {"reviews": CSV}, resume=False)
    assert world.context_grounding.batch_starts == []
