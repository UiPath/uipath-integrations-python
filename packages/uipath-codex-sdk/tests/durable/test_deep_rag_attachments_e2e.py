"""Offline end-to-end tests of DeepRAG on a job attachment (suspend → resume).

The real factory, runtime, bundled codex, gateway shim, tool host and
``UiPathResumeTriggerHandler``, with a mock model and a fake
``context_grounding`` service whose DeepRAG tasks the test finishes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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

from uipath_codex_sdk.attachments import not_found_message
from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.runtime import ATTACHMENT_INSTRUCTIONS

from .._typing import sqlite_storage
from .fakes import DeepRagAttachmentScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
JOB_FOLDER_KEY = "12121212-3434-5656-7878-909090909090"
DOC_ID = "eeeeeeee-0000-0000-0000-000000000001"
DOC = {"ID": DOC_ID, "FullName": "contract.pdf", "MimeType": "application/pdf"}

MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_deep_rag_attachments_tool,
)


class Input(BaseModel):
    document: UiPathAttachment


agent = UiPathCodexAgent(
    model="mock-model",
    input_schema=Input,
    prompt="PLEASE_CALL_DURABLE: research {document} with DeepRAG.",
    tools=[uipath_deep_rag_attachments_tool("DeepRAG")],
    durable_settle_timeout=20.0,
)
"""


@pytest.fixture
def project(project: Path, monkeypatch) -> Path:
    (project / "main.py").write_text(MAIN)
    monkeypatch.setenv("UIPATH_FOLDER_KEY", JOB_FOLDER_KEY)
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


def _final_outputs(result) -> dict:
    text = result.output["output"]
    assert text.startswith("FINAL "), text
    return json.loads(text[len("FINAL ") :])


async def test_deep_rag_on_an_attachment_suspends_then_resumes(
    project, upstream, world, spans
):
    script = DeepRagAttachmentScript(DOC_ID)
    upstream.script = script
    cid = script.call_ids[0]
    cg = world.context_grounding

    # --- run: one start on the attachment, then suspend -----------------
    result, triggers = await _run(project, {"document": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert script.advertised == ("DeepRAG", "mcp__uipath_tools")
    assert ATTACHMENT_INSTRUCTIONS in json.dumps(script.bodies[0])
    [start] = cg.starts
    assert start.attachments == [DOC_ID] and start.index_name is None
    assert start.prompt == DeepRagAttachmentScript.QUERY
    assert (start.folder_key, start.folder_path) == (JOB_FOLDER_KEY, None)
    summary = result.output[cid]
    assert summary["type"] == "WaitDeepRagRaw"
    assert summary["deep_rag_id"] == start.id
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.DEEP_RAG
    assert trig.trigger_name == UiPathResumeTriggerName.DEEP_RAG_RAW
    assert (trig.item_key, trig.folder_key) == (start.id, JOB_FOLDER_KEY)
    names = [s.name for s in spans.get_finished_spans()]
    assert "Tool call - DeepRAG" not in names and "DeepRAG" not in names

    # --- still running: suspends again, nothing restarted ---------------
    n_requests = len(upstream.snapshot())
    cg.tasks[start.id]["lastDeepRagStatus"] = "InProgress"
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert len(upstream.snapshot()) == n_requests

    # --- finished: the model gets the answer ----------------------------
    cg.finish(start.id, text="Either party may terminate with 30 days notice [1].")
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers and len(cg.starts) == 1
    answer = json.loads(_final_outputs(result)[cid])
    assert answer["text"] == "Either party may terminate with 30 days notice [1]."
    assert answer["deepRagId"] == start.id

    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - DeepRAG"]
    [child] = [s for s in finished if s.name == "DeepRAG"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert call.attributes["toolType"] == "Internal"
    assert json.loads(call.attributes["attachments"])[0]["id"] == DOC_ID
    attrs = child.attributes
    assert attrs["span_type"] == "toolCall" and attrs["toolType"] == "Internal"
    assert attrs["tool.name"] == "DeepRAG"
    # One call to the tool-call evaluators, across the suspension too.
    from uipath.eval._helpers.evaluators_helpers import extract_tool_calls_names

    assert extract_tool_calls_names(finished) == ["DeepRAG"]
    assert attrs["deep_rag_id"] == start.id  # kept across the suspension
    assert json.loads(attrs["input"])["attachment"]["FullName"] == "contract.pdf"
    assert json.loads(attrs["output"])["deepRagId"] == start.id
    assert child.status.status_code.name == "OK"


async def test_unknown_attachment_starts_nothing(project, upstream, world):
    unknown = "eeeeeeee-0000-0000-0000-00000000ffff"
    script = DeepRagAttachmentScript(unknown)
    upstream.script = script
    result, triggers = await _run(project, {"document": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers and world.context_grounding.starts == []
    assert not_found_message(unknown) in json.dumps(_final_outputs(result))


async def test_start_error_goes_back_to_the_model(project, upstream, world):
    import httpx
    from uipath.platform.errors import EnrichedException

    request = httpx.Request("POST", "https://t/ecs_/v2/deeprag/create")
    response = httpx.Response(
        400, json={"message": "No valid files to use"}, request=request
    )
    world.context_grounding.fail_start = EnrichedException(
        httpx.HTTPStatusError("400", request=request, response=response)
    )
    upstream.script = DeepRagAttachmentScript(DOC_ID)
    result, triggers = await _run(project, {"document": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    assert "Could not start DeepRAG on attachment 'contract.pdf' (HTTP 400)" in (
        json.dumps(_final_outputs(result))
    )


FROM_INPUT_MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_deep_rag_attachments_tool,
)


class Input(BaseModel):
    other: UiPathAttachment
    document: UiPathAttachment | None = None


agent = UiPathCodexAgent(
    model="mock-model",
    input_schema=Input,
    prompt="PLEASE_CALL_DURABLE: research {other} with DeepRAG.",
    tools=[uipath_deep_rag_attachments_tool("DeepRAG", attachment_from_input="document")],
    durable_settle_timeout=20.0,
)
"""


async def test_attachment_from_input_falls_back_to_the_model(project, upstream, world):
    # The agent input has no 'document': the model is shown the field (as
    # optional, with a note) and the attachment it passes is used.
    (project / "main.py").write_text(FROM_INPUT_MAIN)
    script = DeepRagAttachmentScript(DOC_ID)
    upstream.script = script
    result, triggers = await _run(project, {"other": DOC}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert "Set from the agent input 'document'" in json.dumps(script.bodies[0])
    [start] = world.context_grounding.starts
    assert start.attachments == [DOC_ID]
    [trig] = triggers
    assert trig.item_key == start.id
