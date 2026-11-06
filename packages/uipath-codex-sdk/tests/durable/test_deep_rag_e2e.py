"""Offline end-to-end tests of the DeepRAG durable tool (suspend → resume).

Same set-up as ``test_e2e``: the real factory, runtime, bundled codex, gateway
shim, tool host and ``UiPathResumeTriggerHandler``, with a mock model and a
fake ``context_grounding`` service whose DeepRAG tasks the test finishes.
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

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory

from .._typing import sqlite_storage
from ..contract.mock_responses import input_items, summarize_input
from .fakes import INDEX_FOLDER_KEY, DeepRagScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"

DEEP_RAG_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent, uipath_deep_rag_tool

agent = UiPathCodexAgent(
    model="mock-model",
    prompt="PLEASE_CALL_DURABLE: research the invoices with Invoice_Research.",
    tools=[
        uipath_deep_rag_tool(
            "Invoice Research",
            index_name="invoices",
            folder_path="Coded",
            description="Researches the company's invoices.",
        )
    ],
    durable_settle_timeout=20.0,
)
"""


@pytest.fixture
def project(project: Path) -> Path:
    (project / "main.py").write_text(DEEP_RAG_MAIN)
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


async def test_deep_rag_suspends_then_resumes_with_the_answer(project, upstream, world):
    script = DeepRagScript()
    upstream.script = script
    cid = script.call_ids[0]
    cg = world.context_grounding

    # --- run: starts the task and suspends on it --------------------------
    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert script.advertised == ("Invoice_Research", "mcp__uipath_tools")
    [start] = cg.starts
    assert start.index_name == "invoices"
    assert start.prompt == DeepRagScript.QUERY
    assert start.glob_pattern == "**/*"
    assert start.citation_mode.value == "Inline"
    assert (start.folder_key, start.folder_path) == (INDEX_FOLDER_KEY, None)
    assert start.name.startswith("task-")
    summary = result.output[cid]
    assert summary["deep_rag_id"] == start.id
    assert summary["deep_rag_status"] == "Queued"
    assert summary["folder_key"] == INDEX_FOLDER_KEY
    assert summary["type"] == "WaitDeepRagRaw"
    [trig] = triggers
    assert trig.interrupt_id == cid
    assert trig.trigger_type == UiPathResumeTriggerType.DEEP_RAG
    assert trig.trigger_name == UiPathResumeTriggerName.DEEP_RAG_RAW
    assert trig.item_key == start.id
    assert trig.folder_key == INDEX_FOLDER_KEY
    n_requests = len(upstream.snapshot())

    # --- still in progress: --resume suspends again -----------------------
    cg.tasks[start.id]["lastDeepRagStatus"] = "InProgress"
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in triggers] == [cid]
    assert len(upstream.snapshot()) == n_requests

    # --- finished: the model gets langchain's DeepRAG result --------------
    cg.finish(start.id)
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    answer = json.loads(_final_outputs(result)[cid])
    assert answer == {
        "text": "Acme Corp owes $4,200 [1].",
        "citations": [
            {
                "ordinal": 1,
                "page_number": 1,
                "source": "invoice.jpg",
                "reference": "https://ref/1",
            }
        ],
        "deepRagId": start.id,
    }
    assert len(cg.starts) == 1  # never started again


async def test_failed_deep_rag_is_a_result_not_a_fault(project, upstream, world):
    script = DeepRagScript()
    upstream.script = script
    cid = script.call_ids[0]
    cg = world.context_grounding

    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    [start] = cg.starts
    cg.finish(start.id, "Failed", failure_reason="No documents matched.")
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert json.loads(_final_outputs(result)[cid]) == {
        "status": "failed",
        "deepRagId": start.id,
        "error": "No documents matched.",
    }


async def test_start_error_goes_back_to_the_model(project, upstream, world):
    from uipath.platform.errors import ContextGroundingIndexNotFoundError

    script = DeepRagScript()
    upstream.script = script
    world.context_grounding.fail_start = ContextGroundingIndexNotFoundError("invoices")

    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    blob = json.dumps([summarize_input(r.body) for r in upstream.snapshot()])
    assert "Context grounding index 'invoices' was not found." in json.dumps(
        [input_items(r.body) for r in upstream.snapshot()]
    ), blob


async def test_deep_rag_spans_stay_open_across_the_suspension(
    project, upstream, world, spans
):
    script = DeepRagScript()
    upstream.script = script
    cg = world.context_grounding

    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    names = [s.name for s in spans.get_finished_spans()]
    assert "Tool call - Invoice_Research" not in names
    assert "Invoice Research" not in names  # held open, like the Tool call

    [start] = cg.starts
    cg.finish(start.id)
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - Invoice_Research"]
    [child] = [s for s in finished if s.name == "Invoice Research"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert call.attributes["toolType"] == "ContextGrounding"
    assert call.attributes["context_retrieval_mode"] == "DeepRAG"
    assert call.attributes["index_name"] == "invoices"
    attrs = child.attributes
    assert attrs["span_type"] == "contextGroundingTool"
    assert attrs["retrieval_mode"] == "DeepRAG"
    assert attrs["query"] == DeepRagScript.QUERY
    assert attrs["citation_mode"] == "Inline"
    assert attrs["deep_rag_id"] == start.id
    assert json.loads(attrs["output"])["deepRagId"] == start.id
    assert child.status.status_code.name == "OK"
    # Both were started in the first execution and ended in the second.
    assert child.start_time < call.end_time and child.end_time <= call.end_time
