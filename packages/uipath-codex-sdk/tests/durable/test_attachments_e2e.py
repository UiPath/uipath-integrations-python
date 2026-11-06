"""Offline end-to-end tests of job attachments.

The real factory → runtime → bundled codex → gateway shim (mock model) →
``uipath_tools`` host, with Analyze Files' blob storage and LLM Gateway call
faked, and fake Orchestrator jobs for the durable process tool.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

import uipath_codex_sdk.attachment_tools as at
from uipath_codex_sdk.attachments import not_found_message
from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.runtime import ATTACHMENT_INSTRUCTIONS

from ..contract.mock_responses import (
    advertised_tools,
    find_tool,
    function_call_item,
    input_items,
    last_user_text,
    message_item,
)

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
DOC_ID = "dddddddd-0000-0000-0000-000000000001"
REPORT_ID = "dddddddd-0000-0000-0000-000000000002"
UNKNOWN_ID = "dddddddd-0000-0000-0000-00000000ffff"
DOC = {"ID": DOC_ID, "FullName": "invoice.pdf", "MimeType": "application/pdf"}
REPORT = {"ID": REPORT_ID, "FullName": "summary.md", "MimeType": "text/markdown"}
CALL_ID = "call_attachments0000000000001"

ANALYZE_MAIN = textwrap.dedent(
    """
    from pydantic import BaseModel

    from uipath_codex_sdk import (
        UiPathAttachment,
        UiPathCodexAgent,
        uipath_analyze_files_tool,
    )


    class Input(BaseModel):
        document: UiPathAttachment
        question: str


    class Output(BaseModel):
        answer: str
        source: UiPathAttachment | None = None


    agent = UiPathCodexAgent(
        model="mock-model",
        input_schema=Input,
        output_schema=Output,
        prompt="Answer '{question}' about the document.",
        tools=[uipath_analyze_files_tool()],
    )
    """
)

ANALYZE_PREFILLED_MAIN = textwrap.dedent(
    """
    from pydantic import BaseModel

    from uipath_codex_sdk import (
        UiPathAttachment,
        UiPathCodexAgent,
        analyze_files_tool_from_resource,
    )
    from uipath_codex_sdk.attachment_tools import analyze_files_input_schema


    class Input(BaseModel):
        documents: list[UiPathAttachment]
        question: str


    class Output(BaseModel):
        answer: str


    analyze = analyze_files_tool_from_resource(
        {
            "$resourceType": "tool",
            "type": "Internal",
            "name": "Analyze Files",
            "description": "Analyze the attached files.",
            "inputSchema": analyze_files_input_schema(),
            "outputSchema": {"type": "object", "properties": {}},
            "properties": {"toolType": "analyze-attachments"},
            "settings": {},
            "argumentProperties": {
                "attachments": {
                    "variant": "argument",
                    "argumentPath": "$.documents",
                    "isSensitive": False,
                }
            },
        }
    )

    agent = UiPathCodexAgent(
        model="mock-model",
        input_schema=Input,
        output_schema=Output,
        prompt="Answer '{question}' about the documents.",
        tools=[analyze],
    )
    """
)

PROCESS_MAIN = textwrap.dedent(
    """
    from pydantic import BaseModel

    from uipath_codex_sdk import UiPathAttachment, UiPathCodexAgent, uipath_process_tool


    class Input(BaseModel):
        document: UiPathAttachment


    class SummarizeOut(BaseModel):
        report: UiPathAttachment


    class Output(BaseModel):
        original: UiPathAttachment
        report: UiPathAttachment


    agent = UiPathCodexAgent(
        model="mock-model",
        input_schema=Input,
        output_schema=Output,
        prompt="PLEASE_CALL_DURABLE: summarize {document} with Summarize.",
        tools=[
            uipath_process_tool(
                "Summarize",
                folder_path="Shared",
                input_schema=Input,
                output_schema=SummarizeOut,
            )
        ],
        durable_settle_timeout=20.0,
    )
    """
)


class Script:
    """Call ``tool`` once with ``args``; then answer ``final(outputs)``."""

    def __init__(
        self, tool: str, args: dict[str, Any], final: Callable[[dict[str, Any]], str]
    ) -> None:
        self.tool, self.args, self.final = tool, args, final
        self.prompts: list[str] = []
        self.bodies: list[dict[str, Any]] = []
        self.tool_specs: list[dict[str, Any]] = []

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        self.bodies.append(body)
        self.prompts.append(last_user_text(body))
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if CALL_ID not in seen:
            tools = advertised_tools(body)
            name, ns = find_tool(tools, self.tool)
            self.tool_specs = tools
            if name:
                return [function_call_item(CALL_ID, name, self.args, ns)]
        outputs = {
            it["call_id"]: _output_text(it.get("output"))
            for it in items
            if it.get("type") == "function_call_output"
        }
        return [message_item(self.final(outputs), idx)]


def _output_text(output: Any) -> Any:
    """An MCP tool's result text (Codex sends a list after a ``Wall time`` part)."""
    if isinstance(output, list):
        texts = [p.get("text", "") for p in output if isinstance(p, dict)]
        return texts[-1] if texts else ""
    return output


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


@pytest.fixture
def files(monkeypatch):
    """Analyze Files' blob storage and LLM Gateway."""
    state = SimpleNamespace(requests=[], size=1)

    async def blob_uri(*, key):
        return SimpleNamespace(uri=f"https://blob.test/{key}", name="x")

    def blob(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7 total 42" * state.size)

    class Gateway:
        async def request(self, method, path, *, headers, content):
            state.requests.append(json.loads(content))
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Total: 42"}],
                        }
                    ]
                },
            )

        async def aclose(self) -> None:
            pass

    async def gateway_client(model: str) -> Gateway:
        return Gateway()

    monkeypatch.setattr(
        at,
        "_uipath_client",
        lambda: SimpleNamespace(
            attachments=SimpleNamespace(get_blob_file_access_uri_async=blob_uri)
        ),
    )
    monkeypatch.setattr(
        at,
        "_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(blob)),
    )
    monkeypatch.setattr(at, "_gateway_client", gateway_client)
    return state


async def _run(project: Path, input, *, resume: bool = False):
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
        )
    )
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        return await asyncio.wait_for(
            runtime.execute(input, UiPathExecuteOptions(resume=resume)), 120
        )
    finally:
        await runtime.dispose()
        await factory.dispose()


def _answer(outputs: dict[str, Any]) -> str:
    return json.dumps({"answer": outputs.get(CALL_ID) or "", "source": {"ID": DOC_ID}})


async def test_analyze_files_with_an_input_attachment(project, upstream, files, spans):
    (project / "main.py").write_text(ANALYZE_MAIN)
    script = Script(
        "Analyze_Files",
        {"attachments": [{"ID": DOC_ID}], "analysisTask": "What is the total?"},
        _answer,
    )
    upstream.script = script
    result = await _run(project, {"document": DOC, "question": "the total"})
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert result.output == {
        "answer": json.dumps({"analysisResult": "Total: 42"}),
        "source": {**DOC, "Metadata": None},  # filled in from the known ID
    }
    # The template does not show the document: it follows as a block.
    prompt = script.prompts[0]
    assert prompt.startswith("Answer 'the total' about the document.\n\n")
    assert "Input attachments (JobAttachment objects):" in prompt and DOC_ID in prompt
    assert ATTACHMENT_INSTRUCTIONS in json.dumps(script.bodies[0])
    # Codex keeps the job-attachment reference in the tool's parameters.
    assert "job-attachment" in json.dumps(script.tool_specs)
    [request] = files.requests
    parts = request["input"][0]["content"]
    assert parts[0] == {"type": "input_text", "text": "What is the total?"}
    assert parts[1]["type"] == "input_file" and parts[1]["filename"] == "invoice-pdf"

    finished = {s.name: s for s in spans.get_finished_spans()}
    run = finished["Agent run - agent"]
    assert json.loads(run.attributes["attachments"])[0]["id"] == DOC_ID
    output = finished["Agent output"]
    assert json.loads(output.attributes["attachments"])[0]["direction"] == 2
    call = finished["Tool call - Analyze_Files"]
    assert call.attributes["toolType"] == "Internal"
    assert json.loads(call.attributes["attachments"])[0]["id"] == DOC_ID
    internal = finished["Analyze Files"]
    assert internal.parent.span_id == call.context.span_id
    llm = [
        s
        for s in spans.get_finished_spans()
        if s.name == "LLM call" and s.parent.span_id == internal.context.span_id
    ]
    assert len(llm) == 1 and "attachments" in llm[0].attributes


async def test_unknown_attachment_id_is_a_tool_error(project, upstream, files):
    (project / "main.py").write_text(ANALYZE_MAIN)
    script = Script(
        "Analyze_Files",
        {"attachments": [{"ID": UNKNOWN_ID}], "analysisTask": "x"},
        lambda outputs: json.dumps({"answer": outputs.get(CALL_ID) or ""}),
    )
    upstream.script = script
    result = await _run(project, {"document": DOC, "question": "q"})
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not_found_message(UNKNOWN_ID) in result.output["answer"]
    assert files.requests == []


async def test_an_input_attachment_wins_over_an_unknown_model_id(
    project, upstream, files, spans
):
    # The agent input fills ``attachments`` (argumentProperties): as in
    # LangChain, the model's unknown ID it replaces is not checked.
    (project / "main.py").write_text(ANALYZE_PREFILLED_MAIN)
    script = Script(
        "Analyze_Files",
        {"attachments": [{"ID": UNKNOWN_ID}], "analysisTask": "What is the total?"},
        lambda outputs: json.dumps({"answer": outputs.get(CALL_ID) or ""}),
    )
    upstream.script = script
    result = await _run(project, {"documents": [DOC], "question": "the total"})
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert result.output == {"answer": json.dumps({"analysisResult": "Total: 42"})}
    [request] = files.requests
    assert request["input"][0]["content"][1]["filename"] == "invoice-pdf"
    call = {s.name: s for s in spans.get_finished_spans()}["Tool call - Analyze_Files"]
    assert json.loads(call.attributes["input"])["attachments"] == [DOC]
    assert [a["id"] for a in json.loads(call.attributes["attachments"])] == [DOC_ID]


async def test_a_file_over_the_limit_fails_the_run(project, upstream, files):
    (project / "main.py").write_text(ANALYZE_MAIN)
    files.size = (at.MAX_FILE_BYTES // len(b"%PDF-1.7 total 42")) + 2
    upstream.script = Script(
        "Analyze_Files",
        {"attachments": [{"ID": DOC_ID}], "analysisTask": "x"},
        _answer,
    )
    with pytest.raises(Exception) as info:
        await _run(project, {"document": DOC, "question": "q"})
    error = info.value
    assert "FILE_ERROR" in str(getattr(error, "error_info", None) or error)
    assert files.requests == []


async def test_process_tool_attachments_survive_a_suspension(project, upstream, world):
    (project / "main.py").write_text(PROCESS_MAIN)
    script = Script(
        "Summarize",
        {"document": {"ID": DOC_ID}},
        lambda outputs: json.dumps(
            {"original": {"ID": DOC_ID}, "report": {"ID": REPORT_ID}}
        ),
    )
    upstream.script = script
    result = await _run(project, {"document": DOC})
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    [invocation] = world.invocations
    assert [str(a.id) for a in invocation.attachments] == [DOC_ID]
    assert invocation.input_arguments["document"]["FullName"] == "invoice.pdf"

    world.finish(invocation.job_key, output={"report": REPORT})
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    # The input's attachment (from the saved run input) and the job's
    # (harvested by the tool's output schema) are both known.
    assert result.output == {
        "original": {**DOC, "Metadata": None},
        "report": {**REPORT, "Metadata": None},
    }
