"""Analyze Files: downloads, content parts, the gateway call and its spans."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import uipath_codex_sdk.attachment_tools as at
from uipath_codex_sdk import (
    AttachmentStore,
    ToolContext,
    UiPathAttachment,
    UiPathToolError,
    analyze_files_tool_from_resource,
    uipath_analyze_files_tool,
)
from uipath_codex_sdk.attachments import attachment_paths
from uipath_codex_sdk.runtime import telemetry
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError

PNG_ID = "aaaaaaaa-0000-0000-0000-000000000001"
PDF_ID = "aaaaaaaa-0000-0000-0000-000000000002"
PNG = {"ID": PNG_ID, "FullName": "chart.png", "MimeType": "image/png"}
PDF = {"ID": PDF_ID, "FullName": "q3 report.pdf", "MimeType": "application/pdf"}
BLOBS = {PNG_ID: b"\x89PNG-bytes", PDF_ID: b"%PDF-1.7 bytes"}

RESOURCE = {
    "$resourceType": "tool",
    "type": "Internal",
    "name": "Analyze Files",
    "description": "Analyze the attached files.",
    "inputSchema": at.analyze_files_input_schema(),
    "outputSchema": {"type": "object", "properties": {"analysis": {"type": "string"}}},
    "properties": {"toolType": "analyze-attachments"},
    "settings": {},
}


class Gateway:
    """The LLM Gateway client: records requests, answers with ``reply``."""

    def __init__(self, status: int = 200, reply: str = "The total is 42.") -> None:
        self.status, self.reply = status, reply
        self.requests: list[dict[str, Any]] = []
        self.models: list[str] = []
        self.closed = 0

    async def request(self, method: str, path: str, *, headers, content) -> Any:
        self.requests.append({"headers": headers, "body": json.loads(content)})
        body = (
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": self.reply}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            if self.status < 400
            else {"error": {"message": "unsupported file type"}}
        )
        return httpx.Response(self.status, json=body)

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture
def services(monkeypatch):
    gateway = Gateway()
    sizes: dict[str, int] = {}

    async def blob_uri(*, key):
        return SimpleNamespace(uri=f"https://blob.test/{key}", name="x")

    client = SimpleNamespace(
        attachments=SimpleNamespace(get_blob_file_access_uri_async=blob_uri)
    )

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.strip("/")
        data = BLOBS[key] * sizes.get(key, 1)
        return httpx.Response(200, content=data)

    async def gateway_client(model: str) -> Gateway:
        gateway.models.append(model)
        return gateway

    monkeypatch.setattr(at, "_uipath_client", lambda: client)
    monkeypatch.setattr(
        at,
        "_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(at, "_gateway_client", gateway_client)
    return SimpleNamespace(gateway=gateway, sizes=sizes)


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "get_tracer", lambda: provider.get_tracer("t"))
    return exporter


def _ctx(model: str | None = "gpt-agent") -> ToolContext:
    store = AttachmentStore(
        [UiPathAttachment.model_validate(PNG), UiPathAttachment.model_validate(PDF)]
    )
    return ToolContext("call_af", "rt", "Analyze_Files", None, store, model)


async def test_files_go_to_the_agent_model_as_content_parts(services, spans):
    tool = uipath_analyze_files_tool()
    assert tool.name == "Analyze_Files" and tool.tool_type == "Internal"
    assert attachment_paths(tool.input_schema) == [("attachments", "*")]
    text = await tool.call(
        {"attachments": [PNG, PDF], "analysisTask": "What is the total?"}, _ctx()
    )
    assert json.loads(text) == {"analysisResult": "The total is 42."}
    [request] = services.gateway.requests
    assert services.gateway.models == ["gpt-agent"] and services.gateway.closed == 1
    body = request["body"]
    assert body["instructions"] == at.ANALYZE_FILES_SYSTEM_MESSAGE
    assert body["stream"] is False and body["model"] == "gpt-agent"
    [message] = body["input"]
    task, image, document = message["content"]
    assert task == {"type": "input_text", "text": "What is the total?"}
    assert image == {
        "type": "input_image",
        "image_url": "data:image/png;base64,"
        + base64.b64encode(BLOBS[PNG_ID]).decode(),
    }
    assert document["type"] == "input_file"
    assert document["filename"] == "q3 report-pdf"  # LangChain's sanitizing
    assert document["file_data"].startswith("data:application/pdf;base64,")

    finished = {s.name: s for s in spans.get_finished_spans()}
    internal, llm = finished["Analyze Files"], finished["LLM call"]
    assert llm.parent.span_id == internal.context.span_id
    assert internal.attributes["span_type"] == "toolCall"
    assert internal.attributes["toolName"] == "Analyze Files"
    assert internal.attributes["tool.name"] == "Analyze Files"
    assert internal.attributes["toolType"] == "Internal"
    # Not a second call to the tool-call evaluators: the call is "Tool call".
    from uipath.eval._helpers.evaluators_helpers import extract_tool_calls_names

    assert internal.attributes["tool.synthesized"] is True
    assert extract_tool_calls_names([internal]) == []
    assert json.loads(internal.attributes["output"]) == {
        "analysisResult": "The total is 42."
    }
    assert llm.attributes["span_type"] == "llmCall"
    assert llm.attributes["input"] == "What is the total?"
    assert llm.attributes["model"] == "gpt-agent"
    sent = json.loads(llm.attributes["attachments"])
    assert [(a["id"], a["direction"]) for a in sent] == [(PNG_ID, 1), (PDF_ID, 1)]
    assert "base64" not in json.dumps(dict(llm.attributes))


async def test_no_attachments_short_circuits(services):
    tool = uipath_analyze_files_tool(model="gpt-fixed")
    text = await tool.call({"attachments": [], "analysisTask": "x"}, _ctx())
    assert json.loads(text) == {"analysisResult": "No attachments provided to analyze."}
    assert services.gateway.requests == []
    with pytest.raises(UiPathToolError, match="analysisTask"):
        await tool.call({"attachments": [PNG], "analysisTask": " "}, _ctx())


async def test_a_file_over_the_limit_fails_the_run(services):
    tool = uipath_analyze_files_tool()
    tool.max_file_bytes = 64
    services.sizes[PDF_ID] = 10  # 140 bytes
    with pytest.raises(UiPathCodexRuntimeError) as info:
        await tool.call({"attachments": [PDF], "analysisTask": "x"}, _ctx())
    assert info.value.error_info.title == "File attachment too large"
    assert "FILE_ERROR" in str(info.value.error_info.code)
    assert services.gateway.requests == []


async def test_gateway_errors_and_unsupported_types_are_tool_errors(services):
    tool = uipath_analyze_files_tool(model="gpt-fixed")
    services.gateway.status = 400
    with pytest.raises(UiPathToolError, match="unsupported file type"):
        await tool.call({"attachments": [PDF], "analysisTask": "x"}, _ctx())
    assert services.gateway.models == ["gpt-fixed"]
    tiff = {"ID": PNG_ID, "FullName": "scan.tiff", "MimeType": "image/tiff"}
    with pytest.raises(UiPathToolError, match="cannot analyze"):
        await tool.call({"attachments": [tiff], "analysisTask": "x"}, _ctx())


def test_from_resource():
    tool = analyze_files_tool_from_resource(RESOURCE)
    assert tool.name == "Analyze_Files" and tool.display_name == "Analyze Files"
    assert tool.description == "Analyze the attached files."
    assert tool.input_schema["required"] == ["attachments", "analysisTask"]
    with pytest.raises(ValueError, match="internal tool"):
        analyze_files_tool_from_resource(
            {**RESOURCE, "properties": {"toolType": "batch-transform"}}
        )
    with pytest.raises(ValueError, match="disabled"):
        analyze_files_tool_from_resource({**RESOURCE, "isEnabled": False})


async def test_from_resource_with_attachments_from_the_agent_input(services):
    tool = analyze_files_tool_from_resource(
        {
            **RESOURCE,
            "argumentProperties": {
                "attachments": {
                    "variant": "argument",
                    "argumentPath": "$.document",
                    "isSensitive": False,
                }
            },
        }
    )
    assert tool.input_schema["required"] == ["analysisTask"]
    ctx = ToolContext(
        "c", "rt", "Analyze_Files", {"document": PDF}, _ctx().attachments, "gpt-agent"
    )
    text = await tool.call({"analysisTask": "Summarize"}, ctx)
    assert json.loads(text)["analysisResult"] == "The total is 42."
    [request] = services.gateway.requests
    assert request["body"]["input"][0]["content"][1]["type"] == "input_file"


async def test_from_resource_jsonpath_keys_and_array_builder(services):
    # argumentProperties keys as Agent Builder writes them: JSONPath, and
    # the attachments list built from the agent input.
    tool = analyze_files_tool_from_resource(
        {
            **RESOURCE,
            "argumentProperties": {
                "$['attachments']": {"variant": "arrayBuilder"},
                "$['attachments'][0]": {
                    "variant": "argument",
                    "argumentPath": "$['document']",
                    "isSensitive": False,
                },
                "$['analysisTask']": {
                    "variant": "static",
                    "value": "Summarize",
                    "isSensitive": False,
                },
            },
        }
    )
    assert set(tool.fixed_args) == {"attachments", "analysisTask"}
    assert tool.input_schema["required"] == []
    ctx = ToolContext(
        "c", "rt", "Analyze_Files", {"document": PDF}, _ctx().attachments, "gpt-agent"
    )
    text = await tool.call({}, ctx)
    assert json.loads(text)["analysisResult"] == "The total is 42."
    [request] = services.gateway.requests
    assert request["body"]["input"][0]["content"][1]["type"] == "input_file"


def _from_input_tool():
    """Analyze Files whose ``attachments`` come from the agent input's ``document``."""
    return analyze_files_tool_from_resource(
        {
            **RESOURCE,
            "argumentProperties": {
                "attachments": {
                    "variant": "argument",
                    "argumentPath": "$.document",
                    "isSensitive": False,
                }
            },
        }
    )


def _input_ctx(agent_input: dict[str, Any] | None) -> ToolContext:
    return ToolContext(
        "c", "rt", "Analyze_Files", agent_input, _ctx().attachments, "gpt-agent"
    )


def test_argument_property_stays_visible_and_optional():
    tool = _from_input_tool()
    schema = tool.input_schema
    assert list(schema["properties"]) == ["attachments", "analysisTask"]
    assert (
        "Set from the agent input 'document'"
        in (schema["properties"]["attachments"]["description"])
    )
    # Still an attachment field: the host resolves what the model passes.
    assert attachment_paths(schema) == [("attachments", "*")]
    assert tool.validate_args({"analysisTask": "x"}) == {"analysisTask": "x"}
    ctx = _input_ctx({"document": PDF})
    assert tool.span_args({"analysisTask": "x"}, ctx) == {
        "analysisTask": "x",
        "attachments": PDF,
    }


async def test_unresolved_argument_property_keeps_the_model_attachments(services):
    tool = _from_input_tool()
    text = await tool.call(
        {"attachments": [PNG], "analysisTask": "Describe"}, _input_ctx({})
    )
    assert json.loads(text)["analysisResult"] == "The total is 42."
    [request] = services.gateway.requests
    assert request["body"]["input"][0]["content"][1]["type"] == "input_image"
    # The agent input wins when it has a value.
    await tool.call(
        {"attachments": [PNG], "analysisTask": "Describe"},
        _input_ctx({"document": PDF}),
    )
    assert services.gateway.requests[1]["body"]["input"][0]["content"][1]["type"] == (
        "input_file"
    )


async def test_unresolved_required_argument_property_is_a_tool_error(services):
    tool = _from_input_tool()
    with pytest.raises(UiPathToolError) as info:
        await tool.call({"analysisTask": "Describe"}, _input_ctx({}))
    assert str(info.value) == (
        "Invalid arguments for tool 'Analyze_Files': missing required "
        "argument(s): attachments. The agent input has no value for them; fill "
        "them in and call the tool again."
    )
    assert services.gateway.requests == []


def test_trace_provider_untouched():
    # The fixture patches the package tracer only.
    assert trace.get_tracer_provider() is not None
