"""Job attachments: discovery, the known-attachment store, schemas, prompt,
output, tool-host resolution and process tools."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from uipath_codex_sdk import (
    AttachmentStore,
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_process_tool,
)
from uipath_codex_sdk.attachments import (
    attachment_paths,
    not_found_message,
    span_attachments,
)
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.runtime import (
    ATTACHMENT_INSTRUCTIONS,
    OUTPUT_ATTACHMENT_INSTRUCTIONS,
    UiPathCodexRuntime,
)
from uipath_codex_sdk.runtime.schema import (
    get_entrypoint_schema,
    to_strict_output_schema,
)

from ._typing import attrs
from .test_tool_host import _provider, _text, make_host  # noqa: F401 - fixture

INVOICE_ID = "11111111-2222-3333-4444-000000000001"
REPORT_ID = "11111111-2222-3333-4444-000000000002"
UNKNOWN_ID = "11111111-2222-3333-4444-00000000ffff"
INVOICE = {"ID": INVOICE_ID, "FullName": "invoice.pdf", "MimeType": "application/pdf"}
REPORT = {"ID": REPORT_ID, "FullName": "report.md", "MimeType": "text/markdown"}


class Input(BaseModel):
    invoice: UiPathAttachment
    extra: list[UiPathAttachment] = []
    question: str


class Output(BaseModel):
    summary: str
    report: UiPathAttachment | None = None


def _attachment(data: dict[str, Any]) -> UiPathAttachment:
    return UiPathAttachment.model_validate(data)


def _runtime(**agent: Any) -> UiPathCodexRuntime:
    return UiPathCodexRuntime(
        agent=UiPathCodexAgent(model="m", **agent),
        runtime_id="rt",
        entrypoint="agent",
        file_path="/tmp/main.py",
    )


# --- discovery ----------------------------------------------------------------


def test_paths_of_pydantic_platform_and_raw_schemas():
    assert attachment_paths(Input.model_json_schema()) == [("invoice",), ("extra", "*")]
    platform = get_entrypoint_schema(
        UiPathCodexAgent(model="m", input_schema=Input), "agent"
    ).input
    assert attachment_paths(platform) == [("invoice",), ("extra", "*")]
    raw = {
        "type": "object",
        "properties": {
            "doc": {"anyOf": [{"$ref": "#/$defs/File"}, {"type": "null"}]},
            "nested": {
                "type": "object",
                "properties": {"file": {"x-uipath-resource-kind": "JobAttachment"}},
            },
            "plain": {"type": "string"},
        },
        "$defs": {"File": {"type": "object", "title": "UiPathAttachment"}},
    }
    assert attachment_paths(raw) == [("doc",), ("nested", "file")]
    assert attachment_paths({"type": "object", "properties": {"a": {}}}) == []


def test_platform_schemas_carry_the_job_attachment_definition():
    schema = get_entrypoint_schema(
        UiPathCodexAgent(model="m", input_schema=Input, output_schema=Output), "agent"
    )
    assert schema.input["properties"]["invoice"] == {
        "$ref": "#/definitions/job-attachment"
    }
    definition = schema.input["definitions"]["job-attachment"]
    assert definition["x-uipath-resource-kind"] == "JobAttachment"
    assert definition["required"] == ["ID"]
    assert schema.output["properties"]["report"] == {
        "$ref": "#/definitions/job-attachment"
    }


def test_strict_output_schema_asks_for_a_known_id_only():
    strict = to_strict_output_schema(Output.model_json_schema())
    [attachment] = strict["$defs"].values()
    assert attachment["properties"].keys() == {"ID"}
    assert attachment["required"] == ["ID"]
    assert attachment["additionalProperties"] is False
    assert strict["required"] == ["summary", "report"]
    assert "Metadata" not in json.dumps(strict)


def test_strict_output_schema_moves_described_refs_into_any_of():
    from pydantic import Field

    class Described(BaseModel):
        report: UiPathAttachment = Field(description="The report")

    strict = to_strict_output_schema(Described.model_json_schema())
    report = strict["properties"]["report"]
    assert "$ref" not in report and report["description"] == "The report"
    assert report["anyOf"] == [{"$ref": "#/$defs/Attachment"}]


def test_tool_schema_from_a_model_uses_the_reference():
    tool = uipath_process_tool("p", input_schema=Input)
    assert tool.input_schema["properties"]["invoice"] == {
        "$ref": "#/definitions/job-attachment"
    }
    assert attachment_paths(tool.input_schema) == [("invoice",), ("extra", "*")]


# --- the store ----------------------------------------------------------------


def test_resolve_fills_in_known_attachments_and_reports_unknown_ones():
    store = AttachmentStore([_attachment(INVOICE)])
    paths = [("invoice",), ("extra", "*")]
    args = {
        "invoice": {"ID": INVOICE_ID},
        "extra": [json.dumps({"ID": INVOICE_ID.upper()}), {"ID": UNKNOWN_ID}],
        "question": "q",
    }
    resolved, errors = store.resolve(paths, args)
    assert resolved["invoice"] == INVOICE
    assert resolved["extra"][0] == INVOICE  # a JSON string and any case
    assert errors == [not_found_message(UNKNOWN_ID)]
    assert "Could not find JobAttachment with ID='" in errors[0]
    assert args["invoice"] == {"ID": INVOICE_ID}  # not changed in place
    _, errors = store.resolve(paths, {"invoice": {"ID": "not-a-uuid"}})
    assert errors == [not_found_message("not-a-uuid")]


def test_harvest_and_round_trip():
    store = AttachmentStore()
    found = store.harvest(
        [("files", "*")], {"files": [REPORT, {"ID": "bad"}, None], "other": INVOICE}
    )
    assert [str(a.id) for a in found] == [REPORT_ID]
    assert store.changed and REPORT_ID in store and INVOICE_ID not in store
    again = AttachmentStore.from_json(json.loads(json.dumps(store.to_json())))
    assert again.to_json() == [REPORT] and not again.changed


def test_span_attachments_shape():
    value = span_attachments([_attachment(INVOICE)], "in")
    assert value is not None
    assert json.loads(value) == [
        {
            "id": INVOICE_ID,
            "fileName": "invoice.pdf",
            "mimeType": "application/pdf",
            "provider": 0,
            "direction": 1,
        }
    ]
    out = span_attachments([_attachment(INVOICE)], "out")
    assert out is not None
    assert json.loads(out)[0]["direction"] == 2
    assert span_attachments([], "in") is None


# --- prompt, instructions, output ----------------------------------------------


def test_prompt_renders_attachments_as_json():
    runtime = _runtime(
        input_schema=Input,
        prompt="Answer {question} about {invoice} (named {invoice[FullName]}).",
    )
    text = runtime._render_prompt(
        {"invoice": INVOICE, "extra": [REPORT], "question": "the total"}
    )
    first, block = text.split("\n\n")
    assert first == (
        f"Answer the total about {json.dumps(INVOICE, separators=(',', ':'))} "
        "(named invoice.pdf)."
    )
    # ``extra`` is not in the template: listed after it.
    assert block.startswith("Input attachments (JobAttachment objects):")
    assert json.loads(block.split("\n", 1)[1]) == [REPORT]


def test_prompt_without_a_template_lists_the_attachments():
    class Ask(BaseModel):
        prompt: str
        file: UiPathAttachment

    runtime = _runtime(input_schema=Ask)
    text = runtime._render_prompt({"prompt": "Summarize it.", "file": INVOICE})
    assert text.startswith("Summarize it.\n\nInput attachments")
    assert INVOICE_ID in text


def test_input_attachments_and_instructions():
    runtime = _runtime(input_schema=Input, output_schema=Output, prompt="{question}")
    found = runtime.input_attachments({"invoice": INVOICE, "question": "q"})
    assert [str(a.id) for a in found] == [INVOICE_ID]
    assert runtime.input_attachments({"question": "no invoice"}) == []
    assert runtime._developer_instructions({}) == (
        f"{ATTACHMENT_INSTRUCTIONS} {OUTPUT_ATTACHMENT_INSTRUCTIONS}"
    )
    assert _runtime(prompt="hi")._developer_instructions({}) == ""
    tool_only = _runtime(tools=[uipath_process_tool("p", input_schema=Input)])
    assert ATTACHMENT_INSTRUCTIONS in tool_only._developer_instructions({})


def test_output_fills_in_a_known_attachment():
    runtime = _runtime(output_schema=Output)
    runtime._attachments = AttachmentStore([_attachment(REPORT)])
    output = runtime._build_output(
        json.dumps({"summary": "done", "report": {"ID": REPORT_ID}})
    )
    assert output == {"summary": "done", "report": {**REPORT, "Metadata": None}}
    assert [str(a.id) for a in runtime.output_attachments(output)] == [REPORT_ID]
    assert runtime._build_output(json.dumps({"summary": "s", "report": None})) == {
        "summary": "s",
        "report": None,
    }


def test_output_with_an_unknown_attachment_fails_the_run():
    runtime = _runtime(output_schema=Output)
    with pytest.raises(UiPathCodexRuntimeError) as info:
        runtime._build_output(
            json.dumps({"summary": "s", "report": {"ID": UNKNOWN_ID}})
        )
    assert "OUTPUT_INVALID" in str(info.value.error_info.code)
    assert UNKNOWN_ID in info.value.error_info.detail


# --- tool host ------------------------------------------------------------------


class AttachmentTool:
    """An immediate tool with an attachment argument and attachment output."""

    name = "inspect_file"
    description = "Inspects a file."
    input_schema = {
        "type": "object",
        "properties": {
            "file": {"$ref": "#/definitions/job-attachment"},
            "note": {"type": "string"},
        },
        "required": ["file"],
        "definitions": {
            "job-attachment": {
                "type": "object",
                "x-uipath-resource-kind": "JobAttachment",
                "properties": {"ID": {"type": "string"}},
            }
        },
    }
    output_json_schema = {
        "type": "object",
        "properties": {"copy": {"x-uipath-resource-kind": "JobAttachment"}},
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return dict(args)

    async def call(self, args: dict[str, Any], ctx: Any) -> str:
        self.calls.append(args)
        assert ctx.attachments is not None and ctx.model == "gpt-x"
        return json.dumps({"copy": REPORT})


async def test_host_resolves_arguments_and_harvests_results(make_host):  # noqa: F811
    exporter, tracer = _provider()
    store = AttachmentStore([_attachment(INVOICE)])
    tool = AttachmentTool()
    h = await make_host([tool], tracer=tracer, attachments=store, model="gpt-x")
    _, body = await h.call_tool("inspect_file", {"file": {"ID": INVOICE_ID}})
    assert body["result"]["isError"] is False
    assert tool.calls == [{"file": INVOICE}]
    assert REPORT_ID in store  # harvested by the output schema
    [span] = exporter.get_finished_spans()
    attachments = json.loads(attrs(span)["attachments"])
    assert [(a["id"], a["direction"]) for a in attachments] == [
        (INVOICE_ID, 1),
        (REPORT_ID, 2),
    ]


async def test_host_answers_an_unknown_id_without_running_the_tool(make_host):  # noqa: F811
    tool = AttachmentTool()
    h = await make_host([tool], attachments=AttachmentStore(), model="gpt-x")
    _, body = await h.call_tool("inspect_file", {"file": {"ID": UNKNOWN_ID}})
    assert body["result"]["isError"] is True
    assert _text(body) == not_found_message(UNKNOWN_ID)
    assert tool.calls == []


async def test_a_runtime_error_in_a_tool_fails_the_run(make_host):  # noqa: F811
    from uipath.runtime.errors import UiPathErrorCategory

    from uipath_codex_sdk.runtime.errors import UiPathCodexErrorCode

    error = UiPathCodexRuntimeError(
        UiPathCodexErrorCode.FILE_ERROR,
        "File attachment too large",
        "too big",
        UiPathErrorCategory.USER,
    )

    class Failing(AttachmentTool):
        async def call(self, args: dict[str, Any], ctx: Any) -> str:
            raise error

    h = await make_host(
        [Failing()], attachments=AttachmentStore([_attachment(INVOICE)])
    )
    _, body = await h.call_tool("inspect_file", {"file": {"ID": INVOICE_ID}})
    assert body["result"]["isError"] is True
    assert h.host.fatal_error is error and h.host.fatal.is_set()


# --- process tools ----------------------------------------------------------------


async def test_process_tool_links_attachments_to_the_job(monkeypatch):
    from types import SimpleNamespace

    from uipath.platform.orchestrator import Job

    import uipath_codex_sdk.tools as tools_mod
    from uipath_codex_sdk import DurableToolContext

    calls: list[dict[str, Any]] = []

    async def invoke(**kwargs: Any) -> Job:
        calls.append(kwargs)
        return Job.model_validate({"Id": 1, "Key": "k", "State": "Pending"})

    monkeypatch.setattr(
        tools_mod,
        "_uipath_client",
        lambda: SimpleNamespace(processes=SimpleNamespace(invoke_async=invoke)),
    )
    tool = uipath_process_tool("Summarize", input_schema=Input)
    args = tool.validate_args({"invoice": INVOICE, "extra": [REPORT], "question": "q"})
    assert args["invoice"] == {**INVOICE, "Metadata": None}  # the wire shape
    await tool.start(args, DurableToolContext("c", "rt", "Summarize"))
    [call] = calls
    assert [str(a.id) for a in call["attachments"]] == [INVOICE_ID, REPORT_ID]
    assert call["input_arguments"]["invoice"]["ID"] == INVOICE_ID
