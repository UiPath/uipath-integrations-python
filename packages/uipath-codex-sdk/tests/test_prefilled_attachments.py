"""Pre-filled attachment arguments: the host checks the merged arguments.

As in LangChain, whose ``StaticArgsHandler`` merges the pre-filled values
(``argumentProperties`` / ``static_args`` / ``args_from_input``) into the tool
call before ``job_attachment_wrapper`` resolves its job attachments: a model
value a pre-filled one replaces is not checked (the pre-filled one is), and a
model value no pre-filled one replaces is resolved or refused as before.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    AttachmentStore,
    UiPathAttachment,
    analyze_files_tool_from_resource,
    uipath_escalation_tool,
    uipath_process_tool,
)
from uipath_codex_sdk._fixed_args import fixed_args_from_code, with_fixed_args
from uipath_codex_sdk.attachment_tools import UiPathAnalyzeFilesTool
from uipath_codex_sdk.attachments import not_found_message
from uipath_codex_sdk.batch_transform_tools import batch_transform_tool_from_resource
from uipath_codex_sdk.context_tools import deep_rag_tool_from_resource

from .test_analyze_files import PDF, PNG, PNG_ID, services  # noqa: F401 - fixture
from .test_analyze_files import RESOURCE as ANALYZE_RESOURCE
from .test_attachments import (
    INVOICE,
    INVOICE_ID,
    REPORT,
    REPORT_ID,
    UNKNOWN_ID,
    AttachmentTool,
    Input,
)
from .test_batch_transform_attachments import CSV
from .test_batch_transform_attachments import RESOURCE as BATCH_RESOURCE
from .test_deep_rag_attachments import DOC
from .test_deep_rag_attachments import RESOURCE as DEEP_RAG_RESOURCE
from .test_escalation_tools import fake  # noqa: F401 - fixture
from .test_mocking import (
    RESULT_ID,
    _by_name,
    _with_attachment_argument,
    mockito,
    returns,
    serve,  # noqa: F401 - fixture
    simulating,
    spans,  # noqa: F401 - fixture
)
from .test_mocking import _text as _result_text
from .test_tool_host import _provider, _text, _until, make_host  # noqa: F401

INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice": {"$ref": "#/definitions/job-attachment"},
        "question": {"type": "string"},
    },
    "required": ["invoice", "question"],
    "definitions": {
        "job-attachment": {
            "type": "object",
            "x-uipath-resource-kind": "JobAttachment",
            "properties": {"ID": {"type": "string"}},
        }
    },
}


def _known(*items: dict[str, Any]) -> AttachmentStore:
    return AttachmentStore(UiPathAttachment.model_validate(a) for a in items)


@pytest.fixture
def jobs(monkeypatch):
    """Orchestrator's ``invoke_async``: records each job start."""
    from uipath.platform.orchestrator import Job

    calls: list[dict[str, Any]] = []

    async def invoke(**kwargs: Any) -> Job:
        calls.append(kwargs)
        return Job.model_validate({"Id": 1, "Key": "k", "State": "Pending"})

    monkeypatch.setattr(
        tools_mod,
        "_uipath_client",
        lambda: SimpleNamespace(processes=SimpleNamespace(invoke_async=invoke)),
    )
    return calls


async def _held(h: Any, name: str, args: dict[str, Any]) -> asyncio.Task[Any]:
    """Call durable tool ``name`` and wait until its job started and it is held."""
    task = asyncio.create_task(h.call_tool(name, args))
    await _until(lambda: bool(h.host.held) or task.done())
    assert not task.done(), _text(task.result()[1])
    return task


# --- durable process tools ----------------------------------------------------


@pytest.mark.parametrize("model_value", [{"ID": UNKNOWN_ID}, {"ID": "not-a-uuid"}])
async def test_agent_input_attachment_wins_over_an_unknown_model_id(
    make_host,  # noqa: F811
    jobs,
    model_value,
):
    tool = uipath_process_tool(
        "Summarize", input_schema=Input, args_from_input={"invoice": "doc"}
    )
    h = await make_host(
        [tool],
        with_store=True,
        agent_input={"doc": INVOICE},
        attachments=_known(INVOICE),
    )
    args = {"invoice": model_value, "question": "q"}
    held = await _held(h, "Summarize", args)
    [call] = jobs
    assert call["input_arguments"]["invoice"]["ID"] == INVOICE_ID
    assert [str(a.id) for a in call["attachments"]] == [INVOICE_ID]
    # The model's arguments are what is saved for resume, as sent.
    pending = (await h.store.get_pending())["call_is"]
    assert pending.args == args
    await h.host.release_all()
    await asyncio.wait_for(held, 5)


async def test_a_static_attachment_wins_over_an_unknown_model_id(
    make_host,  # noqa: F811
    jobs,
):
    tool = uipath_process_tool(
        "Summarize", input_schema=INVOICE_SCHEMA, static_args={"invoice": INVOICE}
    )
    assert "invoice" not in tool.input_schema["properties"]
    h = await make_host([tool], with_store=True, attachments=_known(INVOICE))
    # The model passes the hidden field anyway: the static value replaces it.
    held = await _held(h, "Summarize", {"invoice": {"ID": UNKNOWN_ID}, "question": "q"})
    [call] = jobs
    assert call["input_arguments"]["invoice"] == INVOICE
    await h.host.release_all()
    await asyncio.wait_for(held, 5)


async def test_a_static_attachment_the_run_does_not_know_is_refused(
    make_host,  # noqa: F811
    jobs,
):
    # As LangChain, which resolves the tool call once the static args are in.
    tool = uipath_process_tool(
        "Summarize", input_schema=INVOICE_SCHEMA, static_args={"invoice": REPORT}
    )
    h = await make_host([tool], with_store=True, attachments=_known(INVOICE))
    _, body = await h.call_tool("Summarize", {"question": "q"})
    assert body["result"]["isError"] is True
    assert _text(body) == not_found_message(REPORT_ID)
    assert jobs == [] and await h.store.get_pending() == {}


async def test_an_unknown_model_id_with_no_prefill_for_it_is_refused(
    make_host,  # noqa: F811
    jobs,
):
    # The agent input has no ``doc`` (so the model's value stays), and the
    # static field is another one.
    tool = uipath_process_tool(
        "Summarize",
        input_schema=Input,
        static_args={"question": "q"},
        args_from_input={"invoice": "doc"},
    )
    h = await make_host(
        [tool], with_store=True, agent_input={}, attachments=_known(INVOICE)
    )
    _, body = await h.call_tool("Summarize", {"invoice": {"ID": UNKNOWN_ID}})
    assert body["result"]["isError"] is True
    assert _text(body) == not_found_message(UNKNOWN_ID)
    _, body = await h.call_tool(
        "Summarize", {"invoice": {"ID": "not-a-uuid"}}, call_id="call_2"
    )
    assert _text(body) == not_found_message("not-a-uuid")
    assert jobs == [] and await h.store.get_pending() == {}
    # A known one is resolved, and is the job's.
    held = await _held(h, "Summarize", {"invoice": {"ID": INVOICE_ID}})
    [call] = jobs
    assert call["input_arguments"]["invoice"]["FullName"] == "invoice.pdf"
    assert call["input_arguments"]["question"] == "q"
    await h.host.release_all()
    await asyncio.wait_for(held, 5)


async def test_an_unknown_model_id_in_a_list_item_no_prefill_replaces(
    make_host,  # noqa: F811
    jobs,
):
    # ``invoice`` comes from the agent input; ``extra`` is the model's.
    tool = uipath_process_tool(
        "Summarize", input_schema=Input, args_from_input={"invoice": "doc"}
    )
    h = await make_host(
        [tool],
        with_store=True,
        agent_input={"doc": INVOICE},
        attachments=_known(INVOICE, REPORT),
    )
    _, body = await h.call_tool(
        "Summarize",
        {
            "invoice": {"ID": UNKNOWN_ID},
            "extra": [{"ID": REPORT_ID}, {"ID": UNKNOWN_ID}],
            "question": "q",
        },
    )
    assert body["result"]["isError"] is True
    assert _text(body) == not_found_message(UNKNOWN_ID)  # only extra[1]'s
    assert jobs == []


async def test_an_invalid_model_value_in_an_input_field_is_still_refused(
    make_host,  # noqa: F811
    jobs,
):
    # Only an attachment the agent input fills is excused before the merge: a
    # model value of the wrong type elsewhere is refused up front, as before.
    class Order(BaseModel):
        sku: str
        customer: int

    tool = uipath_process_tool(
        "P", input_schema=Order, args_from_input={"customer": "user.id"}
    )
    with pytest.raises(ValueError):
        tool.validate_args({"sku": "A", "customer": "not-int"})
    assert tool.validate_args({"sku": "A"}) == {"sku": "A"}  # missing: excused
    h = await make_host([tool], with_store=True, agent_input={})
    _, body = await h.call_tool("P", {"sku": "A", "customer": "not-int"})
    assert body["result"]["isError"] is True
    assert _text(body).startswith("Invalid arguments for tool 'P':")
    assert jobs == [] and await h.store.get_pending() == {}


# --- escalation: LangChain never resolves its attachments ----------------------

RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "Amount": {"type": "number"},
        "Receipt": {"$ref": "#/definitions/job-attachment"},
    },
    "required": ["Amount", "Receipt"],
    "definitions": INVOICE_SCHEMA["definitions"],
}


async def test_escalation_static_attachment_the_run_does_not_know_creates_the_task(
    make_host,  # noqa: F811
    fake,  # noqa: F811
):
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        folder_path="Shared",
        input_schema=RECEIPT_SCHEMA,
        static_args={"Receipt": REPORT},
    )
    h = await make_host([tool], with_store=True, attachments=_known(INVOICE))
    task = asyncio.create_task(h.call_tool(tool.name, {"Amount": 1}))
    await _until(lambda: bool(h.host.held) or task.done())
    assert not task.done(), _text(task.result()[1])
    [created] = fake.created
    assert created["data"]["Receipt"] == REPORT
    await h.host.release_all()
    await asyncio.wait_for(task, 5)


async def test_escalation_input_attachment_is_not_checked(
    make_host,  # noqa: F811
    fake,  # noqa: F811
):
    # The agent input's attachment replaces the model's known one; neither is
    # checked against the run's attachments (the input's is unknown here).
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        folder_path="Shared",
        input_schema=RECEIPT_SCHEMA,
        args_from_input={"Receipt": "receipt"},
    )
    h = await make_host(
        [tool],
        with_store=True,
        agent_input={"receipt": REPORT},
        attachments=_known(INVOICE),
    )
    args = {"Amount": 1, "Receipt": {"ID": INVOICE_ID}}
    task = asyncio.create_task(h.call_tool(tool.name, args))
    await _until(lambda: bool(h.host.held) or task.done())
    assert not task.done(), _text(task.result()[1])
    [created] = fake.created
    assert created["data"]["Receipt"] == REPORT
    await h.host.release_all()
    await asyncio.wait_for(task, 5)


# --- Analyze Files (immediate) --------------------------------------------------


def _analyze_from(argument_properties: dict[str, Any]) -> UiPathAnalyzeFilesTool:
    return analyze_files_tool_from_resource(
        {**ANALYZE_RESOURCE, "argumentProperties": argument_properties}
    )


async def test_analyze_files_array_builder_wins_over_unknown_model_ids(
    make_host,  # noqa: F811
    services,  # noqa: F811
):
    tool = _analyze_from(
        {
            "$['attachments']": {"variant": "arrayBuilder"},
            "$['attachments'][0]": {
                "variant": "argument",
                "argumentPath": "$['document']",
                "isSensitive": False,
            },
        }
    )
    h = await make_host(
        [tool], agent_input={"document": PDF}, attachments=_known(PNG, PDF), model="m"
    )
    _, body = await h.call_tool(
        "Analyze_Files",
        {"attachments": [{"ID": UNKNOWN_ID}, {"ID": "x"}], "analysisTask": "Sum"},
    )
    assert body["result"]["isError"] is False, _text(body)
    [request] = services.gateway.requests
    [_, part] = request["body"]["input"][0]["content"]
    assert part["type"] == "input_file" and part["file_data"].startswith(
        "data:application/pdf"
    )


async def test_analyze_files_static_items_win_over_unknown_model_ids(
    make_host,  # noqa: F811
    services,  # noqa: F811
):
    # ``attachments[*]``: every item the model sends becomes the static one.
    exporter, tracer = _provider()
    tool = UiPathAnalyzeFilesTool(
        name="Analyze_Files",
        description="Analyze.",
        fixed_args=fixed_args_from_code({"attachments[*]": PNG}, None),
    )
    h = await make_host([tool], tracer=tracer, attachments=_known(PNG), model="m")
    _, body = await h.call_tool(
        "Analyze_Files",
        {"attachments": [{"ID": UNKNOWN_ID}, {"ID": UNKNOWN_ID}], "analysisTask": "?"},
    )
    assert body["result"]["isError"] is False, _text(body)
    [request] = services.gateway.requests
    types = [p["type"] for p in request["body"]["input"][0]["content"][1:]]
    assert types == ["input_image", "input_image"]
    outer = _by_name(exporter)["Tool call - Analyze_Files"]
    assert json.loads(outer.attributes["input"])["attachments"] == [PNG, PNG]
    listed = json.loads(outer.attributes["attachments"])
    assert [a["id"] for a in listed] == [PNG_ID, PNG_ID]


async def test_analyze_files_argument_with_no_input_value_checks_the_model_ids(
    make_host,  # noqa: F811
    services,  # noqa: F811
):
    tool = _analyze_from(
        {
            "attachments": {
                "variant": "argument",
                "argumentPath": "$.documents",
                "isSensitive": False,
            }
        }
    )
    # The agent input has the files: the model's unknown one is replaced.
    h = await make_host(
        [tool],
        agent_input={"documents": [PDF]},
        attachments=_known(PNG, PDF),
        model="m",
    )
    args = {"attachments": [{"ID": UNKNOWN_ID}], "analysisTask": "Sum"}
    _, body = await h.call_tool("Analyze_Files", args)
    assert body["result"]["isError"] is False, _text(body)
    # It has none: the model's are checked.
    h = await make_host([tool], agent_input={}, attachments=_known(PNG, PDF), model="m")
    _, body = await h.call_tool("Analyze_Files", args)
    assert body["result"]["isError"] is True
    assert _text(body) == not_found_message(UNKNOWN_ID)
    assert len(services.gateway.requests) == 1


# --- custom tools ------------------------------------------------------------------


class PrefilledAttachmentTool(AttachmentTool):
    """A custom tool whose ``file`` is pre-filled from the agent input."""

    prefilled_schema = AttachmentTool.input_schema

    def prefilled_args(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(args)
        if agent_input and "doc" in agent_input:
            merged["file"] = agent_input["doc"]
        return merged


async def test_a_custom_tool_hook(make_host):  # noqa: F811
    tool = PrefilledAttachmentTool()
    h = await make_host(
        [tool], agent_input={"doc": INVOICE}, attachments=_known(INVOICE), model="gpt-x"
    )
    _, body = await h.call_tool("inspect_file", {"file": {"ID": UNKNOWN_ID}})
    assert body["result"]["isError"] is False, _text(body)
    assert tool.calls == [{"file": {"ID": UNKNOWN_ID}}]  # the model's, as sent
    h = await make_host([tool], agent_input={}, attachments=_known(INVOICE))
    _, body = await h.call_tool("inspect_file", {"file": {"ID": UNKNOWN_ID}})
    assert _text(body) == not_found_message(UNKNOWN_ID)


# --- simulated (eval-mocked) calls ---------------------------------------------------


async def test_mocked_process_call_sees_the_prefilled_attachment(
    serve,  # noqa: F811
    spans,  # noqa: F811
):
    tool = uipath_process_tool(
        "Summarize", input_schema=INVOICE_SCHEMA, static_args={"invoice": INVOICE}
    )
    merged = {"invoice": INVOICE, "question": "q"}
    with simulating(mockito(returns("Summarize", {"ok": 1}, kwargs=merged))):
        served = await serve([tool], tracer=spans.tracer, attachments=[INVOICE])
    result = await served.call(
        "Summarize", {"invoice": {"ID": UNKNOWN_ID}, "question": "q"}
    )
    assert result["isError"] is False, _result_text(result)
    assert served.probes == [] and served.host.held == {}
    outer = _by_name(spans)["Tool call - Summarize"]
    assert json.loads(outer.attributes["input"]) == merged
    # The hidden static attachment is listed with the call's.
    listed = json.loads(outer.attributes["attachments"])
    assert [a["id"] for a in listed] == [INVOICE_ID]


async def test_mocked_escalation_sees_the_input_attachment(serve):  # noqa: F811
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        input_schema={
            "type": "object",
            "properties": {
                "Amount": {"type": "number"},
                "Receipt": {"$ref": "#/definitions/job-attachment"},
            },
            "required": ["Amount", "Receipt"],
        },
        args_from_input={"Receipt": "receipt"},
    )
    agent_input = {"receipt": INVOICE}
    expected = {"Amount": 120, "Receipt": INVOICE}
    answer = {"action": "approve", "data": {}}
    with simulating(mockito(returns(tool.name.lower(), answer, kwargs=expected))):
        served = await serve([tool], attachments=[INVOICE], agent_input=agent_input)
    args = {"Amount": 120, "Receipt": {"ID": UNKNOWN_ID}}
    result = await served.call(tool.name, args)
    assert result["isError"] is False, _result_text(result)
    # A real start's task data is the same merge.
    assert tool.build_data(args, agent_input) == expected
    # No agent-input value: the model's unknown ID is refused.
    with simulating(mockito(returns(tool.name.lower(), answer))):
        served = await serve([tool], attachments=[INVOICE], agent_input={})
    result = await served.call(tool.name, args)
    assert result["isError"] is True
    assert _result_text(result) == not_found_message(UNKNOWN_ID)


@pytest.mark.parametrize(
    ("resource", "tool_name", "mock_name", "doc", "answer"),
    [
        (
            DEEP_RAG_RESOURCE,
            "Deep_RAG_v2",
            "Deep RAG (v2)",
            DOC,
            {"text": "Due in May.", "citations": [], "deepRagId": "d-1"},
        ),
        (
            BATCH_RESOURCE,
            "Batch_Transform",
            "Batch Transform",
            CSV,
            {"result": {"ID": RESULT_ID, "FullName": "o.csv", "MimeType": "text/csv"}},
        ),
    ],
)
async def test_mocked_attachment_tools_input_wins_over_an_unknown_model_id(
    serve,  # noqa: F811
    resource,
    tool_name,
    mock_name,
    doc,
    answer,
):
    builder = (
        deep_rag_tool_from_resource
        if resource is DEEP_RAG_RESOURCE
        else batch_transform_tool_from_resource
    )
    tool = builder(_with_attachment_argument(resource, "attachment"))
    merged = {"query": "When?", "attachment": doc}
    args = {"query": "When?", "attachment": {"ID": UNKNOWN_ID}}
    with simulating(mockito(returns(mock_name, answer, kwargs=merged))):
        served = await serve([tool], attachments=[doc], agent_input={"document": doc})
    result = await served.call(tool_name, args)
    assert result["isError"] is False, _result_text(result)
    assert json.loads(_result_text(result)) == answer
    assert with_fixed_args(args, tool.fixed_args, {"document": doc}) == merged
    with simulating(mockito(returns(mock_name, answer, kwargs=merged))):
        served = await serve([tool], attachments=[doc], agent_input={})
    result = await served.call(tool_name, args)
    assert result["isError"] is True
    assert _result_text(result) == not_found_message(UNKNOWN_ID)
