"""process_tool_from_resource and the pre-filled arguments (argumentProperties)."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    DurableToolContext,
    UiPathDurableTool,
    UiPathDurableToolError,
    process_tool_from_resource,
)
from uipath_codex_sdk._fixed_args import (
    FixedArg,
    dedupe_parents,
    fixed_args_schema,
    input_name,
    with_fixed_args,
)

from .test_tools import _Client, _job

AGENT_INPUT = {"customer": {"id": "C1", "name": "Ada"}, "region": "EU"}

#: The models report's example: one property of each supported variant.
INVOICE: dict[str, Any] = {
    "$resourceType": "tool",
    "type": "process",
    "name": "Create Invoice",
    "description": "Creates an invoice in the ERP.",
    "isEnabled": True,
    "inputSchema": {
        "type": "object",
        "properties": {
            "customerId": {"type": "string"},
            "apiKey": {"type": "string"},
            "memo": {"type": "string"},
            "currency": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "amount": {"type": "number"},
        },
        "required": ["customerId", "amount", "currency", "apiKey"],
    },
    "outputSchema": {"type": "object", "properties": {"invoiceId": {"type": "string"}}},
    "properties": {
        "processName": "CreateInvoice",
        "folderPath": "Shared/Finance",
        "exampleCalls": [],
    },
    "settings": {"timeout": 600, "maxAttempts": 1, "retryDelay": 0},
    "arguments": {},
    "argumentProperties": {
        "$['currency']": {"variant": "static", "value": "EUR", "isSensitive": False},
        "$['apiKey']": {"variant": "static", "value": "s3cr3t", "isSensitive": True},
        "$['customerId']": {
            "variant": "argument",
            "argumentPath": "$.customer.id",
            "isSensitive": False,
        },
        "$['memo']": {
            "variant": "textBuilder",
            "isSensitive": False,
            "tokens": [
                {"type": "simpleText", "rawString": "Invoice for "},
                {"type": "variable", "rawString": "input.customer.name"},
            ],
        },
        "$['tags']": {"variant": "arrayBuilder"},
        "$['tags'][0]": {"variant": "static", "value": "agent", "isSensitive": False},
        "$['tags'][1]": {
            "variant": "argument",
            "argumentPath": "$.region",
            "isSensitive": False,
        },
    },
}


def _resource(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(INVOICE)
    data.update(changes)
    return data


def _static(value: Any) -> dict[str, Any]:
    return {"variant": "static", "value": value, "isSensitive": False}


def _argument(path: str) -> dict[str, Any]:
    return {"variant": "argument", "argumentPath": path, "isSensitive": False}


@pytest.fixture
def client(monkeypatch):
    holder: dict[str, _Client] = {"c": _Client(job=_job("Pending", FolderKey="fk"))}
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: holder["c"])
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Env/Folder")
    return holder


def _ctx(agent_input: dict[str, Any] | None = None) -> DurableToolContext:
    return DurableToolContext(
        call_id="call_1", runtime_id="rt", tool_name="t", agent_input=agent_input
    )


# --------------------------------------------------------------------------
# the resource
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("type_", "kind", "tool_type", "child"),
    [
        ("process", "process", "Process", True),
        ("agent", "agent", "Agent", True),
        ("api", "process", "Api", True),
        ("processorchestration", "process", "agenticProcess", True),
        ("flow", "process", "Integration", False),  # low-code's fallback
        ("function", "process", "Function", True),
    ],
)
def test_from_resource_shape_per_type(type_, kind, tool_type, child):
    resource = _resource(
        type=type_,
        name="Basic RPA Process",
        argumentProperties={},
        outputSchema={"type": "object", "properties": {}},
    )
    resource["properties"]["exampleCalls"] = [
        {"id": "e1", "input": "{}", "output": "{}"}
    ]
    tool = process_tool_from_resource(resource)
    assert isinstance(tool, UiPathDurableTool)
    assert tool.name == "Basic_RPA_Process"
    assert tool.process_name == "CreateInvoice"
    assert (tool.kind, tool.tool_type) == (kind, tool_type)
    assert tool_type == tools_mod.tool_type_of(tool)
    assert tool.mock_child_span is child
    assert tool.folder_path is None  # the run's folder, as in low-code
    assert tool.mock_name == "Basic RPA Process"
    assert tool.mock_spec.name == "Basic RPA Process"
    assert tool.display_name == "CreateInvoice"
    assert len(tool.example_calls) == 1
    assert tool.description == "Creates an invoice in the ERP."
    assert tool.output_json_schema is None  # EMPTY_SCHEMA: any object
    assert tool.fixed_args == {}
    assert tool.input_schema == INVOICE["inputSchema"]


def test_from_resource_description_and_overrides():
    tool = process_tool_from_resource(
        _resource(description=""),
        tool_name="invoice!",
        run_as_me=True,
        folder_path="Finance/Override",
    )
    assert tool.name == "invoice_"
    assert tool.run_as_me is True
    assert tool.folder_path == "Finance/Override"
    assert tool.description.startswith(
        "Runs the UiPath process 'CreateInvoice' and returns its output."
    )
    assert '"invoiceId"' in tool.description


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"isEnabled": False}, "is disabled"),
        ({"properties": {"folderPath": "x"}}, "has no 'processName'"),
        ({"type": "integration"}, "Not a valid process tool resource"),
        (
            {"argumentProperties": {"x": {"variant": "dynamic", "isSensitive": False}}},
            "Not a valid process tool resource",
        ),
        (
            {
                "argumentProperties": {
                    "$['address']": {"variant": "objectBuilder"},
                    "$['address']['city']": _argument("$.city"),
                }
            },
            "argument property type is unsupported",
        ),
    ],
)
def test_from_resource_refusals(changes, message):
    with pytest.raises(ValueError, match=message) as info:
        process_tool_from_resource(_resource(**changes))
    if "objectBuilder" in str(changes):
        assert "$['address']" in str(info.value)


# --------------------------------------------------------------------------
# argumentProperties
# --------------------------------------------------------------------------
async def test_argument_properties_each_variant(client):
    tool = process_tool_from_resource(INVOICE)
    assert set(tool.fixed_args) == {"currency", "apiKey", "customerId", "memo", "tags"}
    schema = tool.input_schema
    # static, text-builder and array-builder fields are hidden ...
    assert set(schema["properties"]) == {"customerId", "amount"}
    # ... the agent-input one is visible but optional, with a note
    assert schema["required"] == ["amount"]
    assert (
        "agent input 'customer.id'" in schema["properties"]["customerId"]["description"]
    )
    assert schema["properties"]["amount"] == {"type": "number"}
    assert tool.input_json_schema == INVOICE["inputSchema"]  # untouched
    assert tool.mock_spec.input_schema == schema

    args = tool.validate_args({"amount": 10})
    await tool.start(args, _ctx(AGENT_INPUT))
    [call] = client["c"].calls
    assert call["input_arguments"] == {
        "amount": 10,
        "currency": "EUR",
        "apiKey": "s3cr3t",
        "customerId": "C1",
        "memo": "Invoice for Ada",
        "tags": ["agent", "EU"],
    }
    assert call["folder_path"] == "Env/Folder"  # not the resource's folderPath
    assert args == {"amount": 10}  # kept as is for finish / resume


async def test_unresolved_argument_keeps_model_value(client):
    tool = process_tool_from_resource(INVOICE)
    await tool.start({"amount": 1, "customerId": "from-model"}, _ctx({}))
    sent = client["c"].calls[0]["input_arguments"]
    assert sent["customerId"] == "from-model"
    # an unresolved text variable stays raw, an unresolved array item is null
    assert sent["memo"] == "Invoice for input.customer.name"
    assert sent["tags"] == ["agent", None]
    # ... and the agent input's value wins when it has one
    await tool.start({"amount": 1, "customerId": "from-model"}, _ctx(AGENT_INPUT))
    assert client["c"].calls[1]["input_arguments"]["customerId"] == "C1"


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("$['a']['b']", "a.b"),
        ("$['d'][0]", "d[0]"),
        ("$['x'][*]['y']", "x[*].y"),
        ('$["q"]', "q"),
        ("$.a.b", "a.b"),
        ("$.a", "a"),
        ("a.b", "a.b"),
        ("items[*]", "items[*]"),
        ("$filter", "$filter"),  # a field name, not the JSONPath root
        ("$['a.b']", None),  # a dotted property has no Codex name
        ("$..a", None),
        ("$['a'", None),
        ("", None),
    ],
)
def test_input_name(key, name):
    assert input_name(key) == name


async def test_jsonpath_keys_normalized(client):
    resource = _resource(
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "object", "properties": {"b": {"type": "string"}}},
                "d": {"type": "array", "items": {"type": "string"}},
                "x": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"y": {"type": "integer"}},
                    },
                },
                "top": {"type": "string"},
            },
        },
        argumentProperties={
            "$['a']['b']": _static("ab"),
            "$['d'][0]": _static("d0"),
            "$['x'][*]['y']": _static(7),
            "$.top": _argument("$['customer']['name']"),
        },
    )
    tool = process_tool_from_resource(resource)
    assert set(tool.fixed_args) == {"a.b", "d[0]", "x[*].y", "top"}
    schema = tool.input_schema
    assert schema["properties"]["a"]["properties"] == {}
    assert "y" not in schema["properties"]["x"]["items"]["properties"]
    assert "d" in schema["properties"]  # d[0] stays visible, still merged
    await tool.start({"x": [{"y": 1}, {}], "d": ["m", "n"]}, _ctx(AGENT_INPUT))
    assert client["c"].calls[0]["input_arguments"] == {
        "a": {"b": "ab"},
        "d": ["d0", "n"],
        "x": [{"y": 7}, {"y": 7}],
        "top": "Ada",
    }


def test_parent_binding_wins_segment_wise(caplog):
    assert dedupe_parents(["user", "user.name", "users", "users[0]", "u"]) == {
        "user",
        "users",
        "u",
    }
    resource = _resource(
        inputSchema={
            "type": "object",
            "properties": {
                "user": {"type": "object", "properties": {"name": {"type": "string"}}},
                "users": {"type": "array", "items": {"type": "string"}},
            },
        },
        argumentProperties={
            "$['user']": _static({"name": "alice"}),
            "$['user']['name']": _static("bob"),
            "$['users']": _static(["x"]),
        },
    )
    tool = process_tool_from_resource(resource)
    assert set(tool.fixed_args) == {"user", "users"}
    assert tool.fixed_args["user"].value == {"name": "alice"}


def test_array_builder():
    props = {
        "$['files']": {"variant": "arrayBuilder"},
        "$['files'][0]": _static("first"),
        "$['files'][2]": _argument("$.region"),
        "$['files'][3]": {
            "variant": "textBuilder",
            "isSensitive": True,
            "tokens": [
                {"type": "simpleText", "rawString": "Hello, "},
                {"type": "variable", "rawString": "input.customer.name"},
            ],
        },
        "$['files'][4]": _argument("$.missing"),
        "$['names']": {"variant": "arrayBuilder"},
        "$['names'][0]['name']": _static("ignored"),  # not a direct item
        "$['names'][1]": _static("kept"),
        "$['empty']": {"variant": "arrayBuilder"},
    }
    tool = process_tool_from_resource(
        _resource(
            inputSchema={"type": "object", "properties": {}},  # permissive
            argumentProperties=props,
        )
    )
    assert set(tool.fixed_args) == {"files", "names", "empty"}
    merged = with_fixed_args({}, tool.fixed_args, AGENT_INPUT)
    assert merged == {
        "files": ["first", None, "EU", "Hello, Ada", None],
        "names": [None, "kept"],
        "empty": [],
    }


async def test_star_on_missing_array_becomes_list(client):
    resource = _resource(
        inputSchema={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "q": {"type": "string"},
            },
            "required": ["items", "q"],
        },
        argumentProperties={"$['items'][*]": _static("v")},
    )
    tool = process_tool_from_resource(resource)
    schema = tool.input_schema
    assert "items" not in schema["properties"] and schema["required"] == ["q"]
    await tool.start({"q": "x"}, _ctx())
    assert client["c"].calls[0]["input_arguments"] == {"q": "x", "items": ["v"]}
    assert with_fixed_args({"items": ["a", "b"]}, tool.fixed_args, None) == {
        "items": ["v", "v"]
    }


def test_unreachable_path_dropped_with_warning(caplog):
    props = {"$['nope']": _static(1), "$['currency']": _static("EUR")}
    with caplog.at_level(logging.WARNING):
        tool = process_tool_from_resource(_resource(argumentProperties=props))
    assert set(tool.fixed_args) == {"currency"}
    assert "'nope' is not in the tool's input schema" in caplog.text
    # a permissive schema keeps every path, as langchain without args_schema
    permissive = process_tool_from_resource(
        _resource(inputSchema={"type": "object"}, argumentProperties=props)
    )
    assert set(permissive.fixed_args) == {"nope", "currency"}


def test_unsupported_keys_are_left_to_the_model(caplog):
    props = {"$['a.b']": _static(1), "$['currency']": _argument("$..deep")}
    with caplog.at_level(logging.WARNING):
        tool = process_tool_from_resource(_resource(argumentProperties=props))
    assert tool.fixed_args == {}
    assert "is not a supported path" in caplog.text
    assert "not a supported agent input path" in caplog.text


async def test_fixed_attachment_from_agent_input_is_linked(client):
    attachment = {
        "ID": "8e7a7b3c-5f6a-4a43-9f8c-3b8e1f2a9d10",
        "FullName": "invoice.pdf",
        "MimeType": "application/pdf",
    }
    resource = _resource(
        inputSchema={
            "type": "object",
            "properties": {
                "document": {"$ref": "#/definitions/job-attachment"},
                "note": {"type": "string"},
            },
            "required": ["document"],
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "x-uipath-resource-kind": "JobAttachment",
                    "properties": {"ID": {"type": "string"}},
                }
            },
        },
        argumentProperties={"$['document']": _argument("$.invoice")},
    )
    tool = process_tool_from_resource(resource)
    await tool.start({"note": "n"}, _ctx({"invoice": attachment}))
    [call] = client["c"].calls
    assert call["input_arguments"]["document"] == attachment
    [linked] = call["attachments"]
    assert str(linked.id) == attachment["ID"]


def test_span_and_mock_args_show_the_merged_input():
    tool = process_tool_from_resource(INVOICE)
    ctx = _ctx(AGENT_INPUT)
    merged = tool.span_args({"amount": 10}, ctx)
    assert merged["apiKey"] == "s3cr3t"  # clear text, as in low-code
    assert tool.mock_args({"amount": 10}, ctx) == merged


def test_fixed_args_schema_hides_star_arrays_and_notes_inputs():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {"type": "string"}},
            "b": {"type": "string", "description": "B."},
        },
        "required": ["a", "b"],
    }
    out = fixed_args_schema(
        schema, {"a[*]": FixedArg(value="v"), "b": FixedArg(path="x")}
    )
    assert list(out["properties"]) == ["b"]
    assert out["required"] == []
    assert out["properties"]["b"]["description"].startswith("B. (Set from the agent")
    assert fixed_args_schema(schema, {}) is schema


def test_invalid_merged_input_is_a_tool_error():
    from pydantic import BaseModel

    from uipath_codex_sdk import uipath_process_tool

    class In(BaseModel):
        n: int
        tag: str

    tool = uipath_process_tool("p", input_schema=In, static_args={"tag": "x"})
    with pytest.raises(UiPathDurableToolError, match="Invalid arguments for tool 'p'"):
        tool.merged_args({"n": "not a number"}, None)


def test_from_resource_refuses_another_parsed_tool_type():
    from pydantic import TypeAdapter
    from uipath.agent.models.agent import AgentResourceConfig

    from .test_context_tools import POLICY_RESOURCE

    resource = TypeAdapter(AgentResourceConfig).validate_python(POLICY_RESOURCE)
    with pytest.raises(ValueError, match="Not a process tool resource"):
        process_tool_from_resource(resource)  # type: ignore[arg-type]  # deliberately wrong type


@pytest.mark.parametrize(
    ("props", "kept"),
    [
        # a static parent covers it (langchain's parent-wins dedup skips it)
        (
            {
                "$['a']": _static({"b": "1"}),
                "$['a']['b']": {"variant": "objectBuilder"},
            },
            "a",
        ),
        # deeper than a direct item of an array builder
        (
            {
                "$['x']": {"variant": "arrayBuilder"},
                "$['x'][0]": _static({"y": 1}),
                "$['x'][0]['y']": {"variant": "objectBuilder"},
            },
            "x",
        ),
    ],
)
def test_covered_object_builder_is_skipped(props, kept):
    tool = process_tool_from_resource(
        _resource(inputSchema={"type": "object"}, argumentProperties=props)
    )
    assert set(tool.fixed_args) == {kept}


@pytest.mark.parametrize(
    "props",
    [
        {
            "$['x']": {"variant": "arrayBuilder"},
            "$['x'][0]": {"variant": "objectBuilder"},
        },
        {
            "$['x']": {"variant": "arrayBuilder"},
            "$['x'][1]": {"variant": "arrayBuilder"},
            "$['x'][1][0]": {"variant": "objectBuilder"},
        },
    ],
)
def test_object_builder_array_item_is_refused(props):
    # langchain resolves an array builder's direct items, an object builder
    # among them (also in a nested array builder), and fails on it.
    with pytest.raises(ValueError, match="argument property type is unsupported"):
        process_tool_from_resource(
            _resource(inputSchema={"type": "object"}, argumentProperties=props)
        )


def test_unresolved_required_argument_is_left_to_the_model():
    tool = process_tool_from_resource(INVOICE)
    assert "customerId" not in tool.input_schema["required"]
    with pytest.raises(UiPathDurableToolError, match="required argument.*customerId"):
        tool.merged_args(tool.validate_args({"amount": 1}), {})
    merged = tool.merged_args({"amount": 1, "customerId": "M"}, {})
    assert merged["customerId"] == "M"


def test_span_args_are_the_validated_job_input():
    from pydantic import BaseModel

    from uipath_codex_sdk import uipath_process_tool

    class In(BaseModel):
        amount: float
        currency: str
        items: list[str] = []

    tool = uipath_process_tool("p", input_schema=In, static_args={"currency": "EUR"})
    ctx = _ctx()
    args = tool.validate_args({"amount": "2"})
    expected = {"amount": 2.0, "currency": "EUR", "items": []}
    assert tool.merged_args(args, None) == expected
    assert tool.span_args(args, ctx) == expected
    # an input that does not validate shows the unvalidated merge
    assert tool.span_args({"amount": "x"}, ctx) == {"amount": "x", "currency": "EUR"}
