"""Action Center escalation tools: schemas, task creation, results, resources.

No codex here: the tool is called directly, against a fake ``tasks`` service
(the end-to-end suspend/resume runs are in ``tests/durable``).
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
    DurableToolContext,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    UiPathEscalationTool,
    escalation_tool_from_resource,
    uipath_escalation_tool,
)
from uipath_codex_sdk.escalation_tools import (
    _task_from,
    filter_task_data,
    resolve_recipient_config,
    task_url,
)
from uipath_codex_sdk.runtime.errors import UiPathRunTerminatedError
from uipath_codex_sdk.tools import is_durable_tool, tool_type_of

from .durable.fakes import FakeTasks

CTX = DurableToolContext(call_id="c1", runtime_id="rt", tool_name="escalate_x")

#: The low-code HITL example's escalation resource (uipath-agents-python
#: examples/HITL_1), trimmed of UI-only fields.
HITL_RESOURCE: dict[str, Any] = {
    "$resourceType": "escalation",
    "name": "color",
    "description": "Ask color confirmation from the user",
    "channels": [
        {
            "name": "Channel",
            "description": "Channel description",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "Content": {"type": "string"},
                    "Comment": {"type": "string"},
                },
            },
            "outputSchema": {
                "type": "object",
                "properties": {"Comment": {"type": "string"}},
            },
            "outcomeMapping": {"approve": "continue", "reject": "end"},
            "recipients": [{"type": 1, "value": "user-1", "displayName": "Ana"}],
            "type": "actionCenter",
            "properties": {
                "appName": "HITL App",
                "appVersion": 1,
                "folderName": "Shared/HITL",
                "isActionableMessageEnabled": False,
            },
        }
    ],
    "isAgentMemoryEnabled": False,
    "escalationType": 0,
}


def _resource(**channel: Any) -> dict[str, Any]:
    r = json.loads(json.dumps(HITL_RESOURCE))
    r["channels"][0].update(channel)
    return r


@pytest.fixture
def fake(monkeypatch) -> FakeTasks:
    tasks = FakeTasks()
    client = SimpleNamespace(tasks=tasks, assets=None)
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    monkeypatch.delenv("UIPATH_FOLDER_PATH", raising=False)
    return tasks


class Review(BaseModel):
    amount: float
    reason: str


@pytest.fixture
def tracer(monkeypatch):
    """A recording tracer for the tool's own spans."""
    from opentelemetry.sdk.trace import TracerProvider

    t = TracerProvider().get_tracer("test")
    monkeypatch.setattr("uipath_codex_sdk.runtime.telemetry.get_tracer", lambda: t)
    return t


def _hold_spans():
    """Collect held spans as the tool host does around ``start``."""
    from uipath_codex_sdk.runtime.telemetry import collect_held_spans

    return collect_held_spans()


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------
def test_tool_shape():
    tool = uipath_escalation_tool(
        "Expense Approval",
        input_schema=Review,
        static_args={"reason": "fixed"},
        outcome_mapping={"Reject": "end"},
    )
    assert isinstance(tool, UiPathEscalationTool) and is_durable_tool(tool)
    assert tool.name == "escalate_Expense_Approval"
    assert tool.display_name == "Expense Approval"
    assert tool_type_of(tool, tool.kind) == "Escalation"
    # Pre-filled fields are hidden from the model.
    assert list(tool.input_schema["properties"]) == ["amount"]
    assert tool.input_schema["required"] == ["amount"]
    assert "Action Center" in tool.description
    assert tool.validate_args({"amount": 3}) == {"amount": 3}
    with pytest.raises(ValueError):
        tool.validate_args({})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"folder_path": "a", "folder_key": "k"}, "folder_path or folder_key"),
        ({"title": "t", "title_from_input": "x"}, "title or title_from_input"),
        ({"outcome_mapping": {"Reject": "stop"}}, "use 'continue' or 'end'"),
        ({"assignee": "a@b.c", "recipient": object()}, "recipient or assignee"),
    ],
)
def test_invalid_configurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        uipath_escalation_tool("App", **kwargs)


async def test_start_creates_then_assigns_and_returns_wait_escalation(fake, tracer):
    from uipath.platform.common import WaitEscalation

    tool = uipath_escalation_tool(
        "ExpenseApproval",
        folder_path="Shared",
        input_schema=Review,
        title_from_input="request.title",
        assignee="manager@example.com",
        priority="High",
        labels=["finance"],
        args_from_input={"reason": "{{request.reason}}"},
    )
    ctx = DurableToolContext(
        "c1",
        "rt",
        tool.name,
        agent_input={"request": {"title": "Offsite", "reason": "Team event"}},
    )
    token = _hold_spans()
    value = await tool.start({"amount": 1250}, ctx)
    from uipath_codex_sdk.runtime.telemetry import held_spans

    [child] = held_spans(token)
    assert isinstance(value, WaitEscalation)
    [created] = fake.created
    assert created["title"] == "Offsite"
    assert created["data"] == {"amount": 1250.0, "reason": "Team event"}
    assert (created["app_folder_path"], created["app_folder_key"]) == ("Shared", None)
    assert (created["priority"], created["labels"]) == ("High", ["finance"])
    [assignment] = fake.assignments
    assert assignment["json"]["taskAssignments"][0]["UserNameOrEmail"] == (
        "manager@example.com"
    )
    assert value.action.key == created["key"]
    assert (value.app_name, value.app_folder_path, value.app_folder_key) == (
        "ExpenseApproval",
        "Shared",
        None,
    )
    assert child["output_attribute"] == "result"
    assert child["output_keys"] == ["outcome", "output"]
    assert child["attributes"]["assignedTo"] == "manager@example.com"
    assert tool.started_output(value)["task_key"] == created["key"]


async def test_folder_key_is_sent_alone_unless_a_binding_moves_the_app(fake):
    from uipath.platform.common import ResourceOverwritesContext
    from uipath.platform.common._bindings import GenericResourceOverwrite

    tool = uipath_escalation_tool("App", folder_key="fk-1")
    value = await tool.start({}, CTX)
    assert (fake.created[0]["app_folder_path"], fake.created[0]["app_folder_key"]) == (
        None,
        "fk-1",
    )
    assert (value.app_folder_path, value.app_folder_key) == (None, "fk-1")

    tool2 = uipath_escalation_tool("App", folder_path="Shared")

    async def overwrites():
        return {
            "app.App.Shared": GenericResourceOverwrite.model_validate(
                {"resource_type": "app", "name": "ProdApp", "folderPath": "Prod"}
            )
        }

    async with ResourceOverwritesContext(overwrites):
        value2 = await tool2.start({}, CTX)
    created = fake.created[1]
    assert (created["app_name"], created["app_folder_path"]) == ("ProdApp", "Prod")
    assert (value2.app_name, value2.app_folder_path) == ("ProdApp", "Prod")


async def test_create_failure_is_a_durable_tool_error(fake):
    fake.fail_create = Exception("App 'X' was not found")
    tool = uipath_escalation_tool("X")
    with pytest.raises(UiPathDurableToolError, match="App 'X' was not found"):
        await tool.start({}, CTX)


async def test_assignment_failure_keeps_the_task(fake, tracer):
    fake.fail_assign = RuntimeError("no such user")
    tool = uipath_escalation_tool("App", assignee="ghost@example.com")
    token = _hold_spans()
    value = await tool.start({}, CTX)
    from uipath_codex_sdk.runtime.telemetry import held_spans

    [child] = held_spans(token)
    assert value.action.key == fake.created[0]["key"]
    assert child["attributes"]["assignmentError"] == "no such user"


async def test_finish_maps_every_outcome():
    tool = uipath_escalation_tool(
        "App",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        outcome_mapping={"Reject": "end"},
    )
    task = {
        "key": "k",
        "id": 7,
        "status": 2,
        "action": "Approve",
        "data": {"q": "in", "Answer": "yes"},
        "completedByUser": {"emailAddress": "r@example.com"},
        "isDeleted": False,
    }
    assert json.loads(await tool.finish({}, task)) == {
        "status": "completed",
        "outcome": "Approve",
        "output": {"Answer": "yes"},  # no output schema: the non-input fields
        "task_id": 7,
        "assigned_to": "r@example.com",
    }
    with pytest.raises(UiPathRunTerminatedError) as info:
        await tool.finish({}, {**task, "action": "Reject"})
    assert json.loads(info.value.result_text)["outcome"] == "Reject"
    assert "directive end" in str(info.value.error_info.title)
    with pytest.raises(UiPathRunTerminatedError, match="deleted"):
        await tool.finish({}, {**task, "isDeleted": True})
    for status in (0, 1, "Pending"):
        with pytest.raises(UiPathDurableToolPending):
            await tool.finish({}, {**task, "status": status})
    # Typed answers (uipath run --resume '{...}').
    typed = json.loads(await tool.finish({}, {"action": "Approve", "data": {"x": 1}}))
    assert (typed["outcome"], typed["output"]) == ("Approve", {"x": 1})
    form = json.loads(await tool.finish({}, {"Answer": "no", "status": "closed"}))
    assert (form["outcome"], form["output"]) == (
        None,
        {"Answer": "no", "status": "closed"},
    )


def test_task_detection_is_strict():
    assert _task_from({"key": "k", "status": 2}).key == "k"
    assert _task_from({"isDeleted": True}).is_deleted is True
    # A form that happens to have a "status" or "key" field is form data.
    assert _task_from({"status": "ok"}).data == {"status": "ok"}
    assert _task_from({"key": "k"}).data == {"key": "k"}
    assert _task_from("yes").data == {"value": "yes"}


def test_filter_task_data_matches_langchain():
    data = {"in": 1, "out": 2, "extra": 3}
    inputs: dict[str, Any] = {"properties": {"in": {}}}
    assert filter_task_data(data, inputs, None) == {"out": 2, "extra": 3}
    assert filter_task_data(data, inputs, {"properties": {"out": {}}}) == {"out": 2}


def test_task_url(monkeypatch):
    monkeypatch.delenv("UIPATH_URL", raising=False)
    assert task_url(5) is None
    monkeypatch.setenv("UIPATH_URL", "https://cloud.test/o/t/")
    assert task_url(5) == "https://cloud.test/o/t/actions_/tasks/5"


# --------------------------------------------------------------------------
# agent.json resources
# --------------------------------------------------------------------------
def test_from_resource_maps_the_channel():
    tool = escalation_tool_from_resource(HITL_RESOURCE)
    assert tool.name == "escalate_color"
    assert (tool.app_name, tool.folder_path) == ("HITL App", "Shared/HITL")
    assert tool.display_name == "HITL App"
    assert tool.description.startswith("Ask color confirmation from the user")
    assert tool.outcome_mapping == {"approve": "continue", "reject": "end"}
    assert tool.title == "Escalation Task"
    assert tool.output_json_schema == {
        "type": "object",
        "properties": {"Comment": {"type": "string"}},
    }
    recipient = asyncio.run(tool.resolve_recipient({}))
    assert recipient is not None
    assert (recipient.type.value, recipient.value, recipient.display_name) == (
        "UserId",
        "user-1",
        "Ana",
    )


def test_from_resource_title_and_argument_properties():
    tokens = [
        {"type": "simpleText", "rawString": "Review "},
        {"type": "variable", "rawString": "input.topic"},
    ]
    tool = escalation_tool_from_resource(
        _resource(
            taskTitleV2={"type": "textBuilder", "tokens": tokens},
            argumentProperties={
                "Content": {
                    "variant": "argument",
                    "argumentPath": "$.topic",
                    "isSensitive": False,
                },
                "Comment": {
                    "variant": "textBuilder",
                    "isSensitive": False,
                    "tokens": [{"type": "simpleText", "rawString": "auto"}],
                },
            },
        )
    )
    # The text builder is hidden; the argument stays visible but optional.
    assert list(tool.input_schema["properties"]) == ["Content"]
    assert (
        "Set from the agent input 'topic'"
        in (tool.input_schema["properties"]["Content"]["description"])
    )
    assert tool.resolve_title({"topic": "colors"}) == "Review colors"
    data = tool.build_data({}, {"topic": "colors"})
    assert data == {"Content": "colors", "Comment": "auto"}
    dynamic = escalation_tool_from_resource(
        _resource(taskTitleV2={"type": "dynamic", "argumentPath": "subject"})
    )
    assert dynamic.resolve_title({"subject": "From input"}) == "From input"


def test_from_resource_jsonpath_keys_and_array_builder():
    # LangChain's JSONPath keys are converted (``$['Content']`` → ``Content``),
    # and an array builder collects its items.
    tool = escalation_tool_from_resource(
        _resource(
            argumentProperties={
                "$['Content']": {
                    "variant": "argument",
                    "argumentPath": "$['topic']",
                    "isSensitive": False,
                },
                "$['Tags']": {"variant": "arrayBuilder"},
                "$['Tags'][0]": {
                    "variant": "static",
                    "value": "agent",
                    "isSensitive": False,
                },
            },
        )
    )
    assert set(tool.fixed_args) == {"Content", "Tags"}
    assert "Content" in tool.input_schema["properties"]  # an argument: visible
    assert "Tags" not in tool.input_schema["properties"]
    data = tool.build_data({"Comment": "c"}, {"topic": "colors"})
    assert data == {"Comment": "c", "Content": "colors", "Tags": ["agent"]}
    ctx = DurableToolContext(
        call_id="c1", runtime_id="rt", tool_name="t", agent_input={"topic": "colors"}
    )
    assert tool.span_args({"Comment": "c"}, ctx) == data  # the Tool call's input


def _argument_resource() -> dict[str, Any]:
    """HITL with a required ``Content`` read from the agent input's ``topic``."""
    resource = _resource(
        argumentProperties={
            "Content": {
                "variant": "argument",
                "argumentPath": "$.topic",
                "isSensitive": False,
            },
            "Comment": {"variant": "static", "value": "auto", "isSensitive": False},
        }
    )
    resource["channels"][0]["inputSchema"]["required"] = ["Content", "Comment"]
    return resource


def test_argument_property_is_optional_for_the_model():
    tool = escalation_tool_from_resource(_argument_resource())
    schema = tool.input_schema
    assert list(schema["properties"]) == ["Content"]  # the static one is hidden
    assert schema["required"] == []
    assert schema["properties"]["Content"]["description"] == (
        "Set from the agent input 'topic' when it has a value; fill it only if "
        "it does not."
    )
    assert tool.validate_args({}) == {}
    # resolved: the agent input wins over the model
    assert tool.build_data({"Content": "model"}, {"topic": "colors"}) == {
        "Content": "colors",
        "Comment": "auto",
    }
    # unresolved: the model's value is kept (langchain's StaticArgsHandler)
    assert tool.build_data({"Content": "model"}, {}) == {
        "Content": "model",
        "Comment": "auto",
    }
    # unresolved and not filled: a tool error asking the model to fill it
    with pytest.raises(UiPathDurableToolError) as info:
        tool.build_data({}, {"other": 1})
    assert str(info.value) == (
        "Invalid arguments for tool 'escalate_color': missing required "
        "argument(s): Content. The agent input has no value for them; fill "
        "them in and call the tool again."
    )


async def test_unresolved_required_argument_creates_no_task(fake):
    tool = escalation_tool_from_resource(_argument_resource())
    ctx = DurableToolContext("c1", "rt", tool.name, {})
    with pytest.raises(UiPathDurableToolError, match="missing required argument"):
        await tool.start({}, ctx)
    assert fake.created == []
    await tool.start({"Content": "teal"}, ctx)
    [created] = fake.created
    assert created["data"] == {"Content": "teal", "Comment": "auto"}


def test_required_argument_under_every_array_element():
    # ``items[*].owner`` from the agent input: optional for the model, but
    # an element still without one once merged is a tool error (langchain
    # keeps it required in each item).
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "owner": {"type": "string"},
                        },
                        "required": ["sku", "owner"],
                    },
                }
            },
            "required": ["items"],
        },
        args_from_input={"items[*].owner": "user.name"},
    )
    item = tool.input_schema["properties"]["items"]["items"]
    assert item["required"] == ["sku"]
    assert (
        "Set from the agent input 'user.name'"
        in item["properties"]["owner"]["description"]
    )
    two = {"items": [{"sku": "a"}, {"sku": "b", "owner": "Bo"}]}
    assert tool.build_data(two, {"user": {"name": "Al"}}) == {
        "items": [{"sku": "a", "owner": "Al"}, {"sku": "b", "owner": "Al"}]
    }
    with pytest.raises(
        UiPathDurableToolError, match=r"argument\(s\): items\[\*\]\.owner\."
    ):
        tool.build_data(two, {})
    filled = {"items": [{"sku": "a", "owner": "Cy"}, {"sku": "b", "owner": "Bo"}]}
    assert tool.build_data(filled, {}) == filled
    assert tool.build_data({"items": []}, {}) == {"items": []}


def test_coded_args_from_input_with_a_pydantic_schema():
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        input_schema=Review,
        args_from_input={"reason": "{{request.reason}}"},
    )
    assert set(tool.input_schema["properties"]) == {"amount", "reason"}
    assert tool.input_schema["required"] == ["amount"]
    assert tool.build_data({"amount": 1, "reason": "model"}, {})["reason"] == "model"
    assert (
        tool.build_data({"amount": 1, "reason": "model"}, {"request": {"reason": "in"}})
    )["reason"] == "in"
    with pytest.raises(UiPathDurableToolError, match=r"argument\(s\): reason\."):
        tool.build_data({"amount": 1}, {})


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"isEnabled": False}, "is disabled"),
        ({"channels": []}, "has no channels"),
    ],
)
def test_from_resource_refusals(change, match):
    with pytest.raises(ValueError, match=match):
        escalation_tool_from_resource({**HITL_RESOURCE, **change})


def test_from_resource_refuses_quick_forms_and_missing_app():
    quick = json.loads(json.dumps(HITL_RESOURCE))
    quick["channels"][0]["type"] = "actionCenterQuickForm"
    quick["channels"][0]["properties"] = {"schema": {"schemaId": "s"}}
    with pytest.raises(ValueError, match="not supported yet"):
        escalation_tool_from_resource(quick)
    no_app = json.loads(json.dumps(HITL_RESOURCE))
    no_app["channels"][0]["properties"]["appName"] = None
    with pytest.raises(ValueError, match="has no 'appName'"):
        escalation_tool_from_resource(no_app)


async def test_recipient_kinds(monkeypatch):
    from uipath.agent.models.agent import AgentEscalationChannel

    async def asset(name: str, folder_path: str | None = None):
        return SimpleNamespace(value=f"{name}@example.com")

    client = SimpleNamespace(assets=SimpleNamespace(retrieve_async=asset))
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)

    def recipient(raw: dict[str, Any]):
        channel = AgentEscalationChannel.model_validate(
            {**HITL_RESOURCE["channels"][0], "recipients": [raw]}
        )
        return channel.recipients[0]

    cases = [
        ({"type": "UserEmail", "value": "a@b.c"}, ("UserEmail", "a@b.c")),
        ({"type": "GroupName", "value": "Finance"}, ("GroupName", "Finance")),
        (
            {"type": "AssetUserEmail", "assetName": "approver", "folderPath": "Shared"},
            ("UserEmail", "approver@example.com"),
        ),
        ({"type": "ArgumentEmail", "argumentName": "who"}, ("UserEmail", "x@y.z")),
        (
            {"type": "Workload", "value": "g1", "displayName": "Team A"},
            ("Workload", "Team A"),
        ),
    ]
    for raw, (kind, value) in cases:
        got = await resolve_recipient_config(recipient(raw), {"who": "x@y.z"})
        assert got is not None
        assert (got.type.value, got.value, got.display_name) == (kind, value, value)
    with pytest.raises(ValueError, match="no value for 'who'"):
        await resolve_recipient_config(
            recipient({"type": "ArgumentEmail", "argumentName": "who"}), {}
        )


# --------------------------------------------------------------------------
# runtime: suspend value round-trip and trigger rebuild
# --------------------------------------------------------------------------
async def test_wait_escalation_survives_the_store_and_rebuilds_without_side_effects(
    monkeypatch,
):
    from uipath.core.triggers import UiPathResumeTriggerName, UiPathResumeTriggerType
    from uipath.platform.action_center import Task
    from uipath.platform.common import WaitEscalation
    from uipath.platform.resume_triggers import UiPathResumeTriggerHandler

    from uipath_codex_sdk.runtime.runtime import _call_operation, _rebuilt_trigger
    from uipath_codex_sdk.runtime.session_store import (
        PendingCall,
        rebuild_suspend_value,
    )

    value = WaitEscalation(
        action=Task.model_validate({"id": 9, "key": "k9", "status": 0}),
        app_name="App",
        app_folder_path="Shared",
    )
    call = PendingCall(
        call_id="c", tool="t", args={}, value=value, value_type="WaitEscalation"
    )
    stored = PendingCall.from_stored(json.loads(json.dumps(call.to_stored())))
    assert stored is not None
    rebuilt = rebuild_suspend_value(stored.value_type, stored.value)
    assert isinstance(rebuilt, WaitEscalation) and rebuilt.action.key == "k9"
    assert _call_operation(stored) == {"task_id": "9", "task_key": "k9"}
    trigger = await _rebuilt_trigger(UiPathResumeTriggerHandler(), rebuilt)
    assert trigger is not None
    assert trigger.trigger_type == UiPathResumeTriggerType.TASK
    assert trigger.trigger_name == UiPathResumeTriggerName.ESCALATION
    assert (trigger.item_key, trigger.folder_path) == ("k9", "Shared")
