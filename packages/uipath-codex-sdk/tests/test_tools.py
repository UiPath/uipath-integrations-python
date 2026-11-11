"""uipath_process_tool / uipath_agent_tool: start, finish and validation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job

import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    DurableToolContext,
    UiPathDurableTool,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    uipath_agent_tool,
    uipath_process_tool,
)
from uipath_codex_sdk.tools import sanitize_tool_name


class AddIn(BaseModel):
    a: int
    b: int


class AddOut(BaseModel):
    sum: int


def _job(state: str, **extra: Any) -> Job:
    return Job.model_validate({"Id": 7, "Key": "job-7", "State": state, **extra})


class _Client:
    def __init__(self, job: Job | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._job, self._error = job, error
        self.processes = SimpleNamespace(invoke_async=self._invoke)
        self.jobs = SimpleNamespace(extract_output_async=self._extract)

    async def _invoke(self, **kwargs: Any) -> Job | None:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._job

    async def _extract(self, job: Job) -> str | None:
        return job.output_arguments


@pytest.fixture
def client(monkeypatch):
    holder: dict[str, _Client] = {"c": _Client()}
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: holder["c"])
    return holder


CTX = DurableToolContext(call_id="call_1", runtime_id="rt", tool_name="a_plus_b")


def test_factories_and_protocol():
    tool = uipath_process_tool(
        "a plus b!", input_schema=AddIn, output_schema=AddOut, description="Adds."
    )
    assert isinstance(tool, UiPathDurableTool)
    assert tool.name == "a_plus_b_"
    assert tool.process_name == "a plus b!"
    assert tool.kind == "process"
    assert tool.input_schema["required"] == ["a", "b"]
    assert tool.description.startswith("Adds.")
    assert '"sum"' in tool.description  # output schema hint
    agent = uipath_agent_tool("Research_Agent", tool_name="research")
    assert agent.kind == "agent" and agent.name == "research"
    assert "agent 'Research_Agent'" in agent.description
    assert sanitize_tool_name("x" * 100) == "x" * 64


def test_validate_args_with_model_and_schema():
    tool = uipath_process_tool("p", input_schema=AddIn)
    assert tool.validate_args({"a": "1", "b": 2}) == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        tool.validate_args({"a": 1})
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    raw = uipath_process_tool("p", input_schema=schema)
    assert raw.validate_args({"q": "x", "n": 3}) == {"q": "x", "n": 3}
    for bad in ({"n": 3}, {"q": 1}, {"q": "x", "n": True}, {"q": "x", "z": 1}):
        with pytest.raises(ValueError):
            raw.validate_args(bad)
    assert uipath_process_tool("p").validate_args({"anything": 1}) == {"anything": 1}


async def test_start_returns_wait_job_raw_with_folder_key_only(client, monkeypatch):
    monkeypatch.delenv("UIPATH_FOLDER_PATH", raising=False)
    client["c"] = _Client(job=_job("Pending", FolderKey="fk-1"))
    tool = uipath_process_tool("a_plus_b", folder_path="Shared", run_as_me=True)
    value = await tool.start({"a": 1, "b": 2}, CTX)
    assert isinstance(value, WaitJobRaw)
    assert value.job.key == "job-7"
    # setting both folder fields breaks the job lookup on every resume
    assert value.process_folder_key == "fk-1" and value.process_folder_path is None
    [call] = client["c"].calls
    assert call == {
        "name": "a_plus_b",
        "input_arguments": {"a": 1, "b": 2},
        "folder_path": "Shared",
        "run_as_me": True,
        "parent_span_id": None,  # no current span
    }


async def test_start_parents_the_job_under_the_current_span(client):
    # The tool host's Tool call span, not invoke_async's own span.
    from opentelemetry.sdk.trace import TracerProvider

    client["c"] = _Client(job=_job("Pending", FolderKey="fk-1"))
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("Tool call - p") as span:
        await uipath_process_tool("p").start({}, CTX)
    [call] = client["c"].calls
    assert call["parent_span_id"] == format(span.get_span_context().span_id, "016x")


async def test_start_without_folder_key_falls_back_to_path(client, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Env/Folder")
    client["c"] = _Client(job=_job("Pending"))
    value = await uipath_process_tool("p").start({}, CTX)
    assert value.process_folder_key is None
    assert value.process_folder_path == "Env/Folder"
    assert client["c"].calls[0]["folder_path"] == "Env/Folder"
    assert client["c"].calls[0]["run_as_me"] is None


class _Enriched(Exception):
    def __init__(self, status: int, code: str | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status
        self.error_info = SimpleNamespace(error_code=code, message=message)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            _Enriched(404, "1002", "AssociatedProcessNotFound"),
            "Could not find process for tool 'p'",
        ),
        (_Enriched(400, "1100", "x"), "Could not find folder for tool 'p'"),
        (_Enriched(409, None, "busy"), "Cannot start process for tool 'p': busy"),
        (RuntimeError("boom"), "Failed to start the job for tool 'p': boom"),
    ],
)
async def test_start_errors_are_model_facing(client, error, expected):
    client["c"] = _Client(error=error)
    with pytest.raises(UiPathDurableToolError) as exc:
        await uipath_process_tool("p").start({}, CTX)
    assert expected in str(exc.value)


async def test_finish_successful(client):
    tool = uipath_process_tool("p")
    job = _job("Successful", OutputArguments=json.dumps({"sum": 3}))
    assert await tool.finish({}, job) == json.dumps({"sum": 3})
    empty = await tool.finish({}, _job("Successful"))
    assert json.loads(empty) == {"status": "successful", "job_key": "job-7"}
    # a serialized Job (e.g. from a stored payload) is recognised too
    as_dict = job.model_dump(by_alias=True)
    assert await tool.finish({}, as_dict) == json.dumps({"sum": 3})


async def test_finish_faulted_and_stopped(client):
    tool = uipath_process_tool("p")
    faulted = json.loads(await tool.finish({}, _job("Faulted", Info="Boom")))
    assert faulted == {"status": "faulted", "error": "Boom", "job_key": "job-7"}
    detailed = json.loads(
        await tool.finish({}, _job("Faulted", JobError={"Code": "E1", "Title": "Bad"}))
    )
    assert detailed["status"] == "faulted"
    assert detailed["error"] == "Bad"
    assert detailed["details"] == {"code": "E1", "title": "Bad"}
    unknown = json.loads(await tool.finish({}, _job("Faulted")))
    assert unknown["error"] == "Unknown error"
    stopped = json.loads(await tool.finish({}, _job("Stopped")))
    assert stopped == {"status": "stopped", "job_key": "job-7"}


@pytest.mark.parametrize("state", ["Running", "Pending", "Suspended", "Resumed"])
async def test_finish_not_final_is_pending(client, state):
    with pytest.raises(UiPathDurableToolPending):
        await uipath_process_tool("p").finish({}, _job(state))


@pytest.mark.parametrize("state", ["Stopping", "Terminating"])
async def test_finish_stopping_is_final(client, state):
    """The trigger handler reports Stopping/Terminating as fired, so they must
    be final here too; otherwise the in-process resume loop never ends."""
    result = json.loads(await uipath_process_tool("p").finish({}, _job(state)))
    assert result == {"status": state.lower(), "job_key": "job-7"}


async def test_finish_raw_payloads(client):
    tool = uipath_process_tool("p")
    assert await tool.finish({}, {"sum": 7}) == json.dumps({"sum": 7})
    assert await tool.finish({}, "plain text") == "plain text"
    assert await tool.finish({}, 42) == "42"
    assert await tool.finish({}, None) == "null"


# --------------------------------------------------------------------------
# pre-filled arguments (static_args / args_from_input)
# --------------------------------------------------------------------------
class Order(BaseModel):
    sku: str
    quantity: int
    channel: str
    customer: str


@pytest.mark.parametrize("factory", [uipath_process_tool, uipath_agent_tool])
async def test_static_args_and_args_from_input(client, factory):
    client["c"] = _Client(job=_job("Pending", FolderKey="fk-1"))
    tool = factory(
        "PlaceOrder",
        input_schema=Order,
        static_args={"channel": "codex"},
        args_from_input={"customer": "{{user.email}}"},
    )
    schema = tool.input_schema
    assert "channel" not in schema["properties"]  # hidden
    assert "agent input 'user.email'" in schema["properties"]["customer"]["description"]
    assert schema["required"] == ["sku", "quantity"]  # customer is optional now
    # the model's arguments validate without the required pre-filled fields
    args = tool.validate_args({"sku": "A-1", "quantity": "2"})
    assert args == {"sku": "A-1", "quantity": "2"}
    ctx = DurableToolContext(
        call_id="c1",
        runtime_id="rt",
        tool_name=tool.name,
        agent_input={"user": {"email": "ana@example.com"}},
    )
    await tool.start(args, ctx)
    # the merged input is validated (and coerced) by the model
    assert client["c"].calls[0]["input_arguments"] == {
        "sku": "A-1",
        "quantity": 2,
        "channel": "codex",
        "customer": "ana@example.com",
    }
    assert args == {"sku": "A-1", "quantity": "2"}  # not changed by start
    # an agent input without the value: the model's is used
    await tool.start({**args, "customer": "bob"}, CTX)
    assert client["c"].calls[1]["input_arguments"]["customer"] == "bob"
    # nothing to fill it with: the model gets the validation error
    with pytest.raises(UiPathDurableToolError, match="Invalid arguments"):
        await tool.start(args, CTX)


def test_code_paths_accept_jsonpath_and_nesting():
    schema = {
        "type": "object",
        "properties": {
            "meta": {"type": "object", "properties": {"source": {"type": "string"}}},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"taxed": {"type": "boolean"}},
                },
            },
        },
    }
    tool = uipath_process_tool(
        "p",
        input_schema=schema,
        static_args={"$['meta']['source']": "codex", "lines[*].taxed": True},
        args_from_input={"$.meta": "$.request.meta"},
    )
    assert set(tool.fixed_args) == {"meta.source", "lines[*].taxed", "meta"}
    assert tool.fixed_args["meta"].path == "request.meta"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"static_args": {"a": 1}, "args_from_input": {"a": "x"}},
            "'a' is in both static_args and args_from_input",
        ),
        ({"static_args": {"zzz": 1}}, "'zzz' is not in its input schema"),
        ({"args_from_input": {"a": "$..deep"}}, "is not a supported path"),
    ],
)
def test_code_api_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        uipath_process_tool("p", input_schema=AddIn, **kwargs)
    # a permissive schema takes any field
    assert uipath_process_tool("p", static_args={"zzz": 1}).fixed_args


def test_no_prefilled_args_keeps_the_schema_and_validation():
    tool = uipath_process_tool("p", input_schema=AddIn)
    assert tool.input_schema is tool.input_json_schema
    assert tool.merged_args({"a": 1, "b": 2}, None) == {"a": 1, "b": 2}
    assert tool.span_args({"a": 1}, CTX) == {"a": 1}
    with pytest.raises(ValueError):
        tool.validate_args({"a": 1})


def test_validate_args_with_prefilled_fields_still_checks_the_model():
    tool = uipath_process_tool(
        "PlaceOrder",
        input_schema=Order,
        static_args={"channel": "codex"},
        args_from_input={"customer": "user.email"},
    )
    # a missing agent-input field is fine (filled at start), the rest is checked
    assert tool.validate_args({"sku": "A", "quantity": 1}) == {
        "sku": "A",
        "quantity": 1,
    }
    for bad in ({"quantity": 1}, {"sku": "A", "quantity": "many"}):
        with pytest.raises(ValueError):
            tool.validate_args(bad)
