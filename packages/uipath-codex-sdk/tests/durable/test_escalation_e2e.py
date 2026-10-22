"""Offline end-to-end tests of the Action Center escalation tool (suspend → resume).

Same set-up as ``test_e2e``: the real factory, runtime, bundled codex, gateway
shim, tool host and ``UiPathResumeTriggerHandler``, with a mock model and a
fake ``tasks`` service whose tasks the test completes, rejects or deletes.
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
from .fakes import EscalationScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"

ESCALATION_MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import UiPathCodexAgent, uipath_escalation_tool


class Review(BaseModel):
    amount: float
    reason: str
    source: str


class Decision(BaseModel):
    Comment: str


agent = UiPathCodexAgent(
    model="mock-model",
    prompt="PLEASE_CALL_DURABLE: ask a manager to approve the expense.",
    tools=[
        uipath_escalation_tool(
            "ExpenseApproval",
            folder_path="Shared",
            input_schema=Review,
            output_schema=Decision,
            title="Expense approval",
            assignee="manager@example.com",
            static_args={"source": "codex-agent"},
            outcome_mapping={"Reject": "end"},
        )
    ],
    durable_settle_timeout=20.0,
)
"""


@pytest.fixture
def project(project: Path) -> Path:
    (project / "main.py").write_text(ESCALATION_MAIN)
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


async def test_escalation_suspends_then_resumes_with_the_decision(
    project, upstream, world
):
    script = EscalationScript()
    upstream.script = script
    cid = script.call_ids[0]
    tasks = world.tasks

    # --- run: creates the task, assigns it, suspends ----------------------
    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert script.advertised == ("escalate_ExpenseApproval", "mcp__uipath_tools")
    [created] = tasks.created
    assert created["title"] == "Expense approval"
    assert created["data"] == {
        "amount": 1250.0,
        "reason": "Team offsite",
        "source": "codex-agent",  # pre-filled, hidden from the model
    }
    assert created["app_name"] == "ExpenseApproval"
    assert (created["app_folder_path"], created["app_folder_key"]) == ("Shared", None)
    assert "recipient" not in created and "assignee" not in created  # assigned after
    [assignment] = tasks.assignments
    assert assignment["json"]["taskAssignments"][0]["UserNameOrEmail"] == (
        "manager@example.com"
    )
    task = tasks.tasks[created["key"]]
    summary = result.output[cid]
    assert summary["task_id"] == task["id"]
    assert summary["task_key"] == created["key"]
    assert summary["app_name"] == "ExpenseApproval"
    assert summary["type"] == "WaitEscalation"
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.TASK
    assert trig.trigger_name == UiPathResumeTriggerName.ESCALATION
    assert (trig.item_key, trig.folder_path) == (created["key"], "Shared")

    # --- still open: --resume suspends again ------------------------------
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert [t.interrupt_id for t in triggers] == [cid]

    # --- approved: the model gets langchain's escalation result -----------
    tasks.complete(created["key"], "Approve", {"Comment": "Fine by me"})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    assert json.loads(_final_outputs(result)[cid]) == {
        "status": "completed",
        "outcome": "Approve",
        "output": {"Comment": "Fine by me"},  # only the output schema's fields
        "task_id": task["id"],
        "assigned_to": "reviewer@example.com",
    }
    assert len(tasks.created) == 1  # never created again


@pytest.mark.parametrize("ending", ["reject", "delete"])
async def test_end_outcome_and_deleted_task_end_the_run(
    project, upstream, world, spans, ending
):
    from uipath_codex_sdk.runtime.errors import UiPathRunTerminatedError

    script = EscalationScript()
    upstream.script = script
    tasks = world.tasks
    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    [created] = tasks.created
    if ending == "reject":
        tasks.complete(created["key"], "Reject", {"Comment": "Too expensive"})
    else:
        tasks.delete(created["key"])
    _new_robot(project)
    with pytest.raises(Exception) as info:
        await _run(project, None, resume=True)
    error = info.value
    assert "TERMINATION_ESCALATION_REJECTED" in str(
        getattr(error, "error_info", None) or error
    ) or isinstance(error, UiPathRunTerminatedError)
    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - escalate_ExpenseApproval"]
    [child] = [s for s in finished if s.name == "ExpenseApproval"]
    assert call.status.status_code.name == "OK"  # closed once, with the outcome
    result_attr = json.loads(child.attributes["result"])
    if ending == "reject":
        assert result_attr == {
            "outcome": "Reject",
            "output": {"Comment": "Too expensive"},
        }
    else:
        assert result_attr == {
            "outcome": "The escalation task was deleted",
            "output": None,
        }


async def test_assignment_failure_still_waits_on_the_task(
    project, upstream, world, spans
):
    script = EscalationScript()
    upstream.script = script
    world.tasks.fail_assign = RuntimeError("user not found")
    result, triggers = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert len(world.tasks.created) == 1 and len(triggers) == 1
    [created] = world.tasks.created
    world.tasks.complete(created["key"], "Approve", {"Comment": "ok"})
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    [child] = [s for s in spans.get_finished_spans() if s.name == "ExpenseApproval"]
    assert child.attributes["assignmentError"] == "user not found"
    assert "assignedTo" not in child.attributes


async def test_typed_resume_answer(project, upstream, world):
    script = EscalationScript()
    upstream.script = script
    cid = script.call_ids[0]
    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    reads = len(world.tasks.retrieve_calls)  # the wrapper's check after suspending
    _new_robot(project)
    result, _ = await _run(
        project, {"action": "Approve", "data": {"Comment": "typed"}}, resume=True
    )
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    answer = json.loads(_final_outputs(result)[cid])
    assert (answer["outcome"], answer["output"]) == ("Approve", {"Comment": "typed"})
    assert len(world.tasks.retrieve_calls) == reads  # the resume did not read it


async def test_escalation_spans_stay_open_across_the_suspension(
    project, upstream, world, spans, monkeypatch
):
    monkeypatch.setenv("UIPATH_URL", "https://cloud.test/org/tenant")
    script = EscalationScript()
    upstream.script = script
    result, _ = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    names = [s.name for s in spans.get_finished_spans()]
    assert "Tool call - escalate_ExpenseApproval" not in names
    assert "ExpenseApproval" not in names
    [created] = world.tasks.created
    task_id = world.tasks.tasks[created["key"]]["id"]
    world.tasks.complete(created["key"], "Approve", {"Comment": "Fine"})
    _new_robot(project)
    result, _ = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - escalate_ExpenseApproval"]
    [child] = [s for s in finished if s.name == "ExpenseApproval"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert call.attributes["toolType"] == "Escalation"
    assert call.attributes["tool.kind"] == "escalation"
    # The span shows low-code's {outcome, output}; the evaluators' text is full.
    assert json.loads(call.attributes["output"]) == {
        "outcome": "Approve",
        "output": {"Comment": "Fine"},
    }
    content = json.loads(json.loads(call.attributes["output.value"])["content"])
    assert content["task_id"] == task_id and content["status"] == "completed"
    attrs = child.attributes
    assert attrs["span_type"] == "escalationTool"
    assert attrs["channelType"] == "actionCenter"
    assert attrs["taskId"] == str(task_id)
    assert attrs["taskUrl"] == f"https://cloud.test/org/tenant/actions_/tasks/{task_id}"
    assert attrs["assignedTo"] == "manager@example.com"
    assert json.loads(attrs["arguments"])["source"] == "codex-agent"
    assert json.loads(attrs["result"]) == {
        "outcome": "Approve",
        "output": {"Comment": "Fine"},
    }
    assert "output" not in attrs
    assert child.end_time <= call.end_time
