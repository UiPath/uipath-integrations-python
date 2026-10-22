"""Offline end-to-end tests of a process tool built from an agent.json resource.

Same set-up as ``test_e2e``: the real factory, runtime, bundled codex, gateway
shim, tool host and ``UiPathResumeTriggerHandler``, with a mock model and fake
Orchestrator jobs. The tool comes from ``process_tool_from_resource`` with one
``argumentProperties`` entry of each supported variant: the model fills only
``amount``, the job gets the merged input. The mocked variant runs the same
case under ``UiPathMockRuntime``, as ``uipath eval`` does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.core.triggers import UiPathResumeTriggerName, UiPathResumeTriggerType
from uipath.eval._execution_context import ExecutionSpanCollector
from uipath.eval.mocks import UiPathMockRuntime
from uipath.eval.mocks._types import MockingContext, MockitoMockingStrategy
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory

from .._typing import sqlite_storage
from ..contract.mock_responses import input_items, summarize_input
from .fakes import IntegrationScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
TOOL = "Create_Invoice"
CALL_ID = "call_createinvoice0000000001"
AGENT_INPUT = {"customer": {"id": "C1", "name": "Ada"}, "region": "EU"}
#: What the job gets: the model's ``amount`` plus every pre-filled field.
MERGED = {
    "amount": 10,
    "currency": "EUR",
    "apiKey": "s3cr3t",
    "customerId": "C1",
    "memo": "Invoice for Ada",
    "tags": ["agent", "EU"],
}

MAIN = """
import json

from pydantic import BaseModel

from uipath_codex_sdk import UiPathCodexAgent, process_tool_from_resource


class Customer(BaseModel):
    id: str
    name: str


class Input(BaseModel):
    customer: Customer
    region: str


RESOURCE = json.loads('''
{
  "$resourceType": "tool",
  "type": "process",
  "name": "Create Invoice",
  "description": "Creates an invoice in the ERP.",
  "isEnabled": true,
  "inputSchema": {
    "type": "object",
    "properties": {
      "customerId": {"type": "string"},
      "apiKey": {"type": "string"},
      "memo": {"type": "string"},
      "currency": {"type": "string"},
      "tags": {"type": "array", "items": {"type": "string"}},
      "amount": {"type": "number"}
    },
    "required": ["customerId", "amount", "currency", "apiKey"]
  },
  "outputSchema": {"type": "object", "properties": {"invoiceId": {"type": "string"}}},
  "properties": {"processName": "CreateInvoice", "folderPath": "Shared/Finance"},
  "settings": {"timeout": 600},
  "argumentProperties": {
    "$['currency']": {"variant": "static", "value": "EUR", "isSensitive": false},
    "$['apiKey']": {"variant": "static", "value": "s3cr3t", "isSensitive": true},
    "$['customerId']": {"variant": "argument", "argumentPath": "$.customer.id",
                        "isSensitive": false},
    "$['memo']": {"variant": "textBuilder", "isSensitive": false, "tokens": [
      {"type": "simpleText", "rawString": "Invoice for "},
      {"type": "variable", "rawString": "input.customer.name"}]},
    "$['tags']": {"variant": "arrayBuilder"},
    "$['tags'][0]": {"variant": "static", "value": "agent", "isSensitive": false},
    "$['tags'][1]": {"variant": "argument", "argumentPath": "$.region",
                     "isSensitive": false}
  }
}
''')

agent = UiPathCodexAgent(
    model="mock-model",
    prompt="Create the invoice for the customer.",
    input_schema=Input,
    tools=[process_tool_from_resource(RESOURCE)],
    durable_settle_timeout=20.0,
)
"""


class InvoiceScript(IntegrationScript):
    """Mock model: call ``Create_Invoice`` with only the amount, then answer
    with every tool output; records the advertised tool spec."""

    def __init__(self) -> None:
        super().__init__(tool=TOOL, args={"amount": 10})
        self.call_id = CALL_ID


@pytest.fixture
def project(project: Path, monkeypatch) -> Path:
    (project / "main.py").write_text(MAIN)
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Coded/Run")
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


def _factory(project: Path, *, resume: bool) -> UiPathCodexRuntimeFactory:
    return UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
        )
    )


async def _run(project: Path, input, *, resume: bool):
    factory = _factory(project, resume=resume)
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


def _tool_spec(script: InvoiceScript) -> dict[str, Any]:
    """The advertised ``Create_Invoice`` spec (inside its MCP namespace)."""

    def walk(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("name") == TOOL and "parameters" in node:
                return node
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    spec = walk(script.tool_specs)
    assert spec is not None, script.tool_specs
    return spec


def _assert_paired(body: dict[str, Any], expected_output: str) -> None:
    items = input_items(body)
    summary = summarize_input(body)
    calls = [
        i
        for i, it in enumerate(items)
        if it.get("type") == "function_call" and it.get("call_id") == CALL_ID
    ]
    outs = [
        (i, it)
        for i, it in enumerate(items)
        if it.get("type") == "function_call_output" and it.get("call_id") == CALL_ID
    ]
    assert len(calls) == 1 and len(outs) == 1, summary
    (oi, out) = outs[0]
    output = out["output"]
    if isinstance(output, list):  # a live MCP result: content parts
        output = [p.get("text") for p in output if isinstance(p, dict)][-1]
    assert output == expected_output, summary
    assert oi > calls[0], summary
    assert "turn_aborted" not in json.dumps(items), summary


async def test_resource_tool_suspends_with_the_merged_input_then_resumes(
    project, upstream, world, spans
):
    script = InvoiceScript()
    upstream.script = script

    # --- run: the model fills only the amount; the job gets everything ---
    result, triggers = await _run(project, AGENT_INPUT, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert script.advertised == (TOOL, "mcp__uipath_tools")
    parameters = _tool_spec(script)["parameters"]
    assert set(parameters["properties"]) == {"customerId", "amount"}
    assert parameters["required"] == ["amount"]
    [inv] = world.invocations
    assert inv.name == "CreateInvoice"
    assert inv.input_arguments == MERGED
    assert inv.folder_path == "Coded/Run"  # the run's folder, not folderPath
    assert result.output[CALL_ID]["arguments"] == {"amount": 10}
    [trig] = triggers
    assert trig.trigger_type == UiPathResumeTriggerType.JOB
    assert trig.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert trig.item_key == inv.job_key
    # the Tool call span stays open across the suspension
    assert not [s for s in spans.get_finished_spans() if s.name.startswith("Tool")]

    # --- the job finished: the run resumes and the output is paired ------
    world.finish(inv.job_key, "Successful", {"invoiceId": "INV-7"})
    _new_robot(project)
    result, triggers = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    assert len(world.invocations) == 1  # never started again
    expected = json.dumps({"invoiceId": "INV-7"})
    _assert_paired(upstream.snapshot()[-1].body, expected)
    [call] = [s for s in spans.get_finished_spans() if s.name == f"Tool call - {TOOL}"]
    assert call.attributes["toolType"] == "Process"
    assert json.loads(call.attributes["input"]) == MERGED  # low-code's tool call


def mockito(*behaviors: dict[str, Any]) -> MockitoMockingStrategy:
    return MockitoMockingStrategy.model_validate(
        {"type": "mockito", "config": list(behaviors)}
    )


async def test_mocked_resource_tool_gets_the_merged_input(
    project, upstream, world, spans
):
    script = InvoiceScript()
    upstream.script = script
    # The behavior matches only on the merged input, as in LangChain.
    strategy = mockito(
        {
            "function": "Create Invoice",
            "arguments": {"kwargs": MERGED},
            "then": [{"type": "return", "value": {"invoiceId": "SIM-1"}}],
        }
    )
    factory = _factory(project, resume=False)
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    mocked = UiPathMockRuntime(
        runtime,
        MockingContext(strategy=strategy, name="case", inputs=AGENT_INPUT),
        span_collector=ExecutionSpanCollector(),
        execution_id="e1",
    )
    try:
        result = await mocked.execute(AGENT_INPUT, UiPathExecuteOptions(resume=False))
        storage = await sqlite_storage(factory)
        triggers = await storage.get_triggers(RUNTIME_ID)
    finally:
        await runtime.dispose()
        await factory.dispose()
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert world.invocations == [] and not triggers  # no job, no suspension
    _assert_paired(upstream.snapshot()[-1].body, json.dumps({"invoiceId": "SIM-1"}))
    finished = {s.name: s for s in spans.get_finished_spans()}
    child = finished["Simulated result: CreateInvoice"]
    assert child.attributes["span_type"] == "processTool"
    assert json.loads(child.attributes["arguments"]) == MERGED
    assert json.loads(finished[f"Tool call - {TOOL}"].attributes["input"]) == MERGED
