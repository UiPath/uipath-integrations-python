"""Offline end-to-end tests of eval tool mocking with durable tools.

The real factory, runtime, bundled codex, gateway shim and tool host, wrapped
in ``UiPathMockRuntime`` the way ``uipath eval`` wraps each case: a simulated
durable call is answered in the same turn (no job, task or extraction, no
suspension), the others suspend and resume as usual.
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
from uipath.eval._execution_context import ExecutionSpanCollector
from uipath.eval.mocks import UiPathMockRuntime
from uipath.eval.mocks._types import (
    LLMMockingStrategy,
    MockingContext,
    MockitoMockingStrategy,
    ToolSimulation,
)
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory

from .._typing import sqlite_storage
from ..contract.mock_responses import (
    advertised_tools,
    find_tool,
    function_call_item,
    input_items,
    message_item,
)
from .fakes import AddScript, EscalationScript
from .test_cli import _uipath
from .test_ixp_e2e import DOC, EXTRACT_ID, VALIDATE_ID
from .test_ixp_e2e import MAIN as IXP_MAIN
from .test_ixp_e2e import Script as IxpScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"

MIXED_MAIN = """
from pydantic import BaseModel

from uipath_codex_sdk import (
    UiPathCodexAgent,
    uipath_escalation_tool,
    uipath_process_tool,
)


class AddIn(BaseModel):
    a: int
    b: int


class Review(BaseModel):
    amount: float
    reason: str


class Decision(BaseModel):
    Comment: str


agent = UiPathCodexAgent(
    model="mock-model",
    prompt="PLEASE_CALL_DURABLE: add the numbers and ask a manager.",
    tools=[
        uipath_process_tool(
            "a_plus_b",
            folder_path="Shared/Durable",
            description="Adds a and b.",
            input_schema=AddIn,
        ),
        uipath_escalation_tool(
            "ExpenseApproval",
            folder_path="Shared",
            input_schema=Review,
            output_schema=Decision,
            outcome_mapping={"reject": "end"},
        ),
    ],
    durable_settle_timeout=20.0,
)
"""

ESCALATION = "escalate_ExpenseApproval"


def mockito(*behaviors: dict[str, Any]) -> MockitoMockingStrategy:
    return MockitoMockingStrategy.model_validate(
        {"type": "mockito", "config": list(behaviors)}
    )


def returns(function: str, value: Any) -> dict[str, Any]:
    return {"function": function, "then": [{"type": "return", "value": value}]}


async def _run(project: Path, input, strategy: Any, *, resume: bool = False):
    """One execution of the case, as ``uipath eval`` runs it."""
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file="state.db", resume=resume
        )
    )
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    mocked = UiPathMockRuntime(
        runtime,
        MockingContext(strategy=strategy, name="case", inputs=input or {}),
        span_collector=ExecutionSpanCollector(),
        execution_id="e1",
    )
    try:
        result = await mocked.execute(input, UiPathExecuteOptions(resume=resume))
        storage = await sqlite_storage(factory)
        return result, await storage.get_triggers(RUNTIME_ID)
    finally:
        await runtime.dispose()
        await factory.dispose()


def _new_robot(project: Path) -> None:
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _text(output: Any) -> str:
    """A function_call_output as text (a live MCP result is content parts)."""
    if isinstance(output, list):
        texts = [p.get("text", "") for p in output if isinstance(p, dict)]
        return texts[-1] if texts else ""
    return output if isinstance(output, str) else json.dumps(output)


def _outputs(body: dict[str, Any], call_id: str) -> list[str]:
    return [
        _text(it.get("output"))
        for it in input_items(body)
        if it.get("type") == "function_call_output" and it.get("call_id") == call_id
    ]


def _first_with_output(mock, call_id: str):
    for req in mock.snapshot():
        if _outputs(req.body, call_id):
            return req
    return None


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


# --------------------------------------------------------------------------
# a./b. a mocked process tool never starts a job and never suspends
# --------------------------------------------------------------------------
async def test_mockito_mocked_process_answers_in_the_same_turn(
    project, upstream, world
):
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]
    result, triggers = await _run(
        project, {}, mockito(returns("a_plus_b", {"sum": 42}))
    )
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert world.invocations == [] and not triggers
    req = _first_with_output(upstream, cid)
    assert req is not None
    assert _outputs(req.body, cid) == ['{"sum": 42}']
    assert result.output["output"].startswith("FINAL ")


async def test_llm_mocked_process_needs_no_network(
    project, upstream, world, monkeypatch
):
    import uipath.eval.mocks._llm_mocker as llm_mocker

    prompts: list[str] = []

    async def generate(llm, messages, *, schema, **kw):
        prompts.append(messages[0]["content"])
        return {"sum": 99}

    monkeypatch.setattr(llm_mocker, "generate_structured_output", generate)
    monkeypatch.setattr(
        llm_mocker,
        "UiPath",
        lambda: type("U", (), {"_config": None, "_execution_context": None})(),
    )
    monkeypatch.setattr(llm_mocker, "UiPathLlmChatService", lambda *a, **k: object())
    script = AddScript(n_calls=1)
    upstream.script = script
    cid = script.call_ids[0]
    strategy = LLMMockingStrategy(
        prompt="Return a plausible sum.",
        tools_to_simulate=[ToolSimulation(name="a_plus_b")],
    )
    result, _ = await _run(project, {}, strategy)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert world.invocations == []
    [prompt] = prompts
    assert '"kwargs": {"a": 17, "b": 25}' in prompt
    req = _first_with_output(upstream, cid)
    assert req is not None and _outputs(req.body, cid) == ['{"sum": 99}']


# --------------------------------------------------------------------------
# c. a mocked escalation next to a real process call
# --------------------------------------------------------------------------
class BothScript(AddScript):
    """The durable ``a_plus_b`` call and the escalation in one response."""

    ESCALATION_ID = "call_escalation00000000000001"

    def __init__(self) -> None:
        super().__init__(n_calls=1)

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        outputs = {
            it["call_id"]: _text(it.get("output"))
            for it in items
            if it.get("type") == "function_call_output"
        }
        if self.call_ids[0] in seen:
            return [message_item("FINAL " + json.dumps(outputs, sort_keys=True), idx)]
        tools = advertised_tools(body)
        add, ns = find_tool(tools, "a_plus_b")
        esc, ens = find_tool(tools, ESCALATION)
        if not add or not esc:
            return None
        return [
            function_call_item(self.ESCALATION_ID, esc, EscalationScript.ARGS, ens),
            function_call_item(self.call_ids[0], add, {"a": 1, "b": 2}, ns),
        ]


async def test_mocked_escalation_with_a_real_process(project, upstream, world):
    (project / "main.py").write_text(MIXED_MAIN)
    script = BothScript()
    upstream.script = script
    cid, eid = script.call_ids[0], BothScript.ESCALATION_ID
    strategy = mockito(
        returns(ESCALATION.lower(), {"action": "approve", "data": {"Comment": "fine"}})
    )
    result, triggers = await _run(project, {}, strategy)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert list(result.output) == [cid]  # only the process call is pending
    assert [t.interrupt_id for t in triggers] == [cid]
    assert world.tasks.created == []  # no Action Center task
    [inv] = world.invocations

    world.finish(inv.job_key, "Successful", {"sum": 3})
    _new_robot(project)
    result, triggers = await _run(project, None, strategy, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    final = upstream.snapshot()[-1].body
    assert _outputs(final, cid) == [json.dumps({"sum": 3})]
    [escalation] = _outputs(final, eid)  # answered once, never re-mocked
    assert json.loads(escalation) == {
        "status": "completed",
        "outcome": "approve",
        "output": {"Comment": "fine"},
        "task_id": None,
        "assigned_to": None,
    }
    assert world.tasks.created == []


# --------------------------------------------------------------------------
# d./e. IXP extraction and its Document Validation
# --------------------------------------------------------------------------
def extraction(total: str = "999") -> dict[str, Any]:
    from .fakes import FakeDocuments

    return {
        "extractionResult": {
            "DocumentId": "doc-simulated",
            "ResultsVersion": 1,
            "ResultsDocument": {"Fields": []},
        },
        "projectId": "simulated-project",
        "projectType": "IXP",
        "tag": "live",
        "documentTypeId": "00000000-0000-0000-0000-000000000000",
        "dataProjection": FakeDocuments.projection(total),
    }


@pytest.fixture
def ixp_project(project: Path, monkeypatch) -> Path:
    (project / "main.py").write_text(IXP_MAIN)
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Coded")
    return project


async def test_mocked_extraction_feeds_a_real_validation(ixp_project, upstream, world):
    from .fakes import FakeDocuments

    upstream.script = IxpScript()
    docs = world.documents
    strategy = mockito(returns("Invoice Extraction", extraction()))
    result, triggers = await _run(ixp_project, {"invoice": DOC}, strategy)
    # The extraction was simulated; the validation task is real.
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert docs.starts == [] and world.attachments.downloads == []
    assert list(result.output) == [VALIDATE_ID]
    [validation] = docs.validation_starts
    sent = validation["extraction_response"]
    assert sent.project_id == "simulated-project"
    assert sent.extraction_result.document_id == "doc-simulated"
    req = _first_with_output(upstream, EXTRACT_ID)
    assert req is not None
    assert json.loads(_outputs(req.body, EXTRACT_ID)[0]) == FakeDocuments.projection(
        "999"
    )


async def test_mocked_extraction_and_validation_touch_no_documents(
    ixp_project, upstream, world
):
    from .fakes import FakeDocuments

    upstream.script = IxpScript()
    docs = world.documents
    validated = {"data": FakeDocuments.projection("1000")}
    strategy = mockito(
        returns("Invoice Extraction", extraction()),
        returns("Validate Invoice", validated),
    )
    result, triggers = await _run(ixp_project, {"invoice": DOC}, strategy)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    assert not triggers
    assert docs.starts == [] and docs.validation_starts == []
    final = upstream.snapshot()[-1].body
    assert json.loads(_outputs(final, VALIDATE_ID)[0]) == validated


# --------------------------------------------------------------------------
# f. a simulated escalation outcome mapped to "end" ends the run
# --------------------------------------------------------------------------
async def test_mocked_escalation_outcome_mapped_to_end(project, upstream, world):
    from uipath_codex_sdk.runtime.errors import UiPathRunTerminatedError

    (project / "main.py").write_text(MIXED_MAIN)
    upstream.script = EscalationScript()
    strategy = mockito(
        returns(ESCALATION.lower(), {"action": "reject", "data": {"Comment": "no"}})
    )
    with pytest.raises(Exception) as info:
        await _run(project, {}, strategy)
    error = info.value
    assert isinstance(error, UiPathRunTerminatedError)
    assert "TERMINATION_ESCALATION_REJECTED" in str(error.error_info.code)
    assert world.tasks.created == [] and world.invocations == []


# --------------------------------------------------------------------------
# g. tracing: low-code's "Simulated result" child under the Tool call
# --------------------------------------------------------------------------
async def test_mocked_process_trace(project, upstream, world, spans):
    script = AddScript(n_calls=1)
    upstream.script = script
    result, _ = await _run(project, {}, mockito(returns("a_plus_b", {"sum": 42})))
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    [call] = [s for s in finished if s.name == "Tool call - a_plus_b"]
    [child] = [s for s in finished if s.name == "Simulated result: a_plus_b"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert child.attributes["span_type"] == "processTool"
    assert child.attributes["__uipath_response_mocked"] is True
    assert json.loads(call.attributes["output.value"]) == {"content": '{"sum": 42}'}
    [run] = [s for s in finished if s.name.startswith("Agent run")]
    assert call.parent is not None and call.parent.span_id == run.context.span_id


# --------------------------------------------------------------------------
# h. the real ``uipath eval`` CLI, offline
# --------------------------------------------------------------------------
CONTAINS_EVALUATOR = {
    "version": "1.0",
    "id": "evaluator-contains",
    "description": "The answer contains the simulated sum.",
    "evaluatorTypeId": "uipath-contains",
    "evaluatorConfig": {
        "name": "Contains",
        "case_sensitive": False,
        "targetOutputKey": "*",
    },
}

MOCKITO_SET = {
    "version": "1.0",
    "id": "mocked-set",
    "name": "Mocked durable tool",
    "evaluatorRefs": ["evaluator-contains"],
    "evaluations": [
        {
            "id": "mocked-sum",
            "name": "Mocked sum",
            "inputs": {},
            "evaluationCriterias": {"evaluator-contains": {"searchText": "4242"}},
            "mockingStrategy": {
                "type": "mockito",
                "config": [
                    {
                        "function": "a_plus_b",
                        "arguments": {"kwargs": {"a": 17, "b": 25}},
                        "then": [{"type": "return", "value": {"sum": 4242}}],
                    }
                ],
            },
        }
    ],
}

STAMP = "2026-01-01T00:00:00.000Z"
LEGACY_SET = {
    "id": "legacy-set",
    "fileName": "legacy.json",
    "name": "Legacy simulated tools",
    "evaluatorRefs": [],
    "createdAt": STAMP,
    "updatedAt": STAMP,
    "evaluations": [
        {
            "id": "legacy-sum",
            "name": "Legacy sum",
            "inputs": {},
            "expectedOutput": {},
            "evalSetId": "legacy-set",
            "createdAt": STAMP,
            "updatedAt": STAMP,
            "simulateTools": True,
            "simulationInstructions": "Return the sum 7777.",
            "toolsToSimulate": [{"name": "a_plus_b"}],
        }
    ],
}


@pytest.fixture
def eval_project(cli_env):
    proj, server = cli_env
    sets = proj / "evaluations" / "eval-sets"
    evaluators = proj / "evaluations" / "evaluators"
    sets.mkdir(parents=True)
    evaluators.mkdir(parents=True)
    (evaluators / "contains.json").write_text(json.dumps(CONTAINS_EVALUATOR))
    (sets / "mocked.json").write_text(json.dumps(MOCKITO_SET))
    (sets / "legacy.json").write_text(json.dumps(LEGACY_SET))
    return proj, server


def test_uipath_eval_with_a_mockito_strategy(eval_project, world):
    proj, server = eval_project
    script = AddScript(n_calls=1)
    server.script = script
    res = _uipath(
        "eval",
        "agent",
        "evaluations/eval-sets/mocked.json",
        "--no-report",
        "--output-file",
        "eval-out.json",
    )
    assert res.exit_code == 0, res.output
    assert world.invocations == []  # no Orchestrator job
    out = json.loads((proj / "eval-out.json").read_text())
    [case] = out["evaluationSetResults"]
    [score] = case["evaluationRunResults"]
    assert score["result"]["score"] == 1.0, out
    cid = script.call_ids[0]
    assert _outputs(server.snapshot()[-1].body, cid) == ['{"sum": 4242}']


def test_uipath_eval_with_a_legacy_simulated_tool(eval_project, world, monkeypatch):
    import uipath.eval.mocks._llm_mocker as llm_mocker

    seen: list[dict[str, Any]] = []

    async def generate(llm, messages, *, schema, **kw):
        seen.append(kw.get("completion_kwargs") or {})
        return {"sum": 7777}

    monkeypatch.setattr(llm_mocker, "generate_structured_output", generate)
    monkeypatch.setattr(
        llm_mocker,
        "UiPath",
        lambda: type("U", (), {"_config": None, "_execution_context": None})(),
    )
    monkeypatch.setattr(llm_mocker, "UiPathLlmChatService", lambda *a, **k: object())
    proj, server = eval_project
    script = AddScript(n_calls=1)
    server.script = script
    res = _uipath(
        "eval",
        "agent",
        "evaluations/eval-sets/legacy.json",
        "--no-report",
        "--output-file",
        "legacy-out.json",
    )
    assert res.exit_code == 0, res.output
    assert world.invocations == []
    # The LLM mocker runs on the agent's model (``metadata.settings.model``).
    assert [k.get("model") for k in seen] == ["mock-model"]
    cid = script.call_ids[0]
    assert _outputs(server.snapshot()[-1].body, cid) == ['{"sum": 7777}']
