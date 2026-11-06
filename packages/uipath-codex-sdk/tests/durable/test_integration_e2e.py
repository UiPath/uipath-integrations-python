"""Offline end-to-end tests of Integration Service (immediate) tools.

The real factory → runtime → bundled codex 0.161 → gateway shim (mock model)
→ ``uipath_tools`` MCP host, with a fake ``connections`` service standing in
for Integration Service (and fake Orchestrator jobs for the durable sibling).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
)

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.session_store import CodexSessionStore
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage
from uipath_codex_sdk.runtime.tool_host import UNKNOWN_OUTCOME_OUTPUT

from ..contract.mock_responses import input_items, summarize_input
from .fakes import (
    IS_CALL_ID,
    IS_TOOL,
    IntegrationScript,
    MixedScript,
    enriched,
)

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"
CONN = "22222222-aaaa-bbbb-cccc-000000000002"
RESULT = {"formattedResults": "UiPath Codex SDK — docs", "results": [{"url": "u"}]}

IS_TOOL_DECL = f"""
uipath_integration_tool(
    "Web Search",
    tool_name="{IS_TOOL}",
    connection_id="{CONN}",
    connector_key="uipath-uipath-airdk",
    object_name="v2::webSearch",
    tool_path="/v2/webSearch",
    input_schema=SearchIn,
    static_args={{"provider": "GoogleCustomSearch"}},
    description="Searches the public web.",
)
"""


def _main(*, durable: bool, settle: float = 20.0) -> str:
    tools = [IS_TOOL_DECL.strip()]
    if durable:
        tools.insert(
            0,
            'uipath_process_tool("a_plus_b", folder_path="Shared/Durable", '
            'description="Adds a and b.", input_schema=AddIn)',
        )
    return (
        textwrap.dedent(
            """
        from pydantic import BaseModel

        from uipath_codex_sdk import (
            UiPathCodexAgent,
            uipath_integration_tool,
            uipath_process_tool,
        )


        class AddIn(BaseModel):
            a: int
            b: int


        class SearchIn(BaseModel):
            query: str


        agent = UiPathCodexAgent(
            model="mock-model",
            prompt="PLEASE_CALL_DURABLE: search the web.",
            tools=[TOOLS],
            durable_settle_timeout=SETTLE,
        )
        """
        )
        .replace("TOOLS", ",\n        ".join(tools))
        .replace("SETTLE", repr(settle))
    )


async def _run(project: Path, input, *, resume: bool):
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


def _new_robot(project: Path) -> None:
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _text(output: Any) -> str:
    """A function_call_output as text.

    Codex hands a live MCP result to the model as content parts (a ``Wall
    time`` header, then the tool text); an injected output is a string.
    """
    if isinstance(output, list):
        texts = [p.get("text", "") for p in output if isinstance(p, dict)]
        return texts[-1] if texts else ""
    return output if isinstance(output, str) else json.dumps(output)


def _final(result) -> dict[str, str]:
    text = result.output["output"]
    assert text.startswith("FINAL "), text
    return {k: _text(v) for k, v in json.loads(text[len("FINAL ") :]).items()}


def _tool_spec(tools: list[dict[str, Any]], needle: str) -> dict[str, Any] | None:
    for t in tools:
        for sub in [t, *(t.get("tools") or [])]:
            if needle in (sub.get("name") or ""):
                return sub
    return None


def _outputs_for(body: dict[str, Any], call_id: str) -> list[dict[str, Any]]:
    return [
        it
        for it in input_items(body)
        if it.get("type", "").endswith("_output") and it.get("call_id") == call_id
    ]


async def _session_rows(project: Path) -> tuple[Any, Any]:
    db = project / "__uipath" / "state.db"
    if not db.exists():
        return {}, None
    storage = SqliteResumableStorage(str(db))
    try:
        store = CodexSessionStore(storage, RUNTIME_ID)
        return await store.get_pending(), await store.get_thread()
    finally:
        await storage.dispose()


@pytest.fixture
def spans():
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(  # type: ignore[attr-defined]
        SimpleSpanProcessor(exporter)
    )
    yield exporter
    exporter.shutdown()


# --------------------------------------------------------------------------
async def test_integration_tool_only_agent(project, upstream, world, spans):
    (project / "main.py").write_text(_main(durable=False))
    world.connections.responses["/v2/webSearch"] = RESULT
    script = IntegrationScript()
    upstream.script = script

    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    name, ns = script.advertised or (None, None)
    assert name == IS_TOOL and ns == "mcp__uipath_tools", script.advertised
    spec = _tool_spec(script.tool_specs, IS_TOOL)
    assert spec is not None
    schema_text = json.dumps(spec)
    assert "query" in schema_text and "provider" not in schema_text
    assert "[long-running]" not in schema_text

    assert json.loads(_final(result)[IS_CALL_ID]) == RESULT
    [inv] = world.connections.invocations
    assert inv.connection_id == CONN
    assert inv.activity_input == {
        "query": "uipath codex",
        "provider": "GoogleCustomSearch",
    }
    md = inv.activity_metadata
    assert md.object_path == "/v2/webSearch" and md.method_name == "POST"
    assert md.parameter_location_info.body_fields == ["query", "provider"]
    assert world.invocations == []  # no job
    pending, thread = await _session_rows(project)
    assert pending == {} and thread is None  # the session store is not used

    # --- tracing: one Tool call span from the host, with the IS child --------
    finished = list(spans.get_finished_spans())
    [run] = [s for s in finished if s.name.startswith("Agent run - ")]
    calls = [s for s in finished if s.name == f"Tool call - {IS_TOOL}"]
    assert len(calls) == 1, [s.name for s in finished]  # no duplicate Mcp item span
    [call] = calls
    assert call.parent is not None and call.parent.span_id == run.context.span_id
    assert call.attributes["toolType"] == "Integration"
    assert call.attributes["tool.server"] == "uipath_tools"
    assert call.attributes["connector_key"] == "uipath-uipath-airdk"
    [child] = [s for s in finished if s.name == "Web Search"]
    assert child.attributes["span_type"] == "integrationTool"
    assert child.parent is not None and child.parent.span_id == call.context.span_id

    from uipath.eval._helpers.evaluators_helpers import (
        extract_tool_calls,
        extract_tool_calls_names,
        extract_tool_calls_outputs,
    )

    ours: list[ReadableSpan] = [call]
    assert extract_tool_calls_names(ours) == [IS_TOOL]
    assert [c.args for c in extract_tool_calls(ours)] == [{"query": "uipath codex"}]
    [out] = extract_tool_calls_outputs(ours)
    assert json.loads(out.output) == RESULT
    # Across the whole run, the IS call is the only tool call the evals see.
    assert extract_tool_calls_names(
        [s for s in finished if s.context.trace_id == run.context.trace_id]
    ) == [IS_TOOL]


async def test_integration_error_reaches_the_model(project, upstream, world):
    (project / "main.py").write_text(_main(durable=False))
    world.connections.fail = enriched(400, message="query is too long")
    upstream.script = IntegrationScript()

    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    output = _final(result)[IS_CALL_ID]
    assert "Integration service returned an error" in output, output
    assert "query is too long" in output
    assert len(world.connections.invocations) == 1


@pytest.mark.parametrize("is_after", [False, True], ids=["is_before", "is_after"])
async def test_integration_and_durable_calls_in_one_response(
    project, upstream, world, is_after
):
    """The IS call is answered while the durable call is held; the run
    suspends on the job, and on resume both outputs are paired exactly once."""
    (project / "main.py").write_text(_main(durable=True))
    world.connections.responses["/v2/webSearch"] = RESULT
    script = MixedScript(is_after=is_after)
    upstream.script = script
    cid = script.call_ids[0]

    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert list(result.output) == [cid]
    assert len(world.connections.invocations) == 1
    [job] = world.invocations
    world.finish(job.job_key, "Successful", {"sum": 3})

    _new_robot(project)
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    final = _final(result)
    assert json.loads(final[cid]) == {"sum": 3}
    assert json.loads(final[IS_CALL_ID]) == RESULT
    assert len(world.connections.invocations) == 1  # never re-run

    last = upstream.snapshot()[-1].body
    is_outs = _outputs_for(last, IS_CALL_ID)
    assert len(is_outs) == 1, summarize_input(last)
    assert "Not executed" not in _text(is_outs[0]["output"])
    assert "aborted" not in _text(is_outs[0]["output"])
    assert len(_outputs_for(last, cid)) == 1, summarize_input(last)


async def test_integration_call_cut_off_by_the_settle_timeout(project, upstream, world):
    """An IS call still running when the settle timeout ends is cancelled; the
    model is told its outcome is unknown, not that it was not executed."""
    (project / "main.py").write_text(_main(durable=True, settle=2.0))
    world.connections.gate = asyncio.Event()  # never released
    script = MixedScript(is_after=True)
    upstream.script = script
    cid = script.call_ids[0]

    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    assert len(world.connections.invocations) == 1
    [job] = world.invocations
    world.finish(job.job_key, "Successful", {"sum": 3})

    world.connections.gate = None
    _new_robot(project)
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    final = _final(result)
    assert json.loads(final[cid]) == {"sum": 3}
    assert final[IS_CALL_ID] == UNKNOWN_OUTCOME_OUTPUT
    assert len(world.connections.invocations) == 1
