"""Offline end-to-end tests of the job's trace spans.

Same set-up as ``test_e2e``: the real factory, runtime, bundled codex, gateway
shim and tool host, with a mock model and fake Orchestrator jobs.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.core.tracing.processors import UiPathExecutionBatchTraceProcessor
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeContext,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.tracing import JsonLinesFileExporter, LiveTrackingSpanProcessor

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.session_store import CodexSessionStore
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage

from .._typing import as_result, attrs, end_ns, start_ns
from .fakes import AddScript, SiblingScript

pytestmark = pytest.mark.contract

RUNTIME_ID = "default"

PLAIN_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(model="mock-model", prompt="Say hi to {name}.")
"""


@pytest.fixture
def spans():
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]
    yield exporter
    exporter.shutdown()


class RecordingLlmOps:
    """Stands in for ``LlmOpsHttpExporter``: records each upsert as sent."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.enabled = True
        self._lock = threading.Lock()

    def upsert_span(self, span, status_override=None):
        if not self.enabled:
            return None
        with self._lock:
            self.upserts.append(
                {
                    "name": span.name,
                    "span_id": span.get_span_context().span_id,
                    "status_override": status_override,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes or {}),
                }
            )
        return None

    def shutdown(self) -> None:
        pass

    def of(self, span_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [u for u in self.upserts if u["span_id"] == span_id]


@pytest.fixture
def llmops(spans):
    """A live-tracking processor (what ``uipath run`` adds for a job)."""
    exporter = RecordingLlmOps()
    # One worker: upserts land in order, and ``drain`` waits for all of them.
    processor = LiveTrackingSpanProcessor(exporter, max_workers=1)  # type: ignore[arg-type]
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]

    def drain() -> RecordingLlmOps:
        processor.executor.submit(lambda: None).result()
        return exporter

    yield drain
    exporter.enabled = False  # the provider keeps its processors


@pytest.fixture
def trace_file(tmp_path):
    """What ``uipath run --trace-file`` adds: JSON lines behind a batch processor."""
    path = tmp_path / "traces.jsonl"
    processor = UiPathExecutionBatchTraceProcessor(JsonLinesFileExporter(str(path)))
    trace.get_tracer_provider().add_span_processor(processor)  # type: ignore[attr-defined]

    def read() -> list[dict[str, Any]]:
        processor.force_flush()
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    yield read
    processor.shutdown()


async def _run(project: Path, input, *, resume: bool, **context):
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"),
            state_file="state.db",
            resume=resume,
            **context,
        )
    )
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        return await runtime.execute(input, UiPathExecuteOptions(resume=resume))
    finally:
        await runtime.dispose()
        await factory.dispose()


async def _saved_trace(project: Path) -> dict[str, Any] | None:
    storage = SqliteResumableStorage(str(project / "__uipath" / "state.db"))
    try:
        return await CodexSessionStore(storage, RUNTIME_ID).get_trace()
    finally:
        await storage.dispose()


def _new_robot(project: Path) -> None:
    shutil.rmtree(project / "__uipath" / "codex_home", ignore_errors=True)


def _by_name(spans: list[ReadableSpan], prefix: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name.startswith(prefix)]


def _parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent is not None else None


async def test_one_agent_run_across_suspend_and_resumes(
    project, upstream, world, spans, llmops, trace_file, monkeypatch
):
    """suspend → resume (job still running: suspends again) → resume (done):
    one Agent run, opened by the first execution and completed by the last."""
    monkeypatch.setenv("UIPATH_TRACE_ID", "0af7651916cd43dd8448eb211c80319c")
    upstream.script = AddScript(n_calls=1)

    # --- first execution: suspends on the job ---------------------------
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    first = spans.get_finished_spans()
    # The Agent run and the durable Tool call stay open across the suspension.
    assert _by_name(first, "Agent run - ") == []
    assert _by_name(first, "Tool call - ") == []
    saved = await _saved_trace(project)
    assert saved is not None
    root_id = int(saved["span_id"], 16)
    trace_id = int(saved["trace_id"], 16)
    assert saved["name"] == "Agent run - agent"
    assert saved["parent_span_id"] is None
    assert saved["attributes"]["status"] == "suspended"
    assert "a_plus_b" in saved["attributes"]["userPrompt"]

    llm_calls = _by_name(first, "LLM call")
    assert llm_calls and all(_parent_id(s) == root_id for s in llm_calls)
    assert all(attrs(s)["span_type"] == "llmCall" for s in llm_calls)
    # The call that asked for the tool records it, as the gateway returned it.
    assert any("a_plus_b" in attrs(s).get("toolCalls", "") for s in llm_calls)
    # The LLM Gateway got the LLM call span as the trace parent.
    traceparents = [
        r.headers.get("x-uipath-traceparent-id") for r in upstream.snapshot()
    ]
    expected = {
        f"00-0af7651916cd43dd8448eb211c80319c-{s.context.span_id:016x}-01"
        for s in llm_calls
    }
    assert traceparents and set(traceparents) <= expected

    # LLMOps saw both open spans as still running: never as finished.
    [inv] = world.invocations
    tool_id = int(inv.parent_span_id, 16)
    live = llmops()
    run_upserts = live.of(root_id)
    assert run_upserts and all(u["end_time"] is None for u in run_upserts)
    assert run_upserts[-1]["status_override"] == "Unset"
    assert run_upserts[-1]["attributes"]["status"] == "suspended"
    tool_upserts = live.of(tool_id)
    assert tool_upserts and all(u["end_time"] is None for u in tool_upserts)
    assert tool_upserts[-1]["status_override"] == "Unset"
    started = json.loads(tool_upserts[-1]["attributes"]["output"])
    assert started["status"] == "started" and started["job_key"] == inv.job_key

    # --- resume while the job runs: suspends again under the same run ---
    _new_robot(project)
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    second = spans.get_finished_spans()
    assert _by_name(second, "Agent run - ") == []
    assert _by_name(second, "Tool call - ") == []
    # As for low-code agents: no group span. This execution only re-read the
    # (fake, untraced) trigger and suspended again: it adds no span at all.
    assert second[len(first) :] == ()
    assert await _saved_trace(project) == saved
    assert all(u["end_time"] is None for u in llmops().of(root_id))

    # --- resume after the job finished: the run completes ---------------
    world.finish(inv.job_key, "Successful", {"sum": 42})
    _new_robot(project)
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    every = spans.get_finished_spans()
    third = every[len(second) :]
    # The resumed execution's spans sit directly under the restored Agent run.
    assert _by_name(third, "Resume") == []
    resumed_llm = _by_name(third, "LLM call")
    assert resumed_llm
    assert all(_parent_id(s) == root_id for s in resumed_llm)
    assert all(s.context.trace_id == trace_id for s in third)

    # The durable Tool call: completed once, the span the child job hangs under.
    [tool] = _by_name(every, "Tool call - a_plus_b")
    assert tool.context.span_id == tool_id
    assert _parent_id(tool) == root_id
    assert json.loads(attrs(tool)["input"]) == {"a": 17, "b": 25}
    assert json.loads(attrs(tool)["output"]) == {"sum": 42}
    assert tool.status.status_code.name == "OK"
    # Completed when its result was handed back, before the model saw it.
    assert start_ns(tool) < end_ns(tool) <= min(start_ns(s) for s in resumed_llm)

    # One Agent run for the whole job: original id and start, final status.
    [run] = _by_name(every, "Agent run - ")
    assert run.context.span_id == root_id
    assert run.context.trace_id == trace_id
    assert run.parent is None
    assert run.start_time == saved["start_time_ns"]
    assert run.start_time < min(s.start_time for s in every if s is not run)
    assert run.end_time >= max(s.end_time for s in every if s is not run)
    assert attrs(run)["status"] == "successful"
    assert attrs(run)["resumed"] is False
    assert attrs(run)["model"] == "mock-model"
    assert json.loads(attrs(run)["output"]) == result.output
    assert run.status.status_code.name == "OK"
    [output] = _by_name(every, "Agent output")
    assert _parent_id(output) == root_id
    assert json.loads(attrs(output)["output"]) == result.output
    assert all(s.context.trace_id == trace_id for s in every)
    assert await _saved_trace(project) is None

    # The trace file holds the whole job as one tree.
    lines = trace_file()
    assert len(lines) == len(every)
    [file_run] = [ln for ln in lines if ln["name"] == "Agent run - agent"]
    assert file_run["context"]["span_id"] == f"0x{root_id:016x}"
    assert file_run["parent_id"] is None
    [file_tool] = [ln for ln in lines if ln["name"] == "Tool call - a_plus_b"]
    assert file_tool["context"]["span_id"] == f"0x{tool_id:016x}"
    ids = {ln["context"]["span_id"] for ln in lines}
    assert all(ln["parent_id"] in ids for ln in lines if ln is not file_run)

    # LLMOps got the completions with the original ids.
    live = llmops()
    assert live.of(root_id)[-1]["end_time"] == run.end_time
    assert live.of(root_id)[-1]["status_override"] is None
    assert live.of(tool_id)[-1]["end_time"] == tool.end_time


async def test_streamed_resume_completes_the_agent_run(project, upstream, world, spans):
    """``uipath run`` without a job streams: same single Agent run."""
    upstream.script = AddScript(n_calls=1)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    saved = await _saved_trace(project)
    assert saved is not None
    [inv] = world.invocations
    world.finish(inv.job_key, "Successful", {"sum": 42})
    _new_robot(project)

    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file="state.db", resume=True
        )
    )
    runtime = await factory.new_runtime("agent", RUNTIME_ID)
    try:
        events = [
            e async for e in runtime.stream(None, UiPathStreamOptions(resume=True))
        ]
    finally:
        await runtime.dispose()
        await factory.dispose()
    assert as_result(events[-1]).status == UiPathRuntimeStatus.SUCCESSFUL
    finished = spans.get_finished_spans()
    [run] = _by_name(finished, "Agent run - ")
    assert run.context.span_id == int(saved["span_id"], 16)
    assert attrs(run)["status"] == "successful"
    assert _by_name(finished, "Resume") == []
    llm_calls = _by_name(finished, "LLM call")
    assert llm_calls
    assert all(_parent_id(s) == run.context.span_id for s in llm_calls)
    [tool] = _by_name(finished, "Tool call - a_plus_b")
    assert tool.context.span_id == int(inv.parent_span_id, 16)
    assert await _saved_trace(project) is None


async def test_faulted_resume_completes_the_run_and_open_calls_with_error(
    project, upstream, world, spans, monkeypatch
):
    """The resume faults after the first job's result was read, while the
    second job still runs: the Agent run and the open call complete with ERROR."""
    upstream.script = AddScript(n_calls=2)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    saved = await _saved_trace(project)
    assert saved is not None
    first, second = world.invocations
    world.finish(first.job_key, "Successful", {"sum": 42})

    async def broken(self, completed):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(CodexSessionStore, "set_completed", broken)
    _new_robot(project)
    with pytest.raises(RuntimeError, match="database is locked"):
        await _run(project, None, resume=True)

    finished = spans.get_finished_spans()
    [run] = _by_name(finished, "Agent run - ")
    assert run.context.span_id == int(saved["span_id"], 16)
    assert run.start_time == saved["start_time_ns"]
    assert attrs(run)["status"] == "faulted"
    assert run.status.status_code.name == "ERROR"
    assert "database is locked" in attrs(run)["error"]
    assert "output" not in attrs(run)  # not the suspension's summary
    assert _by_name(finished, "Resume") == []
    # Each durable Tool call completes exactly once, with its own id.
    tools = _by_name(finished, "Tool call - a_plus_b")
    assert sorted(t.context.span_id for t in tools) == sorted(
        int(i.parent_span_id, 16) for i in world.invocations
    )
    used = next(t for t in tools if t.context.span_id == int(first.parent_span_id, 16))
    assert json.loads(attrs(used)["output"]) == {"sum": 42}
    unused = next(t for t in tools if t is not used)
    assert unused.status.status_code.name == "ERROR"
    assert "before the job result was used" in attrs(unused)["error"]
    assert await _saved_trace(project) is None


async def test_fresh_run_forgets_a_saved_agent_run(project, upstream, world, spans):
    """A non-resume run on a kept state file starts its own Agent run."""
    upstream.script = AddScript(n_calls=1)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    saved = await _saved_trace(project)
    assert saved is not None

    (project / "main.py").write_text(PLAIN_MAIN)
    result = await _run(project, {"name": "Ada"}, resume=False, keep_state_file=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    [run] = _by_name(spans.get_finished_spans(), "Agent run - ")
    assert run.context.span_id != int(saved["span_id"], 16)
    assert run.parent is None
    assert attrs(run)["resumed"] is False
    assert await _saved_trace(project) is None


async def test_agent_without_durable_tools_is_one_agent_run(
    project, upstream, world, spans
):
    (project / "main.py").write_text(PLAIN_MAIN)
    upstream.script = AddScript()
    result = await _run(project, {"name": "Ada"}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    [run] = _by_name(finished, "Agent run - ")
    assert run.parent is None
    assert attrs(run)["status"] == "successful"
    assert run.status.status_code.name == "OK"
    assert _by_name(finished, "Resume") == []
    [output] = _by_name(finished, "Agent output")
    assert _parent_id(output) == run.context.span_id
    assert all(
        _parent_id(s) == run.context.span_id for s in _by_name(finished, "LLM call")
    )
    assert await _saved_trace(project) is None


async def test_in_process_resume_stays_in_one_agent_run(
    project, upstream, world, spans, monkeypatch
):
    """A job already done at suspend time is resumed by the fired-trigger loop
    inside the same execution: one live Agent run, no Resume span."""
    upstream.script = AddScript(n_calls=1)
    original = world.client

    def finishing_client():
        for inv in world.invocations:
            if world.jobs[inv.job_key].state == "Pending":
                world.finish(inv.job_key, "Successful", {"sum": 42})
        return original()

    import uipath.platform.resume_triggers._protocol as protocol

    monkeypatch.setattr(protocol, "UiPath", finishing_client)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    [run] = _by_name(finished, "Agent run - ")
    assert attrs(run)["status"] == "successful"
    assert _by_name(finished, "Resume") == []
    [inv] = world.invocations
    [tool] = _by_name(finished, "Tool call - a_plus_b")
    assert tool.context.span_id == int(inv.parent_span_id, 16)
    assert _parent_id(tool) == run.context.span_id
    assert json.loads(attrs(tool)["output"]) == {"sum": 42}
    [output] = _by_name(finished, "Agent output")
    assert _parent_id(output) == run.context.span_id
    assert await _saved_trace(project) is None


async def test_items_codex_runs_itself_are_traced(project, upstream, world, spans):
    """A shell command next to the durable call: a Tool call span."""
    upstream.script = SiblingScript(sibling_after=False)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED, result
    finished = spans.get_finished_spans()
    saved = await _saved_trace(project)
    assert saved is not None
    [shell] = _by_name(finished, "Tool call - shell")
    assert _parent_id(shell) == int(saved["span_id"], 16)
    assert attrs(shell)["toolType"] == "Internal"
    assert "command" in json.loads(attrs(shell)["input"])


async def _set_trace(project: Path, record: dict[str, Any] | None) -> None:
    storage = SqliteResumableStorage(str(project / "__uipath" / "state.db"))
    try:
        await CodexSessionStore(storage, RUNTIME_ID).set_trace(record)
    finally:
        await storage.dispose()


async def _suspend_on_one_job(project, upstream, world) -> dict[str, Any]:
    upstream.script = AddScript(n_calls=1)
    result = await _run(project, {}, resume=False)
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    saved = await _saved_trace(project)
    assert saved is not None
    [inv] = world.invocations
    world.finish(inv.job_key, "Successful", {"sum": 42})
    _new_robot(project)
    return saved


async def test_resumed_spans_keep_the_execution_id(
    project, upstream, world, spans, trace_file
):
    """Under ``uipath eval`` the execution span's ``execution.id`` reaches every
    span of a resumed execution, although its parent is the restored root."""
    saved = await _suspend_on_one_job(project, upstream, world)
    before = len(spans.get_finished_spans())
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span(
        "execution", attributes={"execution.id": "item-1"}
    ):
        result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    resumed = [s for s in spans.get_finished_spans()[before:] if s.name != "execution"]
    names = {s.name for s in resumed}
    assert {"LLM call", "Agent output", "Agent run - agent"} <= names
    assert _by_name(resumed, "LLM call")
    assert all(attrs(s).get("execution.id") == "item-1" for s in resumed), [
        s.name for s in resumed if attrs(s).get("execution.id") != "item-1"
    ]
    [run] = _by_name(resumed, "Agent run - ")
    assert run.context.span_id == int(saved["span_id"], 16)


@pytest.mark.parametrize(
    "record",
    [
        None,  # the trace key is gone
        {"span_id": "zz", "trace_id": "q"},  # unusable ids
        "no-start",  # usable ids, no start time
    ],
)
async def test_resume_without_a_usable_saved_agent_run_still_runs(
    project, upstream, world, spans, record
):
    """A missing or broken saved trace record degrades the trace, never the job."""
    saved = await _suspend_on_one_job(project, upstream, world)
    restorable = record == "no-start"
    if restorable:
        record = {k: v for k, v in saved.items() if k != "start_time_ns"}
    await _set_trace(project, record)  # type: ignore[arg-type]
    result = await _run(project, None, resume=True)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL, result
    finished = spans.get_finished_spans()
    runs = _by_name(finished, "Agent run - ")
    if restorable:
        # Restored, but it cannot be completed without its start time.
        assert runs == []
        llm_calls = _by_name(finished, "LLM call")
        assert llm_calls
        assert all(_parent_id(s) == int(saved["span_id"], 16) for s in llm_calls)
    else:
        # A new Agent run for this execution.
        [run] = runs
        assert run.context.span_id != int(saved["span_id"], 16)
        assert attrs(run)["status"] == "successful"
    assert _by_name(finished, "Resume") == []
    [tool] = _by_name(finished, "Tool call - a_plus_b")
    assert json.loads(attrs(tool)["output"]) == {"sum": 42}
    assert await _saved_trace(project) is None


def _fail_shim_start(monkeypatch) -> None:
    """Fault the run: the gateway shim cannot start."""
    from uipath_codex_sdk.runtime.gateway import CodexGatewayShim

    async def start(self):
        raise RuntimeError("no port")

    monkeypatch.setattr(CodexGatewayShim, "start", start)


async def test_a_failing_tool_span_cleanup_still_ends_the_agent_run(
    project, upstream, world, spans, monkeypatch
):
    from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime

    async def broken(self, error):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(UiPathCodexRuntime, "fail_open_tool_spans", broken)
    _fail_shim_start(monkeypatch)
    upstream.script = AddScript(n_calls=1)
    with pytest.raises(RuntimeError, match="no port"):
        await _run(project, {}, resume=False)
    [run] = _by_name(spans.get_finished_spans(), "Agent run - ")
    assert attrs(run)["status"] == "faulted"
    assert run.status.status_code.name == "ERROR"


async def test_a_failing_tool_span_cleanup_still_completes_a_resumed_run(
    project, upstream, world, spans, monkeypatch
):
    from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime

    saved = await _suspend_on_one_job(project, upstream, world)

    async def broken(self, error):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(UiPathCodexRuntime, "fail_open_tool_spans", broken)
    _fail_shim_start(monkeypatch)
    with pytest.raises(RuntimeError, match="no port"):
        await _run(project, None, resume=True)
    finished = spans.get_finished_spans()
    [run] = _by_name(finished, "Agent run - ")
    assert run.context.span_id == int(saved["span_id"], 16)
    assert attrs(run)["status"] == "faulted"
    assert run.status.status_code.name == "ERROR"
    assert await _saved_trace(project) is None
