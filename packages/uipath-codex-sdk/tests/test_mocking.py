"""Eval tool mocking: the simulation gate, mock specs and the host's mocked path.

Real mockers: ``set_execution_context`` builds the LLM, mockito or
simulated-component mocker, as ``UiPathMockRuntime`` does in ``uipath eval``.
The LLM mocker's structured-output call is patched, and the platform client
raises, so a mocked call provably never reaches the tenant.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from typing import Any

import pytest
from aiohttp import ClientSession, ClientTimeout, web
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from uipath.eval._execution_context import ExecutionSpanCollector
from uipath.eval.mocks import ComponentSimulationConfig, ExampleCall
from uipath.eval.mocks._mock_runtime import (
    clear_execution_context,
    set_execution_context,
)
from uipath.eval.mocks._types import (
    LLMMockingStrategy,
    MockingContext,
    MockitoMockingStrategy,
    ToolSimulation,
    UnknownMockingStrategy,
)

import uipath_codex_sdk.attachment_tools as at
import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    AttachmentStore,
    ToolContext,
    UiPathAttachment,
    analyze_files_tool_from_resource,
    batch_transform_tool_from_resource,
    context_tool_from_resource,
    deep_rag_tool_from_resource,
    escalation_tool_from_resource,
    integration_tool_from_resource,
    ixp_extraction_tool_from_resource,
    process_tool_from_resource,
    uipath_agent_tool,
    uipath_analyze_files_tool,
    uipath_context_tool,
    uipath_deep_rag_attachments_tool,
    uipath_escalation_tool,
    uipath_process_tool,
    vs_escalation_tool_from_resource,
)
from uipath_codex_sdk.batch_transform_tools import BATCH_TRANSFORM_OUTPUT_SCHEMA
from uipath_codex_sdk.runtime import telemetry
from uipath_codex_sdk.runtime import tool_host as host_mod
from uipath_codex_sdk.runtime.errors import UiPathCodexErrorCode
from uipath_codex_sdk.runtime.mocking import (
    MOCKED_ANNOTATION_KEY,
    MockSpec,
    is_simulated,
    mock_spec_of,
)
from uipath_codex_sdk.runtime.session_store import CodexSessionStore
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage
from uipath_codex_sdk.runtime.tool_host import DurableToolHost

from .durable.fakes import FakeDocuments
from .test_analyze_files import PNG
from .test_analyze_files import RESOURCE as ANALYZE_RESOURCE
from .test_batch_transform_attachments import CSV
from .test_batch_transform_attachments import RESOURCE as BATCH_RESOURCE
from .test_context_tools import POLICY_RESOURCE, _with
from .test_deep_rag_attachments import RESOURCE as DEEP_RAG_RESOURCE
from .test_escalation_tools import HITL_RESOURCE
from .test_integration_tools import WEB_SEARCH_RESOURCE
from .test_ixp_tools import IXP_RESOURCE, _vs_resource
from .test_tool_host import FakeTool

DOC_ID = "dddddddd-0000-0000-0000-000000000001"
DOC = {"ID": DOC_ID, "FullName": "invoice.pdf", "MimeType": "application/pdf"}
RESULT_ID = "eeeeeeee-0000-0000-0000-000000000001"


class AddIn(BaseModel):
    a: int
    b: int


class AddOut(BaseModel):
    sum: int


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def llm(*names: str) -> LLMMockingStrategy:
    return LLMMockingStrategy(
        prompt="Simulate the tool.",
        tools_to_simulate=[ToolSimulation(name=n) for n in names],
    )


def mockito(*behaviors: dict[str, Any]) -> MockitoMockingStrategy:
    return MockitoMockingStrategy.model_validate(
        {"type": "mockito", "config": list(behaviors)}
    )


def returns(function: str, *values: Any, kwargs: dict | None = None) -> dict:
    behavior: dict[str, Any] = {
        "function": function,
        "then": [{"type": "return", "value": v} for v in values],
    }
    if kwargs is not None:
        behavior["arguments"] = {"kwargs": kwargs}
    return behavior


@contextlib.contextmanager
def simulating(strategy: Any = None, components: list[Any] | None = None):
    """The mocking ContextVars ``UiPathMockRuntime`` sets around an execution."""
    set_execution_context(
        MockingContext(strategy=strategy, components=components, name="case"),
        ExecutionSpanCollector(),
        "exec-1",
    )
    try:
        yield
    finally:
        clear_execution_context()


def extraction(total: str = "4200") -> dict[str, Any]:
    """An ``ExtractionResponseIXP`` as the mocker returns it (aliases)."""
    return {
        "extractionResult": {
            "DocumentId": "doc-1",
            "ResultsVersion": 1,
            "ResultsDocument": {"Fields": []},
        },
        "projectId": "p1",
        "projectType": "IXP",
        "tag": "live",
        "documentTypeId": "00000000-0000-0000-0000-000000000000",
        "dataProjection": FakeDocuments.projection(total),
    }


@pytest.fixture(autouse=True)
def no_platform(monkeypatch):
    """Any platform call fails the test."""

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the platform was called")

    monkeypatch.setattr(tools_mod, "_uipath_client", refuse)
    monkeypatch.setattr(at, "_uipath_client", refuse)
    monkeypatch.setattr(at, "_gateway_client", refuse)


@pytest.fixture
def llm_answers(monkeypatch):
    """The LLM mocker's structured-output call, answered by tool name."""
    import uipath.eval.mocks._llm_mocker as llm_mocker

    answers: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []

    async def generate(llm_service, messages, *, schema, description, **kw):
        calls.append({"schema": schema, "prompt": messages[0]["content"]})
        name = description.split("'")[1]
        return answers[name]

    monkeypatch.setattr(llm_mocker, "generate_structured_output", generate)
    monkeypatch.setattr(
        llm_mocker,
        "UiPath",
        lambda: type("U", (), {"_config": None, "_execution_context": None})(),
    )
    monkeypatch.setattr(llm_mocker, "UiPathLlmChatService", lambda *a, **k: object())
    answers["__calls__"] = calls
    return answers


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    monkeypatch.setattr(telemetry, "get_tracer", lambda: tracer)
    exporter.tracer = tracer  # type: ignore[attr-defined]
    return exporter


def _by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {s.name: s for s in exporter.get_finished_spans()}


class Served:
    def __init__(self, host: DurableToolHost, url: str, store: CodexSessionStore):
        self.host, self.url, self.store = host, url, store
        self.probes: list[str] = []
        self._id = 0

    async def call(
        self, name: str, args: dict[str, Any], call_id: str | None = "call_1"
    ) -> dict[str, Any]:
        self._id += 1
        params: dict[str, Any] = {"name": name, "arguments": args}
        if call_id is not None:
            params["_meta"] = {"callId": call_id}
        msg = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": params,
        }
        headers = {"Authorization": f"Bearer {self.host.token}"}
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            async with s.post(self.url, json=msg, headers=headers) as r:
                return (await r.json())["result"]


@pytest.fixture
async def serve(tmp_path, monkeypatch):
    """A tool host on a local port, built in the current (mocking) context."""
    cleanups: list[Any] = []
    pushed: list[Any] = []
    monkeypatch.setattr(host_mod, "push_running", lambda span: pushed.append(span))

    async def make(
        tools: list[Any],
        *,
        tracer: Any = None,
        attachments: list[dict[str, Any]] = (),  # type: ignore[assignment]
        agent_input: dict[str, Any] | None = None,
    ) -> Served:
        storage = SqliteResumableStorage(str(tmp_path / f"s{len(cleanups)}.db"))
        store = CodexSessionStore(storage, "rt")
        host = DurableToolHost(
            tools,
            store=store,
            runtime_id="rt",
            context=contextvars.copy_context(),
            tracer=tracer,
            agent_input=agent_input,
            attachments=AttachmentStore(
                [UiPathAttachment.model_validate(a) for a in attachments]
            ),
        )
        app = web.Application()
        app.router.add_route("*", "/mcp", host.handle)
        runner = web.AppRunner(app, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        served = Served(host, f"http://127.0.0.1:{port}/mcp", store)

        async def probe(call_id: str) -> bool:
            served.probes.append(call_id)
            return True

        host.bind(probe)
        cleanups.append((host, runner, storage))
        return served

    make.pushed = pushed  # type: ignore[attr-defined]
    yield make
    for host, runner, storage in cleanups:
        await host.release_all()
        await runner.cleanup()
        await storage.dispose()


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


# --------------------------------------------------------------------------
# 1. the gate: the mockers' own name matching
# --------------------------------------------------------------------------
def test_is_simulated_without_a_context():
    assert is_simulated("a_plus_b") is False


@pytest.mark.parametrize(
    ("strategy", "components", "name", "expected"),
    [
        (llm("Get User"), None, "Get User", True),
        # The LLM mocker matches exactly: is_tool_simulated's "_" → " " would
        # let this through and the mocker would then refuse it.
        (llm("Get User"), None, "Get_User", False),
        (mockito(returns("a_plus_b", 1)), None, "a_plus_b", True),
        (mockito(returns("a_plus_b", 1)), None, "a plus b", False),
        (mockito(returns("other", 1)), None, "a_plus_b", False),
        (None, [ComponentSimulationConfig(component_id="Get_User")], "Get User", True),
        (None, [ComponentSimulationConfig(component_id="Get User")], "Get_User", True),
        (None, [ComponentSimulationConfig(component_id="Get User")], "Get User", True),
        (None, [ComponentSimulationConfig(component_id="Get User")], "Other", False),
    ],
)
def test_is_simulated_matrix(strategy, components, name, expected):
    with simulating(strategy, components):
        assert is_simulated(name) is expected


def test_is_simulated_is_false_when_no_mocker_could_be_built():
    with simulating(UnknownMockingStrategy(type="weird")):
        assert is_simulated("a_plus_b") is False


# --------------------------------------------------------------------------
# 2. mock specs: LangChain's names, schemas and example calls
# --------------------------------------------------------------------------
EXAMPLE = {"id": "ex1", "input": '{"a": 1}', "output": '{"sum": 1}'}


def _escalation_with_examples() -> dict[str, Any]:
    r = json.loads(json.dumps(HITL_RESOURCE))
    r["channels"][0]["properties"]["exampleCalls"] = [EXAMPLE]
    return r


def _ixp_with_examples() -> dict[str, Any]:
    r = json.loads(json.dumps(IXP_RESOURCE))
    r["properties"]["exampleCalls"] = [EXAMPLE]
    return r


def test_process_and_agent_specs():
    tool = uipath_process_tool(
        "Add Numbers",
        tool_name="add",
        input_schema=AddIn,
        output_schema=AddOut,
        example_calls=[ExampleCall.model_validate(EXAMPLE)],
    )
    spec = tool.mock_spec
    assert spec.name == "Add Numbers"  # the process name, not the tool name
    assert spec.output_schema["properties"] == {
        "sum": {"title": "Sum", "type": "integer"}
    }
    assert [e.id for e in spec.example_calls] == ["ex1"]
    assert spec.input_schema == tool.input_schema
    bare = uipath_agent_tool("Helper")
    assert bare.mock_spec.output_schema == {"type": "object", "properties": {}}
    assert bare.mock_spec.example_calls == []


def test_escalation_spec_is_langchains():
    tool = escalation_tool_from_resource(_escalation_with_examples())
    spec = tool.mock_spec
    assert tool.name == "escalate_color" and spec.name == "escalate_color"
    assert spec.output_schema["required"] == ["action", "data"]
    props = spec.output_schema["properties"]
    assert props["action"]["enum"] == ["approve", "reject"]
    assert props["data"]["properties"] == {"Comment": {"type": "string"}}
    assert [e.id for e in spec.example_calls] == ["ex1"]
    coded = uipath_escalation_tool("ExpenseApproval")
    assert coded.mock_spec.name == "escalate_expenseapproval"
    assert coded.mock_spec.output_schema["properties"]["data"] == {"type": "object"}


def test_ixp_and_validation_specs():
    ixp = ixp_extraction_tool_from_resource(_ixp_with_examples())
    assert ixp.name == "Invoice_Extraction"
    spec = ixp.mock_spec
    assert spec.name == "Invoice Extraction"
    assert "dataProjection" in spec.output_schema["properties"]
    assert [e.id for e in spec.example_calls] == ["ex1"]
    vs = vs_escalation_tool_from_resource(_vs_resource())
    vspec = vs.mock_spec
    assert vspec.name == "Validate Invoice" and vspec.example_calls == []
    data = vspec.output_schema["properties"]["data"]
    assert {b.get("type") for b in data["anyOf"]} == {"array", "null"}


def test_context_and_deep_rag_specs():
    search = context_tool_from_resource(POLICY_RESOURCE)
    assert search.mock_spec.name == "Company Policies"
    assert search.mock_spec.output_schema["required"] == ["documents"]
    assert search.mock_spec.example_calls == []
    deep = context_tool_from_resource(
        _with(retrievalMode="DeepRAG", citationMode={"value": "Inline"})
    )
    spec = deep.mock_spec
    assert spec.name == "Company Policies"
    assert set(spec.output_schema["required"]) == {"text", "citations", "deepRagId"}
    attached = deep_rag_tool_from_resource(DEEP_RAG_RESOURCE)
    assert attached.name == "Deep_RAG_v2"
    assert attached.mock_spec.name == "Deep RAG (v2)"
    # An empty resource output schema: DeepRAG's result.
    assert "deepRagId" in attached.mock_spec.output_schema["properties"]
    with_schema = json.loads(json.dumps(DEEP_RAG_RESOURCE, default=str))
    with_schema["outputSchema"] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    own = deep_rag_tool_from_resource(with_schema).mock_spec
    assert own.output_schema["properties"] == {"answer": {"type": "string"}}
    assert uipath_deep_rag_attachments_tool("Contract Research").mock_spec.name == (
        "Contract Research"
    )


def test_batch_integration_and_analyze_specs():
    batch = batch_transform_tool_from_resource(BATCH_RESOURCE)
    assert batch.mock_spec.name == "Batch Transform"
    assert batch.mock_spec.output_schema == BATCH_TRANSFORM_OUTPUT_SCHEMA
    resource = json.loads(json.dumps(WEB_SEARCH_RESOURCE))
    resource["name"] = "Get User (v2)"
    resource["properties"]["exampleCalls"] = [EXAMPLE]
    integration = integration_tool_from_resource(resource)
    assert integration.name == "Get_User_v2"
    assert integration.mock_spec.name == "Get User (v2)"
    assert "formattedResults" in integration.mock_spec.output_schema["properties"]
    assert [e.id for e in integration.mock_spec.example_calls] == ["ex1"]
    analyze = analyze_files_tool_from_resource(ANALYZE_RESOURCE)
    assert analyze.mock_spec.name == "Analyze Files"
    assert analyze.mock_spec.output_schema["properties"] == {
        "analysis": {"type": "string"}
    }
    coded = uipath_analyze_files_tool()
    assert coded.mock_spec.output_schema["required"] == ["analysisResult"]


class CustomTool:
    """A user tool: no mock_spec, so the spec comes from its attributes."""

    name = "lookup"
    description = "Looks a key up."
    input_schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    mocks_in_call = False

    def __init__(self, text: str = "real", attr: dict[str, Any] | None = None) -> None:
        self.text = text
        self.calls = 0
        for k, v in (attr or {}).items():
            setattr(self, k, v)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return dict(args)

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        self.calls += 1
        return self.text


def test_custom_tool_spec_from_attributes():
    spec = mock_spec_of(CustomTool())
    assert spec == MockSpec(
        "lookup",
        "Looks a key up.",
        CustomTool.input_schema,
        {"type": "object", "properties": {}},
        [],
    )
    out = {"type": "object", "properties": {"v": {"type": "string"}}}
    spec = mock_spec_of(
        CustomTool(attr={"display_name": "Lookup Key", "output_json_schema": out})
    )
    assert spec.name == "Lookup Key" and spec.output_schema == out


# --------------------------------------------------------------------------
# 3. the host: a simulated durable call is answered, never started or held
# --------------------------------------------------------------------------
async def test_mocked_durable_call_answers_without_starting(serve):
    tool = FakeTool(error=AssertionError("start must not run"))
    with simulating(mockito(returns("a_plus_b", {"sum": 3}, kwargs={"a": 1, "b": 2}))):
        served = await serve([tool])
    result = await served.call("a_plus_b", {"a": 1, "b": 2})
    assert result == {
        "content": [{"type": "text", "text": '{"sum": 3}'}],
        "isError": False,
    }
    assert tool.started == [] and served.probes == []
    assert served.host.held == {} and served.host.mocking_count == 0
    assert await served.store.get_pending() == {}
    assert served.host.immediate_outcome("call_1") == '{"sum": 3}'
    assert serve.pushed == []


async def test_retry_gets_the_first_answer_and_a_raise_is_a_tool_error(serve):
    behavior = {
        "function": "a_plus_b",
        "then": [
            {"type": "return", "value": {"sum": 3}},
            {"type": "raise", "value": {"_target_": "builtins.ValueError"}},
        ],
    }
    with simulating(mockito(behavior)):
        served = await serve([FakeTool(error=AssertionError("start"))])
    first = await served.call("a_plus_b", {"a": 1, "b": 2}, call_id="c1")
    retry = await served.call("a_plus_b", {"a": 1, "b": 2}, call_id="c1")
    assert first == retry and first["isError"] is False
    second = await served.call("a_plus_b", {"a": 1, "b": 2}, call_id="c2")
    assert second["isError"] is True
    assert _text(second) == "Tool 'a_plus_b' failed: ValueError"
    # mockito arguments that no behavior matches: a tool error too
    with simulating(mockito(returns("a_plus_b", {"sum": 3}, kwargs={"a": 1, "b": 2}))):
        served = await serve([FakeTool(error=AssertionError("start"))])
    other = await served.call("a_plus_b", {"a": 5, "b": 2})
    assert other["isError"] is True and "Called but not expected" in _text(other)


class SlowShaping(FakeTool):
    """A durable tool whose mocked result is shaped only once released."""

    def __init__(self) -> None:
        super().__init__(error=AssertionError("start must not run"))
        self.release = asyncio.Event()

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        await self.release.wait()
        return json.dumps(value)


async def test_retry_while_the_mocked_call_runs_waits_for_it(serve):
    behavior = {
        "function": "a_plus_b",
        "then": [
            {"type": "return", "value": {"sum": 3}},
            {"type": "raise", "value": {"_target_": "builtins.ValueError"}},
        ],
    }
    tool = SlowShaping()
    with simulating(mockito(behavior)):
        served = await serve([tool])
    first = asyncio.create_task(served.call("a_plus_b", {"a": 1, "b": 2}))
    for _ in range(100):
        if served.host.mocking_count:
            break
        await asyncio.sleep(0.01)
    # In flight: counted for the suspend watcher, neither held nor starting.
    assert served.host.mocking_count == 1 and served.host.inflight_count == 1
    assert served.host.held == {} and served.host.starting_count == 0
    retry = asyncio.create_task(served.call("a_plus_b", {"a": 1, "b": 2}))
    await asyncio.sleep(0.1)
    tool.release.set()
    assert await first == await retry
    assert (await first)["isError"] is False  # the mocker was asked once
    assert served.host.mocking_count == 0


async def test_unlisted_durable_tool_is_held_as_before(serve):
    tool = FakeTool()
    with simulating(mockito(returns("someone_else", 1))):
        served = await serve([tool])
    task = asyncio.create_task(served.call("a_plus_b", {"a": 1, "b": 2}))
    for _ in range(100):
        if served.host.held:
            break
        await asyncio.sleep(0.02)
    assert list(served.host.held) == ["call_1"] and served.probes == ["call_1"]
    assert not task.done()
    await served.host.release_all()
    await asyncio.wait_for(task, 5)


async def test_llm_strategy_mocks_a_durable_call(serve, llm_answers):
    llm_answers["a_plus_b"] = {"sum": 42}
    tool = uipath_process_tool(
        "a_plus_b", input_schema=AddIn, output_schema=AddOut, description="Adds."
    )
    with simulating(llm("a_plus_b")):
        served = await serve([tool])
    result = await served.call("a_plus_b", {"a": 40, "b": 2})
    assert result["isError"] is False and json.loads(_text(result)) == {"sum": 42}
    [call] = llm_answers["__calls__"]
    assert call["schema"]["properties"] == {"sum": {"type": "integer"}}  # cleaned
    assert '"kwargs": {"a": 40, "b": 2}' in call["prompt"]


# --------------------------------------------------------------------------
# 4. custom immediate tools are gated by the host; built-ins mock in call
# --------------------------------------------------------------------------
async def test_custom_immediate_tool_is_mocked_by_the_host(serve):
    tool = CustomTool()
    with simulating(mockito(returns("lookup", {"value": "simulated"}))):
        served = await serve([tool])
    result = await served.call("lookup", {"key": "k"})
    assert json.loads(_text(result)) == {"value": "simulated"} and tool.calls == 0
    plain = CustomTool()
    with simulating(mockito(returns("lookup", "plain text"))):
        served = await serve([plain])
    assert _text(await served.call("lookup", {"key": "k"}, call_id=None)) == (
        "plain text"
    )


async def test_tool_that_mocks_in_call_is_not_gated(serve):
    tool = CustomTool(attr={"mocks_in_call": True})
    with simulating(mockito(returns("lookup", {"value": "simulated"}))):
        served = await serve([tool])
    result = await served.call("lookup", {"key": "k"})
    assert _text(result) == "real" and tool.calls == 1


# --------------------------------------------------------------------------
# 5. spans: low-code's "Simulated result" child, nothing held
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("factory", "span_type"),
    [
        (uipath_process_tool, "processTool"),
        (uipath_agent_tool, "agentTool"),
    ],
)
async def test_mocked_process_spans(serve, spans, factory, span_type):
    tool = factory("a_plus_b", input_schema=AddIn)
    with simulating(mockito(returns("a_plus_b", {"sum": 3}))):
        served = await serve([tool], tracer=spans.tracer)
    await served.call("a_plus_b", {"a": 1, "b": 2})
    by_name = _by_name(spans)
    outer = by_name["Tool call - a_plus_b"]
    child = by_name["Simulated result: a_plus_b"]
    assert json.loads(outer.attributes["output.value"]) == {"content": '{"sum": 3}'}
    assert outer.attributes["span_type"] == "toolCall"
    assert MOCKED_ANNOTATION_KEY not in outer.attributes
    assert child.attributes["span_type"] == span_type
    assert child.attributes[MOCKED_ANNOTATION_KEY] is True
    assert json.loads(child.attributes["output"]) == {"sum": 3}
    assert child.parent.span_id == outer.context.span_id
    assert serve.pushed == []


async def test_custom_tool_marker_sits_on_the_tool_call(serve, spans):
    with simulating(mockito(returns("lookup", {"value": "v"}))):
        served = await serve([CustomTool()], tracer=spans.tracer)
    await served.call("lookup", {"key": "k"})
    [span] = spans.get_finished_spans()
    assert span.name == "Tool call - lookup"  # never renamed
    assert span.attributes[MOCKED_ANNOTATION_KEY] is True
    assert span.attributes["tool.server"] == "uipath_tools"


# --------------------------------------------------------------------------
# 6. shaping: the durable tools' result shapes and side effects
# --------------------------------------------------------------------------
async def test_mocked_escalation_approve(serve, spans):
    tool = escalation_tool_from_resource(HITL_RESOURCE)
    answer = {"action": "approve", "data": {"Comment": "ok", "Content": "dropped"}}
    with simulating(mockito(returns("escalate_color", answer))):
        served = await serve([tool], tracer=spans.tracer)
    result = await served.call("escalate_color", {"Content": "blue"})
    assert result["isError"] is False
    assert json.loads(_text(result)) == {
        "status": "completed",
        "outcome": "approve",
        "output": {"Comment": "ok"},
        "task_id": None,
        "assigned_to": None,
    }
    child = _by_name(spans)["Simulated result: HITL App"]
    assert child.attributes["span_type"] == "escalationTool"
    assert json.loads(child.attributes["result"]) == {
        "outcome": "approve",
        "output": {"Comment": "ok"},
    }
    assert not served.host.fatal.is_set()


async def test_mocked_escalation_matches_on_its_prefilled_fields(serve):
    # The mocker sees the model's arguments with the hidden fields merged in,
    # as langchain's StaticArgsHandler does: a behavior on them matches.
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        input_schema={
            "type": "object",
            "properties": {
                "Amount": {"type": "number"},
                "Department": {"type": "string"},
                "Requester": {"type": "string"},
            },
        },
        static_args={"Department": "Finance"},
        args_from_input={"Requester": "user.name"},
    )
    expected = {"Amount": 120, "Department": "Finance", "Requester": "Ana"}
    answer = {"action": "approve", "data": {}}
    with simulating(mockito(returns(tool.name.lower(), answer, kwargs=expected))):
        served = await serve([tool], agent_input={"user": {"name": "Ana"}})
    assert "Department" not in tool.input_schema["properties"]
    result = await served.call(tool.name, {"Amount": 120})
    assert result["isError"] is False, _text(result)
    assert json.loads(_text(result))["outcome"] == "approve"


async def test_mocked_escalation_unresolved_argument(serve, spans):
    # An args_from_input field the agent input lacks: the mocker sees the
    # model's value; a required one neither filled is a tool error.
    tool = uipath_escalation_tool(
        "ExpenseApproval",
        input_schema={
            "type": "object",
            "properties": {
                "Amount": {"type": "number"},
                "Requester": {"type": "string"},
            },
            "required": ["Amount", "Requester"],
        },
        args_from_input={"Requester": "user.name"},
    )
    expected = {"Amount": 120, "Requester": "Bo"}
    answer = {"action": "approve", "data": {}}
    with simulating(mockito(returns(tool.name.lower(), answer, kwargs=expected))):
        served = await serve([tool], tracer=spans.tracer, agent_input={})
    assert tool.input_schema["required"] == ["Amount"]
    result = await served.call(tool.name, {"Amount": 120, "Requester": "Bo"})
    assert result["isError"] is False, _text(result)
    outer = _by_name(spans)[f"Tool call - {tool.name}"]
    assert json.loads(outer.attributes["input"]) == expected
    result = await served.call(tool.name, {"Amount": 120}, call_id="call_2")
    assert result["isError"] is True
    assert "missing required argument(s): Requester" in _text(result)
    # Refused before the escalationTool child opens, as a real start.
    names = [s.name for s in spans.get_finished_spans()]
    assert names.count(f"Tool call - {tool.name}") == 2
    assert "ExpenseApproval" not in names


def _with_attachment_argument(resource: dict[str, Any], key: str) -> dict[str, Any]:
    """``resource`` with ``key`` read from the agent input's ``document``."""
    data = json.loads(json.dumps(resource))
    data["argumentProperties"] = {
        key: {"variant": "argument", "argumentPath": "$.document", "isSensitive": False}
    }
    return data


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
async def test_mocked_attachment_tool_sees_the_merged_attachment(
    serve, spans, resource, tool_name, mock_name, doc, answer
):
    builder = (
        deep_rag_tool_from_resource
        if resource is DEEP_RAG_RESOURCE
        else batch_transform_tool_from_resource
    )
    tool = builder(_with_attachment_argument(resource, "attachment"))
    merged = {"query": "When?", "attachment": doc}
    with simulating(mockito(returns(mock_name, answer, kwargs=merged))):
        served = await serve(
            [tool],
            tracer=spans.tracer,
            attachments=[doc],
            agent_input={"document": doc},
        )
    result = await served.call(tool_name, {"query": "When?"})
    assert result["isError"] is False, _text(result)
    assert json.loads(_text(result)) == answer
    outer = _by_name(spans)[f"Tool call - {tool_name}"]
    assert json.loads(outer.attributes["input"]) == merged
    # The attachment taken from the agent input is the span's too.
    listed = json.loads(outer.attributes["attachments"])
    assert [a["id"] for a in listed if a["direction"] == 1] == [doc["ID"]]
    # The agent input has none: the model's (host-resolved) attachment.
    with simulating(mockito(returns(mock_name, answer, kwargs=merged))):
        served = await serve(
            [tool], tracer=spans.tracer, attachments=[doc], agent_input={}
        )
    result = await served.call(
        tool_name, {"query": "When?", "attachment": {"ID": doc["ID"]}}
    )
    assert result["isError"] is False, _text(result)
    spans.clear()
    result = await served.call(tool_name, {"query": "When?"}, call_id="call_2")
    assert result["isError"] is True
    assert "missing required argument(s): attachment" in _text(result)
    # Refused before the tool's child span opens, as a real start.
    assert [s.name for s in spans.get_finished_spans()] == [f"Tool call - {tool_name}"]


async def test_mocked_analyze_files_sees_the_merged_attachments(spans):
    tool = analyze_files_tool_from_resource(
        _with_attachment_argument(ANALYZE_RESOURCE, "attachments")
    )
    store = AttachmentStore([UiPathAttachment.model_validate(PNG)])
    ctx = ToolContext("c1", "rt", "Analyze_Files", {"document": PNG}, store, "gpt")
    behavior = returns(
        "Analyze Files",
        {"analysis": "42"},
        kwargs={"attachments": [PNG], "analysisTask": "Total?"},
    )
    with simulating(mockito(behavior)):
        text = await tool.call({"analysisTask": "Total?"}, ctx)
        assert json.loads(text) == {"analysis": "42"}
        # unresolved: the model's attachments
        unresolved = ToolContext("c1", "rt", "Analyze_Files", {}, store, "gpt")
        text = await tool.call(
            {"analysisTask": "Total?", "attachments": [PNG]}, unresolved
        )
        assert json.loads(text) == {"analysis": "42"}
        with pytest.raises(at.UiPathToolError, match="missing required"):
            await tool.call({"analysisTask": "Total?"}, unresolved)


async def test_mocked_process_matches_on_its_prefilled_fields(serve):
    # As for an escalation: the mocker sees the job input the call would
    # start with, so a behavior on the pre-filled fields matches.
    tool = uipath_process_tool(
        "PlaceOrder",
        input_schema={
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "channel": {"type": "string"},
                "customer": {"type": "string"},
            },
            "required": ["sku", "channel"],
        },
        static_args={"channel": "codex"},
        args_from_input={"customer": "user.email"},
    )
    expected = {"sku": "A-1", "channel": "codex", "customer": "ana@example.com"}
    with simulating(mockito(returns("PlaceOrder", {"ok": True}, kwargs=expected))):
        served = await serve([tool], agent_input={"user": {"email": "ana@example.com"}})
    result = await served.call(tool.name, {"sku": "A-1"})
    assert result["isError"] is False, _text(result)
    assert json.loads(_text(result)) == {"ok": True}


async def test_mocked_process_span_arguments_are_merged(serve, spans):
    from .test_process_tools import AGENT_INPUT, INVOICE

    tool = process_tool_from_resource(INVOICE)
    with simulating(mockito(returns("Create Invoice", {"invoiceId": "I-1"}))):
        served = await serve([tool], tracer=spans.tracer, agent_input=AGENT_INPUT)
    result = await served.call("Create_Invoice", {"amount": 10})
    assert result["isError"] is False, _text(result)
    merged = {
        "amount": 10,
        "currency": "EUR",
        "apiKey": "s3cr3t",  # isSensitive: clear text, as in low-code
        "customerId": "C1",
        "memo": "Invoice for Ada",
        "tags": ["agent", "EU"],
    }
    by_name = _by_name(spans)
    outer = by_name["Tool call - Create_Invoice"]
    assert json.loads(outer.attributes["input"]) == merged
    assert outer.attributes["toolType"] == "Process"
    child = by_name["Simulated result: CreateInvoice"]
    assert child.attributes["span_type"] == "processTool"
    assert json.loads(child.attributes["arguments"]) == merged


async def test_mocked_flow_has_no_process_span(serve, spans):
    # low-code opens no processTool span for a Flow: the marker sits on the
    # Tool call, whose toolType is low-code's Integration fallback.
    from .test_process_tools import _resource

    tool = process_tool_from_resource(_resource(type="flow", argumentProperties={}))
    with simulating(mockito(returns("Create Invoice", {"invoiceId": "I-1"}))):
        served = await serve([tool], tracer=spans.tracer)
    await served.call(
        "Create_Invoice",
        {"amount": 1, "customerId": "C", "currency": "EUR", "apiKey": "k"},
    )
    [span] = spans.get_finished_spans()
    assert span.name == "Tool call - Create_Invoice"
    assert span.attributes["toolType"] == "Integration"
    assert span.attributes[MOCKED_ANNOTATION_KEY] is True


class CustomerIn(BaseModel):
    customer_id: str
    amount: int


@pytest.mark.parametrize("mocked", [False, True])
async def test_invalid_merged_input_is_a_tool_error(serve, spans, mocked):
    # The agent input's value fails the model once merged: the call is
    # answered with a tool error, no job starts (the platform is refused)
    # and the Tool call span ends with the error.
    tool = uipath_process_tool(
        "CreateInvoice",
        input_schema=CustomerIn,
        args_from_input={"customer_id": "customer.id"},
    )
    strategy = mockito(returns("CreateInvoice", {"ok": True})) if mocked else None
    with simulating(strategy) if mocked else contextlib.nullcontext():
        served = await serve(
            [tool], tracer=spans.tracer, agent_input={"customer": {"id": 5}}
        )
    result = await served.call("CreateInvoice", {"amount": 1})
    assert result["isError"] is True
    assert "Invalid arguments for tool 'CreateInvoice'" in _text(result)
    finished = spans.get_finished_spans()
    assert [s.name for s in finished] == ["Tool call - CreateInvoice"]
    assert finished[0].status.status_code.name == "ERROR"
    assert json.loads(finished[0].attributes["input"]) == {
        "amount": 1,
        "customer_id": 5,
    }
    assert await served.store.get_pending() == {}
    assert serve.pushed == []


async def test_unresolved_required_argument_is_a_tool_error(serve, spans):
    # A resource's agent-input field is optional for the model; when the
    # agent input has no value either, the model is asked to fill it.
    from .test_process_tools import INVOICE

    tool = process_tool_from_resource(INVOICE)
    served = await serve([tool], tracer=spans.tracer, agent_input={})
    result = await served.call("Create_Invoice", {"amount": 10})
    assert result["isError"] is True
    assert "missing required argument(s): customerId" in _text(result)
    [span] = spans.get_finished_spans()
    assert span.status.status_code.name == "ERROR"


@pytest.mark.parametrize(
    "answer",
    [
        {"action": "reject", "data": {"Comment": "no"}},  # mapped to "end"
        {"action": "approve", "data": {}, "is_deleted": True},
    ],
)
async def test_mocked_escalation_that_ends_the_run(serve, spans, answer):
    tool = escalation_tool_from_resource(HITL_RESOURCE)
    with simulating(mockito(returns("escalate_color", answer))):
        served = await serve([tool], tracer=spans.tracer)
    result = await served.call("escalate_color", {"Content": "blue"})
    assert result["isError"] is True
    assert served.host.fatal.is_set()
    error = served.host.fatal_error
    assert error is not None
    assert error.error_info.code.endswith(
        UiPathCodexErrorCode.TERMINATION_ESCALATION_REJECTED.value
    )
    outer = _by_name(spans)["Tool call - escalate_color"]
    assert outer.status.status_code.name == "ERROR"


async def test_mocked_ixp_keeps_the_extraction_for_validation(serve, spans):
    from uipath.platform.documents import ExtractionResponseIXP

    ixp = ixp_extraction_tool_from_resource(IXP_RESOURCE)
    vs = vs_escalation_tool_from_resource(_vs_resource())
    with simulating(
        mockito(
            returns("Invoice Extraction", extraction()),
            returns("Validate Invoice", {"data": FakeDocuments.projection("4250")}),
        )
    ):
        served = await serve([ixp, vs], tracer=spans.tracer, attachments=[DOC])
    # validation first: no extraction yet → the "call it first" error, no spans
    early = await served.call("Validate_Invoice", {}, call_id="v0")
    assert early["isError"] is True and "Call it first" in _text(early)
    assert spans.get_finished_spans() == ()
    result = await served.call("Invoice_Extraction", {"attachment": {"ID": DOC_ID}})
    assert result["isError"] is False
    assert json.loads(_text(result)) == FakeDocuments.projection()
    stored = served.host.results.get("Invoice Extraction")
    assert ExtractionResponseIXP.model_validate(stored).data_projection is not None
    assert stored["projectId"] == "p1"  # alias-keyed, as LangChain's round trip
    child = _by_name(spans)["Simulated result: Invoice Extraction"]
    assert child.attributes["span_type"] == "ixpTool"
    validated = await served.call("Validate_Invoice", {}, call_id="v1")
    assert json.loads(_text(validated)) == {"data": FakeDocuments.projection("4250")}
    # the vsEscalationTool child is named after the app, as a real one
    child = _by_name(spans)["Simulated result: HITL App"]
    assert child.attributes["span_type"] == "vsEscalationTool"


async def test_malformed_ixp_result_is_a_tool_error(serve):
    ixp = ixp_extraction_tool_from_resource(IXP_RESOURCE)
    with simulating(mockito(returns("Invoice Extraction", {"nope": 1}))):
        served = await serve([ixp], attachments=[DOC])
    result = await served.call("Invoice_Extraction", {"attachment": {"ID": DOC_ID}})
    assert result["isError"] is True
    assert "Unexpected IXP extraction result" in _text(result)
    assert served.host.results.get("Invoice Extraction") is None


async def test_mocked_batch_transform_result_becomes_known(serve, spans):
    tool = batch_transform_tool_from_resource(BATCH_RESOURCE)
    answer = {
        "result": {"ID": RESULT_ID, "FullName": "output.csv", "MimeType": "text/csv"}
    }
    with simulating(mockito(returns("Batch Transform", answer))):
        served = await serve([tool], tracer=spans.tracer, attachments=[CSV])
    result = await served.call(
        "Batch_Transform", {"attachment": {"ID": CSV["ID"]}, "query": "Add topics."}
    )
    assert json.loads(_text(result)) == answer
    assert served.host.attachments.get(RESULT_ID) is not None
    child = _by_name(spans)["Simulated result: Batch_Transform"]
    assert child.attributes["retrieval_mode"] == "BatchTransform"


async def test_mocked_deep_rag_passes_the_value_through(serve, spans):
    tool = context_tool_from_resource(
        _with(retrievalMode="DeepRAG", citationMode={"value": "Inline"})
    )
    answer = {"text": "Due in May.", "citations": [], "deepRagId": "d-1"}
    with simulating(mockito(returns("Company Policies", answer))):
        served = await serve([tool], tracer=spans.tracer)
    result = await served.call("Company_Policies", {"query": "When?"})
    assert json.loads(_text(result)) == answer
    child = _by_name(spans)["Simulated result: Company Policies"]
    assert child.attributes["span_type"] == "contextGroundingTool"
    assert child.attributes["query"] == "When?"


# --------------------------------------------------------------------------
# 7. the built-in immediate tools mock inside call, under their span
# --------------------------------------------------------------------------
CTX = ToolContext(call_id="c1", runtime_id="rt", tool_name="t")


async def test_integration_tool_is_mocked_by_resource_name(spans, llm_answers):
    resource = json.loads(json.dumps(WEB_SEARCH_RESOURCE))
    resource["name"] = "Get User (v2)"
    tool = integration_tool_from_resource(resource)
    llm_answers["Get User (v2)"] = {"formattedResults": "simulated"}
    with simulating(llm("Get User (v2)")):
        text = await tool.call({"query": "x"}, CTX)
    assert json.loads(text) == {"formattedResults": "simulated"}
    child = _by_name(spans)["Simulated result: Get User (v2)"]
    assert child.attributes["span_type"] == "integrationTool"
    assert child.attributes[MOCKED_ANNOTATION_KEY] is True
    # The model-facing name is not the mock name: the real call runs.
    with (
        simulating(llm("Get_User_v2")),
        pytest.raises(AssertionError, match="platform"),
    ):
        await tool.call({"query": "x"}, CTX)


async def test_context_search_is_mocked_without_a_retriever_span(spans):
    tool = uipath_context_tool("Company Policies", index_name="i")
    docs = {"documents": [{"metadata": {}, "page_content": "Travel by train."}]}
    with simulating(mockito(returns("Company Policies", docs))):
        text = await tool.call({"query": "travel"}, CTX)
    assert json.loads(text) == docs
    names = [s.name for s in spans.get_finished_spans()]
    assert names == ["Simulated result: Company Policies"]
    [span] = spans.get_finished_spans()
    assert span.attributes["span_type"] == "contextGroundingTool"
    assert json.loads(span.attributes["output"]) == docs


def _analyze_ctx() -> ToolContext:
    store = AttachmentStore([UiPathAttachment.model_validate(PNG)])
    return ToolContext("c1", "rt", "Analyze_Files", None, store, "gpt")


async def test_analyze_files_is_mocked_with_the_full_attachments(spans):
    tool = uipath_analyze_files_tool()
    # The mocker sees the attachment as the full JobAttachment (langchain's).
    behavior = returns(
        "Analyze Files",
        {"analysisResult": "42"},
        kwargs={"attachments": [PNG], "analysisTask": "Total?"},
    )
    with simulating(mockito(behavior)):
        text = await tool.call(
            {"attachments": [{"ID": PNG["ID"]}], "analysisTask": "Total?"},
            _analyze_ctx(),
        )
    assert json.loads(text) == {"analysisResult": "42"}
    [span] = spans.get_finished_spans()
    assert span.name == "Simulated result: Analyze Files"
    assert span.attributes["span_type"] == "toolCall"


@pytest.mark.parametrize(
    "args",
    [
        {"attachments": [], "analysisTask": "Total?"},  # the no-files answer
        {"attachments": [PNG]},  # no analysisTask
        {"analysisTask": "Total?"},  # no attachments
    ],
)
async def test_analyze_files_argument_checks_are_inside_the_mock(spans, args):
    """As langchain's mocked tool_fn: the mocker answers before the checks."""
    tool = uipath_analyze_files_tool()
    with simulating(mockito(returns("Analyze Files", {"analysisResult": "42"}))):
        text = await tool.call(args, _analyze_ctx())
    assert json.loads(text) == {"analysisResult": "42"}
    [span] = spans.get_finished_spans()
    assert span.name == "Simulated result: Analyze Files"
    # not simulated: the checks still answer
    with pytest.raises(Exception, match="analysisTask"):
        await uipath_analyze_files_tool().call({"attachments": [PNG]}, _analyze_ctx())


async def test_context_search_mocker_gets_langchains_arguments():
    docs = {"documents": [{"metadata": {}, "page_content": "Travel by train."}]}
    tool = uipath_context_tool(
        "Company Policies", index_name="i", folder_path_prefix="hr"
    )
    with simulating(
        mockito(returns("Company Policies", docs, kwargs={"query": "travel"}))
    ):
        assert json.loads(await tool.call({"query": "travel"}, CTX)) == docs
    # A static query: no arguments.
    static = uipath_context_tool("Company Policies", index_name="i", query="travel")
    with simulating(mockito(returns("Company Policies", docs, kwargs={}))):
        assert json.loads(await static.call({}, CTX)) == docs
    # A query set from the agent input: that query (langchain applies it).
    from_input = uipath_context_tool(
        "Company Policies", index_name="i", query_from_input="topic"
    )
    ctx = ToolContext("c1", "rt", "t", {"topic": "travel"})
    with simulating(
        mockito(returns("Company Policies", docs, kwargs={"query": "travel"}))
    ):
        assert json.loads(await from_input.call({"query": "other"}, ctx)) == docs


async def test_integration_mocker_gets_the_activity_input():
    tool = integration_tool_from_resource(WEB_SEARCH_RESOURCE)
    payload = {"query": "x", "provider": "GoogleCustomSearch"}  # static merged
    answer = {"formattedResults": "simulated"}
    with simulating(mockito(returns("Web Search", answer, kwargs=payload))):
        assert json.loads(await tool.call({"query": "x"}, CTX)) == answer


async def test_mocked_deep_rag_on_an_attachment(serve, spans):
    tool = deep_rag_tool_from_resource(DEEP_RAG_RESOURCE)
    answer = {"text": "Due in May.", "citations": [], "deepRagId": "d-1"}
    with simulating(mockito(returns("Deep RAG (v2)", answer))):
        served = await serve([tool], tracer=spans.tracer, attachments=[DOC])
    result = await served.call(
        "Deep_RAG_v2", {"attachment": {"ID": DOC_ID}, "query": "When?"}
    )
    assert result["isError"] is False and json.loads(_text(result)) == answer
    assert served.probes == [] and served.host.held == {}
    by_name = _by_name(spans)
    child = by_name["Simulated result: Deep_RAG_v2"]  # as a real call's span
    assert child.attributes["span_type"] == "toolCall"
    assert child.attributes["citation_mode"] == "Inline"
    assert child.attributes[MOCKED_ANNOTATION_KEY] is True
    assert json.loads(child.attributes["output"]) == answer
    assert child.parent.span_id == by_name["Tool call - Deep_RAG_v2"].context.span_id


async def test_simulated_component_answers_a_durable_call(serve, monkeypatch):
    import uipath.eval.mocks._simulate_component_mocker as component_mocker

    payloads: list[dict[str, Any]] = []

    class Service:
        async def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
            payloads.append(payload)
            return {"status": 1, "simulatedOutput": {"sum": 7}}

    monkeypatch.setattr(
        component_mocker, "_create_simulate_component_service", lambda: Service()
    )
    tool = FakeTool(error=AssertionError("start must not run"))
    # "a plus b" matches "a_plus_b" for both the gate and the mocker.
    with simulating(components=[ComponentSimulationConfig(component_id="a plus b")]):
        served = await serve([tool])
    result = await served.call("a_plus_b", {"a": 3, "b": 4})
    assert result["isError"] is False and json.loads(_text(result)) == {"sum": 7}
    assert tool.started == [] and served.probes == []
    [payload] = payloads
    assert payload["componentId"] == "a plus b"
    assert payload["input"] == {"args": [], "kwargs": {"a": 3, "b": 4}}


async def test_runtime_error_from_a_mocked_result_fails_the_run(serve, spans):
    from uipath.runtime.errors import UiPathErrorCategory

    from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError

    error = UiPathCodexRuntimeError(
        UiPathCodexErrorCode.EXECUTION_FAILED,
        "Bad result",
        "no",
        UiPathErrorCategory.USER,
    )

    class Failing(FakeTool):
        async def mocked_output(
            self, args: dict[str, Any], value: Any, ctx: Any
        ) -> str:
            raise error

    with simulating(mockito(returns("a_plus_b", {"sum": 3}))):
        served = await serve(
            [Failing(error=AssertionError("start"))], tracer=spans.tracer
        )
    result = await served.call("a_plus_b", {"a": 1, "b": 2})
    assert result["isError"] is True and _text(result) == "Tool 'a_plus_b' failed: no"
    assert served.host.fatal.is_set() and served.host.fatal_error is error
    outer = _by_name(spans)["Tool call - a_plus_b"]
    assert outer.status.status_code.name == "ERROR"


def test_mocks_in_call_flags():
    assert integration_tool_from_resource(WEB_SEARCH_RESOURCE).mocks_in_call is True
    assert uipath_context_tool("c", index_name="i").mocks_in_call is True
    assert uipath_analyze_files_tool().mocks_in_call is True
    assert getattr(uipath_process_tool("p"), "mocks_in_call", False) is False
