"""DurableToolHost: the MCP endpoint durable tools are served from."""

from __future__ import annotations

import asyncio
import contextvars
import json
from typing import Any

import pytest
from aiohttp import ClientSession, ClientTimeout, web
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath_codex_sdk import (
    DurableToolContext,
    UiPathDurableToolError,
    UiPathToolError,
    uipath_process_tool,
)
from uipath_codex_sdk.runtime.session_store import STATE_STARTED, CodexSessionStore
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage
from uipath_codex_sdk.runtime.tool_host import (
    LONG_RUNNING_TAG,
    UNKNOWN_OUTCOME_OUTPUT,
    DurableToolHost,
)

from ._typing import attrs

REQUEST_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_var", default="unset"
)


class FakeTool:
    name = "a_plus_b"
    description = "Adds a and b."
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    def __init__(
        self, error: Exception | None = None, gate: asyncio.Event | None = None
    ) -> None:
        self.error = error
        self.gate = gate
        self.started: list[tuple[dict[str, Any], DurableToolContext, str]] = []
        self.spans: list[Any] = []

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return uipath_process_tool("x", input_schema=self.input_schema).validate_args(
            args
        )

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        self.started.append((args, ctx, REQUEST_VAR.get()))
        self.spans.append(trace.get_current_span())
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        from uipath.platform.common import WaitJobRaw
        from uipath.platform.orchestrator import Job

        job = Job.model_validate(
            {"Id": 1, "Key": "job-1", "State": "Pending", "FolderKey": "fk"}
        )
        return WaitJobRaw(job=job, process_folder_key="fk")

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        return "done"


class Harness:
    def __init__(
        self, host: DurableToolHost, url: str, store: CodexSessionStore
    ) -> None:
        self.host, self.url, self.store = host, url, store
        self._id = 0

    async def rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        token: str | None = None,
        timeout: float = 10,
        notify: bool = False,
    ) -> tuple[int, Any]:
        self._id += 1
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        if not notify:
            msg["id"] = self._id
        headers = {"Authorization": f"Bearer {token or self.host.token}"}
        async with ClientSession(timeout=ClientTimeout(total=timeout)) as s:
            async with s.post(self.url, json=msg, headers=headers) as r:
                body = await r.json() if r.content_type == "application/json" else None
                return r.status, body

    async def call(self, args: dict[str, Any], call_id: str | None = "call_1", **kw):
        params: dict[str, Any] = {"name": "a_plus_b", "arguments": args}
        if call_id is not None:
            params["_meta"] = {"callId": call_id}
        return await self.rpc("tools/call", params, **kw)


@pytest.fixture
async def make_harness(tmp_path):
    cleanups: list[Any] = []

    async def make(
        tool: FakeTool,
        probe_result: bool = True,
        tag: bool = True,
        tracer: trace.Tracer | None = None,
    ) -> Harness:
        storage = SqliteResumableStorage(str(tmp_path / f"s{len(cleanups)}.db"))
        store = CodexSessionStore(storage, "rt")
        REQUEST_VAR.set("from-run-context")
        host = DurableToolHost(
            [tool],
            store=store,
            runtime_id="rt",
            context=contextvars.copy_context(),
            tag_descriptions=tag,
            tracer=tracer,
        )
        REQUEST_VAR.set("unset")

        async def probe(call_id: str) -> bool:
            return probe_result

        host.bind(probe)
        app = web.Application()
        app.router.add_route("*", "/mcp", host.handle)
        runner = web.AppRunner(app, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        cleanups.append((host, runner, storage))
        return Harness(host, f"http://127.0.0.1:{port}/mcp", store)

    yield make
    for host, runner, storage in cleanups:
        await host.release_all()
        await runner.cleanup()
        await storage.dispose()


async def test_protocol_basics(make_harness):
    h = await make_harness(FakeTool())
    async with ClientSession() as s:
        async with s.get(h.url) as r:
            assert r.status == 405
    status, _ = await h.rpc("tools/list", token="wrong")
    assert status == 401
    status, body = await h.rpc("initialize", {"protocolVersion": "2025-03-26"})
    assert status == 200
    assert body["result"]["protocolVersion"] == "2025-03-26"
    assert "tools" in body["result"]["capabilities"]
    status, _ = await h.rpc("notifications/initialized", notify=True)
    assert status == 202
    status, body = await h.rpc("tools/list")
    [tool] = body["result"]["tools"]
    assert tool["name"] == "a_plus_b"
    assert tool["description"] == f"{LONG_RUNNING_TAG} Adds a and b."
    assert tool["inputSchema"]["required"] == ["a", "b"]
    status, body = await h.rpc("ping")
    assert body["result"] == {}
    status, body = await h.rpc("resources/list")
    assert body["error"]["code"] == -32601


async def test_untagged_descriptions(make_harness):
    h = await make_harness(FakeTool(), tag=False)
    _, body = await h.rpc("tools/list")
    assert body["result"]["tools"][0]["description"] == "Adds a and b."


async def test_call_is_held_and_recorded(make_harness):
    tool = FakeTool()
    h = await make_harness(tool)
    task = asyncio.create_task(h.call({"a": 1, "b": 2}))
    for _ in range(100):
        if h.host.held:
            break
        await asyncio.sleep(0.02)
    assert list(h.host.held) == ["call_1"]
    assert h.host.starting_count == 0
    assert not task.done()  # never answered
    [(args, ctx, seen_var)] = tool.started
    assert args == {"a": 1, "b": 2}
    assert ctx.call_id == "call_1" and ctx.runtime_id == "rt"
    # started inside the context captured when the run began
    assert seen_var == "from-run-context"
    pending = await h.store.get_pending()
    assert pending["call_1"].state == STATE_STARTED
    assert pending["call_1"].value_type == "WaitJobRaw"
    assert pending["call_1"].suspend_value().job.key == "job-1"
    await h.host.release_all()
    status, body = await asyncio.wait_for(task, 5)
    assert body["result"]["isError"] is True


@pytest.mark.parametrize(
    ("args", "call_id", "expected"),
    [
        ({"a": 1}, "call_1", "Invalid arguments"),
        ({"a": 1, "b": 2}, None, "no call id"),
    ],
)
async def test_rejected_calls_answer_immediately(make_harness, args, call_id, expected):
    tool = FakeTool()
    h = await make_harness(tool)
    status, body = await h.call(args, call_id=call_id)
    assert body["result"]["isError"] is True
    assert expected in body["result"]["content"][0]["text"]
    assert not tool.started and not h.host.held


async def test_unknown_tool(make_harness):
    h = await make_harness(FakeTool())
    _, body = await h.rpc(
        "tools/call", {"name": "nope", "arguments": {}, "_meta": {"callId": "c"}}
    )
    assert body["result"]["isError"] is True


async def test_call_not_in_rollout_is_not_started(make_harness):
    tool = FakeTool()
    h = await make_harness(tool, probe_result=False)
    _, body = await h.call({"a": 1, "b": 2})
    assert body["result"]["isError"] is True
    assert "not recorded" in body["result"]["content"][0]["text"]
    assert not tool.started
    assert await h.store.get_pending() == {}


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UiPathDurableToolError("Could not find process"), "Could not find process"),
        (RuntimeError("kaput"), "failed to start: kaput"),
    ],
)
async def test_start_failure_is_reported_to_the_model(make_harness, error, expected):
    h = await make_harness(FakeTool(error=error))
    _, body = await h.call({"a": 1, "b": 2})
    assert body["result"]["isError"] is True
    assert expected in body["result"]["content"][0]["text"]
    assert not h.host.held
    assert await h.store.get_pending() == {}  # the intent record is removed


async def _until(predicate, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


async def test_start_runs_under_a_tool_span(make_harness):
    """The job's ParentSpanId comes from the current span: the Tool call span.

    A started job leaves the span open; its identity is saved with the call.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tool = FakeTool()
    h = await make_harness(tool, tracer=provider.get_tracer("test"))
    task = asyncio.create_task(h.call({"a": 1, "b": 2}))
    await _until(lambda: bool(h.host.held))
    assert exporter.get_finished_spans() == ()
    [current] = tool.spans
    assert current.is_recording()
    [call] = (await h.store.get_pending()).values()
    span = call.span
    assert span is not None
    assert span["name"] == "Tool call - a_plus_b"
    assert span["span_id"] == format(current.get_span_context().span_id, "016x")
    assert span["start_time_ns"] == current.start_time
    attributes = span["attributes"]
    assert attributes["call_id"] == "call_1"
    assert attributes["span_type"] == "toolCall"
    assert attributes["tool.name"] == "a_plus_b"
    assert attributes["input"] == '{"a": 1, "b": 2}'
    assert json.loads(attributes["output"]) == {
        "status": "started",
        "job_key": "job-1",
        "job_state": "Pending",
    }
    await h.host.release_all()
    await asyncio.wait_for(task, 5)


async def test_failed_start_ends_its_tool_span(make_harness):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tool = FakeTool(error=RuntimeError("boom"))
    h = await make_harness(tool, tracer=provider.get_tracer("test"))
    status, body = await h.call({"a": 1, "b": 2})
    assert status == 200 and body["result"]["isError"]
    [span] = exporter.get_finished_spans()
    assert span.name == "Tool call - a_plus_b"
    assert span.status.status_code.name == "ERROR"
    assert await h.store.get_pending() == {}


async def test_duplicate_call_while_starting_starts_one_job(make_harness):
    """A retried tools/call for a call id whose start is in flight is held;
    it never starts a second job."""
    gate = asyncio.Event()
    tool = FakeTool(gate=gate)
    h = await make_harness(tool)
    first = asyncio.create_task(h.call({"a": 1, "b": 2}))
    await _until(lambda: h.host.starting_count == 1)
    second = asyncio.create_task(h.call({"a": 1, "b": 2}))
    await asyncio.sleep(0.2)
    assert len(tool.started) == 1
    gate.set()
    await _until(lambda: bool(h.host.held))
    await asyncio.sleep(0.1)
    assert len(tool.started) == 1
    assert not first.done() and not second.done()
    assert list(await h.store.get_pending()) == ["call_1"]
    await h.host.release_all()
    await asyncio.wait_for(asyncio.gather(first, second), 5)


async def test_call_after_close_starts_nothing(make_harness):
    """Once the runtime is stopping codex, a late call is answered at once."""
    tool = FakeTool()
    h = await make_harness(tool)
    h.host.close()
    assert h.host.closing
    _, body = await h.call({"a": 1, "b": 2})
    assert body["result"]["isError"] is True
    assert "being suspended" in body["result"]["content"][0]["text"]
    assert not tool.started and not h.host.held
    assert await h.store.get_pending() == {}


# --------------------------------------------------------------------------
# immediate tools (answered right away, never held)
# --------------------------------------------------------------------------
class FakeImmediate:
    """An immediate tool: echoes its arguments, or fails / waits on demand."""

    kind = "integration"

    def __init__(
        self,
        name: str = "web_search",
        *,
        error: BaseException | None = None,
        gate: asyncio.Event | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.name = name
        self.description = "Searches the web."
        self.input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        self.error = error
        self.gate = gate
        self.timeout_sec = timeout_sec
        self.span_attributes = {"connector_key": "uipath-uipath-airdk"}
        self.calls: list[tuple[dict[str, Any], Any, str]] = []
        self.spans: list[Any] = []
        self.cancelled = False

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args.get("query"), str):
            raise ValueError("query must be a string")
        return dict(args)

    async def call(self, args: dict[str, Any], ctx: Any) -> str:
        self.calls.append((args, ctx, REQUEST_VAR.get()))
        self.spans.append(trace.get_current_span())
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return json.dumps({"hits": [args["query"]]})


class HostHarness(Harness):
    async def call_tool(
        self, name: str, args: dict[str, Any], call_id: str | None = "call_is", **kw
    ):
        params: dict[str, Any] = {"name": name, "arguments": args}
        if call_id is not None:
            params["_meta"] = {"callId": call_id}
        return await self.rpc("tools/call", params, **kw)


@pytest.fixture
async def make_host(tmp_path):
    cleanups: list[Any] = []

    async def make(
        tools: list[Any],
        *,
        with_store: bool = False,
        tracer: trace.Tracer | None = None,
        agent_input: dict[str, Any] | None = None,
        attachments: Any = None,
        model: str | None = None,
    ) -> HostHarness:
        storage = (
            SqliteResumableStorage(str(tmp_path / f"h{len(cleanups)}.db"))
            if with_store
            else None
        )
        store = CodexSessionStore(storage, "rt") if storage is not None else None
        REQUEST_VAR.set("from-run-context")
        host = DurableToolHost(
            tools,
            store=store,
            runtime_id="rt",
            context=contextvars.copy_context(),
            tracer=tracer,
            agent_input=agent_input,
            attachments=attachments,
            model=model,
        )
        REQUEST_VAR.set("unset")
        if with_store:
            probes: list[str] = []

            async def probe(call_id: str) -> bool:
                probes.append(call_id)
                return True

            host.bind(probe)
            host.probes = probes  # type: ignore[attr-defined]
        app = web.Application()
        app.router.add_route("*", "/mcp", host.handle)
        runner = web.AppRunner(app, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        cleanups.append((host, runner, storage))
        return HostHarness(host, f"http://127.0.0.1:{port}/mcp", store)  # type: ignore[arg-type]

    yield make
    for host, runner, storage in cleanups:
        await host.release_all()
        await host.cancel_inflight()
        await runner.cleanup()
        if storage is not None:
            await storage.dispose()


def _text(body: dict[str, Any]) -> str:
    return body["result"]["content"][0]["text"]


async def test_immediate_call_is_answered_without_probe_or_store(make_host):
    tool = FakeImmediate()
    h = await make_host([tool], agent_input={"topic": "x"})
    status, body = await h.call_tool("web_search", {"query": "codex"})
    assert status == 200
    assert body["result"]["isError"] is False
    assert json.loads(_text(body)) == {"hits": ["codex"]}
    [(args, ctx, seen_var)] = tool.calls
    assert args == {"query": "codex"}
    assert ctx.call_id == "call_is" and ctx.runtime_id == "rt"
    assert ctx.tool_name == "web_search" and ctx.agent_input == {"topic": "x"}
    assert seen_var == "from-run-context"
    assert not h.host.held and h.host.inflight_count == 0
    assert h.host.immediate_outcome("call_is") == _text(body)
    assert not h.host.has_durable


async def test_immediate_call_without_a_call_id(make_host):
    tool = FakeImmediate()
    h = await make_host([tool])
    _, body = await h.call_tool("web_search", {"query": "q"}, call_id=None)
    assert body["result"]["isError"] is False
    assert tool.calls[0][1].call_id is None


async def test_only_durable_tools_are_tagged_and_calls_do_not_mix(make_host):
    durable, immediate = FakeTool(), FakeImmediate()
    h = await make_host([durable, immediate], with_store=True)
    _, body = await h.rpc("tools/list")
    descriptions = {t["name"]: t["description"] for t in body["result"]["tools"]}
    assert descriptions == {
        "a_plus_b": f"{LONG_RUNNING_TAG} Adds a and b.",
        "web_search": "Searches the web.",
    }
    # One model response: a durable call (held) and an immediate one (answered).
    held = asyncio.create_task(h.call({"a": 1, "b": 2}))
    await _until(lambda: bool(h.host.held))
    _, body = await h.call_tool("web_search", {"query": "q"})
    assert body["result"]["isError"] is False
    assert list(h.host.held) == ["call_1"]
    assert h.host.probes == ["call_1"]  # type: ignore[attr-defined]
    assert list(await h.store.get_pending()) == ["call_1"]
    assert not held.done()
    await h.host.release_all()
    await asyncio.wait_for(held, 5)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            UiPathToolError("Integration service returned an error"),
            "Integration service returned an error",
        ),
        (RuntimeError("kaput"), "Tool 'web_search' failed: kaput"),
    ],
)
async def test_immediate_errors_are_is_error_results(
    make_host, error, expected, caplog
):
    h = await make_host([FakeImmediate(error=error)])
    _, body = await h.call_tool("web_search", {"query": "q"})
    assert body["result"]["isError"] is True
    assert _text(body) == expected
    if isinstance(error, RuntimeError):
        assert "tool web_search failed" in caplog.text


async def test_immediate_invalid_arguments(make_host):
    tool = FakeImmediate()
    h = await make_host([tool])
    _, body = await h.call_tool("web_search", {"query": 1})
    assert body["result"]["isError"] is True
    assert "Invalid arguments for tool 'web_search'" in _text(body)
    assert not tool.calls


def _provider() -> tuple[InMemorySpanExporter, trace.Tracer]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


async def test_immediate_timeout_cancels_the_call(make_host):
    exporter, tracer = _provider()
    tool = FakeImmediate(gate=asyncio.Event(), timeout_sec=0.1)
    h = await make_host([tool], tracer=tracer)
    _, body = await h.call_tool("web_search", {"query": "q"})
    assert body["result"]["isError"] is True
    assert _text(body) == "Tool 'web_search' did not finish within 0.1s."
    await _until(lambda: tool.cancelled)
    [span] = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert attrs(span)["error"] == "interrupted"
    assert h.host.inflight_count == 0


async def test_immediate_call_span(make_host):
    exporter, tracer = _provider()
    tool = FakeImmediate()
    with tracer.start_as_current_span("Agent run"):
        h = await make_host([tool], tracer=tracer)
    _, body = await h.call_tool("web_search", {"query": "q"})
    text = _text(body)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    run, call = spans["Agent run"], spans["Tool call - web_search"]
    assert call.parent is not None and call.parent.span_id == run.context.span_id
    call_attrs = attrs(call)
    assert call_attrs["span_type"] == "toolCall"
    assert (
        call_attrs["toolType"] == "Integration"
        and call_attrs["tool.kind"] == "integration"
    )
    assert call_attrs["tool.server"] == "uipath_tools"
    assert call_attrs["call_id"] == "call_is"
    assert call_attrs["connector_key"] == "uipath-uipath-airdk"
    assert json.loads(call_attrs["output.value"]) == {"content": text}
    assert json.loads(call_attrs["output"]) == {"hits": ["q"]}
    assert call.status.status_code.name == "OK"
    [current] = tool.spans
    assert current.get_span_context().span_id == call.context.span_id


async def test_integration_tool_child_span(make_host, monkeypatch):
    """The real IS tool: Tool call → integrationTool child (toolName, arguments)."""
    import uipath_codex_sdk.tools as tools_mod
    from uipath_codex_sdk import uipath_integration_tool

    from .durable.fakes import FakeConnections

    conns = FakeConnections()
    conns.responses["/v2/webSearch"] = {"results": []}
    monkeypatch.setattr(
        tools_mod, "_uipath_client", lambda: type("C", (), {"connections": conns})()
    )
    exporter, tracer = _provider()
    # The child span uses the package tracer: route it to our exporter.
    monkeypatch.setattr("uipath_codex_sdk.runtime.telemetry.get_tracer", lambda: tracer)
    tool = uipath_integration_tool(
        "Web Search",
        tool_name="web_search",
        connection_id="conn-1",
        connector_key="uipath-uipath-airdk",
        tool_path="/v2/webSearch",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        static_args={"provider": "GoogleCustomSearch"},
    )
    h = await make_host([tool], tracer=tracer)
    _, body = await h.call_tool("web_search", {"query": "q"})
    assert body["result"]["isError"] is False, body
    spans = {s.name: s for s in exporter.get_finished_spans()}
    call, child = spans["Tool call - web_search"], spans["Web Search"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert attrs(child)["span_type"] == "integrationTool"
    assert attrs(child)["toolName"] == "Web Search"
    assert json.loads(attrs(child)["arguments"]) == {
        "query": "q",
        "provider": "GoogleCustomSearch",
    }
    assert json.loads(attrs(child)["result"]) == {"results": []}
    assert attrs(call)["connection_id"] == "conn-1"
    assert attrs(call)["tool.display_name"] == "Web Search"
    [inv] = conns.invocations
    assert inv.activity_input == {"query": "q", "provider": "GoogleCustomSearch"}


async def test_context_tool_child_span(make_host, monkeypatch):
    """The real context tool: Tool call → contextGroundingTool RETRIEVER child."""
    from types import SimpleNamespace

    from uipath.eval.evaluators.legacy_context_precision_evaluator import (
        LegacyContextPrecisionEvaluator,
    )
    from uipath.platform.context_grounding import UnifiedQueryResult

    import uipath_codex_sdk.tools as tools_mod
    from uipath_codex_sdk import uipath_context_tool

    class Search:
        async def unified_search_async(self, **kwargs: Any) -> Any:
            return UnifiedQueryResult.model_validate(
                {
                    "semanticResults": {
                        "values": [
                            {
                                "content": "Book travel via the portal.",
                                "source": "travel.pdf",
                                "page_number": 1,
                                "score": 0.9,
                            }
                        ],
                        "metadata": {"operation_id": "op-1", "strategy": "s"},
                    }
                }
            )

    client = SimpleNamespace(context_grounding=Search(), folders=None)
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    exporter, tracer = _provider()
    monkeypatch.setattr("uipath_codex_sdk.runtime.telemetry.get_tracer", lambda: tracer)
    tool = uipath_context_tool(
        "Company Policies", index_name="company-policies", number_of_results=3
    )
    h = await make_host([tool], tracer=tracer)
    _, body = await h.call_tool("Company_Policies", {"query": "travel"})
    assert body["result"]["isError"] is False, body
    spans = {s.name: s for s in exporter.get_finished_spans()}
    call, child = spans["Tool call - Company_Policies"], spans["Company Policies"]
    retriever = spans["ContextGroundingRetriever"]
    assert child.parent is not None and child.parent.span_id == call.context.span_id
    assert retriever.parent is not None
    assert retriever.parent.span_id == child.context.span_id
    assert attrs(call)["toolType"] == "ContextGrounding"
    assert attrs(call)["tool.kind"] == "context"
    assert attrs(call)["index_name"] == "company-policies"
    assert attrs(call)["context_retrieval_mode"] == "Semantic"
    child_attrs = attrs(child)
    assert child_attrs["span_type"] == "contextGroundingTool"
    assert (
        child_attrs["retrieval_mode"] == "Semantic" and child_attrs["query"] == "travel"
    )
    assert (
        child_attrs["number_of_results"] == 3 and child_attrs["system_index"] is False
    )
    docs = json.loads(child_attrs["output"])["documents"]
    assert docs[0]["page_content"] == "Book travel via the portal."
    # Only the retriever span is an OpenInference span: the LLMOps exporter
    # takes SpanType from openinference.span.kind.
    for span in (call, child):
        assert "openinference.span.kind" not in attrs(span)
    assert attrs(retriever)["openinference.span.kind"] == "RETRIEVER"
    assert attrs(retriever)["input.value"] == "travel"
    assert attrs(retriever)["output.value"] == child_attrs["output"]
    # What the context precision evaluator finds in this trace.
    evaluator = LegacyContextPrecisionEvaluator.__new__(LegacyContextPrecisionEvaluator)
    [grounding] = evaluator._extract_context_groundings(list(spans.values()))
    assert grounding["query"] == "travel" and len(grounding["chunks"]) == 1


async def test_immediate_call_after_close_is_refused(make_host):
    tool = FakeImmediate()
    h = await make_host([tool])
    h.host.close()
    _, body = await h.call_tool("web_search", {"query": "q"})
    assert body["result"]["isError"] is True
    assert "being suspended" in _text(body)
    assert not tool.calls


async def test_concurrent_immediate_calls_and_cancel_inflight(make_host):
    gate = asyncio.Event()
    tool = FakeImmediate(gate=gate)
    h = await make_host([tool])
    first = asyncio.create_task(h.call_tool("web_search", {"query": "a"}, "c1"))
    second = asyncio.create_task(h.call_tool("web_search", {"query": "b"}, "c2"))
    await _until(lambda: h.host.inflight_count == 2)
    assert sorted(h.host.inflight_ids) == ["c1", "c2"]
    gate.set()
    (_, b1), (_, b2) = await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert not b1["result"]["isError"] and not b2["result"]["isError"]
    assert h.host.inflight_count == 0
    assert json.loads(h.host.immediate_outcome("c2") or "") == {"hits": ["b"]}

    # A call cut off by the suspension: cancelled, outcome unknown.
    tool.gate = asyncio.Event()
    third = asyncio.create_task(h.call_tool("web_search", {"query": "c"}, "c3"))
    await _until(lambda: h.host.inflight_count == 1)
    await h.host.cancel_inflight()
    assert tool.cancelled
    assert h.host.immediate_outcome("c3") == UNKNOWN_OUTCOME_OUTPUT
    _, b3 = await asyncio.wait_for(third, 5)
    assert b3["result"]["isError"] is True
    assert _text(b3) == UNKNOWN_OUTCOME_OUTPUT
    assert h.host.inflight_count == 0


def test_durable_tools_need_a_store():
    with pytest.raises(ValueError, match="session store"):
        DurableToolHost([FakeTool()], store=None, runtime_id="rt")
