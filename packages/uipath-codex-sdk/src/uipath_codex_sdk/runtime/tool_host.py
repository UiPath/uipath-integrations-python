"""MCP endpoint that serves UiPath tools to Codex.

It serves two kinds of tools on one server:

* **durable** tools (below), whose calls are held until the run suspends;
* **immediate** tools (:class:`~uipath_codex_sdk.tools.UiPathTool`, e.g.
  Integration Service activities), answered as soon as their work is done:
  the arguments are validated, ``tool.call`` runs in a copy of the run's
  context under a ``Tool call`` span, with a host-side timeout, and the result
  (or an ``isError`` result) is returned. No rollout probe, no session store,
  no hold. A call still running when the run suspends is cancelled and
  reported to the model as having an unknown outcome.

Mounted on the gateway shim at ``/mcp`` and registered with Codex as the
remote MCP server :data:`SERVER_NAME`. It is a stateless streamable-HTTP MCP
server that answers with plain JSON (no SSE stream, no sessions) and only
accepts its own per-run bearer token.

``tools/call`` for a durable tool:

1. validates the arguments (invalid → an ``isError`` result straight away);
2. waits until Codex has written the ``function_call`` with this call id to the
   rollout — it is recorded before the call runs but not flushed, and a
   suspend must never lose it;
3. records the intent, starts the job (``tool.start``, inside the context the
   run started in, so resource overwrites and the ``Agent run`` span apply,
   under a ``Tool call`` span that becomes the job's ``ParentSpanId``), and
   saves the pending call with that span, which stays open until the
   execution that uses the job's result completes it;
4. never answers: the call is held until the runtime stops Codex and suspends.

Job attachments: an argument at an attachment field of a tool's input schema
is resolved against the run's known attachments before validation (the model
may pass just ``{"ID": ...}``); an unknown ID is answered with LangChain's
error text and the tool does not run. As in LangChain (whose
``StaticArgsHandler`` merges the pre-filled values into the tool call before
``job_attachment_wrapper`` resolves it), the IDs checked are those of the
arguments once the tool's pre-filled values are merged in
(``prefilled_args``): a model value a pre-filled one replaces is not checked,
the pre-filled one is (an escalation's pre-filled ones are not: LangChain
never resolves an escalation's attachments). The tool still gets the model's arguments (resolved),
which are what is saved for resume; it merges its pre-filled values itself.
The attachments found at the attachment fields of a tool's output schema
become known to the run.

A tool that raises a ``UiPathRuntimeError`` (e.g. Analyze Files given a file
over its size limit) fails the run, as in LangChain: the error is recorded in
:attr:`DurableToolHost.fatal_error` and the runtime stops Codex.

Eval tool mocking (:mod:`~uipath_codex_sdk.runtime.mocking`): a call the
active mocker simulates is answered with the mocked value in the same turn,
under a ``Tool call`` span, after its arguments are validated. A durable one
starts no job and is never held or saved, so the run does not suspend on it.
A custom immediate tool is mocked the same way; the built-in immediate tools
(``mocks_in_call``) mock their platform call inside ``call`` instead.

A start failure is answered with an ``isError`` result the model can act on.
Once the runtime starts stopping Codex (:meth:`DurableToolHost.close`), a late
call is answered with an error and starts nothing.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Coroutine

from aiohttp import web
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from uipath.runtime.errors import UiPathRuntimeError

from ..attachments import (
    AttachmentStore,
    Path,
    attachment_paths,
    merge_span_attachments,
    span_attachments,
    to_attachment,
    values_at,
)
from ..tool_results import ToolResultStore
from ..tools import (
    AnyUiPathTool,
    DurableToolContext,
    ToolContext,
    UiPathDurableTool,
    UiPathDurableToolError,
    UiPathTool,
    UiPathToolError,
    is_durable_tool,
    tool_type_of,
)
from .errors import UiPathRunTerminatedError
from .mocking import (
    MockSpan,
    MockSpec,
    is_simulated,
    mark_simulated,
    mock_spec_of,
    mocked_text,
    wrap,
)
from .session_store import STATE_STARTED, STATE_STARTING, CodexSessionStore, PendingCall
from .telemetry import (
    _json,
    _only_keys,
    collect_held_spans,
    end_tool_call,
    fail_span,
    held_spans,
    push_running,
    set_tool_output,
    span_record,
    start_tool_call,
)

logger = logging.getLogger(__name__)

#: MCP server name the tool host is registered under (namespace ``mcp__uipath_tools``).
SERVER_NAME = "uipath_tools"
#: Env var Codex reads the tool host bearer token from.
TOKEN_ENV = "UIPATH_CODEX_TOOLS_TOKEN"
#: Prefix added to durable tool descriptions when instructions are enabled.
LONG_RUNNING_TAG = "[long-running]"

#: Timeout of an immediate tool that sets none.
DEFAULT_IMMEDIATE_TIMEOUT_SEC = 300.0
#: How long :meth:`DurableToolHost.cancel_inflight` waits for cancelled calls.
CANCEL_WAIT_SEC = 5.0
#: Result reported for an immediate call cut off by a suspension.
UNKNOWN_OUTCOME_OUTPUT = (
    "The run was suspended while this call was still running; its outcome is "
    "unknown and it may have taken effect. Check before calling it again."
)

_PROTOCOL_VERSION = "2025-06-18"
_TOKEN_BYTES = 32

#: Probe: ``await probe(call_id)`` → True once the call is in the rollout.
CallProbe = Callable[[str], Awaitable[bool]]


@dataclass
class HeldCall:
    """A durable call whose job started and whose request is being held."""

    call_id: str
    tool: str
    value: Any
    held_at: float


class DurableToolHost:
    """Serves UiPath tools over MCP: holds durable calls, answers immediate ones.

    Args:
        tools: Durable and immediate tools to serve, by model-facing name.
        store: Session store the pending durable calls are written to
            (required when there is a durable tool).
        runtime_id: Runtime id handed to each tool.
        context: Context the tools run in (a copy per call).
        tag_descriptions: Prefix durable descriptions with ``[long-running]``.
        tracer: Tracer for the tool spans (default: the package tracer).
        agent_input: The run's validated input (``ToolContext.agent_input``).
        attachments: The run's known job attachments.
        model: The agent's model (``ToolContext.model``).
    """

    def __init__(
        self,
        tools: list[UiPathDurableTool | UiPathTool],
        *,
        store: CodexSessionStore | None,
        runtime_id: str,
        context: contextvars.Context | None = None,
        tag_descriptions: bool = True,
        tracer: trace.Tracer | None = None,
        agent_input: dict[str, Any] | None = None,
        attachments: AttachmentStore | None = None,
        model: str | None = None,
        results: ToolResultStore | None = None,
    ) -> None:
        self._tools: dict[str, Any] = {tool.name: tool for tool in tools}
        self.attachments = attachments if attachments is not None else AttachmentStore()
        self._model = model
        self.results = results if results is not None else ToolResultStore()
        #: The error of a tool that failed the run (see the module docstring).
        self.fatal_error: Exception | None = None
        self.fatal = asyncio.Event()
        self._durable_names = frozenset(
            tool.name for tool in tools if is_durable_tool(tool)
        )
        if self._durable_names and store is None:
            raise ValueError("Durable tools need a session store.")
        self._store = store
        self._agent_input = agent_input
        self._runtime_id = runtime_id
        self._context = context
        self._tag = tag_descriptions
        self._tracer = tracer
        self.token = secrets.token_urlsafe(_TOKEN_BYTES)
        self._probe: CallProbe | None = None
        self._held: dict[str, HeldCall] = {}
        self._starting: set[str] = set()
        self._order = 0
        self._release = asyncio.Event()
        self._last_activity = time.monotonic()
        self._lock = asyncio.Lock()
        self._closing = False
        self._inflight: dict[str, asyncio.Task[tuple[str, bool]]] = {}
        self._outcomes: dict[str, str] = {}
        self._failed: set[str] = set()
        self._mocking: set[str] = set()
        self._anon = 0

    # --- state read by the suspend watcher ---------------------------------

    @property
    def held(self) -> dict[str, HeldCall]:
        return dict(self._held)

    @property
    def starting_count(self) -> int:
        return len(self._starting)

    @property
    def mocking_count(self) -> int:
        """Simulated durable calls being answered (neither held nor starting)."""
        return len(self._mocking)

    @property
    def last_activity(self) -> float:
        return self._last_activity

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def has_durable(self) -> bool:
        return bool(self._durable_names)

    @property
    def inflight_count(self) -> int:
        """Immediate calls whose work is running."""
        return len(self._inflight)

    @property
    def inflight_ids(self) -> list[str]:
        return list(self._inflight)

    def immediate_outcome(self, call_id: str) -> str | None:
        """Text an immediate call was answered with (or its unknown outcome)."""
        return self._outcomes.get(call_id)

    async def cancel_inflight(self) -> None:
        """Cancel the immediate calls still running (teardown).

        Each one is recorded with :data:`UNKNOWN_OUTCOME_OUTPUT`: its side
        effect may already have happened.
        """
        tasks = dict(self._inflight)
        if not tasks:
            return
        for call_id, task in tasks.items():
            if not call_id.startswith("_anon"):
                self._outcomes[call_id] = UNKNOWN_OUTCOME_OUTPUT
            task.cancel()
        await asyncio.wait(list(tasks.values()), timeout=CANCEL_WAIT_SEC)

    def close(self) -> None:
        """Stop starting jobs: Codex is being stopped.

        Set by the runtime just before it kills Codex. A ``tools/call`` that
        arrives later (Codex had started the call but the request was still in
        flight) is answered with an error instead of starting a job the
        suspension would not know about.
        """
        self._closing = True

    def bind(self, probe: CallProbe) -> None:
        """Set how to check that a call is in the rollout (once the thread is known)."""
        self._probe = probe

    def set_order_base(self, base: int) -> None:
        self._order = max(self._order, base)

    async def release_all(self) -> None:
        """Let every held request finish (teardown; the client is gone by then)."""
        self._release.set()
        await asyncio.sleep(0)

    def tool_list(self) -> list[dict[str, Any]]:
        out = []
        for tool in self._tools.values():
            description = tool.description
            durable = tool.name in self._durable_names
            if durable and self._tag and not description.startswith(LONG_RUNNING_TAG):
                description = f"{LONG_RUNNING_TAG} {description}"
            out.append(
                {
                    "name": tool.name,
                    "description": description,
                    "inputSchema": tool.input_schema,
                }
            )
        return out

    # --- HTTP ---------------------------------------------------------------

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.method != "POST":
            # Stateless server: no standalone SSE stream and no sessions.
            return web.Response(status=405, headers={"Allow": "POST"})
        if request.headers.get("Authorization", "") != f"Bearer {self.token}":
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            message = json.loads(await request.read())
        except ValueError:
            return _rpc_error(None, -32700, "parse error", status=400)
        if not isinstance(message, dict):
            return _rpc_error(None, -32600, "batches are not supported", status=400)

        method = message.get("method")
        mid = message.get("id")
        if mid is None:
            return web.Response(status=202)  # notification
        params = message.get("params") or {}
        if method == "initialize":
            version = params.get("protocolVersion") or _PROTOCOL_VERSION
            return _rpc_result(
                mid,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "uipath-tools", "version": "1.0.0"},
                },
            )
        if method == "ping":
            return _rpc_result(mid, {})
        if method == "tools/list":
            return _rpc_result(mid, {"tools": self.tool_list()})
        if method == "tools/call":
            return await self._tools_call(mid, params)
        return _rpc_error(mid, -32601, f"method not found: {method}")

    async def _tools_call(self, mid: Any, params: dict[str, Any]) -> web.StreamResponse:
        self._last_activity = time.monotonic()
        if self._closing:
            return _tool_error(mid, _CLOSING_TEXT)
        name = params.get("name")
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return _tool_error(mid, f"Unknown tool: {name!r}")
        if name not in self._durable_names:
            return await self._call_immediate(mid, tool, params)
        meta = params.get("_meta") or {}
        call_id = meta.get("callId") if isinstance(meta, dict) else None
        if not isinstance(call_id, str) or not call_id:
            return _tool_error(
                mid, f"Tool '{name}' cannot run: the call carries no call id."
            )
        raw_args, errors = self._resolve_attachments(tool, params)
        if errors:
            return _tool_error(mid, "\n".join(errors))
        try:
            args = tool.validate_args(raw_args)
        except ValueError as e:
            return _tool_error(mid, f"Invalid arguments for tool '{name}': {e}")
        spec = self._simulated_spec(tool)
        if spec is not None:
            # Simulated: answered now; no probe, no pending call, no hold.
            return await self._call_mocked(mid, tool, call_id, args, spec, durable=True)
        # The same call again (a client retry): keep holding, never start a
        # second job. No await between this check and the add below, so a
        # duplicate cannot slip in while the first one starts.
        if call_id in self._held or call_id in self._starting:
            return await self._hold(mid)

        self._starting.add(call_id)
        try:
            error = await self._start(tool, call_id, args)
        finally:
            self._starting.discard(call_id)
            self._last_activity = time.monotonic()
        if error is not None:
            return _tool_error(mid, error)
        return await self._hold(mid)

    # --- immediate tools ----------------------------------------------------

    async def _call_immediate(
        self, mid: Any, tool: UiPathTool, params: dict[str, Any]
    ) -> web.StreamResponse:
        meta = params.get("_meta") or {}
        call_id = meta.get("callId") if isinstance(meta, dict) else None
        if not isinstance(call_id, str) or not call_id:
            call_id = None
        raw_args, errors = self._resolve_attachments(tool, params)
        if errors:
            return _tool_error(mid, "\n".join(errors))
        try:
            args = tool.validate_args(raw_args)
        except ValueError as e:
            return _tool_error(mid, f"Invalid arguments for tool '{tool.name}': {e}")
        if self._closing:  # validation may have yielded
            return _tool_error(mid, _CLOSING_TEXT)
        if not getattr(tool, "mocks_in_call", False):
            spec = self._simulated_spec(tool)
            if spec is not None:
                return await self._call_mocked(
                    mid, tool, call_id, args, spec, durable=False
                )
        ctx = ToolContext(
            call_id=call_id,
            runtime_id=self._runtime_id,
            tool_name=tool.name,
            agent_input=self._agent_input,
            attachments=self.attachments,
            model=self._model,
        )
        key = self._task_key(call_id)
        task = self._spawn_in_context(self._traced_call(tool, args, ctx))
        timeout = getattr(tool, "timeout_sec", None) or DEFAULT_IMMEDIATE_TIMEOUT_SEC
        return await self._await_hosted(mid, key, call_id, task, timeout, tool.name)

    def _task_key(self, call_id: str | None) -> str:
        """Key of a hosted call in ``_inflight`` (its call id when free)."""
        if call_id is None or call_id in self._inflight:
            self._anon += 1
            return f"_anon{self._anon}"
        return call_id

    async def _await_hosted(
        self,
        mid: Any,
        key: str,
        call_id: str | None,
        task: asyncio.Task[tuple[str, bool]],
        timeout: float,
        name: str,
    ) -> web.StreamResponse:
        """Answer with a hosted call's ``(text, is_error)``, within ``timeout``.

        The call is in flight (``inflight_count``, cancelled by
        :meth:`cancel_inflight`) until it ends; its answer is recorded by call
        id for :meth:`immediate_outcome`.
        """
        self._inflight[key] = task
        self._last_activity = time.monotonic()
        try:
            try:
                text, is_error = await asyncio.wait_for(asyncio.shield(task), timeout)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.wait([task], timeout=CANCEL_WAIT_SEC)
                text = f"Tool '{name}' did not finish within {timeout:g}s."
                is_error = True
            except asyncio.CancelledError:
                if task.cancelled() and call_id is not None:
                    # Cancelled by cancel_inflight: the outcome is recorded.
                    text = self._outcomes.get(call_id, UNKNOWN_OUTCOME_OUTPUT)
                    is_error = True
                else:
                    task.cancel()  # the request itself was cancelled
                    raise
        finally:
            self._inflight.pop(key, None)
            self._last_activity = time.monotonic()
        if call_id is not None and call_id not in self._outcomes:
            self._outcomes[call_id] = text
            if is_error:
                self._failed.add(call_id)
        if is_error:
            return _tool_error(mid, text)
        return _tool_result(mid, text)

    async def _traced_call(
        self, tool: UiPathTool, args: dict[str, Any], ctx: ToolContext
    ) -> tuple[str, bool]:
        """``tool.call`` under a ``Tool call`` span; ``(text, is_error)``.

        Runs inside the run's context (a child of ``Agent run`` / ``Resume``).
        The span ends when the call does; it is never saved.
        """
        kind = str(getattr(tool, "kind", None) or "function")
        extra = getattr(tool, "span_attributes", None)
        shown = _span_args(tool, args, ctx)
        span = start_tool_call(
            tool.name,
            tool_type=tool_type_of(tool, kind),
            call_id=ctx.call_id,
            arguments=shown,
            tracer=self._tracer,
            attachments=span_attachments(input_attachments(tool, shown), "in"),
            **{
                "tool.kind": kind,
                "tool.server": SERVER_NAME,
                **(extra if isinstance(extra, dict) else {}),
            },
        )
        with trace.use_span(span, end_on_exit=False):
            try:
                text = await tool.call(args, ctx)
            except UiPathToolError as e:
                end_tool_call(span, error=str(e) or type(e).__name__)
                return str(e), True
            except asyncio.CancelledError:
                end_tool_call(span, error="interrupted")
                raise
            except UiPathRuntimeError as e:
                # Fails the run (LangChain raises it out of the agent graph).
                logger.error("tool %s failed the run: %s", tool.name, e)
                fail_span(span, e)
                span.end()
                if self.fatal_error is None:
                    self.fatal_error = e
                self.fatal.set()
                return f"Tool '{tool.name}' failed: {e}", True
            except Exception as e:
                logger.exception("tool %s failed", tool.name)
                fail_span(span, e)
                span.end()
                return f"Tool '{tool.name}' failed: {e}", True
        if not isinstance(text, str):
            text = json.dumps(text, default=str)
        parsed = _parsed(text)
        harvested = self.attachments.harvest(output_paths(tool), parsed)
        if harvested:
            merged = merge_span_attachments(
                (getattr(span, "attributes", None) or {}).get("attachments"),
                span_attachments(harvested, "out"),
            )
            if merged:
                span.set_attribute("attachments", merged)
        end_tool_call(span, output=parsed, content=text)
        return text, False

    # --- simulated (eval-mocked) calls ---------------------------------------

    def _simulated_spec(self, tool: AnyUiPathTool) -> MockSpec | None:
        """The tool's mock spec when the run's mocker simulates it, else None.

        Checked in the run's context: the handler's own may not have the
        mocking ContextVars.
        """

        def check() -> MockSpec | None:
            try:
                from uipath.eval.mocks._mock_context import mocker_context
            except Exception:  # pragma: no cover - eval extras missing
                return None
            if mocker_context.get() is None:
                return None
            spec = mock_spec_of(tool)
            return spec if is_simulated(spec.name) else None

        if self._context is None:
            return check()
        return self._context.run(check)

    async def _call_mocked(
        self,
        mid: Any,
        tool: Any,
        call_id: str | None,
        args: dict[str, Any],
        spec: MockSpec,
        *,
        durable: bool,
    ) -> web.StreamResponse:
        """Answer a simulated call with the mocked value (see :meth:`_traced_mock`).

        A client retry of the same call id gets the first answer: the mocker
        is not asked again (a mockito answer sequence stays in step).
        """
        if call_id is not None:
            if call_id in self._outcomes:
                text = self._outcomes[call_id]
                if call_id in self._failed:
                    return _tool_error(mid, text)
                return _tool_result(mid, text)
            running = self._inflight.get(call_id)
            if running is not None:
                await asyncio.wait([running])
                if running.cancelled():
                    return _tool_error(
                        mid, self._outcomes.get(call_id, UNKNOWN_OUTCOME_OUTPUT)
                    )
                text, is_error = running.result()
                return _tool_error(mid, text) if is_error else _tool_result(mid, text)
        ctx: ToolContext | DurableToolContext
        if durable:
            assert call_id is not None  # durable calls always carry one
            ctx = DurableToolContext(
                call_id=call_id,
                runtime_id=self._runtime_id,
                tool_name=tool.name,
                agent_input=self._agent_input,
                attachments=self.attachments,
                model=self._model,
                results=self.results,
            )
        else:
            ctx = ToolContext(
                call_id=call_id,
                runtime_id=self._runtime_id,
                tool_name=tool.name,
                agent_input=self._agent_input,
                attachments=self.attachments,
                model=self._model,
            )
        key = self._task_key(call_id)
        task = self._spawn_in_context(
            self._traced_mock(tool, args, ctx, spec, durable=durable)
        )
        if durable:
            self._mocking.add(key)
        timeout = (
            None if durable else getattr(tool, "timeout_sec", None)
        ) or DEFAULT_IMMEDIATE_TIMEOUT_SEC
        try:
            return await self._await_hosted(mid, key, call_id, task, timeout, tool.name)
        finally:
            self._mocking.discard(key)

    async def _traced_mock(
        self,
        tool: Any,
        args: dict[str, Any],
        ctx: ToolContext | DurableToolContext,
        spec: MockSpec,
        *,
        durable: bool,
    ) -> tuple[str, bool]:
        """The mocker's value under a ``Tool call`` span; ``(text, is_error)``.

        * ``tool.mock_precheck(args, ctx)``, when the tool has one, runs
          first: a :class:`UiPathToolError` from it is the answer, with no
          span (as a real ``start`` failing before its span);
        * the mocked function runs under the tool's child span
          (``tool.start_mock_span(args, ctx)`` → :class:`MockSpan`), renamed
          ``Simulated result: <name>`` when mocked, else under the ``Tool
          call`` span itself (a custom tool);
        * ``tool.mocked_output(args, value, ctx)`` turns the value into the
          model's text (a durable tool's ``finish`` shape); without it the
          value is dumped as JSON. A :class:`UiPathRunTerminatedError` from it
          (an escalation outcome mapped to ``end``) fails the run.

        The spans end with the call: nothing is pushed as running or saved.
        """
        generation_error = _mock_generation_error()
        precheck = getattr(tool, "mock_precheck", None)
        if callable(precheck):
            try:
                checked = precheck(args, ctx)
                if inspect.isawaitable(checked):
                    await checked
            except UiPathToolError as e:
                return str(e), True
            except Exception as e:
                logger.exception("mock precheck of %s failed", tool.name)
                return f"Tool '{tool.name}' failed: {str(e) or type(e).__name__}", True
        if durable:
            kind = str(getattr(tool, "kind", "durable"))
            attrs: dict[str, Any] = {"tool.kind": kind}
        else:
            kind = str(getattr(tool, "kind", None) or "function")
            attrs = {"tool.kind": kind, "tool.server": SERVER_NAME}
        extra = getattr(tool, "span_attributes", None)
        shown = _span_args(tool, args, ctx)
        span = start_tool_call(
            tool.name,
            tool_type=tool_type_of(tool, kind),
            call_id=ctx.call_id,
            arguments=shown,
            tracer=self._tracer,
            attachments=span_attachments(input_attachments(tool, shown), "in"),
            **{**attrs, **(extra if isinstance(extra, dict) else {})},
        )
        mspan: MockSpan | None = None

        def fail(error: BaseException, text: str | None = None) -> None:
            if mspan is not None:
                fail_span(mspan.span, error)
                mspan.span.end()
            if text is None:
                fail_span(span, error)
                span.end()
            else:
                end_tool_call(
                    span,
                    output=_parsed(text),
                    content=text,
                    error=str(error) or type(error).__name__,
                )

        with trace.use_span(span, end_on_exit=False):
            try:
                hook = getattr(tool, "start_mock_span", None)
                mspan = hook(args, ctx) if callable(hook) else None
                inner = mspan.span if mspan is not None else span
                with trace.use_span(inner, end_on_exit=False):
                    try:
                        value = await wrap(spec, _not_mocked(tool))(
                            **_mock_args(tool, args, ctx)
                        )
                    except generation_error as e:
                        # A mockito ``raise`` answer or arguments no behavior
                        # matches, an LLM failure: a tool error (as in
                        # langchain, where it is raised out of the tool).
                        cause = e.__cause__
                        detail = (
                            str(e).strip()
                            or (str(cause).strip() if cause else "")
                            or type(cause or e).__name__
                        )
                        raise UiPathToolError(
                            f"Tool '{tool.name}' failed: {detail}"
                        ) from e
                if mspan is not None:
                    mark_simulated(mspan.span)
                shaper = getattr(tool, "mocked_output", None)
                text = (
                    await shaper(args, value, ctx)
                    if callable(shaper)
                    else mocked_text(value)
                )
                if not isinstance(text, str):
                    text = json.dumps(text, default=str)
            except UiPathToolError as e:
                fail(e)
                return str(e), True
            except asyncio.CancelledError as e:
                fail(e)
                raise
            except UiPathRunTerminatedError as e:
                # The simulated result ends the run, as the real one would on
                # resume (an escalation outcome mapped to ``end``).
                logger.info("simulated tool %s ended the run: %s", tool.name, e)
                if mspan is not None:
                    mspan.span.set_attribute(
                        mspan.output_attribute,
                        _json(_only_keys(_parsed(e.result_text), mspan.output_keys)),
                    )
                fail(e, e.result_text)
                if self.fatal_error is None:
                    self.fatal_error = e
                self.fatal.set()
                return e.result_text, True
            except UiPathRuntimeError as e:
                logger.error("simulated tool %s failed the run: %s", tool.name, e)
                fail(e)
                if self.fatal_error is None:
                    self.fatal_error = e
                self.fatal.set()
                return f"Tool '{tool.name}' failed: {e}", True
            except Exception as e:
                logger.exception("simulated tool %s failed", tool.name)
                fail(e)
                return f"Tool '{tool.name}' failed: {str(e) or type(e).__name__}", True
        parsed = _parsed(text)
        harvested = self.attachments.harvest(output_paths(tool), parsed)
        if harvested:
            merged = merge_span_attachments(
                (getattr(span, "attributes", None) or {}).get("attachments"),
                span_attachments(harvested, "out"),
            )
            if merged:
                span.set_attribute("attachments", merged)
        if mspan is not None:
            mspan.span.set_attribute(
                mspan.output_attribute,
                _json(_only_keys(parsed, mspan.output_keys)),
            )
            mspan.span.set_status(Status(StatusCode.OK))
            mspan.span.end()
        end_tool_call(span, output=parsed, content=text)
        return text, False

    def _resolve_attachments(
        self, tool: Any, params: dict[str, Any]
    ) -> tuple[Any, list[str]]:
        """The call's arguments with known attachments filled in, and errors.

        The errors are those of the arguments merged with the tool's
        pre-filled values (LangChain's order, see the module docstring); the
        arguments returned are the model's, with the known attachments filled
        in (an unknown one a pre-filled value replaces is left as is).

        A tool with ``checks_prefilled_attachments = False`` (an escalation,
        which LangChain never resolves) has only the model's own values
        checked, those the pre-filled ones leave in place.
        """
        raw_args = params.get("arguments") or {}
        if not isinstance(raw_args, dict):
            return raw_args, []
        paths = input_paths(tool)
        merged = _prefilled_args(tool, raw_args, self._agent_input)
        if merged is None:
            if not paths:
                return raw_args, []
            return self.attachments.resolve(paths, raw_args)
        if not getattr(tool, "checks_prefilled_attachments", True):
            resolved, own = self.attachments.resolve(paths, raw_args)
            _, kept = self.attachments.resolve(paths, merged)
            return resolved, [e for e in own if e in kept]
        _, errors = self.attachments.resolve(prefilled_paths(tool), merged)
        if not paths:
            return raw_args, errors
        resolved, _ = self.attachments.resolve(paths, raw_args)
        return resolved, errors

    def _spawn_in_context(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        loop = asyncio.get_running_loop()
        if self._context is None:
            return loop.create_task(coro)
        # A copy per call: one Context object cannot be entered by two tasks.
        context = self._context.run(contextvars.copy_context)
        return loop.create_task(coro, context=context)

    # --- durable tools ------------------------------------------------------

    async def _start(
        self, tool: UiPathDurableTool, call_id: str, args: dict[str, Any]
    ) -> str | None:
        """Start the job and record it; return a model-facing error, or None."""
        if self._probe is None or not await self._probe(call_id):
            logger.warning(
                "durable call %s is not in the rollout; not starting it", call_id
            )
            return (
                f"Tool '{tool.name}' could not be started because the call was not "
                "recorded. Call it again."
            )
        if self._closing:
            return _CLOSING_TEXT
        assert self._store is not None  # durable tools always have one
        async with self._lock:
            self._order += 1
            order = self._order
        pending = PendingCall(
            call_id=call_id,
            tool=tool.name,
            args=args,
            state=STATE_STARTING,
            order=order,
        )
        await self._store.put_pending(pending)
        try:
            value, span = await self._run_in_context(
                self._traced_start(
                    tool,
                    args,
                    DurableToolContext(
                        call_id=call_id,
                        runtime_id=self._runtime_id,
                        tool_name=tool.name,
                        agent_input=self._agent_input,
                        attachments=self.attachments,
                        model=self._model,
                        results=self.results,
                    ),
                )
            )
        except UiPathDurableToolError as e:
            await self._store.remove_pending([call_id])
            return str(e)
        except UiPathRuntimeError as e:
            # Fails the run (e.g. an IXP extraction without a document).
            logger.error("durable tool %s failed the run: %s", tool.name, e)
            await self._store.remove_pending([call_id])
            if self.fatal_error is None:
                self.fatal_error = e
            self.fatal.set()
            return f"Tool '{tool.name}' failed: {e}"
        except Exception as e:
            logger.exception("durable tool %s failed to start", tool.name)
            await self._store.remove_pending([call_id])
            return f"Tool '{tool.name}' failed to start: {e}"

        pending.state = STATE_STARTED
        pending.value = value
        pending.value_type = type(value).__name__
        pending.span = span
        try:
            await self._store.put_pending(pending)
        except Exception:
            logger.exception(
                "durable call %s started its job but could not be saved; the job "
                "is orphaned",
                call_id,
            )
            raise
        self._held[call_id] = HeldCall(call_id, tool.name, value, time.monotonic())
        return None

    async def _traced_start(
        self, tool: UiPathDurableTool, args: dict[str, Any], ctx: DurableToolContext
    ) -> tuple[Any, dict[str, Any] | None]:
        """``tool.start`` under a ``Tool call`` span (the job's ``ParentSpanId``).

        Runs inside the run's context, so the span is a child of the run's
        current span (``Agent run``, or ``Resume`` in a resumed execution).
        A started job leaves the span open, pushed to LLMOps as running with
        the job key; returns the suspend value and the span's saved identity,
        which the execution that uses the job's result completes. A start
        failure ends the span with the error.
        """
        kind = str(getattr(tool, "kind", "durable"))
        extra = getattr(tool, "span_attributes", None)
        shown = _span_args(tool, args, ctx)
        span = start_tool_call(
            tool.name,
            tool_type=tool_type_of(tool, kind),
            call_id=ctx.call_id,
            arguments=shown,
            tracer=self._tracer,
            attachments=span_attachments(input_attachments(tool, shown), "in"),
            **{"tool.kind": kind, **(extra if isinstance(extra, dict) else {})},
        )
        with trace.use_span(span, end_on_exit=False):
            token = collect_held_spans()
            try:
                value = await tool.start(args, ctx)
            except BaseException as e:
                fail_span(span, e)
                span.end()
                raise
            finally:
                children = held_spans(token)
        set_tool_output(span, _started_output(tool, value))
        push_running(span)
        record = span_record(span)
        if record is not None and children:
            record["children"] = children
        keys = getattr(tool, "span_output_keys", None)
        if record is not None and keys:
            record["output_keys"] = list(keys)
        return value, record

    async def _run_in_context(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if self._context is None:
            return await coro
        return await self._spawn_in_context(coro)

    async def _hold(self, mid: Any) -> web.StreamResponse:
        """Never answer (until teardown): the Codex turn stays blocked here."""
        await self._release.wait()
        return _tool_error(mid, "The run was suspended while this call was pending.")


def input_paths(tool: Any) -> list[Path]:
    """Attachment fields of a tool's input schema (as the model sees it)."""
    try:
        return attachment_paths(tool.input_schema)
    except Exception:
        return []


def prefilled_paths(tool: Any) -> list[Path]:
    """Attachment fields of a tool's input, pre-filled ones included.

    Those of ``prefilled_schema`` (the input with its pre-filled fields, which
    ``input_schema`` hides) and of ``input_schema``.
    """
    paths = input_paths(tool)
    try:
        schema = getattr(tool, "prefilled_schema", None)
    except Exception:
        logger.debug("prefilled_schema of %s failed", tool.name, exc_info=True)
        schema = None
    if not isinstance(schema, dict):
        return paths
    full = attachment_paths(schema)
    return [*full, *(p for p in paths if p not in full)]


def _prefilled_args(
    tool: Any, args: dict[str, Any], agent_input: dict[str, Any] | None
) -> dict[str, Any] | None:
    """``tool.prefilled_args(args, agent_input)``; None without the hook (or on error)."""
    hook = getattr(tool, "prefilled_args", None)
    if not callable(hook):
        return None
    try:
        merged = hook(args, agent_input)
    except Exception:
        logger.debug("prefilled_args of %s failed", tool.name, exc_info=True)
        return None
    return merged if isinstance(merged, dict) else None


def output_paths(tool: Any) -> list[Path]:
    """Attachment fields of a tool's output schema (``output_json_schema``)."""
    schema = getattr(tool, "output_json_schema", None)
    return attachment_paths(schema) if isinstance(schema, dict) else []


def input_attachments(tool: Any, args: Any) -> list[Any]:
    """The attachments in a call's (resolved, pre-filled) arguments."""
    found = (to_attachment(v) for v in values_at(args, prefilled_paths(tool)))
    return [a for a in found if a is not None]


def _started_output(tool: Any, value: Any) -> dict[str, Any]:
    """The ``Tool call`` output while suspended: the tool's, else the job's."""
    hook = getattr(tool, "started_output", None)
    if callable(hook):
        try:
            out = hook(value)
            if isinstance(out, dict):
                return out
        except Exception:
            logger.debug("started_output of %s failed", tool.name, exc_info=True)
    job = getattr(value, "job", None)
    return {
        "status": "started",
        "job_key": getattr(job, "key", None),
        "job_state": getattr(getattr(job, "state", None), "value", None)
        or getattr(job, "state", None),
    }


class _NoSuchError(Exception):
    """Stands in for an exception type that cannot be imported."""


def _mock_generation_error() -> type[Exception]:
    """``UiPathMockResponseGenerationError``: the mocker could not answer."""
    try:
        from uipath.eval.mocks._mocker import UiPathMockResponseGenerationError
    except Exception:  # pragma: no cover - eval extras missing
        return _NoSuchError
    return UiPathMockResponseGenerationError


def _span_args(tool: Any, args: dict[str, Any], ctx: Any) -> Any:
    """The ``Tool call`` span's input: the tool's ``span_args`` hook, else the model's.

    A tool with hidden pre-filled fields shows them merged in, as low-code's
    tool call does (values marked sensitive included, as there).
    """
    hook = getattr(tool, "span_args", None)
    if not callable(hook):
        return args
    try:
        return hook(args, ctx)
    except Exception:
        logger.debug("span_args of %s failed", tool.name, exc_info=True)
        return args


def _mock_args(tool: Any, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """The arguments the mocker sees: the tool's ``mock_args`` hook, else the model's.

    A tool with hidden pre-filled fields merges them in, as langchain's
    ``StaticArgsHandler`` does before the mocked function runs.
    """
    hook = getattr(tool, "mock_args", None)
    return hook(args, ctx) if callable(hook) else args


def _not_mocked(tool: Any) -> Callable[..., Awaitable[Any]]:
    """The function a simulated call wraps: reached only if the mocker declines."""

    async def not_mocked(**_kwargs: Any):  # no return annotation: kept as is
        raise UiPathToolError(
            f"Tool '{tool.name}' was selected for simulation but no simulated "
            "result was produced."
        )

    return not_mocked


_CLOSING_TEXT = (
    "The run is being suspended; this call was not started. Call it again if "
    "it is still needed."
)


# --- JSON-RPC helpers -------------------------------------------------------


def _rpc_result(mid: Any, result: dict[str, Any]) -> web.Response:
    return web.json_response({"jsonrpc": "2.0", "id": mid, "result": result})


def _rpc_error(mid: Any, code: int, message: str, *, status: int = 200) -> web.Response:
    return web.json_response(
        {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}},
        status=status,
    )


def _parsed(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _tool_result(mid: Any, text: str) -> web.Response:
    return _rpc_result(
        mid, {"content": [{"type": "text", "text": text}], "isError": False}
    )


def _tool_error(mid: Any, text: str) -> web.Response:
    return _rpc_result(
        mid, {"content": [{"type": "text", "text": text}], "isError": True}
    )


__all__ = [
    "DEFAULT_IMMEDIATE_TIMEOUT_SEC",
    "input_paths",
    "output_paths",
    "prefilled_paths",
    "DurableToolHost",
    "HeldCall",
    "LONG_RUNNING_TAG",
    "SERVER_NAME",
    "TOKEN_ENV",
    "UNKNOWN_OUTCOME_OUTPUT",
]
