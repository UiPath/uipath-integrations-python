"""Runtime that owns the Codex run loop and routes it through UiPath.

For each run the runtime starts a loopback gateway shim, points a bundled
``codex`` subprocess at it via a custom ``model_provider``, starts a thread and
runs the rendered prompt, then maps the Codex ``TurnResult`` to a
``UiPathRuntimeResult``.

Durable tools (``agent.tools``) add a suspend/resume cycle (immediate tools,
such as Integration Service activities, are served on the same MCP server and
answered right away; they never suspend the run):

* **suspend** – the shim also serves the tools over MCP (``uipath_tools``).
  A call starts its Orchestrator job and is never answered, so the turn stays
  blocked. Once nothing else is running, a watcher SIGKILLs the codex
  app-server (a graceful stop would record the call as "aborted by user"),
  the rollout is saved to the state database, and the run returns SUSPENDED
  with one ``WaitJobRaw`` per pending call, keyed by its ``call_id``.
* **resume** – each finished job becomes the text result of its call. Once no
  call is pending, the rollout is restored, the thread resumed, the results
  injected with ``thread/inject_items`` under the original call ids, and a new
  turn asks the model to continue (it may suspend again).

Tracing (see :mod:`.telemetry`): the ``Agent run`` span is opened by the
factory's wrapper; this runtime adds the ``Tool call`` spans of the items Codex
runs and completes the still-open ``Tool call`` span of each durable call with
its job's result on resume. Model calls are traced in the gateway shim,
durable tool starts in the tool host.
"""

from __future__ import annotations

import asyncio
import contextvars
import glob
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, cast

from opentelemetry import context as context_api
from uipath.core.errors import UiPathPendingTriggerError
from uipath.platform.common import (
    WaitDeepRag,
    WaitDeepRagRaw,
    WaitJob,
    WaitTask,
)
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeEvent,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathRuntimeStorageProtocol,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath.runtime.events import (
    UiPathRuntimeMessageEvent,
    UiPathRuntimeStateEvent,
    UiPathRuntimeStatePhase,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from ..agent import UiPathCodexAgent
from ..attachments import (
    Attachment,
    AttachmentStore,
    attachment_paths,
    attachments_block,
    model_attachments,
    model_has_attachments,
    span_attachments,
    to_attachment,
    values_at,
    with_tickets,
)
from ..mcp import (
    McpServer,
    RemoteMcpServer,
    build_mcp_config_overrides,
    build_mcp_env,
    missing_bearer_token_env_vars,
)
from ..tool_results import ToolResultStore
from ..tools import (
    UiPathDurableTool,
    UiPathDurableToolPending,
    UiPathTool,
    is_durable_tool,
    tool_type_of,
)
from .errors import (
    UiPathCodexErrorCode,
    UiPathCodexRuntimeError,
    UiPathRunTerminatedError,
)
from .gateway import PROVIDER_ID, CodexGatewayShim
from .schema import get_entrypoint_schema, prompt_fields, to_strict_output_schema
from .session_store import (
    STATE_STARTED,
    CodexSessionStore,
    DanglingCall,
    EntrypointRecord,
    PendingCall,
    ThreadRecord,
    cut_torn_tail,
    dangling_calls,
    encode_rollout,
    find_rollout,
    has_call,
    parse_rollout,
    write_rollout,
)
from .telemetry import (
    TOOL_ITEM_TYPES,
    ItemSpans,
    complete_tool_call,
    end_tool_call,
    start_tool_call,
)
from .tool_host import (
    DEFAULT_IMMEDIATE_TIMEOUT_SEC,
    SERVER_NAME,
    TOKEN_ENV,
    DurableToolHost,
    input_paths,
    output_paths,
)
from .triggers import CodexResumeTriggerHandler

logger = logging.getLogger(__name__)

#: ``--config`` overrides that keep codex from calling home (ChatGPT plugins
#: backend) — a robot has no use for it and it can stall start-up. Applied to
#: agents with durable tools (or ``agent.hermetic=True``); other agents run
#: with the codex configuration they always had.
HERMETIC_OVERRIDES: tuple[str, ...] = (
    "features.plugins=false",
    'chatgpt_base_url="http://127.0.0.1:9/backend-api"',
)


#: ``UiPathRuntimeResult.metadata`` key carrying ``{call_id: {tool, arguments}}``
#: of a suspension (see ``factory.UiPathCodexResumableRuntime``).
DURABLE_CALLS_METADATA = "durable_calls"

#: User message of the turn that follows the injected results.
CONTINUE_PROMPT = "The pending tool call(s) above have completed. Continue the task."
#: Result injected for a sibling call that never ran because the run suspended.
NOT_EXECUTED_OUTPUT = (
    "Not executed: the run was suspended while waiting for another tool. "
    "Call it again if still needed."
)
#: Developer instruction added when the agent has durable tools.
DURABLE_INSTRUCTIONS = (
    "Tools marked [long-running] start a UiPath job, a research task or a human "
    "review that can take minutes or hours. Call independent ones together in a "
    "single response, and never call one again while you wait for its result."
)

#: Developer instruction added when the agent handles job attachments.
ATTACHMENT_INSTRUCTIONS = (
    "Files are passed around as JobAttachment objects with an ID, a FullName "
    "and a MimeType. You cannot open them yourself: to work with a file's "
    "contents, pass its JobAttachment to a tool that accepts attachments. Only "
    "use attachment IDs from your input or from tool results, never made-up ones."
)
#: Added to :data:`ATTACHMENT_INSTRUCTIONS` when the output has attachment fields.
OUTPUT_ATTACHMENT_INSTRUCTIONS = (
    "To return a file in your final output, set the attachment field to the ID "
    "of a JobAttachment from your input or a tool result."
)

#: Directory (under the runtime dir) used as CODEX_HOME by default.
CODEX_HOME_DIR = "codex_home"
#: Durable calls hold for a day before codex gives up on them.
DURABLE_TOOL_TIMEOUT_SEC = 86400.0
#: Codex's per-call timeout margin over the host's own immediate-tool timeout.
IMMEDIATE_TOOL_TIMEOUT_MARGIN_SEC = 30.0
#: How long nothing may happen before the watcher suspends.
QUIET_WINDOW_SEC = 1.0
#: How often the watcher looks.
WATCH_INTERVAL_SEC = 0.1
#: How long a tools/call waits for its function_call to reach the rollout.
ROLLOUT_PROBE_TIMEOUT_SEC = 5.0
ROLLOUT_PROBE_INTERVAL_SEC = 0.05
#: Pause before re-suspending when a tool's ``finish`` reports the job as still
#: pending although its trigger fired, so the in-process resume loop does not
#: poll Orchestrator in a tight loop.
RESUSPEND_BACKOFF_SEC = 2.0
#: State-database namespace / key of the input a durable run started with.
#: ``uipath run agent --file input.json --resume`` passes it again; it is
#: recognised as "no resume input" rather than as the result of a job.
RUN_INPUT_NAMESPACE = "codex_run"
RUN_INPUT_KEY = "input"
#: State-database key of the job attachments a suspended run knows.
RUN_ATTACHMENTS_KEY = "attachments"
#: State-database key of the results durable tools keep for each other.
RUN_RESULTS_KEY = "tool_results"
#: How long the watcher waits for the killed app-server to exit.
KILL_WAIT_SEC = 10.0
#: How long the run waits for the watcher to finish once it asked to suspend.
WATCHER_JOIN_SEC = 30.0

#: Graph node ids (see ``schema.get_agent_graph``) that state events name.
AGENT_NODE = "agent"
TOOLS_NODE = "tools"


def _node_for(item_type: Any) -> str:
    """Graph node a Codex item of ``item_type`` ran on.

    Shares ``telemetry.TOOL_ITEM_TYPES`` so an item that gets a tool span is
    the same item the debugger shows on the ``tools`` node.
    """
    return TOOLS_NODE if item_type in TOOL_ITEM_TYPES else AGENT_NODE


def _mcp_result_text(result: Any) -> Any:
    """Pull the text out of an MCP tool result's content list, if present."""
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
    return result


def _clean_codex_error(message: str) -> str:
    """Codex surfaces upstream failures as a JSON string; extract the message."""
    try:
        payload = json.loads(message)
    except ValueError:
        return message
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return str(payload["error"].get("message", message))
    return message


def normalize_resume_input(value: Any, pending_ids: list[str]) -> dict[str, Any]:
    """Map a resume payload to ``{call_id: resume_value}`` for the pending calls.

    Accepted shapes:

    * ``{call_id: value, ...}`` (a deployed resume, or a local ``--resume``
      with no input): the ids that are pending are taken, the rest ignored;
    * ``{}`` or ``None``: nothing has fired (``--resume`` while jobs run);
    * any other value: the result of the single pending call (``uipath run
      agent '{"output": 42}' --resume``). With several pending calls it is
      ambiguous and ignored.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        if not value:
            return {}
        keyed = {k: v for k, v in value.items() if k in pending_ids}
        if keyed:
            return keyed
    if len(pending_ids) == 1:
        return {pending_ids[0]: value}
    logger.warning(
        "Resume input is not keyed by call id and %d calls are pending; "
        "specify which one it answers as {call_id: value}. Pending: %s",
        len(pending_ids),
        pending_ids,
    )
    return {}


def _descendants(pid: int) -> list[int]:
    """Descendant pids of ``pid`` (Linux ``/proc``; empty elsewhere)."""
    if not os.path.isdir("/proc"):
        return []
    children: dict[int, list[int]] = {}
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(stat) as f:
                rest = f.read().rsplit(")", 1)[1].split()
            children.setdefault(int(rest[1]), []).append(int(stat.split("/")[2]))
        except (OSError, IndexError, ValueError):
            continue
    out: list[int] = []
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def _kill_pids(pids: list[int]) -> None:
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for pid in pids:
        try:
            os.kill(pid, sigkill)
        except (ProcessLookupError, PermissionError, OSError):
            pass


# --------------------------------------------------------------------------
# turn bookkeeping
# --------------------------------------------------------------------------
class _ItemTracker:
    """Started/completed thread items of the running turn."""

    def __init__(self, durable_names: frozenset[str] = frozenset()) -> None:
        self._durable_names = durable_names
        self.started: dict[str, dict[str, Any]] = {}
        self.completed: dict[str, dict[str, Any]] = {}
        self.last_activity = time.monotonic()

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def on_started(self, item: dict[str, Any]) -> None:
        self.touch()
        if isinstance(item.get("id"), str):
            self.started[item["id"]] = item

    def on_completed(self, item: dict[str, Any]) -> None:
        self.touch()
        if isinstance(item.get("id"), str):
            self.completed[item["id"]] = item

    def _is_durable(self, item: dict[str, Any]) -> bool:
        """A call to a durable tool (held by the host until the run suspends)."""
        return self._is_hosted(item) and item.get("tool") in self._durable_names

    @staticmethod
    def _is_hosted(item: dict[str, Any]) -> bool:
        """A call served by the tool host, which traces it itself."""
        return item.get("type") == "mcpToolCall" and item.get("server") == SERVER_NAME

    def running_durable(self) -> list[str]:
        """Ids of durable (``uipath_tools``) calls that started, not completed."""
        return [
            iid
            for iid, item in self.started.items()
            if iid not in self.completed and self._is_durable(item)
        ]

    def running_non_durable(self) -> list[str]:
        """Ids of tool-ish items that started and have not completed."""
        return [
            iid
            for iid, item in self.started.items()
            if iid not in self.completed
            and not self._is_durable(item)
            and item.get("type") not in ("userMessage", "agentMessage", "reasoning")
        ]

    def sibling_output(self, call_id: str) -> str | None:
        """The output a completed sibling call produced, as text, if known."""
        item = self.completed.get(call_id)
        if item is None:
            return None
        itype = item.get("type")
        if itype == "commandExecution":
            text = item.get("aggregated_output") or ""
            code = item.get("exit_code")
            return f"Exit code: {code}\nOutput:\n{text}" if code is not None else text
        if itype == "mcpToolCall":
            if item.get("error"):
                return json.dumps(item["error"], default=str)
            result = _mcp_result_text(item.get("result"))
            return (
                result if isinstance(result, str) else json.dumps(result, default=str)
            )
        return json.dumps(item, default=str)


@dataclass
class _SuspendState:
    requested: bool = False
    killed: bool = False
    error: Exception | None = None


def _raise_fatal(host: DurableToolHost | None, cause: BaseException | None) -> None:
    """Raise the error of a tool that failed the run, if one did."""
    if host is not None and host.fatal_error is not None:
        raise host.fatal_error from cause


def _dumps_fields(fields: dict[str, Any]) -> str:
    return json.dumps(fields) if fields else ""


def _call_job_key(call: Any) -> str | None:
    """Key of the job a saved durable call started, if it recorded one."""
    value = getattr(call, "value", None)
    job = value.get("job") if isinstance(value, dict) else None
    if not isinstance(job, dict):
        return None
    return job.get("key") or job.get("Key")


def _call_operation(call: Any) -> dict[str, str]:
    """What a saved durable call waits on: ``job_key``, ``deep_rag_id`` or ``task_id``."""
    key = _call_job_key(call)
    if key:
        return {"job_key": key}
    value = getattr(call, "value", None)
    deep_rag = value.get("deep_rag") if isinstance(value, dict) else None
    if isinstance(deep_rag, dict) and deep_rag.get("id"):
        return {"deep_rag_id": str(deep_rag["id"])}
    extraction = value.get("extraction") if isinstance(value, dict) else None
    if isinstance(extraction, dict):
        op = extraction.get("operationId") or extraction.get("operation_id")
        if op:
            return {
                "extraction_id": str(op),
                "document_id": str(
                    extraction.get("documentId") or extraction.get("document_id") or ""
                ),
            }
    validation = value.get("extraction_validation") if isinstance(value, dict) else None
    if isinstance(validation, dict):
        op = validation.get("operationId") or validation.get("operation_id")
        if op:
            out = {"validation_operation_id": str(op)}
            task_url = value.get("task_url") if isinstance(value, dict) else None
            if task_url:
                out["task_url"] = str(task_url)
            return out
    batch = value.get("batch_transform") if isinstance(value, dict) else None
    if isinstance(batch, dict) and batch.get("id"):
        return {"batch_transform_id": str(batch["id"])}
    task = value.get("action") if isinstance(value, dict) else None
    if isinstance(task, dict) and (task.get("id") is not None or task.get("key")):
        out = {"task_id": str(task.get("id"))} if task.get("id") is not None else {}
        if task.get("key"):
            out["task_key"] = str(task["key"])
        return out
    return {}


def _resumed_tool_event(
    call: Any, result: str, execution_id: str
) -> UiPathRuntimeStateEvent:
    """``● tools`` event for a durable call finished on resume.

    Mirrors the shape of a live ``mcpToolCall`` event (server / tool /
    arguments / result / status) and adds the job key and ``resumed: True``.
    """
    status = "completed"
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("status"), str):
        status = parsed["status"].lower()
    return UiPathRuntimeStateEvent(
        payload={
            "server": SERVER_NAME,
            "tool": call.tool,
            "arguments": call.args,
            **(_call_operation(call) or {"job_key": None}),
            "result": parsed if parsed is not None else result,
            "status": status,
            "resumed": True,
        },
        node_name=TOOLS_NODE,
        execution_id=execution_id,
        phase=UiPathRuntimeStatePhase.UPDATED,
        metadata={
            "source": "codex",
            "item_type": "durableToolResult",
            "call_id": call.call_id,
        },
    )


async def _rebuilt_trigger(handler: Any, value: Any) -> Any | None:
    """A resume trigger to read a saved suspend value with, or None.

    ``create_trigger`` is side-effect free for a ``WaitJob`` and a
    ``WaitTask`` (``WaitEscalation``) only: for a
    ``WaitDeepRag`` it registers the DeepRAG run's external trigger with
    Orchestrator again, so that trigger is built directly, and so is a
    Batch Transform's (``WaitBatchTransformFile``, which carries the
    ``destination_path`` its reader needs).
    """
    from uipath.platform.common import (
        WaitDocumentExtraction,
        WaitDocumentExtractionValidation,
    )

    from ..batch_transform_tools import WaitBatchTransformFile

    if isinstance(value, (WaitDocumentExtraction, WaitDocumentExtractionValidation)):
        # IXP operations: no folder (the platform clears it), and the reader
        # needs project_id and tag in the payload.
        from uipath.core.serialization import serialize_object
        from uipath.core.triggers import (
            UiPathResumeTrigger,
            UiPathResumeTriggerName,
            UiPathResumeTriggerType,
        )

        is_extraction = isinstance(value, WaitDocumentExtraction)
        op: Any = (
            value.extraction
            if isinstance(value, WaitDocumentExtraction)
            else value.extraction_validation
        )
        trigger = UiPathResumeTrigger.model_validate(
            {
                "triggerType": UiPathResumeTriggerType.IXP_EXTRACTION
                if is_extraction
                else UiPathResumeTriggerType.IXP_VS_ESCALATION,
                "itemKey": op.operation_id,
            }
        )
        # The names the platform gives them (a validation trigger is "Api").
        trigger.trigger_name = (
            UiPathResumeTriggerName.EXTRACTION
            if is_extraction
            else UiPathResumeTriggerName.API
        )
        payload = serialize_object(value)
        payload.setdefault("project_id", op.project_id)
        payload.setdefault("tag", op.tag)
        trigger.payload = payload
        return trigger
    if isinstance(value, WaitBatchTransformFile):
        from uipath.core.serialization import serialize_object
        from uipath.core.triggers import (
            UiPathResumeTrigger,
            UiPathResumeTriggerName,
            UiPathResumeTriggerType,
        )

        trigger = UiPathResumeTrigger.model_validate(
            {
                "triggerType": UiPathResumeTriggerType.BATCH_RAG,
                "itemKey": value.batch_transform.id,
                "folderKey": value.index_folder_key,
                "folderPath": value.index_folder_path,
            }
        )
        trigger.trigger_name = UiPathResumeTriggerName.BATCH_RAG
        trigger.payload = serialize_object(value)
        return trigger
    if isinstance(value, (WaitJob, WaitTask)):
        return await handler.create_trigger(value)
    if isinstance(value, WaitDeepRag):
        from uipath.core.triggers import (
            UiPathResumeTrigger,
            UiPathResumeTriggerName,
            UiPathResumeTriggerType,
        )

        trigger = UiPathResumeTrigger.model_validate(
            {
                "triggerType": UiPathResumeTriggerType.DEEP_RAG,
                "itemKey": value.deep_rag.id,
                "folderKey": value.index_folder_key,
                "folderPath": value.index_folder_path,
            }
        )
        # Excluded from the model's aliases: set after validation.
        trigger.trigger_name = (
            UiPathResumeTriggerName.DEEP_RAG_RAW
            if isinstance(value, WaitDeepRagRaw)
            else UiPathResumeTriggerName.DEEP_RAG
        )
        trigger.payload = value.model_dump(mode="json")
        return trigger
    return None


@dataclass
class _ResumePlan:
    thread_id: str
    items: list[dict[str, Any]] = field(default_factory=list)


class UiPathCodexRuntime:
    """Executes a single :class:`UiPathCodexAgent` via the Codex SDK.

    Args:
        agent: The agent definition.
        runtime_id: UiPath runtime id (the job key when deployed).
        entrypoint: Entrypoint name the agent was loaded for.
        file_path: Path of the module the agent was loaded from.
        storage: Runtime storage (the state database). Required for durable
            tools; the session state lives there.
        codex_home: ``CODEX_HOME`` for an agent with durable tools;
            ``agent.codex_home`` takes precedence. Defaults to
            ``<project>/__uipath/codex_home``. Agents without durable tools
            ignore it and keep codex's own ``CODEX_HOME`` resolution.
    """

    def __init__(
        self,
        *,
        agent: UiPathCodexAgent,
        runtime_id: str,
        entrypoint: str,
        file_path: str,
        storage: UiPathRuntimeStorageProtocol | None = None,
        codex_home: str | None = None,
    ) -> None:
        self.agent = agent
        self.runtime_id = runtime_id
        self.entrypoint = entrypoint
        self.file_path = file_path
        self.storage = storage
        self._codex_home = codex_home
        self._resumed_events: list[UiPathRuntimeEvent] = []
        #: Durable calls whose saved ``Tool call`` span was completed.
        self._completed_spans: set[str] = set()
        self._session = (
            CodexSessionStore(storage, runtime_id) if storage is not None else None
        )
        self._shim: CodexGatewayShim | None = None
        #: The job attachments this run knows (its input's, tools' results).
        self._attachments = AttachmentStore()
        #: Results durable tools keep for each other (IXP extraction → validation).
        self._results = ToolResultStore()

    # --- execution --------------------------------------------------------

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        # Single code path: drain the stream and return its terminal result.
        result: UiPathRuntimeResult | None = None
        stream_options = (
            UiPathStreamOptions(**options.model_dump()) if options else None
        )
        async for event in self.stream(input, stream_options):
            if isinstance(event, UiPathRuntimeResult):
                result = event
        if result is None:  # pragma: no cover - stream always yields a result
            return UiPathRuntimeResult(
                output={"output": ""}, status=UiPathRuntimeStatus.SUCCESSFUL
            )
        return result

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Run a Codex turn, emitting one UiPath event per completed Codex item.

        Each Codex ``ItemCompletedNotification`` (reasoning, command execution,
        agent message, ...) becomes a message/state event; the terminal
        ``TurnCompletedNotification`` yields the final ``UiPathRuntimeResult``.
        With durable tools the terminal result may be SUSPENDED instead.
        """
        self._refuse_breakpoints(options)
        tools = self._durable_tools()
        resume = bool(options is not None and getattr(options, "resume", False))
        plan: _ResumePlan | None = None
        if tools and resume:
            self._resumed_events = []
            self._attachments = await self._load_attachments()
            self._results = await self._load_results()
            prepared = await self._prepare_resume(input)
            # Show what each finished durable call returned before the run
            # continues (or suspends again on the calls still running).
            for event in self._resumed_events:
                yield event
            self._resumed_events = []
            if isinstance(prepared, UiPathRuntimeResult):
                yield prepared
                return
            plan = prepared
            prompt = CONTINUE_PROMPT
        else:
            prompt = self._render_prompt(input or {})
            self._attachments = AttachmentStore(self.input_attachments(input or {}))
            self._results = ToolResultStore()
            if tools:
                session = self._require_session()
                await session.clear_all()
                await session.set_entrypoint(EntrypointRecord(self.entrypoint))
                await self._save_run_input(input)
                await self._save_attachments(force=True)
                await self._save_results(force=True)
                # A kept state file may still hold triggers of an earlier
                # suspension; the resumable wrapper would merge them in.
                save_triggers = getattr(self.storage, "save_triggers", None)
                if save_triggers is not None:
                    await save_triggers(self.runtime_id, [])

        if plan is not None:
            agent_input = await self._load_run_input()
        else:
            agent_input = input
        async for event in self._run(
            prompt,
            tools,
            plan,
            agent_input=agent_input if isinstance(agent_input, dict) else None,
        ):
            yield event

    @staticmethod
    def _refuse_breakpoints(options: UiPathStreamOptions | None) -> None:
        """Answer a breakpoint request out loud instead of ignoring it.

        The agent loop runs inside the codex app-server, so this runtime owns
        no node it could hold execution at. ``uipath debug`` still offers
        stepping and node breakpoints, and a debugger that accepts them and
        then runs straight through looks broken rather than unsupported.
        """
        if options is None or not options.breakpoints:
            return
        logger.warning(
            "Breakpoints are not supported by the Codex runtime, so this run "
            "continues to completion. The agent loop runs inside the codex "
            "app-server and exposes no node to hold at. A durable tool (a "
            "UiPath process tool) is the supported way to suspend a run."
        )

    async def get_schema(self) -> UiPathRuntimeSchema:
        return get_entrypoint_schema(self.agent, self.entrypoint)

    async def dispose(self) -> None:
        if self._shim is not None:
            await self._shim.stop()
            self._shim = None

    # --- the codex run ----------------------------------------------------

    async def _run(
        self,
        prompt: str,
        tools: dict[str, UiPathDurableTool],
        plan: _ResumePlan | None,
        agent_input: dict[str, Any] | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
        from openai_codex.errors import CodexError
        from openai_codex.generated.v2_all import (
            ErrorNotification,
            ItemCompletedNotification,
            ItemStartedNotification,
            ThreadTokenUsageUpdatedNotification,
            TurnCompletedNotification,
            TurnStatus,
        )

        codex_home = self._codex_home_dir(tools)
        # Every durable path below only runs with tools, where it is set.
        durable_home = codex_home or ""
        host: DurableToolHost | None = None
        _, immediate = self._agent_tools()
        if tools or immediate:
            host = DurableToolHost(
                [*tools.values(), *immediate.values()],
                store=self._require_session() if tools else None,
                runtime_id=self.runtime_id,
                context=contextvars.copy_context(),
                tag_descriptions=self.agent.durable_tool_instructions,
                agent_input=agent_input,
                attachments=self._attachments,
                model=self.agent.model,
                results=self._results,
            )
        if plan is not None:
            await self._restore_rollout(durable_home, plan.thread_id)

        shim = CodexGatewayShim(
            default_model=self.agent.model,
            mcp_handler=host,
            # The handlers run on aiohttp's tasks: hand them the Agent run span.
            trace_context=context_api.get_current(),
        )
        await shim.start()
        self._shim = shim
        item_dicts: list[dict[str, Any]] = []
        tracker = _ItemTracker(frozenset(tools))
        item_spans = ItemSpans()
        suspend = _SuspendState()
        watcher: asyncio.Task[None] | None = None
        fatal_watch: asyncio.Task[None] | None = None
        codex: Any = None
        thread_id: str | None = None
        try:
            servers = self._mcp_servers(shim, host)
            # Warn about remote MCP servers whose bearer token can't be
            # resolved — codex silently skips them (no tools) otherwise.
            for name, var in missing_bearer_token_env_vars(
                servers, dict(os.environ)
            ).items():
                logger.warning(
                    "MCP server %r: bearer_token_env_var %r is not set and no "
                    "literal bearer_token was provided; codex will skip this "
                    "server and its tools will be unavailable.",
                    name,
                    var,
                )
            codex_bin = self.agent.codex_bin or os.environ.get("UIPATH_CODEX_BIN")
            env = {**shim.env(), **build_mcp_env(servers)}
            if codex_home is not None:
                env["CODEX_HOME"] = codex_home
            config = CodexConfig(
                codex_bin=codex_bin,
                config_overrides=(
                    *shim.config_overrides(),
                    *(HERMETIC_OVERRIDES if self._hermetic(tools) else ()),
                    *build_mcp_config_overrides(servers),
                ),
                env=env,
                cwd=self._project_dir(),
            )
            codex = AsyncCodex(config)
            await codex.__aenter__()
            thread_kwargs: dict[str, Any] = {
                "model": self.agent.model,
                "model_provider": PROVIDER_ID,
                "sandbox": Sandbox(self.agent.sandbox),
                "approval_mode": ApprovalMode(self.agent.approval_mode),
                "base_instructions": self.agent.base_instructions,
            }
            instructions = self._developer_instructions(tools)
            if instructions:
                thread_kwargs["developer_instructions"] = instructions
            if plan is not None:
                thread = await codex.thread_resume(
                    plan.thread_id, cwd=self._project_dir(), **thread_kwargs
                )
                if thread.id != plan.thread_id:
                    raise UiPathCodexRuntimeError(
                        UiPathCodexErrorCode.SESSION_MISMATCH,
                        "Resumed the wrong Codex thread",
                        f"Expected thread {plan.thread_id}, codex resumed {thread.id}.",
                        UiPathErrorCategory.SYSTEM,
                    )
                await self._inject_items(codex, thread.id, plan.items)
            else:
                thread = await codex.thread_start(**thread_kwargs)
            thread_id = thread.id
            if host is not None and tools:
                await self._require_session().set_thread(ThreadRecord(thread.id))
                host.bind(self._rollout_probe(durable_home, thread.id))

            run_kwargs: dict[str, Any] = {}
            if self.agent.output_schema is not None:
                run_kwargs["output_schema"] = to_strict_output_schema(
                    self.agent.output_schema.model_json_schema()
                )
            if host is not None:
                fatal_watch = asyncio.create_task(self._stop_on_fatal(codex, host))
            try:
                handle = await thread.turn(prompt, **run_kwargs)
                if host is not None and tools:
                    watcher = asyncio.create_task(
                        self._watch(
                            codex, shim, host, tracker, suspend, durable_home, thread.id
                        )
                    )
                async for notification in handle.stream():
                    payload = notification.payload
                    tracker.touch()
                    if isinstance(payload, ItemStartedNotification):
                        item = payload.item.model_dump(mode="json")
                        tracker.on_started(item)
                        if not _ItemTracker._is_hosted(item):
                            item_spans.start(item)
                    elif isinstance(payload, ItemCompletedNotification):
                        item = payload.item.model_dump(mode="json")
                        tracker.on_completed(item)
                        if not _ItemTracker._is_hosted(item):
                            item_spans.finish(item)
                        item_dicts.append(item)
                        event = self._item_to_event(item)
                        if event is not None:
                            yield event
                    elif isinstance(payload, TurnCompletedNotification):
                        if payload.turn.status == TurnStatus.failed:
                            error = payload.turn.error
                            raise UiPathCodexRuntimeError(
                                UiPathCodexErrorCode.EXECUTION_FAILED,
                                "Codex execution failed",
                                _clean_codex_error(
                                    getattr(error, "message", "") or "turn failed"
                                ),
                                UiPathErrorCategory.USER,
                            )
                    elif isinstance(payload, ErrorNotification):
                        # Turn-scoped error. If Codex will retry, log and
                        # continue; otherwise it's fatal — surface it (it
                        # doesn't always arrive as a failed turn/completed).
                        err = getattr(payload, "error", None)
                        msg = _clean_codex_error(
                            getattr(err, "message", "") or "codex error"
                        )
                        if getattr(payload, "will_retry", False):
                            logger.warning("codex error (will retry): %s", msg)
                        else:
                            raise UiPathCodexRuntimeError(
                                UiPathCodexErrorCode.EXECUTION_FAILED,
                                "Codex execution failed",
                                msg,
                                UiPathErrorCategory.USER,
                            )
                    elif isinstance(payload, ThreadTokenUsageUpdatedNotification):
                        pass  # token usage → tracing milestone
            except UiPathCodexRuntimeError as e:
                _raise_fatal(host, e)
                raise
            except CodexError as e:
                _raise_fatal(host, e)
                # The suspend watcher SIGKILLed the app-server: the stream ends
                # with a closed transport. That is the expected path.
                if suspend.error is not None:
                    raise suspend.error from e
                if not suspend.requested:
                    raise UiPathCodexRuntimeError(
                        UiPathCodexErrorCode.EXECUTION_FAILED,
                        "Codex execution failed",
                        _clean_codex_error(str(e)),
                        UiPathErrorCategory.USER,
                    ) from e
            except Exception as e:
                _raise_fatal(host, e)
                if suspend.error is not None:
                    raise suspend.error from e
                if not suspend.requested:
                    raise UiPathCodexRuntimeError(
                        UiPathCodexErrorCode.EXECUTION_FAILED,
                        "Codex execution failed",
                        _clean_codex_error(str(e)),
                        UiPathErrorCategory.USER,
                    ) from e
                logger.debug("codex stream ended after suspend kill: %r", e)
            if watcher is not None and suspend.requested:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(watcher), timeout=WATCHER_JOIN_SEC
                    )
                except asyncio.TimeoutError as e:
                    raise UiPathCodexRuntimeError(
                        UiPathCodexErrorCode.ROLLOUT_CHECKPOINT_FAILED,
                        "Cannot suspend the Codex run",
                        "The codex app-server did not stop after it was killed.",
                        UiPathErrorCategory.SYSTEM,
                    ) from e
            if suspend.error is not None:
                raise suspend.error
            _raise_fatal(host, None)
        finally:
            if fatal_watch is not None and not fatal_watch.done():
                fatal_watch.cancel()
            item_spans.close_all("suspended" if suspend.killed else "interrupted")
            if watcher is not None and not watcher.done():
                watcher.cancel()
                try:
                    await watcher
                except (asyncio.CancelledError, Exception):
                    pass
            if codex is not None:
                try:
                    await codex.close()
                except Exception:
                    logger.debug("codex close failed", exc_info=True)
            if host is not None:
                # Only after codex is gone: a live codex would take the release
                # answer as the tool result and keep sampling.
                await host.release_all()
                # Immediate calls cut off by the suspension: their outcome is
                # unknown (recorded for the dangling-call outputs below).
                await host.cancel_inflight()
            await shim.stop()
            self._shim = None

        if suspend.killed and thread_id is not None:
            yield await self._finish_suspend(durable_home, thread_id, tracker, host)
            return
        if tools:
            await self._require_session().clear_suspension()
        yield UiPathRuntimeResult(
            output=self._build_output(self._final_text(item_dicts)),
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _mcp_servers(
        self, shim: CodexGatewayShim, host: DurableToolHost | None
    ) -> dict[str, McpServer]:
        servers: dict[str, McpServer] = dict(self.agent.mcp_servers or {})
        if host is None:
            return servers
        if SERVER_NAME in servers:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.CONFIG_INVALID,
                "Reserved MCP server name",
                f"The MCP server name {SERVER_NAME!r} is reserved for UiPath tools.",
                UiPathErrorCategory.USER,
            )
        if host.has_durable:
            tool_timeout = DURABLE_TOOL_TIMEOUT_SEC
        else:
            # The host enforces each immediate tool's own timeout; codex's
            # per-server one only has to outlast the longest of them.
            _, immediate = self._agent_tools()
            tool_timeout = (
                max(
                    (
                        getattr(t, "timeout_sec", None) or DEFAULT_IMMEDIATE_TIMEOUT_SEC
                        for t in immediate.values()
                    ),
                    default=DEFAULT_IMMEDIATE_TIMEOUT_SEC,
                )
                + IMMEDIATE_TOOL_TIMEOUT_MARGIN_SEC
            )
        servers[SERVER_NAME] = RemoteMcpServer(
            url=shim.mcp_url,
            bearer_token_env_var=TOKEN_ENV,
            bearer_token=host.token,
            startup_timeout_sec=30.0,
            tool_timeout_sec=tool_timeout,
            default_tools_approval_mode="approve",
            supports_parallel_tool_calls=True,
            # A durable call must be a top-level function_call in the rollout
            # (it is held, then resumed by call_id). Keep the tools out of
            # tool-search deferral and out of code-mode ``exec`` scripts;
            # immediate tools share the server, and so the setting.
            omit_tools_from=["deferred", "code_mode"],
        )
        return servers

    # --- suspend ------------------------------------------------------------

    def _rollout_probe(self, codex_home: str, thread_id: str):
        async def probe(call_id: str) -> bool:
            deadline = time.monotonic() + ROLLOUT_PROBE_TIMEOUT_SEC
            while True:
                path = find_rollout(codex_home, thread_id)
                if path is not None:
                    try:
                        data = await asyncio.to_thread(path.read_bytes)
                    except OSError:
                        data = b""
                    if has_call(parse_rollout(cut_torn_tail(data)), call_id):
                        return True
                if time.monotonic() >= deadline:
                    return False
                await asyncio.sleep(ROLLOUT_PROBE_INTERVAL_SEC)

        return probe

    async def _watch(
        self,
        codex: Any,
        shim: CodexGatewayShim,
        host: DurableToolHost,
        tracker: _ItemTracker,
        suspend: _SuspendState,
        codex_home: str,
        thread_id: str,
    ) -> None:
        """Suspend once a durable call is held and the turn has gone quiet.

        Runs beside the notification loop, which cannot end by itself while a
        call is held (no notifications arrive, and the tool timeout is a day).
        So a watcher failure must not go unnoticed: it is recorded as the run
        error and Codex is stopped, which ends the stream.
        """
        try:
            await self._watch_until_suspended(
                codex, shim, host, tracker, suspend, codex_home, thread_id
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("The durable-tool suspend watcher failed")
            if suspend.error is None:
                suspend.error = UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.ROLLOUT_CHECKPOINT_FAILED,
                    "Cannot suspend the Codex run",
                    f"The suspend watcher failed: {e}",
                    UiPathErrorCategory.SYSTEM,
                )
            host.close()
            await self._force_stop(codex)

    async def _watch_until_suspended(
        self,
        codex: Any,
        shim: CodexGatewayShim,
        host: DurableToolHost,
        tracker: _ItemTracker,
        suspend: _SuspendState,
        codex_home: str,
        thread_id: str,
    ) -> None:
        first_held: float | None = None
        rollout_size = -1
        rollout_changed = time.monotonic()
        while True:
            await asyncio.sleep(WATCH_INTERVAL_SEC)
            held = host.held
            if not held:
                first_held = None
                continue
            now = time.monotonic()
            if first_held is None:
                first_held = now
            try:
                path = find_rollout(codex_home, thread_id)
                size = path.stat().st_size if path is not None else -1
            except OSError:
                size = -1
            if size != rollout_size:
                rollout_size, rollout_changed = size, now
            if host.starting_count:
                continue
            last = max(
                tracker.last_activity,
                host.last_activity,
                shim.last_activity,
                rollout_changed,
            )
            if now - last < QUIET_WINDOW_SEC:
                continue
            running = tracker.running_non_durable()
            # Durable calls codex started whose tools/call has not reached the
            # host yet (by count: every arrived call is held, starting or
            # being answered with a simulated result).
            arriving = len(tracker.running_durable()) > (
                len(held) + host.starting_count + getattr(host, "mocking_count", 0)
            )
            # Immediate calls whose work runs on the host (they may reach it
            # before their item/started arrives).
            hosted = getattr(host, "inflight_count", 0)
            busy = shim.inflight > 0 or bool(running) or arriving or hosted > 0
            if busy and now - first_held < self.agent.durable_settle_timeout:
                continue
            if busy:
                logger.warning(
                    "Suspending after %.0fs although call(s) %s are still "
                    "running%s; they will be reported as not executed (UiPath "
                    "tool calls already running: as having an unknown outcome).",
                    self.agent.durable_settle_timeout,
                    running or list(getattr(host, "inflight_ids", [])),
                    " (and a durable call has not arrived)" if arriving else "",
                )
            # From here on a late tools/call starts nothing: the suspension
            # below would record it as "not executed".
            host.close()
            proc = self._kill_app_server(codex, suspend)
            if proc is None:
                # No way to stop it cleanly: fail the run (unblocks the stream).
                await codex.close()
                return
            try:
                await asyncio.to_thread(proc.wait, KILL_WAIT_SEC)
            except subprocess.TimeoutExpired:
                # SIGKILL was sent; a process stuck in the kernel cannot write
                # to the rollout any more either.
                logger.warning(
                    "The killed codex app-server (pid %s) has not exited after "
                    "%.0fs; suspending anyway.",
                    proc.pid,
                    KILL_WAIT_SEC,
                )
            suspend.killed = True
            return

    async def _stop_on_fatal(self, codex: Any, host: DurableToolHost) -> None:
        """Stop Codex once a tool failed the run (``host.fatal_error``)."""
        await host.fatal.wait()
        host.close()
        await self._force_stop(codex)

    @staticmethod
    async def _force_stop(codex: Any) -> None:
        """End the codex stream after a watcher failure (the run fails)."""
        sync = getattr(getattr(codex, "_client", None), "_sync", None)
        proc = getattr(sync, "_proc", None)
        try:
            if proc is not None and hasattr(proc, "kill"):
                children = _descendants(proc.pid)
                proc.kill()
                _kill_pids(children)
            else:
                await codex.close()
        except Exception:
            logger.debug("could not stop codex after a watcher failure", exc_info=True)

    @staticmethod
    def _kill_app_server(codex: Any, suspend: _SuspendState) -> Any:
        """SIGKILL the codex app-server — never a graceful stop.

        ``turn/interrupt``, ``close()`` and shutdown all persist the held call
        as "aborted by user" plus a ``<turn_aborted>`` marker, which the model
        would see on resume.
        """
        sync = getattr(getattr(codex, "_client", None), "_sync", None)
        proc = getattr(sync, "_proc", None)
        if proc is None or not hasattr(proc, "kill"):
            suspend.error = UiPathCodexRuntimeError(
                UiPathCodexErrorCode.SUSPEND_KILL_UNAVAILABLE,
                "Cannot suspend the Codex run",
                "The codex app-server process handle is not available in this "
                "openai-codex version, so the run cannot be suspended cleanly.",
                UiPathErrorCategory.SYSTEM,
            )
            return None
        children = _descendants(proc.pid)
        suspend.requested = True
        proc.kill()
        # MCP stdio servers and PTY sessions would outlive their parent.
        _kill_pids(children)
        return proc

    async def _finish_suspend(
        self,
        codex_home: str,
        thread_id: str,
        tracker: _ItemTracker,
        host: DurableToolHost | None = None,
    ) -> UiPathRuntimeResult:
        """Save the rollout and dangling calls; return SUSPENDED on the jobs."""
        session = self._require_session()
        await self._save_attachments()
        await self._save_results()
        path = find_rollout(codex_home, thread_id)
        if path is None:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.ROLLOUT_CHECKPOINT_FAILED,
                "Cannot suspend the Codex run",
                f"No rollout file for thread {thread_id} under {codex_home}.",
                UiPathErrorCategory.SYSTEM,
            )
        raw = await asyncio.to_thread(path.read_bytes)
        is_zst = path.name.endswith(".zst")
        data = raw if is_zst else cut_torn_tail(raw)
        pending = {
            cid: call
            for cid, call in (await session.get_pending()).items()
            if call.state == STATE_STARTED
        }
        if not pending:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.ROLLOUT_CHECKPOINT_FAILED,
                "Cannot suspend the Codex run",
                "The run was stopped for a durable tool call that has no saved job.",
                UiPathErrorCategory.SYSTEM,
            )
        dangling: list[DanglingCall] = []
        if not is_zst:
            for cid, itype in dangling_calls(parse_rollout(data)):
                if cid in pending:
                    dangling.append(DanglingCall(cid, itype, durable=True))
                    continue
                output = tracker.sibling_output(cid)
                if output is None and host is not None:
                    # An immediate call answered (or cut off) without its
                    # item/completed reaching us.
                    output = host.immediate_outcome(cid)
                if output is None:
                    if cid in tracker.started:
                        logger.warning(
                            "Sibling call %s started but its output is unknown; "
                            "the model is told it was not executed.",
                            cid,
                        )
                    output = NOT_EXECUTED_OUTPUT
                dangling.append(DanglingCall(cid, itype, durable=False, output=output))
            missing = [
                cid for cid in pending if cid not in {d.call_id for d in dangling}
            ]
            if missing:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.ROLLOUT_CHECKPOINT_FAILED,
                    "Cannot suspend the Codex run",
                    f"The rollout does not hold the pending call(s) {missing}.",
                    UiPathErrorCategory.SYSTEM,
                )
        rel = os.path.relpath(path, codex_home)
        await session.set_rollout(encode_rollout(rel, data))
        await session.set_thread(ThreadRecord(thread_id, rel))
        await session.set_dangling(dangling)
        await session.set_completed({})
        logger.info(
            "Codex run suspended on %d durable call(s): %s",
            len(pending),
            list(pending),
        )
        return self._suspended_result(pending)

    @staticmethod
    def _suspended_result(pending: dict[str, PendingCall]) -> UiPathRuntimeResult:
        # Live models, not dicts: a dict would silently become an API trigger.
        ordered = sorted(pending.values(), key=lambda c: c.order)
        return UiPathRuntimeResult(
            output={call.call_id: call.suspend_value() for call in ordered},
            status=UiPathRuntimeStatus.SUSPENDED,
            # Read by the factory's resumable wrapper to show a slim summary
            # of the suspension instead of the full trigger values.
            metadata={
                DURABLE_CALLS_METADATA: {
                    call.call_id: {"tool": call.tool, "arguments": call.args}
                    for call in ordered
                }
            },
        )

    # --- resume -------------------------------------------------------------

    async def _prepare_resume(self, input: Any) -> _ResumePlan | UiPathRuntimeResult:
        """Finish fired calls; re-suspend on the rest, or plan the injection."""
        session = self._require_session()
        thread = await session.get_thread()
        rollout = await session.get_rollout()
        pending = await session.get_pending()
        if thread is None or rollout is None:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.SESSION_NOT_FOUND,
                "Nothing to resume",
                "No suspended Codex session was found for this run in the state "
                "database.",
                UiPathErrorCategory.USER,
            )
        record = await session.get_entrypoint()
        if record is not None and record.entrypoint != self.entrypoint:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.ENTRYPOINT_MISMATCH,
                "Resumed on a different entrypoint",
                f"The run was suspended on '{record.entrypoint}' and cannot be "
                f"resumed on '{self.entrypoint}'.",
                UiPathErrorCategory.USER,
            )
        pending = {cid: c for cid, c in pending.items() if c.state == STATE_STARTED}
        completed = await session.get_completed()
        dangling = await session.get_dangling()
        if not pending and not completed and not dangling:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.SESSION_NOT_FOUND,
                "Nothing to resume",
                "The Codex run is not suspended on any durable tool call.",
                UiPathErrorCategory.USER,
            )
        tools = self._durable_tools()
        fired = await self._resume_values(input, pending)
        attachments_out: dict[str, list[Attachment]] = {}
        done: list[str] = []
        still_running = False
        for cid, value in fired.items():
            call = pending[cid]
            tool = tools.get(call.tool)
            if tool is None:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.TOOL_NOT_FOUND,
                    "Durable tool not found",
                    f"Call {cid} was made to tool '{call.tool}', which the agent no "
                    "longer declares.",
                    UiPathErrorCategory.USER,
                )
            try:
                text = await tool.finish(call.args, value)
            except UiPathRunTerminatedError as e:
                # The result ends the run: close this call's spans with it,
                # keep what finished before it, then fault the run.
                self._trace_resumed_call(call, tool, e.result_text)
                self._completed_spans.add(cid)
                if done:
                    await session.set_completed(completed)
                    await session.remove_pending(done)
                raise
            except UiPathDurableToolPending as e:
                logger.info("durable call %s is still pending: %s", cid, e)
                still_running = True
                continue
            except Exception as e:
                logger.exception(
                    "durable tool %s failed to finish call %s", call.tool, cid
                )
                text = json.dumps({"status": "error", "error": str(e)})
                key = getattr(tool, "result_key", None)
                if isinstance(key, str):
                    self._results.clear_pending(key)
            else:
                harvested = self._harvest(tool, text)
                if harvested:
                    attachments_out[cid] = harvested
                self._store_result(tool, call.args, value)
            completed[cid] = text
            done.append(cid)
            self._resumed_events.append(
                _resumed_tool_event(call, text, self.runtime_id)
            )
            self._trace_resumed_call(call, tool, text, attachments_out.get(cid))
            self._completed_spans.add(cid)
        # Before the calls are marked done: a crash in between would lose the
        # results of calls that never run again.
        await self._save_results()
        if done:
            await session.set_completed(completed)
            await session.remove_pending(done)
        await self._save_attachments()
        remaining = {cid: c for cid, c in pending.items() if cid not in done}
        if remaining:
            if still_running:
                await asyncio.sleep(RESUSPEND_BACKOFF_SEC)
            logger.info("Codex run still waiting on %s", list(remaining))
            return self._suspended_result(remaining)

        items: list[dict[str, Any]] = []
        if dangling:
            for d in dangling:
                output = completed.get(d.call_id) if d.durable else d.output
                items.append(
                    {
                        "type": d.output_type,
                        "call_id": d.call_id,
                        "output": output if output is not None else NOT_EXECUTED_OUTPUT,
                    }
                )
        else:
            items = [
                {"type": "function_call_output", "call_id": cid, "output": text}
                for cid, text in completed.items()
            ]
        return _ResumePlan(thread_id=thread.thread_id, items=items)

    async def _resume_values(
        self, input: Any, pending: dict[str, PendingCall]
    ) -> dict[str, Any]:
        """``{call_id: resume_value}`` of the pending calls that can finish now.

        An input keyed by pending call ids, or a bare value for the only
        pending call, is an explicit answer (:func:`normalize_resume_input`).
        ``None`` means the resumable wrapper already read the stored triggers.

        An input that answers no call (``{}``, the run's own agent input as
        passed again by ``uipath run --file input.json --resume``, or an
        unkeyed value while several calls are pending) is not an answer, but
        the wrapper took it as one and deleted the trigger of a single pending
        call. Read those calls' jobs here, so the run does not use the input
        as a job result, nor re-suspend only for the wrapper's fired-trigger
        loop to find the job finished in a second cycle.
        """
        ordered = sorted(pending.values(), key=lambda c: c.order)
        ids = [c.call_id for c in ordered]
        if input is None or (
            isinstance(input, dict) and any(k in pending for k in input)
        ):
            return normalize_resume_input(input, ids)
        if input == {}:
            # Also what the wrapper passes when no stored trigger fired: the
            # calls that still have one were just read.
            return await self._read_pending_jobs(ordered, stored_too=False)
        if len(ids) == 1 and input != await self._load_run_input():
            return normalize_resume_input(input, ids)
        logger.info(
            "Resume input answers no pending call (%s); reading their jobs", ids
        )
        return await self._read_pending_jobs(ordered, stored_too=True)

    async def _read_pending_jobs(
        self, calls: list[PendingCall], *, stored_too: bool
    ) -> dict[str, Any]:
        """Read the triggers of ``calls``; return the values of fired ones.

        A call whose stored trigger was deleted gets one rebuilt from its saved
        suspend value (:func:`_rebuilt_trigger`): only for a ``WaitJob``, a
        ``WaitTask`` or a ``WaitDeepRag``, as building a trigger from other values may start
        work (``InvokeProcess`` starts a job). ``stored_too`` also reads calls whose trigger is still stored.
        The stored triggers of fired calls are deleted, as the wrapper does,
        so a later suspension does not carry them over and hand their job to
        another call.
        """
        get_triggers = getattr(self.storage, "get_triggers", None)
        stored: dict[str, list[Any]] = {}
        if get_triggers is not None:
            for trigger in await get_triggers(self.runtime_id) or []:
                stored.setdefault(trigger.interrupt_id, []).append(trigger)
        handler = CodexResumeTriggerHandler()
        fired: dict[str, Any] = {}
        used: list[Any] = []
        for call in calls:
            triggers = stored.get(call.call_id)
            if triggers and not stored_too:
                continue
            try:
                if not triggers:
                    trigger = await _rebuilt_trigger(handler, call.suspend_value())
                    if trigger is None:
                        continue
                    triggers = [trigger]
                fired[call.call_id] = await handler.read_trigger(triggers[0])
            except UiPathPendingTriggerError:
                continue
            except Exception:
                logger.warning(
                    "Could not read the result of durable call %s",
                    call.call_id,
                    exc_info=True,
                )
                continue
            used.extend(stored.get(call.call_id) or [])
        delete_triggers = getattr(self.storage, "delete_triggers", None)
        if used and delete_triggers is not None:
            await delete_triggers(self.runtime_id, used)
        return fired

    async def _save_run_input(self, input: Any) -> None:
        """Remember the agent input of a fresh run (see ``RUN_INPUT_KEY``)."""
        if self.storage is None:
            return
        try:
            await self.storage.set_value(
                self.runtime_id, RUN_INPUT_NAMESPACE, RUN_INPUT_KEY, {"value": input}
            )
        except Exception:  # not storable: a resume then never matches it
            logger.debug("Could not save the run input", exc_info=True)

    async def _load_run_input(self) -> Any:
        if self.storage is None:
            return None
        raw = await self.storage.get_value(
            self.runtime_id, RUN_INPUT_NAMESPACE, RUN_INPUT_KEY
        )
        return raw.get("value") if isinstance(raw, dict) else None

    async def _save_attachments(self, *, force: bool = False) -> None:
        """Remember the known attachments for a resume (durable runs only)."""
        if self.storage is None or not (force or self._attachments.changed):
            return
        try:
            await self.storage.set_value(
                self.runtime_id,
                RUN_INPUT_NAMESPACE,
                RUN_ATTACHMENTS_KEY,
                {"value": self._attachments.to_json()},
            )
            self._attachments.changed = False
        except Exception:
            logger.warning("Could not save the run's job attachments", exc_info=True)

    async def _load_attachments(self) -> AttachmentStore:
        """The attachments a suspended run knew: its input's, then the saved ones."""
        store = AttachmentStore()
        run_input = await self._load_run_input()
        if isinstance(run_input, dict):
            for attachment in self.input_attachments(run_input):
                store.register(attachment)
        if self.storage is not None:
            raw = await self.storage.get_value(
                self.runtime_id, RUN_INPUT_NAMESPACE, RUN_ATTACHMENTS_KEY
            )
            saved = raw.get("value") if isinstance(raw, dict) else None
            for attachment in AttachmentStore.from_json(saved).attachments:
                store.register(attachment)
        store.changed = False
        return store

    def _store_result(self, tool: Any, args: Any, resume_value: Any) -> None:
        """Keep the result a durable tool offers (``stored_result``)."""
        hook = getattr(tool, "stored_result", None)
        if not callable(hook):
            return
        try:
            stored = hook(args, resume_value)
        except Exception:
            logger.warning("Could not keep the result of %s", tool.name, exc_info=True)
            return
        if stored is not None:
            key, value = stored
            self._results.put(key, value)

    async def _save_results(self, *, force: bool = False) -> None:
        if self.storage is None or not (force or self._results.changed):
            return
        try:
            await self.storage.set_value(
                self.runtime_id,
                RUN_INPUT_NAMESPACE,
                RUN_RESULTS_KEY,
                {"value": self._results.to_json()},
            )
            self._results.changed = False
        except Exception:
            logger.warning("Could not save the run's tool results", exc_info=True)

    async def _load_results(self) -> ToolResultStore:
        if self.storage is None:
            return ToolResultStore()
        raw = await self.storage.get_value(
            self.runtime_id, RUN_INPUT_NAMESPACE, RUN_RESULTS_KEY
        )
        return ToolResultStore.from_json(
            raw.get("value") if isinstance(raw, dict) else None
        )

    def _harvest(self, tool: Any, text: str) -> list[Attachment]:
        """Register the attachments a durable tool's result carries."""
        paths = output_paths(tool)
        if not paths:
            return []
        try:
            data = json.loads(text)
        except ValueError:
            return []
        return self._attachments.harvest(paths, data)

    # --- job attachments --------------------------------------------------

    def input_attachments(self, input: dict[str, Any]) -> list[Attachment]:
        """The attachments of a run input (none when it does not validate)."""
        if self.agent.input_schema is None or not isinstance(input, dict):
            return []
        try:
            return model_attachments(self.agent.input_schema(**input))
        except Exception:
            return []

    def output_attachments(self, output: Any) -> list[Attachment]:
        """The attachments of the run output (as built by ``_build_output``)."""
        if self.agent.output_schema is None:
            return []
        paths = attachment_paths(self.agent.output_schema.model_json_schema())
        found = (to_attachment(v) for v in values_at(output, paths))
        return [a for a in found if a is not None]

    def _uses_attachments(self) -> bool:
        if model_has_attachments(self.agent.input_schema) or model_has_attachments(
            self.agent.output_schema
        ):
            return True
        durable, immediate = self._agent_tools()
        return any(input_paths(t) for t in [*durable.values(), *immediate.values()])

    def _developer_instructions(self, tools: dict[str, UiPathDurableTool]) -> str:
        """The thread's developer instructions (the same on start and resume)."""
        parts: list[str] = []
        if tools and self.agent.durable_tool_instructions:
            parts.append(DURABLE_INSTRUCTIONS)
        if self._uses_attachments():
            text = ATTACHMENT_INSTRUCTIONS
            if model_has_attachments(self.agent.output_schema):
                text += " " + OUTPUT_ATTACHMENT_INSTRUCTIONS
            parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _trace_resumed_call(
        call: PendingCall,
        tool: UiPathDurableTool,
        text: str,
        attachments: list[Attachment] | None = None,
    ) -> None:
        """Complete the call's ``Tool call`` span with the job's result.

        The span was left open when the job started (the job hangs under it).
        A call saved without one (before the job was known to have started,
        or by an older version) gets a short ``resumed`` span instead.
        """
        try:
            result: Any = json.loads(text)
        except ValueError:
            result = text
        status = result.get("status") if isinstance(result, dict) else None
        error = (
            f"{'job' if _call_job_key(call) else 'call'} {status}"
            if status in ("faulted", "error", "failed")
            else None
        )
        out = span_attachments(attachments, "out") if attachments else None
        if call.span is not None:
            complete_tool_call(call.span, output=result, error=error, attachments=out)
            return
        span = start_tool_call(
            call.tool,
            tool_type=tool_type_of(tool, getattr(tool, "kind", "durable")),
            call_id=call.call_id,
            arguments=call.args,
            resumed=True,
            job_key=_call_job_key(call),
            deep_rag_id=_call_operation(call).get("deep_rag_id"),
            batch_transform_id=_call_operation(call).get("batch_transform_id"),
            extraction_id=_call_operation(call).get("extraction_id"),
            validation_operation_id=_call_operation(call).get(
                "validation_operation_id"
            ),
            task_id=_call_operation(call).get("task_id"),
            attachments=out,
        )
        end_tool_call(span, output=result, error=error)

    async def fail_open_tool_spans(self, error: str) -> None:
        """Complete every still-open durable ``Tool call`` span with ``error``.

        Called by the factory's wrapper when the run faults: no later
        execution will use those jobs' results.
        """
        if self._session is None or not self.agent.tools:
            return
        for call in (await self._session.get_pending()).values():
            if call.span is None:
                continue
            if call.call_id not in self._completed_spans:
                complete_tool_call(call.span, error=error)
            call.span = None
            await self._session.put_pending(call)

    async def _restore_rollout(self, codex_home: str, thread_id: str) -> None:
        rollout = await self._require_session().get_rollout()
        if rollout is None:  # pragma: no cover - checked in _prepare_resume
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.SESSION_NOT_FOUND,
                "Nothing to resume",
                "The suspended Codex rollout is missing from the state database.",
                UiPathErrorCategory.USER,
            )
        target = await asyncio.to_thread(write_rollout, codex_home, thread_id, rollout)
        logger.debug("restored codex rollout to %s", target)

    @staticmethod
    async def _inject_items(
        codex: Any, thread_id: str, items: list[dict[str, Any]]
    ) -> None:
        """Pair the results with their calls via raw ``thread/inject_items``.

        The typed SDK has no ``inject_items``; ``turn/start`` tool output would
        drop the call id. A wrong call id is silently dropped by codex, and no
        injection at all makes it write "aborted".
        """
        if not items:
            return
        sync = getattr(getattr(codex, "_client", None), "_sync", None)
        request_raw = getattr(sync, "_request_raw", None)
        if request_raw is None:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.EXECUTION_FAILED,
                "Cannot resume the Codex run",
                "thread/inject_items is not reachable in this openai-codex version.",
                UiPathErrorCategory.SYSTEM,
            )
        await asyncio.to_thread(
            request_raw, "thread/inject_items", {"threadId": thread_id, "items": items}
        )

    # --- helpers ----------------------------------------------------------

    def _durable_tools(self) -> dict[str, UiPathDurableTool]:
        return self._agent_tools()[0]

    def _agent_tools(
        self,
    ) -> tuple[dict[str, UiPathDurableTool], dict[str, UiPathTool]]:
        """``(durable, immediate)`` tools of the agent, by name."""
        durable: dict[str, UiPathDurableTool] = {}
        immediate: dict[str, UiPathTool] = {}
        for tool in self.agent.tools or []:
            name = getattr(tool, "name", None)
            if name in durable or name in immediate:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.CONFIG_INVALID,
                    "Duplicate tool name",
                    f"Two tools are named '{name}'.",
                    UiPathErrorCategory.USER,
                )
            is_durable = is_durable_tool(tool)
            is_immediate = isinstance(tool, UiPathTool)
            if is_durable == is_immediate:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.CONFIG_INVALID,
                    "Invalid tool",
                    f"Tool {name!r} must implement either UiPathDurableTool "
                    "(start/finish) or UiPathTool (call)"
                    + (", not both." if is_durable else "."),
                    UiPathErrorCategory.USER,
                )
            if is_durable:
                durable[tool.name] = cast(UiPathDurableTool, tool)
            else:
                immediate[tool.name] = cast(UiPathTool, tool)
        return durable, immediate

    def _require_session(self) -> CodexSessionStore:
        if self._session is None:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.CONFIG_INVALID,
                "No state storage",
                "Durable tools need the runtime state storage; create the runtime "
                "through UiPathCodexRuntimeFactory.",
                UiPathErrorCategory.SYSTEM,
            )
        return self._session

    def _codex_home_dir(self, tools: dict[str, UiPathDurableTool]) -> str | None:
        """``CODEX_HOME`` to run codex with, or None to leave it as it was.

        Durable tools need a run-scoped home (the rollout is saved from and
        restored into it). Agents without them keep codex's own resolution
        (``$CODEX_HOME`` or ``~/.codex``) unless ``agent.codex_home`` is set.
        """
        if self.agent.codex_home:
            path = self.agent.codex_home
        elif tools:
            path = self._codex_home or os.path.join(
                self._project_dir(), "__uipath", CODEX_HOME_DIR
            )
        else:
            return None
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        return path

    def _hermetic(self, tools: dict[str, UiPathDurableTool]) -> bool:
        if self.agent.hermetic is not None:
            return self.agent.hermetic
        return bool(tools)

    def _project_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.file_path)) or os.getcwd()

    def _render_prompt(self, input: dict[str, Any]) -> str:
        """The first user message; no side effects (the factory calls it too).

        An attachment renders as a JobAttachment JSON object. Attachments the
        prompt does not show (a template without their field, or a plain
        ``prompt``/``message`` input) are listed in a block after it.
        """
        fields = input
        model: Any = None
        attachment_fields: set[str] = set()
        if self.agent.input_schema is not None:
            try:
                model = self.agent.input_schema(**input)
                fields, attachment_fields = with_tickets(model, model.model_dump())
            except Exception as e:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.INPUT_INVALID,
                    "Invalid input",
                    f"Input does not match the agent input schema: {e}",
                    UiPathErrorCategory.USER,
                ) from e

        text = self._format_prompt(fields)
        if not attachment_fields:
            return text
        if self.agent.prompt is not None:
            hidden_fields = attachment_fields - set(prompt_fields(self.agent.prompt))
        elif text == _dumps_fields(fields):
            hidden_fields = set()  # the whole input is the prompt
        else:
            hidden_fields = attachment_fields
        hidden = [
            a
            for name in sorted(hidden_fields)
            for a in model_attachments(getattr(model, name, None))
        ]
        if not hidden:
            return text
        block = attachments_block(hidden)
        return f"{text}\n\n{block}" if text else block

    def _format_prompt(self, fields: dict[str, Any]) -> str:
        if self.agent.prompt is not None:
            try:
                return self.agent.prompt.format(**fields)
            except KeyError as e:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.INPUT_INVALID,
                    "Missing prompt field",
                    f"Prompt template references {e} which is not in the input.",
                    UiPathErrorCategory.USER,
                ) from e
            except (AttributeError, IndexError, TypeError) as e:
                # ``{order[id]}`` given a string, ``{order.city}`` given a dict.
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.INPUT_INVALID,
                    "Invalid prompt field",
                    f"Input does not fit the prompt template: {e}",
                    UiPathErrorCategory.USER,
                ) from e

        for key in ("prompt", "message", "input", "query"):
            value = fields.get(key)
            if isinstance(value, str) and value:
                return value
        return _dumps_fields(fields)

    def _item_to_event(self, item: dict[str, Any]) -> UiPathRuntimeEvent | None:
        """Map a completed Codex thread item to a UiPath runtime event.

        State events name the graph node the item ran on (``agent`` or
        ``tools``), so a debugger can match them against the schema graph.
        The console debugger prints only that name as the header (``● tools``);
        the payload keys it prints below (``command`` / ``server`` + ``tool``)
        and ``metadata["item_type"]`` still tell a shell command from a tool.
        """
        itype = item.get("type")
        if itype == "userMessage":
            return None  # don't echo the input back as an event
        if itype == "agentMessage":
            return UiPathRuntimeMessageEvent(
                payload={"text": item.get("text", ""), "phase": item.get("phase")},
                metadata={"source": "codex", "item_type": itype},
                execution_id=self.runtime_id,
            )
        if itype == "reasoning":
            # A reasoning item has `summary: list[str]` and `content: list[str]`
            # (no `text`). Prefer the summary, fall back to full content.
            parts = item.get("summary") or item.get("content") or []
            if isinstance(parts, list):
                text = "\n".join(p for p in parts if isinstance(p, str))
            else:
                text = str(parts)
            if not text:
                return (
                    None  # models often emit empty reasoning summaries; skip the noise
                )
            return UiPathRuntimeStateEvent(
                payload={"text": text},
                node_name=AGENT_NODE,
                execution_id=self.runtime_id,
                phase=UiPathRuntimeStatePhase.UPDATED,
                metadata={"source": "codex", "item_type": itype},
            )
        if itype == "commandExecution":
            return UiPathRuntimeStateEvent(
                payload={
                    "command": item.get("command"),
                    "output": item.get("aggregated_output"),
                    "status": item.get("status"),
                },
                node_name=TOOLS_NODE,
                execution_id=self.runtime_id,
                phase=UiPathRuntimeStatePhase.UPDATED,
                metadata={"source": "codex", "item_type": itype},
            )
        if itype == "mcpToolCall":
            return UiPathRuntimeStateEvent(
                payload={
                    "server": item.get("server"),
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "result": _mcp_result_text(item.get("result")),
                    "status": item.get("status"),
                    "error": item.get("error"),
                },
                node_name=TOOLS_NODE,
                execution_id=self.runtime_id,
                phase=UiPathRuntimeStatePhase.UPDATED,
                metadata={"source": "codex", "item_type": itype},
            )
        # fileChange / webSearch / others → generic state event
        return UiPathRuntimeStateEvent(
            payload=item,
            node_name=_node_for(itype),
            execution_id=self.runtime_id,
            phase=UiPathRuntimeStatePhase.UPDATED,
            metadata={"source": "codex", "item_type": itype},
        )

    def _final_text(self, items: list[dict[str, Any]]) -> str:
        """Final assistant text, matching the SDK's selection.

        Mirrors ``_final_assistant_response_from_items``: prefer the
        ``final_answer`` agent message; otherwise fall back to the last
        phase-less agent message — never a ``commentary`` message.
        """
        fallback: str | None = None
        for item in reversed(items):
            if item.get("type") != "agentMessage":
                continue
            phase = item.get("phase")
            if phase == "final_answer":
                return item.get("text") or ""
            if phase is None and fallback is None:
                fallback = item.get("text") or ""
        return fallback or ""

    def _build_output(self, final_text: str) -> dict[str, Any]:
        """The run output: ``output_schema``'s fields, else ``{"output": text}``.

        The model writes an attachment field as ``{"ID": ...}``; it must name
        an attachment the run knows, which fills in the rest. An unknown ID
        fails the run.
        """
        schema = self.agent.output_schema
        if schema is not None:
            paths = attachment_paths(schema.model_json_schema())
            data: Any = None
            if paths:
                try:
                    data = json.loads(final_text)
                except ValueError:
                    data = None
            if isinstance(data, dict):
                data, errors = self._attachments.resolve(paths, data)
                if errors:
                    raise UiPathCodexRuntimeError(
                        UiPathCodexErrorCode.OUTPUT_INVALID,
                        "Invalid output attachment",
                        "The agent output names a job attachment the run does not "
                        "know: " + " ".join(errors),
                        UiPathErrorCategory.USER,
                    )
            try:
                parsed = (
                    schema.model_validate(data)
                    if isinstance(data, dict)
                    else schema.model_validate_json(final_text)
                )
                return parsed.model_dump(mode="json", by_alias=True)
            except Exception:
                logger.debug("output did not match output_schema; returning raw text")
        return {"output": final_text}


__all__ = ["UiPathCodexRuntime", "normalize_resume_input"]
