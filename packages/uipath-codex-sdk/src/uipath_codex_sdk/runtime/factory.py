"""Factory for creating Codex SDK runtimes from codex.json configuration."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any, AsyncGenerator

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeEvent,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathRuntimeStorageProtocol,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory

from ..agent import UiPathCodexAgent
from ..attachments import span_attachments
from ..tools import is_durable_tool
from .chat_events import chat_attachments, extract_user_message, is_chat_input
from .chat_runtime import UiPathCodexChatRuntime, chat_parked
from .config import CodexConfig
from .errors import UiPathCodexErrorCode, UiPathCodexRuntimeError
from .loader import CodexAgentLoader
from .runtime import (
    CODEX_HOME_DIR,
    CONTINUE_PROMPT,
    DURABLE_CALLS_METADATA,
    UiPathCodexRuntime,
)
from .schema import get_entrypoint_schema
from .session_store import CodexSessionStore
from .storage import SqliteResumableStorage
from .telemetry import (
    attribute_values,
    complete_span,
    emit_agent_output,
    enable_gateway_model_spans,
    fail_span,
    propagated_attributes,
    push_running,
    reset_resumed_attributes,
    restored_span,
    resumed_attributes,
    set_attributes,
    span_record,
    start_agent_run,
)
from .triggers import CodexResumeTriggerHandler

logger = logging.getLogger(__name__)

_AGENT_TYPE_CODED = "uipath_coded"
_AGENT_FRAMEWORK = "codex_sdk"
_DEFAULT_RUNTIME_DIR = "__uipath"
_DEFAULT_STATE_FILE = "state.db"
_STATE_FILE_SIDECARS = ("-wal", "-shm")


class UiPathCodexRuntimeFactory:
    """Discovers entrypoints from codex.json and builds Codex runtimes.

    Every runtime is wrapped in ``UiPathResumableRuntime`` over a SQLite state
    database, so a durable-tool suspension is persisted as resume triggers and
    restored on the next ``--resume`` (or job resume).

    A conversational agent (``uipath.json`` → ``runtimeOptions.isConversational``,
    or a chat host's conversation id) runs on :class:`UiPathCodexChatRuntime`.
    """

    def __init__(self, context: UiPathRuntimeContext):
        self.context = context
        # Let the LLM Gateway record the Model run spans (under our LLM call
        # spans), as for every other UiPath agent.
        enable_gateway_model_spans()
        self._config: CodexConfig | None = None
        self._agent_cache: dict[str, UiPathCodexAgent] = {}
        self._agent_loaders: dict[str, CodexAgentLoader] = {}
        self._lock = asyncio.Lock()
        self._storage_lock = asyncio.Lock()
        self._storage: SqliteResumableStorage | None = None
        self._codex_home: str | None = None

    def _load_config(self) -> CodexConfig:
        if self._config is None:
            self._config = CodexConfig()
        return self._config

    def discover_entrypoints(self) -> list[str]:
        config = self._load_config()
        if not config.exists:
            return []
        return config.entrypoint

    async def get_storage(self) -> UiPathRuntimeStorageProtocol | None:
        return await self._get_storage()

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        return UiPathRuntimeFactorySettings(
            agent_type=_AGENT_TYPE_CODED,
            agent_framework=_AGENT_FRAMEWORK,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        agent = await self._resolve_agent(entrypoint)
        loader = self._agent_loaders[entrypoint]
        storage = await self._get_storage()
        conversational = self._is_conversational()
        options: dict[str, Any] = {
            "agent": agent,
            "runtime_id": runtime_id,
            "entrypoint": entrypoint,
            "file_path": os.path.abspath(loader.file_path),
            "storage": storage,
            # Only durable and conversational agents run on a run-scoped
            # CODEX_HOME; the others (immediate tools only, too) keep codex's
            # own resolution (and nothing of theirs is wiped).
            "codex_home": (
                self._get_codex_home()
                if (
                    conversational or any(is_durable_tool(t) for t in agent.tools or [])
                )
                and not agent.codex_home
                else None
            ),
        }
        delegate: UiPathCodexRuntime = (
            UiPathCodexChatRuntime(exchange_id=self.context.exchange_id, **options)
            if conversational
            else UiPathCodexRuntime(**options)
        )
        return UiPathCodexResumableRuntime(
            delegate=delegate,
            storage=storage,
            trigger_manager=CodexResumeTriggerHandler(),
            runtime_id=runtime_id,
            conversational=conversational,
            conversation_id=self.context.conversation_id,
            exchange_id=self.context.exchange_id,
        )

    async def dispose(self) -> None:
        for loader in self._agent_loaders.values():
            await loader.cleanup()
        self._agent_loaders.clear()
        self._agent_cache.clear()
        async with self._storage_lock:
            storage, self._storage = self._storage, None
        if storage is not None:
            await storage.dispose()

    def _is_conversational(self) -> bool:
        """Whether the agent runs as a conversational (chat) agent.

        The chat host's conversation id is the live signal; otherwise
        ``uipath.json`` → ``runtimeOptions.isConversational`` decides (what
        ``uipath pack`` copies into the package, and what ``uipath dev``
        runs locally). A missing or unreadable file means no.
        """
        if self.context.conversation_id is not None:
            return True
        config_path = self.context.config_path or "uipath.json"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, ValueError):
            return False
        options = config.get("runtimeOptions") if isinstance(config, dict) else None
        return isinstance(options, dict) and bool(options.get("isConversational"))

    # --- state ------------------------------------------------------------

    def _should_reset_state(self) -> bool:
        """Whether a run starts from empty state (state.db and CODEX_HOME).

        langchain's rule: only the ``runtime_dir`` + ``state_file`` location is
        reset (never an explicit ``state_file_path``), and only on a fresh
        local run. A resumed run either carries the resume flag or runs under
        a job id, and reads its session out of the state database.
        """
        return (
            self.context.state_file_path is None
            and bool(self.context.runtime_dir)
            and bool(self.context.state_file)
            and not self.context.resume
            and self.context.job_id is None
            and not self.context.keep_state_file
        )

    def _runtime_dir(self) -> str:
        return self.context.runtime_dir or _DEFAULT_RUNTIME_DIR

    def _get_storage_path(self) -> str:
        if self.context.state_file_path is not None:
            return self.context.state_file_path
        if not (self.context.runtime_dir and self.context.state_file):
            os.makedirs(_DEFAULT_RUNTIME_DIR, exist_ok=True)
            return os.path.join(_DEFAULT_RUNTIME_DIR, _DEFAULT_STATE_FILE)
        path = os.path.join(self.context.runtime_dir, self.context.state_file)
        if self._should_reset_state():
            for candidate in (path, *(path + s for s in _STATE_FILE_SIDECARS)):
                if os.path.exists(candidate):
                    try:
                        os.remove(candidate)
                    except OSError:
                        logger.warning("Could not reset the state file %s", candidate)
        os.makedirs(self.context.runtime_dir, exist_ok=True)
        return path

    async def _get_storage(self) -> SqliteResumableStorage:
        if self._storage is not None:
            return self._storage
        async with self._storage_lock:
            if self._storage is None:
                self._storage = SqliteResumableStorage(self._get_storage_path())
            return self._storage

    def _get_codex_home(self) -> str:
        """Run-scoped ``CODEX_HOME``, reset under the same rule as state.db."""
        if self._codex_home is None:
            path = os.path.abspath(os.path.join(self._runtime_dir(), CODEX_HOME_DIR))
            if self._should_reset_state() and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
            self._codex_home = path
        return self._codex_home

    # --- helpers ----------------------------------------------------------

    async def _resolve_agent(self, entrypoint: str) -> UiPathCodexAgent:
        async with self._lock:
            if entrypoint in self._agent_cache:
                return self._agent_cache[entrypoint]

            config = self._load_config()
            if not config.exists:
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.CONFIG_MISSING,
                    "Invalid configuration",
                    "Failed to load codex.json configuration",
                    UiPathErrorCategory.DEPLOYMENT,
                )
            if entrypoint not in config.agents:
                available = ", ".join(config.entrypoint)
                raise UiPathCodexRuntimeError(
                    UiPathCodexErrorCode.AGENT_NOT_FOUND,
                    "Agent not found",
                    f"Agent '{entrypoint}' not found. Available: {available}",
                    UiPathErrorCategory.DEPLOYMENT,
                )

            loader = CodexAgentLoader.from_path_string(
                entrypoint, config.agents[entrypoint]
            )
            self._agent_loaders[entrypoint] = loader
            agent = await loader.load()
            self._agent_cache[entrypoint] = agent
            return agent


class UiPathCodexResumableRuntime(UiPathResumableRuntime):
    """``UiPathResumableRuntime`` that traces the job and reports a suspension
    compactly.

    A job is one ``Agent run`` span across all its executions, as for a
    low-code agent. It is current while the resume triggers are read and the
    Codex run executes, so every span of the job is under it:

    * the first execution opens it. On a suspension it is not ended: it is
      pushed to LLMOps as still running (``status=suspended``) and its
      identity is saved in the state database;
    * a resumed execution parents its spans directly under the saved span,
      as low-code agents do;
    * the execution that ends the job completes the saved span (same id and
      start time) with ``status`` = successful or faulted, after an ``Agent
      output`` span on success.

    A fresh (non-resume) execution forgets any saved span. An in-process
    resume (fired triggers) runs inside the execution's span.

    The delegate's suspended ``output`` must stay the typed suspend values
    (``WaitJobRaw``...): the base class turns them into resume triggers. Once
    the triggers exist, the returned result's ``output`` is only what
    ``uipath run`` prints and what lands in ``output.json`` (Orchestrator
    resumes from ``resumeTriggers``), so it is replaced by
    ``{call_id: {tool, arguments, job_key, job_state, folder_key}}``.

    A ``conversational`` runtime traces each chat exchange as its own
    ``Conversational agent run`` span: an exchange that ends (SUSPENDED with
    no trigger) completes it, and only an exchange parked on durable calls
    saves it for the execution that resumes the exchange.
    """

    def __init__(
        self,
        *args: Any,
        conversational: bool = False,
        conversation_id: str | None = None,
        exchange_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conversational = conversational
        self.conversation_id = conversation_id
        self.exchange_id = exchange_id

    async def _restore_resume_input(
        self, input: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Keep the stored triggers when a chat message is the resume input.

        The base class takes any explicit input as the answer to the stored
        trigger(s) and deletes them. A user message answers no durable call:
        the exchange suspends again on the same calls (with their triggers).
        """
        if self.conversational and (is_chat_input(input) or isinstance(input, str)):
            return input
        return await super()._restore_resume_input(input)

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        run = await self._open_agent_run(input, options)
        try:
            result = await super().execute(input, options)
            await run.finish(result)
            return result
        except BaseException as e:
            await run.fail(e)
            raise
        finally:
            run.detach()

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        run = await self._open_agent_run(input, options)
        try:
            async for event in super().stream(input, options):
                if isinstance(event, UiPathRuntimeResult):
                    await run.finish(event)
                yield event
        except BaseException as e:
            await run.fail(e)
            raise
        finally:
            run.detach()

    def _trace_store(self) -> CodexSessionStore | None:
        if not hasattr(self.storage, "get_value"):  # pragma: no cover
            return None
        return CodexSessionStore(self.storage, self.runtime_id)  # type: ignore[arg-type]

    async def _open_agent_run(self, input: Any, options: Any) -> _AgentRun:
        # Read before the run: the in-process resume loop sets ``resume``.
        resumed = bool(options is not None and getattr(options, "resume", False))
        store = self._trace_store()
        saved: dict[str, Any] | None = None
        if store is not None:
            try:
                saved = await store.get_trace()
                # A chat exchange parked on durable calls keeps its span even
                # when the next run is not a resume (a fresh execution on the
                # same runtime id). Only a parked exchange: a faulted one may
                # leave pending calls, but its span has ended.
                parked = (
                    self.conversational
                    and saved is not None
                    and chat_parked(await store.get_chat())
                )
                if not resumed and saved is not None and not parked:
                    await store.set_trace(None)
                    saved = None
            except Exception:
                logger.warning("Could not read the saved trace context", exc_info=True)
                saved = None
        delegate = self.delegate
        # Before the restored root becomes current: its children cannot
        # inherit ``execution.id`` from a non-recording parent.
        propagated = propagated_attributes()
        root: Span | None = None
        if saved is not None:
            try:
                root = restored_span(saved)
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring an unusable saved trace context")
                if store is not None:
                    try:
                        await store.set_trace(None)
                    except Exception:
                        logger.warning("Could not clear the saved trace context")
        if saved is None or root is None:
            # A new Agent run (also on resume when nothing was saved).
            return _AgentRun(
                self._start_agent_span(input, options),
                store=store,
                delegate=delegate,
                conversational=self.conversational,
            )
        return _AgentRun(
            root,
            store=store,
            delegate=delegate,
            saved=saved,
            propagated=propagated,
            conversational=self.conversational,
        )

    def _start_agent_span(self, input: Any, options: Any) -> Span:
        runtime = self.delegate
        resumed = bool(options is not None and getattr(options, "resume", False))
        if not isinstance(runtime, UiPathCodexRuntime):  # pragma: no cover
            return trace.get_tracer(__name__).start_span("Agent run")
        agent = runtime.agent
        if isinstance(runtime, UiPathCodexChatRuntime):
            return self._start_exchange_span(runtime, input)
        user_prompt: str | None = CONTINUE_PROMPT if resumed else None
        if not resumed:
            try:
                user_prompt = runtime._render_prompt(input or {})
            except Exception:
                user_prompt = None  # invalid input: the run itself reports it
        schema = get_entrypoint_schema(agent, runtime.entrypoint)
        return start_agent_run(
            runtime.entrypoint,
            agent_name=agent.name if agent.name != "agent" else runtime.entrypoint,
            model=agent.model,
            input=None if resumed else input,
            system_prompt=agent.base_instructions,
            user_prompt=user_prompt,
            input_schema=schema.input,
            output_schema=schema.output,
            resumed=resumed,
            attachments=(
                None
                if resumed
                else span_attachments(runtime.input_attachments(input or {}), "in")
            ),
        )

    def _start_exchange_span(self, runtime: UiPathCodexChatRuntime, input: Any) -> Span:
        """A chat exchange's span, named and shaped as low-code's.

        Its ``input`` is the exchange input (the messages); there is no
        ``userPrompt``, and no ``resumed``: a span opened here is always a
        new exchange (every exchange after the first is a job resume in
        Orchestrator), and the durable resume of a parked exchange reuses its
        saved span. ``threadId`` is set when the exchange ends.
        """
        agent = runtime.agent
        schema = get_entrypoint_schema(agent, runtime.entrypoint, conversational=True)
        message = extract_user_message(input)
        return start_agent_run(
            runtime.entrypoint,
            agent_name=agent.name if agent.name != "agent" else runtime.entrypoint,
            model=agent.model,
            input=input if isinstance(input, dict) else None,
            system_prompt=agent.base_instructions,
            user_prompt=None,
            input_schema=schema.input,
            output_schema=schema.output,
            resumed=False,
            attachments=(
                span_attachments(chat_attachments(message), "in")
                if message is not None
                else None
            ),
            is_conversational=True,
            extra={
                "conversationId": self.conversation_id or self.runtime_id,
                "exchangeId": self.exchange_id,
            },
        )

    async def _handle_suspension(
        self, result: UiPathRuntimeResult
    ) -> UiPathRuntimeResult:
        handled = await super()._handle_suspension(result)
        if handled.status != UiPathRuntimeStatus.SUSPENDED or not isinstance(
            result.output, dict
        ):
            return handled
        calls = (result.metadata or {}).get(DURABLE_CALLS_METADATA) or {}
        summary: dict[str, Any] = {}
        for call_id, value in result.output.items():
            entry: dict[str, Any] = dict(calls.get(call_id) or {})
            job = getattr(value, "job", None)
            if job is not None:
                entry["job_key"] = getattr(job, "key", None)
                state = getattr(job, "state", None)
                entry["job_state"] = getattr(state, "value", state)
            deep_rag = getattr(value, "deep_rag", None)
            if deep_rag is not None:
                entry["deep_rag_id"] = getattr(deep_rag, "id", None)
                entry["deep_rag_status"] = getattr(
                    deep_rag, "last_deep_rag_status", None
                )
            extraction = getattr(value, "extraction", None)
            if extraction is not None and hasattr(extraction, "operation_id"):
                entry["extraction_id"] = extraction.operation_id
                entry["document_id"] = getattr(extraction, "document_id", None)
                entry["project_id"] = getattr(extraction, "project_id", None)
            validation = getattr(value, "extraction_validation", None)
            if validation is not None:
                entry["validation_operation_id"] = validation.operation_id
                entry["task_url"] = getattr(value, "task_url", None)
            batch = getattr(value, "batch_transform", None)
            if batch is not None:
                entry["batch_transform_id"] = getattr(batch, "id", None)
                entry["batch_transform_status"] = getattr(
                    batch, "last_batch_rag_status", None
                )
            task = getattr(value, "action", None)
            if task is not None and hasattr(task, "is_deleted"):
                entry["task_id"] = getattr(task, "id", None)
                entry["task_key"] = getattr(task, "key", None)
                entry["task_title"] = getattr(task, "title", None)
                entry["app_name"] = getattr(value, "app_name", None)
            folder_key = (
                getattr(value, "process_folder_key", None)
                or getattr(value, "index_folder_key", None)
                or getattr(value, "app_folder_key", None)
            )
            if folder_key:
                entry["folder_key"] = folder_key
            entry.setdefault("type", type(value).__name__)
            summary[call_id] = entry
        handled.output = summary
        return handled


def _error_text(error: Any) -> str:
    if error is None:
        return ""
    return str(getattr(error, "detail", "") or getattr(error, "title", "") or "")


class _AgentRun:
    """The job's ``Agent run`` span as one execution sees it.

    Either a live span this execution opened (``saved`` is None), or the span
    an earlier execution saved, restored as the (non-recording) parent of this
    execution's spans.
    """

    def __init__(
        self,
        root: Span,
        *,
        store: CodexSessionStore | None,
        delegate: Any,
        saved: dict[str, Any] | None = None,
        propagated: dict[str, Any] | None = None,
        conversational: bool = False,
    ) -> None:
        self.root = root
        self.saved = saved
        self._conversational = conversational
        # Attributes this execution's spans carry (``execution.id``) that the
        # restored root cannot hand down.
        self._propagated = dict(propagated or {})
        self._store = store
        self._delegate = delegate
        self._done = False
        self._token: object | None = context_api.attach(trace.set_span_in_context(root))
        # The restored root cannot hand ``execution.id`` down; set it directly.
        self._attributes_token = (
            resumed_attributes(self._propagated) if saved is not None else None
        )

    def _output_attachments(self, output: Any) -> str | None:
        runtime = self._delegate
        if not isinstance(runtime, UiPathCodexRuntime):
            return None
        return span_attachments(runtime.output_attachments(output), "out")

    def detach(self) -> None:
        if self._attributes_token is not None:
            reset_resumed_attributes(self._attributes_token)
            self._attributes_token = None
        if self._token is not None:
            _detach(self._token)
            self._token = None

    async def finish(self, result: UiPathRuntimeResult) -> None:
        if self._done:
            return
        self._done = True
        if (
            result.status == UiPathRuntimeStatus.SUSPENDED
            and self._conversational
            and not result.triggers
        ):
            # The chat exchange is over (the job waits for the next message).
            extra = await self._chat_attributes()
            await self._safely(self._complete("successful", extra=extra))
            return
        if result.status == UiPathRuntimeStatus.SUSPENDED:
            try:
                await self._suspend(result.output)
            except Exception:
                logger.warning("Could not suspend the Agent run span", exc_info=True)
            return
        if result.status == UiPathRuntimeStatus.SUCCESSFUL:
            try:
                emit_agent_output(
                    result.output,
                    context=trace.set_span_in_context(self.root),
                    attachments=self._output_attachments(result.output),
                    **self._propagated,
                )
            except Exception:
                logger.warning("Could not emit the Agent output span", exc_info=True)
            await self._safely(self._complete("successful", output=result.output))
            return
        status = str(getattr(result.status, "value", result.status)).lower()
        error = _error_text(result.error) or status
        # Each step on its own: a failing one must not leave the run open.
        await self._safely(self._fail_tool_spans(error))
        await self._safely(self._complete(status, error=error))

    async def fail(self, error: BaseException) -> None:
        if self._done:
            return
        self._done = True
        # ``CancelledError()`` and the like have no message: still an error.
        text = str(error) or type(error).__name__
        await self._safely(self._fail_tool_spans(text))
        if self.saved is None:
            try:
                fail_span(self.root, error)
                self.root.set_attribute("status", "faulted")
            except Exception:
                logger.warning("Could not fail the Agent run span", exc_info=True)
            finally:
                self.root.end()
        else:
            await self._safely(self._complete("faulted", error=text))

    @staticmethod
    async def _safely(step: Any) -> None:
        try:
            await step
        except Exception:
            logger.warning("Could not finish the Agent run trace", exc_info=True)

    async def _chat_attributes(self) -> dict[str, Any]:
        """``threadId`` of a chat exchange: the conversation's Codex thread."""
        if not self._conversational or self._store is None:
            return {}
        try:
            thread = await self._store.get_thread()
        except Exception:
            logger.debug("could not read the Codex thread id", exc_info=True)
            return {}
        return {"threadId": thread.thread_id} if thread is not None else {}

    async def _suspend(self, output: Any) -> None:
        if self.saved is not None:
            return  # still running; the saved span carries forward unchanged
        # Not ended: the next execution completes it.
        set_attributes(self.root, status="suspended", output=output)
        for key, value in (await self._chat_attributes()).items():
            self.root.set_attribute(key, value)
        push_running(self.root)
        record = span_record(self.root)
        if record is not None and self._store is not None:
            await self._store.set_trace(record)

    async def _complete(
        self,
        status: str,
        *,
        output: Any = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.saved is None:
            try:
                set_attributes(self.root, status=status, output=output)
                for key, value in (extra or {}).items():
                    self.root.set_attribute(key, value)
                if error is None:
                    self.root.set_status(Status(StatusCode.OK))
                else:
                    self.root.set_status(Status(StatusCode.ERROR, error))
                    self.root.set_attribute("error", error)
            finally:
                self.root.end()
            return
        try:
            complete_span(
                self.saved,
                attributes={
                    **self._propagated,
                    **(extra or {}),
                    **attribute_values(status=status, output=output),
                },
                error=error,
                # A faulted run has no output: not the suspension's summary.
                drop=("output", "output.value") if output is None else (),
            )
        finally:
            if self._store is not None:
                await self._store.set_trace(None)

    async def _fail_tool_spans(self, error: str) -> None:
        if isinstance(self._delegate, UiPathCodexRuntime):
            await self._delegate.fail_open_tool_spans(
                f"The agent run failed before the job result was used: {error}"
            )


def _detach(token: object) -> None:
    """Detach the agent run context, quietly when it cannot be.

    A stream closed from another context (an abandoned generator finalised by
    the event loop, or ``aclose()`` from another task) cannot reset the
    context variable; that context is gone anyway. ``context_api.detach``
    catches that ``ValueError`` itself and logs "Failed to detach context" at
    ERROR, so call the runtime context it wraps, which raises instead.
    """
    runtime_context = getattr(context_api, "_RUNTIME_CONTEXT", None)
    try:
        if runtime_context is not None:
            runtime_context.detach(token)
        else:  # pragma: no cover - a future OpenTelemetry without it
            context_api.detach(token)  # type: ignore[arg-type]
    except Exception:
        logger.debug("could not detach the agent run context", exc_info=True)


__all__ = ["UiPathCodexResumableRuntime", "UiPathCodexRuntimeFactory"]
