"""Runtime class for executing Claude Agent SDK agents within the UiPath framework."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from claude_agent_sdk import (
    TERMINAL_TASK_STATUSES,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    Message,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from opentelemetry import context as context_api
from pydantic import ValidationError
from uipath.core.serialization import serialize_json
from uipath.core.tracing import traced
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath.runtime.events import UiPathRuntimeEvent, UiPathRuntimeStateEvent
from uipath.runtime.schema import UiPathRuntimeSchema

from ..agent import ClaudeAgent
from ..interrupts import (
    PendingSuspend,
    SuspendChannel,
    ToolIndex,
    active_channel,
    uipath_tool_index,
)
from ._telemetry import GatewayCallTelemetry
from .errors import UiPathClaudeSDKErrorCode, UiPathClaudeSDKRuntimeError
from .gateway import GatewayShim, GatewayShimError
from .schema import get_agent_graph, get_entrypoints_schema
from .session_paths import ClaudeSessionPaths
from .session_store import ClaudeSessionStore, EntrypointRecord, TranscriptRecord
from .suspend import build_suspend_hooks

logger = logging.getLogger(__name__)
cli_logger = logger.getChild("cli")

RESUME_PROMPT = "Continue."

# These are the bounded task types whose completion causes Claude Code to run a follow-up turn.
_WAITED_BACKGROUND_TASK_TYPES = frozenset({"local_agent", "local_workflow"})


def _resume_shape_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """Say whether this run is resuming, without recording the payload.

    The input a resume carries is the answer a person or a platform trigger
    gave, so it is exactly the kind of thing not to copy onto a span.
    """
    return {"resuming": inputs.get("resuming")}


def log_cli_stderr(line: str) -> None:
    """Route one line of the CLI subprocess's stderr into the runtime's logs.

    The SDK pipes that stream only when a callback is registered, so without
    this the CLI's own diagnostics are inherited by the parent process: visible
    in a local terminal and nowhere else, not in execution.log, not in the dev
    server output, and not in a deployed job's logs. The SDK swallows anything
    this raises and drops nothing else, so it stays a single log call.
    """
    cli_logger.warning("claude cli: %s", line)


_SUSPEND_SAFETY_ENV = {"CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1"}
"""Environment the CLI subprocess needs for a suspension to be trustworthy.

On the non-streaming fallback the assistant message keeps its full content
array, which trips the CLI's solo-only guard: the defer is discarded and the
tool executes for real while the run reports success. Disabling the fallback
turns that into a loud API error instead. It is applied only to an agent that
can suspend, so an agent without UiPath tools
keeps the CLI's own retry behaviour.
"""


@dataclass
class RunContext:
    """Per-invocation state shared between the run loop and its helpers.

    Attributes:
        channel: The suspend channel this run's hook and UiPath tools share.
        sdk_options: Options handed to ``ClaudeSDKClient``.
        session_id: Claude session the SDK was asked to re-attach to, if any.
        pending: The deferred call this run is resolving, if any.
        resuming: Whether this invocation is resuming a suspended job.
    """

    channel: SuspendChannel
    sdk_options: ClaudeAgentOptions
    session_id: str | None
    pending: PendingSuspend | None
    resuming: bool = False


class UiPathClaudeSDKRuntime:
    """A runtime class for executing Claude Agent SDK agents within the UiPath framework.

    Owns the ``ClaudeSDKClient`` run loop and maps SDK messages to UiPath
    runtime events. LLM routing is an explicit opt-in: an agent declaring
    ``uipath_llm`` runs behind a local gateway shim pointed at the tenant,
    and an agent without one gets no injected LLM environment at all.

    Suspension is opt-in too. A run suspends durably when a tool the developer
    built with ``uipath_tool_server`` calls ``interrupt``: the CLI parks the
    call, the runtime persists the pending record and returns SUSPENDED, and a
    later process rebuilds the same hook, seeds the trigger payload and
    re-attaches to the Claude session so the parked call finishes with a real
    tool result.

    Args:
        agent: The loaded ClaudeAgent definition.
        session_store: Persists the Claude session id, the pending suspension
            and the entrypoint the run started on.
        session_paths: Durable Claude config and working directories, both
            inside the runtime directory so they survive a suspension.
        runtime_id: Unique identifier for this runtime instance.
        entrypoint: Agent entrypoint name (for schema reporting).
    """

    def __init__(
        self,
        agent: ClaudeAgent,
        session_store: ClaudeSessionStore,
        session_paths: ClaudeSessionPaths,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        self.agent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self._session_store = session_store
        self._session_paths = session_paths
        self._shim: GatewayShim | None = None
        # Assigned by the runtime factory, replaceable by a host.
        self.telemetry: GatewayCallTelemetry | None = None

    # --- Instrumentation ---------------------------------------------------

    def _adopt_span_parent(self, channel: SuspendChannel | None = None) -> None:
        """Give the turn to everything that traces outside its context.

        Only valid while the instrumentor's agent span is current, which is
        during iteration of the response. The gateway shim and the tool bodies
        both run in tasks that never see that span, so each is handed the
        context to attach instead.
        """
        current = context_api.get_current()
        if self.telemetry is not None:
            self.telemetry.adopt_parent(current)
        if channel is not None:
            channel.adopt_turn_parent(current)

    def _flush_gateway_spans(self) -> None:
        """Turn the run's upstream calls into the only LLM spans it emits.

        The agent and tool spans come from the Claude Agent SDK instrumentor,
        which sees no model call to report: the CLI makes them in its own
        process, so the gateway shim is the only observer of the real traffic.
        """
        if self.telemetry is not None:
            self.telemetry.flush_gateway_calls()

    # --- LLM access -------------------------------------------------------

    async def _build_llm_env(self) -> dict[str, str]:
        """Return the SDK subprocess env vars for LLM access.

        With ``agent.uipath_llm`` set, a gateway shim is started and the CLI
        is pointed at it. Without it nothing is injected and the Claude Agent
        SDK reaches Anthropic on whatever credentials it finds for itself.
        """
        llm = self.agent.uipath_llm
        if llm is None:
            logger.info(
                "LLM routing for '%s': direct Anthropic access, the agent declares "
                "no uipath_llm so no LLM environment is injected.",
                self.agent.name,
            )
            return {}

        telemetry = self.telemetry
        if self._shim is None:
            self._shim = GatewayShim(
                llm,
                on_call=telemetry.on_gateway_call if telemetry else None,
                trace_headers=telemetry.trace_context_headers if telemetry else None,
            )
        try:
            await self._shim.start()
        except Exception:
            await self._stop_llm_gateway()
            raise
        logger.info(
            "LLM routing for '%s': UiPath LLM Gateway, requested model '%s' "
            "resolved to '%s' behind %s.",
            self.agent.name,
            llm.model,
            self._shim.resolved_model,
            self._shim.base_url,
        )
        return self._shim.build_env()

    async def _stop_llm_gateway(self) -> None:
        """Stop the gateway shim if this runtime started one."""
        shim, self._shim = self._shim, None
        if shim is None:
            return
        try:
            await shim.stop()
        except Exception as e:
            logger.warning("Failed to stop the gateway shim cleanly: %s", e)

    # --- Input/output mapping ---------------------------------------------

    def _get_user_message(self, input: dict[str, Any]) -> str:
        """Build the user message string for client.query().

        Renders the agent's prompt template from validated input fields when
        declared; otherwise uses the 'input' field or a JSON dump of the input.
        """
        validated: dict[str, Any] = input
        if self.agent.input_schema is not None:
            try:
                model = self.agent.input_schema.model_validate(input)
            except ValidationError as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.INPUT_VALIDATION_ERROR,
                    "Invalid input",
                    f"Input does not match the agent's input schema: {e}",
                    UiPathErrorCategory.USER,
                ) from e
            validated = model.model_dump()

        if self.agent.prompt:
            try:
                return self.agent.prompt.format(**validated)
            except (KeyError, IndexError) as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.INPUT_VALIDATION_ERROR,
                    "Invalid prompt template",
                    f"Failed to render prompt template from input: {e}",
                    UiPathErrorCategory.USER,
                ) from e

        if isinstance(validated.get("input"), str):
            return validated["input"]
        return json.dumps(validated)

    def _map_result_output(self, message: ResultMessage) -> dict[str, Any]:
        if self.agent.output_schema is not None:
            structured = message.structured_output
            try:
                validated = self.agent.output_schema.model_validate(structured or {})
            except ValidationError as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.OUTPUT_VALIDATION_ERROR,
                    "Invalid structured output",
                    f"Agent output does not match the declared output schema: {e}",
                    UiPathErrorCategory.USER,
                ) from e
            return json.loads(serialize_json(validated))
        # SDK-native structured output: options.output_format set by the user.
        if self.agent.options.output_format is not None and isinstance(
            message.structured_output, dict
        ):
            return message.structured_output
        logger.warning(
            "Entrypoint '%s' declares no output_schema, returning the agent's final "
            "text under 'result'. Set output_schema on the agent for typed output.",
            self.entrypoint or self.agent.name,
        )
        return {"result": message.result or ""}

    def create_runtime_error(self, e: Exception) -> Exception:
        """Map a raw exception before it is re-raised from stream()."""
        if isinstance(e, UiPathClaudeSDKRuntimeError):
            return e
        if isinstance(e, GatewayShimError):
            return UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.GATEWAY_PROXY_ERROR,
                "LLM gateway routing failed",
                f"Error: {e}",
                UiPathErrorCategory.USER,
            )
        if isinstance(e, TimeoutError):
            return UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.AGENT_TIMEOUT,
                "Agent execution timed out",
                f"Error: {e}",
                UiPathErrorCategory.USER,
            )
        return UiPathClaudeSDKRuntimeError(
            UiPathClaudeSDKErrorCode.AGENT_EXECUTION_ERROR,
            "Agent execution failed",
            f"Error: {e}",
            UiPathErrorCategory.USER,
        )

    # --- SDK options ------------------------------------------------------

    @staticmethod
    def _warn_on_overridden_safety_env(
        declared: dict[str, str] | None, safety_env: dict[str, str]
    ) -> None:
        """Say so when a declared value is about to lose to a safety value.

        Silently winning would leave a developer reading their own env back and
        seeing something else run.
        """
        for name, required in safety_env.items():
            declared_value = (declared or {}).get(name)
            if declared_value is not None and declared_value != required:
                logger.warning(
                    "%s=%s was declared but a suspending agent needs %s, so the "
                    "declared value is ignored.",
                    name,
                    declared_value,
                    required,
                )

    def _build_sdk_options(
        self,
        *,
        env: dict[str, str],
        workspace: Path,
        channel: SuspendChannel,
        resume_session_id: str | None = None,
    ) -> ClaudeAgentOptions:
        """Derive execution options from the user's options.

        Injects execution-scoped fields (env, cwd, model, output_format,
        resume) while preserving everything the developer configured.
        ``setting_sources`` defaults to [] to isolate from user/project/local
        Claude config. The suspend hook is added only when the agent registered
        UiPath tools, so an agent that declares none gets the tools, hooks and
        prompt it wrote and nothing else.
        """
        base = self.agent.options
        tool_index = uipath_tool_index(base.mcp_servers)
        safety_env = _SUSPEND_SAFETY_ENV if tool_index else {}
        self._warn_on_overridden_safety_env(base.env, safety_env)
        overrides: dict[str, Any] = {
            "env": {**base.env, **env, **safety_env},
            "cwd": workspace,
        }
        if self._shim is not None:
            overrides["model"] = self._shim.resolved_model
        if base.stderr is None:
            overrides["stderr"] = log_cli_stderr
        if base.setting_sources is None:
            overrides["setting_sources"] = []
        if self.agent.output_schema is not None and base.output_format is None:
            overrides["output_format"] = {
                "type": "json_schema",
                "schema": self.agent.output_schema.model_json_schema(),
            }
        if resume_session_id is not None:
            overrides["resume"] = resume_session_id
        self._wire_suspend_hook(base, overrides, channel, tool_index)
        return replace(base, **overrides)

    @staticmethod
    def _wire_suspend_hook(
        base: ClaudeAgentOptions,
        overrides: dict[str, Any],
        channel: SuspendChannel,
        tool_index: ToolIndex,
    ) -> None:
        """Add the ``PreToolUse`` hook that runs the agent's UiPath tool bodies.

        Every process that runs or resumes a session rebuilds it: hooks are live
        objects and cannot be persisted. Nothing else is injected, so an agent
        with no UiPath tools is handed the options it declared.
        """
        if not tool_index:
            return

        hooks: dict[str, list[HookMatcher]] = {
            event: list(matchers) for event, matchers in (base.hooks or {}).items()
        }
        for event, matchers in build_suspend_hooks(channel, tool_index).items():
            hooks[event] = [*matchers, *hooks.get(event, [])]
        overrides["hooks"] = hooks

    # --- Suspension lifecycle ----------------------------------------------

    async def _session_id_for_run(self, resuming: bool) -> str | None:
        """The Claude session this run re-attaches to.

        A resume without a stored session id would silently start a fresh
        conversation that has forgotten the parked call, so it fails instead.
        """
        if not resuming:
            return None
        session_id = await self._session_store.get_session_id()
        if session_id is None:
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.SESSION_NOT_FOUND,
                "No Claude session to resume",
                f"Runtime '{self.runtime_id}' was asked to resume but no Claude "
                "session id was persisted. The runtime directory holding the "
                "session state was not remounted, or the run never started.",
                UiPathErrorCategory.SYSTEM,
            )
        return session_id

    @traced(
        name="claude_suspend_restore",
        run_type="uipath",
        input_processor=_resume_shape_only,
        hide_output=True,
    )
    async def _restore_pending(
        self,
        channel: SuspendChannel,
        input: dict[str, Any] | None,
        resuming: bool,
    ) -> PendingSuspend | None:
        """Seed the resume payload for the parked call, when this run resolves one.

        The resumable wrapper hands back either the ``{interrupt_id: payload}``
        map it built from a fired trigger, or, when the caller passed input
        explicitly, that input unwrapped. Both shapes have to resolve the parked
        call, otherwise ``uipath run agent '{"answer": "..."}' --resume`` drops
        the answer and suspends again under a fresh interrupt id.

        A resume that carries nothing yet restores the record without resolving
        it, so the re-parked call keeps its interrupt id instead of minting a
        second trigger for the same wait.
        """
        if not resuming:
            return None

        pending = await self._session_store.get_pending_suspend()
        if pending is None:
            return None
        if not input:
            logger.info(
                "Interrupt %s is still pending and this resume carries no payload, "
                "so the parked call stays parked.",
                pending.interrupt_id,
            )
            channel.restore(pending)
            return None

        payload = (
            input[pending.interrupt_id] if pending.interrupt_id in input else input
        )
        await self._check_entrypoint()
        channel.restore(pending)
        channel.resolve(pending.interrupt_id, payload)
        return pending

    async def _check_entrypoint(self) -> None:
        """Refuse to resume a session that was started on another entrypoint."""
        record = await self._session_store.get_entrypoint()
        if record is None or self.entrypoint is None:
            return
        if record.entrypoint == self.entrypoint:
            return
        raise UiPathClaudeSDKRuntimeError(
            UiPathClaudeSDKErrorCode.ENTRYPOINT_MISMATCH,
            "Resumed on a different entrypoint",
            f"The suspended run started on '{record.entrypoint}' and cannot be "
            f"resumed on '{self.entrypoint}'.",
            UiPathErrorCategory.USER,
        )

    @staticmethod
    def _refuse_breakpoints(options: UiPathStreamOptions | None) -> None:
        """Answer a breakpoint request out loud instead of ignoring it.

        The agent loop runs inside the claude CLI subprocess, so this runtime
        owns no node it could hold execution at, and the graph it reports has
        no node to name either. ``uipath debug`` still offers stepping and
        node breakpoints, and a debugger that accepts them and then runs
        straight through looks broken rather than unsupported.
        """
        if options is None or not options.breakpoints:
            return
        logger.warning(
            "Breakpoints are not supported by the Claude SDK runtime, so this run "
            "continues to completion. The agent loop runs inside the claude CLI "
            "subprocess and exposes no node to hold at. A tool that calls "
            "interrupt is the supported way to stop a run for a human."
        )

    @traced(name="claude_transcript_capture", run_type="uipath")
    async def _capture_transcript(self, session_id: str | None) -> None:
        """Store the CLI's session transcript so a resumed process can restore it.

        The platform carries the state database across a suspension and nothing
        else, so without this the CLI finds no session on the far side and
        silently starts a fresh one.
        """
        if not session_id:
            return
        try:
            source = self._session_paths.find_transcript(session_id)
        except ValueError as e:
            logger.warning("Not storing a transcript: %s", e)
            return
        if source is None:
            logger.warning(
                "Session %s has no transcript on disk, so a resumed process will "
                "not be able to continue it.",
                session_id,
            )
            return
        await self._session_store.set_transcript(
            TranscriptRecord(
                session_id=session_id,
                project_dir=source.parent.name,
                content=source.read_text(encoding="utf-8"),
            )
        )
        logger.debug(
            "Stored the transcript for session %s (%d bytes).",
            session_id,
            source.stat().st_size,
        )

    @traced(name="claude_transcript_restore", run_type="uipath")
    async def _restore_transcript(self, session_id: str | None) -> None:
        """Put a stored transcript back before the CLI looks for its session.

        Written under the name this run's working directory encodes to, not the
        one the previous process recorded, so the working directory is free to
        differ between the two.
        """
        if not session_id:
            return
        record = await self._session_store.get_transcript()
        if record is None or record.session_id != session_id:
            return
        target = self._session_paths.transcript_path(session_id)
        if target.exists():
            return
        if record.project_dir != target.parent.name:
            logger.warning(
                "Session %s was filed under %s and is being restored under %s. "
                "That is expected when the working directory moved, and a sign "
                "the CLI changed its encoding when it did not.",
                session_id,
                record.project_dir,
                target.parent.name,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record.content, encoding="utf-8")
        logger.info(
            "Restored the transcript for session %s under %s.",
            session_id,
            target.parent.name,
        )

    async def _open_run(
        self,
        input: dict[str, Any] | None,
        options: UiPathStreamOptions | None,
    ) -> RunContext:
        """Build the per-invocation suspend channel and SDK options."""
        self._refuse_breakpoints(options)
        channel = SuspendChannel()
        resuming = bool(options and options.resume)
        session_id = await self._session_id_for_run(resuming)
        pending = await self._restore_pending(channel, input, resuming)
        env = await self._build_llm_env()
        self._session_paths.ensure()
        await self._restore_transcript(session_id)
        return RunContext(
            channel=channel,
            pending=pending,
            resuming=resuming,
            session_id=session_id,
            sdk_options=self._build_sdk_options(
                env={**env, **self._session_paths.env_overrides()},
                workspace=self._session_paths.workspace,
                channel=channel,
                resume_session_id=session_id,
            ),
        )

    def _user_message_for_run(
        self, input: dict[str, Any] | None, context: RunContext
    ) -> str:
        """The prompt that starts the turn.

        Resolving a parked call needs no new instruction: the payload reaches
        the model as that call's tool result. A resume that carries no payload
        yet leaves the call parked and re-suspends, so resume input must never
        be validated against the entrypoint input schema.
        """
        if context.pending is not None or context.resuming:
            return RESUME_PROMPT
        return self._get_user_message(input or {})

    async def _on_result_message(
        self, message: ResultMessage, context: RunContext
    ) -> None:
        """Record the session, verify continuity and surface a failed turn."""
        self._check_session_continuity(message, context)
        await self._session_store.set_session_id(message.session_id)
        if message.deferred_tool_use is not None:
            return
        if message.is_error:
            self._raise_result_error(message)

    @staticmethod
    def _check_session_continuity(message: ResultMessage, context: RunContext) -> None:
        """Detect a session the CLI could not find and started fresh instead.

        A lost transcript is never reported by the CLI, so a resume that was
        meant to deliver a parked call would otherwise look like a run that
        succeeded and forgot everything.
        """
        if context.session_id is None or context.sdk_options.fork_session:
            return
        if message.session_id == context.session_id:
            return
        detail = (
            f"Asked to resume Claude session '{context.session_id}' but the run "
            f"answered from '{message.session_id}'. The session transcript is "
            "missing from the runtime directory."
        )
        if context.pending is None:
            logger.warning("%s Earlier conversation history is lost.", detail)
            return
        raise UiPathClaudeSDKRuntimeError(
            UiPathClaudeSDKErrorCode.SESSION_MISMATCH,
            "Claude session was not resumed",
            f"{detail} Interrupt {context.pending.interrupt_id} cannot be "
            "delivered to the agent.",
            UiPathErrorCategory.SYSTEM,
        )

    async def _suspended_result(self, context: RunContext) -> UiPathRuntimeResult:
        """Persist the parked call and report it as an interrupt."""
        pending = context.channel.pending
        if pending is None:
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.INTERRUPT_STATE_ERROR,
                "Deferred call was not claimed",
                "The CLI parked a tool call that the UiPath suspend hook never "
                "claimed, so there is nothing to suspend on.",
                UiPathErrorCategory.SYSTEM,
            )
        await self._session_store.set_pending_suspend(pending)
        await self._session_store.set_entrypoint(
            EntrypointRecord(entrypoint=self.entrypoint or self.agent.name)
        )
        await self._capture_transcript(await self._session_store.get_session_id())
        logger.info(
            "Suspended on interrupt %s from %s.",
            pending.interrupt_id,
            pending.tool_name,
        )
        return UiPathRuntimeResult(
            output={pending.interrupt_id: pending.value},
            status=UiPathRuntimeStatus.SUSPENDED,
        )

    def _refuse_ignored_deferral(self, context: RunContext) -> None:
        """Fail a run whose suspension was silently dropped by the CLI.

        The hook asked for a deferral and the turn came back without one, so
        the parked call ran for real. Reporting success here would mean a job
        that skipped its human step and told nobody. The known trigger is the
        non-streaming fallback, which ``_SUSPEND_SAFETY_ENV`` disables, so
        reaching this is either that flag being overridden or a change in CLI
        behaviour.
        """
        if context.channel.deferrals_requested == 0:
            return
        raise UiPathClaudeSDKRuntimeError(
            UiPathClaudeSDKErrorCode.SUSPEND_IGNORED,
            "Suspension was ignored",
            "The agent asked to suspend but the run finished without "
            "suspending, so the tool executed instead of parking.",
            UiPathErrorCategory.SYSTEM,
        )

    async def _clear_resolved_pending(self, context: RunContext) -> None:
        """Drop the stored record once the parked call has really run."""
        if context.pending is None:
            return
        if context.channel.pending is not None:
            logger.warning(
                "Interrupt %s was resumed but its tool never ran, so the pending "
                "record is kept for another resume.",
                context.pending.interrupt_id,
            )
            return
        await self._session_store.clear_pending_suspend()

    # --- Message mapping ---------------------------------------------------

    @staticmethod
    def _map_assistant(message: AssistantMessage) -> list[UiPathRuntimeStateEvent]:
        events: list[UiPathRuntimeStateEvent] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="assistant",
                        payload={"text": block.text},
                    )
                )
            elif isinstance(block, ToolUseBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="tool_call",
                        payload={"tool": block.name, "input": block.input},
                    )
                )
            elif isinstance(block, ThinkingBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="thinking",
                        payload={"thinking": block.thinking},
                    )
                )
            else:
                logger.warning(
                    "Unhandled assistant block type: %s", type(block).__name__
                )
        return events

    @staticmethod
    def _map_tool_results(message: UserMessage) -> list[UiPathRuntimeStateEvent]:
        if isinstance(message.content, list):
            return [
                UiPathRuntimeStateEvent(
                    node_name="tool_result",
                    payload={
                        "tool_use_id": block.tool_use_id,
                        "content": str(block.content),
                    },
                )
                for block in message.content
                if isinstance(block, ToolResultBlock)
            ]
        return [
            UiPathRuntimeStateEvent(
                node_name="tool_result",
                payload={"content": str(message.content)},
            )
        ]

    @staticmethod
    def _map_system(message: SystemMessage) -> UiPathRuntimeStateEvent | None:
        if isinstance(message, (TaskStartedMessage, TaskProgressMessage)):
            return UiPathRuntimeStateEvent(
                node_name="task",
                payload={
                    "task_id": message.task_id,
                    "description": message.description,
                },
            )
        if isinstance(message, TaskNotificationMessage):
            return UiPathRuntimeStateEvent(
                node_name="task",
                payload={
                    "task_id": message.task_id,
                    "status": message.status,
                    "summary": message.summary,
                },
            )
        if isinstance(message, TaskUpdatedMessage):
            return UiPathRuntimeStateEvent(
                node_name="task",
                payload={
                    "task_id": message.task_id,
                    "status": message.status,
                    "patch": message.patch,
                },
            )
        return None

    @staticmethod
    def _update_background_tasks(message: Message, active_tasks: set[str]) -> None:
        """Update the bounded background-task ledger from one SDK message."""
        if isinstance(message, TaskStartedMessage):
            if message.task_type in _WAITED_BACKGROUND_TASK_TYPES:
                active_tasks.add(message.task_id)
        elif isinstance(message, TaskNotificationMessage):
            active_tasks.discard(message.task_id)
        elif isinstance(message, TaskUpdatedMessage):
            if message.status in TERMINAL_TASK_STATUSES:
                active_tasks.discard(message.task_id)

    async def _receive_run_messages(
        self, client: ClaudeSDKClient
    ) -> AsyncGenerator[Message, None]:
        """Receive through the final result after bounded background work.

        ``ClaudeSDKClient.receive_response()`` stops at the first
        ``ResultMessage``. For a background subagent that result ends only the
        spawning turn: the task later wakes the parent and produces another
        result. It is called once per turn, rather than reading the underlying
        stream directly, because it is the method the tracing instrumentor
        wraps: a turn read any other way produces no agent span and no tool
        spans. Both methods draw on the same stream, so calling it again picks
        up where the last turn stopped.

        A deferred tool use is a real run boundary and must be returned to the
        UiPath resumable runtime immediately. It is therefore not held open by
        this local-task ledger.
        """
        active_tasks: set[str] = set()
        while True:
            result: ResultMessage | None = None
            async for message in client.receive_response():
                self._update_background_tasks(message, active_tasks)
                yield message
                if isinstance(message, ResultMessage):
                    result = message

            if result is None:
                break
            if result.deferred_tool_use is not None or not active_tasks:
                return
            logger.info(
                "Claude turn ended with %d background task(s) still running; "
                "waiting for their follow-up turn.",
                len(active_tasks),
            )

        if active_tasks:
            task_ids = ", ".join(sorted(active_tasks))
            raise RuntimeError(
                "Claude's message stream ended while background tasks were still "
                f"running: {task_ids}"
            )

    @staticmethod
    def _raise_result_error(message: ResultMessage) -> None:
        parts = [message.result or "Agent run failed"]
        if message.subtype:
            parts.append(f"subtype={message.subtype}")
        if message.stop_reason:
            parts.append(f"stop_reason={message.stop_reason}")
        if message.errors:
            parts.append(f"errors={message.errors}")
        if message.permission_denials:
            parts.append(f"permission_denials={message.permission_denials}")
        raise RuntimeError(" | ".join(parts))

    # --- Runtime protocol ---------------------------------------------------

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events.

        Yields:
            UiPathRuntimeStateEvent for each intermediate step, then
            UiPathRuntimeResult as the final event: SUCCESSFUL with the mapped
            output, or SUSPENDED with ``{interrupt_id: suspend_value}`` when the
            model parked a UiPath interrupt call.
        """
        try:
            context = await self._open_run(input, options)
            user_message = self._user_message_for_run(input, context)
            logger.info("Claude SDK agent workspace: %s", self._session_paths.workspace)

            result_message: ResultMessage | None = None
            try:
                with active_channel(context.channel):
                    async with ClaudeSDKClient(options=context.sdk_options) as client:
                        await client.query(user_message)
                        async for message in self._receive_run_messages(client):
                            self._adopt_span_parent(context.channel)
                            if isinstance(message, AssistantMessage):
                                for event in self._map_assistant(message):
                                    yield event
                            elif isinstance(message, UserMessage):
                                for event in self._map_tool_results(message):
                                    yield event
                            elif isinstance(message, ResultMessage):
                                await self._on_result_message(message, context)
                                result_message = message
                            elif isinstance(message, SystemMessage):
                                mapped = self._map_system(message)
                                if mapped is not None:
                                    yield mapped
            finally:
                await self._stop_llm_gateway()
                self._flush_gateway_spans()

            if result_message is not None and result_message.deferred_tool_use:
                yield await self._suspended_result(context)
                return

            self._refuse_ignored_deferral(context)
            await self._clear_resolved_pending(context)
            output = self._map_result_output(result_message) if result_message else {}
            yield UiPathRuntimeResult(
                output=output, status=UiPathRuntimeStatus.SUCCESSFUL
            )
        except Exception as e:
            raise self.create_runtime_error(e) from e

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent and return the final result."""
        stream_options = (
            UiPathStreamOptions(**options.model_dump()) if options else None
        )
        result: UiPathRuntimeResult | None = None
        async for event in self.stream(input, stream_options):
            if isinstance(event, UiPathRuntimeResult):
                result = event
        return result or UiPathRuntimeResult(output={})

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this Claude SDK agent runtime."""
        entrypoints_schema = get_entrypoints_schema(self.agent)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint or "",
            uniqueId=str(uuid4()),
            type="agent",
            input=entrypoints_schema.get("input", {}),
            output=entrypoints_schema.get("output", {}),
            graph=get_agent_graph(self.agent),
        )

    async def dispose(self) -> None:
        """Cleanup runtime resources."""
        await self._stop_llm_gateway()
