"""Opt-in interrupt surface for a Claude Agent SDK agent running on UiPath.

Nothing here is installed unless the developer asks for it. An agent whose
``mcp_servers`` holds no server built by :func:`uipath_tool_server` runs with
exactly the tools, hooks and prompt it declared, and an agent that runs
elsewhere runs unchanged on UiPath.

The developer writes an ordinary SDK tool and calls :func:`interrupt` inside it
with a platform model imported from ``uipath.platform.common``::

    from uipath.platform.common import CreateTask
    from uipath_claude_sdk import interrupt, uipath_tool_server

    @tool("review", "Get a human to review it.", {"ticket_id": str, "label": str})
    async def review(args: dict[str, Any]) -> dict[str, Any]:
        action_data = await interrupt(
            CreateTask(
                app_name="escalation_agent_app",
                title="Action Required: Review classification",
                data={"AgentOutput": f"Classified {args['ticket_id']}"},
                app_folder_path="Shared",
            )
        )
        approved = action_data["Answer"] is True
        return {"content": [{"type": "text", "text": f"Approved: {approved}"}]}

    server = uipath_tool_server("tickets", tools=[review])
    options = ClaudeAgentOptions(mcp_servers={"tickets": server})

The value handed to :func:`interrupt` reaches the platform untouched.
``UiPathResumeTriggerHandler`` decides the trigger kind from the value's type,
so every platform interrupt model works without any code here knowing about it,
and a value it does not recognise degrades to an API trigger. A list becomes
sibling triggers for one interrupt, resolved by whichever fires first.

How the body runs
-----------------

A tool handler cannot pause and resume across a process boundary, and the CLI's
``defer`` decision fires before the handler runs. The tool body is therefore
executed by the ``PreToolUse`` hook, and the handler returns what the hook
already computed for that ``tool_use_id``:

* Suspend pass. :func:`interrupt` raises a private carrier, the hook records the
  pending suspension against the CLI's ``tool_use_id`` and defers the call.
* Resume pass. The CLI re-issues the same ``tool_use_id``. :func:`interrupt`
  returns the resolved payload, the body runs to completion, and the handler
  delivers its return value instead of executing a second time.
* No interrupt at all. The body completes on the first pass and the handler
  delivers its return value. The body still runs exactly once.

Across a suspension the body therefore runs twice, exactly as a LangGraph node
does. Work performed BEFORE the :func:`interrupt` call happens twice, so keep
side effects after the call or make them idempotent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from claude_agent_sdk import (
    McpSdkServerConfig,
    SdkMcpTool,
    create_sdk_mcp_server,
)
from opentelemetry import context as context_api
from opentelemetry.context import Context
from pydantic import BaseModel
from uipath.core.tracing import traced
from uipath.platform.common import interrupt_models

__all__ = [
    "INTERRUPT_SUSPEND_MODELS",
    "UIPATH_MCP_SERVER_NAME",
    "InterruptToolBinding",
    "InterruptOutsideRunError",
    "PendingSuspend",
    "SuspendAlreadyPendingError",
    "SuspendChannel",
    "SuspendChannelError",
    "ToolIndex",
    "ToolOutcome",
    "UnknownInterruptError",
    "active_channel",
    "call_key",
    "interrupt",
    "run_tool_body",
    "uipath_tool_index",
    "uipath_tool_server",
]

logger = logging.getLogger(__name__)

UIPATH_MCP_SERVER_NAME = "uipath"

ToolBody = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_NOT_DEFERRED_MESSAGE = (
    "No resumed input is available for this call, so the run was never "
    "suspended. Do not retry the same call. Continue with the information you "
    "already have, or tell the user that waiting for input is unavailable."
)

_OUTSIDE_RUN_MESSAGE = (
    "interrupt() was called outside a UiPath run, so there is no channel to "
    "suspend on. A tool can only suspend when its server was built with "
    "uipath_tool_server() and registered in ClaudeAgentOptions.mcp_servers, "
    "and when the agent is executed by the UiPath Claude SDK runtime."
)


class SuspendChannelError(RuntimeError):
    """Base error for misuse of a :class:`SuspendChannel`."""


class SuspendAlreadyPendingError(SuspendChannelError):
    """Raised when a second suspend is claimed while one is already in flight."""


class UnknownInterruptError(SuspendChannelError):
    """Raised when an interrupt id does not match the channel's pending record."""


class InterruptOutsideRunError(SuspendChannelError):
    """Raised when :func:`interrupt` is called with no UiPath run around it."""


INTERRUPT_SUSPEND_MODELS: dict[str, type[BaseModel]] = {
    name: value
    for name, value in vars(interrupt_models).items()
    if inspect.isclass(value)
    and issubclass(value, BaseModel)
    and value is not BaseModel
}
"""Every platform interrupt model, keyed by its class name.

A suspend value that crosses a suspension is serialized, and the platform's
trigger creator dispatches on the value's type, so a value reloaded as a plain
dict silently becomes an API trigger. Persisting the class name and looking it
up here rebuilds the type the creator needs.
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifying_inputs_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """Name the suspension on its span without recording what it carries.

    A suspend value is whatever the tool passed to :func:`interrupt`: an
    approval's payload, a question, a record to write. It reaches LLM Ops as a
    span attribute if it is recorded, so only the identifiers go there.
    """
    return {
        name: inputs.get(name)
        for name in ("tool_name", "tool_use_id")
        if name in inputs
    }


class _InterruptRaised(BaseException):
    """Carries a suspend value out of a tool body on the suspend pass.

    It derives from ``BaseException`` so a developer's ``except Exception``
    around their own work cannot swallow a suspension.
    """

    def __init__(self, value: Any) -> None:
        super().__init__("A UiPath interrupt escaped its tool body.")
        self.value = value


@dataclass(frozen=True)
class _CallContext:
    """What :func:`interrupt` needs to decide between raising and returning."""

    channel: SuspendChannel
    tool_use_id: str | None


@dataclass
class ToolOutcome:
    """What a tool body produced, held until the handler delivers it."""

    result: dict[str, Any] | None = None
    error: BaseException | None = None
    tool_use_id: str | None = None

    def deliver(self) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return self.result or {}


@dataclass(frozen=True)
class InterruptToolBinding:
    """One developer tool the suspend hook is allowed to execute.

    Attributes:
        token: Opaque per-tool identifier, so a stashed result cannot be picked
            up by a same-named tool from another server.
        body: The handler the developer wrote, before wrapping.
    """

    token: str
    body: ToolBody


ToolIndex = dict[str, InterruptToolBinding]


@dataclass
class PendingSuspend:
    """One suspension awaiting a resume payload.

    Attributes:
        interrupt_id: Identifier the runtime reports in the SUSPENDED output map
            and that the resumed process seeds a payload against. It is the
            CLI's ``tool_use_id``, which is stable across defer and re-issue.
        tool_name: Model-visible tool name, e.g. ``mcp__tickets__review``.
        value: What lands in the SUSPENDED output map for this interrupt. The
            platform's trigger creator decides the trigger kind from its type,
            and a list yields sibling triggers resolved by the first to fire.
        tool_use_id: The CLI's tool_use id for the deferred call, equal to
            ``interrupt_id`` and kept as its own field for the stored record.
        requested_at: ISO 8601 UTC timestamp of the claim.
    """

    interrupt_id: str
    tool_name: str
    value: Any
    tool_use_id: str | None = None
    requested_at: str = field(default_factory=_utc_now_iso)


class SuspendChannel:
    """Per-run rendezvous between the PreToolUse hook, the tools and the runtime.

    The hook runs a developer tool body and leaves behind either a pending
    suspension or the body's outcome. The runtime reads :attr:`pending` to build
    its SUSPENDED result. On resume a fresh channel is restored from the
    persisted record and seeded with the trigger payload, which the re-run body
    receives from :func:`interrupt`.
    """

    def __init__(self) -> None:
        self.pending: PendingSuspend | None = None
        self.deferrals_requested = 0
        self.turn_parent: Context | None = None
        self._resolved: dict[str, Any] = {}
        self._outcomes: dict[str, deque[ToolOutcome]] = {}
        self._lock = asyncio.Lock()

    def adopt_turn_parent(self, parent: Context) -> None:
        """Record the trace context a tool body should run under.

        The runtime calls this while the turn's agent span is current. Only
        valid from there, and it moves on with the turn.
        """
        self.turn_parent = parent

    @traced(
        name="claude_suspend_claim",
        run_type="uipath",
        input_processor=_identifying_inputs_only,
    )
    async def claim(self, tool_name: str, value: Any, tool_use_id: str) -> str:
        """Register a new suspension and return its interrupt id.

        Args:
            tool_name: Model-visible name of the tool being deferred.
            value: The suspend value for the SUSPENDED output map.
            tool_use_id: The CLI's tool_use id for this call.

        Returns:
            The interrupt id, which is ``tool_use_id``.

        Raises:
            SuspendAlreadyPendingError: If a different suspension is in flight.
        """
        async with self._lock:
            if self.pending is not None and self.pending.interrupt_id != tool_use_id:
                raise SuspendAlreadyPendingError(
                    f"Interrupt {self.pending.interrupt_id} for "
                    f"{self.pending.tool_name} is already in flight."
                )
            self.deferrals_requested += 1
            self.pending = PendingSuspend(
                interrupt_id=tool_use_id,
                tool_name=tool_name,
                value=value,
                tool_use_id=tool_use_id,
            )
            return tool_use_id

    def restore(self, pending: PendingSuspend) -> None:
        """Reinstate a persisted pending record in a resumed process.

        Args:
            pending: The record persisted when the run suspended.

        Raises:
            SuspendAlreadyPendingError: If a suspension is already in flight.
        """
        if self.pending is not None:
            raise SuspendAlreadyPendingError(
                f"Interrupt {self.pending.interrupt_id} is already in flight."
            )
        self.pending = pending

    def resolve(self, interrupt_id: str, payload: Any) -> None:
        """Seed the trigger payload for a restored suspension.

        Args:
            interrupt_id: Identifier of the pending record.
            payload: What the re-run body receives from :func:`interrupt`.

        Raises:
            UnknownInterruptError: If no pending record carries that id.
        """
        if self.pending is None or self.pending.interrupt_id != interrupt_id:
            raise UnknownInterruptError(
                f"No pending suspension with interrupt id {interrupt_id}."
            )
        self._resolved[interrupt_id] = payload

    def resolved_for(self, interrupt_id: str | None) -> Any | None:
        """Return the seeded payload, or ``None`` when there is none."""
        if interrupt_id is None:
            return None
        return self._resolved.get(interrupt_id)

    def is_resolved(self, interrupt_id: str | None) -> bool:
        """Whether a payload has been seeded for ``interrupt_id``."""
        return interrupt_id is not None and interrupt_id in self._resolved

    def complete(self, interrupt_id: str) -> None:
        """Drop the pending record and its payload once delivered to the model."""
        self._resolved.pop(interrupt_id, None)
        if self.pending is not None and self.pending.interrupt_id == interrupt_id:
            self.pending = None

    def stash(self, key: str, outcome: ToolOutcome) -> None:
        """Hold what a tool body produced until its handler is called.

        One key can hold several outcomes. A model is free to make the same call
        twice in one turn, and the key is built from the tool and its arguments
        because that is all a handler has to recognise itself by: the SDK hands
        a handler its arguments and no ``tool_use_id``. Holding a queue rather
        than a slot is what stops the second of two identical calls from finding
        nothing stashed and running the body again.
        """
        self._outcomes.setdefault(key, deque()).append(outcome)

    def take(self, key: str) -> ToolOutcome | None:
        """Remove and return the oldest stashed outcome, or ``None`` when empty."""
        queued = self._outcomes.get(key)
        if not queued:
            return None
        outcome = queued.popleft()
        if not queued:
            del self._outcomes[key]
        return outcome


_ACTIVE_CHANNEL: ContextVar[SuspendChannel | None] = ContextVar(
    "uipath_claude_sdk_active_channel", default=None
)

_ACTIVE_CALL: ContextVar[_CallContext | None] = ContextVar(
    "uipath_claude_sdk_active_call", default=None
)


@contextmanager
def _under_turn(channel: SuspendChannel) -> Iterator[None]:
    """Run a tool body inside the trace context of the turn that called it.

    The client's tasks inherit the context from before the run began, so a span
    a tool opens would otherwise attach to whatever was current back then and
    surface outside the turn rather than under it. The runtime records the turn
    on the channel, which is the same handoff the gateway spans use.
    """
    parent = channel.turn_parent
    if parent is None:
        yield
        return
    token = context_api.attach(parent)
    try:
        yield
    finally:
        context_api.detach(token)


@contextmanager
def active_channel(channel: SuspendChannel) -> Iterator[None]:
    """Make ``channel`` the one this run's UiPath tools resolve against.

    The runtime enters this before it constructs the SDK client, so every task
    the client spawns inherits the binding.
    """
    token = _ACTIVE_CHANNEL.set(channel)
    try:
        yield
    finally:
        _ACTIVE_CHANNEL.reset(token)


async def interrupt(value: Any) -> Any:
    """Suspend the run on ``value`` and return what resumes it.

    On the suspend pass this never returns: it raises a private carrier the
    ``PreToolUse`` hook turns into a deferred call, and the run reports
    SUSPENDED with ``value`` as its interrupt payload. On the resume pass the
    same tool body runs again and this returns the resolved payload instead.

    Args:
        value: What the platform turns into a resume trigger, normally a model
            from ``uipath.platform.common``. A list becomes sibling triggers
            resolved by whichever fires first.

    Returns:
        The payload the fired trigger carried back.

    Raises:
        InterruptOutsideRunError: If no UiPath run is wrapping this call.
    """
    call = _ACTIVE_CALL.get()
    if call is None:
        raise InterruptOutsideRunError(_OUTSIDE_RUN_MESSAGE)
    if call.channel.is_resolved(call.tool_use_id):
        return call.channel.resolved_for(call.tool_use_id)
    raise _InterruptRaised(value)


def _arguments_key(args: Mapping[str, Any]) -> str:
    """A stable key for one call's arguments.

    The hook and the handler see the same arguments and nothing else they share,
    so the arguments are what correlates the stashed outcome with the call that
    produced it.
    """
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(sorted(args.items(), key=lambda item: item[0]))


def call_key(binding: InterruptToolBinding, args: Mapping[str, Any]) -> str:
    """The stash key for one call of one tool."""
    return f"{binding.token}:{_arguments_key(args)}"


async def run_tool_body(
    channel: SuspendChannel,
    binding: InterruptToolBinding,
    args: dict[str, Any],
    tool_name: str,
    tool_use_id: str,
) -> ToolOutcome | None:
    """Run one developer tool body on behalf of the ``PreToolUse`` hook.

    Args:
        channel: The run's suspend channel.
        binding: The tool whose body to run.
        args: Arguments the model passed.
        tool_name: Model-visible tool name, recorded on the pending suspension.
        tool_use_id: The CLI's tool_use id, which becomes the interrupt id.

    Returns:
        The body's outcome, or ``None`` when it suspended and the call must be
        deferred.

    Raises:
        SuspendAlreadyPendingError: If another interrupt is already in flight.
    """
    token = _ACTIVE_CALL.set(_CallContext(channel=channel, tool_use_id=tool_use_id))
    try:
        with _under_turn(channel):
            result = await binding.body(args)
    except _InterruptRaised as raised:
        await channel.claim(tool_name, raised.value, tool_use_id)
        return None
    except Exception as error:
        return ToolOutcome(error=error, tool_use_id=tool_use_id)
    else:
        return ToolOutcome(result=result, tool_use_id=tool_use_id)
    finally:
        _ACTIVE_CALL.reset(token)


def _settle_delivered(channel: SuspendChannel, tool_use_id: str | None) -> None:
    """Clear the pending record once its resolved value reaches the model."""
    pending = channel.pending
    if pending is None or pending.interrupt_id != tool_use_id:
        return
    if channel.is_resolved(pending.interrupt_id):
        channel.complete(pending.interrupt_id)


async def _run_unstashed(
    channel: SuspendChannel, body: ToolBody, args: dict[str, Any]
) -> dict[str, Any]:
    """Run a body the hook did not run, and refuse to suspend from there.

    The handler is past the point where a call can be parked, so a body that
    tries to interrupt here is told so as a tool error the model can act on
    rather than faulting the job. The known cause is a defer the CLI dropped,
    which the runtime also fails the run over.
    """
    token = _ACTIVE_CALL.set(_CallContext(channel=channel, tool_use_id=None))
    try:
        with _under_turn(channel):
            return await body(args)
    except _InterruptRaised:
        logger.warning(
            "A UiPath tool tried to suspend from its handler, so its call was "
            "never parked."
        )
        return {
            "content": [{"type": "text", "text": _NOT_DEFERRED_MESSAGE}],
            "is_error": True,
        }
    finally:
        _ACTIVE_CALL.reset(token)


def _wrap(tool_def: SdkMcpTool[Any]) -> tuple[SdkMcpTool[Any], InterruptToolBinding]:
    """Replace a tool's handler with one that delivers the hook's outcome."""
    binding = InterruptToolBinding(token=uuid4().hex, body=tool_def.handler)

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        channel = _ACTIVE_CHANNEL.get()
        if channel is None:
            return await binding.body(args)
        outcome = channel.take(call_key(binding, args))
        if outcome is None:
            return await _run_unstashed(channel, binding.body, args)
        _settle_delivered(channel, outcome.tool_use_id)
        return outcome.deliver()

    wrapped = SdkMcpTool(
        name=tool_def.name,
        description=tool_def.description,
        input_schema=tool_def.input_schema,
        handler=handler,
        annotations=tool_def.annotations,
    )
    return wrapped, binding


_REGISTRY: WeakKeyDictionary[Any, dict[str, InterruptToolBinding]] = WeakKeyDictionary()
"""Server instance to its tools, keyed by bare name, for what this module built.

Keying on the instance lets :func:`uipath_tool_server` return a plain
``McpSdkServerConfig`` while the runtime can still tell which servers are its
own, and lets the qualified names be derived from the ``mcp_servers`` key the
developer actually used.
"""


def uipath_tool_server(
    name: str,
    tools: list[SdkMcpTool[Any]],
    version: str = "1.0.0",
) -> McpSdkServerConfig:
    """Build an in-process MCP server whose tools may call :func:`interrupt`.

    The returned config is an ordinary ``McpSdkServerConfig``: register it under
    whatever key you like in ``ClaudeAgentOptions.mcp_servers``. Its tools keep
    the names, descriptions and schemas you gave them, and outside a UiPath run
    they behave like any other SDK tool.

    Args:
        name: MCP server name.
        tools: Tools built with the SDK's ``@tool`` decorator.
        version: Server version string.

    Returns:
        A config to place in ``ClaudeAgentOptions.mcp_servers``.
    """
    pairs = [_wrap(tool_def) for tool_def in tools]
    config = create_sdk_mcp_server(
        name=name, version=version, tools=[wrapped for wrapped, _ in pairs]
    )
    _REGISTRY[config["instance"]] = {
        wrapped.name: binding for wrapped, binding in pairs
    }
    return config


def uipath_tool_index(mcp_servers: Any) -> ToolIndex:
    """Map every UiPath tool in a mapping to the body the hook must run.

    Args:
        mcp_servers: ``ClaudeAgentOptions.mcp_servers`` as the developer wrote
            it. Anything that is not a mapping yields an empty index, which is
            how an agent ends up with no UiPath wiring at all.

    Returns:
        Model-visible tool name to its binding.
    """
    if not isinstance(mcp_servers, Mapping):
        return {}
    index: ToolIndex = {}
    for server_name, config in mcp_servers.items():
        instance = config.get("instance") if isinstance(config, Mapping) else None
        if instance is None:
            continue
        for tool_name, binding in _REGISTRY.get(instance, {}).items():
            index[f"mcp__{server_name}__{tool_name}"] = binding
    return index
