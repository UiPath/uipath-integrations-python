"""PreToolUse hook that runs a UiPath tool's body and parks it when it suspends.

The Claude Code CLI parks a tool call without running it when a ``PreToolUse``
hook returns ``permissionDecision: "defer"``, writes a marker into the session
transcript, and ends the turn with ``ResultMessage.deferred_tool_use`` set. A
handler cannot pause across that boundary and the defer fires before the handler
runs, so the body is executed here instead and the handler returns what this
hook left behind.

Hooks cannot be persisted. Every process that runs or resumes a session rebuilds
them with :func:`build_suspend_hooks`, and only when the agent registered a
server built by :func:`~uipath_claude_sdk.interrupts.uipath_tool_server`.
"""

from __future__ import annotations

import logging
from typing import Literal

from claude_agent_sdk import HookContext, HookInput, HookJSONOutput, HookMatcher
from claude_agent_sdk.types import PreToolUseHookSpecificOutput

from ..interrupts import (
    SuspendAlreadyPendingError,
    SuspendChannel,
    ToolIndex,
    call_key,
    run_tool_body,
)

__all__ = ["TOOL_BODY_TIMEOUT_SECONDS", "build_suspend_hooks"]

logger = logging.getLogger(__name__)

TOOL_BODY_TIMEOUT_SECONDS = 600.0
"""Hook timeout in seconds, forwarded verbatim to the CLI.

The developer's whole tool body runs inside this hook, so the SDK's 60s default
would cap every UiPath tool at a minute of real work.
"""

_IN_FLIGHT_REASON = (
    "A UiPath interrupt is already in flight for this run. Only one can be "
    "pending at a time, and a tool that suspends must be the last tool call in "
    "its turn. Wait for the pending one to resolve, then re-issue this call on "
    "its own."
)

_UNCORRELATABLE_REASON = (
    "This tool call carries no tool_use id, so it cannot be correlated across a "
    "suspension. Re-issue the call on its own."
)


def _decision(
    decision: Literal["allow", "deny", "defer"], reason: str | None = None
) -> HookJSONOutput:
    specific: PreToolUseHookSpecificOutput = {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
    }
    if reason is not None:
        specific["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": specific}


def build_suspend_hooks(
    channel: SuspendChannel, tool_index: ToolIndex
) -> dict[str, list[HookMatcher]]:
    """Build the ``PreToolUse`` hooks that execute and gate the UiPath tools.

    Args:
        channel: The run's suspend channel, shared with the runtime.
        tool_index: Model-visible tool name to the developer's original handler,
            as :func:`~uipath_claude_sdk.interrupts.uipath_tool_index` built it.

    Returns:
        A mapping to merge into ``ClaudeAgentOptions.hooks``.
    """

    async def gate(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        if hook_input["hook_event_name"] != "PreToolUse":
            return {}

        tool_name = hook_input["tool_name"]
        binding = tool_index.get(tool_name)
        if binding is None:
            return {}

        pending = channel.pending
        if pending is not None and pending.interrupt_id != tool_use_id:
            return _decision("deny", _IN_FLIGHT_REASON)
        if tool_use_id is None:
            return _decision("deny", _UNCORRELATABLE_REASON)

        args = dict(hook_input["tool_input"])
        try:
            outcome = await run_tool_body(
                channel, binding, args, tool_name, tool_use_id
            )
        except SuspendAlreadyPendingError:
            return _decision("deny", _IN_FLIGHT_REASON)

        if outcome is None:
            return _decision("defer")
        if outcome.error is not None:
            logger.warning(
                "%s raised %s, which the model sees as a tool error.",
                tool_name,
                type(outcome.error).__name__,
            )
        channel.stash(call_key(binding, args), outcome)
        return _decision("allow")

    return {
        "PreToolUse": [
            HookMatcher(matcher=None, hooks=[gate], timeout=TOOL_BODY_TIMEOUT_SECONDS)
        ]
    }
