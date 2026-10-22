"""Error types for the UiPath Codex SDK runtime."""

from __future__ import annotations

from enum import Enum

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
    UiPathRuntimeError,
)


class UiPathCodexErrorCode(str, Enum):
    """Stable error codes surfaced by the Codex SDK runtime."""

    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_IMPORT_ERROR = "AGENT_IMPORT_ERROR"
    AGENT_LOAD_FAILURE = "AGENT_LOAD_FAILURE"
    AGENT_TYPE_ERROR = "AGENT_TYPE_ERROR"
    INPUT_INVALID = "INPUT_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    FILE_ERROR = "FILE_ERROR"
    IXP_FAILED = "IXP_FAILED"

    # durable (suspend/resume) tools
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    ENTRYPOINT_MISMATCH = "ENTRYPOINT_MISMATCH"
    ROLLOUT_CHECKPOINT_FAILED = "ROLLOUT_CHECKPOINT_FAILED"
    SUSPEND_KILL_UNAVAILABLE = "SUSPEND_KILL_UNAVAILABLE"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TERMINATION_ESCALATION_REJECTED = "TERMINATION_ESCALATION_REJECTED"


class UiPathCodexRuntimeError(UiPathRuntimeError):
    """Runtime error raised by the Codex SDK integration."""

    def __init__(
        self,
        code: UiPathCodexErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.USER,
    ) -> None:
        # UiPathRuntimeError types its code as the SDK's own enum; ours has the
        # same shape, so go straight to the base with the same "Python" prefix.
        UiPathBaseRuntimeError.__init__(
            self,
            code=code.value,
            title=title,
            detail=detail,
            category=category,
            prefix="Python",
        )


class UiPathRunTerminatedError(UiPathCodexRuntimeError):
    """A durable tool's result ends the run (an escalation outcome mapped to "end").

    Raised by ``finish``; the runtime closes the call's spans with
    ``result_text`` and faults the run with this error.
    """

    def __init__(self, title: str, detail: str, result_text: str) -> None:
        super().__init__(
            UiPathCodexErrorCode.TERMINATION_ESCALATION_REJECTED,
            title,
            detail,
            UiPathErrorCategory.USER,
        )
        self.result_text = result_text


__all__ = [
    "UiPathCodexErrorCode",
    "UiPathCodexRuntimeError",
    "UiPathRunTerminatedError",
]
