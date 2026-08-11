"""Error handling for the Claude SDK runtime."""

from enum import Enum

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
    UiPathErrorCode,
)


class UiPathClaudeSDKErrorCode(Enum):
    """Error codes specific to the Claude SDK runtime."""

    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    STREAM_ERROR = "STREAM_ERROR"

    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_TYPE_ERROR = "AGENT_TYPE_ERROR"
    AGENT_VALUE_ERROR = "AGENT_VALUE_ERROR"
    AGENT_LOAD_FAILURE = "AGENT_LOAD_FAILURE"
    AGENT_IMPORT_ERROR = "AGENT_IMPORT_ERROR"

    INPUT_VALIDATION_ERROR = "INPUT_VALIDATION_ERROR"
    OUTPUT_VALIDATION_ERROR = "OUTPUT_VALIDATION_ERROR"

    GATEWAY_PROXY_ERROR = "GATEWAY_PROXY_ERROR"

    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    ENTRYPOINT_MISMATCH = "ENTRYPOINT_MISMATCH"
    INTERRUPT_STATE_ERROR = "INTERRUPT_STATE_ERROR"
    SUSPEND_IGNORED = "SUSPEND_IGNORED"


class UiPathClaudeSDKRuntimeError(UiPathBaseRuntimeError):
    """Custom exception for Claude SDK runtime errors with structured error information."""

    def __init__(
        self,
        code: UiPathClaudeSDKErrorCode | UiPathErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
    ):
        super().__init__(
            code.value, title, detail, category, status, prefix="ClaudeSDK"
        )


__all__ = [
    "UiPathClaudeSDKErrorCode",
    "UiPathClaudeSDKRuntimeError",
]
