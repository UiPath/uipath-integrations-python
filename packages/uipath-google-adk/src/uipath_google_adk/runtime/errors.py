"""Error handling for Google ADK runtime."""

from enum import Enum

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
    UiPathErrorCode,
)


class UiPathGoogleADKErrorCode(Enum):
    """Error codes specific to Google ADK runtime."""

    AGENT_EXECUTION_FAILURE = "AGENT_EXECUTION_FAILURE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    SERIALIZE_OUTPUT_ERROR = "SERIALIZE_OUTPUT_ERROR"

    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_TYPE_ERROR = "AGENT_TYPE_ERROR"
    AGENT_VALUE_ERROR = "AGENT_VALUE_ERROR"
    AGENT_LOAD_ERROR = "AGENT_LOAD_ERROR"
    AGENT_IMPORT_ERROR = "AGENT_IMPORT_ERROR"


class UiPathGoogleADKRuntimeError(UiPathBaseRuntimeError):
    """Custom exception for Google ADK runtime errors with structured error information."""

    def __init__(
        self,
        code: UiPathGoogleADKErrorCode | UiPathErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
    ):
        super().__init__(
            code.value, title, detail, category, status, prefix="Google-ADK"
        )


__all__ = [
    "UiPathGoogleADKErrorCode",
    "UiPathGoogleADKRuntimeError",
]
