"""Error handling for PydanticAI runtime."""

from enum import Enum

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
    UiPathErrorCode,
)


class UiPathPydanticAIErrorCode(Enum):
    """Error codes specific to PydanticAI runtime."""

    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    SERIALIZE_OUTPUT_ERROR = "SERIALIZE_OUTPUT_ERROR"
    STREAM_ERROR = "STREAM_ERROR"

    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_TYPE_ERROR = "AGENT_TYPE_ERROR"
    AGENT_VALUE_ERROR = "AGENT_VALUE_ERROR"
    AGENT_LOAD_FAILURE = "AGENT_LOAD_FAILURE"
    AGENT_IMPORT_ERROR = "AGENT_IMPORT_ERROR"

    SCHEMA_INFERENCE_ERROR = "SCHEMA_INFERENCE_ERROR"


class UiPathPydanticAIRuntimeError(UiPathBaseRuntimeError):
    """Custom exception for PydanticAI runtime errors with structured error information."""

    def __init__(
        self,
        code: UiPathPydanticAIErrorCode | UiPathErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
    ):
        super().__init__(
            code.value, title, detail, category, status, prefix="PydanticAI"
        )


__all__ = [
    "UiPathPydanticAIErrorCode",
    "UiPathPydanticAIRuntimeError",
]
