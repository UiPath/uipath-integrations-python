"""Tests for the LlamaIndex runtime error type."""

from uipath.runtime.errors import UiPathErrorCategory

from uipath_llamaindex.runtime.errors import (
    UiPathLlamaIndexErrorCode,
    UiPathLlamaIndexRuntimeError,
)


def test_runtime_error_carries_structured_fields():
    error = UiPathLlamaIndexRuntimeError(
        code=UiPathLlamaIndexErrorCode.WORKFLOW_NOT_FOUND,
        title="Not found",
        detail="workflow missing",
        category=UiPathErrorCategory.USER,
    )

    assert error.error_info.code == "LlamaIndex.WORKFLOW_NOT_FOUND"
    assert error.error_info.title == "Not found"
    assert error.error_info.detail == "workflow missing"
