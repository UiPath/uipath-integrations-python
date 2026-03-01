"""Tests for PydanticAI runtime execution."""

import pytest

from uipath_pydantic_ai.runtime.errors import (
    UiPathPydanticAIErrorCode,
    UiPathPydanticAIRuntimeError,
)


def test_error_handling():
    """Test that error handling works correctly."""
    error = UiPathPydanticAIRuntimeError(
        code=UiPathPydanticAIErrorCode.AGENT_EXECUTION_ERROR,
        title="Test error",
        detail="This is a test error",
    )

    # Verify error can be created and contains the detail message
    assert isinstance(error, UiPathPydanticAIRuntimeError)
    assert "This is a test error" in str(error)

    # Verify error can be raised
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc_info:
        raise error

    assert "This is a test error" in str(exc_info.value)


def test_error_codes():
    """Test that all error codes are accessible."""
    assert (
        UiPathPydanticAIErrorCode.AGENT_EXECUTION_ERROR.value == "AGENT_EXECUTION_ERROR"
    )
    assert UiPathPydanticAIErrorCode.AGENT_TIMEOUT.value == "AGENT_TIMEOUT"
    assert (
        UiPathPydanticAIErrorCode.SERIALIZE_OUTPUT_ERROR.value
        == "SERIALIZE_OUTPUT_ERROR"
    )
    assert UiPathPydanticAIErrorCode.STREAM_ERROR.value == "STREAM_ERROR"
    assert UiPathPydanticAIErrorCode.CONFIG_MISSING.value == "CONFIG_MISSING"
    assert UiPathPydanticAIErrorCode.CONFIG_INVALID.value == "CONFIG_INVALID"
    assert UiPathPydanticAIErrorCode.AGENT_NOT_FOUND.value == "AGENT_NOT_FOUND"
    assert UiPathPydanticAIErrorCode.AGENT_TYPE_ERROR.value == "AGENT_TYPE_ERROR"
    assert UiPathPydanticAIErrorCode.AGENT_VALUE_ERROR.value == "AGENT_VALUE_ERROR"
    assert UiPathPydanticAIErrorCode.AGENT_LOAD_FAILURE.value == "AGENT_LOAD_FAILURE"
    assert UiPathPydanticAIErrorCode.AGENT_IMPORT_ERROR.value == "AGENT_IMPORT_ERROR"
    assert (
        UiPathPydanticAIErrorCode.SCHEMA_INFERENCE_ERROR.value
        == "SCHEMA_INFERENCE_ERROR"
    )


def test_runtime_input_preparation():
    """Test that runtime correctly prepares agent input (conversational mode)."""
    from pydantic_ai import Agent

    from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime

    agent = Agent("test", name="test_agent")
    runtime = UiPathPydanticAIRuntime(agent=agent)

    # Test string messages -> (prompt, None deps)
    prompt, deps = runtime._prepare_input({"messages": "Hello"})
    assert prompt == "Hello"
    assert deps is None

    # Test empty input
    prompt, deps = runtime._prepare_input(None)
    assert prompt == ""
    assert deps is None

    prompt, deps = runtime._prepare_input({})
    assert prompt == ""
    assert deps is None

    # Test non-string/non-list messages
    prompt, deps = runtime._prepare_input({"messages": 123})
    assert prompt == ""
    assert deps is None

    # Test list input
    prompt, deps = runtime._prepare_input(
        {"messages": [{"content": "Hello"}, {"content": "World"}]}
    )
    assert "Hello" in prompt
    assert "World" in prompt
    assert deps is None


def test_runtime_structured_input_preparation():
    """Test that runtime correctly prepares structured deps input."""
    from pydantic import BaseModel
    from pydantic_ai import Agent

    from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime

    class MyInput(BaseModel):
        query: str
        max_results: int = 5

    agent = Agent("test", name="test_agent", deps_type=MyInput)
    runtime = UiPathPydanticAIRuntime(agent=agent)

    # Test structured input -> deps model
    prompt, deps = runtime._prepare_input({"query": "test", "max_results": 3})
    assert prompt == ""
    assert isinstance(deps, MyInput)
    assert deps.query == "test"
    assert deps.max_results == 3


def test_runtime_structured_input_with_messages():
    """Test that deps model with a 'messages' field uses it as prompt."""
    from pydantic import BaseModel
    from pydantic_ai import Agent

    from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime

    class ReviewInput(BaseModel):
        messages: str
        review_type: str

    agent = Agent("test", name="test_agent", deps_type=ReviewInput)
    runtime = UiPathPydanticAIRuntime(agent=agent)

    prompt, deps = runtime._prepare_input(
        {"messages": "Review this code", "review_type": "security"}
    )
    assert prompt == "Review this code"
    assert isinstance(deps, ReviewInput)
    assert deps.review_type == "security"
