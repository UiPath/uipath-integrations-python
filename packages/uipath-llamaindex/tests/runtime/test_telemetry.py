"""Tests for the ToolCallAttributeNormalizer span processor."""

import json
from types import SimpleNamespace
from typing import Any, cast

from opentelemetry.sdk.trace import ReadableSpan

from uipath_llamaindex.runtime._telemetry import ToolCallAttributeNormalizer


def _tool_span(attributes: dict[str, Any]) -> ReadableSpan:
    return cast(ReadableSpan, SimpleNamespace(name="tool-span", _attributes=attributes))


def test_input_value_kwargs_wrapper_is_unwrapped():
    attrs = {
        "openinference.span.kind": "TOOL",
        "input.value": json.dumps({"kwargs": {"city": "Paris"}}),
    }
    ToolCallAttributeNormalizer().on_end(_tool_span(attrs))

    assert json.loads(attrs["input.value"]) == {"city": "Paris"}


def test_output_value_is_reshaped_to_common_format():
    attrs = {
        "openinference.span.kind": "TOOL",
        "output.value": json.dumps(
            {"raw_output": "sunny", "is_error": False, "tool_call_id": "call-1"}
        ),
    }
    ToolCallAttributeNormalizer().on_end(_tool_span(attrs))

    assert json.loads(attrs["output.value"]) == {
        "content": "sunny",
        "status": "success",
        "tool_call_id": "call-1",
    }


def test_non_tool_spans_are_left_untouched():
    original = json.dumps({"kwargs": {"a": 1}})
    attrs = {"openinference.span.kind": "LLM", "input.value": original}
    ToolCallAttributeNormalizer().on_end(_tool_span(attrs))

    assert attrs["input.value"] == original


def test_span_without_attributes_returns_early():
    # Should not raise when the span exposes no attributes.
    ToolCallAttributeNormalizer().on_end(_tool_span({}))
