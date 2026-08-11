"""
Simple trace assertion - just check that expected spans exist with required attributes.
"""
import json
from typing import Any


def load_traces(traces_file: str) -> list[dict[str, Any]]:
    """Load traces from a JSONL file."""
    traces = []
    with open(traces_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def load_expected_traces(expected_file: str) -> list[dict[str, Any]]:
    """Load expected trace definitions from a JSON file."""
    with open(expected_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("required_spans", [])


def get_attributes(span: dict[str, Any]) -> dict[str, Any]:
    """Parse attributes from a span."""
    if "attributes" in span and isinstance(span["attributes"], dict):
        return span["attributes"]
    attributes_str = span.get("Attributes", "{}")
    try:
        return json.loads(attributes_str)
    except json.JSONDecodeError:
        return {}


def matches_value(expected_value: Any, actual_value: Any) -> bool:
    """Check if an actual value matches the expected value."""
    if expected_value == "*":
        return True
    if isinstance(expected_value, list):
        return actual_value in expected_value
    return expected_value == actual_value


def matches_expected(span: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check if a span matches the expected definition."""
    expected_name = expected.get("name")
    actual_name = span.get("name") or span.get("Name")
    if isinstance(expected_name, list):
        if actual_name not in expected_name:
            return False
    elif expected_name != actual_name:
        return False
    if "attributes" in expected:
        actual_attrs = get_attributes(span)
        for key, expected_value in expected["attributes"].items():
            if key not in actual_attrs:
                return False
            if not matches_value(expected_value, actual_attrs[key]):
                return False
    return True


def assert_traces(traces_file: str, expected_file: str) -> None:
    """Assert that all expected traces exist in the traces file."""
    traces = load_traces(traces_file)
    expected_spans = load_expected_traces(expected_file)
    print(f"Loaded {len(traces)} traces from {traces_file}")
    print(f"Checking {len(expected_spans)} expected spans...")
    missing_spans = []
    for expected in expected_spans:
        found = False
        name = expected["name"]
        name_str = name if isinstance(name, str) else f"[{' | '.join(name)}]"
        for span in traces:
            if matches_expected(span, expected):
                found = True
                print(f"✓ Found span: {name_str}")
                break
        if not found:
            missing_spans.append(name_str)
            print(f"✗ Missing span: {name_str}")
    if missing_spans:
        raise AssertionError(
            f"Missing expected spans: {', '.join(missing_spans)}\n"
            f"Expected {len(expected_spans)} spans, found {len(expected_spans) - len(missing_spans)}"
        )
    print(f"\n✓ All {len(expected_spans)} expected spans found!")
