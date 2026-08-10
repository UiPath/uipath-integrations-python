"""Version-drift guard for the pinned claude-agent-sdk and its bundled CLI.

Almost everything this package depends on for suspend and resume lives in the
Claude Code CLI binary shipped inside the claude-agent-sdk wheel, not in the
Python package. A routine dependency bump can therefore change runtime
behaviour while every Python signature stays identical. These tests are the
tripwire: they are hermetic (no subprocess, no network, no tenant, no API key)
and they fail loudly when the bundled CLI or the Python-side primitives move.
"""

from __future__ import annotations

import dataclasses
from typing import Any, get_args, get_origin, get_type_hints

import claude_agent_sdk
from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_agent_sdk._cli_version import __cli_version__
from claude_agent_sdk.types import (
    DeferredToolUse,
    HookMatcher,
    PreToolUseHookSpecificOutput,
    ResultMessage,
)

from tests.conftest import make_result_message

VALIDATED_CLI_VERSIONS = frozenset({"2.1.224"})


def test_bundled_cli_version_is_validated():
    assert __cli_version__ in VALIDATED_CLI_VERSIONS, (
        f"claude-agent-sdk {claude_agent_sdk.__version__} bundles Claude Code CLI "
        f"{__cli_version__}, which this package has not been validated against. "
        "The suspend primitive is implemented in the CLI binary, so a Python-only "
        "review proves nothing. Before adding a version here, drive a real agent "
        "against this CLI and confirm the full round trip: a PreToolUse hook "
        "returning permissionDecision 'defer' parks the call without running the "
        "tool and reports it as ResultMessage.deferred_tool_use, and resuming that "
        "session in a NEW process re-issues the same tool_use_id so the handler "
        "runs with its original arguments. Then add "
        f'"{__cli_version__}" to VALIDATED_CLI_VERSIONS in this file.'
    )


def test_defer_is_an_accepted_permission_decision():
    hints = get_type_hints(PreToolUseHookSpecificOutput)
    assert "defer" in get_args(hints["permissionDecision"])


def test_deferred_tool_use_shape():
    fields = {field.name: field for field in dataclasses.fields(DeferredToolUse)}
    assert set(fields) == {"id", "name", "input"}

    hints = get_type_hints(DeferredToolUse)
    assert hints["id"] is str
    assert hints["name"] is str
    assert get_origin(hints["input"]) is dict
    assert get_args(hints["input"]) == (str, Any)


def test_result_message_carries_deferred_tool_use():
    field_names = {field.name for field in dataclasses.fields(ResultMessage)}
    assert "deferred_tool_use" in field_names

    hints = get_type_hints(ResultMessage)
    assert DeferredToolUse in get_args(hints["deferred_tool_use"])


def test_suspend_signal_is_deferred_tool_use_not_terminal_reason():
    """The runtime detects a suspension through deferred_tool_use, never
    terminal_reason.

    terminal_reason is present on the currently pinned sdk but was absent on
    0.2.110, where the defer round trip was also verified to work. Reading it
    would therefore couple the runtime to a field that is not guaranteed across
    the supported range, for no benefit: deferred_tool_use is the signal and it
    carries the pending call as well.
    """
    result = make_result_message()
    assert result.deferred_tool_use is None


def test_hook_matcher_accepts_a_timeout():
    field_names = {field.name for field in dataclasses.fields(HookMatcher)}
    assert "timeout" in field_names

    matcher = HookMatcher(matcher="Bash", hooks=[], timeout=1.5)
    assert matcher.timeout == 1.5


def test_sdk_mcp_server_primitives_are_importable():
    assert callable(tool)
    assert callable(create_sdk_mcp_server)
