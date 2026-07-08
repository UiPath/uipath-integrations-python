"""Tests for the runtime factory and registration."""

from __future__ import annotations

import json

import pytest
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactoryRegistry,
)

from uipath_claude_sdk.runtime import register_runtime_factory
from uipath_claude_sdk.runtime.conversational_runtime import (
    UiPathClaudeSDKConversationalRuntime,
)
from uipath_claude_sdk.runtime.errors import UiPathClaudeSDKRuntimeError
from uipath_claude_sdk.runtime.factory import UiPathClaudeSDKRuntimeFactory
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime

_MAIN_PY = (
    "from claude_agent_sdk import ClaudeAgentOptions\n"
    "from uipath_claude_sdk import ClaudeAgent\n"
    "agent = ClaudeAgent(options=ClaudeAgentOptions(model='claude-sonnet-4-5'))\n"
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text(_MAIN_PY)
    (tmp_path / "claude.json").write_text(
        json.dumps({"agents": {"agent": "main.py:agent"}})
    )
    return tmp_path


def _make_factory(tmp_path) -> UiPathClaudeSDKRuntimeFactory:
    return UiPathClaudeSDKRuntimeFactory(
        context=UiPathRuntimeContext(runtime_dir=str(tmp_path / "__uipath"))
    )


def test_discover_entrypoints(project):
    factory = _make_factory(project)
    assert factory.discover_entrypoints() == ["agent"]


async def test_new_runtime_standard(project):
    factory = _make_factory(project)
    runtime = await factory.new_runtime(entrypoint="agent", runtime_id="rt-1")
    assert isinstance(runtime, UiPathClaudeSDKRuntime)
    assert not isinstance(runtime, UiPathClaudeSDKConversationalRuntime)
    await factory.dispose()


async def test_new_runtime_conversational_from_uipath_json(project):
    (project / "uipath.json").write_text(
        json.dumps({"runtimeOptions": {"isConversational": True}})
    )
    factory = _make_factory(project)
    runtime = await factory.new_runtime(entrypoint="agent", runtime_id="rt-1")
    assert isinstance(runtime, UiPathResumableRuntime)
    assert isinstance(runtime.delegate, UiPathClaudeSDKConversationalRuntime)
    await factory.dispose()


async def test_new_runtime_conversational_from_conversation_id(project):
    factory = UiPathClaudeSDKRuntimeFactory(
        context=UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), conversation_id="conv-1"
        )
    )
    runtime = await factory.new_runtime(entrypoint="agent", runtime_id="rt-1")
    assert isinstance(runtime, UiPathResumableRuntime)
    assert isinstance(runtime.delegate, UiPathClaudeSDKConversationalRuntime)
    await factory.dispose()


async def test_unknown_entrypoint(project):
    factory = _make_factory(project)
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="not found. Available"):
        await factory.new_runtime(entrypoint="missing", runtime_id="rt-1")
    await factory.dispose()


async def test_get_settings(project):
    factory = _make_factory(project)
    settings = await factory.get_settings()
    assert settings is not None
    assert settings.agent_framework == "claude-agent-sdk"
    assert settings.agent_type == "uipath_coded"
    await factory.dispose()


def test_register_runtime_factory():
    register_runtime_factory()
    factories = UiPathRuntimeFactoryRegistry.get_all()
    assert factories.get("claude") == "claude.json"
