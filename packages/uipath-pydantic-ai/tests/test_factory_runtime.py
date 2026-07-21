"""Tests for the PydanticAI runtime factory orchestration."""

import json
from pathlib import Path

import pytest
from uipath.runtime import UiPathRuntimeContext

from uipath_pydantic_ai.runtime.errors import (
    UiPathPydanticAIErrorCode,
    UiPathPydanticAIRuntimeError,
)
from uipath_pydantic_ai.runtime.factory import UiPathPydanticAIRuntimeFactory
from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime


def _project(tmp_path: Path, agents: dict[str, str]) -> None:
    """Write a pydantic_ai.json plus a main.py exposing a TestModel agent."""
    (tmp_path / "pydantic_ai.json").write_text(json.dumps({"agents": agents}))
    (tmp_path / "main.py").write_text(
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "agent = Agent(TestModel(), name='factory_agent')\n"
    )


def _factory() -> UiPathPydanticAIRuntimeFactory:
    return UiPathPydanticAIRuntimeFactory(context=UiPathRuntimeContext())


def test_discover_entrypoints_empty_without_config(tmp_path, monkeypatch):
    """With no pydantic_ai.json present, no entrypoints are discovered."""
    monkeypatch.chdir(tmp_path)
    assert _factory().discover_entrypoints() == []


def test_discover_entrypoints_lists_configured_agents(tmp_path, monkeypatch):
    """Configured agent names are returned as entrypoints."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    assert _factory().discover_entrypoints() == ["agent"]


@pytest.mark.asyncio
async def test_load_agent_config_missing_raises(tmp_path, monkeypatch):
    """Loading an agent without a config file raises CONFIG_MISSING."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await _factory()._load_agent("agent")
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.CONFIG_MISSING.value
    )


@pytest.mark.asyncio
async def test_load_agent_unknown_entrypoint_raises(tmp_path, monkeypatch):
    """Requesting an entrypoint absent from config raises AGENT_NOT_FOUND."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await _factory()._load_agent("nope")
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_NOT_FOUND.value
    )


@pytest.mark.asyncio
async def test_resolve_agent_caches_result(tmp_path, monkeypatch):
    """Resolving the same entrypoint twice returns the cached Agent instance."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    factory = _factory()

    first = await factory._resolve_agent("agent")
    second = await factory._resolve_agent("agent")
    assert first is second


@pytest.mark.asyncio
async def test_new_runtime_builds_runtime_for_agent(tmp_path, monkeypatch):
    """new_runtime returns a runtime bound to the resolved agent."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    factory = _factory()

    runtime = await factory.new_runtime("agent", runtime_id="rt-1")
    assert isinstance(runtime, UiPathPydanticAIRuntime)


@pytest.mark.asyncio
async def test_discover_runtimes_one_per_entrypoint(tmp_path, monkeypatch):
    """discover_runtimes yields a runtime for each configured entrypoint."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    runtimes = await _factory().discover_runtimes()
    assert len(runtimes) == 1
    assert isinstance(runtimes[0], UiPathPydanticAIRuntime)


@pytest.mark.asyncio
async def test_dispose_clears_caches_and_loaders(tmp_path, monkeypatch):
    """dispose() cleans up loaders and empties the agent cache."""
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, {"agent": "main.py:agent"})
    factory = _factory()
    await factory._resolve_agent("agent")
    assert factory._agent_cache

    await factory.dispose()
    assert factory._agent_cache == {}
    assert factory._agent_loaders == {}


@pytest.mark.asyncio
async def test_factory_storage_and_settings_are_none(tmp_path, monkeypatch):
    """The factory advertises no shared storage or settings."""
    monkeypatch.chdir(tmp_path)
    factory = _factory()
    assert await factory.get_storage() is None
    assert await factory.get_settings() is None
