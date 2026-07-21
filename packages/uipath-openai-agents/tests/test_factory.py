"""Tests for UiPathOpenAIAgentRuntimeFactory."""

import pytest
from uipath.runtime import UiPathRuntimeContext

from uipath_openai_agents.runtime.errors import (
    UiPathOpenAIAgentsErrorCode,
    UiPathOpenAIAgentsRuntimeError,
)
from uipath_openai_agents.runtime.factory import UiPathOpenAIAgentRuntimeFactory
from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime

MOCK_AGENT = (
    'from agents import Agent\nagent = Agent(name="basic", instructions="hi")\n'
)


@pytest.fixture
def factory() -> UiPathOpenAIAgentRuntimeFactory:
    return UiPathOpenAIAgentRuntimeFactory(context=UiPathRuntimeContext())


def _write_project(tmp_path, monkeypatch) -> None:
    (tmp_path / "main.py").write_text(MOCK_AGENT)
    (tmp_path / "openai_agents.json").write_text(
        '{"agents": {"basic": "main.py:agent"}}'
    )
    monkeypatch.chdir(tmp_path)


def test_discover_entrypoints_empty_without_config(
    factory: UiPathOpenAIAgentRuntimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """With no openai_agents.json, no entrypoints are discovered."""
    monkeypatch.chdir(tmp_path)

    assert factory.discover_entrypoints() == []


@pytest.mark.asyncio
async def test_load_agent_missing_config_raises_config_missing(
    factory: UiPathOpenAIAgentRuntimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Loading an agent without a config file raises CONFIG_MISSING."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(UiPathOpenAIAgentsRuntimeError) as exc:
        await factory._load_agent("basic")

    assert exc.value.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.CONFIG_MISSING.value
    )


@pytest.mark.asyncio
async def test_load_agent_unknown_entrypoint_raises_agent_not_found(
    factory: UiPathOpenAIAgentRuntimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Requesting an entrypoint absent from config raises AGENT_NOT_FOUND."""
    _write_project(tmp_path, monkeypatch)

    with pytest.raises(UiPathOpenAIAgentsRuntimeError) as exc:
        await factory._load_agent("nope")

    assert exc.value.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.AGENT_NOT_FOUND.value
    )


@pytest.mark.asyncio
async def test_new_runtime_loads_and_caches_agent(
    factory: UiPathOpenAIAgentRuntimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """new_runtime resolves the configured agent and reuses the cached instance."""
    _write_project(tmp_path, monkeypatch)

    runtime = await factory.new_runtime("basic", "rt-1")

    assert isinstance(runtime, UiPathOpenAIAgentRuntime)
    assert runtime.agent.name == "basic"
    # second resolution comes from the cache (same Agent object)
    runtime2 = await factory.new_runtime("basic", "rt-2")
    assert isinstance(runtime2, UiPathOpenAIAgentRuntime)
    assert runtime2.agent is runtime.agent


@pytest.mark.asyncio
async def test_dispose_clears_caches_and_loaders(
    factory: UiPathOpenAIAgentRuntimeFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """dispose clears cached agents and loaders."""
    _write_project(tmp_path, monkeypatch)
    await factory.new_runtime("basic", "rt-1")

    await factory.dispose()

    assert factory._agent_cache == {}
    assert factory._agent_loaders == {}
