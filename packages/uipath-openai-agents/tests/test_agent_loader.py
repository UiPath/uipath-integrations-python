"""Tests for OpenAiAgentLoader path parsing, loading and resolution."""

import pytest
from agents import Agent

from uipath_openai_agents.runtime.agent import OpenAiAgentLoader
from uipath_openai_agents.runtime.errors import (
    UiPathOpenAIAgentsErrorCode,
    UiPathOpenAIAgentsRuntimeError,
)


def test_from_path_string_rejects_missing_variable_separator() -> None:
    """A path without ':' is rejected as an invalid config format."""
    with pytest.raises(UiPathOpenAIAgentsRuntimeError) as exc:
        OpenAiAgentLoader.from_path_string("agent", "main.py")

    assert exc.value.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.CONFIG_INVALID.value
    )


def test_from_path_string_splits_file_and_variable() -> None:
    """A valid 'file:variable' path is split into its components."""
    loader = OpenAiAgentLoader.from_path_string("agent", "pkg/main.py:my_agent")

    assert loader.file_path == "pkg/main.py"
    assert loader.variable_name == "my_agent"


@pytest.mark.asyncio
async def test_load_rejects_file_outside_working_directory(tmp_path) -> None:
    """Loading a file outside the cwd is refused for safety."""
    loader = OpenAiAgentLoader("agent", "/etc/hosts", "agent")

    with pytest.raises(UiPathOpenAIAgentsRuntimeError) as exc:
        await loader.load()

    assert exc.value.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.AGENT_VALUE_ERROR.value
    )


@pytest.mark.asyncio
async def test_load_reports_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A non-existent agent file inside the cwd raises AGENT_NOT_FOUND."""
    monkeypatch.chdir(tmp_path)
    loader = OpenAiAgentLoader("agent", "missing.py", "agent")

    with pytest.raises(UiPathOpenAIAgentsRuntimeError) as exc:
        await loader.load()

    assert exc.value.error_info.code.endswith(
        UiPathOpenAIAgentsErrorCode.AGENT_NOT_FOUND.value
    )


@pytest.mark.asyncio
async def test_resolve_agent_invokes_sync_factory() -> None:
    """A callable returning an Agent is invoked and its result used."""
    agent = Agent(name="factory", instructions="x")
    loader = OpenAiAgentLoader("agent", "main.py", "agent")

    resolved = await loader._resolve_agent(lambda: agent)

    assert resolved is agent


@pytest.mark.asyncio
async def test_resolve_agent_enters_async_context_manager_and_cleans_up() -> None:
    """An async context manager is entered on resolve and exited on cleanup."""
    agent = Agent(name="ctx", instructions="x")
    exited = []

    class AgentCM:
        async def __aenter__(self):
            return agent

        async def __aexit__(self, *exc):
            exited.append(True)

    loader = OpenAiAgentLoader("agent", "main.py", "agent")

    resolved = await loader._resolve_agent(AgentCM())
    assert resolved is agent

    await loader.cleanup()
    assert exited == [True]
