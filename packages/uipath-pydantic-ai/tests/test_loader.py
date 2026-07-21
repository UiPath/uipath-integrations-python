"""Tests for the PydanticAI agent loader."""

import os
from pathlib import Path

import pytest
from pydantic_ai import Agent

from uipath_pydantic_ai.runtime.errors import (
    UiPathPydanticAIErrorCode,
    UiPathPydanticAIRuntimeError,
)
from uipath_pydantic_ai.runtime.loader import PydanticAiAgentLoader


def _write_agent_file(directory: str, body: str, filename: str = "main.py") -> str:
    path = os.path.join(directory, filename)
    Path(path).write_text(body)
    return path


# ============= PATH PARSING =============


def test_from_path_string_parses_file_and_variable():
    """A valid 'file:variable' string is split into its parts."""
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:my_agent")
    assert loader.file_path == "main.py"
    assert loader.variable_name == "my_agent"


def test_from_path_string_rejects_missing_colon():
    """A path without a colon raises a CONFIG_INVALID error."""
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        PydanticAiAgentLoader.from_path_string("agent", "main.py")
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.CONFIG_INVALID.value
    )


# ============= LOAD SUCCESS PATHS =============


@pytest.mark.asyncio
async def test_load_direct_agent_instance(tmp_path, monkeypatch):
    """A module exposing a plain Agent instance loads successfully."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(
        str(tmp_path),
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "agent = Agent(TestModel(), name='direct')\n",
    )
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:agent")
    agent = await loader.load()
    assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_load_factory_function(tmp_path, monkeypatch):
    """A module exposing a sync factory function is called to produce the Agent."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(
        str(tmp_path),
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "def build():\n"
        "    return Agent(TestModel(), name='factory')\n",
    )
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:build")
    agent = await loader.load()
    assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_load_async_factory_function(tmp_path, monkeypatch):
    """A module exposing an async factory coroutine is awaited to produce the Agent."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(
        str(tmp_path),
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "async def build():\n"
        "    return Agent(TestModel(), name='async_factory')\n",
    )
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:build")
    agent = await loader.load()
    assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_load_async_context_manager_and_cleanup(tmp_path, monkeypatch):
    """An async-context-manager object is entered on load and exited on cleanup."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(
        str(tmp_path),
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.models.test import TestModel\n"
        "class Managed:\n"
        "    exited = False\n"
        "    async def __aenter__(self):\n"
        "        return Agent(TestModel(), name='ctx')\n"
        "    async def __aexit__(self, *a):\n"
        "        type(self).exited = True\n"
        "cm = Managed()\n",
    )
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:cm")
    agent = await loader.load()
    assert isinstance(agent, Agent)
    # cleanup should invoke __aexit__ on the stored context manager
    assert loader._context_manager is not None
    await loader.cleanup()
    assert loader._context_manager is None


# ============= LOAD ERROR PATHS =============


@pytest.mark.asyncio
async def test_load_rejects_path_outside_cwd(tmp_path, monkeypatch):
    """A file path that escapes the working directory raises AGENT_VALUE_ERROR."""
    monkeypatch.chdir(tmp_path)
    loader = PydanticAiAgentLoader.from_path_string("agent", "../outside.py:agent")
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_VALUE_ERROR.value
    )


@pytest.mark.asyncio
async def test_load_missing_file(tmp_path, monkeypatch):
    """A non-existent agent file raises AGENT_NOT_FOUND."""
    monkeypatch.chdir(tmp_path)
    loader = PydanticAiAgentLoader.from_path_string("agent", "missing.py:agent")
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_NOT_FOUND.value
    )


@pytest.mark.asyncio
async def test_load_missing_variable(tmp_path, monkeypatch):
    """A module that lacks the requested variable raises AGENT_NOT_FOUND."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(str(tmp_path), "x = 1\n")
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:agent")
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_NOT_FOUND.value
    )


@pytest.mark.asyncio
async def test_load_wrong_type(tmp_path, monkeypatch):
    """A variable that is not a pydantic_ai.Agent raises AGENT_TYPE_ERROR."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(str(tmp_path), "agent = 'not an agent'\n")
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:agent")
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_TYPE_ERROR.value
    )


@pytest.mark.asyncio
async def test_load_module_execution_error(tmp_path, monkeypatch):
    """A module that raises at import time surfaces AGENT_LOAD_FAILURE."""
    monkeypatch.chdir(tmp_path)
    _write_agent_file(str(tmp_path), "raise RuntimeError('boom')\n")
    loader = PydanticAiAgentLoader.from_path_string("agent", "main.py:agent")
    with pytest.raises(UiPathPydanticAIRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code.endswith(
        UiPathPydanticAIErrorCode.AGENT_LOAD_FAILURE.value
    )


@pytest.mark.asyncio
async def test_cleanup_without_context_manager_is_noop():
    """cleanup() is safe to call when no context manager was entered."""
    loader = PydanticAiAgentLoader("agent", "main.py", "agent")
    await loader.cleanup()  # should not raise
    assert loader._context_manager is None
