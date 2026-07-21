"""Tests for AgentFrameworkAgentLoader."""

import pytest

from uipath_agent_framework.runtime.errors import (
    UiPathAgentFrameworkRuntimeError,
)
from uipath_agent_framework.runtime.loader import AgentFrameworkAgentLoader


def test_from_path_string_parses_file_and_variable():
    loader = AgentFrameworkAgentLoader.from_path_string("bot", "main.py:agent")
    assert loader.file_path == "main.py"
    assert loader.variable_name == "agent"


def test_from_path_string_without_colon_raises_config_invalid():
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        AgentFrameworkAgentLoader.from_path_string("bot", "main.py")
    assert exc.value.error_info.code == "Agent-Framework.CONFIG_INVALID"


@pytest.mark.asyncio
async def test_load_rejects_path_outside_cwd():
    loader = AgentFrameworkAgentLoader("bot", "/etc/passwd", "agent")
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code == "Agent-Framework.AGENT_VALUE_ERROR"


@pytest.mark.asyncio
async def test_load_missing_file_raises_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = AgentFrameworkAgentLoader("bot", "missing.py", "agent")
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code == "Agent-Framework.AGENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_load_missing_variable_raises_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod_a.py").write_text("something = 1\n")
    loader = AgentFrameworkAgentLoader("bot", "mod_a.py", "agent")
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code == "Agent-Framework.AGENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_load_rejects_non_workflow_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod_b.py").write_text("agent = object()\n")
    loader = AgentFrameworkAgentLoader("bot", "mod_b.py", "agent")
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code == "Agent-Framework.AGENT_TYPE_ERROR"


@pytest.mark.asyncio
async def test_load_wraps_module_execution_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod_c.py").write_text("raise RuntimeError('boom')\n")
    loader = AgentFrameworkAgentLoader("bot", "mod_c.py", "agent")
    with pytest.raises(UiPathAgentFrameworkRuntimeError) as exc:
        await loader.load()
    assert exc.value.error_info.code == "Agent-Framework.AGENT_LOAD_ERROR"


@pytest.mark.asyncio
async def test_resolve_agent_calls_sync_factory():
    loader = AgentFrameworkAgentLoader("bot", "main.py", "agent")
    sentinel = object()
    resolved = await loader._resolve_agent(lambda: sentinel)
    assert resolved is sentinel


@pytest.mark.asyncio
async def test_resolve_agent_awaits_async_factory():
    loader = AgentFrameworkAgentLoader("bot", "main.py", "agent")
    sentinel = object()

    async def factory():
        return sentinel

    resolved = await loader._resolve_agent(factory)
    assert resolved is sentinel


@pytest.mark.asyncio
async def test_resolve_agent_enters_async_context_manager_and_cleanup():
    loader = AgentFrameworkAgentLoader("bot", "main.py", "agent")
    yielded = object()
    exited = []

    class _CM:
        async def __aenter__(self):
            return yielded

        async def __aexit__(self, *args):
            exited.append(args)

    resolved = await loader._resolve_agent(_CM())
    assert resolved is yielded

    await loader.cleanup()
    assert exited == [(None, None, None)]
    assert loader._context_manager is None
