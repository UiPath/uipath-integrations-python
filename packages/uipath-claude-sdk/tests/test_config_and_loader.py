"""Tests for claude.json config parsing and the agent loader."""

from __future__ import annotations

import json
import os

import pytest

from uipath_claude_sdk import ClaudeAgent
from uipath_claude_sdk.runtime.config import ClaudeConfig
from uipath_claude_sdk.runtime.errors import UiPathClaudeSDKRuntimeError
from uipath_claude_sdk.runtime.loader import ClaudeAgentLoader


def test_config_string_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claude.json").write_text(
        json.dumps({"agents": {"agent": "main.py:agent"}})
    )
    config = ClaudeConfig()
    assert config.exists
    assert config.entrypoint == ["agent"]
    assert config.agents["agent"] == "main.py:agent"


def test_config_invalid_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claude.json").write_text(json.dumps({"agents": {"bad": 42}}))
    with pytest.raises(ValueError, match="Missing or invalid 'agents'"):
        _ = ClaudeConfig().agents


def test_loader_invalid_path_format():
    with pytest.raises(UiPathClaudeSDKRuntimeError):
        ClaudeAgentLoader.from_path_string("agent", "main.py")


async def test_loader_loads_claude_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text(
        "from claude_agent_sdk import ClaudeAgentOptions\n"
        "from uipath_claude_sdk import ClaudeAgent\n"
        "agent = ClaudeAgent(options=ClaudeAgentOptions(model='claude-sonnet-4-5'))\n"
    )
    loader = ClaudeAgentLoader.from_path_string("agent", "main.py:agent")
    agent = await loader.load()
    assert isinstance(agent, ClaudeAgent)
    assert agent.options.model == "claude-sonnet-4-5"


async def test_loader_wraps_bare_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text(
        "from claude_agent_sdk import ClaudeAgentOptions\n"
        "agent = ClaudeAgentOptions(model='claude-sonnet-4-5')\n"
    )
    loader = ClaudeAgentLoader.from_path_string("agent", "main.py:agent")
    agent = await loader.load()
    assert isinstance(agent, ClaudeAgent)
    assert agent.options.model == "claude-sonnet-4-5"
    assert agent.name == "agent"


async def test_loader_rejects_paths_outside_cwd(tmp_path, monkeypatch):
    outside = tmp_path / "outside.py"
    outside.write_text("agent = None\n")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    loader = ClaudeAgentLoader.from_path_string(
        "agent", os.path.join("..", "outside.py") + ":agent"
    )
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="working directory"):
        await loader.load()


async def test_loader_rejects_wrong_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("agent = 42\n")
    loader = ClaudeAgentLoader.from_path_string("agent", "main.py:agent")
    with pytest.raises(
        UiPathClaudeSDKRuntimeError, match="Expected uipath_claude_sdk.ClaudeAgent"
    ):
        await loader.load()
