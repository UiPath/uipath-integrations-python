"""Tests for AgentFrameworkConfig."""

import json

import pytest

from uipath_agent_framework.runtime.config import AgentFrameworkConfig


class TestAgentFrameworkConfig:
    """Tests for AgentFrameworkConfig class."""

    def test_exists_returns_false_when_no_file(self):
        """Test exists returns False when config file doesn't exist."""
        config = AgentFrameworkConfig(config_path="/nonexistent/agent_framework.json")
        assert config.exists is False

    def test_exists_returns_true_when_file_exists(self, tmp_path):
        """Test exists returns True when config file exists."""
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text('{"agents": {"agent": "main.py:agent"}}')
        config = AgentFrameworkConfig(config_path=str(config_file))
        assert config.exists is True

    def test_agents_parses_valid_config(self, tmp_path):
        """Test agents property parses valid config correctly."""
        config_data = {"agents": {"agent": "main.py:agent", "other": "other.py:bot"}}
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text(json.dumps(config_data))

        config = AgentFrameworkConfig(config_path=str(config_file))
        assert config.agents == {"agent": "main.py:agent", "other": "other.py:bot"}

    def test_entrypoint_returns_agent_keys(self, tmp_path):
        """Test entrypoint returns list of agent names."""
        config_data = {"agents": {"agent": "main.py:agent", "other": "other.py:bot"}}
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text(json.dumps(config_data))

        config = AgentFrameworkConfig(config_path=str(config_file))
        assert sorted(config.entrypoint) == ["agent", "other"]

    def test_missing_agents_key_raises_value_error(self, tmp_path):
        """Test that missing 'agents' key raises ValueError."""
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text('{"other": "value"}')

        config = AgentFrameworkConfig(config_path=str(config_file))
        with pytest.raises(ValueError, match="Missing 'agents' key"):
            _ = config.agents

    def test_agents_not_dict_raises_value_error(self, tmp_path):
        """Test that non-dict 'agents' value raises ValueError."""
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text('{"agents": ["agent1", "agent2"]}')

        config = AgentFrameworkConfig(config_path=str(config_file))
        with pytest.raises(ValueError, match="'agents' must be a dictionary"):
            _ = config.agents

    def test_invalid_json_raises_value_error(self, tmp_path):
        """Test that invalid JSON raises ValueError."""
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text("not valid json {{{")

        config = AgentFrameworkConfig(config_path=str(config_file))
        with pytest.raises(ValueError, match="Invalid JSON"):
            _ = config.agents

    def test_file_not_found_raises_error(self):
        """Test that accessing agents on missing file raises FileNotFoundError."""
        config = AgentFrameworkConfig(config_path="/nonexistent/agent_framework.json")
        with pytest.raises(FileNotFoundError):
            _ = config.agents

    def test_agents_are_cached(self, tmp_path):
        """Test that agents are loaded only once (cached)."""
        config_data = {"agents": {"agent": "main.py:agent"}}
        config_file = tmp_path / "agent_framework.json"
        config_file.write_text(json.dumps(config_data))

        config = AgentFrameworkConfig(config_path=str(config_file))
        agents1 = config.agents
        agents2 = config.agents
        assert agents1 is agents2
