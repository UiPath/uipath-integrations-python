"""Tests for PydanticAI runtime factory."""

import json
import os
import tempfile

import pytest

from uipath_pydantic_ai.runtime.config import PydanticAiConfig


def test_config_exists():
    """Test config existence check."""
    config = PydanticAiConfig(config_path="nonexistent.json")
    assert config.exists is False


def test_config_load_agents():
    """Test loading agents from config file."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump({"agents": {"agent": "main.py:agent"}}, f)
        f.close()

        config = PydanticAiConfig(config_path=f.name)
        assert config.exists is True
        assert "agent" in config.agents
        assert config.agents["agent"] == "main.py:agent"
        assert config.entrypoint == ["agent"]
    finally:
        os.unlink(f.name)


def test_config_multiple_agents():
    """Test loading multiple agents from config."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(
            {
                "agents": {
                    "agent1": "main.py:agent1",
                    "agent2": "other.py:agent2",
                }
            },
            f,
        )
        f.close()

        config = PydanticAiConfig(config_path=f.name)
        assert len(config.agents) == 2
        assert "agent1" in config.entrypoint
        assert "agent2" in config.entrypoint
    finally:
        os.unlink(f.name)


def test_config_missing_agents_key():
    """Test that missing 'agents' key raises ValueError."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump({"other": "value"}, f)
        f.close()

        config = PydanticAiConfig(config_path=f.name)
        with pytest.raises(ValueError, match="Missing 'agents' key"):
            _ = config.agents
    finally:
        os.unlink(f.name)


def test_config_invalid_json():
    """Test that invalid JSON raises ValueError."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        f.write("not valid json{")
        f.close()

        config = PydanticAiConfig(config_path=f.name)
        with pytest.raises(ValueError, match="Invalid JSON"):
            _ = config.agents
    finally:
        os.unlink(f.name)


def test_config_file_not_found():
    """Test that missing config file raises FileNotFoundError."""
    config = PydanticAiConfig(config_path="nonexistent_config.json")
    with pytest.raises(FileNotFoundError):
        _ = config.agents
