"""Tests for OpenAiAgentsConfig loading."""

import pytest

from uipath_openai_agents.runtime.config import OpenAiAgentsConfig


def test_agents_and_entrypoint_parsed_from_file(tmp_path) -> None:
    """A valid config exposes the agents map and derived entrypoint list."""
    cfg_file = tmp_path / "openai_agents.json"
    cfg_file.write_text('{"agents": {"basic": "main.py:agent"}}')

    config = OpenAiAgentsConfig(str(cfg_file))

    assert config.exists is True
    assert config.agents == {"basic": "main.py:agent"}
    assert config.entrypoint == ["basic"]


def test_missing_agents_key_raises_value_error(tmp_path) -> None:
    """A config without the 'agents' key is rejected."""
    cfg_file = tmp_path / "openai_agents.json"
    cfg_file.write_text('{"other": 1}')

    with pytest.raises(ValueError, match="Missing 'agents' key"):
        _ = OpenAiAgentsConfig(str(cfg_file)).agents


def test_invalid_json_raises_value_error(tmp_path) -> None:
    """Malformed JSON surfaces as a ValueError referencing the path."""
    cfg_file = tmp_path / "openai_agents.json"
    cfg_file.write_text("{not json")

    with pytest.raises(ValueError, match="Invalid JSON"):
        _ = OpenAiAgentsConfig(str(cfg_file)).agents


def test_missing_file_raises_file_not_found(tmp_path) -> None:
    """Loading agents from a non-existent config raises FileNotFoundError."""
    config = OpenAiAgentsConfig(str(tmp_path / "nope.json"))

    assert config.exists is False
    with pytest.raises(FileNotFoundError):
        _ = config.agents
