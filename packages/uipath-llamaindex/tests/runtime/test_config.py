"""Tests for LlamaIndexConfig loader."""

import json
import os

import pytest

from uipath_llamaindex.runtime.config import LlamaIndexConfig


def _write(tmp_path, content: str) -> str:
    path = os.path.join(tmp_path, "llama_index.json")
    with open(path, "w") as f:
        f.write(content)
    return path


def test_workflows_and_entrypoints_loaded(tmp_path):
    path = _write(tmp_path, json.dumps({"workflows": {"agent": "main.py:workflow"}}))
    config = LlamaIndexConfig(config_path=path)

    assert config.exists is True
    assert config.workflows == {"agent": "main.py:workflow"}
    assert config.entrypoints == ["agent"]


def test_missing_file_raises_file_not_found(tmp_path):
    config = LlamaIndexConfig(config_path=os.path.join(tmp_path, "nope.json"))
    assert config.exists is False
    with pytest.raises(FileNotFoundError):
        _ = config.workflows


def test_missing_workflows_field_raises_value_error(tmp_path):
    path = _write(tmp_path, json.dumps({"other": {}}))
    with pytest.raises(ValueError, match="Missing required 'workflows' field"):
        _ = LlamaIndexConfig(config_path=path).workflows


def test_workflows_must_be_a_dict(tmp_path):
    path = _write(tmp_path, json.dumps({"workflows": ["a", "b"]}))
    with pytest.raises(ValueError, match="'workflows' must be a dictionary"):
        _ = LlamaIndexConfig(config_path=path).workflows


def test_invalid_json_raises_value_error(tmp_path):
    path = _write(tmp_path, "{not valid json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        _ = LlamaIndexConfig(config_path=path).workflows
