"""Tests for `UiPathLlamaIndexRuntimeFactory.is_conversational`."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from uipath_llamaindex.runtime.factory import UiPathLlamaIndexRuntimeFactory


def _make_factory(config_path: str) -> UiPathLlamaIndexRuntimeFactory:
    """Build a factory with a minimal context pointing at the given config path.

    We bypass `__init__` to avoid the instrumentation side effects in
    `_setup_instrumentation`, which are not relevant to this property.
    """
    factory = UiPathLlamaIndexRuntimeFactory.__new__(UiPathLlamaIndexRuntimeFactory)
    factory.context = MagicMock()
    factory.context.config_path = config_path
    return factory


def test_is_conversational_true_when_flag_set(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({"runtimeOptions": {"isConversational": True}}))

    factory = _make_factory(str(config_path))

    assert factory.is_conversational is True


def test_is_conversational_false_when_flag_set_false(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({"runtimeOptions": {"isConversational": False}}))

    factory = _make_factory(str(config_path))

    assert factory.is_conversational is False


def test_is_conversational_false_when_flag_missing(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({"runtimeOptions": {}}))

    factory = _make_factory(str(config_path))

    assert factory.is_conversational is False


def test_is_conversational_false_when_runtime_options_missing(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({}))

    factory = _make_factory(str(config_path))

    assert factory.is_conversational is False


def test_is_conversational_false_when_file_missing(tmp_path: Path):
    factory = _make_factory(str(tmp_path / "does-not-exist.json"))

    assert factory.is_conversational is False


def test_is_conversational_false_when_file_unparseable(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text("{ not json")

    factory = _make_factory(str(config_path))

    assert factory.is_conversational is False


def test_is_conversational_false_when_file_unreadable(tmp_path: Path):
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({"runtimeOptions": {"isConversational": True}}))

    factory = _make_factory(str(config_path))

    with patch("builtins.open", side_effect=PermissionError("denied")):
        assert factory.is_conversational is False


def test_is_conversational_is_cached(tmp_path: Path):
    """`cached_property` should only read the file once across accesses."""
    config_path = tmp_path / "uipath.json"
    config_path.write_text(json.dumps({"runtimeOptions": {"isConversational": True}}))

    factory = _make_factory(str(config_path))

    # First access reads the file.
    assert factory.is_conversational is True

    # If we now delete the file, a re-read would return False — but the cached
    # value must still be True.
    config_path.unlink()
    assert factory.is_conversational is True
