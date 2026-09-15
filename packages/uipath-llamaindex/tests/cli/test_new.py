"""Tests for the `uipath new` LlamaIndex scaffolding middleware."""

import pytest

from uipath_llamaindex._cli import cli_new
from uipath_llamaindex._cli.cli_new import llamaindex_new_middleware


def test_middleware_scaffolds_project_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The middleware writes main.py, llama_index.json and pyproject into cwd."""
    monkeypatch.chdir(tmp_path)

    result = llamaindex_new_middleware("my-agent")

    assert result.should_continue is False
    for name in ("main.py", "llama_index.json", "pyproject.toml"):
        assert (tmp_path / name).exists()
    pyproject = (tmp_path / "pyproject.toml").read_text()
    assert 'name = "my-agent"' in pyproject
    assert "uipath-llamaindex" in pyproject


def test_middleware_writes_pyproject_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scaffold generated pyproject.toml twice per run; guard against that."""
    monkeypatch.chdir(tmp_path)
    calls = []
    original = cli_new.generate_pyproject

    def counting(directory, project_name):
        calls.append(directory)
        return original(directory, project_name)

    monkeypatch.setattr(cli_new, "generate_pyproject", counting)
    result = llamaindex_new_middleware("demo")

    assert result.should_continue is False
    assert len(calls) == 1
