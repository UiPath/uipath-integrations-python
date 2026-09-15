"""Tests for the `uipath new` OpenAI agents scaffolding middleware."""

import os

import pytest

from uipath_openai_agents._cli import cli_new
from uipath_openai_agents._cli.cli_new import openai_agents_new_middleware


def test_middleware_scaffolds_project_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The middleware writes main.py, config, AGENTS.md and pyproject into cwd."""
    monkeypatch.chdir(tmp_path)

    result = openai_agents_new_middleware("my_agent")

    assert result.should_continue is False
    for name in ("main.py", "openai_agents.json", "AGENTS.md", "pyproject.toml"):
        assert os.path.exists(tmp_path / name)
    # project name is threaded into the generated pyproject
    assert 'name = "my_agent"' in (tmp_path / "pyproject.toml").read_text()


def test_middleware_reports_stacktrace_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A generation failure returns a non-continuing result requesting a stacktrace."""
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli_new, "generate_pyproject", boom)
    errors: list[str] = []
    monkeypatch.setattr(cli_new.console, "error", lambda msg, **kw: errors.append(msg))

    result = openai_agents_new_middleware("my_agent")

    assert errors and "disk full" in errors[0]
    assert result.should_continue is False
    assert result.should_include_stacktrace is True


def test_middleware_writes_pyproject_once(tmp_path, monkeypatch):
    """The scaffold generated pyproject.toml twice per run; guard against that."""
    monkeypatch.chdir(tmp_path)
    calls = []
    original = cli_new.generate_pyproject

    def counting(directory, project_name):
        calls.append(directory)
        return original(directory, project_name)

    monkeypatch.setattr(cli_new, "generate_pyproject", counting)
    result = openai_agents_new_middleware("demo")

    assert result.should_continue is False
    assert len(calls) == 1
