"""Tests for the `uipath new` Agent Framework scaffolding middleware."""

from uipath_agent_framework._cli import cli_new
from uipath_agent_framework._cli.cli_new import (
    agent_framework_new_middleware,
    generate_pyproject,
)


def test_generate_pyproject_writes_project_name(tmp_path):
    generate_pyproject(str(tmp_path), "my-agent")
    content = (tmp_path / "pyproject.toml").read_text()
    assert 'name = "my-agent"' in content
    assert "uipath-agent-framework" in content


def test_middleware_scaffolds_project_files(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    result = agent_framework_new_middleware("demo")

    assert result.should_continue is False
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "agent_framework.json").exists()
    assert (tmp_path / "pyproject.toml").exists()


def test_middleware_reports_error_with_stacktrace_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli_new, "generate_pyproject", boom)
    # console.error exits via the click context, which isn't active in tests.
    monkeypatch.setattr(cli_new.console, "error", lambda *a, **k: None)
    result = agent_framework_new_middleware("demo")

    assert result.should_continue is False
    assert result.should_include_stacktrace is True
