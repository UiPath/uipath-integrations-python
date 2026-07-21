"""Tests for the `uipath new` PydanticAI scaffolding middleware."""

import os
import shutil

import click
import pytest

from uipath_pydantic_ai._cli.cli_new import pydantic_ai_new_middleware


def test_new_middleware_scaffolds_project(tmp_path, monkeypatch):
    """The middleware writes main.py, pydantic_ai.json and pyproject.toml, then stops."""
    monkeypatch.chdir(tmp_path)

    result = pydantic_ai_new_middleware("my-agent")

    assert result.should_continue is False
    assert os.path.exists(tmp_path / "main.py")
    assert os.path.exists(tmp_path / "pydantic_ai.json")
    pyproject = (tmp_path / "pyproject.toml").read_text()
    assert 'name = "my-agent"' in pyproject
    assert "uipath-pydantic-ai" in pyproject


def test_new_middleware_reports_failure(tmp_path, monkeypatch):
    """A generation failure is caught and reported via console.error (which exits)."""
    monkeypatch.chdir(tmp_path)

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", _boom)

    # console.error() calls click's ctx.exit(1); provide an active context so the
    # except branch reports the failure instead of raising "no active click context".
    with click.Context(click.Command("new")):
        with pytest.raises((SystemExit, click.exceptions.Exit)):
            pydantic_ai_new_middleware("broken")
