"""Tests for the `uipath new` PydanticAI scaffolding middleware."""

import os
import shutil
import tomllib
from importlib.metadata import version

import click
import pytest
from packaging.requirements import Requirement

from uipath_pydantic_ai._cli.cli_new import (
    generate_pyproject,
    pydantic_ai_new_middleware,
)


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


def test_scaffold_accepts_the_current_release(tmp_path):
    generate_pyproject(str(tmp_path), "my-agent")
    project = tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]
    requirement = next(
        Requirement(dep)
        for dep in project["dependencies"]
        if Requirement(dep).name == "uipath-pydantic-ai"
    )
    assert requirement.specifier.contains(version("uipath-pydantic-ai"))
