"""What ``uipath new`` leaves on disk for a Claude SDK agent.

The command is registered as a ``new`` middleware, so it runs in the user's
current directory and reports through the platform console rather than
returning anything the CLI inspects. These tests pin the scaffold it writes and
the contract it returns, including the failure path, where a raised error has to
become a reported one or the CLI would carry on and scaffold a second time.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

import click
import pytest

from uipath_claude_sdk._cli import cli_new
from uipath_claude_sdk._cli.cli_new import claude_new_middleware
from uipath_claude_sdk.middlewares import register_middleware

TEMPLATES = Path(cli_new.__file__).parent / "_templates"


@pytest.fixture
def in_project(tmp_path: Path):
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous)


def test_new_writes_the_agent_and_its_config(in_project: Path):
    result = claude_new_middleware("weather-agent")

    assert result.should_continue is False
    assert (in_project / "main.py").read_text() == (
        TEMPLATES / "main.py.template"
    ).read_text()
    assert (in_project / "claude.json").read_text() == (
        TEMPLATES / "claude.json.template"
    ).read_text()


def test_new_names_the_project_after_the_agent(in_project: Path):
    claude_new_middleware("weather-agent")

    project = tomllib.loads((in_project / "pyproject.toml").read_text())["project"]
    assert project["name"] == "weather-agent"
    assert project["description"] == "weather-agent"
    assert project["requires-python"] == ">=3.11"
    assert any(dep.startswith("uipath-claude-sdk") for dep in project["dependencies"])


def test_new_fails_the_command_instead_of_half_scaffolding(
    in_project: Path, monkeypatch
):
    """``console.error`` ends the command, so the middleware never returns here."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(shutil, "copyfile", refuse)

    with click.Context(click.Command("new")):
        with pytest.raises(click.exceptions.Exit) as exit_info:
            claude_new_middleware("weather-agent")

    assert exit_info.value.exit_code == 1
    assert not (in_project / "pyproject.toml").exists()


def test_new_is_registered_as_the_new_middleware():
    from uipath._cli.middlewares import Middlewares

    register_middleware()

    assert claude_new_middleware in Middlewares._middlewares["new"]
