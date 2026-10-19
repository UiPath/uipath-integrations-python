"""Tests for the llamaindex project scaffold."""

import tomllib
from importlib.metadata import version

from packaging.requirements import Requirement

from uipath_llamaindex._cli.cli_new import generate_pyproject


def test_scaffold_accepts_the_current_release(tmp_path):
    generate_pyproject(str(tmp_path), "my-agent")
    project = tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]
    requirement = next(
        Requirement(dep)
        for dep in project["dependencies"]
        if Requirement(dep).name == "uipath-llamaindex"
    )
    assert requirement.specifier.contains(version("uipath-llamaindex"))
