"""Tests for _cli/cli_new.py — project scaffolding middleware."""

import tomllib
from importlib.metadata import version

from packaging.requirements import Requirement

from uipath_google_adk._cli.cli_new import (
    generate_pyproject,
    generate_script,
    google_adk_new_middleware,
)


class TestGeneratePyproject:
    def test_writes_pyproject_with_project_name(self, tmp_path):
        generate_pyproject(str(tmp_path), "my-agent")
        content = (tmp_path / "pyproject.toml").read_text()
        assert 'name = "my-agent"' in content
        assert "uipath-google-adk" in content

    def test_accepts_the_current_release(self, tmp_path):
        generate_pyproject(str(tmp_path), "my-agent")
        project = tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]
        requirement = next(
            Requirement(dep)
            for dep in project["dependencies"]
            if Requirement(dep).name == "uipath-google-adk"
        )
        assert requirement.specifier.contains(version("uipath-google-adk"))


class TestGenerateScript:
    def test_copies_templates(self, tmp_path):
        generate_script(str(tmp_path))
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / "google_adk.json").exists()


class TestMiddleware:
    def test_creates_project_and_stops_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = google_adk_new_middleware("demo")
        assert result.should_continue is False
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / "google_adk.json").exists()
        assert (tmp_path / "pyproject.toml").exists()
