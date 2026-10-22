import os
import shutil

import click
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult

console = ConsoleLogger()


def generate_script(target_directory):
    for template, target in (
        ("main.py.template", "main.py"),
        ("codex.json.template", "codex.json"),
    ):
        src = os.path.join(os.path.dirname(__file__), "_templates", template)
        shutil.copyfile(src, os.path.join(target_directory, target))


def generate_pyproject(target_directory, project_name):
    toml_content = f"""[project]
name = "{project_name}"
version = "0.0.1"
description = "{project_name}"
authors = [{{ name = "John Doe", email = "john.doe@myemail.com" }}]
dependencies = [
    "uipath-codex-sdk>=0.1.0, <0.2.0",
]
requires-python = ">=3.11"

[dependency-groups]
dev = [
    "uipath-dev>=0.0.74",
]
"""
    with open(os.path.join(target_directory, "pyproject.toml"), "w") as f:
        f.write(toml_content)


def codex_new_middleware(name: str) -> MiddlewareResult:
    """Middleware to create a demo Codex SDK agent."""
    directory = os.getcwd()
    try:
        with console.spinner(f"Creating new agent {name} in current directory ..."):
            generate_script(directory)
            console.success("Created 'main.py' file.")
            console.success("Created 'codex.json' file.")
            generate_pyproject(directory, name)
            console.success("Created 'pyproject.toml' file.")
            init_command = "uipath init"
            run_command = """uipath run agent '{"topic": "robots"}'"""
            console.hint(
                f""" Initialize project: {click.style(init_command, fg="cyan")}"""
            )
            console.hint(f""" Run agent: {click.style(run_command, fg="cyan")}""")
        return MiddlewareResult(should_continue=False)
    except Exception as e:
        console.error(f"Error creating demo agent {str(e)}")
        return MiddlewareResult(should_continue=False, should_include_stacktrace=True)
