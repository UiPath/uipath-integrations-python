import os
import shutil

import click
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult

console = ConsoleLogger()


def generate_script(target_directory):
    template_script_path = os.path.join(
        os.path.dirname(__file__), "_templates/main.py.template"
    )
    target_path = os.path.join(target_directory, "main.py")

    shutil.copyfile(template_script_path, target_path)

    template_openai_agents_json_path = os.path.join(
        os.path.dirname(__file__), "_templates/openai_agents.json.template"
    )
    target_path = os.path.join(target_directory, "openai_agents.json")
    shutil.copyfile(template_openai_agents_json_path, target_path)

    # Copy OpenAI-specific AGENTS.md template
    template_agents_md_path = os.path.join(
        os.path.dirname(__file__), "_templates/AGENTS.md.template"
    )
    target_agents_md = os.path.join(target_directory, "AGENTS.md")
    if os.path.exists(template_agents_md_path):
        shutil.copyfile(template_agents_md_path, target_agents_md)


def generate_pyproject(target_directory, project_name):
    project_toml_path = os.path.join(target_directory, "pyproject.toml")
    toml_content = f"""[project]
name = "{project_name}"
version = "0.0.1"
description = "{project_name}"
authors = [{{ name = "John Doe", email = "john.doe@myemail.com" }}]
dependencies = [
    "uipath-openai-agents>=0.0.1, <0.1.0",
]
requires-python = ">=3.11"
"""

    with open(project_toml_path, "w") as f:
        f.write(toml_content)


def openai_agents_new_middleware(name: str) -> MiddlewareResult:
    """Middleware to create demo OpenAI agent"""

    directory = os.getcwd()

    try:
        with console.spinner(f"Creating new agent {name} in current directory ..."):
            generate_pyproject(directory, name)
            generate_script(directory)
            console.success("Created 'main.py' file.")
            console.success("Created 'openai_agents.json' file.")
            console.success("Created 'AGENTS.md' file.")
            generate_pyproject(directory, name)
            console.success("Created 'pyproject.toml' file.")
            init_command = """uipath init"""
            run_command = """uipath run agent '{"messages": "What is the weather in San Francisco?"}'"""
            console.hint(
                f""" Initialize project: {click.style(init_command, fg="cyan")}"""
            )
            console.hint(f""" Run agent: {click.style(run_command, fg="cyan")}""")
        return MiddlewareResult(should_continue=False)
    except Exception as e:
        console.error(f"Error creating demo agent {str(e)}")
        return MiddlewareResult(
            should_continue=False,
            should_include_stacktrace=True,
        )
