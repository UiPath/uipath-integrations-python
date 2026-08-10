# Contributing to UiPath Claude Agent SDK

## Local Development Setup

### Prerequisites

1. **Install Python 3.11 or later**:
    - Download and install Python from the official [Python website](https://www.python.org/downloads/)
    - Verify the installation by running:
        ```sh
        python3.11 --version
        ```

    Alternative: [mise](https://mise.jdx.dev/lang/python.html)

    The package is pinned to 3.11 through `.python-version` and is tested on 3.11, 3.12 and 3.13.

2. **Install [uv](https://docs.astral.sh/uv/)**:
    ```sh
    pip install uv
    ```

3. **Create a virtual environment in the current working directory**:
    ```sh
    uv venv
    ```

4. **Install dependencies**:
    ```sh
    uv sync --all-extras
    ```

### Checks

Run these from `packages/uipath-claude-sdk`. They are the same commands CI runs:

```sh
uv run mypy --config-file pyproject.toml .
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`uv run ruff format .` rewrites the files in place instead of only reporting.

### Use SDK Locally
1. Create a folder on your own device `mkdir project; cd project`
2. Initialize the python project `uv` `uv init . --python 3.11`
3. Obtain the project path `PATH_TO_SDK=/Users/YOUR_USER/uipath-integrations-python/packages/uipath-claude-sdk/`
4. Install the sdk in editable mode `uv add --editable ${PATH_TO_SDK}`

:information_source: Instead of cloning the project into `.venv/lib/python3.11/site-packages/uipath-claude-sdk`, this mode creates a file named `_uipath-claude-sdk.pth` inside `.venv/lib/python3.11/site-packages`. That file contains the value of `PATH_TO_SDK`, which is added to `sys.path`, the list of directories where python searches for packages. (Run `python -c 'import sys; print(sys.path)'` to see the entries.)
