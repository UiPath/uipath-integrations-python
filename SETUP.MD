# SETUP.MD

This file documents how to provision a clean development environment for the five packages in this repo (`uipath-agent-framework`, `uipath-google-adk`, `uipath-llamaindex`, `uipath-openai-agents`, `uipath-pydantic-ai`), run the build, execute the tests, and validate a sample code change end-to-end. It is intended both as a quick reference for human contributors and as a structured guide for automated environment-setup tooling.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 0.5+

### Supported platforms

`uv` is shell- and OS-agnostic, so the commands below run unchanged on every supported platform:

- [x] Linux
- [x] Windows
- [x] macOS

## Environment Variables

None required for environment setup, build, or unit tests. The suites under the `Test` section run fully offline and require no external authentication.

> **All commands below must be run from the repository root.** The `uv --directory packages/<name>` invocations resolve each subpackage relative to the current working directory. The first line of `## Setup` enforces this by `cd`-ing to the git root.

## Setup

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m pip install --upgrade uv

# Sync all five packages (each is independent)
uv --directory packages/uipath-agent-framework sync --all-extras
uv --directory packages/uipath-google-adk sync --all-extras
uv --directory packages/uipath-llamaindex sync --all-extras
uv --directory packages/uipath-openai-agents sync --all-extras
uv --directory packages/uipath-pydantic-ai sync --all-extras
```

## Verify Setup

```bash
uv --version
uv --directory packages/uipath-pydantic-ai run python --version
uv --directory packages/uipath-agent-framework run python -c "import uipath_agent_framework; print('uipath-agent-framework ok')"
uv --directory packages/uipath-google-adk run python -c "import uipath_google_adk; print('uipath-google-adk ok')"
uv --directory packages/uipath-llamaindex run python -c "import uipath_llamaindex; print('uipath-llamaindex ok')"
uv --directory packages/uipath-openai-agents run python -c "import uipath_openai_agents; print('uipath-openai-agents ok')"
uv --directory packages/uipath-pydantic-ai run python -c "import uipath_pydantic_ai; print('uipath-pydantic-ai ok')"
```

## Build

N/A

## Test

```bash
uv --directory packages/uipath-agent-framework run pytest
uv --directory packages/uipath-google-adk run pytest
uv --directory packages/uipath-llamaindex run pytest
uv --directory packages/uipath-openai-agents run pytest
uv --directory packages/uipath-pydantic-ai run pytest
```

## Sample Code Change

### The change

Add a new `agent_count` property to `PydanticAiConfig` in `packages/uipath-pydantic-ai/src/uipath_pydantic_ai/runtime/config.py`, immediately after the existing `entrypoint` property:

```python
@property
def agent_count(self) -> int:
    """Number of agents defined in the configuration."""
    return len(self.agents)
```

Then create `packages/uipath-pydantic-ai/tests/test_config_agent_count.py` with two pytest tests:

```python
"""Tests for PydanticAiConfig.agent_count."""

import json
from pathlib import Path

from uipath_pydantic_ai.runtime.config import PydanticAiConfig


def test_agent_count_single(tmp_path: Path) -> None:
    config_path = tmp_path / "pydantic_ai.json"
    config_path.write_text(json.dumps({"agents": {"main": "main:agent"}}))
    cfg = PydanticAiConfig(str(config_path))
    assert cfg.agent_count == 1


def test_agent_count_multiple(tmp_path: Path) -> None:
    config_path = tmp_path / "pydantic_ai.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "alpha": "alpha:agent",
                    "beta": "beta:agent",
                    "gamma": "gamma:agent",
                }
            }
        )
    )
    cfg = PydanticAiConfig(str(config_path))
    assert cfg.agent_count == 3
```

### Verification

```bash
uv --directory packages/uipath-pydantic-ai run pytest tests/test_config_agent_count.py -v
```

## Test with a real UiPath Coded Agent

The unit tests above are necessary but not sufficient — they don't exercise the package end-to-end through a real agent. The flow below validates changes against a live runtime:

1. Apply the code changes locally.
2. Run the unit tests (see the `Sample Code Change` section above).
3. Scaffold a coded UiPath agent (PydanticAI / OpenAI / Google ADK / LlamaIndex / Agent Framework, matching the package you changed) that exercises the changed code path.
4. In the downstream project's `pyproject.toml`, add this local library as an editable dependency (substitute the package you changed):

   ```toml
   [tool.uv.sources]
   uipath-pydantic-ai = { path = "../path/to/uipath-integrations-python/packages/uipath-pydantic-ai", editable = true }
   ```

5. Exercise the new behavior end-to-end:

   ```bash
   uv run uipath run <agent-name> --input '{...}'
   ```

6. (Optional) Open a PR and apply the `build:dev` label — this publishes the development version to Test PyPI.
7. The PR description is updated automatically with instructions for pointing the downstream agent at the Test PyPI dev version.
8. Push the dev version to UiPath with [`uipath push`](https://uipath.github.io/uipath-python/cli/#push), then deploy it to Orchestrator or Studio Web with [`uipath deploy`](https://uipath.github.io/uipath-python/cli/#deploy), and run it in cloud to confirm the changes behave correctly against the real platform.
9. Once validation is done, close the dev PR — these PRs are not meant to be merged; their only purpose was to publish a Test PyPI build for end-to-end validation.
