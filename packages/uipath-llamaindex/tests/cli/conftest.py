from pathlib import Path

import pytest

from uipath_llamaindex.runtime import register_runtime_factory

# Register the LlamaIndex runtime factory for CLI tests
register_runtime_factory()

# Get the tests directory path relative to this conftest file
TESTS_DIR = Path(__file__).parent.parent
MOCKS_DIR = TESTS_DIR / "mocks"


@pytest.fixture
def simple_script_basic_config() -> str:
    mock_file = MOCKS_DIR / "simple_script_basic_config.py"
    with open(mock_file, "r") as file:
        data = file.read()
    return data


@pytest.fixture
def simple_script_custom_config() -> str:
    mock_file = MOCKS_DIR / "simple_script_custom_config.py"
    with open(mock_file, "r") as file:
        data = file.read()
    return data


@pytest.fixture
def llama_config() -> str:
    mock_file = MOCKS_DIR / "llama_index.json"
    with open(mock_file, "r") as file:
        data = file.read()
    return data
