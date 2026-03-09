import shutil
import tempfile
from typing import Generator

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for test files.

    Cleanup ignores errors to avoid PermissionError on Windows when
    SQLiteSession worker-thread connections still hold the state.db file open.
    """
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
