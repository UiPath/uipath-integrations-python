"""Tests for GoogleADKAgentLoader."""


import pytest

from uipath_google_adk.runtime.errors import UiPathGoogleADKRuntimeError
from uipath_google_adk.runtime.loader import GoogleADKAgentLoader


class TestGoogleADKAgentLoaderFromPathString:
    """Tests for GoogleADKAgentLoader.from_path_string()."""

    def test_valid_path_string(self):
        """Test parsing valid path string."""
        loader = GoogleADKAgentLoader.from_path_string("test", "main.py:agent")
        assert loader.name == "test"
        assert loader.file_path == "main.py"
        assert loader.variable_name == "agent"

    def test_path_with_multiple_colons(self):
        """Test path with multiple colons uses first split."""
        loader = GoogleADKAgentLoader.from_path_string("test", "path/to/file.py:var:extra")
        assert loader.file_path == "path/to/file.py"
        assert loader.variable_name == "var:extra"

    def test_invalid_path_raises_error(self):
        """Test path without colon raises error."""
        with pytest.raises(UiPathGoogleADKRuntimeError) as exc_info:
            GoogleADKAgentLoader.from_path_string("test", "main.py")
        assert "Invalid path format" in str(exc_info.value)


class TestGoogleADKAgentLoaderLoad:
    """Tests for GoogleADKAgentLoader.load()."""

    @pytest.mark.asyncio
    async def test_load_nonexistent_file_raises_error(self):
        """Test loading nonexistent file raises error."""
        loader = GoogleADKAgentLoader(
            name="test",
            file_path="nonexistent_file_12345.py",
            variable_name="agent",
        )
        with pytest.raises(UiPathGoogleADKRuntimeError) as exc_info:
            await loader.load()
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_outside_cwd_raises_error(self):
        """Test loading file outside CWD raises error."""
        loader = GoogleADKAgentLoader(
            name="test",
            file_path="/tmp/../../../etc/passwd",
            variable_name="agent",
        )
        with pytest.raises(UiPathGoogleADKRuntimeError):
            await loader.load()

    @pytest.mark.asyncio
    async def test_load_missing_variable_raises_error(self, tmp_path, monkeypatch):
        """Test loading with missing variable raises error."""
        monkeypatch.chdir(tmp_path)

        # Create a simple Python file without the expected variable
        agent_file = tmp_path / "test_agent.py"
        agent_file.write_text("x = 42\n")

        loader = GoogleADKAgentLoader(
            name="test",
            file_path="test_agent.py",
            variable_name="agent",
        )
        with pytest.raises(UiPathGoogleADKRuntimeError) as exc_info:
            await loader.load()
        assert "not found in module" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_wrong_type_raises_error(self, tmp_path, monkeypatch):
        """Test loading non-BaseAgent object raises error."""
        monkeypatch.chdir(tmp_path)

        # Create a Python file with a non-agent variable
        agent_file = tmp_path / "test_agent.py"
        agent_file.write_text("agent = 'not an agent'\n")

        loader = GoogleADKAgentLoader(
            name="test",
            file_path="test_agent.py",
            variable_name="agent",
        )
        with pytest.raises(UiPathGoogleADKRuntimeError) as exc_info:
            await loader.load()
        assert "Expected BaseAgent" in str(exc_info.value)
