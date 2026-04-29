"""Tests for uipath init command with openai-agents."""

import json
import os

from click.testing import CliRunner
from uipath._cli import cli


class TestInit:
    """Test init command for OpenAI agents."""

    def test_init_basic_config_generation(
        self,
        runner: CliRunner,
        temp_dir: str,
        simple_agent_basic: str,
        simple_agent_translation: str,
        openai_agents_config: str,
    ) -> None:
        """Test configuration file generation with basic agent."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create agent scripts (config references both)
            with open("main.py", "w") as f:
                f.write(simple_agent_basic)

            with open("translation.py", "w") as f:
                f.write(simple_agent_translation)

            with open("openai_agents.json", "w") as f:
                f.write(openai_agents_config)

            with open("pyproject.toml", "w") as f:
                f.write(
                    '[project]\nname = "test-project"\nversion = "0.1.0"\n'
                    'description = "Test project"\n'
                )

            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert os.path.exists("entry-points.json")

            with open("entry-points.json", "r") as f:
                config = json.load(f)

                # Verify config structure
                assert "entryPoints" in config

                # Verify entryPoints properties
                entry = config["entryPoints"][0]
                assert entry["filePath"] == "basic"
                assert entry["type"] == "agent"

                # Verify input schema
                assert "input" in entry
                input_schema = entry["input"]
                assert input_schema["type"] == "object"
                assert "properties" in input_schema
                assert "required" in input_schema
                assert isinstance(input_schema["properties"], dict)
                assert isinstance(input_schema["required"], list)

                # OpenAI agents use default messages input
                assert "messages" in input_schema["properties"]
                assert "messages" in input_schema["required"]

                # Verify output schema (default since no output_type specified)
                assert "output" in entry
                output_schema = entry["output"]
                assert "properties" in output_schema
                # Default output schema has "result" field
                assert "result" in output_schema["properties"]
                assert "required" in output_schema
                assert output_schema["type"] == "object"

    def test_init_translation_agent_config_generation(
        self,
        runner: CliRunner,
        temp_dir: str,
        simple_agent_basic: str,
        simple_agent_translation: str,
        openai_agents_config: str,
    ) -> None:
        """Test configuration file generation with translation agent."""
        with runner.isolated_filesystem(temp_dir=temp_dir):
            # Create agent scripts
            with open("main.py", "w") as f:
                f.write(simple_agent_basic)

            with open("translation.py", "w") as f:
                f.write(simple_agent_translation)

            with open("openai_agents.json", "w") as f:
                f.write(openai_agents_config)

            with open("pyproject.toml", "w") as f:
                f.write(
                    '[project]\nname = "test-project"\nversion = "0.1.0"\n'
                    'description = "Test project"\n'
                )

            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert os.path.exists("entry-points.json")

            with open("entry-points.json", "r") as f:
                config = json.load(f)

                # Verify config structure
                assert "entryPoints" in config

                # Find the translation entry point
                translation_entry = None
                for entry in config["entryPoints"]:
                    if entry["filePath"] == "translation":
                        translation_entry = entry
                        break

                assert translation_entry is not None
                assert translation_entry["type"] == "agent"

                # Verify input schema - should use default messages
                assert "input" in translation_entry
                input_schema = translation_entry["input"]
                assert input_schema["type"] == "object"
                assert "properties" in input_schema
                assert "messages" in input_schema["properties"]
                assert "messages" in input_schema["required"]

                # Verify output schema from agent's output_type
                assert "output" in translation_entry
                output_schema = translation_entry["output"]
                assert "properties" in output_schema

                # Verify output properties from TranslationOutput
                out_props = output_schema["properties"]
                assert "original_text" in out_props
                assert out_props["original_text"]["type"] == "string"
                assert out_props["original_text"]["description"] == "The original text"

                assert "translated_text" in out_props
                assert out_props["translated_text"]["type"] == "string"
                assert (
                    out_props["translated_text"]["description"] == "The translated text"
                )

                assert "target_language" in out_props
                assert out_props["target_language"]["type"] == "string"
                assert (
                    out_props["target_language"]["description"] == "The target language"
                )

                # Verify required fields in output
                assert "required" in output_schema
                assert "original_text" in output_schema["required"]
                assert "translated_text" in output_schema["required"]
                assert "target_language" in output_schema["required"]
                assert output_schema["type"] == "object"
