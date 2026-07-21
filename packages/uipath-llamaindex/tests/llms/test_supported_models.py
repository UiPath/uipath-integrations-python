"""Tests for the supported model identifier registries."""

from uipath_llamaindex.llms.supported_models import (
    BedrockModel,
    GeminiModel,
    OpenAIModel,
)


def test_openai_model_enum_exposes_string_values():
    assert OpenAIModel.GPT_4_1_2025_04_14.value == "gpt-4.1-2025-04-14"
    assert OpenAIModel("gpt-4o-mini-2024-07-18") is OpenAIModel.GPT_4O_MINI_2024_07_18


def test_gemini_and_bedrock_identifiers_are_plain_strings():
    assert GeminiModel.gemini_2_5_flash == "gemini-2.5-flash"
    assert (
        BedrockModel.anthropic_claude_sonnet_4_5
        == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
