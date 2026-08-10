"""Shared fixtures for uipath-claude-sdk tests."""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from pydantic import BaseModel

from uipath_claude_sdk import ClaudeAgent


class SampleInput(BaseModel):
    topic: str


class SampleOutput(BaseModel):
    summary: str


@pytest.fixture
def structured_agent() -> ClaudeAgent:
    return ClaudeAgent(
        options=ClaudeAgentOptions(model="anthropic.claude-sonnet-4-5-20250929-v1:0"),
        input_schema=SampleInput,
        output_schema=SampleOutput,
        prompt="Summarize {topic}",
        name="research",
    )


@pytest.fixture
def plain_agent() -> ClaudeAgent:
    return ClaudeAgent(
        options=ClaudeAgentOptions(model="anthropic.claude-sonnet-4-5-20250929-v1:0"),
    )


def make_result_message(
    *,
    result: str | None = "done",
    structured_output: Any = None,
    is_error: bool = False,
    session_id: str = "session-1",
) -> ResultMessage:
    return ResultMessage(
        subtype="success" if not is_error else "error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        result=result,
        structured_output=structured_output,
    )


class FakeClaudeSDKClient:
    """Fake ClaudeSDKClient yielding a scripted message sequence."""

    last_options: ClaudeAgentOptions | None = None
    last_query: str | None = None
    scripted_messages: list[Any] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        type(self).last_options = options

    async def __aenter__(self) -> "FakeClaudeSDKClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def query(self, message: str) -> None:
        type(self).last_query = message

    async def receive_response(self):
        for message in type(self).scripted_messages:
            yield message


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClaudeSDKClient]:
    """Patch ClaudeSDKClient in both runtime modules and bypass the gateway proxy."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.runtime.ClaudeSDKClient", FakeClaudeSDKClient
    )
    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.conversational_runtime.ClaudeSDKClient",
        FakeClaudeSDKClient,
    )
    FakeClaudeSDKClient.scripted_messages = [
        AssistantMessage(
            content=[
                TextBlock(text="Hello"),
                ToolUseBlock(id="tu-1", name="my_tool", input={"a": 1}),
            ],
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        make_result_message(result="Hello", structured_output={"summary": "Hello"}),
    ]
    FakeClaudeSDKClient.last_options = None
    FakeClaudeSDKClient.last_query = None
    return FakeClaudeSDKClient
