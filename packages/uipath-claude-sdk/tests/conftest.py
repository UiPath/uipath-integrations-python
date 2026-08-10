"""Shared fixtures for uipath-claude-sdk tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import DeferredToolUse
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from pydantic import BaseModel

from uipath_claude_sdk import ClaudeAgent
from uipath_claude_sdk.runtime.session_paths import ClaudeSessionPaths
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage


@pytest.fixture(autouse=True)
def uninstrumented_claude_agent_sdk():
    """Undo the global SDK patching a runtime factory installs.

    ``ClaudeAgentSDKInstrumentor`` monkeypatches ``claude_agent_sdk`` for the
    whole process, so a test that builds a factory would otherwise decide what
    every later test runs against.
    """
    yield
    instrumentor = ClaudeAgentSDKInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


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
    deferred_tool_use: DeferredToolUse | None = None,
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
        deferred_tool_use=deferred_tool_use,
    )


def make_session_paths(tmp_path: Path, runtime_id: str = "rt-1") -> ClaudeSessionPaths:
    return ClaudeSessionPaths.for_runtime(tmp_path / "__uipath", runtime_id)


@pytest.fixture
async def storage(tmp_path: Path):
    """A state database disposed once the test finishes."""
    instance = SqliteResumableStorage(str(tmp_path / "state.db"))
    try:
        yield instance
    finally:
        await instance.dispose()


@pytest.fixture
def session_store(storage: SqliteResumableStorage) -> ClaudeSessionStore:
    return ClaudeSessionStore(storage, "rt-1")


class FakeClaudeSDKClient:
    """Fake ClaudeSDKClient yielding a scripted message sequence."""

    last_options: ClaudeAgentOptions | None = None
    last_query: str | None = None
    scripted_messages: list[Any] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        type(self).last_options = options
        self._stream: Any = None

    async def __aenter__(self) -> "FakeClaudeSDKClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def query(self, message: str) -> None:
        type(self).last_query = message

    async def receive_response(self):
        """Stop at the result, as the SDK's own convenience method does."""
        async for message in self.receive_messages():
            yield message
            if isinstance(message, ResultMessage):
                return

    async def receive_messages(self):
        """Draw from one stream, so a later turn does not replay an earlier one."""
        if self._stream is None:
            self._stream = iter(type(self).scripted_messages)
        for message in self._stream:
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
