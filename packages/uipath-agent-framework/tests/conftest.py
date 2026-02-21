"""Shared test helpers for agent framework e2e tests."""

from __future__ import annotations

from typing import Any

from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def extract_system_text(messages: list[dict[str, Any]]) -> str:
    """Extract system/developer message text from OpenAI-format messages.

    The OpenAI client sends content as a list of content parts:
    [{"type": "text", "text": "..."}]
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("system", "developer"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return " ".join(parts)
    return ""


def make_chat_completion(text: str) -> ChatCompletion:
    """Create a mock OpenAI ChatCompletion response."""
    return ChatCompletion(
        id="test-completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


async def make_streaming_response(text: str):
    """Create an async iterable of ChatCompletionChunks for streaming."""
    yield ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=text),
                finish_reason=None,
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
    )
    yield ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def make_mock_response(text: str, stream: bool = False):
    """Return either a ChatCompletion or a streaming async iterable."""
    if stream:
        return make_streaming_response(text)
    return make_chat_completion(text)


def make_tool_call_completion(tool_name: str, arguments: str = "{}") -> ChatCompletion:
    """Create a mock ChatCompletion with a tool call."""
    return ChatCompletion(
        id="test-completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=f"call_{tool_name}",
                            function=Function(name=tool_name, arguments=arguments),
                            type="function",
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


async def make_streaming_tool_call(tool_name: str, arguments: str = "{}"):
    """Create streaming chunks for a tool call response."""
    yield ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id=f"call_{tool_name}",
                            function=ChoiceDeltaToolCallFunction(
                                name=tool_name,
                                arguments=arguments,
                            ),
                            type="function",
                        )
                    ],
                ),
                finish_reason=None,
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
    )
    yield ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="tool_calls",
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def make_tool_call_response(
    tool_name: str, arguments: str = "{}", stream: bool = False
):
    """Return either a tool call ChatCompletion or streaming chunks."""
    if stream:
        return make_streaming_tool_call(tool_name, arguments)
    return make_tool_call_completion(tool_name, arguments)
