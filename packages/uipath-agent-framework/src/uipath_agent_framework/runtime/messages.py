"""Chat message mapper for Agent Framework <-> UiPath Conversation format.

Handles:
- Inbound: UiPath conversation messages -> Agent Framework message input
- Outbound: Agent Framework streaming updates -> UiPath conversation message events
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agent_framework import Content
from pydantic import ValidationError
from uipath.core.chat import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
    UiPathConversationMessage,
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
    UiPathConversationToolCallEndEvent,
    UiPathConversationToolCallEvent,
    UiPathConversationToolCallStartEvent,
    UiPathInlineValue,
)

logger = logging.getLogger(__name__)


class AgentFrameworkChatMessagesMapper:
    """Maps between UiPath Conversation format and Agent Framework format.

    Maintains state across events to properly track message lifecycle
    and correlate tool calls with their responses.
    """

    def __init__(self) -> None:
        self._current_message_id: str | None = None
        self._message_started: bool = False
        self._pending_tool_calls: dict[str, str] = {}

    # -- Inbound: UiPath messages -> string or list for agent.run() --

    def map_messages_to_input(self, messages: Any) -> str:
        """Convert UiPath input messages to a string for agent.run().

        Agent Framework agents accept str or list[ChatMessage] for run().
        For simplicity, we extract text and pass as string.
        """
        if isinstance(messages, str):
            return messages

        if isinstance(messages, list) and messages:
            first = messages[0]

            # UiPath conversation message objects
            if isinstance(first, UiPathConversationMessage):
                return self._extract_text_from_uipath_messages(messages)

            # Dicts -> try parsing as UiPath conversation messages
            if isinstance(first, dict):
                try:
                    parsed = [
                        UiPathConversationMessage.model_validate(m) for m in messages
                    ]
                    return self._extract_text_from_uipath_messages(parsed)
                except ValidationError:
                    pass

                # Fallback: extract text from dicts
                return self._extract_text_from_raw(messages)

            # Simple strings
            if isinstance(first, str):
                return "\n".join(messages)

        return str(messages) if messages else ""

    def _extract_text_from_uipath_messages(
        self, messages: list[UiPathConversationMessage]
    ) -> str:
        """Extract text from the last user message in UiPath conversation format."""
        for msg in reversed(messages):
            if msg.role == "user":
                text = self._extract_text_from_content_parts(msg)
                if text:
                    return text
        # Fallback: extract from last message regardless of role
        if messages:
            text = self._extract_text_from_content_parts(messages[-1])
            if text:
                return text
        return ""

    @staticmethod
    def _extract_text_from_content_parts(msg: UiPathConversationMessage) -> str:
        """Extract text from UiPath message content parts."""
        texts: list[str] = []
        if msg.content_parts:
            for cp in msg.content_parts:
                if cp.mime_type.startswith("text/") and isinstance(
                    cp.data, UiPathInlineValue
                ):
                    text = str(cp.data.inline)
                    if text:
                        texts.append(text)
        return "".join(texts)

    @staticmethod
    def _extract_text_from_raw(messages: list[Any]) -> str:
        """Extract text from raw message list (fallback)."""
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict):
                parts.append(msg.get("content", str(msg)))
            else:
                parts.append(str(msg))
        return "\n".join(parts)

    # -- Outbound: Agent Framework updates -> UiPath conversation events --

    def map_streaming_content(
        self, content: Content
    ) -> list[UiPathConversationMessageEvent]:
        """Convert an Agent Framework Content to UiPath conversation message events.

        Maps:
        - Text content -> ContentPartChunk (streaming text)
        - FunctionCallContent -> ToolCallStart
        - FunctionResultContent -> ToolCallEnd
        """
        events: list[UiPathConversationMessageEvent] = []

        if content.type == "function_call":
            # During streaming, only the first chunk carries content.name.
            # Subsequent chunks are partial argument fragments — skip them.
            if not content.name:
                return events

            tool_call_id = content.call_id or f"{content.name}_{uuid4().hex[:8]}"

            events.extend(self._ensure_message_started())
            if self._current_message_id:
                self._pending_tool_calls[tool_call_id] = self._current_message_id
            args = content.arguments if isinstance(content.arguments, dict) else None
            events.append(
                self._make_tool_call_start_event(
                    self._current_message_id or "",
                    tool_call_id,
                    content.name,
                    args,
                )
            )
            return events

        if content.type == "function_result":
            if content.call_id and content.call_id in self._pending_tool_calls:
                message_id = self._pending_tool_calls.pop(content.call_id)
                result = content.result or {}
                events.append(
                    self._make_tool_call_end_event(message_id, content.call_id, result)
                )
            # Close the current message after all tool calls complete
            # so that the agent's text response starts a new message.
            if not self._pending_tool_calls:
                events.extend(self.close_message())
            return events

        # Text content
        text = self._extract_text_from_content(content)
        if text:
            events.extend(self._ensure_message_started())
            if self._current_message_id:
                events.append(
                    self._make_content_chunk_event(self._current_message_id, text)
                )

        return events

    def close_message(self) -> list[UiPathConversationMessageEvent]:
        """Close the current message if open. Safety net for end of stream."""
        if self._message_started and self._current_message_id:
            events = [self._make_message_end_event(self._current_message_id)]
            self._message_started = False
            self._current_message_id = None
            return events
        return []

    @staticmethod
    def _extract_text_from_content(content: Content) -> str:
        """Extract text from an Agent Framework Content object."""
        if content.text:
            return str(content.text)
        return ""

    def _ensure_message_started(self) -> list[UiPathConversationMessageEvent]:
        """Start a new message if not already started."""
        if self._message_started:
            return []
        self._current_message_id = str(uuid4())
        self._message_started = True
        return [self._make_message_start_event(self._current_message_id)]

    @staticmethod
    def _get_timestamp() -> str:
        """Format current time as ISO 8601 UTC with milliseconds."""
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _get_content_part_id(message_id: str) -> str:
        return f"chunk-{message_id}-0"

    @classmethod
    def _make_message_start_event(
        cls, message_id: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            start=UiPathConversationMessageStartEvent(
                role="assistant", timestamp=cls._get_timestamp()
            ),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=cls._get_content_part_id(message_id),
                start=UiPathConversationContentPartStartEvent(mime_type="text/plain"),
            ),
        )

    @classmethod
    def _make_content_chunk_event(
        cls, message_id: str, text: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=cls._get_content_part_id(message_id),
                chunk=UiPathConversationContentPartChunkEvent(data=text),
            ),
        )

    @classmethod
    def _make_tool_call_start_event(
        cls,
        message_id: str,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            tool_call=UiPathConversationToolCallEvent(
                tool_call_id=tool_call_id,
                start=UiPathConversationToolCallStartEvent(
                    tool_name=tool_name,
                    timestamp=cls._get_timestamp(),
                    input=args,
                ),
            ),
        )

    @classmethod
    def _make_tool_call_end_event(
        cls, message_id: str, tool_call_id: str, output: Any
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            tool_call=UiPathConversationToolCallEvent(
                tool_call_id=tool_call_id,
                end=UiPathConversationToolCallEndEvent(
                    timestamp=cls._get_timestamp(),
                    output=output,
                ),
            ),
        )

    @classmethod
    def _make_message_end_event(cls, message_id: str) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            end=UiPathConversationMessageEndEvent(),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=cls._get_content_part_id(message_id),
                end=UiPathConversationContentPartEndEvent(),
            ),
        )


__all__ = ["AgentFrameworkChatMessagesMapper"]
