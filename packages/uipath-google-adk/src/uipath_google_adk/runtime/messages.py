"""Chat message mapper for Google ADK <-> UiPath Conversation format.

Handles:
- Inbound: UiPath conversation messages → Google ADK Content list
- Outbound: Google ADK streaming events → UiPath conversation message events
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from google.adk.events.event import Event
from google.genai import types
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

_TRANSFER_FN = "transfer_to_agent"


class GoogleADKChatMessagesMapper:
    """Maps between UiPath Conversation format and Google ADK format.

    Maintains state across events to properly track message lifecycle
    and correlate tool calls with their responses.
    """

    def __init__(self) -> None:
        self._current_message_id: str | None = None
        self._message_started: bool = False
        # tool_call_id -> message_id (for correlating responses)
        self._pending_tool_calls: dict[str, str] = {}

    # ── Inbound: UiPath messages → Google ADK Content ────────────────

    def map_messages(self, messages: list[Any]) -> list[types.Content]:
        """Parse UiPath conversation messages and convert to Google ADK Content list.

        Returns a list of Content objects representing the conversation.
        The caller should inject all but the last into the session as history,
        and use the last as new_message for run_async().
        """
        if not isinstance(messages, list) or not messages:
            return []

        first = messages[0]

        # Already Content objects — pass through
        if isinstance(first, types.Content):
            return messages

        # UiPath conversation messages
        if isinstance(first, UiPathConversationMessage):
            return self._map_messages_internal(messages)

        # List of dicts → try parsing as UiPath conversation messages
        if isinstance(first, dict):
            try:
                parsed = [UiPathConversationMessage.model_validate(m) for m in messages]
                return self._map_messages_internal(parsed)
            except ValidationError:
                pass

        # Fallback: treat as simple text messages
        text = self._extract_text_from_raw(messages)
        if text:
            return [types.Content(role="user", parts=[types.Part(text=text)])]

        return []

    def _map_messages_internal(
        self, messages: list[UiPathConversationMessage]
    ) -> list[types.Content]:
        """Convert UiPathConversationMessage list to Google ADK Content list."""
        contents: list[types.Content] = []

        for msg in messages:
            text = self._extract_text_from_content_parts(msg)

            if msg.role == "user":
                if text:
                    contents.append(
                        types.Content(role="user", parts=[types.Part(text=text)])
                    )

            elif msg.role == "assistant":
                parts: list[types.Part] = []

                if text:
                    parts.append(types.Part(text=text))

                # Add function calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tc.name,
                                    args=tc.input or {},
                                    id=tc.tool_call_id,
                                )
                            )
                        )

                if parts:
                    contents.append(types.Content(role="model", parts=parts))

                # Add function responses for completed tool calls
                if msg.tool_calls:
                    response_parts: list[types.Part] = []
                    for tc in msg.tool_calls:
                        if tc.result is not None:
                            output = self._normalize_tool_output(tc.result.output)
                            response_parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tc.name,
                                        response=output,
                                        id=tc.tool_call_id,
                                    )
                                )
                            )

                    if response_parts:
                        contents.append(
                            types.Content(role="user", parts=response_parts)
                        )

        return contents

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

    @staticmethod
    def _normalize_tool_output(output: Any) -> dict[str, Any]:
        """Normalize tool output to a dict for FunctionResponse.response."""
        if output is None:
            return {}
        if isinstance(output, dict):
            return output
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"result": output}
        return {"result": str(output)}

    # ── Outbound: Google ADK events → UiPath conversation events ─────

    def map_event(self, event: Event) -> list[UiPathConversationMessageEvent]:
        """Convert a Google ADK Event to UiPath conversation message events.

        Maps:
        - Partial text → ContentPartChunk (token-by-token streaming)
        - Function calls → ToolCallStart
        - Function responses → ToolCallEnd + MessageEnd (when all done)
        - Final response → MessageEnd
        """
        author = event.author
        if not author or author == "user":
            return []

        events: list[UiPathConversationMessageEvent] = []

        # Function calls → ToolCallStart
        # Skip partial events: with SSE streaming, ADK may emit partial
        # function call events before the complete one.
        function_calls = [
            fc for fc in event.get_function_calls() if fc.name != _TRANSFER_FN
        ]
        if function_calls and not event.partial:
            events.extend(self._ensure_message_started())
            for fc in function_calls:
                tool_call_id = fc.id or f"{fc.name}_{uuid4().hex[:8]}"
                if self._current_message_id:
                    self._pending_tool_calls[tool_call_id] = self._current_message_id
                events.append(
                    self._make_tool_call_start_event(
                        self._current_message_id or "",
                        tool_call_id,
                        fc.name or "unknown",
                        fc.args,
                    )
                )
            return events

        # Function responses → ToolCallEnd
        function_responses = [
            fr for fr in event.get_function_responses() if fr.name != _TRANSFER_FN
        ]
        if function_responses:
            for fr in function_responses:
                fr_tool_call_id = self._resolve_tool_call_id(fr)
                if not fr_tool_call_id:
                    continue
                message_id = self._pending_tool_calls.pop(
                    fr_tool_call_id, self._current_message_id
                )
                if message_id:
                    events.append(
                        self._make_tool_call_end_event(
                            message_id, fr_tool_call_id, fr.response or {}
                        )
                    )

            # If all tool calls for current message are resolved, close it
            if (
                self._message_started
                and self._current_message_id
                and not any(
                    mid == self._current_message_id
                    for mid in self._pending_tool_calls.values()
                )
            ):
                events.append(self._make_message_end_event(self._current_message_id))
                self._message_started = False
                self._current_message_id = None
            return events

        # Partial streaming content → ContentPartChunk
        if event.partial and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if p.text)
            if text:
                events.extend(self._ensure_message_started())
                if self._current_message_id:
                    events.append(
                        self._make_content_chunk_event(self._current_message_id, text)
                    )
            return events

        # Final response → close message
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if p.text)
            if text:
                if not self._message_started:
                    # Non-streaming final response: emit full text
                    events.extend(self._ensure_message_started())
                if self._current_message_id:
                    events.append(
                        self._make_content_chunk_event(self._current_message_id, text)
                    )
            if self._message_started and self._current_message_id:
                events.append(self._make_message_end_event(self._current_message_id))
                self._message_started = False
                self._current_message_id = None
            return events

        return events

    def close_message(self) -> list[UiPathConversationMessageEvent]:
        """Close the current message if open. Safety net for end of stream."""
        if self._message_started and self._current_message_id:
            events = [self._make_message_end_event(self._current_message_id)]
            self._message_started = False
            self._current_message_id = None
            return events
        return []

    def _resolve_tool_call_id(self, fr: types.FunctionResponse) -> str | None:
        """Resolve tool_call_id for a FunctionResponse.

        Prefers id match, falls back to name match.
        """
        # Match by id
        if fr.id and fr.id in self._pending_tool_calls:
            return fr.id

        # Match by name (first pending call with same name)
        if fr.name:
            for tc_id, _ in self._pending_tool_calls.items():
                if tc_id.startswith(fr.name):
                    return tc_id

        # Return id even if not in pending (might be tracked elsewhere)
        return fr.id

    # ── Helpers ───────────────────────────────────────────────────────

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


__all__ = ["GoogleADKChatMessagesMapper"]
