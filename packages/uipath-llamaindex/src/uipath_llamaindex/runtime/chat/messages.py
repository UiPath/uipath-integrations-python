import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)
from uipath.core.chat import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
    UiPathConversationToolCallEndEvent,
    UiPathConversationToolCallEvent,
    UiPathConversationToolCallStartEvent,
)

from uipath_llamaindex.runtime.storage import SqliteResumableStorage

logger = logging.getLogger(__name__)

STORAGE_NAMESPACE_EVENT_MAPPER = "chat-event-mapper"
STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP = "tool_id_map"


class UiPathChatMessagesMapper:
    """Stateful mapper that converts LlamaIndex agent events to UiPath message events.

    Maintains state across events to properly track:
    - The current AI message ID (generated per agent turn, since LlamaIndex doesn't provide one)
    - Pending tool calls per message ID for correct message_end timing

    When a storage backend is provided, the tool_id → message_id mapping is persisted
    so that it survives workflow suspension and can be correctly resolved on resume.
    """

    def __init__(
        self,
        runtime_id: str,
        storage: SqliteResumableStorage | None = None,
    ) -> None:
        self.runtime_id = runtime_id
        self.storage = storage
        self._current_message_id: str | None = None
        self._storage_lock = asyncio.Lock()
        # In-memory fallback state used when no storage is provided
        self._pending_tool_calls: dict[str, set[str]] = {}
        self._tool_id_to_message_id: dict[str, str] = {}

    @staticmethod
    def map_input(input: dict[str, Any]) -> dict[str, Any]:
        """Map UiPath chat message format to LlamaIndex expected format.

        If the input already has 'user_msg', return it unchanged.
        If the input has a 'messages' array (UiPath chat format), extract the
        text content from the first message's contentParts and map it to
        'user_msg'.
        """
        if "user_msg" in input:
            return input

        messages = input.get("messages")
        if not messages or not isinstance(messages, list):
            return input

        first_message = messages[0]
        content_parts = first_message.get("contentParts", [])

        text_parts = [
            part["data"]["inline"]
            for part in content_parts
            if part.get("mimeType") == "text/plain"
            and isinstance(part.get("data"), dict)
            and part["data"].get("inline")
        ]

        if text_parts:
            return {"user_msg": " ".join(text_parts)}

        return input

    def get_timestamp(self) -> str:
        """Format current time as ISO 8601 UTC with milliseconds: 2025-01-04T10:30:00.123Z"""
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def get_content_part_id(self, message_id: str) -> str:
        return f"chunk-{message_id}-0"

    async def map_event(
        self,
        event: AgentStream | AgentOutput | ToolCall | ToolCallResult,
    ) -> list[UiPathConversationMessageEvent] | None:
        """Convert a LlamaIndex agent event into UiPath conversation message events.

        Returns a list of events to emit, or None if the event should be skipped.
        """
        if isinstance(event, AgentStream):
            return self._map_agent_stream(event)

        if isinstance(event, AgentOutput):
            return await self._map_agent_output(event)

        # ToolCall start is handled via AgentOutput to have the message_id available
        if isinstance(event, ToolCall):
            return None

        if isinstance(event, ToolCallResult):
            return await self._map_tool_call_result(event)

        return None

    def _map_agent_stream(
        self, event: AgentStream
    ) -> list[UiPathConversationMessageEvent] | None:
        events: list[UiPathConversationMessageEvent] = []

        # First stream chunk of a new agent turn: generate a fresh message ID
        if self._current_message_id is None:
            self._current_message_id = str(uuid4())
            events.append(self._create_message_start_event(self._current_message_id))

        if event.delta:
            events.append(
                self._create_content_chunk_event(self._current_message_id, event.delta)
            )

        return events if events else None

    async def _map_agent_output(
        self, event: AgentOutput
    ) -> list[UiPathConversationMessageEvent] | None:
        message_id = self._current_message_id
        # Reset for the next turn regardless of outcome
        self._current_message_id = None

        if message_id is None:
            return None

        events: list[UiPathConversationMessageEvent] = []

        if event.tool_calls:
            if self.storage is not None:
                async with self._storage_lock:
                    existing: dict[str, str] | None = await self.storage.get_value(
                        self.runtime_id,
                        STORAGE_NAMESPACE_EVENT_MAPPER,
                        STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
                    )
                    tool_id_to_message_id: dict[str, str] = existing or {}

                    for tool_call in event.tool_calls:
                        tool_id_to_message_id[tool_call.tool_id] = message_id
                        events.append(
                            self._create_tool_call_start_event(
                                message_id=message_id,
                                tool_call_id=tool_call.tool_id,
                                tool_name=tool_call.tool_name,
                                input=tool_call.tool_kwargs,
                            )
                        )

                    await self.storage.set_value(
                        self.runtime_id,
                        STORAGE_NAMESPACE_EVENT_MAPPER,
                        STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
                        tool_id_to_message_id,
                    )
            else:
                # In-memory fallback (no suspend/resume support)
                pending: set[str] = set()
                for tool_call in event.tool_calls:
                    self._tool_id_to_message_id[tool_call.tool_id] = message_id
                    pending.add(tool_call.tool_id)
                    events.append(
                        self._create_tool_call_start_event(
                            message_id=message_id,
                            tool_call_id=tool_call.tool_id,
                            tool_name=tool_call.tool_name,
                            input=tool_call.tool_kwargs,
                        )
                    )
                self._pending_tool_calls[message_id] = pending
            # message_end will be emitted once the last ToolCallResult comes in
        else:
            # No tool calls: this is the final text response, close the message now
            events.append(self._create_message_end_event(message_id))

        return events if events else None

    async def _map_tool_call_result(
        self, event: ToolCallResult
    ) -> list[UiPathConversationMessageEvent] | None:
        message_id, is_last = await self._get_message_id_for_tool_call(event.tool_id)
        if message_id is None:
            logger.warning(
                "ToolCallResult received for unknown tool_id '%s' — skipping.",
                event.tool_id,
            )
            return None

        output = event.tool_output.content if event.tool_output else None

        events: list[UiPathConversationMessageEvent] = [
            self._create_tool_call_end_event(
                message_id=message_id,
                tool_call_id=event.tool_id,
                output=output,
            )
        ]

        # Close the message once all tool calls for it have completed
        if is_last:
            events.append(self._create_message_end_event(message_id))

        return events

    async def _get_message_id_for_tool_call(
        self, tool_id: str
    ) -> tuple[str | None, bool]:
        """Look up the message_id for a tool_id and remove it from the map.

        Returns (message_id, is_last) where is_last is True when no other
        pending tool calls remain for the same message.
        """
        if self.storage is not None:
            async with self._storage_lock:
                tool_id_to_message_id: (
                    dict[str, str] | None
                ) = await self.storage.get_value(
                    self.runtime_id,
                    STORAGE_NAMESPACE_EVENT_MAPPER,
                    STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
                )

                if tool_id_to_message_id is None:
                    logger.error(
                        "attempt to lookup tool_id %s when no map present in storage",
                        tool_id,
                    )
                    return None, False

                message_id = tool_id_to_message_id.get(tool_id)
                if message_id is None:
                    logger.error(
                        "tool_id to message map does not contain tool_id %s",
                        tool_id,
                    )
                    return None, False

                del tool_id_to_message_id[tool_id]

                await self.storage.set_value(
                    self.runtime_id,
                    STORAGE_NAMESPACE_EVENT_MAPPER,
                    STORAGE_KEY_TOOL_ID_TO_MESSAGE_ID_MAP,
                    tool_id_to_message_id,
                )

                is_last = message_id not in tool_id_to_message_id.values()

            return message_id, is_last

        # In-memory fallback
        message_id = self._tool_id_to_message_id.pop(tool_id, None)
        if message_id is None:
            return None, False

        pending = self._pending_tool_calls.get(message_id)
        if pending is not None:
            pending.discard(tool_id)
            if not pending:
                del self._pending_tool_calls[message_id]
                return message_id, True

        return message_id, False

    # ── Factory helpers ────────────────────────────────────────────────────────

    def _create_message_start_event(
        self, message_id: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            start=UiPathConversationMessageStartEvent(
                role="assistant", timestamp=self.get_timestamp()
            ),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                start=UiPathConversationContentPartStartEvent(mime_type="text/plain"),
            ),
        )

    def _create_content_chunk_event(
        self, message_id: str, text: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                chunk=UiPathConversationContentPartChunkEvent(data=text),
            ),
        )

    def _create_message_end_event(
        self, message_id: str
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            end=UiPathConversationMessageEndEvent(),
            content_part=UiPathConversationContentPartEvent(
                content_part_id=self.get_content_part_id(message_id),
                end=UiPathConversationContentPartEndEvent(),
            ),
        )

    def _create_tool_call_start_event(
        self, message_id: str, tool_call_id: str, tool_name: str, input: dict[str, Any]
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            tool_call=UiPathConversationToolCallEvent(
                tool_call_id=tool_call_id,
                start=UiPathConversationToolCallStartEvent(
                    tool_name=tool_name,
                    timestamp=self.get_timestamp(),
                    input=input,
                ),
            ),
        )

    def _create_tool_call_end_event(
        self, message_id: str, tool_call_id: str, output: str | None
    ) -> UiPathConversationMessageEvent:
        return UiPathConversationMessageEvent(
            message_id=message_id,
            tool_call=UiPathConversationToolCallEvent(
                tool_call_id=tool_call_id,
                end=UiPathConversationToolCallEndEvent(
                    timestamp=self.get_timestamp(),
                    output=output,
                ),
            ),
        )


__all__ = ["UiPathChatMessagesMapper"]
