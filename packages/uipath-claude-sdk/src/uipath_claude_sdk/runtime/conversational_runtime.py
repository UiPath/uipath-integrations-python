"""Conversational runtime for Claude Agent SDK agents.

Runs one conversation exchange per invocation: streams the assistant response
as UiPath conversation message events, then suspends on nothing, which is how
a chat host reads "the turn is over, waiting for the user". The Claude SDK
session id is persisted so the SDK re-attaches to the same conversation
(``ClaudeAgentOptions(resume=session_id)``), and a stable per-conversation
workspace directory preserves files across exchanges.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
)
from uipath.core.chat.content import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
)
from uipath.core.chat.message import (
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
)
from uipath.runtime import (
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.events import (
    UiPathRuntimeEvent,
    UiPathRuntimeMessageEvent,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from ..agent import ClaudeAgent
from ..interrupts import active_channel
from .runtime import RESUME_PROMPT, RunContext, UiPathClaudeSDKRuntime
from .schema import get_agent_graph, get_entrypoints_schema
from .session_paths import ClaudeSessionPaths
from .session_store import ClaudeSessionStore

logger = logging.getLogger(__name__)


class UiPathClaudeSDKConversationalRuntime(UiPathClaudeSDKRuntime):
    """Conversational (multi-turn) runtime for Claude Agent SDK agents.

    Every exchange ends suspended, so the job survives to serve the next one.
    An exchange the model parked in suspends on that interrupt and leaves the
    exchange open until it resolves; a finished exchange suspends with no
    interrupt at all, which is what closes it.

    Args:
        agent: The loaded ClaudeAgent definition.
        session_store: Persists the Claude session id between exchanges, plus
            any interrupt the conversation is suspended on.
        session_paths: Durable Claude config and working directories. The
            workspace is keyed on the conversation, so files created in one
            exchange are still there in the next.
        runtime_id: Unique identifier for this runtime instance.
        entrypoint: Agent entrypoint name (for schema reporting).
    """

    def __init__(
        self,
        agent: ClaudeAgent,
        session_store: ClaudeSessionStore,
        session_paths: ClaudeSessionPaths,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        super().__init__(
            agent=agent,
            session_store=session_store,
            session_paths=session_paths,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def _session_id_for_run(self, resuming: bool) -> str | None:
        """Re-attach to the conversation's session on every exchange after the first."""
        return await self._session_store.get_session_id()

    def _user_message_for_run(
        self, input: dict[str, Any] | None, context: RunContext
    ) -> str:
        """The message that starts the exchange.

        Every exchange after the first arrives as a resume, so the base
        behaviour of replacing the prompt with a fixed continuation would
        discard what the user just said. A resume resolving a parked interrupt
        is the exception: there the payload reaches the model as that call's
        tool result, and no new instruction belongs in the turn.
        """
        if context.pending is not None:
            return RESUME_PROMPT
        return self._get_user_message(input or {}) or RESUME_PROMPT

    # --- Input mapping ------------------------------------------------------

    def _get_user_message(self, input: dict[str, Any]) -> str:
        """Extract the user prompt from UiPath conversation format input.

        On resume, the input is the resume map {interrupt_id: resume_data}
        produced by UiPathResumableRuntime. The resume data carries the next
        user message in the same conversation format.
        """
        message = self._extract_messages(input)
        if message:
            return message

        for value in input.values():
            if isinstance(value, dict):
                message = self._extract_messages(value)
                if message:
                    return message
            elif isinstance(value, str) and value:
                return value

        return ""

    @classmethod
    def _extract_messages(cls, input: dict[str, Any]) -> str:
        """Extract text from the last user message in UiPath conversation format.

        Expects: [{"role": "user", "contentParts": [{"data": {"inline": "..."}}]}]
        """
        messages = input.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role and role != "user":
                continue
            text = cls._extract_text_from_content_parts(msg)
            if text:
                return text

        if isinstance(messages[-1], dict):
            return cls._extract_text_from_content_parts(messages[-1])

        return ""

    @staticmethod
    def _extract_text_from_content_parts(msg: dict[str, Any]) -> str:
        content_parts = msg.get("contentParts")
        if not isinstance(content_parts, list):
            return ""

        texts: list[str] = []
        for cp in content_parts:
            if not isinstance(cp, dict):
                continue
            data = cp.get("data")
            if isinstance(data, dict) and "inline" in data:
                inline = data["inline"]
                if isinstance(inline, str) and inline:
                    texts.append(inline)
        return "".join(texts)

    # --- Conversation event helpers ------------------------------------------

    @staticmethod
    def _get_timestamp() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def _message_start_events(
        self, message_id: str, content_part_id: str
    ) -> UiPathRuntimeMessageEvent:
        return UiPathRuntimeMessageEvent(
            payload=UiPathConversationMessageEvent(
                message_id=message_id,
                start=UiPathConversationMessageStartEvent(
                    role="assistant",
                    timestamp=self._get_timestamp(),
                ),
                content_part=UiPathConversationContentPartEvent(
                    content_part_id=content_part_id,
                    start=UiPathConversationContentPartStartEvent(
                        mime_type="text/plain",
                    ),
                ),
            ),
        )

    @staticmethod
    def _message_chunk_event(
        message_id: str, content_part_id: str, text: str
    ) -> UiPathRuntimeMessageEvent:
        return UiPathRuntimeMessageEvent(
            payload=UiPathConversationMessageEvent(
                message_id=message_id,
                content_part=UiPathConversationContentPartEvent(
                    content_part_id=content_part_id,
                    chunk=UiPathConversationContentPartChunkEvent(
                        data=text,
                    ),
                ),
            ),
        )

    @staticmethod
    def _message_end_event(
        message_id: str, content_part_id: str
    ) -> UiPathRuntimeMessageEvent:
        return UiPathRuntimeMessageEvent(
            payload=UiPathConversationMessageEvent(
                message_id=message_id,
                end=UiPathConversationMessageEndEvent(),
                content_part=UiPathConversationContentPartEvent(
                    content_part_id=content_part_id,
                    end=UiPathConversationContentPartEndEvent(),
                ),
            ),
        )

    # --- Runtime protocol -----------------------------------------------------

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Run one conversation exchange and suspend for the next user message.

        Yields:
            UiPathRuntimeMessageEvent for assistant text (message start,
            content-part chunks, message end), UiPathRuntimeStateEvent for
            tool activity, then a SUSPENDED UiPathRuntimeResult: carrying the
            interrupt the model parked on, or carrying nothing when the turn
            simply finished.
        """
        try:
            context = await self._open_run(input, options)
            user_message = self._user_message_for_run(input, context)

            result_message: ResultMessage | None = None
            try:
                with active_channel(context.channel):
                    async with ClaudeSDKClient(options=context.sdk_options) as client:
                        await client.query(user_message)
                        async for message in self._receive_run_messages(client):
                            self._adopt_span_parent(context.channel)
                            if isinstance(message, AssistantMessage):
                                for event in self._map_conversational_assistant(
                                    message
                                ):
                                    yield event
                            elif isinstance(message, UserMessage):
                                for state_event in self._map_tool_results(message):
                                    yield state_event
                            elif isinstance(message, ResultMessage):
                                await self._on_result_message(message, context)
                                result_message = message
                            elif isinstance(message, SystemMessage):
                                mapped = self._map_system(message)
                                if mapped is not None:
                                    yield mapped
            finally:
                await self._stop_llm_gateway()
                self._flush_gateway_spans()

            if result_message is not None and result_message.deferred_tool_use:
                yield await self._suspended_result(context)
                return

            self._refuse_ignored_deferral(context)
            await self._clear_resolved_pending(context)
            await self._capture_transcript(await self._session_store.get_session_id())
            yield UiPathRuntimeResult(status=UiPathRuntimeStatus.SUSPENDED)
        except Exception as e:
            raise self.create_runtime_error(e) from e

    def _map_conversational_assistant(
        self, message: AssistantMessage
    ) -> list[UiPathRuntimeEvent]:
        """Map an assistant message to conversation events (text) and state events (tools/thinking)."""
        events: list[UiPathRuntimeEvent] = []

        text_blocks = [b for b in message.content if isinstance(b, TextBlock)]
        if text_blocks:
            message_id = str(uuid4())
            content_part_id = f"chunk-{message_id}-0"
            events.append(self._message_start_events(message_id, content_part_id))
            for block in text_blocks:
                if block.text:
                    events.append(
                        self._message_chunk_event(
                            message_id, content_part_id, block.text
                        )
                    )
            events.append(self._message_end_event(message_id, content_part_id))

        non_text = AssistantMessage(
            content=[b for b in message.content if not isinstance(b, TextBlock)],
            model=message.model,
        )
        events.extend(self._map_assistant(non_text))
        return events

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get the conversational schema (UiPath conversation message format)."""
        entrypoints_schema = get_entrypoints_schema(self.agent, conversational=True)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint or "",
            uniqueId=str(uuid4()),
            type="agent",
            input=entrypoints_schema.get("input", {}),
            output=entrypoints_schema.get("output", {}),
            graph=get_agent_graph(self.agent),
        )
