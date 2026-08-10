"""Conversational runtime for Claude Agent SDK agents.

Runs one conversation exchange per invocation: streams the assistant response
as UiPath conversation message events, then suspends with an API resume
trigger. The next user message resumes the runtime; the Claude SDK session id
is persisted so the SDK re-attaches to the same conversation
(``ClaudeAgentOptions(resume=session_id)``), and a stable per-conversation
workspace directory preserves files across exchanges.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
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
from .instrumentor import ClaudeSdkInstrumentor
from .runtime import UiPathClaudeSDKRuntime
from .schema import get_agent_graph, get_entrypoints_schema
from .session_store import ClaudeSessionStore

logger = logging.getLogger(__name__)


class UiPathClaudeSDKConversationalRuntime(UiPathClaudeSDKRuntime):
    """Conversational (multi-turn) runtime for Claude Agent SDK agents.

    Args:
        agent: The loaded ClaudeAgent definition.
        session_store: Persists the Claude session id between exchanges.
        workspace_root: Stable directory reused across exchanges of the same
            conversation.
        runtime_id: Unique identifier for this runtime instance.
        entrypoint: Agent entrypoint name (for schema reporting).
        agenthub_config: AgentHub billing/consumption config header value.
    """

    def __init__(
        self,
        agent: ClaudeAgent,
        session_store: ClaudeSessionStore,
        workspace_root: Path,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        agenthub_config: str = "conversationalagentsruntime",
    ):
        super().__init__(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            agenthub_config=agenthub_config,
        )
        self._session_store = session_store
        self._workspace_root = workspace_root

    # --- Input mapping ------------------------------------------------------

    def _get_user_message(self, input: dict[str, Any]) -> str:
        """Extract the user prompt from UiPath conversation format input.

        On resume, the input is the resume map {interrupt_id: resume_data}
        produced by UiPathResumableRuntime — the resume data carries the next
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
            tool activity, then a SUSPENDED UiPathRuntimeResult carrying a new
            API resume trigger interrupt.
        """
        try:
            mapper = (
                ClaudeSdkInstrumentor(self.span_factory, self.agent.options.model or "")
                if self.span_factory
                else None
            )

            user_message = self._get_user_message(input or {})
            env = await self._build_llm_env()

            workspace = self._workspace_root
            workspace.mkdir(parents=True, exist_ok=True)

            session_id = await self._session_store.get_session_id()
            sdk_options = self._build_sdk_options(
                env=env,
                workspace=workspace,
                resume_session_id=session_id,
            )

            try:
                async with ClaudeSDKClient(options=sdk_options) as client:
                    await client.query(user_message)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            if mapper:
                                mapper.on_assistant_message(message)
                            for event in self._map_conversational_assistant(message):
                                yield event
                        elif isinstance(message, UserMessage):
                            if mapper:
                                mapper.on_user_message(message)
                            for state_event in self._map_tool_results(message):
                                yield state_event
                        elif isinstance(message, ResultMessage):
                            if message.is_error:
                                self._raise_result_error(message)
                            await self._session_store.set_session_id(message.session_id)
                        elif isinstance(message, SystemMessage):
                            mapped = self._map_system(message)
                            if mapped is not None:
                                yield mapped
            finally:
                if mapper:
                    mapper.cleanup()

            # Suspend for the next user message. The plain dict suspend value
            # produces an API resume trigger via UiPathResumableRuntime.
            yield UiPathRuntimeResult(
                output={str(uuid4()): {}},
                status=UiPathRuntimeStatus.SUSPENDED,
            )
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
