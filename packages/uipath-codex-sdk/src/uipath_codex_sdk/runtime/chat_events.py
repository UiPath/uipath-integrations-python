"""UiPath conversation (chat) events of a Codex turn, and the chat input.

:class:`CodexChatEventMapper` turns the thread notifications of one exchange
into ``UiPathConversationMessageEvent`` payloads, in the order the chat host
expects (as LangChain emits them):

* an assistant message: ``startMessage`` + ``startContentPart``
  (``text/markdown``), one ``chunk`` per ``item/agentMessage/delta`` (or the
  whole text when nothing streamed), ``endContentPart``;
* each tool call that follows it attaches to that message: ``startToolCall``
  + ``executingToolCall``, then ``endToolCall`` with its result;
* ``endMessage`` once the message's last tool call has ended, or when the
  next message starts, or at the end of the turn.

MCP tool calls (UiPath tools and your own servers) and shell commands are
tool calls in the chat; reasoning, file edits and web searches stay state
events. The mapper's state (the open messages and tool calls) is saved when
an exchange suspends on a durable call, so the resumed execution can end the
call's ``Tool call`` card on the message that started it.

The input helpers read the newest user message of a ``{"messages": [...]}``
input: its text parts, and the files uploaded with it as job attachments.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from uipath.core.chat import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
    UiPathConversationExecutingToolCallEvent,
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
    UiPathConversationToolCallEndEvent,
    UiPathConversationToolCallEvent,
    UiPathConversationToolCallStartEvent,
)
from uipath.runtime.events import UiPathRuntimeMessageEvent

from ..attachments import Attachment

#: Codex writes markdown; LangChain streams the same mime type.
TEXT_MIME_TYPE = "text/markdown"
#: The chat tool name of a shell command (as its ``Tool call`` span is named).
SHELL_TOOL = "shell"
#: Codex item types shown as tool calls in the chat.
CHAT_TOOL_ITEM_TYPES = frozenset({"mcpToolCall", "commandExecution"})
#: Prefix of the URI of a file uploaded in the chat.
ATTACHMENT_URN_PREFIX = "urn:uipath:cas:file:orchestrator:"
#: User text of a message that only carries files.
ATTACHMENT_ONLY_TEXT = "The user attached the file(s) below."
#: Roles of a message the user wrote (an empty role counts as the user's).
USER_ROLES = ("user", "")


def now_iso() -> str:
    """ISO timestamp with milliseconds and ``Z``, as the chat host writes them."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def content_part_id(message_id: str) -> str:
    """The id of a message's single text part (LangChain's convention)."""
    return f"chunk-{message_id}-0"


def _event(message_id: str, **fields: Any) -> UiPathRuntimeMessageEvent:
    return UiPathRuntimeMessageEvent(
        payload=UiPathConversationMessageEvent(message_id=message_id, **fields),
        metadata={"source": "codex"},
    )


def parse_output(value: Any) -> Any:
    """A tool result for ``endToolCall``: JSON text parsed, the rest as is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _mcp_text(result: Any) -> Any:
    """The text of an MCP tool result's first content item, if it has one."""
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
    return result


def tool_call_info(item: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """``(tool name, input, metadata)`` of a tool item, for ``startToolCall``.

    An MCP tool is named by its own name, as its ``Tool call`` span is (the
    server goes in the metadata); a shell command is ``shell`` with its
    ``command`` as the input.
    """
    if item.get("type") == "commandExecution":
        return SHELL_TOOL, {"command": item.get("command")}, {}
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        arguments = parse_output(arguments)
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"arguments": arguments}
    return (
        str(item.get("tool") or "tool"),
        arguments,
        {"server": item.get("server")},
    )


def tool_call_result(item: dict[str, Any]) -> tuple[Any, bool]:
    """``(output, is_error)`` of a completed tool item, for ``endToolCall``."""
    status = str(item.get("status") or "")
    if item.get("type") == "commandExecution":
        code = item.get("exit_code")
        failed = status in ("failed", "declined") or (code is not None and code != 0)
        return item.get("aggregated_output") or "", failed
    error = item.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        return message, True
    return parse_output(_mcp_text(item.get("result"))), status == "failed"


# --------------------------------------------------------------------------
# the mapper
# --------------------------------------------------------------------------
@dataclass
class _Message:
    """An assistant message still open in the chat."""

    id: str
    #: Whether its text part is open (started, not ended).
    text_open: bool = False
    #: Ids of its tool calls that have not ended.
    calls: list[str] = field(default_factory=list)

    def to_stored(self) -> dict[str, Any]:
        return {"id": self.id, "text_open": self.text_open, "calls": self.calls}


class CodexChatEventMapper:
    """Chat events of one exchange, from Codex thread notifications."""

    def __init__(self) -> None:
        self._messages: dict[str, _Message] = {}
        #: agentMessage item id -> message id.
        self._items: dict[str, str] = {}
        #: Tool call id (the item id, also the Responses ``call_id``) -> message id.
        self._calls: dict[str, str] = {}
        #: The message new tool calls attach to.
        self._current: str | None = None
        #: agentMessage items whose text arrived as deltas.
        self._streamed: set[str] = set()

    # --- persistence --------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What a resumed execution needs to end the open tool calls."""
        return {
            "tool_calls": dict(self._calls),
            "open_messages": [m.to_stored() for m in self._messages.values()],
        }

    @classmethod
    def from_state(cls, state: Any) -> CodexChatEventMapper:
        mapper = cls()
        if not isinstance(state, dict):
            return mapper
        for raw in state.get("open_messages") or []:
            if isinstance(raw, dict) and isinstance(raw.get("id"), str):
                calls = [str(c) for c in raw.get("calls") or []]
                mapper._messages[raw["id"]] = _Message(
                    raw["id"], bool(raw.get("text_open")), calls
                )
        for call_id, message_id in (state.get("tool_calls") or {}).items():
            if message_id in mapper._messages:
                mapper._calls[str(call_id)] = message_id
        return mapper

    @property
    def has_open(self) -> bool:
        return bool(self._messages)

    # --- notifications --------------------------------------------------------

    def on_item_started(self, item: dict[str, Any]) -> list[UiPathRuntimeMessageEvent]:
        itype = item.get("type")
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return []
        if itype == "agentMessage":
            events = self._close_current()
            events += self._start_text(item_id)
            return events
        if itype in CHAT_TOOL_ITEM_TYPES and item_id not in self._calls:
            return self._start_call(item_id, *tool_call_info(item))
        return []

    def on_delta(self, item_id: str, delta: str) -> list[UiPathRuntimeMessageEvent]:
        events: list[UiPathRuntimeMessageEvent] = []
        if item_id not in self._items:
            events += self._close_current()
            events += self._start_text(item_id)
        self._streamed.add(item_id)
        if delta:
            events.append(self._chunk(self._items[item_id], delta))
        return events

    def on_item_completed(
        self, item: dict[str, Any]
    ) -> list[UiPathRuntimeMessageEvent]:
        itype = item.get("type")
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return []
        events: list[UiPathRuntimeMessageEvent] = []
        if itype == "agentMessage":
            if item_id not in self._items:
                events += self._close_current()
                events += self._start_text(item_id)
            message_id = self._items[item_id]
            text = item.get("text") or ""
            if item_id not in self._streamed and text:
                events.append(self._chunk(message_id, text))
            message = self._messages.get(message_id)
            if message is not None and message.text_open:
                message.text_open = False
                events.append(
                    _event(
                        message_id,
                        content_part=UiPathConversationContentPartEvent(
                            content_part_id=content_part_id(message_id),
                            end=UiPathConversationContentPartEndEvent(),
                        ),
                    )
                )
            # Stays open: the tool calls that follow attach to it.
            return events
        if itype in CHAT_TOOL_ITEM_TYPES:
            if item_id not in self._calls:
                events += self._start_call(item_id, *tool_call_info(item))
            output, is_error = tool_call_result(item)
            events += self._end_call(item_id, output=output, is_error=is_error)
        return events

    def on_resumed_call(
        self,
        call_id: str,
        tool: str,
        arguments: Any,
        output: Any,
        is_error: bool,
    ) -> list[UiPathRuntimeMessageEvent]:
        """A durable call that finished while the exchange was suspended."""
        events: list[UiPathRuntimeMessageEvent] = []
        if call_id not in self._calls:
            # Its start was not saved (an older state): show it whole.
            args = arguments if isinstance(arguments, dict) else {}
            events += self._start_call(call_id, tool, args, {"server": None})
        events += self._end_call(
            call_id, output=parse_output(output), is_error=is_error
        )
        return events

    def cancel_calls(self, keep: Iterable[str] = ()) -> list[UiPathRuntimeMessageEvent]:
        """End every open tool call but ``keep`` as cancelled."""
        kept = set(keep)
        events: list[UiPathRuntimeMessageEvent] = []
        for call_id in [c for c in self._calls if c not in kept]:
            events += self._end_call(call_id, cancelled=True)
        return events

    def close_turn(self) -> list[UiPathRuntimeMessageEvent]:
        """End everything still open: the turn is over."""
        events = self.cancel_calls()
        for message_id in list(self._messages):
            events += self._end_message(message_id)
        self._current = None
        return events

    # --- builders -------------------------------------------------------------

    def _new_message(self, *, text: bool) -> tuple[str, UiPathRuntimeMessageEvent]:
        message_id = str(uuid.uuid4())
        self._messages[message_id] = _Message(message_id, text_open=text)
        self._current = message_id
        start = UiPathConversationMessageStartEvent(
            role="assistant", timestamp=now_iso()
        )
        if not text:
            return message_id, _event(message_id, start=start)
        return message_id, _event(
            message_id,
            start=start,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=content_part_id(message_id),
                start=UiPathConversationContentPartStartEvent(mime_type=TEXT_MIME_TYPE),
            ),
        )

    def _start_text(self, item_id: str) -> list[UiPathRuntimeMessageEvent]:
        message_id, event = self._new_message(text=True)
        self._items[item_id] = message_id
        return [event]

    @staticmethod
    def _chunk(message_id: str, text: str) -> UiPathRuntimeMessageEvent:
        return _event(
            message_id,
            content_part=UiPathConversationContentPartEvent(
                content_part_id=content_part_id(message_id),
                chunk=UiPathConversationContentPartChunkEvent(data=text),
            ),
        )

    def _start_call(
        self,
        call_id: str,
        tool: str,
        tool_input: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[UiPathRuntimeMessageEvent]:
        events: list[UiPathRuntimeMessageEvent] = []
        message = self._messages.get(self._current) if self._current else None
        if message is None or message.text_open:
            events += self._close_current()
            message_id, start = self._new_message(text=False)
            events.append(start)
            message = self._messages[message_id]
        message.calls.append(call_id)
        self._calls[call_id] = message.id
        meta = {k: v for k, v in metadata.items() if v is not None}
        events.append(
            _event(
                message.id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=call_id,
                    start=UiPathConversationToolCallStartEvent(
                        tool_name=tool,
                        timestamp=now_iso(),
                        input=tool_input,
                        metadata=meta or None,
                    ),
                ),
            )
        )
        # No confirmation is asked for: the call runs straight away.
        events.append(
            _event(
                message.id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=call_id,
                    executing=UiPathConversationExecutingToolCallEvent(
                        timestamp=now_iso(), input=tool_input
                    ),
                ),
            )
        )
        return events

    def _end_call(
        self,
        call_id: str,
        *,
        output: Any = None,
        is_error: bool = False,
        cancelled: bool = False,
    ) -> list[UiPathRuntimeMessageEvent]:
        message_id = self._calls.pop(call_id, None)
        if message_id is None:
            return []
        events = [
            _event(
                message_id,
                tool_call=UiPathConversationToolCallEvent(
                    tool_call_id=call_id,
                    end=UiPathConversationToolCallEndEvent(
                        timestamp=now_iso(),
                        output=output,
                        is_error=is_error,
                        cancelled=True if cancelled else None,
                    ),
                ),
            )
        ]
        message = self._messages.get(message_id)
        if message is not None:
            if call_id in message.calls:
                message.calls.remove(call_id)
            if not message.calls and not message.text_open:
                events += self._end_message(message_id)
        return events

    def _close_current(self) -> list[UiPathRuntimeMessageEvent]:
        """End the current message unless tool calls still run on it."""
        message = self._messages.get(self._current) if self._current else None
        self._current = None
        if message is None or message.calls:
            return []
        return self._end_message(message.id)

    def _end_message(self, message_id: str) -> list[UiPathRuntimeMessageEvent]:
        message = self._messages.pop(message_id, None)
        if message is None:
            return []
        if self._current == message_id:
            self._current = None
        events: list[UiPathRuntimeMessageEvent] = []
        for call_id in list(message.calls):
            self._calls.pop(call_id, None)
        if message.text_open:
            events.append(
                _event(
                    message_id,
                    content_part=UiPathConversationContentPartEvent(
                        content_part_id=content_part_id(message_id),
                        end=UiPathConversationContentPartEndEvent(),
                    ),
                )
            )
        events.append(_event(message_id, end=UiPathConversationMessageEndEvent()))
        return events


# --------------------------------------------------------------------------
# the chat input
# --------------------------------------------------------------------------
def _messages_of(input: Any) -> list[Any] | None:
    """The top-level ``messages`` list of an input, or None.

    A resume map's values are not looked into: a durable call's job output
    may hold chat-shaped messages (a conversational agent run as a process
    does), and is still that call's result, not a user message. Codex makes
    no API trigger, so the chat host never resumes a run with an
    ``{interrupt_id: {"messages": [...]}}`` map.
    """
    if not isinstance(input, dict):
        return None
    messages = input.get("messages")
    return messages if isinstance(messages, list) else None


def is_chat_input(input: Any) -> bool:
    """Whether ``input`` carries chat messages (a user's new message)."""
    return _messages_of(input) is not None


def _dump(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(by_alias=True, mode="json")
        except Exception:
            return None
        return dumped if isinstance(dumped, dict) else None
    return None


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("contentParts")
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def extract_user_message(input: Any) -> dict[str, Any] | None:
    """The newest message the user wrote (text or files), camelCase, or None.

    Only that message goes to the model: everything earlier is already in
    the Codex thread, so it does not matter whether the host replays the
    history or sends only the new message.
    """
    messages = _messages_of(input)
    for raw in reversed(messages or []):
        message = _dump(raw)
        if message is None:
            continue
        if str(message.get("role") or "") not in USER_ROLES:
            continue
        if user_text(message) or chat_attachments(message):
            return message
    return None


def user_text(message: dict[str, Any]) -> str:
    """The inline text parts of a message, joined."""
    texts: list[str] = []
    for part in _parts(message):
        mime = str(part.get("mimeType") or "text/plain")
        data = part.get("data")
        if not mime.startswith("text/") or not isinstance(data, dict):
            continue
        inline = data.get("inline")
        if isinstance(inline, str) and inline:
            texts.append(inline)
    return "\n".join(texts)


def chat_attachments(message: dict[str, Any]) -> list[Attachment]:
    """The files uploaded with a message, as job attachments.

    A file part's ``data.uri`` is ``urn:uipath:cas:file:orchestrator:<id>``;
    the id is the Orchestrator attachment's, as LangChain reads it.
    """
    found: list[Attachment] = []
    for part in _parts(message):
        data = part.get("data")
        uri = data.get("uri") if isinstance(data, dict) else None
        if not isinstance(uri, str) or not uri.startswith(ATTACHMENT_URN_PREFIX):
            continue
        try:
            attachment_id = uuid.UUID(uri.rsplit(":", 1)[-1])
        except ValueError:
            continue
        found.append(
            Attachment(
                id=attachment_id,
                full_name=str(part.get("name") or attachment_id),
                mime_type=str(part.get("mimeType") or "application/octet-stream"),
            )
        )
    return found


def message_fingerprint(message: dict[str, Any]) -> str:
    """A message's identity: its ``messageId``, else a hash of its parts."""
    message_id = message.get("messageId")
    if isinstance(message_id, str) and message_id:
        return message_id
    blob = json.dumps(_parts(message), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


__all__ = [
    "ATTACHMENT_ONLY_TEXT",
    "CodexChatEventMapper",
    "TEXT_MIME_TYPE",
    "chat_attachments",
    "content_part_id",
    "extract_user_message",
    "is_chat_input",
    "message_fingerprint",
    "now_iso",
    "parse_output",
    "tool_call_info",
    "tool_call_result",
    "user_text",
]
