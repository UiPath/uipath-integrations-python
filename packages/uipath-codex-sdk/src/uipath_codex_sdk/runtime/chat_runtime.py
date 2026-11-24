"""Conversational (chat) runtime: one chat exchange per run.

A conversation is one Codex thread. Each run takes the newest user message
from the ``messages`` input, resumes the thread (restoring its rollout from
the state database first, as a durable resume does), runs one turn with that
message, and streams the reply as UiPath conversation message events (see
:mod:`.chat_events`). The rollout is then checkpointed and the run ends
**SUSPENDED with no trigger**: the job stays alive for the next message, and
the chat host (``UiPathChatRuntime``) reads "no trigger" as the end of the
exchange and sends ``endExchange``.

Which case a run is in is read from the stored state, never from the resume
flag: ``uipath dev`` runs every exchange as a fresh execution (``resume`` is
False, same runtime id), Orchestrator as a job resume.

* the exchange is parked (it suspended on durable calls, and the chat state
  says so): the run is the durable resume (results injected, the model
  continues). A new user message arriving meanwhile is not processed: the
  run suspends again on the same calls;
* the conversation has a thread: the next exchange (whatever an exchange
  that faulted left behind, such as the pending call of a job it started,
  is dropped);
* nothing is stored: the first exchange.

The chat events of a parked exchange's open message belong to that
exchange: the chat host sends every event under the exchange it serves. A
resume that runs for another exchange (a new message met calls that had
all finished, or the run serves another exchange id than the one that
parked) ends the old message silently; the parked exchange's tool card
stays open in the chat, as nothing can close it from another exchange.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from uipath.runtime import (
    UiPathRuntimeEvent,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath.runtime.schema import UiPathRuntimeSchema

from ..attachments import AttachmentStore, attachments_block
from ..tool_results import ToolResultStore
from ..tools import UiPathDurableTool
from .chat_events import (
    ATTACHMENT_ONLY_TEXT,
    CodexChatEventMapper,
    chat_attachments,
    extract_user_message,
    message_fingerprint,
    user_text,
)
from .errors import UiPathCodexErrorCode, UiPathCodexRuntimeError
from .runtime import (
    CONTINUE_PROMPT,
    UiPathCodexRuntime,
    _ItemTracker,
    _ResumePlan,
)
from .schema import get_entrypoint_schema
from .session_store import (
    STATE_STARTED,
    CodexSessionStore,
    EntrypointRecord,
    PendingCall,
)
from .tool_host import DurableToolHost

logger = logging.getLogger(__name__)

#: Key of the chat state that holds the exchange's own user message.
USER_MESSAGE_KEY = "user_message"
#: Key of the chat state that says the exchange is parked on durable calls.
PARKED_KEY = "parked"
#: Key of the chat state that holds the id of the exchange that parked.
EXCHANGE_KEY = "exchange_id"
#: Durable call results that count as a failed tool call in the chat.
_ERROR_STATUSES = frozenset({"faulted", "error", "failed"})


class UiPathCodexChatRuntime(UiPathCodexRuntime):
    """Runs a :class:`UiPathCodexAgent` as a conversational agent.

    Selected by the factory when ``uipath.json`` sets
    ``runtimeOptions.isConversational`` (or the chat host passes a
    conversation id). The agent's ``base_instructions`` are its system
    prompt; ``prompt``, ``input_schema`` and ``output_schema`` do not apply
    (the platform feeds the messages, the reply goes over the chat bridge)
    and are refused.
    """

    _conversational = True

    def __init__(self, *, exchange_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        refused = [
            name
            for name in ("prompt", "input_schema", "output_schema")
            if getattr(self.agent, name) is not None
        ]
        if refused:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.CONFIG_INVALID,
                "Invalid conversational agent",
                "A conversational agent takes its input from the chat and "
                "answers in it, so it cannot set "
                + ", ".join(refused)
                + ". Remove them (use base_instructions for the system prompt), "
                "or turn runtimeOptions.isConversational off in uipath.json.",
                UiPathErrorCategory.USER,
            )
        self._mapper = CodexChatEventMapper()
        #: Fingerprint of the user message this exchange answers.
        self._user_message: str | None = None
        #: The chat host's exchange this run serves (None outside a chat job).
        self._exchange_id = exchange_id
        #: Whether the exchange is parked on durable calls (saved in the state).
        self._parked = False
        #: The exchange that parked (its open message's events belong to it).
        self._parked_exchange: str | None = None

    async def get_schema(self) -> UiPathRuntimeSchema:
        return get_entrypoint_schema(self.agent, self.entrypoint, conversational=True)

    # --- the exchange ---------------------------------------------------------

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Run one chat exchange (or resume one parked on durable calls)."""
        self._refuse_breakpoints(options)
        tools = self._durable_tools()
        session = self._require_session()
        await self._check_entrypoint(session)
        stored = await session.get_chat()
        pending = {
            cid: call
            for cid, call in (await session.get_pending()).items()
            if call.state == STATE_STARTED
        }
        # Only an exchange that suspended is parked: an exchange that faulted
        # may leave the pending call of a job it started, which no execution
        # will resume.
        parked = chat_parked(stored) and bool(
            pending or await session.get_completed() or await session.get_dangling()
        )
        message = extract_user_message(input)
        plan: _ResumePlan | None
        if parked:
            self._mapper = CodexChatEventMapper.from_state(stored)
            self._user_message = stored.get(USER_MESSAGE_KEY)
            self._parked = True
            self._parked_exchange = stored.get(EXCHANGE_KEY)
            new_message = self._is_new_message(input, message)
            if pending and new_message:
                logger.warning(
                    "A user message arrived while the exchange waits on durable "
                    "tool call(s) %s. It is not processed: the exchange keeps "
                    "waiting on the call(s).",
                    sorted(pending),
                )
                yield self._suspended_result(pending)
                return
            # The open message's events go to the exchange that parked only.
            elsewhere = new_message or self._serves_another_exchange()
            if elsewhere:
                logger.info(
                    "Resuming the exchange parked by %s in exchange %s: its "
                    "open message ends without chat events.",
                    self._parked_exchange or "an earlier exchange",
                    self._exchange_id or "of a new message",
                )
            self._resumed_events = []
            self._attachments = await self._load_attachments()
            self._results = await self._load_results()
            prepared = await self._prepare_resume(input)
            for event in self._resumed_events:
                yield event
                chat_events = self._resumed_chat_events(event)
                if not elsewhere:
                    for chat_event in chat_events:
                        yield chat_event
            self._resumed_events = []
            if isinstance(prepared, UiPathRuntimeResult):
                await self._save_chat_state()
                yield prepared
                return
            plan = prepared
            prompt = CONTINUE_PROMPT
            if elsewhere:
                # Its open message is not this exchange's to end.
                self._mapper = CodexChatEventMapper()
            if new_message:
                # Every call had finished: the results go in, then the message.
                prompt = self._chat_prompt(input)
                self._user_message = self._fingerprint(input, message)
        else:
            prompt, plan = await self._start_exchange(session, input, message)
        # Until it suspends again: a run that faults leaves nothing parked.
        self._parked = False
        self._parked_exchange = None
        await self._save_attachments(force=True)
        await self._save_results(force=True)
        await self._save_chat_state()
        async for event in self._run(prompt, tools, plan):
            yield event

    async def _start_exchange(
        self,
        session: CodexSessionStore,
        input: Any,
        message: dict[str, Any] | None,
    ) -> tuple[str, _ResumePlan | None]:
        """The prompt and thread of a new exchange (the first, or a next one)."""
        if self._fingerprint(input, message) is None:
            raise self._no_message_error()
        thread = await session.get_thread()
        rollout = await session.get_rollout()
        # A trigger left over from an earlier suspension would keep the
        # exchange from ending (the host waits on it).
        save_triggers = getattr(self.storage, "save_triggers", None)
        plan: _ResumePlan | None = None
        if thread is not None and rollout is not None:
            # What an exchange that faulted left (the pending call of a job
            # it started...): nothing resumes it, and a later suspension would
            # take it for one of its own calls.
            await session.clear_suspension()
            self._attachments = await self._load_attachments()
            self._results = await self._load_results()
            plan = _ResumePlan(thread_id=thread.thread_id, items=[])
            logger.info(
                "Chat exchange on Codex thread %s (conversation %s)",
                thread.thread_id,
                self.runtime_id,
            )
        else:
            if thread is not None:
                logger.warning(
                    "Codex thread %s has no saved conversation (its first "
                    "exchange did not finish); starting a new thread.",
                    thread.thread_id,
                )
            await session.clear_all()
            await session.set_entrypoint(EntrypointRecord(self.entrypoint))
            self._attachments = AttachmentStore()
            self._results = ToolResultStore()
        if save_triggers is not None:
            await save_triggers(self.runtime_id, [])
        self._mapper = CodexChatEventMapper()
        prompt = self._chat_prompt(input)
        self._user_message = self._fingerprint(input, message)
        return prompt, plan

    async def _check_entrypoint(self, session: CodexSessionStore) -> None:
        record = await session.get_entrypoint()
        if record is not None and record.entrypoint != self.entrypoint:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.ENTRYPOINT_MISMATCH,
                "Resumed on a different entrypoint",
                f"The conversation started on '{record.entrypoint}' and cannot "
                f"continue on '{self.entrypoint}'.",
                UiPathErrorCategory.USER,
            )

    # --- the user message -------------------------------------------------------

    @staticmethod
    def _fingerprint(input: Any, message: dict[str, Any] | None) -> str | None:
        if message is not None:
            return message_fingerprint(message)
        if isinstance(input, str) and input.strip():
            return message_fingerprint(
                {"contentParts": [{"data": {"inline": input.strip()}}]}
            )
        return None

    def _serves_another_exchange(self) -> bool:
        """Whether this run serves another exchange than the one that parked."""
        return (
            self._exchange_id is not None
            and self._parked_exchange is not None
            and self._exchange_id != self._parked_exchange
        )

    def _is_new_message(self, input: Any, message: dict[str, Any] | None) -> bool:
        """Whether ``input`` is a user message other than this exchange's own.

        A job resumed for a durable call may be handed the input it started
        with again; that message was answered already and is no new one.
        """
        fingerprint = self._fingerprint(input, message)
        return fingerprint is not None and fingerprint != self._user_message

    def _chat_prompt(self, input: Any) -> str:
        """The turn's user message: the newest message's text and files.

        Files uploaded in the chat become job attachments the run knows (for
        attachment tools, in this exchange and the next ones), listed in the
        prompt as the input attachments of a non-chat run are.
        """
        message = extract_user_message(input)
        if message is None:
            if isinstance(input, str) and input.strip():
                return input.strip()
            raise self._no_message_error()
        text = user_text(message)
        attachments = chat_attachments(message)
        if not attachments:
            return text
        for attachment in attachments:
            self._attachments.register(attachment)
        block = attachments_block(attachments)
        return f"{text or ATTACHMENT_ONLY_TEXT}\n\n{block}"

    @staticmethod
    def _no_message_error() -> UiPathCodexRuntimeError:
        return UiPathCodexRuntimeError(
            UiPathCodexErrorCode.INPUT_INVALID,
            "No user message",
            "A conversational agent's input is "
            '{"messages": [{"role": "user", "contentParts": [...]}]} with '
            "text or a file in the newest user message.",
            UiPathErrorCategory.USER,
        )

    async def _resume_values(
        self, input: Any, pending: dict[str, PendingCall]
    ) -> dict[str, Any]:
        """As the base, but a chat message answers no durable call."""
        if extract_user_message(input) is not None or isinstance(input, str):
            ordered = sorted(pending.values(), key=lambda c: c.order)
            return await self._read_pending_jobs(ordered, stored_too=True)
        return await super()._resume_values(input, pending)

    # --- events -----------------------------------------------------------------

    def _item_started_events(self, item: dict[str, Any]) -> list[UiPathRuntimeEvent]:
        return list(self._mapper.on_item_started(item))

    def _item_completed_events(self, item: dict[str, Any]) -> list[UiPathRuntimeEvent]:
        events: list[UiPathRuntimeEvent] = list(self._mapper.on_item_completed(item))
        if item.get("type") != "agentMessage":
            # Tools, reasoning, file edits...: the state events of a non-chat
            # run too (the chat host passes them on to the debugger).
            events.extend(super()._item_completed_events(item))
        return events

    def _notification_events(self, payload: Any) -> list[UiPathRuntimeEvent]:
        from openai_codex.generated.v2_all import AgentMessageDeltaNotification

        if isinstance(payload, AgentMessageDeltaNotification):
            return list(self._mapper.on_delta(payload.item_id, payload.delta))
        return []

    def _resumed_chat_events(
        self, event: UiPathRuntimeEvent
    ) -> list[UiPathRuntimeEvent]:
        """``endToolCall`` of a durable call that finished while suspended."""
        metadata = getattr(event, "metadata", None) or {}
        payload = getattr(event, "payload", None)
        if metadata.get("item_type") != "durableToolResult" or not isinstance(
            payload, dict
        ):
            return []
        return list(
            self._mapper.on_resumed_call(
                str(metadata.get("call_id")),
                str(payload.get("tool")),
                payload.get("arguments"),
                payload.get("result"),
                str(payload.get("status")) in _ERROR_STATUSES,
            )
        )

    async def _completed_events(
        self,
        codex_home: str,
        thread_id: str | None,
        tools: dict[str, UiPathDurableTool],
        items: list[dict[str, Any]],
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """End the exchange: checkpoint the thread, then SUSPENDED on nothing."""
        session = self._require_session()
        if thread_id is not None:
            await self._checkpoint_rollout(codex_home, thread_id)
        await self._save_attachments()
        await self._save_results()
        await session.clear_suspension()
        for event in self._mapper.close_turn():
            yield event
        await self._save_chat_state()
        logger.info(
            "Chat exchange done on Codex thread %s (conversation %s)",
            thread_id,
            self.runtime_id,
        )
        yield UiPathRuntimeResult(status=UiPathRuntimeStatus.SUSPENDED)

    async def _suspend_events(
        self,
        codex_home: str,
        thread_id: str,
        tracker: _ItemTracker,
        host: DurableToolHost | None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Park the exchange on its durable calls; their cards stay open."""
        result = await self._finish_suspend(codex_home, thread_id, tracker, host)
        held = set(result.output) if isinstance(result.output, dict) else set()
        # Calls cut off by the suspension never report back.
        for event in self._mapper.cancel_calls(keep=held):
            yield event
        self._parked = True
        self._parked_exchange = self._exchange_id
        await self._save_chat_state()
        yield result

    async def _save_chat_state(self) -> None:
        await self._require_session().set_chat(
            {
                **self._mapper.state(),
                USER_MESSAGE_KEY: self._user_message,
                PARKED_KEY: self._parked,
                EXCHANGE_KEY: self._parked_exchange,
            }
        )


def chat_parked(state: Any) -> bool:
    """Whether a stored chat state is of an exchange parked on durable calls."""
    return isinstance(state, dict) and state.get(PARKED_KEY) is True


__all__ = ["UiPathCodexChatRuntime", "chat_parked"]
