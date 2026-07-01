"""LlamaIndex governance event handler for UiPath.

Provides governance for LlamaIndex agents/workflows. Unlike the ADK / OpenAI /
Agent-Framework integrations — which install per-agent callbacks or middleware —
LlamaIndex routes everything (LLM calls, tool calls) through its global
**instrumentation dispatcher** (the same mechanism the package already uses for
OpenInference tracing). So this adapter governs by registering a
:class:`GovernanceEventHandler` on the **root dispatcher**, which receives every
event propagated from child dispatchers:

- ``LLMChatStartEvent``   → BEFORE_MODEL  (scans the latest input message)
- ``LLMChatEndEvent``     → AFTER_MODEL   (scans the response)
- ``AgentToolCallEvent``  → TOOL_CALL     (tool name + arguments)

The dispatcher is process-global, so registration is process-wide — which fits
the coded-agent model (one workflow per process). :func:`install_governance`
therefore returns the ``agent`` unchanged (nothing is mutated on it); the wiring
lives on the dispatcher. A second install (a reused process serving a new
runtime) **rebinds** that one handler to the new run's evaluator / session
rather than silently ignoring it — the most-recent install governs.
:func:`uninstall_governance` removes the handler so the global dispatcher does
not retain the evaluator after the runtime is gone; the factory calls it on
dispose.

Because the dispatcher is process-global and LlamaIndex events do not carry a
stable per-run identity, this adapter does not isolate two *concurrently*
executing runtimes in the same process — they would share the latest-installed
evaluator. That is a property of LlamaIndex's global instrumentation and matches
the one-workflow-per-process runtime model.

LlamaIndex does **not** emit a tool-*end* instrumentation event, so AFTER_TOOL
is not wired here; a tool's result is governed at the next ``LLMChatStartEvent``
where it is fed back to the model as input (analogous to how the OpenAI adapter
handles its missing tool-args).

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host and are intentionally not fired here.

The evaluator protocol comes from ``uipath-core``; this package contributes
only the LlamaIndex-specific wiring. Governance is installed by the runtime
factory: passing an ``evaluator`` to ``new_runtime`` calls
:func:`install_governance`, which registers the handler on the dispatcher. No
adapter registry, no entry point, no import-time side effects.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator. The handler only extracts payloads and calls
the matching ``evaluate_*`` method; :class:`GovernanceBlockException` propagates
(aborting the run), anything else is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from llama_index.core.instrumentation import (  # type: ignore[attr-defined]
    get_dispatcher,
)
from llama_index.core.instrumentation.event_handlers.base import (  # type: ignore[attr-defined]
    BaseEventHandler,
)
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from pydantic import PrivateAttr
from uipath.core.adapters import EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters.
_BEFORE_MODEL_TEXT_CAP = 64000


def install_governance(
    agent: Any,
    evaluator: EvaluatorProtocol,
    *,
    agent_name: str,
    session_id: str,
) -> Any:
    """Register the governance event handler on the root dispatcher.

    Returns the ``agent`` unchanged — LlamaIndex governance is wired on the
    process-global instrumentation dispatcher, not on the agent object. If a
    governance handler is already registered (a reused process serving a new
    runtime), it is **rebound** to this run's evaluator / session instead of
    being left pointing at the previous run.

    Called by :class:`UiPathLlamaIndexRuntimeFactory` when an ``evaluator``
    is supplied to ``new_runtime``.
    """
    dispatcher = get_dispatcher()
    for handler in dispatcher.event_handlers:
        if isinstance(handler, GovernanceEventHandler):
            handler.rebind(
                evaluator=evaluator, agent_name=agent_name, session_id=session_id
            )
            logger.debug("Rebound existing governance handler to the new runtime")
            return agent
    callbacks = GovernanceCallbacks(
        evaluator=evaluator, agent_name=agent_name, session_id=session_id
    )
    dispatcher.add_event_handler(GovernanceEventHandler(callbacks=callbacks))
    logger.debug("Registered governance event handler on LlamaIndex dispatcher")
    return agent


def uninstall_governance(agent: Any = None) -> Any:
    """Remove the governance handler(s) from the root dispatcher.

    The instrumentation dispatcher is process-global, so a registered handler
    (and the evaluator it holds) would otherwise outlive the runtime. The
    factory calls this on ``dispose`` to release it. Returns ``agent`` unchanged.
    Safe to call when nothing is registered.
    """
    dispatcher = get_dispatcher()
    handlers = dispatcher.event_handlers
    remaining = [h for h in handlers if not isinstance(h, GovernanceEventHandler)]
    if len(remaining) != len(handlers):
        # event_handlers is a plain list; mutate in place to avoid a pydantic
        # attribute re-assignment on the Dispatcher model.
        handlers[:] = remaining
        logger.debug("Removed governance event handler from LlamaIndex dispatcher")
    return agent


class GovernanceEventHandler(BaseEventHandler):
    """Routes LlamaIndex instrumentation events to a governance evaluator.

    A pydantic model (``BaseEventHandler`` is one), so the evaluator + state
    are held in a private attribute. ``handle`` is called synchronously by the
    dispatcher for every event; we dispatch the three governance-relevant
    types and ignore the rest.
    """

    _callbacks: "GovernanceCallbacks" = PrivateAttr()

    def __init__(self, callbacks: "GovernanceCallbacks", **data: Any) -> None:
        super().__init__(**data)
        self._callbacks = callbacks

    @classmethod
    def class_name(cls) -> str:
        return "GovernanceEventHandler"

    def rebind(
        self,
        evaluator: EvaluatorProtocol,
        agent_name: str,
        session_id: str,
    ) -> None:
        """Re-point the single process-global handler at a new runtime."""
        self._callbacks.rebind(
            evaluator=evaluator, agent_name=agent_name, session_id=session_id
        )

    def handle(self, event: Any, **kwargs: Any) -> Any:
        # The dispatcher calls ``handle`` synchronously and inline with the
        # instrumented call. That is deliberate: a BEFORE_MODEL / TOOL_CALL
        # governance decision must complete (and be able to BLOCK) *before* the
        # underlying LLM / tool call proceeds — an async, out-of-band check
        # could not gate it. The evaluator is expected to be fast.
        if isinstance(event, LLMChatStartEvent):
            self._callbacks.before_model(event.messages)
        elif isinstance(event, LLMChatEndEvent):
            self._callbacks.after_model(event.response)
        elif isinstance(event, AgentToolCallEvent):
            self._callbacks.tool_call(event.tool, event.arguments)
        return None


class GovernanceCallbacks:
    """Holds the evaluator + per-attach state, called by the event handler.

    :class:`GovernanceBlockException` is re-raised (it aborts the run);
    anything else is logged and swallowed so a governance bug never breaks an
    agent run.
    """

    def __init__(
        self,
        evaluator: EvaluatorProtocol,
        agent_name: str,
        session_id: str,
    ) -> None:
        self._evaluator = evaluator
        self._agent_name = agent_name
        self._session_id = session_id
        # ``trace_id`` is intentionally NOT held here. A single uuid minted at
        # install time would be identical for every call. Trace correlation is
        # owned by the layer below (OTel span / HTTP resolve at call time),
        # matching the LangChain adapter.
        self._session_state: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0}

    def rebind(
        self,
        evaluator: EvaluatorProtocol,
        agent_name: str,
        session_id: str,
    ) -> None:
        """Re-point this callback set at a new run.

        Called when the process-global handler is reused for a fresh runtime —
        updates the evaluator and identifiers and resets the per-run counters so
        state does not bleed across runtimes.
        """
        self._evaluator = evaluator
        self._agent_name = agent_name
        self._session_id = session_id
        self._session_state = {"tool_calls": 0, "llm_calls": 0}

    def before_model(self, messages: Any) -> None:
        """Evaluate BEFORE_MODEL on the latest input message (see ADK rationale)."""
        try:
            self._evaluator.evaluate_before_model(
                model_input=_latest_message_text(messages),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
            # Count only calls that passed governance — a DENY raises above, so
            # a blocked call must not inflate the counter.
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001 - governance must not break the run
            logger.warning("before_model governance check failed (continuing): %s", e)

    def after_model(self, response: Any) -> None:
        """Evaluate AFTER_MODEL on the chat response text."""
        try:
            self._evaluator.evaluate_after_model(
                model_output=_response_text(response),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    def tool_call(self, tool: Any, arguments: Any) -> None:
        """Evaluate TOOL_CALL with the tool name + arguments."""
        try:
            self._evaluator.evaluate_tool_call(
                tool_name=getattr(tool, "name", None) or "unknown",
                tool_args=_coerce_args(arguments),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                session_state=self._session_state,
            )
            # Count only calls that passed governance; the evaluator saw the
            # count of prior tool calls, and a DENY raises before this bump.
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("tool_call governance check failed (continuing): %s", e)


# --------------------------------------------------------------------------
# Text / argument extraction
# --------------------------------------------------------------------------


def _latest_message_text(messages: Any) -> str:
    """Text of the most-recent message in a chat request."""
    if not messages:
        return ""
    if isinstance(messages, (list, tuple)):
        return _message_text(messages[-1])
    return _message_text(messages)


def _message_text(message: Any) -> str:
    """Pull text from a ``ChatMessage`` (``.content`` / ``.blocks``) or a str."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message[:_BEFORE_MODEL_TEXT_CAP]
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content[:_BEFORE_MODEL_TEXT_CAP]
    # Multimodal ChatMessage carries typed blocks. Walk them for text (a
    # TextBlock exposes ``.text``) rather than ``str(message)``, which would
    # serialize the pydantic repr — dict-syntax noise that pollutes the
    # regex-scanned blob. Non-text blocks (image/binary) have no scannable text.
    blocks = getattr(message, "blocks", None)
    if isinstance(blocks, (list, tuple)):
        texts = [
            t for b in blocks if isinstance((t := getattr(b, "text", None)), str) and t
        ]
        if texts:
            return "\n".join(texts)[:_BEFORE_MODEL_TEXT_CAP]
    return str(message)[:_BEFORE_MODEL_TEXT_CAP]


def _response_text(response: Any) -> str:
    """Pull assistant text from a ``ChatResponse`` (``.message.content``)."""
    if response is None:
        return ""
    message = getattr(response, "message", None)
    if message is not None:
        return _message_text(message)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text[:_BEFORE_MODEL_TEXT_CAP]
    return str(response)[:_BEFORE_MODEL_TEXT_CAP]


def _coerce_args(arguments: Any) -> Dict[str, Any]:
    """Normalise tool arguments (JSON string / Mapping / list / None) to a dict.

    ``AgentToolCallEvent.arguments`` is usually a JSON-encoded string; other
    call sites may hand a dict directly. Non-dict payloads are preserved (not
    dropped) so an arg-based policy can still scan them: a list-shaped arg
    (common with MCP tools) is wrapped under ``_``, and malformed JSON is kept
    raw under ``_raw`` — a payload governance can't parse must not be a way to
    slip past it.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"_": parsed}
        except (TypeError, ValueError):
            return {"_raw": arguments}
    # list / tuple / other structured args — preserve rather than drop to {}.
    return {"_": arguments}


__all__: List[str] = [
    "GovernanceCallbacks",
    "GovernanceEventHandler",
    "install_governance",
    "uninstall_governance",
]
