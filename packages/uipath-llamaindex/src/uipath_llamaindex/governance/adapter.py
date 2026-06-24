"""LlamaIndex adapter for UiPath governance.

Provides governance for LlamaIndex agents/workflows. Unlike the ADK / OpenAI /
Agent-Framework adapters — which install per-agent callbacks or middleware —
LlamaIndex routes everything (LLM calls, tool calls) through its global
**instrumentation dispatcher** (the same mechanism the package already uses for
OpenInference tracing). So this adapter governs by registering a
:class:`GovernanceEventHandler` on the **root dispatcher**, which receives every
event propagated from child dispatchers:

- ``LLMChatStartEvent``   → BEFORE_MODEL  (scans the latest input message)
- ``LLMChatEndEvent``     → AFTER_MODEL   (scans the response)
- ``AgentToolCallEvent``  → TOOL_CALL     (tool name + arguments)

The dispatcher is process-global, so registration is process-wide — which fits
the coded-agent model (one workflow per process). :meth:`attach` therefore
returns the ``agent`` unchanged (nothing is mutated on it); the wiring lives on
the dispatcher. :meth:`detach` removes the handler.

LlamaIndex does **not** emit a tool-*end* instrumentation event, so AFTER_TOOL
is not wired here; a tool's result is governed at the next ``LLMChatStartEvent``
where it is fed back to the model as input (analogous to how the OpenAI adapter
handles its missing tool-args).

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host and are intentionally not fired here.

Contracts and the evaluator protocol come from ``uipath-core``; this package
contributes only the LlamaIndex-specific implementation and registers it with
the adapter registry via the ``uipath.governance.adapters`` entry point.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator. The handler only extracts payloads and calls
the matching ``evaluate_*`` method; :class:`GovernanceBlockException` propagates
(aborting the run), anything else is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from uuid import uuid4

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
from uipath.core.adapters import BaseAdapter, EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters.
_BEFORE_MODEL_TEXT_CAP = 64000


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for the LlamaIndex framework.

    Detects LlamaIndex workflows/agents and governs them by registering a
    :class:`GovernanceEventHandler` on the root instrumentation dispatcher.
    """

    @property
    def name(self) -> str:
        return "LlamaIndex"

    def can_handle(self, agent: Any) -> bool:
        """Return True only for a LlamaIndex ``Workflow`` (incl. agent workflows)."""
        try:
            from workflows import Workflow
        except ImportError:
            return False
        return isinstance(agent, Workflow)

    def attach(
        self,
        agent: Any,
        agent_id: str,
        session_id: str,
        evaluator: EvaluatorProtocol,
    ) -> Any:
        """Register the governance event handler on the root dispatcher.

        Returns the ``agent`` unchanged — LlamaIndex governance is wired on the
        process-global dispatcher, not on the agent object. Idempotent: a
        second attach is a no-op while a handler is already registered.
        """
        dispatcher = get_dispatcher()
        if any(isinstance(h, GovernanceEventHandler) for h in dispatcher.event_handlers):
            return agent  # idempotent — already governed
        callbacks = GovernanceCallbacks(
            evaluator=evaluator, agent_name=agent_id, session_id=session_id
        )
        dispatcher.add_event_handler(GovernanceEventHandler(callbacks=callbacks))
        logger.debug("Registered governance event handler on LlamaIndex dispatcher")
        return agent

    def detach(self, governed: Any) -> Any:
        """Remove the governance event handler from the root dispatcher."""
        dispatcher = get_dispatcher()
        dispatcher.event_handlers = [
            h
            for h in dispatcher.event_handlers
            if not isinstance(h, GovernanceEventHandler)
        ]
        return governed


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

    def handle(self, event: Any, **kwargs: Any) -> Any:
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
        self._trace_id = str(uuid4())
        self._session_state: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0}

    def before_model(self, messages: Any) -> None:
        """Evaluate BEFORE_MODEL on the latest input message (see ADK rationale)."""
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            self._evaluator.evaluate_before_model(
                model_input=_latest_message_text(messages),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
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
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    def tool_call(self, tool: Any, arguments: Any) -> None:
        """Evaluate TOOL_CALL with the tool name + arguments."""
        try:
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
            self._evaluator.evaluate_tool_call(
                tool_name=getattr(tool, "name", None) or "unknown",
                tool_args=_coerce_args(arguments),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
                session_state=self._session_state,
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
    """Pull text from a ``ChatMessage`` (``.content``) or a bare string."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message[:_BEFORE_MODEL_TEXT_CAP]
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content[:_BEFORE_MODEL_TEXT_CAP]
    # Newer ChatMessage carries typed blocks; fall back to str().
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
    """Normalise tool arguments (JSON string / Mapping / None) to a dict.

    ``AgentToolCallEvent.arguments`` is a JSON-encoded string; other call
    sites may hand a dict directly.
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
            return {}
    return {}


__all__: List[str] = [
    "GovernanceCallbacks",
    "GovernanceEventHandler",
    "LlamaIndexAdapter",
]