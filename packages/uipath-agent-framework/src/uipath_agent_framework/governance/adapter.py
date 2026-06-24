"""Microsoft Agent Framework adapter for UiPath governance.

Provides governance for ``agent_framework`` agents (``Agent`` and
``WorkflowAgent`` graphs). The framework runs agents through a middleware
pipeline that it rebuilds from ``agent.middleware`` on **every** ``run`` call
("Re-categorize self.middleware at runtime to support dynamic changes"). So,
like the Google ADK and OpenAI Agents adapters — and unlike the LangChain
adapter, which wraps the ``Runnable`` — this adapter installs governance by
appending middleware to each agent's ``middleware`` list in place:

- :class:`GovernanceChatMiddleware` (a ``ChatMiddleware``) brackets the LLM
  call → BEFORE_MODEL before ``call_next`` / AFTER_MODEL after it.
- :class:`GovernanceFunctionMiddleware` (a ``FunctionMiddleware``) brackets a
  tool call → TOOL_CALL before ``call_next`` / AFTER_TOOL after it.

Both subclass the framework's middleware base classes because the framework's
``categorize_middleware`` sorts middleware into chat/function/agent pipelines
by ``isinstance`` — a duck-typed object would be silently dropped.

Because the mutation is in place, :meth:`AgentFrameworkAdapter.attach` returns
the **original agent**. For a ``WorkflowAgent`` the inner agents reachable via
``workflow.executors[*]._agent`` are governed too, so a multi-agent app is
covered end to end.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host, so they are not fired here. The framework's
``AgentMiddleware`` slot is therefore left untouched.

Contracts and the evaluator protocol come from ``uipath-core``; this package
contributes only the Agent-Framework-specific implementation and self-registers
it with the global adapter registry when
``uipath_agent_framework.governance`` is imported.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator. Each middleware only extracts the relevant
payload and calls the matching ``evaluate_*`` method;
:class:`GovernanceBlockException` is allowed to propagate (it aborts the run),
anything else is logged and swallowed so a governance bug never breaks a run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List
from uuid import uuid4

from agent_framework._middleware import (
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
)
from uipath.core.adapters import BaseAdapter, EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters so
# scan-time budgets are consistent across hooks. A long conversation history is
# governed at the LLM layer by scanning only the latest message, not the full
# prompt — see :meth:`GovernanceCallbacks._latest_message_text`.
_BEFORE_MODEL_TEXT_CAP = 64000


class AgentFrameworkAdapter(BaseAdapter):
    """Adapter for the Microsoft Agent Framework.

    Detects ``agent_framework`` agents and appends governance middleware to
    every agent reachable through a ``WorkflowAgent``'s executors (or the
    single agent itself).
    """

    @property
    def name(self) -> str:
        return "AgentFramework"

    def can_handle(self, agent: Any) -> bool:
        """Return True only for an ``agent_framework`` ``BaseAgent``."""
        try:
            from agent_framework import BaseAgent
        except ImportError:
            return False
        return isinstance(agent, BaseAgent)

    def attach(
        self,
        agent: Any,
        agent_id: str,
        session_id: str,
        evaluator: EvaluatorProtocol,
    ) -> Any:
        """Append governance middleware to the agent graph (mutated in place).

        Returns the original ``agent`` — the framework rebuilds the middleware
        pipeline from ``agent.middleware`` on each ``run``, so the in-place
        append is what wires governance into execution.
        """
        callbacks = GovernanceCallbacks(
            evaluator=evaluator, agent_name=agent_id, session_id=session_id
        )
        targets = _iter_agents(agent)
        installed = 0
        for node in targets:
            existing = list(getattr(node, "middleware", None) or [])
            if any(isinstance(m, _GOVERNANCE_MIDDLEWARE) for m in existing):
                continue  # idempotent — already governed
            # Governance runs first so it can BLOCK before user middleware.
            node.middleware = [
                GovernanceChatMiddleware(callbacks),
                GovernanceFunctionMiddleware(callbacks),
                *existing,
            ]
            installed += 1
        if not targets:
            logger.warning(
                "AgentFrameworkAdapter found no agent in %s — hooks will not fire",
                type(agent).__name__,
            )
        else:
            logger.debug("Installed governance middleware on %d agent(s)", installed)
        return agent

    def detach(self, governed: Any) -> Any:
        """Strip governance middleware from each agent and return the graph."""
        for node in _iter_agents(governed):
            existing = getattr(node, "middleware", None)
            if not existing:
                continue
            kept = [m for m in existing if not isinstance(m, _GOVERNANCE_MIDDLEWARE)]
            node.middleware = kept or None
        return governed


def _iter_agents(root: Any) -> List[Any]:
    """Return every agent node carrying a ``middleware`` slot.

    A plain ``Agent`` is itself the target. A ``WorkflowAgent`` exposes its
    inner agents through ``workflow.executors[*]._agent`` (the same traversal
    the breakpoint middleware uses), so a multi-agent app is governed end to
    end. Cycles / pathological size are bounded by an id-visited set and a cap.
    """
    found: List[Any] = []
    seen: set[int] = set()

    def _add(node: Any) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if hasattr(node, "middleware"):
            found.append(node)

    _add(root)
    workflow = getattr(root, "workflow", None)
    executors = getattr(workflow, "executors", None)
    if isinstance(executors, Mapping):
        for executor in list(executors.values()):
            inner = getattr(executor, "_agent", None)
            if inner is not None and len(seen) < 1000:
                _add(inner)
    return found


class GovernanceCallbacks:
    """Holds the governance evaluator + per-attach state shared by the two
    middleware classes.

    Each method extracts the relevant payload and calls the matching
    ``evaluate_*`` method. :class:`GovernanceBlockException` is re-raised (it
    aborts the run); anything else is logged and swallowed so a governance bug
    never breaks an agent run.
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

    # ----- Model --------------------------------------------------------

    def before_model(self, messages: Any) -> None:
        """Evaluate BEFORE_MODEL on the latest message only (see ADK rationale)."""
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            self._evaluator.evaluate_before_model(
                model_input=self._latest_message_text(messages),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001 - governance must not break the run
            logger.warning("before_model governance check failed (continuing): %s", e)

    def after_model(self, result: Any) -> None:
        """Evaluate AFTER_MODEL on the chat response text."""
        try:
            self._evaluator.evaluate_after_model(
                model_output=self._response_text(result),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    # ----- Tools --------------------------------------------------------

    def before_tool(self, function: Any, arguments: Any) -> None:
        """Evaluate TOOL_CALL with the tool name + arguments."""
        try:
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
            self._evaluator.evaluate_tool_call(
                tool_name=getattr(function, "name", None) or "unknown",
                tool_args=_coerce_args(arguments),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
                session_state=self._session_state,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("before_tool governance check failed (continuing): %s", e)

    def after_tool(self, function: Any, result: Any) -> None:
        """Evaluate AFTER_TOOL with the tool result."""
        try:
            self._evaluator.evaluate_after_tool(
                tool_name=getattr(function, "name", None) or "unknown",
                tool_result="" if result is None else _stringify(result),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_tool governance check failed (continuing): %s", e)

    # ----- Text extraction ---------------------------------------------

    @classmethod
    def _latest_message_text(cls, messages: Any) -> str:
        """Text of the most-recent message in a chat request."""
        if not messages:
            return ""
        if isinstance(messages, (list, tuple)):
            return cls._message_text(messages[-1])
        return cls._message_text(messages)

    @classmethod
    def _message_text(cls, message: Any) -> str:
        """Pull text from a ``Message`` (``.text``) or a bare string."""
        if message is None:
            return ""
        if isinstance(message, str):
            return message[:_BEFORE_MODEL_TEXT_CAP]
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text[:_BEFORE_MODEL_TEXT_CAP]
        return _stringify(message)[:_BEFORE_MODEL_TEXT_CAP]

    @classmethod
    def _response_text(cls, result: Any) -> str:
        """Pull text from a ``ChatResponse`` (``.text``) or fallbacks."""
        if result is None:
            return ""
        text = getattr(result, "text", None)
        if isinstance(text, str) and text:
            return text[:_BEFORE_MODEL_TEXT_CAP]
        messages = getattr(result, "messages", None)
        if isinstance(messages, (list, tuple)) and messages:
            return cls._message_text(messages[-1])
        return _stringify(result)[:_BEFORE_MODEL_TEXT_CAP]


class GovernanceChatMiddleware(ChatMiddleware):
    """Brackets each LLM call: BEFORE_MODEL, then ``call_next``, then AFTER_MODEL."""

    def __init__(self, callbacks: GovernanceCallbacks) -> None:
        self._cb = callbacks

    async def process(
        self, context: ChatContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        self._cb.before_model(getattr(context, "messages", None))
        await call_next()
        self._cb.after_model(getattr(context, "result", None))


class GovernanceFunctionMiddleware(FunctionMiddleware):
    """Brackets each tool call: TOOL_CALL, then ``call_next``, then AFTER_TOOL."""

    def __init__(self, callbacks: GovernanceCallbacks) -> None:
        self._cb = callbacks

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        function = getattr(context, "function", None)
        self._cb.before_tool(function, getattr(context, "arguments", None))
        await call_next()
        self._cb.after_tool(function, getattr(context, "result", None))


# Tuple used for isinstance idempotency / detach checks.
_GOVERNANCE_MIDDLEWARE = (GovernanceChatMiddleware, GovernanceFunctionMiddleware)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _coerce_args(arguments: Any) -> Dict[str, Any]:
    """Normalise tool arguments (Mapping / pydantic model / None) to a dict."""
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    model_dump = getattr(arguments, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # noqa: BLE001 - fall through to empty
            pass
    return {}


def _stringify(value: Any) -> str:
    """Render a dict / object payload as compact, scannable text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)