"""Microsoft Agent Framework governance middleware for UiPath.

Provides governance for ``agent_framework`` agents (``Agent`` and
``WorkflowAgent`` graphs). The framework runs agents through a middleware
pipeline that it rebuilds from ``agent.middleware`` on **every** ``run`` call
("Re-categorize self.middleware at runtime to support dynamic changes"). So,
like the Google ADK and OpenAI Agents integrations — and unlike the LangChain
one, which wraps the ``Runnable`` — governance is installed by appending
middleware to each agent's ``middleware`` list in place:

- :class:`GovernanceChatMiddleware` (a ``ChatMiddleware``) brackets the LLM
  call → BEFORE_MODEL before ``call_next`` / AFTER_MODEL after it.
- :class:`GovernanceFunctionMiddleware` (a ``FunctionMiddleware``) brackets a
  tool call → TOOL_CALL before ``call_next`` / AFTER_TOOL after it.

Both subclass the framework's middleware base classes because the framework's
``categorize_middleware`` sorts middleware into chat/function/agent pipelines
by ``isinstance`` — a duck-typed object would be silently dropped.

Because the mutation is in place, :func:`install_governance` returns the
**original agent**. For a ``WorkflowAgent`` the inner agents reachable via
``workflow.executors[*]._agent`` are governed too, so a multi-agent app is
covered end to end.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host, so they are not fired here. The framework's
``AgentMiddleware`` slot is therefore left untouched.

The evaluator protocol comes from ``uipath-core``; this package contributes
only the Agent-Framework-specific wiring. Governance is installed by the
runtime factory: passing an ``evaluator`` to ``new_runtime`` calls
:func:`install_governance` on the resolved agent. No adapter registry, no
entry point, no import-time side effects.

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

from agent_framework._middleware import (
    ChatContext,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
)
from uipath.core.adapters import EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters so
# scan-time budgets are consistent across hooks. A long conversation history is
# governed at the LLM layer by scanning only the latest message, not the full
# prompt — see :meth:`GovernanceCallbacks._latest_message_text`.
_BEFORE_MODEL_TEXT_CAP = 64000

# Hard cap on how many nodes the workflow walk visits, guarding against cyclic
# or pathologically deep (nested) workflows. Hitting it is logged, not silent.
_MAX_GRAPH_NODES = 1000


def install_governance(
    agent: Any,
    evaluator: EvaluatorProtocol,
    *,
    agent_name: str,
    session_id: str,
) -> Any:
    """Append governance middleware to the agent graph (mutated in place).

    Returns the original ``agent`` — the framework rebuilds the middleware
    pipeline from ``agent.middleware`` on each ``run``, so the in-place append
    is what wires governance into execution. Idempotent: an already-governed
    agent is skipped. For a ``WorkflowAgent`` the inner agents reachable via
    ``workflow.executors[*]._agent`` are governed too.

    Called by :class:`UiPathAgentFrameworkRuntimeFactory` when an ``evaluator``
    is supplied to ``new_runtime``.
    """
    callbacks = GovernanceCallbacks(
        evaluator=evaluator, agent_name=agent_name, session_id=session_id
    )
    targets = _iter_agents(agent)
    if not targets:
        logger.warning(
            "install_governance found no agent in %s — hooks will not fire",
            type(agent).__name__,
        )
        return agent

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
    logger.debug("Installed governance middleware on %d agent(s)", installed)
    return agent


def _iter_agents(root: Any) -> List[Any]:
    """Return every agent node carrying a ``middleware`` slot.

    A plain ``Agent`` is itself the target. A ``WorkflowAgent`` exposes its
    inner agents through ``workflow.executors[*]._agent``. Those inner agents
    can themselves be ``WorkflowAgent``s (workflow-of-workflows), so the walk
    **recurses** through nested workflows rather than stopping one level down —
    otherwise a nested workflow's agents would run ungoverned. Cycles and
    pathological depth are bounded by an id-visited set and a hard cap
    (``_MAX_GRAPH_NODES``), which logs rather than silently truncating.
    """
    found: List[Any] = []
    seen: set[int] = set()
    stack: List[Any] = [root]
    capped = False
    while stack:
        if len(seen) >= _MAX_GRAPH_NODES:
            capped = True
            break
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if hasattr(node, "middleware"):
            found.append(node)
        workflow = getattr(node, "workflow", None)
        executors = getattr(workflow, "executors", None)
        if isinstance(executors, Mapping):
            for executor in executors.values():
                inner = getattr(executor, "_agent", None)
                if inner is not None:
                    stack.append(inner)
    if capped:
        logger.warning(
            "install_governance stopped walking the agent graph at the %d-node "
            "cap; agents beyond it will not be governed",
            _MAX_GRAPH_NODES,
        )
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
        # ``trace_id`` is intentionally NOT held here. A single uuid minted at
        # install time would be identical for every call. Trace correlation is
        # owned by the layer below (OTel span / HTTP resolve at call time),
        # matching the LangChain adapter.
        self._session_state: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0}

    # ----- Model --------------------------------------------------------

    def before_model(self, messages: Any) -> None:
        """Evaluate BEFORE_MODEL on the latest message only (see ADK rationale)."""
        try:
            self._evaluator.evaluate_before_model(
                model_input=self._latest_message_text(messages),
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

    def after_model(self, result: Any) -> None:
        """Evaluate AFTER_MODEL on the chat response text."""
        try:
            self._evaluator.evaluate_after_model(
                model_output=self._response_text(result),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    # ----- Tools --------------------------------------------------------

    def before_tool(self, function: Any, arguments: Any) -> None:
        """Evaluate TOOL_CALL with the tool name + arguments."""
        try:
            self._evaluator.evaluate_tool_call(
                tool_name=getattr(function, "name", None) or "unknown",
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
            logger.warning("before_tool governance check failed (continuing): %s", e)

    def after_tool(self, function: Any, result: Any) -> None:
        """Evaluate AFTER_TOOL with the tool result."""
        try:
            self._evaluator.evaluate_after_tool(
                tool_name=getattr(function, "name", None) or "unknown",
                tool_result="" if result is None else _stringify(result),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_tool governance check failed (continuing): %s", e)

    # ----- Text extraction ---------------------------------------------
    # Payload text is read defensively via ``.text`` rather than
    # isinstance-checking agent-framework message/response models: those
    # shapes are still pre-release (rc) and not stable public types, so we
    # avoid coupling extraction to types that may move.

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

        if getattr(context, "stream", False):
            # Streaming: after ``call_next`` ``context.result`` is a
            # ResponseStream, not finalized text — reading it for AFTER_MODEL
            # yields nothing. Register a result hook so AFTER_MODEL runs on the
            # finalized ``ChatResponse`` the framework assembles once the stream
            # is consumed.
            hooks = getattr(context, "stream_result_hooks", None)
            if isinstance(hooks, list):
                hooks.append(self._govern_streamed_result)
            else:  # pragma: no cover - defensive: framework always provides it
                logger.debug(
                    "ChatContext has no stream_result_hooks; AFTER_MODEL will "
                    "not run for this streamed response"
                )
            await call_next()
            return

        try:
            await call_next()
        finally:
            # AFTER_MODEL must run even if the model call raised, so audit and
            # rules still observe whatever result is present.
            self._cb.after_model(getattr(context, "result", None))

    def _govern_streamed_result(self, response: Any) -> Any:
        """``stream_result_hook``: govern the finalized streamed ``ChatResponse``.

        Returns the response unchanged (governance observes, it does not
        rewrite). A DENY raised here still propagates to abort the run.
        """
        self._cb.after_model(response)
        return response


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
        try:
            await call_next()
        finally:
            # AFTER_TOOL must run even if the tool call raised.
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
        except Exception as e:  # noqa: BLE001
            # Don't silently drop the args from governance visibility — surface
            # that they couldn't be coerced.
            logger.warning(
                "governance: could not coerce %s tool args to a dict (%s); "
                "TOOL_CALL will see empty args",
                type(arguments).__name__,
                e,
            )
    return {}


def _stringify(value: Any, cap: int = _BEFORE_MODEL_TEXT_CAP) -> str:
    """Render a dict / object payload as compact, scannable text, capped.

    Bounded by ``cap`` so an oversized tool result or message payload can't
    hand a multi-megabyte string to the evaluator. Callers that slice the
    result again (the ``_message_text`` / ``_response_text`` fallbacks) are
    unaffected.
    """
    if isinstance(value, str):
        return value[:cap]
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:cap]
    except (TypeError, ValueError):
        return str(value)[:cap]
