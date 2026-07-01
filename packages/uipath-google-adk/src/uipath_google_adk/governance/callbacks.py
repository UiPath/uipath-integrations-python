"""Google ADK governance callbacks for UiPath.

Provides governance for Google ADK agents (``google.adk.agents.LlmAgent``
and any ``BaseAgent`` tree containing them). Unlike the LangChain integration
— which wraps a ``Runnable`` and intercepts ``invoke`` / ``ainvoke`` — ADK
agents are executed by a ``Runner`` that holds its **own** reference to
the agent object. Replacing ``runtime.agent`` with a proxy would never
reach the ``Runner``. So :func:`install_governance` installs governance
directly onto each ``LlmAgent``'s native callback attributes, mutating them
in place:

- ``before_model_callback``  → BEFORE_MODEL
- ``after_model_callback``   → AFTER_MODEL
- ``before_tool_callback``   → TOOL_CALL
- ``after_tool_callback``    → AFTER_TOOL

Because the mutation is in place, :func:`install_governance` returns the
**original agent** (hooks installed) rather than a wrapping proxy.
Returning a proxy here would also break ADK's own ``isinstance(agent,
LlmAgent)`` checks in output-schema / graph resolution, since ``LlmAgent``
is a Pydantic model.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are intentionally
*not* fired from here — they are owned by the governance host. Firing them
here too would duplicate every boundary evaluation.

The evaluator protocol comes from ``uipath-core``; this package contributes
only the ADK-specific wiring. Governance is installed by the runtime
factory: passing an ``evaluator`` to ``new_runtime`` calls
:func:`install_governance` on the resolved agent. No adapter registry, no
entry point, no import-time side effects.

Audit emission and enforcement (raising :class:`GovernanceBlockException`
on DENY) are owned by the evaluator itself. Each callback only extracts
the relevant payload and calls the matching ``evaluate_*`` method;
:class:`GovernanceBlockException` is allowed to propagate (it aborts the
``Runner`` run), anything else is logged and swallowed so a governance
bug never breaks an agent run.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from uipath.core.adapters import EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the governance host and the other adapters so
# scan-time budgets are consistent across hooks. A long conversation
# history is governed at the LLM layer by scanning only the latest
# request content, not the full prompt — see
# :meth:`GovernanceCallbacks._latest_request_text`.
_BEFORE_MODEL_TEXT_CAP = 64000

# Hard cap on how many nodes the agent-tree walk visits, guarding against
# cyclic or pathologically deep trees. Hitting it is logged, not silent.
_MAX_GRAPH_NODES = 1000

# Native LlmAgent callback attribute names this adapter manages.
_MODEL_BEFORE = "before_model_callback"
_MODEL_AFTER = "after_model_callback"
_TOOL_BEFORE = "before_tool_callback"
_TOOL_AFTER = "after_tool_callback"


def _is_governance_callable(fn: Any) -> bool:
    """True if ``fn`` is a bound method of a :class:`GovernanceCallbacks`."""
    return isinstance(getattr(fn, "__self__", None), GovernanceCallbacks)


def _find_governance_callbacks(agent: Any) -> "GovernanceCallbacks | None":
    """Return the :class:`GovernanceCallbacks` already installed on ``agent``.

    Scans the four callback slots for a governance-owned callable and returns
    the instance backing it, else ``None``. Used to detect a cached agent that
    was governed by a previous ``new_runtime`` so its metadata can be refreshed
    rather than left stale.
    """
    for attr in (_MODEL_BEFORE, _MODEL_AFTER, _TOOL_BEFORE, _TOOL_AFTER):
        existing = getattr(agent, attr, None)
        handlers = existing if isinstance(existing, list) else [existing]
        for h in handlers:
            if _is_governance_callable(h):
                return h.__self__  # type: ignore[no-any-return]
    return None


def _install_callback(agent: Any, attr: str, fn: Any) -> None:
    """Prepend ``fn`` to an ADK callback slot, preserving existing handlers.

    ADK accepts a single callable or a ``list`` of callables for each
    ``*_callback`` field and runs them in order, stopping early if one
    returns a value (a short-circuit). Governance is prepended (runs
    first) so it always evaluates — and can BLOCK — before any
    user-supplied callback gets a chance to short-circuit the model /
    tool call.

    Idempotent: if a governance callback is already present in the slot,
    this is a no-op (so a double ``attach`` does not stack duplicates).
    """
    existing = getattr(agent, attr, None)
    if existing is None:
        handlers: List[Any] = []
    elif isinstance(existing, list):
        handlers = list(existing)
    else:
        handlers = [existing]
    if any(_is_governance_callable(h) for h in handlers):
        return
    setattr(agent, attr, [fn, *handlers])


def _iter_llm_agents(root: Any) -> List[Any]:
    """Return every ``LlmAgent``-shaped node in the agent tree.

    A node qualifies if it exposes the model-callback surface (duck-typed
    via :data:`_MODEL_BEFORE` so we don't hard-require ``LlmAgent`` to be
    importable). Container agents (``Sequential`` / ``Parallel`` / ``Loop``)
    have no model callbacks themselves but their ``sub_agents`` are walked
    so a multi-agent app is governed end to end.

    ``AgentTool``-wrapped agents are also followed: an agent exposed to another
    agent as a tool carries its target on ``tool.agent`` and lives in ``tools``
    (not ``sub_agents``), so it would otherwise be missed. Cycles and
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
        if hasattr(node, _MODEL_BEFORE):
            found.append(node)
        sub_agents = getattr(node, "sub_agents", None)
        if isinstance(sub_agents, (list, tuple)):
            stack.extend(sub_agents)
        # AgentTool wraps its target agent on ``.agent``; follow tools so an
        # agent-as-tool is governed too.
        tools = getattr(node, "tools", None)
        if isinstance(tools, (list, tuple)):
            for tool in tools:
                wrapped = getattr(tool, "agent", None)
                if wrapped is not None:
                    stack.append(wrapped)
    if capped:
        logger.warning(
            "install_governance stopped walking the agent tree at the %d-node "
            "cap; agents beyond it will not be governed",
            _MAX_GRAPH_NODES,
        )
    return found


def install_governance(
    agent: Any,
    evaluator: EvaluatorProtocol,
    *,
    agent_name: str,
    session_id: str,
) -> Any:
    """Install governance callbacks on the agent tree (mutated in place).

    Walks every ``LlmAgent`` reachable through ``sub_agents`` and prepends
    governance to each model/tool callback slot, preserving existing handlers.
    Returns the original ``agent`` — the ``Runner`` already holds this
    reference, so in-place mutation is what wires governance into execution.
    Idempotent: a slot that already carries a governance callback is skipped.

    Called by :class:`UiPathGoogleADKRuntimeFactory` when an ``evaluator``
    is supplied to ``new_runtime``.
    """
    llm_agents = _iter_llm_agents(agent)
    callbacks: GovernanceCallbacks | None = None
    for node in llm_agents:
        already = _find_governance_callbacks(node)
        if already is not None:
            # Cached agent reused for a new runtime: refresh the evaluator and
            # session/agent so governance attributes to *this* run rather than
            # the first one that installed it (the factory caches agents by
            # entrypoint across runtime_ids).
            already.rebind(
                evaluator=evaluator, agent_name=agent_name, session_id=session_id
            )
            continue
        if callbacks is None:
            callbacks = GovernanceCallbacks(
                evaluator=evaluator,
                agent_name=agent_name,
                session_id=session_id,
            )
        _install_callback(node, _MODEL_BEFORE, callbacks.before_model)
        _install_callback(node, _MODEL_AFTER, callbacks.after_model)
        _install_callback(node, _TOOL_BEFORE, callbacks.before_tool)
        _install_callback(node, _TOOL_AFTER, callbacks.after_tool)
    if not llm_agents:
        logger.warning(
            "install_governance found no LlmAgent in %s — deep hooks will not fire",
            type(agent).__name__,
        )
    else:
        logger.debug(
            "Installed governance callbacks on %d ADK LlmAgent(s)",
            len(llm_agents),
        )
    return agent


class GovernanceCallbacks:
    """Holds the four ADK callbacks bound to one governance evaluator.

    The evaluator owns audit emission and DENY-raising. Each callback
    extracts the relevant payload, calls the matching ``evaluate_*``
    method, and returns ``None`` (never short-circuiting the model / tool
    on its own). :class:`GovernanceBlockException` is allowed to
    propagate — it aborts the ``Runner`` run — anything else is logged
    and swallowed.
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

        Called when a cached agent (already carrying these callbacks) is reused
        for a fresh ``new_runtime`` — updates the evaluator and identifiers and
        resets the per-run counters so state does not bleed across runtimes.
        """
        self._evaluator = evaluator
        self._agent_name = agent_name
        self._session_id = session_id
        self._session_state = {"tool_calls": 0, "llm_calls": 0}

    # ----- Model callbacks -------------------------------------------------

    def before_model(self, callback_context: Any, llm_request: Any) -> None:
        """Evaluate BEFORE_MODEL rules at model start.

        Scans only the **latest request content** — not the full history.
        The model still receives the entire history (this callback does
        not mutate ``llm_request``); the evaluator focuses on the new
        content the agent is about to respond to. Without this scoping, a
        violation in an earlier turn would re-fire on every subsequent
        model call because that text stays in the prompt for context.

        Returns ``None`` so ADK proceeds with the model call.
        """
        try:
            model_input = self._latest_request_text(llm_request)
            self._evaluator.evaluate_before_model(
                model_input=model_input,
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
        except Exception as e:
            logger.warning("before_model governance check failed (continuing): %s", e)
        return None

    def after_model(self, callback_context: Any, llm_response: Any) -> None:
        """Evaluate AFTER_MODEL rules at model end.

        Partial (streamed) responses are skipped — ADK fires
        ``after_model_callback`` for each chunk with ``partial=True`` and
        once more for the aggregated final response. Governing only the
        final response avoids re-scanning the same text token-by-token.

        Returns ``None`` so ADK keeps the model's response unchanged.
        """
        try:
            if getattr(llm_response, "partial", False):
                return None
            content = getattr(llm_response, "content", None)
            model_output = self._content_text(content)
            self._evaluator.evaluate_after_model(
                model_output=model_output,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning("after_model governance check failed (continuing): %s", e)
        return None

    # ----- Tool callbacks --------------------------------------------------

    def before_tool(self, tool: Any, args: Dict[str, Any], tool_context: Any) -> None:
        """Evaluate TOOL_CALL rules at tool start.

        Returns ``None`` so ADK proceeds with the tool call (a non-None
        return would short-circuit it with a substitute result).
        """
        try:
            tool_name = getattr(tool, "name", None) or "unknown"
            self._evaluator.evaluate_tool_call(
                tool_name=tool_name,
                tool_args=self._cap_args(args or {}),
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
        except Exception as e:
            logger.warning("before_tool governance check failed (continuing): %s", e)
        return None

    def after_tool(
        self,
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any,
        tool_response: Any,
    ) -> None:
        """Evaluate AFTER_TOOL rules at tool end.

        Returns ``None`` so ADK keeps the tool's result unchanged.
        """
        try:
            tool_name = getattr(tool, "name", None) or "unknown"
            tool_result = (
                "" if tool_response is None else self._stringify(tool_response)
            )
            self._evaluator.evaluate_after_tool(
                tool_name=tool_name,
                tool_result=tool_result,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning("after_tool governance check failed (continuing): %s", e)
        return None

    # ----- Text extraction -------------------------------------------------
    # Read LlmRequest/LlmResponse/content/parts defensively via getattr
    # rather than isinstance on ADK's typed models: this keeps the adapter
    # from hard-coupling to google-adk internal types that may shift, and
    # lets the tests duck-type the payloads without a google-adk install.

    def _latest_request_text(self, llm_request: Any) -> str:
        """Extract text from the most-recent content in an ``LlmRequest``.

        ``llm_request.contents`` is the full ``list[Content]`` sent to the
        model. We take the last entry — the new user message, or the tool
        ``function_response`` being fed back — and pull its text cleanly
        via :meth:`_content_text`. Returns ``""`` when there is nothing
        extractable.
        """
        contents = getattr(llm_request, "contents", None)
        if not contents:
            return ""
        return self._content_text(contents[-1])

    @classmethod
    def _content_text(cls, content: Any) -> str:
        """Return governance-relevant text from a ``Content`` (or part list).

        Walks ``content.parts`` and pulls, per part:

        - ``part.text`` — plain text.
        - ``part.function_call`` — the tool name plus JSON-encoded
          ``args``; ADK / Gemini routinely carry the user-visible reply in
          a function call (e.g. a "submit final answer" tool).
        - ``part.function_response`` — the tool result fed back to the
          model; relevant when it is the latest content for BEFORE_MODEL.

        Capped at :data:`_BEFORE_MODEL_TEXT_CAP` so a runaway response or
        large tool payload can't blow scan budgets.
        """
        if content is None:
            return ""
        parts = getattr(content, "parts", None)
        if parts is None:
            # Some shapes hand us a bare string or a list of parts.
            if isinstance(content, str):
                return content[:_BEFORE_MODEL_TEXT_CAP]
            if isinstance(content, (list, tuple)):
                parts = content
            else:
                return ""
        collected: List[str] = []
        remaining = _BEFORE_MODEL_TEXT_CAP
        for part in parts:
            if remaining <= 0:
                break
            piece = cls._part_text(part)
            if piece:
                collected.append(piece)
                remaining -= len(piece) + 1
        return "\n".join(collected)[:_BEFORE_MODEL_TEXT_CAP]

    @classmethod
    def _part_text(cls, part: Any) -> str:
        """Return text / function-call args / function-response from one part."""
        pieces: List[str] = []
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            pieces.append(text)

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            name = getattr(function_call, "name", "") or ""
            fc_args = getattr(function_call, "args", None)
            if name:
                pieces.append(name)
            if fc_args:
                pieces.append(cls._stringify(fc_args))

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            response = getattr(function_response, "response", None)
            if response:
                pieces.append(cls._stringify(response))

        return "\n".join(p for p in pieces if p)

    @classmethod
    def _cap_args(cls, args: Dict[str, Any], cap: int = _BEFORE_MODEL_TEXT_CAP) -> Any:
        """Bound the tool-args payload before it reaches the evaluator.

        ``before_tool`` receives args straight from ADK; a huge blob (e.g. a
        tool called with a multi-megabyte string) would otherwise be scanned
        uncapped — contrast with ``after_tool``, which caps its result. Within
        budget the dict is passed through unchanged (so per-key rules still
        work); once its serialized size exceeds ``cap`` it is replaced with a
        single capped, stringified form.
        """
        if not isinstance(args, dict) or not args:
            return args
        blob = cls._stringify(args, cap + 1)
        if len(blob) <= cap:
            return args
        return {"_truncated": blob[:cap]}

    @staticmethod
    def _stringify(value: Any, cap: int = _BEFORE_MODEL_TEXT_CAP) -> str:
        """Render a dict / object payload as compact, scannable text, capped.

        Bounded by ``cap`` so an oversized tool result, function-call args
        blob, or function-response can't hand a multi-megabyte string to the
        evaluator.
        """
        if isinstance(value, str):
            return value[:cap]
        try:
            return json.dumps(value, default=str, ensure_ascii=False)[:cap]
        except (TypeError, ValueError):
            return str(value)[:cap]
