"""OpenAI Agents adapter for UiPath governance.

Provides governance for OpenAI Agents SDK agents (``agents.Agent`` and any
graph of agents reachable via ``handoffs``). Like the Google ADK adapter —
and unlike the LangChain adapter, which wraps a ``Runnable`` and intercepts
``invoke`` / ``ainvoke`` — OpenAI Agents are executed by ``Runner.run`` /
``Runner.run_streamed``, which hold their **own** reference to the agent
object. Replacing ``runtime.agent`` with a proxy would never reach the
``Runner``. So this adapter installs governance directly onto each agent's
native ``hooks`` attribute (an :class:`agents.AgentHooks`), mutating it in
place:

- ``on_llm_start``  → BEFORE_MODEL
- ``on_llm_end``    → AFTER_MODEL
- ``on_tool_start`` → TOOL_CALL
- ``on_tool_end``   → AFTER_TOOL

Because the mutation is in place, :meth:`OpenAIAgentsAdapter.attach` returns
the **original agent** (hooks installed) rather than a wrapping proxy.
``agents.Agent`` validates that ``hooks`` is an ``AgentHooks`` instance, so
:class:`GovernanceAgentHooks` subclasses it (the ADK adapter could duck-type
its callbacks; here the SDK type-checks the slot).

``agent.hooks`` holds a **single** ``AgentHooks`` (not a list, as in ADK), so
when an agent already carries user hooks we *chain*: governance runs first,
then the previously-installed hooks. ``detach`` restores the original.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host, so they are not fired here — that would duplicate every
boundary evaluation. (The SDK's per-agent ``on_start`` / ``on_end`` are
pass-through-only here for that reason.)

Contracts and the evaluator protocol come from ``uipath-core``; this package
contributes only the OpenAI-Agents-specific implementation and registers it
with the adapter registry via the ``uipath.governance.adapters`` entry point.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator itself. Each hook only extracts the relevant
payload and calls the matching ``evaluate_*`` method;
:class:`GovernanceBlockException` is allowed to propagate (it aborts the
``Runner`` run), anything else is logged and swallowed so a governance bug
never breaks an agent run.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from uuid import uuid4

from agents import Agent, AgentHooks
from uipath.core.adapters import BaseAdapter, EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the governance host and the other adapters so
# scan-time budgets are consistent across hooks. A long conversation history is
# governed at the LLM layer by scanning only the latest request content, not the
# full prompt — see :func:`_latest_input_text`.
_BEFORE_MODEL_TEXT_CAP = 64000

# Marks an agent we have already governed so a double ``attach`` is a no-op and
# ``detach`` can restore the hooks slot to whatever was there before.
_PREV_HOOKS_ATTR = "_uipath_governance_prev_hooks"


class OpenAIAgentsAdapter(BaseAdapter):
    """Adapter for the OpenAI Agents SDK.

    Detects ``agents.Agent`` instances and installs governance hooks on every
    agent reachable through the ``handoffs`` graph.
    """

    @property
    def name(self) -> str:
        return "OpenAIAgents"

    def can_handle(self, agent: Any) -> bool:
        """Return True only for an OpenAI Agents ``Agent``."""
        return isinstance(agent, Agent)

    def attach(
        self,
        agent: Any,
        agent_id: str,
        session_id: str,
        evaluator: EvaluatorProtocol,
    ) -> Any:
        """Install governance hooks on the agent graph (mutated in place).

        Returns the original ``agent`` — the ``Runner`` already holds this
        reference, so in-place mutation is what actually wires governance into
        execution. A wrapping proxy would not reach the ``Runner`` and would
        break the SDK's ``isinstance(agent, Agent)`` checks.
        """
        agents = _iter_agents(agent)
        installed = 0
        for node in agents:
            if isinstance(getattr(node, "hooks", None), GovernanceAgentHooks):
                continue  # idempotent — already governed
            prev = getattr(node, "hooks", None)
            hooks = GovernanceAgentHooks(
                evaluator=evaluator,
                agent_name=agent_id,
                session_id=session_id,
                inner=prev,
            )
            # Remember what was there so detach can restore it.
            setattr(node, _PREV_HOOKS_ATTR, prev)
            node.hooks = hooks
            installed += 1
        if not agents:
            logger.warning(
                "OpenAIAgentsAdapter found no Agent in %s — deep hooks will not fire",
                type(agent).__name__,
            )
        else:
            logger.debug("Installed governance hooks on %d OpenAI agent(s)", installed)
        return agent

    def detach(self, governed: Any) -> Any:
        """Restore each agent's original ``hooks`` slot and return the graph."""
        for node in _iter_agents(governed):
            if isinstance(getattr(node, "hooks", None), GovernanceAgentHooks):
                node.hooks = getattr(node, _PREV_HOOKS_ATTR, None)
            if hasattr(node, _PREV_HOOKS_ATTR):
                delattr(node, _PREV_HOOKS_ATTR)
        return governed


def _iter_agents(root: Any) -> List[Any]:
    """Return every agent node reachable through the ``handoffs`` graph.

    A node qualifies if it exposes the ``hooks`` slot. Handoff targets may be
    ``Agent`` instances or ``Handoff`` objects that carry the target on
    ``.agent``; both are followed so a multi-agent app is governed end to end.
    Cycles and pathological depth are bounded by an id-visited set and a hard
    cap.
    """
    found: List[Any] = []
    seen: set[int] = set()
    stack: List[Any] = [root]
    while stack and len(seen) < 1000:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if hasattr(node, "hooks"):
            found.append(node)
        handoffs = getattr(node, "handoffs", None)
        if isinstance(handoffs, (list, tuple)):
            for h in handoffs:
                # A Handoff wraps its target agent on ``.agent``; a bare Agent
                # is itself the target.
                stack.append(getattr(h, "agent", h))
    return found


class GovernanceAgentHooks(AgentHooks):  # type: ignore[type-arg]
    """Per-agent ``AgentHooks`` bound to one governance evaluator.

    The evaluator owns audit emission and DENY-raising. Each hook extracts the
    relevant payload, calls the matching ``evaluate_*`` method, and returns
    ``None``. :class:`GovernanceBlockException` is allowed to propagate — it
    aborts the ``Runner`` run — anything else is logged and swallowed.

    When the agent already carried an ``AgentHooks`` (``inner``), governance
    runs first and then delegates to it, so user hooks keep working.
    """

    def __init__(
        self,
        evaluator: EvaluatorProtocol,
        agent_name: str,
        session_id: str,
        inner: Any = None,
    ) -> None:
        self._evaluator = evaluator
        self._agent_name = agent_name
        self._session_id = session_id
        self._inner = inner
        self._trace_id = str(uuid4())
        self._session_state: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0}

    # ----- Model hooks -----------------------------------------------------

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Any,
        input_items: Any,
    ) -> None:
        """Evaluate BEFORE_MODEL rules immediately before the LLM call.

        Scans only the **latest input item** — not the full history. The model
        still receives the entire history (this hook does not mutate the
        request); the evaluator focuses on the new content the agent is about
        to respond to. Without this scoping, a violation in an earlier turn
        would re-fire on every subsequent model call because that text stays in
        the prompt for context.
        """
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            model_input = _latest_input_text(input_items)
            self._evaluator.evaluate_before_model(
                model_input=model_input,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001 - governance must not break the run
            logger.warning("on_llm_start governance check failed (continuing): %s", e)
        await _delegate(self._inner, "on_llm_start", context, agent, system_prompt, input_items)

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        """Evaluate AFTER_MODEL rules immediately after the LLM response."""
        try:
            model_output = _model_response_text(response)
            self._evaluator.evaluate_after_model(
                model_output=model_output,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("on_llm_end governance check failed (continuing): %s", e)
        await _delegate(self._inner, "on_llm_end", context, agent, response)

    # ----- Tool hooks ------------------------------------------------------

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """Evaluate TOOL_CALL rules immediately before a tool is invoked.

        The OpenAI Agents SDK does not surface tool *arguments* on
        ``on_tool_start`` (only the tool itself), so ``tool_args`` is empty
        here — argument-shaped rules evaluate at AFTER_TOOL via the result, or
        at the model layer where the call's arguments are visible in the output.
        """
        try:
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
            tool_name = getattr(tool, "name", None) or "unknown"
            self._evaluator.evaluate_tool_call(
                tool_name=tool_name,
                tool_args={},
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
                session_state=self._session_state,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("on_tool_start governance check failed (continuing): %s", e)
        await _delegate(self._inner, "on_tool_start", context, agent, tool)

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        """Evaluate AFTER_TOOL rules immediately after a tool is invoked.

        The SDK passes ``tool`` to both ``on_tool_start`` and ``on_tool_end``,
        so the name is read directly here — no start→end correlation is needed
        (unlike callback frameworks whose end hook omits the tool).
        """
        try:
            tool_name = getattr(tool, "name", None) or "unknown"
            tool_result = "" if result is None else _stringify(result)
            self._evaluator.evaluate_after_tool(
                tool_name=tool_name,
                tool_result=tool_result,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("on_tool_end governance check failed (continuing): %s", e)
        await _delegate(self._inner, "on_tool_end", context, agent, tool, result)

    # ----- Pass-through boundaries ----------------------------------------
    # BEFORE_AGENT / AFTER_AGENT are owned by the governance host; here we only
    # forward to any wrapped user hooks so their behaviour is preserved.

    async def on_start(self, context: Any, agent: Any) -> None:
        await _delegate(self._inner, "on_start", context, agent)

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        await _delegate(self._inner, "on_end", context, agent, output)

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        await _delegate(self._inner, "on_handoff", context, agent, source)


# --------------------------------------------------------------------------
# Delegation + text extraction (module-level, sync, duck-typed)
#
# Extraction is duck-typed on purpose: the OpenAI Agents SDK's run-item /
# response shapes are not stable public models, so we read attributes
# defensively rather than isinstance-checking SDK types that may move.
# --------------------------------------------------------------------------


async def _delegate(inner: Any, method: str, *args: Any) -> None:
    """Call ``inner.<method>(*args)`` if a wrapped hooks object provides it.

    User hooks are best-effort: a failure in a chained hook is logged and
    swallowed (it must not abort the run on governance's behalf), except a
    :class:`GovernanceBlockException`, which always propagates.
    """
    if inner is None:
        return
    fn = getattr(inner, method, None)
    if fn is None:
        return
    try:
        await fn(*args)
    except GovernanceBlockException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("chained user hook %s failed (continuing): %s", method, e)


def _latest_input_text(input_items: Any) -> str:
    """Extract text from the most-recent item in an LLM-call input list.

    ``input_items`` is the full ``list`` of response input items sent to the
    model. We take the last entry — the new user message, or the tool
    ``function_call_output`` being fed back — and pull its text via
    :func:`_item_text`. Returns ``""`` when there is nothing extractable.
    """
    if not input_items:
        return ""
    if isinstance(input_items, (list, tuple)):
        return _item_text(input_items[-1])
    return _item_text(input_items)


def _item_text(item: Any) -> str:
    """Return governance-relevant text from one response input/output item.

    Tolerant of both dict-shaped items (``{"role": ..., "content": ...}``,
    ``{"type": "function_call", "name": ..., "arguments": ...}``) and
    object-shaped items (``.content`` / ``.text`` / ``.name`` / ``.arguments``).
    Content may itself be a string or a list of parts (each a dict with
    ``text`` / ``input_text`` / ``output_text`` or an object with ``.text``).
    Capped at :data:`_BEFORE_MODEL_TEXT_CAP`.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item[:_BEFORE_MODEL_TEXT_CAP]

    pieces: List[str] = []

    # A function/tool call carries its intent in name + arguments.
    name = _get(item, "name")
    arguments = _get(item, "arguments")
    if name and (_get(item, "type") in (None, "function_call") or arguments is not None):
        if isinstance(name, str):
            pieces.append(name)
        if arguments is not None:
            pieces.append(_stringify(arguments))

    content = _get(item, "content")
    if content is not None:
        pieces.append(_content_text(content))

    # Tool result fed back to the model.
    output = _get(item, "output")
    if output is not None and not pieces:
        pieces.append(_stringify(output))

    text = "\n".join(p for p in pieces if p)
    return text[:_BEFORE_MODEL_TEXT_CAP]


def _content_text(content: Any) -> str:
    """Return text from a message ``content`` (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out: List[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
                continue
            t = (
                _get(part, "text")
                or _get(part, "input_text")
                or _get(part, "output_text")
            )
            if isinstance(t, str) and t:
                out.append(t)
        return "\n".join(out)
    t = _get(content, "text")
    return t if isinstance(t, str) else ""


def _model_response_text(response: Any) -> str:
    """Extract assistant text + tool-call intent from a ``ModelResponse``.

    ``response.output`` is the ``list`` of output items the model produced
    (assistant messages and function/tool calls). Each is run through
    :func:`_item_text` so both visible replies and tool-call arguments are
    governed. Capped at :data:`_BEFORE_MODEL_TEXT_CAP`.
    """
    if response is None:
        return ""
    output = _get(response, "output")
    if output is None:
        # Some shapes hand back text directly.
        return _item_text(response)
    items = output if isinstance(output, (list, tuple)) else [output]
    collected: List[str] = []
    remaining = _BEFORE_MODEL_TEXT_CAP
    for item in items:
        if remaining <= 0:
            break
        piece = _item_text(item)
        if piece:
            collected.append(piece)
            remaining -= len(piece) + 1
    return "\n".join(collected)[:_BEFORE_MODEL_TEXT_CAP]


def _get(obj: Any, attr: str) -> Any:
    """Read ``attr`` from a dict key or object attribute, else ``None``."""
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _stringify(value: Any) -> str:
    """Render a dict / object payload as compact, scannable text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)