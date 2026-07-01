"""Pydantic AI governance model wrapper for UiPath.

Pydantic AI has the thinnest hook surface of the supported frameworks — there
is no per-agent callback or middleware system. But *everything* an agent does
flows through its ``Model``: the LLM request, the model's tool-call requests
(``ToolCallPart`` in the response), and the tool results fed back on the next
turn (``ToolReturnPart`` in the request). So this adapter governs by wrapping
``agent.model`` with a :class:`GovernanceModel` (a ``pydantic_ai`` ``WrapperModel``)
that brackets every model call:

- BEFORE_MODEL — the latest request message's text (user prompt or tool result
  being fed back), before delegating to the wrapped model.
- AFTER_TOOL   — any ``ToolReturnPart`` in that latest request message.
- AFTER_MODEL  — the ``TextPart`` content of the model's response.
- TOOL_CALL    — each ``ToolCallPart`` the model emits (tool name + arguments).

Both the non-streaming ``request`` and the streaming ``request_stream`` paths
are covered (the runtime uses ``agent.run`` and ``agent.iter`` respectively).

Because the wrap is installed on ``agent.model`` in place,
:func:`install_governance` returns the **original agent**.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host and are intentionally not fired here.

The evaluator protocol comes from ``uipath-core``; this package contributes
only the Pydantic-AI-specific wiring. Governance is installed by the runtime
factory: passing an ``evaluator`` to ``new_runtime`` calls
:func:`install_governance` on the resolved agent. No adapter registry, no
entry point, no import-time side effects.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator. The wrapper only extracts payloads and calls
the matching ``evaluate_*`` method; :class:`GovernanceBlockException` propagates
(aborting the run), anything else is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterable, List

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from uipath.core.adapters import EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters.
_BEFORE_MODEL_TEXT_CAP = 64000


def install_governance(
    agent: Agent,
    evaluator: EvaluatorProtocol,
    *,
    agent_name: str,
    session_id: str,
) -> Agent:
    """Wrap ``agent.model`` with a :class:`GovernanceModel` (mutated in place).

    Returns the original ``agent``. Idempotent: an already-wrapped model is
    left untouched. If the agent has no concrete ``Model`` bound (the model is
    supplied per-run), there is nothing to wrap and a warning is logged.

    Called by :class:`UiPathPydanticAIRuntimeFactory` when an ``evaluator``
    is supplied to ``new_runtime``.
    """
    model = getattr(agent, "model", None)
    if isinstance(model, GovernanceModel):
        return agent  # idempotent — already governed
    if not isinstance(model, Model):
        logger.warning(
            "install_governance: agent has no bound Model to wrap (got %s); "
            "model-layer governance will not fire",
            type(model).__name__,
        )
        return agent
    callbacks = GovernanceCallbacks(
        evaluator=evaluator, agent_name=agent_name, session_id=session_id
    )
    # ``agent.model`` is a property whose setter stores ``_model``; the agent
    # re-reads ``self.model`` on every run (``_get_model``), so this in-place
    # wrap takes effect for all subsequent runs with no stale reference. A model
    # supplied per-run (``agent.run(model=...)``) bypasses this wrap — that path
    # is governed by whatever model the caller passes, not by us.
    agent.model = GovernanceModel(model, callbacks)
    logger.debug("Wrapped Pydantic AI agent model with governance")
    return agent


class GovernanceModel(WrapperModel):
    """A ``WrapperModel`` that brackets every model call with governance."""

    def __init__(self, wrapped: Model, callbacks: "GovernanceCallbacks") -> None:
        super().__init__(wrapped)
        self._callbacks = callbacks

    async def request(
        self,
        messages: List[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        self._callbacks.on_request(messages)
        response = await super().request(
            messages, model_settings, model_request_parameters
        )
        self._callbacks.on_response(response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: List[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        self._callbacks.on_request(messages)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            try:
                yield stream
            finally:
                # Once the caller has consumed the stream the final response is
                # assembled — govern it the same as the non-streaming path. This
                # runs in a ``finally`` so AFTER_MODEL / TOOL_CALL still fire
                # even if the consumer's ``async for`` raised partway through.
                #
                # Streaming governance is inherently post-hoc: the tokens have
                # already been streamed to the caller by the time the response
                # is complete, so a DENY here aborts the run but cannot un-send
                # what was already emitted. The block exception still
                # propagates; any other governance error is logged and swallowed
                # so a governance bug can't break the run.
                try:
                    self._callbacks.on_response(stream.get())
                except GovernanceBlockException:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "after-stream governance check failed (continuing): %s", e
                    )


class GovernanceCallbacks:
    """Holds the evaluator + per-attach state, called by :class:`GovernanceModel`.

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

    # ----- before the model call --------------------------------------

    def on_request(self, messages: Any) -> None:
        """Fire BEFORE_MODEL (latest message text) + AFTER_TOOL (tool returns).

        Only the latest request message is scanned, so a tool result / prompt
        is not re-evaluated on every subsequent model call (the full history is
        re-sent each turn for context).
        """
        latest = self._latest_request(messages)
        if latest is None:
            self._before_model("")
            return
        parts = getattr(latest, "parts", None) or []
        self._before_model(self._parts_input_text(parts))
        for part in parts:
            if isinstance(part, ToolReturnPart):
                self._after_tool(
                    part.tool_name or "unknown",
                    part.content,
                    tool_call_id=getattr(part, "tool_call_id", None),
                )

    # ----- after the model call ---------------------------------------

    def on_response(self, response: Any) -> None:
        """Fire AFTER_MODEL (response text) + TOOL_CALL (each tool-call part)."""
        parts = getattr(response, "parts", None) or []
        self._after_model(self._response_text(parts))
        for part in parts:
            if isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
                self._tool_call(
                    part.tool_name or "unknown",
                    part.args,
                    tool_call_id=getattr(part, "tool_call_id", None),
                )

    # ----- individual evaluate_* wrappers (block-propagate, else swallow) --

    def _before_model(self, text: str) -> None:
        try:
            self._evaluator.evaluate_before_model(
                model_input=text,
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
        except Exception as e:  # noqa: BLE001
            logger.warning("before_model governance check failed (continuing): %s", e)

    def _after_model(self, text: str) -> None:
        try:
            self._evaluator.evaluate_after_model(
                model_output=text,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    def _tool_call(
        self, tool_name: str, args: Any, tool_call_id: str | None = None
    ) -> None:
        try:
            self._evaluator.evaluate_tool_call(
                tool_name=tool_name,
                tool_args=_coerce_args(args),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                session_state=self._session_state,
                tool_call_id=tool_call_id,
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

    def _after_tool(
        self, tool_name: str, content: Any, tool_call_id: str | None = None
    ) -> None:
        try:
            self._evaluator.evaluate_after_tool(
                tool_name=tool_name,
                tool_result="" if content is None else _stringify(content),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                tool_call_id=tool_call_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_tool governance check failed (continuing): %s", e)

    # ----- text extraction --------------------------------------------

    @staticmethod
    def _latest_request(messages: Any) -> Any:
        """Return the most recent ``ModelRequest`` message, or ``None``.

        Scans from the end for the last ``ModelRequest`` rather than blindly
        taking ``messages[-1]``: the history can end with a ``ModelResponse``
        (e.g. mid tool-call round-trips), and treating that as request input
        would scan the wrong side of the exchange.
        """
        if not messages or not isinstance(messages, (list, tuple)):
            return None
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                return message
        return None

    @classmethod
    def _parts_input_text(cls, parts: Any) -> str:
        """Join governance-relevant input text from a request message's parts.

        Covers user prompts and tool-return content (the model's input on a
        follow-up turn). Joined with a running cap so an oversized tool-return
        can't build a multi-megabyte string before the final slice.
        """

        def _pieces() -> Iterable[str]:
            for part in parts:
                if isinstance(part, UserPromptPart):
                    yield _content_text(part.content)
                elif isinstance(part, ToolReturnPart):
                    yield _stringify(part.content)

        return _join_within_cap(_pieces(), _BEFORE_MODEL_TEXT_CAP)

    @classmethod
    def _response_text(cls, parts: Any) -> str:
        """Join ``TextPart`` content from a model response's parts (running cap)."""

        def _pieces() -> Iterable[str]:
            for part in parts:
                if isinstance(part, TextPart) and part.content:
                    yield part.content

        return _join_within_cap(_pieces(), _BEFORE_MODEL_TEXT_CAP)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _join_within_cap(pieces: Iterable[str], cap: int) -> str:
    """Join non-empty ``pieces`` with newlines, stopping once ``cap`` is hit.

    Bounds the work (and the allocation) to ``cap`` characters instead of
    building the full string and slicing afterwards.
    """
    collected: List[str] = []
    remaining = cap
    for piece in pieces:
        if remaining <= 0:
            break
        if not piece:
            continue
        collected.append(piece[:remaining])
        remaining -= len(piece) + 1
    return "\n".join(collected)[:cap]


def _content_text(content: Any) -> str:
    """Render a ``UserPromptPart.content`` (str or list of items) as text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:_BEFORE_MODEL_TEXT_CAP]
    if isinstance(content, (list, tuple)):

        def _pieces() -> Iterable[str]:
            for item in content:
                if isinstance(item, str):
                    yield item
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        yield text

        return _join_within_cap(_pieces(), _BEFORE_MODEL_TEXT_CAP)
    return _stringify(content)


def _coerce_args(args: Any) -> Dict[str, Any]:
    """Normalise ``ToolCallPart.args`` (dict / JSON string / None) to a dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"_": parsed}
        except (TypeError, ValueError):
            return {}
    return {}


def _stringify(value: Any, cap: int = _BEFORE_MODEL_TEXT_CAP) -> str:
    """Render a dict / object payload as compact, scannable text, capped.

    Bounded by ``cap`` so an oversized tool result / return content can't hand
    a multi-megabyte string to the evaluator.
    """
    if isinstance(value, str):
        return value[:cap]
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:cap]
    except (TypeError, ValueError):
        return str(value)[:cap]
