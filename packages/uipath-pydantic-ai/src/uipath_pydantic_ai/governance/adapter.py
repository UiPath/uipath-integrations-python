"""Pydantic AI adapter for UiPath governance.

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

Because the wrap is installed on ``agent.model`` in place, :meth:`attach`
returns the **original agent**; :meth:`detach` restores the original model.

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are owned by the
governance host and are intentionally not fired here.

Contracts and the evaluator protocol come from ``uipath-core``; this package
contributes only the Pydantic-AI-specific implementation and registers it with
the adapter registry via the ``uipath.governance.adapters`` entry point.

Audit emission and enforcement (raising :class:`GovernanceBlockException` on
DENY) are owned by the evaluator. The wrapper only extracts payloads and calls
the matching ``evaluate_*`` method; :class:`GovernanceBlockException` propagates
(aborting the run), anything else is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List
from uuid import uuid4

from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from uipath.core.adapters import BaseAdapter, EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

logger = logging.getLogger(__name__)

# Cap on the text blob passed to BEFORE_MODEL / AFTER_MODEL governance
# evaluation. Sized to match the runtime side and the other adapters.
_BEFORE_MODEL_TEXT_CAP = 64000

# Attribute used to stash the original (unwrapped) model so detach can restore it.
_ORIGINAL_MODEL_ATTR = "_uipath_governance_original_model"


class PydanticAIAdapter(BaseAdapter):
    """Adapter for the Pydantic AI framework.

    Detects ``pydantic_ai.Agent`` instances and wraps their ``model`` with a
    :class:`GovernanceModel`.
    """

    @property
    def name(self) -> str:
        return "PydanticAI"

    def can_handle(self, agent: Any) -> bool:
        """Return True only for a ``pydantic_ai.Agent``."""
        try:
            from pydantic_ai import Agent
        except ImportError:
            return False
        return isinstance(agent, Agent)

    def attach(
        self,
        agent: Any,
        agent_id: str,
        session_id: str,
        evaluator: EvaluatorProtocol,
    ) -> Any:
        """Wrap ``agent.model`` with governance (mutated in place).

        Returns the original ``agent``. If the agent has no concrete ``Model``
        bound (the model is supplied per-run), there is nothing to wrap and a
        warning is logged.
        """
        model = getattr(agent, "model", None)
        if isinstance(model, GovernanceModel):
            return agent  # idempotent — already governed
        if not isinstance(model, Model):
            logger.warning(
                "PydanticAIAdapter: agent has no bound Model to wrap (got %s); "
                "model-layer governance will not fire",
                type(model).__name__,
            )
            return agent
        callbacks = GovernanceCallbacks(
            evaluator=evaluator, agent_name=agent_id, session_id=session_id
        )
        setattr(agent, _ORIGINAL_MODEL_ATTR, model)
        agent.model = GovernanceModel(model, callbacks)
        logger.debug("Wrapped Pydantic AI agent model with governance")
        return agent

    def detach(self, governed: Any) -> Any:
        """Restore the agent's original (unwrapped) model and return it."""
        if isinstance(getattr(governed, "model", None), GovernanceModel):
            original = getattr(governed, _ORIGINAL_MODEL_ATTR, None)
            if original is not None:
                governed.model = original
        if hasattr(governed, _ORIGINAL_MODEL_ATTR):
            delattr(governed, _ORIGINAL_MODEL_ATTR)
        return governed


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
            yield stream
        # After the caller has consumed the stream, the final response is
        # assembled — govern it the same as the non-streaming path. A DENY
        # decision must still abort the run, so the block exception propagates;
        # any other governance error is logged and swallowed.
        try:
            self._callbacks.on_response(stream.get())
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001 - a governance bug must not break the run
            logger.warning("after-stream governance check failed (continuing): %s", e)


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
        self._trace_id = str(uuid4())
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
            if _part_kind(part) == "tool-return":
                self._after_tool(
                    getattr(part, "tool_name", None) or "unknown",
                    getattr(part, "content", None),
                )

    # ----- after the model call ---------------------------------------

    def on_response(self, response: Any) -> None:
        """Fire AFTER_MODEL (response text) + TOOL_CALL (each tool-call part)."""
        parts = getattr(response, "parts", None) or []
        self._after_model(self._response_text(parts))
        for part in parts:
            if _part_kind(part) in ("tool-call", "builtin-tool-call"):
                self._tool_call(
                    getattr(part, "tool_name", None) or "unknown",
                    getattr(part, "args", None),
                )

    # ----- individual evaluate_* wrappers (block-propagate, else swallow) --

    def _before_model(self, text: str) -> None:
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            self._evaluator.evaluate_before_model(
                model_input=text,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
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
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_model governance check failed (continuing): %s", e)

    def _tool_call(self, tool_name: str, args: Any) -> None:
        try:
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
            self._evaluator.evaluate_tool_call(
                tool_name=tool_name,
                tool_args=_coerce_args(args),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
                session_state=self._session_state,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("tool_call governance check failed (continuing): %s", e)

    def _after_tool(self, tool_name: str, content: Any) -> None:
        try:
            self._evaluator.evaluate_after_tool(
                tool_name=tool_name,
                tool_result="" if content is None else _stringify(content),
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                trace_id=self._trace_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("after_tool governance check failed (continuing): %s", e)

    # ----- text extraction --------------------------------------------

    @staticmethod
    def _latest_request(messages: Any) -> Any:
        """Return the most recent message (a ``ModelRequest``) or ``None``."""
        if not messages or not isinstance(messages, (list, tuple)):
            return None
        return messages[-1]

    @classmethod
    def _parts_input_text(cls, parts: Any) -> str:
        """Join governance-relevant input text from a request message's parts.

        Covers user prompts and tool-return content (the model's input on a
        follow-up turn). Capped at :data:`_BEFORE_MODEL_TEXT_CAP`.
        """
        collected: List[str] = []
        for part in parts:
            kind = _part_kind(part)
            if kind == "user-prompt":
                collected.append(_content_text(getattr(part, "content", None)))
            elif kind == "tool-return":
                collected.append(_stringify(getattr(part, "content", None)))
        return "\n".join(p for p in collected if p)[:_BEFORE_MODEL_TEXT_CAP]

    @classmethod
    def _response_text(cls, parts: Any) -> str:
        """Join ``TextPart`` content from a model response's parts."""
        collected: List[str] = []
        for part in parts:
            if _part_kind(part) == "text":
                text = getattr(part, "content", None)
                if isinstance(text, str) and text:
                    collected.append(text)
        return "\n".join(collected)[:_BEFORE_MODEL_TEXT_CAP]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _part_kind(part: Any) -> str:
    """Return a message part's discriminator (``part_kind``), or ``""``."""
    kind = getattr(part, "part_kind", None)
    return kind if isinstance(kind, str) else ""


def _content_text(content: Any) -> str:
    """Render a ``UserPromptPart.content`` (str or list of items) as text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out: List[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    out.append(text)
        return "\n".join(out)
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


def _stringify(value: Any) -> str:
    """Render a dict / object payload as compact, scannable text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)