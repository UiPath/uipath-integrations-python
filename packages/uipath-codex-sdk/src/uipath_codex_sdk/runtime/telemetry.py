"""UiPath trace spans for a Codex run.

A job is one ``Agent run`` span with the low-code agent span schema, so a coded
Codex agent's trace reads like any other agent's, however often it suspends:

* ``Agent run - <entrypoint>`` (``agentRun``)

  * ``LLM call`` (``llmCall``): one per ``/responses`` call through the gateway
    shim. The shim sends the trace context upstream, and the LLM Gateway adds
    its ``Model run`` span under it.
  * ``Tool call - <tool>`` (``toolCall``): durable tools (the child job's
    ``ParentSpanId``), and the MCP tools, commands and other items Codex runs
    itself. An MCP tool is named by its own name (its server is
    ``tool.server``); ``output.value`` is ``{"content": <text>}`` for
    ``uipath.eval``, ``output`` the full result.
  * ``Agent output`` (``agentOutput``): on a successful run.

Spans that outlive a suspension (the ``Agent run`` and the ``Tool call`` of a
durable job) are not ended when the execution suspends: they are pushed to
LLMOps as still running and their identity (:func:`span_record`) is saved in
the state database. A resumed execution parents its spans directly under the
saved ``Agent run`` (:func:`restored_span`), as low-code agents do, and the
execution that finishes one completes it with its original id and start time
(:func:`complete_span`).
"""

from __future__ import annotations

import json
import logging
import os
import time
import weakref
from contextvars import ContextVar, Token
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

logger = logging.getLogger(__name__)

TRACER_NAME = "uipath_codex_sdk"
TRACE_CONTEXT_FLAG = "EnableTraceContextHeaders"
TRACE_SOURCE = "source=codex-sdk"

SPAN_AGENT_RUN = "agentRun"
SPAN_AGENT_OUTPUT = "agentOutput"
SPAN_LLM_CALL = "llmCall"
SPAN_TOOL_CALL = "toolCall"
SPAN_INTEGRATION_TOOL = "integrationTool"
SPAN_CONTEXT_GROUNDING_TOOL = "contextGroundingTool"
SPAN_ESCALATION_TOOL = "escalationTool"
SPAN_IXP_TOOL = "ixpTool"
SPAN_VS_ESCALATION_TOOL = "vsEscalationTool"
SPAN_PROCESS_TOOL = "processTool"
SPAN_AGENT_TOOL = "agentTool"

#: Codex item types that are tool executions (``mcpToolCall`` of a user server,
#: shell commands, file edits, web searches, dynamic tools).
TOOL_ITEM_TYPES = frozenset(
    {"mcpToolCall", "commandExecution", "fileChange", "webSearch", "dynamicToolCall"}
)
#: Cap for text attributes (prompts, model output, tool results).
TEXT_CAP = 64_000


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def enable_gateway_model_spans() -> None:
    """Default ``EnableTraceContextHeaders`` on, as claude-sdk does.

    ``build_trace_context_headers`` reads the flag and nothing sets it for a
    coded agent. This only supplies a default: a deployment that sets
    ``UIPATH_FEATURE_EnableTraceContextHeaders`` still decides.
    """
    try:
        from uipath.core.feature_flags import FeatureFlags

        if FeatureFlags.get_flag(TRACE_CONTEXT_FLAG) is None:
            FeatureFlags.configure_flags({TRACE_CONTEXT_FLAG: True})
    except Exception:
        logger.debug("could not default %s", TRACE_CONTEXT_FLAG, exc_info=True)


def _cap(text: str) -> str:
    return text if len(text) <= TEXT_CAP else text[:TEXT_CAP] + "…"


def _json(value: Any) -> str:
    if isinstance(value, str):
        return _cap(value)
    try:
        return _cap(json.dumps(value, default=str))
    except Exception:
        return _cap(str(value))


def span_attributes(span_type: str, **attrs: Any) -> dict[str, Any]:
    """Attributes every UiPath span carries, plus ``attrs`` (None dropped)."""
    base: dict[str, Any] = {
        "type": span_type,
        "span_type": span_type,
        "uipath.custom_instrumentation": True,
        "executionType": 0 if os.environ.get("UIPATH_IS_DEBUG") == "true" else 1,
    }
    version = os.environ.get("UIPATH_PROCESS_VERSION")
    if version:
        base["agentVersion"] = version
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            base[key] = _cap(value) if isinstance(value, str) else value
        else:
            base[key] = _json(value)
    return base


def attribute_values(**attrs: Any) -> dict[str, Any]:
    """``attrs`` as span attribute values (None dropped, the rest JSON text)."""
    return {
        key: value if isinstance(value, (bool, int, float)) else _json(value)
        for key, value in attrs.items()
        if value is not None
    }


def set_attributes(span: Span, **attrs: Any) -> None:
    for key, value in attribute_values(**attrs).items():
        span.set_attribute(key, value)


def fail_span(span: Span, error: BaseException) -> None:
    # ``CancelledError()`` and the like have no message: still an error.
    text = str(error) or type(error).__name__
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, text))
    span.set_attribute("error", _cap(text))


# --- agent run ----------------------------------------------------------------


def start_agent_run(
    name: str,
    *,
    agent_name: str,
    model: str,
    input: Any,
    system_prompt: str | None,
    user_prompt: str | None,
    input_schema: Any,
    output_schema: Any,
    resumed: bool,
    attachments: str | None = None,
) -> Span:
    return get_tracer().start_span(
        f"Agent run - {name}",
        attributes=span_attributes(
            SPAN_AGENT_RUN,
            agentName=agent_name,
            source="runtime",
            isConversational=False,
            runtimeKind="coded",
            framework="codex_sdk",
            model=model,
            systemPrompt=system_prompt or "",
            userPrompt=user_prompt or "",
            input=input if input is not None else {},
            inputSchema=input_schema,
            outputSchema=output_schema,
            resumed=resumed,
            attachments=attachments,
        ),
    )


def emit_agent_output(
    output: Any, *, context: Context | None = None, **attrs: Any
) -> None:
    """A zero-length ``Agent output`` span under ``context`` (or the current span).

    ``attrs`` are extra attributes, e.g. the ``execution.id`` a non-recording
    (restored) parent cannot pass down.
    """
    span = get_tracer().start_span(
        "Agent output",
        context=context,
        attributes=span_attributes(SPAN_AGENT_OUTPUT, output=output, **attrs),
    )
    span.set_status(Status(StatusCode.OK))
    span.end()


# --- spans across executions --------------------------------------------------


def span_record(span: Span) -> dict[str, Any] | None:
    """What it takes to complete ``span`` from another process (JSON-safe).

    None for a span that records nothing (a restored or non-sampled one).
    """
    ctx = span.get_span_context()
    start = getattr(span, "start_time", None)
    if not span.is_recording() or not ctx.is_valid or start is None:
        return None
    parent = getattr(span, "parent", None)
    attributes = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in (getattr(span, "attributes", None) or {}).items()
    }
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "name": getattr(span, "name", ""),
        "start_time_ns": start,
        "attributes": attributes,
    }


def _span_context(trace_id: str, span_id: str, *, remote: bool) -> SpanContext:
    # Sampled: the default sampler drops every child of an unsampled remote
    # parent, and the batch processors drop an unsampled span.
    return SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        is_remote=remote,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


#: Span attributes the UiPath span processors hand from parent to child.
PROPAGATED_ATTRIBUTES = ("execution.id",)


#: Propagated attributes (``execution.id``) of the resumed execution running in
#: this context; see :func:`resumed_attributes`.
_RESUMED_ATTRIBUTES: ContextVar[dict[str, Any] | None] = ContextVar(
    "uipath_codex_resumed_attributes", default=None
)
_INSTALLED: weakref.WeakSet[Any] = weakref.WeakSet()


def propagated_attributes() -> dict[str, Any]:
    """The current span's propagated attributes (e.g. ``execution.id``).

    ``UiPathExecutionTraceProcessorMixin`` copies them only from a recording
    parent, so the children of a restored (non-recording) span and a span
    completed from a record need them passed explicitly.
    """
    span = trace.get_current_span()
    attributes = getattr(span, "attributes", None) if span.is_recording() else None
    if not attributes:
        # Under a restored Agent run: the resumed execution's own attributes.
        return dict(_RESUMED_ATTRIBUTES.get() or {})
    return {
        key: attributes[key] for key in PROPAGATED_ATTRIBUTES if attributes.get(key)
    }


class _ResumedAttributesProcessor(SpanProcessor):
    """Copies a resumed execution's propagated attributes onto its spans.

    They sit under the restored, non-recording ``Agent run``, which
    ``UiPathExecutionTraceProcessorMixin`` cannot copy ``execution.id`` from.
    """

    def on_start(self, span: Any, parent_context: Context | None = None) -> None:
        attributes = _RESUMED_ATTRIBUTES.get()
        if not attributes:
            return
        present = span.attributes or {}
        for key, value in attributes.items():
            if key not in present:
                span.set_attribute(key, value)


def resumed_attributes(attributes: dict[str, Any]) -> Token[dict[str, Any] | None]:
    """Give every span started in this context ``attributes`` (until reset)."""
    provider = trace.get_tracer_provider()
    add = getattr(provider, "add_span_processor", None)
    if attributes and add is not None and provider not in _INSTALLED:
        add(_ResumedAttributesProcessor())
        _INSTALLED.add(provider)
    return _RESUMED_ATTRIBUTES.set(dict(attributes) or None)


def reset_resumed_attributes(token: Token[dict[str, Any] | None]) -> None:
    try:
        _RESUMED_ATTRIBUTES.reset(token)
    except ValueError:  # reset from another context: that context is gone
        pass


def restored_span(record: dict[str, Any]) -> Span:
    """The saved span as a parent for this execution's spans (records nothing)."""
    return NonRecordingSpan(
        _span_context(record["trace_id"], record["span_id"], remote=True)
    )


def _live_trackers() -> list[Any]:
    """The ``LiveTrackingSpanProcessor``s on the tracer provider (LLMOps export).

    ``uipath run`` adds one only for a job with tracing enabled, after the
    runtime is built, so they are looked up when needed.
    """
    try:
        from uipath.tracing import LiveTrackingSpanProcessor
    except Exception:  # pragma: no cover - uipath always ships it
        return []
    found: list[Any] = []
    stack = [getattr(trace.get_tracer_provider(), "_active_span_processor", None)]
    while stack:
        processor = stack.pop()
        if processor is None:
            continue
        if isinstance(processor, LiveTrackingSpanProcessor):
            found.append(processor)
            continue
        for attr in ("_span_processors", "_processors"):
            stack.extend(getattr(processor, attr, None) or ())
    return found


def push_running(span: Span) -> None:
    """Upsert ``span`` to LLMOps as still running, with its current attributes.

    For a span that stays open across a suspension: ending it would export it
    as finished. Nothing happens without live tracking (a local run).
    """
    try:
        from uipath.platform.common._span_utils import SpanStatus
    except Exception:  # pragma: no cover
        return
    for processor in _live_trackers():
        span_filter = getattr(processor, "span_filter", None)
        if span_filter is not None and not span_filter(span):
            continue
        try:
            # The processor's own pool: non-blocking, and drained at shutdown.
            processor._upsert_span_async(span, status_override=SpanStatus.UNSET)
        except Exception:
            logger.debug("could not push the running span", exc_info=True)


def complete_span(
    record: dict[str, Any],
    *,
    attributes: dict[str, Any] | None = None,
    error: str | None = None,
    drop: tuple[str, ...] = (),
) -> None:
    """End a span saved by an earlier execution: same id and start, end now.

    ``error`` (even an empty one) completes it with ERROR status. ``drop``
    names saved attributes that no longer apply (e.g. a suspension's
    ``output`` on a faulted run).

    The finished span goes through the provider's span processors, as
    ``Span.end()`` would send it: LLMOps live tracking, ``--trace-file`` and
    any other exporter get it once.
    """
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    provider = trace.get_tracer_provider()
    processor = getattr(provider, "_active_span_processor", None)
    if processor is None:
        return
    try:
        saved = {
            key: value
            for key, value in (record.get("attributes") or {}).items()
            if key not in drop
        }
        final = {**saved, **(attributes or {})}
        if error is not None:
            final["error"] = _cap(error)
        parent = record.get("parent_span_id")
        span = ReadableSpan(
            name=str(record["name"]),
            context=_span_context(record["trace_id"], record["span_id"], remote=False),
            parent=(
                _span_context(record["trace_id"], parent, remote=False)
                if parent
                else None
            ),
            resource=getattr(provider, "resource", None),
            attributes=final,
            kind=SpanKind.INTERNAL,
            status=(
                Status(StatusCode.ERROR, error)
                if error is not None
                else Status(StatusCode.OK)
            ),
            start_time=int(record["start_time_ns"]),
            end_time=time.time_ns(),
            instrumentation_scope=InstrumentationScope(TRACER_NAME),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("Ignoring an unusable saved span record", exc_info=True)
        return
    processor.on_end(span)


# --- tool calls ---------------------------------------------------------------


def start_tool_call(
    tool_name: str,
    *,
    tool_type: str,
    call_id: str | None,
    arguments: Any,
    context: Context | None = None,
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``Tool call - <tool>`` span (exported as a ``toolCall``)."""
    args_text = _json(arguments if arguments is not None else {})
    return (tracer or get_tracer()).start_span(
        f"Tool call - {tool_name}",
        context=context,
        attributes=span_attributes(
            SPAN_TOOL_CALL,
            **{
                "tool.name": tool_name,
                "toolName": tool_name,
                "toolType": tool_type,
                "call_id": call_id,
                "input": args_text,
                "input.value": args_text,
            },
            **attrs,
        ),
    )


def _tool_output(output: Any, content: str | None = None) -> dict[str, str]:
    """``output`` (the full value, LLMOps' ``result``) and ``output.value``.

    ``output.value`` is ``{"content": <text>}``, which ``uipath.eval``'s
    tool-output evaluators read; ``content`` defaults to ``output`` as text.
    """
    if content is None:
        content = _json(output)
    return {
        "output": _json(output),
        "output.value": json.dumps({"content": _cap(content)}),
    }


def set_tool_output(span: Span, output: Any, *, content: str | None = None) -> None:
    for key, value in _tool_output(output, content).items():
        span.set_attribute(key, value)


def end_tool_call(
    span: Span,
    *,
    output: Any = None,
    error: str | None = None,
    content: str | None = None,
) -> None:
    if output is not None:
        set_tool_output(span, output, content=content)
    if error:
        span.set_status(Status(StatusCode.ERROR, error))
        span.set_attribute("error", _cap(error))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


#: Child spans a durable tool leaves open across the suspension (see
#: :func:`hold_span`): collected while ``start`` runs, saved with the call.
_held_children: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "codex_held_children", default=None
)


def collect_held_spans() -> Token[list[dict[str, Any]] | None]:
    """Start collecting :func:`hold_span` records (the tool host, around ``start``)."""
    return _held_children.set([])


def held_spans(token: Token[list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    """The records collected since ``token``; stops collecting."""
    records = _held_children.get() or []
    _held_children.reset(token)
    return records


def hold_span(
    span: Span,
    *,
    output_attribute: str = "output",
    output_keys: tuple[str, ...] | None = None,
    result_attachment_key: str | None = None,
) -> bool:
    """Leave ``span`` open until the durable call's result is used.

    For a child span of a durable ``Tool call`` (e.g. a ``contextGroundingTool``
    span around a DeepRAG run): it is pushed to LLMOps as running and saved
    with the call, and the execution that uses the result completes it with
    that result. Returns False, and ends the span, outside a durable start.

    ``output_attribute`` names the attribute the result goes into (low-code
    escalations use ``result``) and ``output_keys``, when set, keeps only
    those keys of a dict result (``outcome``, ``output``).
    ``result_attachment_key`` names the result's attachment field (Batch
    Transform's ``result``): the span then also gets low-code's ``results``
    (``{id, fileName, mimeType}``) and the attachment as ``out``.
    """
    records = _held_children.get()
    record = span_record(span) if records is not None else None
    if records is None or record is None:
        span.set_status(Status(StatusCode.OK))
        span.end()
        return False
    push_running(span)
    record["output_attribute"] = output_attribute
    if output_keys:
        record["output_keys"] = list(output_keys)
    if result_attachment_key:
        record["result_attachment_key"] = result_attachment_key
    records.append(record)
    return True


def _only_keys(value: Any, keys: Any) -> Any:
    """``value`` with only ``keys`` when it is a dict and keys are given."""
    if not keys or not isinstance(value, dict):
        return value
    return {k: value.get(k) for k in keys}


def _result_attachment_attributes(
    record: dict[str, Any], output: Any
) -> dict[str, Any]:
    """Low-code's ``results`` and the ``out`` attachment of a result attachment."""
    from ..attachments import merge_span_attachments, span_attachments, to_attachment

    key = record.get("result_attachment_key")
    value = output.get(key) if key and isinstance(output, dict) else None
    attachment = to_attachment(value) if value else None
    if attachment is None:
        return {}
    merged = merge_span_attachments(
        (record.get("attributes") or {}).get("attachments"),
        span_attachments([attachment], "out"),
    )
    return {
        "results": _json(
            {
                "id": str(attachment.id),
                "fileName": attachment.full_name,
                "mimeType": attachment.mime_type,
            }
        ),
        **({"attachments": merged} if merged else {}),
    }


def complete_tool_call(
    record: dict[str, Any],
    *,
    output: Any = None,
    error: str | None = None,
    attachments: str | None = None,
) -> None:
    """End a ``Tool call`` span saved when its job started (see :func:`complete_span`).

    It joins the current execution (``execution.id``), which reads its result.
    ``attachments`` (the result's, ``out``) are added to the ``attachments``
    the call started with.
    """
    from ..attachments import merge_span_attachments

    merged = merge_span_attachments(
        (record.get("attributes") or {}).get("attachments"), attachments
    )
    # Children first (inner before outer), with the same result.
    for child in record.get("children") or []:
        if isinstance(child, dict):
            attribute = str(child.get("output_attribute") or "output")
            child_output = _only_keys(output, child.get("output_keys"))
            complete_span(
                child,
                attributes={
                    **propagated_attributes(),
                    **({attribute: _json(child_output)} if output is not None else {}),
                    **_result_attachment_attributes(child, output),
                },
                error=error,
            )
    # ``output`` may keep only some keys (``output_keys``); ``output.value``
    # keeps the full text the model received (the tool-output evaluators').
    shown = _only_keys(output, record.get("output_keys"))
    complete_span(
        {k: v for k, v in record.items() if k != "children"},
        attributes={
            **propagated_attributes(),
            **(
                _tool_output(shown, content=_json(output)) if output is not None else {}
            ),
            **({"attachments": merged} if attachments and merged else {}),
        },
        error=error,
    )


def start_integration_tool_span(
    display_name: str,
    *,
    arguments: Any,
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<display name>`` span (``integrationTool``) under the current span.

    The low-code ``IntegrationToolSpanAttributes`` (``toolName``,
    ``arguments``, ``result``): the Integration Service request of a
    ``Tool call``. The SDK's own ``activity_invoke`` span nests under it.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_INTEGRATION_TOOL,
            toolName=display_name,
            arguments=_json(arguments if arguments is not None else {}),
            **attrs,
        ),
    )


def end_integration_tool_span(
    span: Span, *, result: Any = None, error: BaseException | str | None = None
) -> None:
    if result is not None:
        span.set_attribute("result", _json(result))
    if isinstance(error, BaseException):
        fail_span(span, error)
    elif error:
        span.set_status(Status(StatusCode.ERROR, error))
        span.set_attribute("error", _cap(error))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def start_internal_tool_span(
    display_name: str,
    *,
    arguments: Any,
    tool_type: str = "Internal",
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<display name>`` span of a low-code internal tool, under the ``Tool call``.

    A ``toolCall`` span (low-code ``InternalToolSpanAttributes``: ``toolName``,
    ``input``, ``output``). ``tool.name`` and ``toolType`` are set too: the
    LLMOps exporter reads them for every ``toolCall`` span (``toolType``
    defaults to Integration). ``tool.synthesized`` keeps the tool-call
    evaluators from counting it as a second call: the call is the parent
    ``Tool call`` span.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_TOOL_CALL,
            toolName=display_name,
            toolType=tool_type,
            **{"tool.name": display_name, "tool.synthesized": True},
            input=_json(arguments if arguments is not None else {}),
            **{"input.mime_type": "application/json"},
            **attrs,
        ),
    )


def start_context_grounding_span(
    display_name: str,
    *,
    retrieval_mode: str,
    query: str,
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<display name>`` span (``contextGroundingTool``) under the current span.

    The low-code ``ContextGroundingToolSpanAttributes`` (``retrieval_mode``,
    ``query``, ``threshold``, ``number_of_results``, ``folder_path_prefix``,
    ``file_extension``, ``system_index``) of a ``Tool call``; its ``output`` is
    the documents. The search runs under a :func:`start_retriever_span` child.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_CONTEXT_GROUNDING_TOOL,
            retrieval_mode=retrieval_mode,
            query=query,
            **attrs,
        ),
    )


def start_ixp_tool_span(
    display_name: str,
    *,
    arguments: Any,
    project_name: str,
    version_tag: str,
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<tool name>`` span (``ixpTool``) of an IXP extraction.

    The low-code ``IxpToolSpanAttributes``: ``arguments``, ``projectName``,
    ``versionTag``, then ``extractionId`` and ``documentId``; it closes with
    ``result``.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_IXP_TOOL,
            arguments=arguments if arguments is not None else {},
            projectName=project_name,
            versionTag=version_tag,
            **attrs,
        ),
    )


def start_vs_escalation_tool_span(
    display_name: str,
    *,
    ixp_tool_id: str,
    storage_bucket_name: str | None,
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<name>`` span (``vsEscalationTool``) of a Document Validation task.

    The low-code ``VsEscalationToolSpanAttributes``: ``arguments``,
    ``ixpToolId``, ``storageBucketName``, then ``operationId`` and
    ``taskUrl``; it closes with ``result``.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_VS_ESCALATION_TOOL,
            arguments={},
            ixpToolId=ixp_tool_id,
            storageBucketName=storage_bucket_name,
            **attrs,
        ),
    )


def start_process_tool_span(
    process_name: str,
    *,
    arguments: Any,
    kind: str = "process",
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<process name>`` span (``processTool``, or ``agentTool`` for an agent).

    The low-code ``ProcessToolSpanAttributes`` (``arguments``, then the
    ``output``). Only a simulated process call has one: a real call's job is
    linked to the ``Tool call`` span itself (its ``ParentSpanId``).
    """
    span_type = SPAN_AGENT_TOOL if kind == "agent" else SPAN_PROCESS_TOOL
    extra: dict[str, Any] = {"toolName": process_name} if kind == "agent" else {}
    return (tracer or get_tracer()).start_span(
        process_name,
        attributes=span_attributes(
            span_type,
            arguments=arguments if arguments is not None else {},
            **extra,
            **attrs,
        ),
    )


def start_escalation_tool_span(
    display_name: str,
    *,
    arguments: Any,
    channel_type: str = "actionCenter",
    tracer: trace.Tracer | None = None,
    **attrs: Any,
) -> Span:
    """``<app name>`` span (``escalationTool``) under the current span.

    The low-code ``EscalationToolSpanAttributes``: ``arguments`` (the task
    data), ``channelType``, and once the task exists ``taskId``, ``taskUrl``
    and ``assignedTo``; it closes with ``result`` = ``{outcome, output}``.
    """
    return (tracer or get_tracer()).start_span(
        display_name,
        attributes=span_attributes(
            SPAN_ESCALATION_TOOL,
            arguments=arguments if arguments is not None else {},
            channelType=channel_type,
            **attrs,
        ),
    )


def end_context_grounding_span(
    span: Span, *, result: str | None = None, error: BaseException | None = None
) -> None:
    """End it with ``result`` (the documents JSON, kept whole) or ``error``."""
    if result is not None:
        span.set_attribute("output", result)
    _end(span, error)


def start_retriever_span(query: str, *, tracer: trace.Tracer | None = None) -> Span:
    """``ContextGroundingRetriever`` span (OpenInference ``RETRIEVER``).

    The span langchain's OpenInference instrumentation records for its
    ``ContextGroundingRetriever``: ``input.value`` is the query and
    ``output.value`` ``{"documents": [...]}``, which the context precision
    and faithfulness evaluators read. It is separate from the
    ``contextGroundingTool`` span because the LLMOps exporter takes a span's
    ``SpanType`` from ``openinference.span.kind`` when it has one.
    """
    return (tracer or get_tracer()).start_span(
        "ContextGroundingRetriever",
        attributes={
            "openinference.span.kind": "RETRIEVER",
            "input.value": _cap(query),
            "input.mime_type": "text/plain",
        },
    )


def end_retriever_span(
    span: Span, *, result: str | None = None, error: BaseException | None = None
) -> None:
    """End it with ``result`` (the documents JSON, kept whole) or ``error``."""
    if result is not None:
        span.set_attribute("output.value", result)
        span.set_attribute("output.mime_type", "application/json")
    _end(span, error)


def _end(span: Span, error: BaseException | None) -> None:
    if error is not None:
        fail_span(span, error)
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def item_tool(item: dict[str, Any]) -> tuple[str, str, Any] | None:
    """``(tool name, tool type, arguments)`` of a Codex tool item, else None."""
    itype = item.get("type")
    if itype not in TOOL_ITEM_TYPES:
        return None
    if itype == "mcpToolCall":
        return (item.get("tool") or "tool", "Mcp", item.get("arguments"))
    if itype == "dynamicToolCall":
        return (item.get("tool") or "tool", "Internal", item.get("arguments"))
    if itype == "commandExecution":
        return ("shell", "Internal", {"command": item.get("command")})
    if itype == "fileChange":
        return ("apply_patch", "Internal", {"changes": item.get("changes")})
    return ("web_search", "Internal", {"query": item.get("query")})


def item_tool_result(item: dict[str, Any]) -> tuple[Any, str | None]:
    """``(output, error)`` of a completed Codex tool item."""
    itype = item.get("type")
    status = str(item.get("status") or "")
    error: str | None = None
    if itype == "mcpToolCall":
        output = item.get("result")
        err = item.get("error")
        if err:
            error = err.get("message") if isinstance(err, dict) else str(err)
    elif itype == "commandExecution":
        output = {
            "output": item.get("aggregated_output"),
            "exit_code": item.get("exit_code"),
        }
    else:
        output = {k: v for k, v in item.items() if k not in ("id", "type")}
    if error is None and status in ("failed", "declined"):
        error = f"{itype} {status}"
    return output, error


def _change_line(change: Any) -> str:
    if not isinstance(change, dict):
        return str(change)
    kind = change.get("kind")
    kind = kind if isinstance(kind, dict) else {"type": kind}
    line = f"{kind.get('type') or 'update'} {change.get('path') or ''}".rstrip()
    return f"{line} -> {kind['move_path']}" if kind.get("move_path") else line


def item_tool_text(item: dict[str, Any]) -> str | None:
    """Text result of a completed Codex tool item (``output.value``'s content).

    Shell output, an MCP result's text blocks, a summary of the file changes;
    None for the full output as JSON.
    """
    itype = item.get("type")
    if itype == "commandExecution":
        return item.get("aggregated_output") or ""
    if itype == "mcpToolCall":
        result = item.get("result")
        if isinstance(result, dict) and (text := _content_text(result.get("content"))):
            return text
        return None
    if itype == "dynamicToolCall":
        return _content_text(item.get("content_items")) or None
    if itype == "fileChange":
        return "\n".join(_change_line(c) for c in item.get("changes") or [])
    if itype == "webSearch":
        return None if item.get("results") else item.get("query") or ""
    return None


def _item_attributes(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") == "mcpToolCall" and item.get("server"):
        return {"tool.server": item["server"], "mcp.server": item["server"]}
    return {}


class ItemSpans:
    """Open ``Tool call`` spans of the Codex items still running."""

    def __init__(self) -> None:
        self._open: dict[str, Span] = {}

    def start(self, item: dict[str, Any]) -> None:
        described = item_tool(item)
        item_id = item.get("id")
        if described is None or not item_id or item_id in self._open:
            return
        name, tool_type, arguments = described
        self._open[item_id] = start_tool_call(
            name,
            tool_type=tool_type,
            call_id=item_id,
            arguments=arguments,
            **_item_attributes(item),
        )

    def finish(self, item: dict[str, Any]) -> None:
        span = self._open.pop(item.get("id") or "", None)
        if span is None:
            return
        output, error = item_tool_result(item)
        end_tool_call(span, output=output, error=error, content=item_tool_text(item))

    def close_all(self, reason: str) -> None:
        for span in self._open.values():
            span.set_attribute("status", reason)
            span.end()
        self._open.clear()


# --- model calls --------------------------------------------------------------


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("output") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(parts)


def latest_input_text(payload: dict[str, Any]) -> str:
    """Text of the newest Responses ``input`` item that has any."""
    items = payload.get("input")
    if isinstance(items, str):
        return _cap(items)
    if not isinstance(items, list):
        return ""
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            text = _content_text(item.get("output"))
        else:
            text = _content_text(item.get("content"))
        if text:
            return _cap(text)
    return ""


def has_output_schema(payload: dict[str, Any]) -> bool:
    """Whether a Responses request forces a JSON schema on the output text."""
    text = payload.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    return isinstance(fmt, dict) and fmt.get("type") == "json_schema"


def response_summary(
    response: dict[str, Any], *, schema_output: bool = False
) -> dict[str, Any]:
    """Output text, tool calls and usage of a Responses ``response`` object.

    With ``schema_output`` (the request forced a JSON schema), the text of a
    response that also calls tools is dropped: the model fills the schema
    with placeholders (``{"issue_key": "pending"}``) to go with its tool
    calls, and only the final response carries the agent's output.
    """
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    texts.append(block.get("text") or "")
        elif itype in ("function_call", "custom_tool_call"):
            tool_calls.append(
                {
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or item.get("input"),
                }
            )
    usage = response.get("usage") or {}
    if schema_output and tool_calls:
        texts = []
    summary: dict[str, Any] = {"text": "\n".join(t for t in texts if t)}
    if tool_calls:
        summary["toolCalls"] = tool_calls
    if usage:
        summary["usage"] = {
            "promptTokens": usage.get("input_tokens"),
            "completionTokens": usage.get("output_tokens"),
            "totalTokens": usage.get("total_tokens"),
        }
    return summary


class SseCompletedReader:
    """Rebuilds the final ``response`` from a relayed Responses SSE stream.

    Takes the ``response.completed`` event, and fills its ``output`` from the
    ``response.output_item.done`` events when the event carries none (Codex
    itself builds the turn from those). Only the event being received is
    buffered, plus the finished output items.
    """

    _EVENTS = (b"response.completed", b"response.done", b"response.output_item.done")

    def __init__(self) -> None:
        self._buffer = b""
        self._items: list[dict[str, Any]] = []
        self._completed: dict[str, Any] | None = None

    @property
    def response(self) -> dict[str, Any] | None:
        if self._completed is None:
            return None
        if self._completed.get("output") or not self._items:
            return self._completed
        return {**self._completed, "output": list(self._items)}

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            self._parse(event)

    def _parse(self, event: bytes) -> None:
        data = b"\n".join(
            line[5:].lstrip()
            for line in event.split(b"\n")
            if line.startswith(b"data:")
        )
        if not any(name in data for name in self._EVENTS):
            return
        try:
            parsed = json.loads(data)
        except ValueError:
            return
        if not isinstance(parsed, dict):
            return
        if parsed.get("type") == "response.output_item.done":
            item = parsed.get("item")
            if isinstance(item, dict):
                self._items.append(item)
        elif isinstance(parsed.get("response"), dict):
            self._completed = parsed["response"]


class ModelCallSpan:
    """The ``LLM call`` span of one relayed ``/responses`` request."""

    def __init__(self, parent: Context | None, payload: dict[str, Any], model: str):
        self._schema_output = has_output_schema(payload)
        self.span = get_tracer().start_span(
            "LLM call",
            context=parent,
            attributes=span_attributes(
                SPAN_LLM_CALL, input=latest_input_text(payload), model=model
            ),
        )

    def trace_headers(self) -> dict[str, str]:
        """Headers that let the LLM Gateway add its ``Model run`` span here."""
        token = context_api.attach(trace.set_span_in_context(self.span))
        try:
            from uipath.platform.chat.llm_trace_context import (
                build_trace_context_headers,
            )

            return build_trace_context_headers(extra_baggage=[TRACE_SOURCE])
        except Exception:
            logger.debug("could not build trace context headers", exc_info=True)
            return {}
        finally:
            context_api.detach(token)

    def finish(
        self,
        *,
        response: dict[str, Any] | None = None,
        status: int | None = None,
        error: str | None = None,
    ) -> None:
        if response is not None:
            summary = response_summary(response, schema_output=self._schema_output)
            set_attributes(
                self.span,
                output=summary.get("text"),
                content=summary.get("text"),
                toolCalls=summary.get("toolCalls"),
                usage=summary.get("usage"),
            )
        if status is not None:
            self.span.set_attribute("http.status_code", status)
        if error:
            self.span.set_status(Status(StatusCode.ERROR, error))
            self.span.set_attribute("error", _cap(error))
        else:
            self.span.set_status(Status(StatusCode.OK))
        self.span.end()


__all__ = [
    "ItemSpans",
    "attribute_values",
    "ModelCallSpan",
    "SseCompletedReader",
    "TOOL_ITEM_TYPES",
    "TRACER_NAME",
    "complete_span",
    "complete_tool_call",
    "emit_agent_output",
    "enable_gateway_model_spans",
    "end_tool_call",
    "fail_span",
    "get_tracer",
    "item_tool_text",
    "latest_input_text",
    "push_running",
    "response_summary",
    "restored_span",
    "set_attributes",
    "set_tool_output",
    "span_attributes",
    "span_record",
    "start_agent_run",
    "reset_resumed_attributes",
    "resumed_attributes",
    "start_tool_call",
]
