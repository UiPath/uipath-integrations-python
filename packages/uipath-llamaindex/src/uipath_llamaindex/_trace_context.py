"""LLM Gateway trace context for UiPath LlamaIndex models.

By default every UiPath model sends the caller's trace and span to the LLM
Gateway, which records the model call itself ("Model run", with its governance
and guardrail spans) inside the agent's trace. The client-side LLM span would
duplicate it, so the runtime doesn't create one.

``UIPATH_USE_CLIENT_LLM_SPANS=true`` turns that around, for agents whose models
don't go through the Gateway: the client-side LLM span is kept and the
traceparent isn't sent, so the Gateway's spans stay out of the trace. The
baggage (folder / agent / job attribution) is still sent.
"""

import logging
import os

from uipath.platform.chat.llm_trace_context import build_trace_context_headers

logger = logging.getLogger(__name__)

USE_CLIENT_LLM_SPANS_ENV = "UIPATH_USE_CLIENT_LLM_SPANS"
TRACEPARENT_HEADER = "x-uipath-traceparent-id"
_TRACE_SOURCE = "source=llamaindex"


def use_client_llm_spans() -> bool:
    """Whether the client-side LLM span is kept instead of the Gateway's."""
    return os.environ.get(USE_CLIENT_LLM_SPANS_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def trace_context_headers() -> dict[str, str]:
    """Headers for one LLM Gateway request, built from the active span."""
    try:
        headers = build_trace_context_headers(extra_baggage=[_TRACE_SOURCE])
    except Exception:
        logger.debug("could not build trace context headers", exc_info=True)
        return {}
    if use_client_llm_spans():
        headers.pop(TRACEPARENT_HEADER, None)
    return headers
