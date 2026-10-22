"""Eval tool mocking: which calls the active mocker answers, and with what.

``uipath eval`` (and ``uipath run`` / ``uipath debug`` with a
``simulation.json``) wraps the runtime in ``UiPathMockRuntime``, which puts a
mocking context and a mocker in ContextVars. The tool host snapshots that
context when the run starts, so the checks below see it.

* :func:`is_simulated` tells whether the active mocker will answer a mock
  name, with the mockers' own match rules (exact for the LLM and mockito
  mockers; exact or ``_`` → space for simulated components). The public
  ``is_tool_simulated`` normalizes ``_`` for every strategy, which would let
  ``Get_User`` through while the LLM mocker refuses it.
* :func:`mock_spec_of` is what the mocker is told about a tool: its mock name
  (LangChain's ``resource.name``; see :func:`mock_name_of`), description,
  schemas and example calls. A built-in tool defines ``mock_spec`` itself; a
  custom tool gets one from its attributes.
* :func:`wrap` is ``uipath.eval.mocks.mockable`` with a spec.

A simulated durable call is answered in the same turn by the tool host: no
job, task or extraction starts and the run does not suspend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The span attribute ``mockable`` sets on the current span of a mocked call.
MOCKED_ANNOTATION_KEY = "__uipath_response_mocked"
#: Prefix of a mocked child span's name (low-code's).
SIMULATED_PREFIX = "Simulated result: "

#: Output schema of a tool that declares none.
ANY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True)
class MockSpec:
    """What the mocker is told about a tool (``mockable``'s parameters)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    example_calls: list[Any] = field(default_factory=list)


@dataclass
class MockSpan:
    """The child span a mocked durable call runs under (``start_mock_span``).

    ``output_attribute`` names the attribute the result goes into and
    ``output_keys``, when set, keeps only those keys of a dict result (the
    same as the span's ``hold_span`` on a real call).
    """

    span: Any
    output_attribute: str = "output"
    output_keys: tuple[str, ...] | None = None


def mock_name_of(tool: Any) -> str:
    """The name a tool is mocked by: ``mock_name``, else the display name, else ``name``.

    From an ``agent.json`` resource that is ``resource.name`` (the factories
    set ``mock_name`` where the display name differs); for a tool declared in
    code, the name passed to the factory. Escalations lowercase their
    model-facing name in their own ``mock_spec``.
    """
    return str(
        getattr(tool, "mock_name", None)
        or getattr(tool, "display_name", None)
        or tool.name
    )


def mock_spec_of(tool: Any) -> MockSpec:
    """``tool.mock_spec`` when it has one, else a spec from its attributes."""
    spec = getattr(tool, "mock_spec", None)
    if isinstance(spec, MockSpec):
        return spec
    output = getattr(tool, "output_json_schema", None)
    examples = getattr(tool, "example_calls", None)
    return MockSpec(
        name=mock_name_of(tool),
        description=str(getattr(tool, "description", "") or ""),
        input_schema=_input_schema(tool),
        output_schema=output
        if isinstance(output, dict) and output
        else ANY_OBJECT_SCHEMA,
        example_calls=list(examples) if isinstance(examples, (list, tuple)) else [],
    )


def _input_schema(tool: Any) -> dict[str, Any]:
    try:
        schema = tool.input_schema
    except Exception:
        return dict(ANY_OBJECT_SCHEMA)
    return schema if isinstance(schema, dict) else dict(ANY_OBJECT_SCHEMA)


def is_simulated(name: str) -> bool:
    """True when the active mocker will answer ``name``.

    The mockers' own match rules: exact for the LLM and mockito strategies,
    exact or after replacing ``_`` with a space for simulated components. A
    mockito strategy without a behavior for ``name`` is False: the real tool
    runs (the mockito mocker would call it itself).
    """
    try:
        from uipath.eval.mocks._mock_context import mocker_context, mocking_context
        from uipath.eval.mocks._types import (
            LLMMockingStrategy,
            MockitoMockingStrategy,
        )
    except Exception:  # pragma: no cover - eval extras missing
        return False
    ctx, mocker = mocking_context.get(), mocker_context.get()
    if ctx is None or mocker is None:
        # set_execution_context leaves the mocker None when it cannot build one.
        return False
    if ctx.components:
        spaced = name.replace("_", " ")
        return any(
            c.component_id == name or c.component_id.replace("_", " ") == spaced
            for c in ctx.components
        )
    strategy = ctx.strategy
    if isinstance(strategy, LLMMockingStrategy):
        return name in [t.name for t in strategy.tools_to_simulate]
    if isinstance(strategy, MockitoMockingStrategy):
        return any(b.function == name for b in strategy.behaviors)
    return False


def wrap(spec: MockSpec, func: Callable[..., Any]) -> Callable[..., Any]:
    """``mockable(...)(func)`` with ``spec``; ``func`` itself without the eval extras."""
    try:
        from uipath.eval.mocks import mockable
    except Exception:  # pragma: no cover - eval extras missing
        return func
    return mockable(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        output_schema=spec.output_schema,
        example_calls=spec.example_calls,
    )(func)


def is_mocked(span: Any) -> bool:
    """True when ``mockable`` marked ``span`` as answered by the mocker."""
    return bool((getattr(span, "attributes", None) or {}).get(MOCKED_ANNOTATION_KEY))


def mark_simulated(span: Any) -> bool:
    """Rename a mocked child span ``Simulated result: <name>`` (low-code's).

    Returns whether it was mocked. The span keeps its type; the ``Tool call``
    span around it is never renamed.
    """
    if not is_mocked(span):
        return False
    name = str(getattr(span, "name", "") or "")
    if not name.startswith(SIMULATED_PREFIX):
        span.update_name(f"{SIMULATED_PREFIX}{name}")
    return True


def mocked_text(value: Any) -> str:
    """The model-facing text of a mocked value (a tool without ``mocked_output``)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


__all__ = [
    "ANY_OBJECT_SCHEMA",
    "MOCKED_ANNOTATION_KEY",
    "MockSpan",
    "MockSpec",
    "SIMULATED_PREFIX",
    "is_mocked",
    "is_simulated",
    "mark_simulated",
    "mock_name_of",
    "mock_spec_of",
    "mocked_text",
    "wrap",
]
