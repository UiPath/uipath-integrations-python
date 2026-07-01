"""Governance integration for ``uipath-llamaindex``.

Exposes :func:`install_governance` — registers a :class:`GovernanceEventHandler`
on the LlamaIndex root instrumentation dispatcher, which governs LLM/tool events
(BEFORE_MODEL, AFTER_MODEL, TOOL_CALL). Wired into a run by passing an
``evaluator`` to :class:`UiPathLlamaIndexRuntimeFactory`; the factory calls
:func:`install_governance`.

Importing this module has no side effects: no adapter is registered, no global
state is mutated.
"""

from __future__ import annotations

from .event_handler import (
    GovernanceEventHandler,
    install_governance,
    uninstall_governance,
)

__all__ = [
    "GovernanceEventHandler",
    "install_governance",
    "uninstall_governance",
]
