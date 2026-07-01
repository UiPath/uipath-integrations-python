"""Governance integration for ``uipath-pydantic-ai``.

Exposes :func:`install_governance` — wraps a ``pydantic_ai.Agent``'s ``model``
with a :class:`GovernanceModel` that brackets every model call with governance
(BEFORE_MODEL, AFTER_MODEL, TOOL_CALL, AFTER_TOOL). Wired into a run by passing
an ``evaluator`` to :class:`UiPathPydanticAIRuntimeFactory`; the factory calls
:func:`install_governance` on the resolved agent.

Importing this module has no side effects: no adapter is registered, no global
state is mutated.
"""

from __future__ import annotations

from .model import GovernanceCallbacks, GovernanceModel, install_governance

__all__ = [
    "GovernanceCallbacks",
    "GovernanceModel",
    "install_governance",
]