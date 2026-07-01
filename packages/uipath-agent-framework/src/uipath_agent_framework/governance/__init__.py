"""Governance integration for ``uipath-agent-framework``.

Exposes :func:`install_governance` — appends governance middleware
(:class:`GovernanceChatMiddleware` + :class:`GovernanceFunctionMiddleware`) to
each ``agent_framework`` agent's ``middleware`` list, governing model/tool
events (BEFORE_MODEL, AFTER_MODEL, TOOL_CALL, AFTER_TOOL). Wired into a run by
passing an ``evaluator`` to :class:`UiPathAgentFrameworkRuntimeFactory`; the
factory calls :func:`install_governance` on the resolved agent.

Importing this module has no side effects: no adapter is registered, no global
state is mutated.
"""

from __future__ import annotations

from .middleware import (
    GovernanceCallbacks,
    GovernanceChatMiddleware,
    GovernanceFunctionMiddleware,
    install_governance,
)

__all__ = [
    "GovernanceCallbacks",
    "GovernanceChatMiddleware",
    "GovernanceFunctionMiddleware",
    "install_governance",
]