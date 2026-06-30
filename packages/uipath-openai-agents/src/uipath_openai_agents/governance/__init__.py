"""Governance integration for ``uipath-openai-agents``.

Exposes :func:`install_governance` — installs the OpenAI-Agents-specific inner
hooks (BEFORE_MODEL, AFTER_MODEL, TOOL_CALL, AFTER_TOOL) onto an agent's native
``hooks`` slot. Wired into a run by passing an ``evaluator`` to
:class:`UiPathOpenAIAgentRuntimeFactory`; the factory calls
:func:`install_governance` on the resolved agent.

Importing this module has no side effects: no adapter is registered, no global
state is mutated.
"""

from __future__ import annotations

from .hooks import GovernanceAgentHooks, install_governance

__all__ = [
    "GovernanceAgentHooks",
    "install_governance",
]