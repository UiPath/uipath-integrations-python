"""Governance integration for ``uipath-google-adk``.

Exposes :func:`install_governance` — installs governance callbacks
(BEFORE_MODEL, AFTER_MODEL, TOOL_CALL, AFTER_TOOL) on every ``LlmAgent`` in an
ADK agent tree's native ``*_callback`` slots. Wired into a run by passing an
``evaluator`` to :class:`UiPathGoogleADKRuntimeFactory`; the factory calls
:func:`install_governance` on the resolved agent.

Importing this module has no side effects: no adapter is registered, no global
state is mutated.
"""

from __future__ import annotations

from .callbacks import GovernanceCallbacks, install_governance

__all__ = [
    "GovernanceCallbacks",
    "install_governance",
]