"""Governance integration for ``uipath-pydantic-ai``.

Registers :class:`PydanticAIAdapter` with the adapter registry in
``uipath.core.adapters`` so the governance host can attach the
Pydantic-AI-specific governance (BEFORE_MODEL, AFTER_MODEL, TOOL_CALL,
AFTER_TOOL) when it sees a ``pydantic_ai.Agent``.

Registration is **idempotent**: calling :func:`register_governance_adapter`
twice is a no-op on the second call.

Wiring: the package exposes :func:`register_governance_adapter` as an entry
point under ``uipath.governance.adapters``. The governance adapter discovery
path calls it to register the adapter. Importing this module does not, by
itself, mutate the global registry.
"""

from __future__ import annotations

import logging

from uipath.core.adapters import get_adapter_registry

from .adapter import GovernanceCallbacks, GovernanceModel, PydanticAIAdapter

logger = logging.getLogger(__name__)

_registered: bool = False


def register_governance_adapter() -> None:
    """Register :class:`PydanticAIAdapter` with the global registry.

    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return
    registry = get_adapter_registry()
    if any(a.name == "PydanticAI" for a in registry.get_all()):
        _registered = True
        return
    registry.register(PydanticAIAdapter())
    _registered = True
    logger.debug("Registered uipath-pydantic-ai governance adapter")


__all__ = [
    "GovernanceCallbacks",
    "GovernanceModel",
    "PydanticAIAdapter",
    "register_governance_adapter",
]