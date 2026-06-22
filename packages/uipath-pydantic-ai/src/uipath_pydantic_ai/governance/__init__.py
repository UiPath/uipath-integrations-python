"""Governance integration for ``uipath-pydantic-ai``.

Registers :class:`PydanticAIAdapter` with the global adapter registry in
``uipath.core.adapters`` so ``uipath.runtime.governance.GovernanceRuntime`` can
attach the Pydantic-AI-specific governance (BEFORE_MODEL, AFTER_MODEL,
TOOL_CALL, AFTER_TOOL) when it sees a ``pydantic_ai.Agent``.

Registration is **idempotent**: calling :func:`register_governance_adapter`
twice is a no-op on the second call.

Wiring:
    1. Importing this module triggers registration as a side-effect, so any
       caller that does ``import uipath_pydantic_ai.governance`` is opted in.
    2. The package also exposes :func:`register_governance_adapter` as an entry
       point under ``uipath.governance.adapters`` so the registry's entry-point
       discovery can plug us in without an explicit import.
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


# Side-effect registration on module import.
register_governance_adapter()


__all__ = [
    "GovernanceCallbacks",
    "GovernanceModel",
    "PydanticAIAdapter",
    "register_governance_adapter",
]