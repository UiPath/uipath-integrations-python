"""Governance integration for ``uipath-google-adk``.

Registers :class:`GoogleADKAdapter` with the adapter registry in
``uipath.core.adapters`` so the governance host can attach the ADK-specific
inner hooks (BEFORE_MODEL, AFTER_MODEL, TOOL_CALL, AFTER_TOOL) when it sees a
Google ADK agent.

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

from .adapter import GoogleADKAdapter, GovernanceCallbacks

logger = logging.getLogger(__name__)

_registered: bool = False


def register_governance_adapter() -> None:
    """Register :class:`GoogleADKAdapter` with the global registry.

    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return
    registry = get_adapter_registry()
    if any(a.name == "GoogleADK" for a in registry.get_all()):
        _registered = True
        return
    registry.register(GoogleADKAdapter())
    _registered = True
    logger.debug("Registered uipath-google-adk governance adapter")


__all__ = [
    "GoogleADKAdapter",
    "GovernanceCallbacks",
    "register_governance_adapter",
]
