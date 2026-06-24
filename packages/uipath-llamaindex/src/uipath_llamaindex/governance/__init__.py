"""Governance integration for ``uipath-llamaindex``.

Registers :class:`LlamaIndexAdapter` with the adapter registry in
``uipath.core.adapters`` so the governance host can attach the
LlamaIndex-specific governance (BEFORE_MODEL, AFTER_MODEL, TOOL_CALL) when it
sees a LlamaIndex workflow/agent.

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

from .adapter import GovernanceEventHandler, LlamaIndexAdapter

logger = logging.getLogger(__name__)

_registered: bool = False


def register_governance_adapter() -> None:
    """Register :class:`LlamaIndexAdapter` with the global registry.

    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return
    registry = get_adapter_registry()
    if any(a.name == "LlamaIndex" for a in registry.get_all()):
        _registered = True
        return
    registry.register(LlamaIndexAdapter())
    _registered = True
    logger.debug("Registered uipath-llamaindex governance adapter")



__all__ = [
    "GovernanceEventHandler",
    "LlamaIndexAdapter",
    "register_governance_adapter",
]