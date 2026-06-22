"""Governance integration for ``uipath-openai-agents``.

Registers :class:`OpenAIAgentsAdapter` with the global adapter registry in
``uipath.core.adapters`` so ``uipath.runtime.governance.GovernanceRuntime``
can attach the OpenAI-Agents-specific inner hooks (BEFORE_MODEL, AFTER_MODEL,
TOOL_CALL, AFTER_TOOL) when it sees an OpenAI Agents agent.

Registration is **idempotent**: calling :func:`register_governance_adapter`
twice is a no-op on the second call.

Wiring:
    1. Importing this module triggers registration as a side-effect, so any
       caller that does ``import uipath_openai_agents.governance`` is opted in.
    2. The package also exposes :func:`register_governance_adapter` as an entry
       point under ``uipath.governance.adapters`` so the registry's entry-point
       discovery can plug us in without an explicit import.
"""

from __future__ import annotations

import logging

from uipath.core.adapters import get_adapter_registry

from .adapter import GovernanceAgentHooks, OpenAIAgentsAdapter

logger = logging.getLogger(__name__)

_registered: bool = False


def register_governance_adapter() -> None:
    """Register :class:`OpenAIAgentsAdapter` with the global registry.

    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return
    registry = get_adapter_registry()
    if any(a.name == "OpenAIAgents" for a in registry.get_all()):
        _registered = True
        return
    registry.register(OpenAIAgentsAdapter())
    _registered = True
    logger.debug("Registered uipath-openai-agents governance adapter")


# Side-effect registration on module import.
register_governance_adapter()


__all__ = [
    "GovernanceAgentHooks",
    "OpenAIAgentsAdapter",
    "register_governance_adapter",
]