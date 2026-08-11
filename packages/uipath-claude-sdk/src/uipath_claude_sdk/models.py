"""Routing descriptor for models served by the UiPath LLM Gateway.

A :class:`UiPathModel` declares which tenant model an agent should use.
Attaching one to a :class:`~uipath_claude_sdk.agent.UiPathClaudeAgent` is the
explicit opt-in that makes the runtime start its local gateway shim.

How to route to that model is not declared here. The tenant's discovery response
reports the vendor and the wire format for every model it serves, and the shim
reads the route from there, which is how a model name that works in
``uipath-langchain`` works here too.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["UiPathModel"]


@dataclass
class UiPathModel:
    """A routing descriptor for a UiPath-hosted model.

    This is not a client. It holds no connection, opens no socket and is not
    callable. It only records which model to use, and the runtime reads it when
    it builds the environment for the Claude SDK subprocess.

    Args:
        model: Model id as listed by the tenant's discovery endpoint, for
            example ``"claude-sonnet-4-5"``.
    """

    model: str
