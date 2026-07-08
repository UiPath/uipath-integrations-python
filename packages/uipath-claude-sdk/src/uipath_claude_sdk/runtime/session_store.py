"""Persistence of Claude SDK session ids across conversational exchanges."""

from __future__ import annotations

from uipath.runtime import UiPathRuntimeStorageProtocol

_NAMESPACE = "claude_sdk"
_SESSION_KEY = "session_id"


class ClaudeSessionStore:
    """Stores the Claude SDK session id keyed by runtime (conversation) id.

    The Claude Agent SDK persists full conversation state per session; storing
    the session id lets the next exchange re-attach via
    ``ClaudeAgentOptions(resume=session_id)``.
    """

    def __init__(self, storage: UiPathRuntimeStorageProtocol, runtime_id: str) -> None:
        self._storage = storage
        self._runtime_id = runtime_id

    async def get_session_id(self) -> str | None:
        value = await self._storage.get_value(
            self._runtime_id, _NAMESPACE, _SESSION_KEY
        )
        return value if isinstance(value, str) else None

    async def set_session_id(self, session_id: str) -> None:
        await self._storage.set_value(
            self._runtime_id, _NAMESPACE, _SESSION_KEY, session_id
        )
