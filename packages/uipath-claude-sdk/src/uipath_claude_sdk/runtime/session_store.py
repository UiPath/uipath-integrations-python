"""Persistence of Claude SDK session state across exchanges and suspensions."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from uipath.core.serialization import serialize_object
from uipath.runtime import UiPathRuntimeStorageProtocol

from ..interrupts import INTERRUPT_SUSPEND_MODELS, PendingSuspend

logger = logging.getLogger(__name__)

_NAMESPACE = "claude_sdk"
_SESSION_KEY = "session_id"
_PENDING_SUSPEND_KEY = "pending_suspend"
_ENTRYPOINT_KEY = "entrypoint"
_TRANSCRIPT_KEY = "transcript"


@dataclass(frozen=True)
class TranscriptRecord:
    """The Claude CLI session transcript, carried across a suspension.

    A resumed job is handed a fresh runtime directory, so the CLI's own copy is
    gone and it would start a new session rather than continue the parked one.
    It is stored here because the state database is the one thing the platform
    does carry across.

    Attributes:
        session_id: Session the transcript belongs to.
        project_dir: Name of the directory the CLI filed it under, recorded
            rather than derived. That name is the working directory encoded by
            the CLI, and reimplementing the encoding here would be a second
            copy to drift.
        content: The transcript itself, JSON lines.
    """

    session_id: str
    project_dir: str
    content: str

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> TranscriptRecord | None:
        """Rebuild a record, or None when nothing usable was stored."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.warning("Ignoring a transcript record that is not an object.")
            return None
        try:
            return cls(
                session_id=str(raw["session_id"]),
                project_dir=str(raw["project_dir"]),
                content=str(raw["content"]),
            )
        except KeyError as e:
            logger.warning("Ignoring an incomplete transcript record: %s", e)
            return None


@dataclass(frozen=True)
class EntrypointRecord:
    """The entrypoint a suspended run started on.

    A resumed process refuses to continue a session that was started on a
    different entrypoint.

    Attributes:
        entrypoint: Agent name the run started on.
    """

    entrypoint: str

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> EntrypointRecord | None:
        """Rebuild a record, or None when nothing usable was stored."""
        if raw is None:
            return None
        if not isinstance(raw, dict) or "entrypoint" not in raw:
            logger.warning("Ignoring an unusable entrypoint record.")
            return None
        return cls(entrypoint=str(raw["entrypoint"]))


def _value_type_name(value: Any) -> str:
    """Name the type a resumed process has to rebuild.

    A list value is sibling triggers for one interrupt, so what matters is the
    element type, not that the container was a list.
    """
    if isinstance(value, list) and value:
        return type(value[0]).__name__
    return type(value).__name__


def _pending_to_stored(pending: PendingSuspend) -> dict[str, Any]:
    """Flatten a pending record into JSON the state database can hold.

    The suspend value is whatever the interrupt tool produced, often a pydantic
    model, so it is serialized here. Its type name travels alongside because
    serialization erases the type the trigger creator dispatches on.
    """
    return {
        "interrupt_id": pending.interrupt_id,
        "tool_use_id": pending.tool_use_id,
        "tool_name": pending.tool_name,
        "value_type": _value_type_name(pending.value),
        "value": serialize_object(pending.value),
        "requested_at": pending.requested_at,
    }


def _rebuild_value(stored_type: Any, value: Any, interrupt_id: str) -> Any:
    """Restore the suspend value's type from the name stored beside it.

    ``UiPathResumeTriggerCreator`` picks the resume trigger from the value's
    type alone, and its default for anything unrecognised is an API trigger.
    A run that re-parks after being resumed would otherwise mint an API trigger
    where it first minted a task or a job trigger, and then wait forever on a
    resume nobody sends.
    """
    if not isinstance(stored_type, str) or stored_type == _value_type_name(value):
        return value
    model = INTERRUPT_SUSPEND_MODELS.get(stored_type)
    if model is None:
        logger.warning(
            "Interrupt %s was suspended with a %s value, which is not a known "
            "interrupt model, so it is reloaded as stored.",
            interrupt_id,
            stored_type,
        )
        return value
    if isinstance(value, list):
        return [model.model_validate(item) for item in value]
    return model.model_validate(value)


def _pending_from_stored(raw: Any) -> PendingSuspend | None:
    """Rebuild a pending record, or None when nothing usable was stored."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("Ignoring a pending suspend record that is not an object.")
        return None
    try:
        return PendingSuspend(
            interrupt_id=str(raw["interrupt_id"]),
            tool_name=str(raw["tool_name"]),
            value=_rebuild_value(
                raw.get("value_type"), raw["value"], str(raw["interrupt_id"])
            ),
            tool_use_id=None if raw["tool_use_id"] is None else str(raw["tool_use_id"]),
            requested_at=str(raw["requested_at"]),
        )
    except KeyError as e:
        logger.warning("Ignoring an incomplete pending suspend record: %s", e)
        return None


class ClaudeSessionStore:
    """Stores Claude SDK session state keyed by runtime id.

    Holds the SDK session id, so the next exchange or a resumed process
    re-attaches via ``ClaudeAgentOptions(resume=session_id)``, the pending
    deferred tool call a suspension is waiting on, and the entrypoint the run
    started on. Everything lands in the runtime key-value table of the state
    database, which lives inside the persisted runtime directory and is
    therefore remounted when a suspended job resumes.
    """

    def __init__(self, storage: UiPathRuntimeStorageProtocol, runtime_id: str) -> None:
        self._storage = storage
        self._runtime_id = runtime_id

    async def _get(self, key: str) -> Any:
        return await self._storage.get_value(self._runtime_id, _NAMESPACE, key)

    async def _set(self, key: str, value: Any) -> None:
        await self._storage.set_value(self._runtime_id, _NAMESPACE, key, value)

    async def get_session_id(self) -> str | None:
        value = await self._get(_SESSION_KEY)
        return value if isinstance(value, str) else None

    async def set_session_id(self, session_id: str) -> None:
        await self._set(_SESSION_KEY, session_id)

    async def get_pending_suspend(self) -> PendingSuspend | None:
        return _pending_from_stored(await self._get(_PENDING_SUSPEND_KEY))

    async def set_pending_suspend(self, pending: PendingSuspend) -> None:
        await self._set(_PENDING_SUSPEND_KEY, _pending_to_stored(pending))

    async def clear_pending_suspend(self) -> None:
        await self._set(_PENDING_SUSPEND_KEY, None)

    async def get_entrypoint(self) -> EntrypointRecord | None:
        return EntrypointRecord.from_stored(await self._get(_ENTRYPOINT_KEY))

    async def set_entrypoint(self, record: EntrypointRecord) -> None:
        await self._set(_ENTRYPOINT_KEY, record.to_stored())

    async def get_transcript(self) -> TranscriptRecord | None:
        return TranscriptRecord.from_stored(await self._get(_TRANSCRIPT_KEY))

    async def set_transcript(self, record: TranscriptRecord) -> None:
        await self._set(_TRANSCRIPT_KEY, record.to_stored())

    async def clear_transcript(self) -> None:
        await self._set(_TRANSCRIPT_KEY, None)


__all__ = [
    "ClaudeSessionStore",
    "EntrypointRecord",
    "TranscriptRecord",
]
