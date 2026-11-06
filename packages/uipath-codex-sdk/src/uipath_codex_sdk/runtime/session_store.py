"""Persistence of Codex session state across a durable-tool suspension.

Everything lands in the runtime key-value table of the state database, which
is the one thing the platform carries from a suspended job to its resume
(possibly on another robot). That includes the Codex rollout file itself: the
resumed job gets a fresh runtime directory, so ``CODEX_HOME`` is gone and the
thread could not be resumed from it.

Keys (namespace ``codex_sdk``):

* ``thread`` – the thread id and where its rollout lived under ``CODEX_HOME``;
* ``rollout`` – the latest rollout snapshot (zlib + base64);
* ``pending`` – durable calls still waiting on their job, by call id;
* ``completed`` – tool result text of calls whose job has finished;
* ``dangling`` – every call left without an output when Codex was stopped, in
  rollout order, with its output type and (for siblings) the output to inject;
* ``entrypoint`` – the entrypoint the run started on;
* ``trace`` – the job's ``Agent run`` span, left open while the run is
  suspended (see :func:`.telemetry.span_record`);
* ``chat`` – a conversational agent's chat state: the messages and tool calls
  of the exchange still open in the chat (see :mod:`.chat_events`).

A conversational agent keeps ``thread`` and ``rollout`` across exchanges: the
conversation is the Codex thread, checkpointed after every exchange.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import logging
import os
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from uipath.core.serialization import serialize_object
from uipath.runtime import UiPathRuntimeStorageProtocol

logger = logging.getLogger(__name__)

_NAMESPACE = "codex_sdk"
_THREAD_KEY = "thread"
_ROLLOUT_KEY = "rollout"
_PENDING_KEY = "pending"
_COMPLETED_KEY = "completed"
_DANGLING_KEY = "dangling"
_ENTRYPOINT_KEY = "entrypoint"
_TRACE_KEY = "trace"
_CHAT_KEY = "chat"

#: A compressed rollout above this size is stored, with a warning.
ROLLOUT_WARN_BYTES = 8 * 1024 * 1024

_ENC_ZLIB = "zlib+b64"
_ENC_RAW = "raw+b64"

#: Rollout ``response_item`` types that open a tool call, and the output type
#: that answers each one.
CALL_OUTPUT_TYPES: dict[str, str] = {
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
    "local_shell_call": "function_call_output",
}
_OUTPUT_TYPES = frozenset(
    {"function_call_output", "custom_tool_call_output", "mcp_tool_call_output"}
)

STATE_STARTING = "starting"
STATE_STARTED = "started"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreadRecord:
    """The Codex thread a run belongs to.

    Attributes:
        thread_id: Codex thread id.
        rollout_rel: Path of its rollout relative to ``CODEX_HOME``.
    """

    thread_id: str
    rollout_rel: str | None = None

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> ThreadRecord | None:
        if not isinstance(raw, dict) or not raw.get("thread_id"):
            return None
        rel = raw.get("rollout_rel")
        return cls(
            thread_id=str(raw["thread_id"]), rollout_rel=str(rel) if rel else None
        )


@dataclass(frozen=True)
class EntrypointRecord:
    """The entrypoint a suspended run started on."""

    entrypoint: str

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> EntrypointRecord | None:
        if not isinstance(raw, dict) or "entrypoint" not in raw:
            return None
        return cls(entrypoint=str(raw["entrypoint"]))


@dataclass
class PendingCall:
    """A durable tool call whose job has been (or is being) started.

    Attributes:
        call_id: Responses ``call_id``; also the resume trigger interrupt id.
        tool: Model-facing tool name.
        args: Validated arguments the job was started with.
        state: ``"starting"`` (intent recorded, job not yet known) or
            ``"started"``.
        value_type: Class name of the suspend value, so it can be rebuilt as
            a typed model (a plain dict would become an API trigger).
        value: The serialized suspend value.
        order: Arrival order among the run's durable calls.
        span: The call's ``Tool call`` span, left open until the job result
            is used (see :func:`.telemetry.span_record`).
    """

    call_id: str
    tool: str
    args: dict[str, Any]
    state: str = STATE_STARTING
    value_type: str | None = None
    value: Any = None
    order: int = 0
    span: dict[str, Any] | None = None

    def to_stored(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "args": self.args,
            "state": self.state,
            "value_type": self.value_type,
            "value": serialize_object(self.value) if self.value is not None else None,
            "order": self.order,
            "span": self.span,
        }

    @classmethod
    def from_stored(cls, raw: Any) -> PendingCall | None:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                call_id=str(raw["call_id"]),
                tool=str(raw["tool"]),
                args=dict(raw.get("args") or {}),
                state=str(raw.get("state") or STATE_STARTING),
                value_type=raw.get("value_type"),
                value=raw.get("value"),
                order=int(raw.get("order") or 0),
                span=raw["span"] if isinstance(raw.get("span"), dict) else None,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Ignoring an unusable pending call record: %s", e)
            return None

    def suspend_value(self) -> Any:
        """The suspend value as a live model of its original type."""
        return rebuild_suspend_value(self.value_type, self.value)


@dataclass
class DanglingCall:
    """A tool call the rollout holds without an output.

    Attributes:
        call_id: Responses ``call_id``.
        item_type: Rollout item type (``function_call``, ``custom_tool_call``...).
        durable: Whether it is one of our held durable calls.
        output: Text to inject for a non-durable sibling (its real output, or
            a "not executed" note). ``None`` for durable calls.
    """

    call_id: str
    item_type: str
    durable: bool = False
    output: str | None = None

    @property
    def output_type(self) -> str:
        return CALL_OUTPUT_TYPES.get(self.item_type, "function_call_output")

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> DanglingCall | None:
        if not isinstance(raw, dict) or not raw.get("call_id"):
            return None
        return cls(
            call_id=str(raw["call_id"]),
            item_type=str(raw.get("item_type") or "function_call"),
            durable=bool(raw.get("durable")),
            output=raw.get("output"),
        )


@dataclass
class RolloutRecord:
    """A captured rollout file."""

    rel_path: str
    encoding: str
    data: str
    raw_size: int = 0
    stored_size: int = 0

    def to_stored(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_stored(cls, raw: Any) -> RolloutRecord | None:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                rel_path=str(raw["rel_path"]),
                encoding=str(raw["encoding"]),
                data=str(raw["data"]),
                raw_size=int(raw.get("raw_size") or 0),
                stored_size=int(raw.get("stored_size") or 0),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Ignoring an unusable rollout record: %s", e)
            return None

    def content(self) -> bytes:
        blob = base64.b64decode(self.data)
        if self.encoding == _ENC_ZLIB:
            return zlib.decompress(blob)
        return blob


# --------------------------------------------------------------------------
# suspend value types
# --------------------------------------------------------------------------
def _suspend_models() -> dict[str, type]:
    from uipath.platform.common import (
        InvokeProcess,
        InvokeProcessRaw,
        WaitBatchTransform,
        WaitDeepRag,
        WaitDeepRagRaw,
        WaitDocumentExtraction,
        WaitDocumentExtractionValidation,
        WaitEscalation,
        WaitJob,
        WaitJobRaw,
        WaitTask,
    )

    from ..batch_transform_tools import WaitBatchTransformFile

    return {
        m.__name__: m
        for m in (
            WaitBatchTransformFile,
            WaitBatchTransform,
            WaitDocumentExtraction,
            WaitDocumentExtractionValidation,
            WaitJobRaw,
            WaitJob,
            InvokeProcessRaw,
            InvokeProcess,
            WaitDeepRagRaw,
            WaitDeepRag,
            WaitEscalation,
            WaitTask,
        )
    }


def rebuild_suspend_value(value_type: Any, value: Any) -> Any:
    """Rebuild a stored suspend value as a typed model.

    ``UiPathResumeTriggerCreator`` picks the trigger from the value's type; a
    plain dict would silently become an API trigger nobody ever fires.
    """
    if value is None or not isinstance(value_type, str):
        return value
    model = _suspend_models().get(value_type)
    if model is None:
        logger.warning(
            "Unknown suspend value type %s; reloading as stored.", value_type
        )
        return value
    if isinstance(value, model):
        return value
    return model.model_validate(value)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# rollout files
# --------------------------------------------------------------------------
#: File names Codex writes a rollout under.
ROLLOUT_SUFFIXES: tuple[str, ...] = (".jsonl", ".jsonl.zst")
#: Suffix of the temp file a restore is written to before it is moved in place.
RESTORE_TMP_SUFFIX = ".restore-tmp"


def find_rollouts(codex_home: str | os.PathLike[str], thread_id: str) -> list[Path]:
    """Every rollout file of ``thread_id`` under ``codex_home/sessions``.

    Only real rollout names count (``.jsonl`` / ``.jsonl.zst``): a leftover
    restore temp file must never be taken for the live rollout.
    """
    pattern = os.path.join(
        os.fspath(codex_home), "sessions", "**", f"rollout-*{thread_id}.jsonl*"
    )
    return sorted(
        Path(p)
        for p in glob.glob(pattern, recursive=True)
        if p.endswith(ROLLOUT_SUFFIXES)
    )


def find_rollout(codex_home: str | os.PathLike[str], thread_id: str) -> Path | None:
    """The newest rollout of ``thread_id`` (the one Codex would pick)."""
    found = find_rollouts(codex_home, thread_id)
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def cut_torn_tail(data: bytes) -> bytes:
    """Drop a partially written last line (a SIGKILL can land mid-write)."""
    if not data or data.endswith(b"\n"):
        return data
    cut = data.rfind(b"\n")
    dropped = len(data) - (cut + 1)
    logger.warning("Dropping a torn rollout tail of %d bytes.", dropped)
    return data[: cut + 1] if cut >= 0 else b""


def parse_rollout(data: bytes | str) -> list[dict[str, Any]]:
    """Parse rollout JSON lines; unparsable lines are skipped."""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def response_items(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        ln.get("payload") or {} for ln in lines if ln.get("type") == "response_item"
    ]


def has_call(lines: list[dict[str, Any]], call_id: str) -> bool:
    """Whether the rollout already holds the call item for ``call_id``."""
    return any(
        item.get("type") in CALL_OUTPUT_TYPES and item.get("call_id") == call_id
        for item in response_items(lines)
    )


def dangling_calls(lines: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(call_id, item_type)`` of every call without an output, in order."""
    calls: list[tuple[str, str]] = []
    answered: set[str] = set()
    for item in response_items(lines):
        itype = item.get("type")
        cid = item.get("call_id")
        if not isinstance(cid, str):
            continue
        if itype in CALL_OUTPUT_TYPES:
            calls.append((cid, str(itype)))
        elif itype in _OUTPUT_TYPES:
            answered.add(cid)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for cid, itype in calls:
        if cid in answered or cid in seen:
            continue
        seen.add(cid)
        out.append((cid, itype))
    return out


def encode_rollout(rel_path: str, data: bytes) -> RolloutRecord:
    """Compress a rollout for the KV store (``.zst`` files are stored raw)."""
    if rel_path.endswith(".zst"):
        blob, encoding = data, _ENC_RAW
    else:
        blob, encoding = zlib.compress(data, 6), _ENC_ZLIB
    encoded = base64.b64encode(blob).decode("ascii")
    if len(blob) > ROLLOUT_WARN_BYTES:
        logger.warning(
            "The Codex rollout is %d bytes compressed; the state database grows "
            "with every suspension of a long conversation.",
            len(blob),
        )
    return RolloutRecord(
        rel_path=rel_path,
        encoding=encoding,
        data=encoded,
        raw_size=len(data),
        stored_size=len(blob),
    )


def write_rollout(
    codex_home: str | os.PathLike[str], thread_id: str, record: RolloutRecord
) -> Path:
    """Put a stored rollout back under ``codex_home``.

    An existing copy of the thread's rollout (fast-path resume in the same
    process, or a re-mounted runtime dir) is replaced in place; otherwise it
    goes back to its original relative path. There is never a second copy:
    Codex resolves the thread to the newest matching file.
    """
    content = record.content()
    for leftover in Path(codex_home, "sessions").glob(f"**/*{RESTORE_TMP_SUFFIX}"):
        try:
            leftover.unlink()
        except OSError:
            logger.warning("Could not remove a leftover restore file %s", leftover)
    existing = find_rollouts(codex_home, thread_id)
    target = existing[0] if existing else Path(codex_home) / record.rel_path
    for extra in existing[1:]:
        try:
            extra.unlink()
        except OSError:
            logger.warning("Could not remove a stale rollout copy %s", extra)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + RESTORE_TMP_SUFFIX)
    tmp.write_bytes(content)
    os.replace(tmp, target)
    return target


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
class CodexSessionStore:
    """Codex session state of one runtime id, kept in the state database."""

    def __init__(self, storage: UiPathRuntimeStorageProtocol, runtime_id: str) -> None:
        self._storage = storage
        self._runtime_id = runtime_id
        self._pending_lock = asyncio.Lock()

    async def _get(self, key: str) -> Any:
        return await self._storage.get_value(self._runtime_id, _NAMESPACE, key)

    async def _set(self, key: str, value: Any) -> None:
        await self._storage.set_value(self._runtime_id, _NAMESPACE, key, value)

    # thread / entrypoint ---------------------------------------------------
    async def get_thread(self) -> ThreadRecord | None:
        return ThreadRecord.from_stored(await self._get(_THREAD_KEY))

    async def set_thread(self, record: ThreadRecord) -> None:
        await self._set(_THREAD_KEY, record.to_stored())

    async def get_entrypoint(self) -> EntrypointRecord | None:
        return EntrypointRecord.from_stored(await self._get(_ENTRYPOINT_KEY))

    async def set_entrypoint(self, record: EntrypointRecord) -> None:
        await self._set(_ENTRYPOINT_KEY, record.to_stored())

    # trace -----------------------------------------------------------------
    async def get_trace(self) -> dict[str, Any] | None:
        raw = await self._get(_TRACE_KEY)
        return raw if isinstance(raw, dict) and raw.get("span_id") else None

    async def set_trace(self, record: dict[str, Any] | None) -> None:
        await self._set(_TRACE_KEY, record)

    # chat ------------------------------------------------------------------
    async def get_chat(self) -> dict[str, Any]:
        """The chat state of a conversational agent (empty when none)."""
        raw = await self._get(_CHAT_KEY)
        return raw if isinstance(raw, dict) else {}

    async def set_chat(self, state: dict[str, Any] | None) -> None:
        await self._set(_CHAT_KEY, state or None)

    # rollout ---------------------------------------------------------------
    async def get_rollout(self) -> RolloutRecord | None:
        return RolloutRecord.from_stored(await self._get(_ROLLOUT_KEY))

    async def set_rollout(self, record: RolloutRecord) -> None:
        await self._set(_ROLLOUT_KEY, record.to_stored())

    # pending ---------------------------------------------------------------
    async def get_pending(self) -> dict[str, PendingCall]:
        raw = await self._get(_PENDING_KEY)
        out: dict[str, PendingCall] = {}
        if isinstance(raw, dict):
            for cid, value in raw.items():
                call = PendingCall.from_stored(value)
                if call is not None:
                    out[str(cid)] = call
        return out

    async def _write_pending(self, pending: dict[str, PendingCall]) -> None:
        await self._set(
            _PENDING_KEY,
            {cid: call.to_stored() for cid, call in pending.items()} or None,
        )

    async def put_pending(self, call: PendingCall) -> None:
        """Insert or replace one pending call (safe under concurrent calls)."""
        async with self._pending_lock:
            pending = await self.get_pending()
            pending[call.call_id] = call
            await self._write_pending(pending)

    async def remove_pending(self, call_ids: list[str]) -> None:
        async with self._pending_lock:
            pending = await self.get_pending()
            for cid in call_ids:
                pending.pop(cid, None)
            await self._write_pending(pending)

    # completed / dangling ----------------------------------------------------
    async def get_completed(self) -> dict[str, str]:
        raw = await self._get(_COMPLETED_KEY)
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    async def set_completed(self, completed: dict[str, str]) -> None:
        await self._set(_COMPLETED_KEY, dict(completed) or None)

    async def get_dangling(self) -> list[DanglingCall]:
        raw = await self._get(_DANGLING_KEY)
        items = raw.get("calls") if isinstance(raw, dict) else None
        out: list[DanglingCall] = []
        for value in items or []:
            call = DanglingCall.from_stored(value)
            if call is not None:
                out.append(call)
        return out

    async def set_dangling(self, calls: list[DanglingCall]) -> None:
        await self._set(
            _DANGLING_KEY, {"calls": [c.to_stored() for c in calls]} if calls else None
        )

    async def clear_suspension(self) -> None:
        """Forget everything about the last suspension (not the thread)."""
        async with self._pending_lock:
            await self._set(_PENDING_KEY, None)
        await self._set(_COMPLETED_KEY, None)
        await self._set(_DANGLING_KEY, None)

    async def clear_all(self) -> None:
        """Start over: a fresh (non-resume) run of the same runtime id."""
        await self.clear_suspension()
        await self._set(_THREAD_KEY, None)
        await self._set(_ROLLOUT_KEY, None)
        await self._set(_ENTRYPOINT_KEY, None)
        await self._set(_TRACE_KEY, None)
        await self._set(_CHAT_KEY, None)


__all__ = [
    "CALL_OUTPUT_TYPES",
    "CodexSessionStore",
    "DanglingCall",
    "EntrypointRecord",
    "PendingCall",
    "RolloutRecord",
    "STATE_STARTED",
    "STATE_STARTING",
    "ThreadRecord",
    "cut_torn_tail",
    "dangling_calls",
    "encode_rollout",
    "find_rollout",
    "find_rollouts",
    "has_call",
    "parse_rollout",
    "rebuild_suspend_value",
    "write_rollout",
]
