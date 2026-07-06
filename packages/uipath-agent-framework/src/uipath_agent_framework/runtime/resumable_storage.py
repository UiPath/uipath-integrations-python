"""SQLite resumable storage for Agent Framework.

Provides SqliteResumableStorage with resume trigger, key-value, and checkpoint
tables, plus SqliteCheckpointStorage and ScopedCheckpointStorage for the
Agent Framework checkpoint protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiosqlite
from agent_framework import WorkflowCheckpoint
from agent_framework._workflows._checkpoint_encoding import (
    decode_checkpoint_value,
    encode_checkpoint_value,
)
from pydantic import BaseModel
from uipath.runtime import (
    UiPathApiTrigger,
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

logger = logging.getLogger(__name__)


class SqliteResumableStorage:
    """SQLite storage with resume triggers, KV, and checkpoint tables.

    Tables:
    - resume_triggers: interrupt trigger persistence for HITL
    - runtime_kv: arbitrary key-value storage scoped by runtime_id + namespace
    - checkpoints: workflow checkpoint persistence
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self.checkpoint_storage: SqliteCheckpointStorage | None = None

    async def setup(self) -> None:
        """Create all tables and initialize checkpoint storage."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_id TEXT NOT NULL,
                    interrupt_id TEXT NOT NULL,
                    trigger_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resume_triggers_runtime_id
                ON resume_triggers(runtime_id)
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_kv (
                    runtime_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc')),
                    PRIMARY KEY (runtime_id, namespace, key)
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    workflow_name TEXT NOT NULL,
                    checkpoint_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow
                ON checkpoints(workflow_name)
                """
            )
            await conn.commit()

        self.checkpoint_storage = SqliteCheckpointStorage(self)
        logger.debug("Resumable storage tables initialized at %s", self.db_path)

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=30000")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.commit()
        return self._conn

    async def dispose(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Resume trigger persistence
    # ------------------------------------------------------------------

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Save resume triggers, replacing any existing ones for this runtime_id."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "DELETE FROM resume_triggers WHERE runtime_id = ?",
                (runtime_id,),
            )
            for trigger in triggers:
                trigger_dict = self._serialize_trigger(trigger)
                trigger_json = json.dumps(trigger_dict)
                await conn.execute(
                    "INSERT INTO resume_triggers (runtime_id, interrupt_id, trigger_data) VALUES (?, ?, ?)",
                    (runtime_id, trigger.interrupt_id, trigger_json),
                )
            await conn.commit()

        logger.debug("Saved %d triggers for runtime_id=%s", len(triggers), runtime_id)

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        """Retrieve all resume triggers for this runtime_id."""
        conn = await self._get_conn()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT trigger_data FROM resume_triggers WHERE runtime_id = ? ORDER BY id ASC",
                (runtime_id,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return None

        triggers = []
        for row in rows:
            trigger_dict = json.loads(row[0])
            triggers.append(self._deserialize_trigger(trigger_dict))
        return triggers

    async def delete_trigger(
        self, runtime_id: str, trigger: UiPathResumeTrigger
    ) -> None:
        """Delete a specific resume trigger by runtime_id and interrupt_id."""
        await self.delete_triggers(runtime_id, [trigger])

    async def delete_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Delete specific resume triggers by runtime_id and interrupt_id."""
        interrupt_ids = [trigger.interrupt_id for trigger in triggers]
        if not interrupt_ids:
            return

        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                (
                    "DELETE FROM resume_triggers "
                    "WHERE runtime_id = ? "
                    f"AND interrupt_id IN ({','.join('?' for _ in interrupt_ids)})"
                ),
                (runtime_id, *interrupt_ids),
            )
            await conn.commit()

        logger.debug(
            "Deleted %d triggers for runtime_id=%s",
            len(interrupt_ids),
            runtime_id,
        )

    # ------------------------------------------------------------------
    # Key-value storage
    # ------------------------------------------------------------------

    async def set_value(
        self,
        runtime_id: str,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Save arbitrary key-value pair scoped by runtime_id + namespace."""
        value_text = self._dump_value(value)

        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO runtime_kv (runtime_id, namespace, key, value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(runtime_id, namespace, key)
                DO UPDATE SET
                    value = excluded.value,
                    timestamp = (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                """,
                (runtime_id, namespace, key, value_text),
            )
            await conn.commit()

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        """Get arbitrary key-value pair scoped by runtime_id + namespace."""
        conn = await self._get_conn()
        async with self._lock:
            cursor = await conn.execute(
                """
                SELECT value FROM runtime_kv
                WHERE runtime_id = ? AND namespace = ? AND key = ?
                LIMIT 1
                """,
                (runtime_id, namespace, key),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return self._load_value(row[0])

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_trigger(trigger: UiPathResumeTrigger) -> dict[str, Any]:
        """Serialize a resume trigger to a dictionary."""
        trigger_key = (
            trigger.api_resume.inbox_id if trigger.api_resume else trigger.item_key
        )
        payload = (
            json.dumps(trigger.payload)
            if isinstance(trigger.payload, dict)
            else str(trigger.payload)
            if trigger.payload
            else None
        )

        return {
            "type": trigger.trigger_type.value,
            "key": trigger_key,
            "name": trigger.trigger_name.value,
            "payload": payload,
            "interrupt_id": trigger.interrupt_id,
            "folder_path": trigger.folder_path,
            "folder_key": trigger.folder_key,
        }

    @staticmethod
    def _deserialize_trigger(
        trigger_data: dict[str, Any],
    ) -> UiPathResumeTrigger:
        """Deserialize a resume trigger from a dictionary."""
        trigger_type = trigger_data["type"]
        key = trigger_data["key"]
        name = trigger_data["name"]
        folder_path = trigger_data.get("folder_path")
        folder_key = trigger_data.get("folder_key")
        payload = trigger_data.get("payload")
        interrupt_id = trigger_data.get("interrupt_id")

        resume_trigger = UiPathResumeTrigger(
            trigger_type=UiPathResumeTriggerType(trigger_type),
            trigger_name=UiPathResumeTriggerName(name),
            item_key=key,
            folder_path=folder_path,
            folder_key=folder_key,
            payload=payload,
            interrupt_id=interrupt_id,
        )

        if resume_trigger.trigger_type == UiPathResumeTriggerType.API:
            resume_trigger.api_resume = UiPathApiTrigger(
                inbox_id=resume_trigger.item_key,
                request=resume_trigger.payload,
            )

        return resume_trigger

    @staticmethod
    def _dump_value(
        value: str | dict[str, Any] | BaseModel | None,
    ) -> str | None:
        """Serialize a value for KV storage."""
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return "j:" + json.dumps(value.model_dump())
        if isinstance(value, dict):
            return "j:" + json.dumps(value)
        if isinstance(value, str):
            return "s:" + value
        raise TypeError("Value must be str, dict, BaseModel or None.")

    @staticmethod
    def _load_value(raw: str | None) -> Any:
        """Deserialize a value from KV storage."""
        if raw is None:
            return None
        if raw.startswith("s:"):
            return raw[2:]
        if raw.startswith("j:"):
            return json.loads(raw[2:])
        return raw


class SqliteCheckpointStorage:
    """SQLite-backed CheckpointStorage implementation.

    Implements the agent_framework CheckpointStorage protocol using the
    checkpoints table managed by SqliteResumableStorage.
    """

    def __init__(self, storage: SqliteResumableStorage) -> None:
        self._storage = storage

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint and return its ID."""
        checkpoint_dict = checkpoint.to_dict()
        encoded = encode_checkpoint_value(checkpoint_dict)
        checkpoint_data = json.dumps(encoded, ensure_ascii=False)

        conn = await self._storage._get_conn()
        async with self._storage._lock:
            await conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (checkpoint_id, workflow_name, checkpoint_data, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.workflow_name,
                    checkpoint_data,
                    checkpoint.timestamp,
                ),
            )
            await conn.commit()

        logger.debug("Saved checkpoint %s", checkpoint.checkpoint_id)
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Load a checkpoint by ID."""
        conn = await self._storage._get_conn()
        async with self._storage._lock:
            cursor = await conn.execute(
                "SELECT checkpoint_data FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            row = await cursor.fetchone()

        if not row:
            from agent_framework._workflows._checkpoint import (  # type: ignore[attr-defined]
                WorkflowCheckpointException,
            )

            raise WorkflowCheckpointException(
                f"No checkpoint found with ID {checkpoint_id}"
            )

        encoded = json.loads(row[0])
        decoded = decode_checkpoint_value(encoded)
        return WorkflowCheckpoint.from_dict(decoded)

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        """List checkpoint objects for a given workflow name."""
        conn = await self._storage._get_conn()
        async with self._storage._lock:
            cursor = await conn.execute(
                "SELECT checkpoint_data FROM checkpoints WHERE workflow_name = ? ORDER BY timestamp ASC",
                (workflow_name,),
            )
            rows = await cursor.fetchall()

        checkpoints = []
        for row in rows:
            encoded = json.loads(row[0])
            decoded = decode_checkpoint_value(encoded)
            checkpoints.append(WorkflowCheckpoint.from_dict(decoded))
        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID."""
        conn = await self._storage._get_conn()
        async with self._storage._lock:
            cursor = await conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        """Get the latest checkpoint for a given workflow name."""
        conn = await self._storage._get_conn()
        async with self._storage._lock:
            cursor = await conn.execute(
                "SELECT checkpoint_data FROM checkpoints WHERE workflow_name = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1",
                (workflow_name,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        encoded = json.loads(row[0])
        decoded = decode_checkpoint_value(encoded)
        return WorkflowCheckpoint.from_dict(decoded)

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]:
        """List checkpoint IDs for a given workflow name."""
        conn = await self._storage._get_conn()
        async with self._storage._lock:
            cursor = await conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE workflow_name = ? ORDER BY timestamp ASC",
                (workflow_name,),
            )
            rows = await cursor.fetchall()

        return [row[0] for row in rows]


class ScopedCheckpointStorage:
    """Thin wrapper that prefixes workflow_name with a runtime scope.

    When multiple runtimes share the same SqliteResumableStorage, this
    ensures checkpoint isolation by prefixing workflow_name queries with
    ``{runtime_id}::``.
    """

    def __init__(self, delegate: SqliteCheckpointStorage, runtime_id: str) -> None:
        self._delegate = delegate
        self._scope = f"{runtime_id}::"

    def _scoped_name(self, workflow_name: str) -> str:
        return self._scope + workflow_name

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save with scoped workflow_name."""
        checkpoint.workflow_name = self._scoped_name(checkpoint.workflow_name)
        return await self._delegate.save(checkpoint)

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Load by checkpoint_id (globally unique)."""
        return await self._delegate.load(checkpoint_id)

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        """List checkpoints with scoped workflow_name."""
        return await self._delegate.list_checkpoints(
            workflow_name=self._scoped_name(workflow_name)
        )

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete by checkpoint_id (globally unique)."""
        return await self._delegate.delete(checkpoint_id)

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        """Get latest checkpoint with scoped workflow_name."""
        return await self._delegate.get_latest(
            workflow_name=self._scoped_name(workflow_name)
        )

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]:
        """List checkpoint IDs with scoped workflow_name."""
        return await self._delegate.list_checkpoint_ids(
            workflow_name=self._scoped_name(workflow_name)
        )


__all__ = [
    "SqliteResumableStorage",
    "SqliteCheckpointStorage",
    "ScopedCheckpointStorage",
]
