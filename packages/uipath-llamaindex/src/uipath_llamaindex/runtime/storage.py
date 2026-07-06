"""SQLite implementation of UiPathResumableStorageProtocol."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, cast

from pydantic import BaseModel
from uipath.core.errors import ErrorCategory, UiPathFaultedTriggerError
from uipath.runtime import (
    UiPathApiTrigger,
    UiPathResumeTrigger,
    UiPathResumeTriggerName,
    UiPathResumeTriggerType,
)

from ._sqlite import AsyncSqlite


class SqliteResumableStorage:
    """SQLite database storage for resume triggers and workflow context."""

    def __init__(self, storage_path: str):
        """
        Initialize SQLite storage.

        Args:
            storage_path: Path to the SQLite database file
        """
        self.storage_path = storage_path
        self._db: AsyncSqlite | None = None

    async def _get_db(self) -> AsyncSqlite:
        """Get or create database connection."""
        if self._db is None:
            self._db = AsyncSqlite(self.storage_path, timeout=30.0)
            await self._db.connect()
        return self._db

    async def dispose(self) -> None:
        """Dispose of the storage and close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SqliteResumableStorage:
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.dispose()

    async def setup(self) -> None:
        """Ensure storage directory and database tables exist."""
        dir_name = os.path.dirname(self.storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            db = await self._get_db()

            # Table for workflow contexts
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflow_contexts (
                    runtime_id TEXT PRIMARY KEY,
                    context_data BLOB NOT NULL
                )
            """)

            # Table for resume triggers
            await db.execute("""
                CREATE TABLE IF NOT EXISTS resume_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_id TEXT NOT NULL,
                    interrupt_id TEXT NOT NULL,
                    trigger_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resume_triggers_runtime_id
                ON resume_triggers(runtime_id)
                """
            )

            await db.execute(
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

            await db.commit()
        except Exception as exc:
            msg = f"Failed to initialize SQLite storage at {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Save resume trigger to SQLite database."""
        try:
            db = await self._get_db()

            # Delete all existing triggers for this runtime_id
            await db.execute(
                """
                DELETE FROM resume_triggers
                WHERE runtime_id = ?
                """,
                (runtime_id,),
            )

            # Insert new triggers
            for trigger in triggers:
                trigger_dict = self._serialize_trigger(trigger)
                trigger_json = json.dumps(trigger_dict)
                await db.execute(
                    "INSERT INTO resume_triggers (runtime_id, interrupt_id, trigger_data) VALUES (?, ?, ?)",
                    (runtime_id, trigger.interrupt_id, trigger_json),
                )

            await db.commit()
        except Exception as exc:
            msg = f"Failed to save resume triggers to database {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        """Get most recent trigger from SQLite database."""
        try:
            db = await self._get_db()
            rows = await db.fetchall(
                "SELECT trigger_data FROM resume_triggers WHERE runtime_id = ? ORDER BY id ASC",
                (runtime_id,),
            )
        except Exception as exc:
            msg = f"Failed to retrieve resume triggers from database {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

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
        """Delete resume trigger from storage."""
        await self.delete_triggers(runtime_id, [trigger])

    async def delete_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Delete resume triggers from storage."""
        interrupt_ids = [trigger.interrupt_id for trigger in triggers]
        if not interrupt_ids:
            return

        try:
            db = await self._get_db()
            await db.execute(
                f"""
                DELETE FROM resume_triggers
                WHERE runtime_id = ?
                AND interrupt_id IN ({",".join("?" for _ in interrupt_ids)})
                """,
                (runtime_id, *interrupt_ids),
            )
            await db.commit()
        except Exception as exc:
            msg = f"Failed to delete resume triggers from database {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

    async def save_context(self, runtime_id: str, context_dict: dict[str, Any]) -> None:
        """
        Save workflow context to SQLite database.

        Args:
            runtime_id: Unique identifier for the runtime instance
            context_dict: Serialized workflow context dictionary
        """
        context_blob = pickle.dumps(context_dict)

        try:
            db = await self._get_db()
            await db.execute(
                """
                INSERT INTO workflow_contexts (runtime_id, context_data)
                VALUES (?, ?)
                ON CONFLICT(runtime_id) DO UPDATE SET
                    context_data = excluded.context_data
                """,
                (runtime_id, context_blob),
            )
            await db.commit()
        except Exception as exc:
            msg = f"Failed to save workflow context to database {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

    async def load_context(self, runtime_id: str) -> dict[str, Any] | None:
        """
        Load workflow context from SQLite database.

        Args:
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Serialized workflow context dictionary or None if not found
        """
        try:
            db = await self._get_db()
            row = await db.fetchone(
                "SELECT context_data FROM workflow_contexts WHERE runtime_id = ?",
                (runtime_id,),
            )
        except Exception as exc:
            msg = f"Failed to load workflow context from database {self.storage_path!r}: {exc}"
            raise UiPathFaultedTriggerError(ErrorCategory.SYSTEM, msg) from exc

        if not row:
            return None

        return pickle.loads(row[0])

    async def set_value(
        self,
        runtime_id: str,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Save arbitrary key-value pair to database."""
        if not (
            isinstance(value, str)
            or isinstance(value, dict)
            or isinstance(value, BaseModel)
            or value is None
        ):
            raise TypeError("Value must be str, dict, BaseModel or None.")

        value_text = self._dump_value(value)

        db = await self._get_db()
        await db.execute(
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
        await db.commit()

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        """Get arbitrary key-value pair from database (scoped by runtime_id + namespace)."""
        db = await self._get_db()
        row = await db.fetchone(
            """
            SELECT value
            FROM runtime_kv
            WHERE runtime_id = ? AND namespace = ? AND key = ?
            LIMIT 1
            """,
            (runtime_id, namespace, key),
        )

        if not row:
            return None

        return self._load_value(cast(str | None, row[0]))

    def _serialize_trigger(self, trigger: UiPathResumeTrigger) -> dict[str, Any]:
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

    def _deserialize_trigger(self, trigger_data: dict[str, Any]) -> UiPathResumeTrigger:
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
                inbox_id=resume_trigger.item_key, request=resume_trigger.payload
            )

        return resume_trigger

    def _dump_value(self, value: str | dict[str, Any] | BaseModel | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return "j:" + json.dumps(value.model_dump())
        if isinstance(value, dict):
            return "j:" + json.dumps(value)
        return "s:" + value

    def _load_value(self, raw: str | None) -> Any:
        if raw is None:
            return None
        if raw.startswith("s:"):
            return raw[2:]
        if raw.startswith("j:"):
            return json.loads(raw[2:])
        return raw
