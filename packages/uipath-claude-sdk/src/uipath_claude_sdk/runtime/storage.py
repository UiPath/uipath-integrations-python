"""SQLite storage adapter implementing UiPathResumableStorageProtocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
from uipath.core.triggers import UiPathResumeTrigger


class SqliteResumableStorage:
    """SQLite-backed storage for resume triggers and key-value pairs."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)

            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA busy_timeout=30000")
            await self._db.execute("PRAGMA synchronous=NORMAL")

            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS __uipath_resume_triggers "
                "(runtime_id TEXT, data TEXT, "
                "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS __uipath_runtime_kv "
                "(runtime_id TEXT, namespace TEXT, key TEXT, value TEXT, "
                "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY (runtime_id, namespace, key))"
            )
            await self._db.commit()
        return self._db

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        db = await self._ensure_db()
        await db.execute(
            "DELETE FROM __uipath_resume_triggers WHERE runtime_id = ?",
            (runtime_id,),
        )
        for trigger in triggers:
            await db.execute(
                "INSERT INTO __uipath_resume_triggers (runtime_id, data) VALUES (?, ?)",
                (runtime_id, trigger.model_dump_json()),
            )
        await db.commit()

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT data FROM __uipath_resume_triggers WHERE runtime_id = ?",
            (runtime_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None
        return [UiPathResumeTrigger.model_validate_json(row[0]) for row in rows]

    async def delete_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        db = await self._ensure_db()
        for trigger in triggers:
            await db.execute(
                "DELETE FROM __uipath_resume_triggers "
                "WHERE runtime_id = ? AND data = ?",
                (runtime_id, trigger.model_dump_json()),
            )
        await db.commit()

    async def delete_trigger(
        self, runtime_id: str, trigger: UiPathResumeTrigger
    ) -> None:
        await self.delete_triggers(runtime_id, [trigger])

    async def set_value(
        self, runtime_id: str, namespace: str, key: str, value: Any
    ) -> None:
        db = await self._ensure_db()
        await db.execute(
            "INSERT OR REPLACE INTO __uipath_runtime_kv "
            "(runtime_id, namespace, key, value) VALUES (?, ?, ?, ?)",
            (runtime_id, namespace, key, json.dumps(value)),
        )
        await db.commit()

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT value FROM __uipath_runtime_kv "
            "WHERE runtime_id = ? AND namespace = ? AND key = ?",
            (runtime_id, namespace, key),
        )
        row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    async def dispose(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
