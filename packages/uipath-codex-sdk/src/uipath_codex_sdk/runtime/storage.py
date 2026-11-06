"""SQLite implementation of ``UiPathResumableStorageProtocol``.

A port of uipath-langchain's ``SqliteResumableStorage`` onto a plain
``aiosqlite`` connection (Codex has no LangGraph checkpointer to share). The
schema and serialization are kept identical on purpose:

* triggers carry an ``interrupt_id`` column and are deleted by it, never by
  their serialized JSON;
* ``trigger_name`` and ``payload`` are ``exclude=True`` on
  ``UiPathResumeTrigger``, so they are added back before saving. Without them
  a reloaded JOB_RAW trigger would come back as a plain JOB trigger, and
  ``read_trigger`` would raise on a faulted job instead of returning it.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import aiosqlite
from pydantic import BaseModel
from uipath.core.serialization import serialize_json
from uipath.core.triggers import UiPathResumeTrigger

_TRIGGERS_TABLE = "__uipath_resume_triggers"
_KV_TABLE = "__uipath_runtime_kv"


class SqliteResumableStorage:
    """SQLite storage for resume triggers and arbitrary key-value pairs.

    Args:
        db_path: Path of the state database (``__uipath/state.db`` by default).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.rs_table_name = _TRIGGERS_TABLE
        self.kv_table_name = _KV_TABLE
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _ensure_conn(self) -> aiosqlite.Connection:
        """Open the connection and create the tables on first use."""
        if self._conn is not None:
            return self._conn
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        conn = await aiosqlite.connect(self.db_path)
        try:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=30000")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.rs_table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_id TEXT NOT NULL,
                    interrupt_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc'))
                )
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.rs_table_name}_runtime_id
                ON {self.rs_table_name}(runtime_id)
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.kv_table_name} (
                    runtime_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'utc')),
                    PRIMARY KEY (runtime_id, namespace, key)
                )
                """
            )
            await conn.commit()
        except BaseException:
            await conn.close()
            raise
        self._conn = conn
        return conn

    # --- resume triggers --------------------------------------------------

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Replace every stored trigger of ``runtime_id`` with ``triggers``."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                f"DELETE FROM {self.rs_table_name} WHERE runtime_id = ?",
                (runtime_id,),
            )
            for trigger in triggers:
                trigger_data = trigger.model_dump()
                trigger_data["payload"] = trigger.payload
                trigger_data["trigger_name"] = trigger.trigger_name
                await conn.execute(
                    f"""
                    INSERT INTO {self.rs_table_name} (runtime_id, interrupt_id, data)
                    VALUES (?, ?, ?)
                    """,
                    (
                        runtime_id,
                        trigger.interrupt_id or "",
                        serialize_json(trigger_data),
                    ),
                )
            await conn.commit()

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        """All triggers of ``runtime_id`` in insertion order, or None."""
        async with self._lock:
            conn = await self._ensure_conn()
            cursor = await conn.execute(
                f"""
                SELECT data FROM {self.rs_table_name}
                WHERE runtime_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (runtime_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        if not rows:
            return None
        return [
            UiPathResumeTrigger.model_validate_json(cast(str, row[0])) for row in rows
        ]

    async def delete_trigger(
        self, runtime_id: str, trigger: UiPathResumeTrigger
    ) -> None:
        """Delete one trigger (by its interrupt id)."""
        await self.delete_triggers(runtime_id, [trigger])

    async def delete_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Delete triggers by interrupt id."""
        interrupt_ids = [t.interrupt_id for t in triggers if t.interrupt_id]
        if not interrupt_ids:
            return
        placeholders = ",".join("?" for _ in interrupt_ids)
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                f"""
                DELETE FROM {self.rs_table_name}
                WHERE runtime_id = ? AND interrupt_id IN ({placeholders})
                """,
                (runtime_id, *interrupt_ids),
            )
            await conn.commit()

    # --- key-value --------------------------------------------------------

    async def set_value(
        self, runtime_id: str, namespace: str, key: str, value: Any
    ) -> None:
        """Upsert a value (``str``, ``dict``, ``BaseModel`` or ``None``)."""
        if not (
            value is None
            or isinstance(value, (str, dict))
            or isinstance(value, BaseModel)
        ):
            raise TypeError("Value must be str, dict, BaseModel or None.")
        value_text = self._dump_value(value)
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                f"""
                INSERT INTO {self.kv_table_name} (runtime_id, namespace, key, value)
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
        """Read a value, or None when absent."""
        async with self._lock:
            conn = await self._ensure_conn()
            cursor = await conn.execute(
                f"""
                SELECT value FROM {self.kv_table_name}
                WHERE runtime_id = ? AND namespace = ? AND key = ?
                LIMIT 1
                """,
                (runtime_id, namespace, key),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return self._load_value(cast(str | None, row[0]))

    # --- lifecycle --------------------------------------------------------

    async def dispose(self) -> None:
        """Close the connection. Safe to call more than once."""
        async with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            await conn.close()

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _dump_value(value: str | dict[str, Any] | BaseModel | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return "s:" + value
        return "j:" + serialize_json(value)

    @staticmethod
    def _load_value(raw: str | None) -> Any:
        if raw is None:
            return None
        if raw.startswith("s:"):
            return raw[2:]
        if raw.startswith("j:"):
            return json.loads(raw[2:])
        return raw


__all__ = ["SqliteResumableStorage"]
