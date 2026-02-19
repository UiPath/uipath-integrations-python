"""SQLite session store for Agent Framework agents.

Persists AgentSession state between turns using SQLite, keyed by runtime_id.
Each runtime_id maps to an isolated session — conversation history accumulates
across calls via the InMemoryHistoryProvider that Agent Framework auto-injects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class SqliteSessionStore:
    """SQLite-backed store for Agent Framework session state.

    Stores serialized AgentSession dicts (via to_dict/from_dict) in a single
    table, keyed by runtime_id. Thread-safe via asyncio lock.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def setup(self) -> None:
        """Ensure storage directory and database table exist."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    runtime_id TEXT PRIMARY KEY,
                    session_data TEXT NOT NULL
                )
                """
            )
            await conn.commit()
        self._initialized = True
        logger.debug("Session store initialized at %s", self.db_path)

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=30000")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.commit()
        return self._conn

    async def load_session(self, runtime_id: str) -> dict[str, Any] | None:
        """Load a serialized session dict for the given runtime_id.

        Returns None if no session exists for this runtime_id.
        """
        if not self._initialized:
            await self.setup()

        conn = await self._get_conn()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT session_data FROM sessions WHERE runtime_id = ?",
                (runtime_id,),
            )
            row = await cursor.fetchone()

        if not row:
            logger.debug("No session found for runtime_id=%s", runtime_id)
            return None

        logger.debug("Loaded session for runtime_id=%s", runtime_id)
        return json.loads(row[0])

    async def save_session(self, runtime_id: str, session_data: dict[str, Any]) -> None:
        """Save a serialized session dict for the given runtime_id."""
        if not self._initialized:
            await self.setup()

        data_json = json.dumps(session_data)
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                """
                INSERT INTO sessions (runtime_id, session_data)
                VALUES (?, ?)
                ON CONFLICT(runtime_id) DO UPDATE SET
                    session_data = excluded.session_data
                """,
                (runtime_id, data_json),
            )
            await conn.commit()

        logger.debug("Saved session for runtime_id=%s", runtime_id)

    async def dispose(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False


__all__ = ["SqliteSessionStore"]
