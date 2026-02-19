"""Tests for SQLite session store."""

import os
import tempfile

from uipath_agent_framework.runtime.storage import SqliteSessionStore


class TestSqliteSessionStore:
    """Tests for SqliteSessionStore."""

    async def test_setup_creates_db_file(self):
        """Setup creates the SQLite database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            assert os.path.exists(db_path)
            await store.dispose()

    async def test_setup_creates_nested_directories(self):
        """Setup creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "dir", "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            assert os.path.exists(db_path)
            await store.dispose()

    async def test_load_returns_none_for_missing_session(self):
        """Loading a non-existent session returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            result = await store.load_session("nonexistent")
            assert result is None
            await store.dispose()

    async def test_save_and_load_session(self):
        """Saved session data can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            session_data = {
                "session_id": "runtime-123",
                "state": {"memory": {"messages": [{"role": "user", "content": "hi"}]}},
            }
            await store.save_session("runtime-123", session_data)
            loaded = await store.load_session("runtime-123")

            assert loaded == session_data
            await store.dispose()

    async def test_save_overwrites_existing_session(self):
        """Saving with the same runtime_id overwrites the previous data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            await store.save_session("rt-1", {"version": 1})
            await store.save_session("rt-1", {"version": 2})
            loaded = await store.load_session("rt-1")

            assert loaded == {"version": 2}
            await store.dispose()

    async def test_sessions_isolated_by_runtime_id(self):
        """Different runtime_ids have independent sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            await store.save_session("rt-a", {"agent": "alpha"})
            await store.save_session("rt-b", {"agent": "beta"})

            assert await store.load_session("rt-a") == {"agent": "alpha"}
            assert await store.load_session("rt-b") == {"agent": "beta"}
            assert await store.load_session("rt-c") is None
            await store.dispose()

    async def test_dispose_allows_reconnect(self):
        """After dispose, the store can be set up again and data persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            store = SqliteSessionStore(db_path)
            await store.setup()
            await store.save_session("rt-1", {"data": "persisted"})
            await store.dispose()

            # Reconnect to the same DB
            store2 = SqliteSessionStore(db_path)
            await store2.setup()
            loaded = await store2.load_session("rt-1")

            assert loaded == {"data": "persisted"}
            await store2.dispose()

    async def test_auto_setup_on_load(self):
        """Loading without explicit setup triggers auto-setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)

            # No explicit setup() call
            result = await store.load_session("any-id")
            assert result is None
            await store.dispose()

    async def test_auto_setup_on_save(self):
        """Saving without explicit setup triggers auto-setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)

            # No explicit setup() call
            await store.save_session("rt-1", {"key": "value"})
            loaded = await store.load_session("rt-1")

            assert loaded == {"key": "value"}
            await store.dispose()

    async def test_complex_session_data_roundtrip(self):
        """Complex nested session data survives serialization roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SqliteSessionStore(db_path)
            await store.setup()

            session_data = {
                "session_id": "abc-123",
                "state": {
                    "memory": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the weather?",
                                "metadata": {"timestamp": "2025-01-01T00:00:00Z"},
                            },
                            {
                                "role": "assistant",
                                "content": "It's sunny!",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "name": "get_weather",
                                        "arguments": {"city": "SF"},
                                    }
                                ],
                            },
                        ]
                    },
                    "custom_key": [1, 2, 3],
                    "nested": {"a": {"b": {"c": True}}},
                },
            }

            await store.save_session("rt-complex", session_data)
            loaded = await store.load_session("rt-complex")

            assert loaded == session_data
            await store.dispose()
