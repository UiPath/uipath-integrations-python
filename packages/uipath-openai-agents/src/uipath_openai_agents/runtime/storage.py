"""OpenAI Agents SDK SQLiteSession adapter for UiPath resumable storage."""

import asyncio
import json
import os
from typing import Any, cast

from agents import SQLiteSession
from pydantic import BaseModel
from uipath.core.serialization import serialize_json
from uipath.core.triggers import UiPathResumeTrigger


class SqliteResumableStorage:
    """UiPath resumable storage backed by OpenAI Agents SDK SQLiteSession."""

    _STATE_ITEM_TYPE = "uipath_runtime_state"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SQLiteSession] = {}

    def _state_session_id(self, runtime_id: str) -> str:
        return f"__uipath_state__{runtime_id}"

    def _get_state_session(self, runtime_id: str) -> SQLiteSession:
        session_id = self._state_session_id(runtime_id)
        session = self._sessions.get(session_id)
        if session is None:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            session = SQLiteSession(session_id=session_id, db_path=self.db_path)
            self._sessions[session_id] = session
        return session

    async def _load_state(self, runtime_id: str) -> dict[str, Any]:
        session = self._get_state_session(runtime_id)
        items = await session.get_items(limit=1)
        if not items:
            return {"triggers": [], "kv": {}}

        item = items[0]
        if not isinstance(item, dict):
            return {"triggers": [], "kv": {}}

        if item.get("type") != self._STATE_ITEM_TYPE:
            return {"triggers": [], "kv": {}}

        state = item.get("state")
        if not isinstance(state, dict):
            return {"triggers": [], "kv": {}}
        if not isinstance(state.get("triggers"), list):
            state["triggers"] = []
        if not isinstance(state.get("kv"), dict):
            state["kv"] = {}
        return state

    async def _save_state(self, runtime_id: str, state: dict[str, Any]) -> None:
        session = self._get_state_session(runtime_id)
        await session.clear_session()
        state_item: dict[str, Any] = {
            "type": self._STATE_ITEM_TYPE,
            "state": state,
        }
        await session.add_items(
            [
                cast(Any, state_item),
            ]
        )

    async def save_triggers(
        self, runtime_id: str, triggers: list[UiPathResumeTrigger]
    ) -> None:
        """Save resume triggers, replacing all existing triggers for this runtime_id."""
        async with self._lock:
            state = await self._load_state(runtime_id)
            serialized_triggers: list[str] = []

            for trigger in triggers:
                if trigger.interrupt_id is None:
                    raise ValueError("Trigger interrupt_id cannot be None.")

                trigger_data = trigger.model_dump()
                trigger_data["payload"] = trigger.payload
                trigger_data["trigger_name"] = trigger.trigger_name
                serialized_triggers.append(serialize_json(trigger_data))

            state["triggers"] = serialized_triggers
            await self._save_state(runtime_id, state)

    async def get_triggers(self, runtime_id: str) -> list[UiPathResumeTrigger] | None:
        """Get all triggers for runtime_id."""
        async with self._lock:
            state = await self._load_state(runtime_id)

        trigger_payloads = state.get("triggers", [])
        if not trigger_payloads:
            return None

        triggers: list[UiPathResumeTrigger] = []
        for payload in trigger_payloads:
            if isinstance(payload, str):
                triggers.append(UiPathResumeTrigger.model_validate_json(payload))
        return triggers or None

    async def delete_trigger(
        self, runtime_id: str, trigger: UiPathResumeTrigger
    ) -> None:
        """Delete resume trigger from storage."""
        async with self._lock:
            state = await self._load_state(runtime_id)
            serialized = state.get("triggers", [])
            if not isinstance(serialized, list):
                serialized = []

            kept: list[str] = []
            for payload in serialized:
                if not isinstance(payload, str):
                    continue
                existing = UiPathResumeTrigger.model_validate_json(payload)
                if existing.interrupt_id != trigger.interrupt_id:
                    kept.append(payload)

            state["triggers"] = kept
            await self._save_state(runtime_id, state)

    async def set_value(
        self,
        runtime_id: str,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Save arbitrary key-value pair to storage."""
        if not (
            isinstance(value, str)
            or isinstance(value, dict)
            or isinstance(value, BaseModel)
            or value is None
        ):
            raise TypeError("Value must be str, dict, BaseModel or None.")

        async with self._lock:
            state = await self._load_state(runtime_id)
            kv = state.get("kv")
            if not isinstance(kv, dict):
                kv = {}
                state["kv"] = kv

            namespace_data = kv.get(namespace)
            if not isinstance(namespace_data, dict):
                namespace_data = {}
                kv[namespace] = namespace_data

            namespace_data[key] = self._dump_value(value)
            await self._save_state(runtime_id, state)

    async def get_value(self, runtime_id: str, namespace: str, key: str) -> Any:
        """Get arbitrary key-value pair from storage."""
        async with self._lock:
            state = await self._load_state(runtime_id)

        kv = state.get("kv")
        if not isinstance(kv, dict):
            return None
        namespace_data = kv.get(namespace)
        if not isinstance(namespace_data, dict):
            return None

        return self._load_value(namespace_data.get(key))

    async def close(self) -> None:
        """Close all OpenAI Agents SDK SQLite sessions."""
        async with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()

    def _dump_value(self, value: str | dict[str, Any] | BaseModel | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return "s:" + value
        return "j:" + serialize_json(value)

    def _load_value(self, raw: str | None) -> Any:
        if raw is None:
            return None
        if raw.startswith("s:"):
            return raw[2:]
        if raw.startswith("j:"):
            return json.loads(raw[2:])
        return raw
