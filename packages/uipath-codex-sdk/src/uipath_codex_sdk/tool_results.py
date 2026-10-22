"""Results one durable tool keeps for another within a run.

The codex counterpart of langchain's ``inner_state.tools_storage``: an IXP
extraction keeps its full extraction response under its name, and the
Document Validation escalation reads it back. Saved with a suspended run, so
the two calls may be days and robots apart.
"""

from __future__ import annotations

from typing import Any


class ToolResultStore:
    """Values by key (last write wins) plus the keys whose result is on its way."""

    def __init__(
        self,
        values: dict[str, Any] | None = None,
        pending: list[str] | None = None,
    ) -> None:
        self._values: dict[str, Any] = dict(values or {})
        self._pending: set[str] = set(pending or [])
        self.changed = False

    def get(self, key: str) -> Any | None:
        return self._values.get(key)

    def put(self, key: str, value: Any) -> None:
        self._values[key] = value
        self._pending.discard(key)
        self.changed = True

    def mark_pending(self, key: str) -> None:
        """A call that will produce ``key``'s result has started."""
        self._pending.add(key)
        self.changed = True

    def clear_pending(self, key: str) -> None:
        if key in self._pending:
            self._pending.discard(key)
            self.changed = True

    def is_pending(self, key: str) -> bool:
        return key in self._pending

    def to_json(self) -> dict[str, Any]:
        return {"values": self._values, "pending": sorted(self._pending)}

    @classmethod
    def from_json(cls, data: Any) -> "ToolResultStore":
        if not isinstance(data, dict):
            return cls()
        values = data.get("values")
        pending = data.get("pending")
        return cls(
            values if isinstance(values, dict) else None,
            pending if isinstance(pending, list) else None,
        )


__all__ = ["ToolResultStore"]
