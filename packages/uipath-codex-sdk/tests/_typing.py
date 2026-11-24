"""Typed accessors shared by the tests.

OpenTelemetry and ``uipath.runtime`` type several fields as optional that are
always set on the objects the tests inspect (a finished span's attributes and
timestamps, the factory's storage). These helpers assert that once, so the
assertions themselves stay plain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from uipath.runtime import UiPathRuntimeEvent, UiPathRuntimeResult

from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage


def attrs(span: ReadableSpan) -> Mapping[str, Any]:
    """The span's attributes (asserted present)."""
    assert span.attributes is not None, span.name
    return span.attributes


def start_ns(span: ReadableSpan) -> int:
    """The span's start time (asserted set)."""
    assert span.start_time is not None, span.name
    return span.start_time


def end_ns(span: ReadableSpan) -> int:
    """The span's end time (asserted set: the span has finished)."""
    assert span.end_time is not None, span.name
    return span.end_time


def parent_span_id(span: ReadableSpan) -> int:
    """The span's parent's id (asserted to have a parent)."""
    assert span.parent is not None, span.name
    return span.parent.span_id


def as_result(event: UiPathRuntimeEvent) -> UiPathRuntimeResult:
    """A streamed event asserted to be the final result."""
    assert isinstance(event, UiPathRuntimeResult), event
    return event


async def sqlite_storage(factory: UiPathCodexRuntimeFactory) -> SqliteResumableStorage:
    """The factory's storage, asserted to be the SQLite resumable storage."""
    storage = await factory.get_storage()
    assert isinstance(storage, SqliteResumableStorage), storage
    return storage
