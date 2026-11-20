"""Job attachments: Orchestrator files a run receives, passes on and returns.

As in LangChain ReAct agents, the model never opens a file itself. It sees
each attachment as a JobAttachment object (``{"ID", "FullName", "MimeType"}``)
and hands it to the tools that accept attachments (Analyze Files, process and
agent tools). The run keeps the attachments it knows: those of its input and
those tools returned. A tool argument naming an unknown ``ID`` is answered
with an error instead of running the tool, and an attachment field of the
agent output may only name a known one.

Attachment fields are found in JSON schemas by any of:

* ``"$ref": "#/definitions/job-attachment"`` (the platform schema marker,
  ``uipath.runtime.schema.transform_attachments``);
* ``"x-uipath-resource-kind": "JobAttachment"``;
* ``"title": "UiPathAttachment"`` (a Pydantic model's own schema).
"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, ValidationError
from uipath.platform.attachments import Attachment

logger = logging.getLogger(__name__)

#: The attachment type to use in input and output models (the platform's own).
UiPathAttachment = Attachment

ATTACHMENT_TITLE = "UiPathAttachment"
JOB_ATTACHMENT_DEFINITION = "job-attachment"
RESOURCE_KIND_KEY = "x-uipath-resource-kind"
RESOURCE_KIND = "JobAttachment"

#: A path into a JSON value: property names, and ``"*"`` for every list item.
Path = tuple[str, ...]

_MAX_DEPTH = 32


# --- schema discovery ---------------------------------------------------------


def _resolve_ref(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part.replace("~1", "/").replace("~0", "~"))
    return node


def is_attachment_schema(node: Any, root: dict[str, Any] | None = None) -> bool:
    """True for a schema node that describes one job attachment."""
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref.rsplit("/", 1)[-1] == JOB_ATTACHMENT_DEFINITION:
            return True
        if root is not None:
            target = _resolve_ref(ref, root)
            if target is not node and is_attachment_schema(target):
                return True
    return (
        node.get(RESOURCE_KIND_KEY) == RESOURCE_KIND
        or node.get("title") == ATTACHMENT_TITLE
    )


def attachment_paths(schema: Any) -> list[Path]:
    """Paths of the attachment fields of a JSON schema (``$ref`` followed).

    ``("invoice",)`` is a property, ``("files", "*")`` every item of a list.
    Nullable and other ``anyOf``/``oneOf``/``allOf`` branches share the path.
    """
    if not isinstance(schema, dict):
        return []
    root = schema
    found: list[Path] = []

    def walk(node: Any, path: Path, refs: frozenset[str]) -> None:
        if not isinstance(node, dict) or len(path) > _MAX_DEPTH:
            return
        if is_attachment_schema(node, root):
            if path not in found:
                found.append(path)
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref not in refs:
            walk(_resolve_ref(ref, root), path, refs | {ref})
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                walk(sub, (*path, str(name)), refs)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, (*path, "*"), refs)
        for key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for branch in branches:
                    walk(branch, path, refs)

    walk(schema, (), frozenset())
    return found


def path_text(path: Path) -> str:
    """JSONPath-like text of a path (``$.files[*]``), for messages."""
    out = "$"
    for part in path:
        out += "[*]" if part == "*" else f".{part}"
    return out


# --- values at paths ----------------------------------------------------------


def _values_at(data: Any, path: Path) -> Iterator[Any]:
    if not path:
        yield data
        return
    head, rest = path[0], path[1:]
    if head == "*":
        if isinstance(data, list):
            for item in data:
                yield from _values_at(item, rest)
    elif isinstance(data, dict) and head in data:
        yield from _values_at(data[head], rest)


def values_at(data: Any, paths: Iterable[Path]) -> list[Any]:
    """The values found at ``paths`` (missing and null ones skipped)."""
    return [v for p in paths for v in _values_at(data, p) if v]


def _replace_at(data: Any, path: Path, fn: Any) -> Any:
    """``data`` with ``fn(value)`` at ``path`` (``data`` is changed in place)."""
    if not path:
        return fn(data)
    head, rest = path[0], path[1:]
    if head == "*":
        if isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = _replace_at(item, rest, fn)
    elif isinstance(data, dict) and head in data:
        data[head] = _replace_at(data[head], rest, fn)
    return data


def _coerce_json(value: Any) -> Any:
    """A JSON object or list the model sent as a string, parsed."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def dump_attachment(attachment: Attachment) -> dict[str, Any]:
    """The wire shape ``{"ID", "FullName", "MimeType"[, "Metadata"]}``."""
    return attachment.model_dump(by_alias=True, mode="json", exclude_none=True)


def to_attachment(value: Any) -> Attachment | None:
    """``value`` as an :class:`Attachment`, or None when it is not one."""
    if isinstance(value, Attachment):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(by_alias=True)
    try:
        return Attachment.model_validate(value)
    except ValidationError:
        return None


def not_found_message(attachment_id: str) -> str:
    """LangChain's text for an unknown attachment ID (a tool result)."""
    return (
        f"Could not find JobAttachment with ID='{attachment_id}'. "
        "Try invoking the tool again and please make sure that you pass "
        "valid JobAttachment IDs associated with existing JobAttachments in the "
        "current context."
    )


def _id_of(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("ID", value.get("id"))
        return None if raw is None else str(raw)
    return None


def _canonical_id(raw: str) -> str | None:
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return None


# --- the run's known attachments ----------------------------------------------


class AttachmentStore:
    """The attachments a run knows, by ID: its input's and tools' results."""

    def __init__(self, attachments: Iterable[Attachment] = ()) -> None:
        self._known: dict[str, Attachment] = {}
        self.changed = False
        for attachment in attachments:
            self.register(attachment)
        self.changed = False

    def __len__(self) -> int:
        return len(self._known)

    def __contains__(self, attachment_id: object) -> bool:
        return (
            isinstance(attachment_id, str)
            and _canonical_id(attachment_id) in self._known
        )

    @property
    def attachments(self) -> list[Attachment]:
        return list(self._known.values())

    def get(self, attachment_id: str) -> Attachment | None:
        key = _canonical_id(attachment_id)
        return self._known.get(key) if key is not None else None

    def register(self, attachment: Attachment) -> None:
        key = str(attachment.id)
        if self._known.get(key) != attachment:
            self._known[key] = attachment
            self.changed = True

    def harvest(self, paths: Iterable[Path], data: Any) -> list[Attachment]:
        """Register the attachments found at ``paths`` of ``data``; return them.

        A value that is not a valid attachment is skipped.
        """
        found: list[Attachment] = []
        for value in values_at(data, list(paths)):
            attachment = to_attachment(_coerce_json(value))
            if attachment is None:
                logger.warning("Ignoring an invalid job attachment in a tool result")
                continue
            self.register(attachment)
            found.append(attachment)
        return found

    def resolve(
        self, paths: Iterable[Path], args: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """``args`` with each attachment at ``paths`` replaced by the known one.

        The model may pass ``{"ID": ...}`` only (or the object as a JSON
        string). Returns the new arguments and one error per unknown or
        invalid ID (LangChain's text); ``args`` is not changed.
        """
        paths = list(paths)
        if not paths:
            return args, []
        result = copy.deepcopy(args)
        errors: list[str] = []

        def replace(value: Any) -> Any:
            value = _coerce_json(value)
            raw = _id_of(value)
            if raw is None:
                return value
            known = self.get(raw)
            if known is None:
                errors.append(not_found_message(raw))
                return value
            return dump_attachment(known)

        for path in paths:
            result = _replace_at(result, path, replace)
        return result, errors

    # --- persistence (suspend/resume) --------------------------------------

    def to_json(self) -> list[dict[str, Any]]:
        return [dump_attachment(a) for a in self._known.values()]

    @classmethod
    def from_json(cls, data: Any) -> "AttachmentStore":
        items = data if isinstance(data, list) else []
        return cls(a for a in (to_attachment(v) for v in items) if a is not None)


# --- model instances ------------------------------------------------------------


def model_attachments(value: Any) -> list[Attachment]:
    """Every :class:`Attachment` inside a validated model (fields, lists, dicts)."""
    found: list[Attachment] = []

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(node, Attachment):
            found.append(node)
        elif isinstance(node, BaseModel):
            for name in type(node).model_fields:
                walk(getattr(node, name, None), depth + 1)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                walk(item, depth + 1)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item, depth + 1)

    walk(value, 0)
    return found


def model_has_attachments(model_type: type[BaseModel] | None) -> bool:
    if model_type is None:
        return False
    try:
        return bool(attachment_paths(model_type.model_json_schema()))
    except Exception:
        return False


class AttachmentTicket(dict):
    """An attachment in a prompt: formats as compact JSON, indexes like a dict."""

    def __format__(self, spec: str) -> str:
        return format(json.dumps(self, separators=(",", ":")), spec)

    def __str__(self) -> str:
        return json.dumps(self, separators=(",", ":"))


class _TicketList(list):
    def __format__(self, spec: str) -> str:
        return format(json.dumps(self, separators=(",", ":")), spec)

    def __str__(self) -> str:
        return json.dumps(self, separators=(",", ":"))


def ticket(attachment: Attachment) -> AttachmentTicket:
    data = dump_attachment(attachment)
    data.pop("Metadata", None)
    return AttachmentTicket(data)


def with_tickets(
    model: BaseModel, dumped: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    """``model.model_dump()`` with each attachment as an :class:`AttachmentTicket`.

    Returns the fields and the names of the top-level fields that hold an
    attachment.
    """
    top: set[str] = set()

    def swap(node: Any, data: Any) -> Any:
        if isinstance(node, Attachment):
            return ticket(node)
        if isinstance(node, BaseModel) and isinstance(data, dict):
            for name in type(node).model_fields:
                if name in data:
                    data[name] = swap(getattr(node, name, None), data[name])
            return data
        if isinstance(node, (list, tuple)) and isinstance(data, list):
            items = [swap(n, d) for n, d in zip(node, data, strict=False)]
            if any(isinstance(i, AttachmentTicket) for i in items):
                return _TicketList(items)
            return items
        return data

    for name in type(model).model_fields:
        if name not in dumped:
            continue
        value = getattr(model, name, None)
        if model_attachments(value):
            top.add(name)
            dumped[name] = swap(value, dumped[name])
    return dumped, top


def attachments_block(attachments: Iterable[Attachment]) -> str:
    """The prompt block listing input attachments a template does not show."""
    items = [dict(ticket(a)) for a in attachments]
    return "Input attachments (JobAttachment objects):\n" + json.dumps(
        items, separators=(",", ":")
    )


# --- tracing ----------------------------------------------------------------------


def span_attachments(attachments: Iterable[Attachment], direction: str) -> str | None:
    """The ``attachments`` span attribute (``SpanAttachment`` JSON), or None.

    ``direction`` is ``"in"`` or ``"out"``. Never raises.
    """
    try:
        from uipath.tracing import (
            AttachmentDirection,
            AttachmentProvider,
            SpanAttachment,
        )

        kind = AttachmentDirection.IN if direction == "in" else AttachmentDirection.OUT
        items = [
            SpanAttachment(
                id=str(a.id),
                file_name=a.full_name,
                mime_type=a.mime_type,
                provider=AttachmentProvider.ORCHESTRATOR,
                direction=kind,
            ).model_dump(by_alias=True, mode="json")
            for a in attachments
        ]
        return json.dumps(items) if items else None
    except Exception:
        logger.debug("could not build the attachments span attribute", exc_info=True)
        return None


def merge_span_attachments(existing: Any, extra: str | None) -> str | None:
    """Two ``attachments`` attribute values as one (``existing`` first)."""
    if not extra:
        return existing if isinstance(existing, str) else None
    try:
        old = json.loads(existing) if isinstance(existing, str) else []
        return json.dumps([*old, *json.loads(extra)])
    except ValueError:
        return extra


__all__ = [
    "AttachmentStore",
    "AttachmentTicket",
    "UiPathAttachment",
    "attachment_paths",
    "attachments_block",
    "dump_attachment",
    "is_attachment_schema",
    "merge_span_attachments",
    "model_attachments",
    "model_has_attachments",
    "not_found_message",
    "path_text",
    "span_attachments",
    "to_attachment",
    "values_at",
    "with_tickets",
]
