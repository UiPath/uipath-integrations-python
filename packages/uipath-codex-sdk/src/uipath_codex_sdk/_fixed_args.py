"""Pre-filled tool arguments: paths, schema hiding and the call-time merge.

Shared by every tool with fields the model does not fill: static values,
values read from the agent input, text templates and arrays built from those
(agent.json ``argumentProperties``, or ``static_args`` / ``args_from_input``
in code). A leaf module: it imports only the stdlib and ``uipath.agent``, so
``tools``, ``escalation_tools`` and ``integration_tools`` can all use it.

Names are paths into the tool input: ``a.b`` creates objects on the way,
``x[*].y`` sets ``y`` on every element of ``x``, ``x[*]`` sets every element
(an empty or absent ``x`` becomes ``[value]``) and ``x[0].b`` addresses one
element. LangChain's JSONPath keys (``$['a']['b']``, ``$.a.b``) are
converted with :func:`input_name`.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(r"\[(\*|\d+)\]")
#: One JSONPath bracket: ``['name']`` / ``["name"]``, ``[*]`` or ``[N]``.
_BRACKET_RE = re.compile(
    r"""\[\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|(\*|\d+))\s*\]"""
)
_NAME_RE = re.compile(r"[^.\[\]]+")


# --------------------------------------------------------------------------
# path names
# --------------------------------------------------------------------------
def _segments(name: str) -> list[str | int]:
    """``attachments[*].actions[0].text`` → ``["attachments", "*", "actions", 0, "text"]``."""
    out: list[str | int] = []
    for part in name.split("."):
        key, _, rest = part.partition("[")
        if key:
            out.append(key)
        if rest:
            for idx in _SEGMENT_RE.findall("[" + rest):
                out.append("*" if idx == "*" else int(idx))
    return out


def _join(segs: Sequence[str | int]) -> str:
    """The name of ``segs`` (the inverse of :func:`_segments`)."""
    out = ""
    for seg in segs:
        if isinstance(seg, int) or seg == "*":
            out += f"[{seg}]"
        else:
            out += f".{seg}" if out else str(seg)
    return out


def input_name(key: str) -> str | None:
    """A JSONPath key or path as a Codex path name; None when it has no equivalent.

    ``$['a']['b']`` → ``a.b``, ``$['d'][0]`` → ``d[0]``, ``$['x'][*]['y']`` →
    ``x[*].y``, ``$.a.b`` → ``a.b``; a Codex name (``a.b``, ``x[*]``) is kept.
    A bracketed name holding ``.``, ``[`` or ``]``, recursive descent
    (``$..a``) and malformed paths have no equivalent.
    """
    s = key.strip()
    if s == "$" or s.startswith(("$.", "$[")):
        s = s[1:]  # the JSONPath root (``$filter`` is a field name)
    segs: list[str | int] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "[":
            m = _BRACKET_RE.match(s, i)
            if m is None:
                return None
            quoted = m.group(1) if m.group(1) is not None else m.group(2)
            if quoted is not None:
                if not quoted or any(ch in quoted for ch in ".[]"):
                    return None
                segs.append(quoted)
            else:
                idx = m.group(3)
                segs.append("*" if idx == "*" else int(idx))
            i = m.end()
            continue
        if c == ".":
            i += 1
        elif segs:
            return None  # a name right after ``]``
        m = _NAME_RE.match(s, i)
        if m is None:
            return None
        segs.append(m.group(0))
        i = m.end()
    return _join(segs) or None


# --------------------------------------------------------------------------
# reading and writing paths
# --------------------------------------------------------------------------
def _set_path(obj: Any, name: str, value: Any) -> None:
    """Set ``value`` at IS name ``name`` in ``obj``.

    ``a.b.c`` creates the objects on the way; ``a[*].b`` sets ``b`` on every
    element of an existing list ``a`` (an absent list is not created);
    ``a[*]`` sets every element, and turns an empty or absent ``a`` into
    ``[value]`` (as langchain does); ``a[0].b`` addresses (and pads) one
    element.
    """
    segs = _segments(name)
    if segs:
        _set_segments(obj, segs, value)


def _set_segments(obj: Any, segs: list[str | int], value: Any) -> None:
    seg, rest = segs[0], segs[1:]
    if seg == "*":
        if not isinstance(obj, list):
            return
        for i in range(len(obj)):
            if rest:
                _set_segments(obj[i], rest, value)
            else:
                obj[i] = copy.deepcopy(value)
        return
    if isinstance(seg, int):
        if not isinstance(obj, list):
            return
        while len(obj) <= seg:
            obj.append({} if rest and not isinstance(rest[0], int) else None)
        if not rest:
            obj[seg] = value
            return
        if not isinstance(obj[seg], (dict, list)):
            obj[seg] = [] if isinstance(rest[0], int) else {}
        _set_segments(obj[seg], rest, value)
        return
    if not isinstance(obj, dict):
        return
    if not rest:
        obj[seg] = value
        return
    child = obj.get(seg)
    nxt = rest[0]
    if nxt == "*":
        if rest == ["*"] and (child is None or child == []):
            # langchain's apply_static_args: an empty (or absent) array
            # becomes ``[value]`` instead of staying empty.
            obj[seg] = [copy.deepcopy(value)]
        elif isinstance(child, list):
            _set_segments(child, rest, value)
        return
    if isinstance(nxt, int):
        if not isinstance(child, list):
            child = obj[seg] = []
    elif not isinstance(child, dict):
        child = obj[seg] = {}
    _set_segments(child, rest, value)


def _get_path(obj: Any, path: str) -> tuple[bool, Any]:
    """``(found, value)`` of a ``{{x}}`` path (``x``, ``a.b``, ``$.a.b``) in ``obj``."""
    path = path.strip()
    if path.startswith("$"):
        path = path[1:].lstrip(".")
    current = obj
    for seg in _segments(path):
        if isinstance(seg, int):
            if not isinstance(current, list) or seg >= len(current):
                return False, None
            current = current[seg]
        elif isinstance(current, dict) and seg in current:
            current = current[seg]
        else:
            return False, None
    return True, current


# --------------------------------------------------------------------------
# schema editing
# --------------------------------------------------------------------------
def _resolve_ref(root: dict[str, Any], node: Any) -> Any:
    """Follow a local ``$ref``, inlining a copy so shared definitions stay intact."""
    if not isinstance(node, dict) or "$ref" not in node:
        return node
    ref = node["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return node
        target = target[part]
    if not isinstance(target, dict):
        return node
    inlined = copy.deepcopy(target)
    node.pop("$ref")
    node.update({k: v for k, v in inlined.items() if k not in node})
    return node


def _unwrap_any_of(root: dict[str, Any], node: Any) -> Any:
    """The first non-null ``anyOf`` branch of ``node`` (``$ref`` resolved), else ``node``."""
    if not isinstance(node, dict):
        return node
    for branch in node.get("anyOf") or []:
        if isinstance(branch, dict) and branch.get("type") != "null":
            return _resolve_ref(root, branch)
    return node


def _navigate(root: dict[str, Any], segs: list[str | int]) -> dict[str, Any] | None:
    """The schema node at ``segs``, nullable ``anyOf`` unwrapped (also at the leaf)."""
    current: Any = _resolve_ref(root, root)
    for seg in segs:
        if not isinstance(current, dict):
            return None
        current = _unwrap_any_of(root, current)
        if seg in ("*",) or isinstance(seg, int):
            current = current.get("items")
        else:
            props = current.get("properties")
            if not isinstance(props, dict) or seg not in props:
                return None
            current = props[seg]
        current = _resolve_ref(root, current)
    current = _unwrap_any_of(root, current)
    return current if isinstance(current, dict) else None


def _parent_and_key(
    schema: dict[str, Any], name: str
) -> tuple[dict[str, Any], str] | None:
    """The object schema holding IS name ``name``'s leaf, and the leaf key."""
    segs = _segments(name)
    if not segs or not isinstance(segs[-1], str):
        return None  # ``x[*]`` / ``x[0]``: the leaf is an array element
    parent = _navigate(schema, segs[:-1]) if len(segs) > 1 else schema
    if parent is None or not isinstance(parent.get("properties"), dict):
        return None
    return parent, segs[-1]


def _hide_static(schema: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Remove static fields (at any depth) from ``properties`` and ``required``.

    Returns a modified deep copy (``$ref`` on the way are inlined).
    """
    if not names:
        return schema
    schema = copy.deepcopy(schema)
    for name in names:
        found = _parent_and_key(schema, name)
        if found is None:
            continue
        parent, key = found
        parent["properties"].pop(key, None)
        if isinstance(parent.get("required"), list):
            parent["required"] = [r for r in parent["required"] if r != key]
    return schema


def _note_arguments(
    schema: dict[str, Any], arguments: Mapping[str, str]
) -> dict[str, Any]:
    """Keep ``{{x}}`` fields visible but optional, noting where they come from.

    The agent input's value wins at call time; when the input has none, the
    model's value is sent (langchain leaves an unresolved field to the
    model too). Returns a modified deep copy.
    """
    if not arguments:
        return schema
    schema = copy.deepcopy(schema)
    for name, path in arguments.items():
        found = _parent_and_key(schema, name)
        if found is None:
            continue
        parent, key = found
        node = parent["properties"].get(key)
        if not isinstance(node, dict):
            continue
        note = (
            f"Set from the agent input '{path}' when it has a value; fill it "
            "only if it does not."
        )
        description = node.get("description")
        node["description"] = f"{description} ({note})" if description else note
        if isinstance(parent.get("required"), list):
            parent["required"] = [r for r in parent["required"] if r != key]
    return schema


def _without_fields(schema: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """``schema`` without the top-level fields of ``names`` (``a`` of ``a.b``)."""
    tops = {n for n in names if "." not in n and "[" not in n}
    if not tops or not isinstance(schema.get("properties"), dict):
        return schema
    out = copy.deepcopy(schema)
    for name in tops:
        out["properties"].pop(name, None)
    if isinstance(out.get("required"), list):
        out["required"] = [r for r in out["required"] if r not in tops]
    return out


def reachable(schema: dict[str, Any], name: str) -> bool:
    """True when ``schema`` has the field ``name`` (for ``x[*]`` / ``x[0]``: ``x``)."""
    segs = _segments(name)
    if not segs:
        return False
    schema = copy.deepcopy(schema)  # navigating inlines ``$ref``
    if isinstance(segs[-1], str) and segs[-1] != "*":
        found = _parent_and_key(schema, name)
        return found is not None and found[1] in found[0]["properties"]
    return _navigate(schema, segs[:-1]) is not None


# --------------------------------------------------------------------------
# pre-filled arguments
# --------------------------------------------------------------------------
@dataclass
class FixedArg:
    """A pre-filled argument the model does not fill.

    One of: ``value`` (static), ``path`` (from the agent input), ``tokens``
    (a text template rendered from the agent input) or ``items`` (an array
    built from other pre-filled arguments; a gap is None).
    """

    value: Any = None
    path: str | None = None
    tokens: list[Any] | None = None
    items: list[FixedArg | None] | None = None

    @property
    def from_input(self) -> bool:
        """True for an agent-input value the model fills when the input has none."""
        return self.path is not None and self.tokens is None and self.items is None

    def resolve(self, agent_input: dict[str, Any] | None) -> tuple[bool, Any]:
        """``(found, value)``: only an agent-input path can be not found."""
        if self.items is not None:
            values: list[Any] = []
            for item in self.items:
                found, value = (
                    item.resolve(agent_input) if item is not None else (False, None)
                )
                values.append(value if found else None)
            return True, values
        if self.tokens is not None:
            from uipath.agent.utils.text_tokens import build_string_from_tokens

            return True, build_string_from_tokens(self.tokens, agent_input or {})
        if self.path is not None:
            return _get_path(agent_input, self.path)
        return True, copy.deepcopy(self.value)


def dedupe_parents(names: Iterable[str]) -> set[str]:
    """``names`` without those under another name (``a.b`` when ``a`` is there).

    langchain's parent-wins rule, compared by path segment so ``user`` does
    not swallow ``users``.
    """
    segs = {name: _segments(name) for name in names}
    keep: set[str] = set()
    for name, mine in segs.items():
        if not any(
            other != name and len(theirs) < len(mine) and mine[: len(theirs)] == theirs
            for other, theirs in segs.items()
        ):
            keep.add(name)
    return keep


def _variant(prop: Any) -> str:
    variant = getattr(prop, "variant", None)
    return str(getattr(variant, "value", variant) or type(prop).__name__)


def fixed_args_from_properties(
    label: str,
    argument_properties: Mapping[str, Any] | None,
    *,
    schema: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, FixedArg]:
    """agent.json ``argumentProperties`` as pre-filled arguments.

    Keys and argument paths may be JSONPath (``$['a']['b']``). Static,
    argument (agent-input path), text-builder and array-builder properties
    are supported (an array's items are its ``name[N]`` keys); a parent wins
    over the paths under it, as in langchain. Any other property is left to
    the model with a warning, or with ``strict`` an object builder is refused
    (langchain's "Unsupported argument property type") unless a parent path
    covers it. When ``schema`` has properties, a name it does not have is
    dropped with a warning.

    Raises:
        ValueError: ``strict`` and an object-builder property no parent covers.
    """
    from uipath.agent.models.agent import (
        AgentToolArgumentArgumentProperties,
        AgentToolStaticArgumentProperties,
        AgentToolTextBuilderArgumentProperties,
    )

    fixed: dict[str, FixedArg] = {}
    arrays: list[str] = []
    objects: dict[str, str] = {}  # name → key, refused once deduped (strict)
    for key, prop in (argument_properties or {}).items():
        name = input_name(key)
        if name is None:
            logger.warning(
                "%s: argument property key '%s' is not a supported path; the "
                "model fills it.",
                label,
                key,
            )
            continue
        variant = _variant(prop)
        if isinstance(prop, AgentToolStaticArgumentProperties):
            fixed[name] = FixedArg(value=prop.value)
        elif isinstance(prop, AgentToolArgumentArgumentProperties):
            path = input_name(prop.argument_path)
            if path is None:
                logger.warning(
                    "%s: argument property '%s' reads '%s', which is not a "
                    "supported agent input path; the model fills it.",
                    label,
                    key,
                    prop.argument_path,
                )
                continue
            fixed[name] = FixedArg(path=path)
        elif isinstance(prop, AgentToolTextBuilderArgumentProperties):
            fixed[name] = FixedArg(tokens=list(prop.tokens))
        elif variant == "arrayBuilder":
            arrays.append(name)
        elif variant == "objectBuilder" and strict:
            objects[name] = key
        else:
            logger.warning(
                "%s: argument property '%s' (%s) is not supported; the model fills it.",
                label,
                key,
                variant,
            )
    # Innermost arrays first, so an array item can itself be an array.
    for name in sorted(arrays, key=lambda n: len(_segments(n)), reverse=True):
        child = re.compile(rf"^{re.escape(name)}\[(\d+)\]$")
        indexed = {
            int(m.group(1)): arg
            for other, arg in fixed.items()
            if (m := child.match(other)) is not None
        }
        items: list[FixedArg | None] = [None] * (max(indexed) + 1 if indexed else 0)
        for index, arg in indexed.items():
            items[index] = arg
        fixed[name] = FixedArg(items=items)
    if objects:
        _refuse_objects(label, objects, fixed, arrays)
    kept = dedupe_parents(fixed)
    fixed = {name: arg for name, arg in fixed.items() if name in kept}
    if isinstance(schema, dict) and schema.get("properties"):
        for name in list(fixed):
            if not reachable(schema, name):
                logger.warning(
                    "%s: argument property '%s' is not in the tool's input "
                    "schema; ignored.",
                    label,
                    name,
                )
                del fixed[name]
    return fixed


def _refuse_objects(
    label: str,
    objects: Mapping[str, str],
    fixed: Mapping[str, FixedArg],
    arrays: Sequence[str],
) -> None:
    """Refuse an object builder langchain would resolve (and fail on).

    That is one no parent path covers, or a direct item (``name[N]``) of an
    array builder that is resolved: langchain's parent-wins dedup skips any
    other.
    """
    kept = dedupe_parents([*fixed, *objects])
    used = {name for name in arrays if name in kept}
    grown = True
    while grown:  # an array item that is itself an array builder
        items = {n for n in arrays if n not in used and _item_of(n) in used}
        grown = bool(items)
        used |= items
    for name, key in objects.items():
        if name in kept or _item_of(name) in used:
            raise ValueError(
                f"{label}: argument property '{key}' is an objectBuilder; this "
                "argument property type is unsupported."
            )


def _item_of(name: str) -> str | None:
    """The array ``name`` is a direct item of (``x`` of ``x[0]``), else None."""
    segs = _segments(name)
    if len(segs) > 1 and isinstance(segs[-1], int):
        return _join(segs[:-1])
    return None


def fixed_args_from_code(
    static_args: Mapping[str, Any] | None,
    args_from_input: Mapping[str, str] | None,
    *,
    label: str | None = None,
) -> dict[str, FixedArg]:
    """``static_args`` and ``args_from_input`` (field → ``{{a.b}}`` / ``a.b``) as pre-filled arguments.

    Fields and paths may be JSONPath (``$['a']['b']``). With a ``label``, a
    field set in both mappings or a path with no Codex equivalent is refused;
    without one, the field is kept as written and ``args_from_input`` wins.

    Raises:
        ValueError: With ``label``, a field in both mappings or a bad path.
    """

    def name_of(raw: str, what: str) -> str:
        name = input_name(str(raw))
        if name is None:
            if label is not None:
                raise ValueError(f"{label}: {what} '{raw}' is not a supported path.")
            return str(raw)
        return name

    fixed: dict[str, FixedArg] = {
        name_of(name, "field"): FixedArg(value=value)
        for name, value in (static_args or {}).items()
    }
    for name, path in (args_from_input or {}).items():
        field_name = name_of(name, "field")
        if label is not None and field_name in fixed:
            raise ValueError(
                f"{label}: '{name}' is in both static_args and args_from_input."
            )
        fixed[field_name] = FixedArg(
            path=name_of(str(path).strip().strip("{}").strip(), "agent input path")
        )
    return fixed


def with_fixed_args(
    args: dict[str, Any],
    fixed_args: Mapping[str, FixedArg],
    agent_input: dict[str, Any] | None,
) -> dict[str, Any]:
    """The model's arguments with the pre-filled values (they win), as JSON values.

    An agent-input value the input does not have leaves the model's value.
    ``args`` is not changed.
    """
    data = copy.deepcopy(args)
    for name, fixed in fixed_args.items():
        found, value = fixed.resolve(agent_input)
        if found:
            _set_path(data, name, value)
    return json.loads(json.dumps(data, default=str))


def missing_from_input(
    schema: dict[str, Any],
    fixed_args: Mapping[str, FixedArg],
    merged: dict[str, Any],
) -> list[str]:
    """The agent-input fields ``schema`` requires that ``merged`` still lacks.

    That is, the agent input had no value and the model did not fill it
    (:func:`fixed_args_schema` shows such a field as optional). A field under
    an absent object is not reported: that object's own requirement decides.
    Through ``[*]``, every element is checked (``x[*].y`` is reported when
    any element of ``x`` lacks ``y``).
    """
    missing: list[str] = []
    for name, fixed in fixed_args.items():
        segs = _segments(name)
        if not fixed.from_input or not segs or segs[-1] == "*":
            continue
        if not isinstance(segs[-1], str):
            continue  # ``x[0]``: the leaf is an array element
        found = _parent_and_key(copy.deepcopy(schema), name)
        if found is None:
            continue
        parent, key = found
        if key not in (parent.get("required") or []):
            continue
        if any(
            isinstance(container, dict) and key not in container
            for container in _values_at(merged, segs[:-1])
        ):
            missing.append(name)
    return missing


def _values_at(obj: Any, segs: Sequence[str | int]) -> list[Any]:
    """The values at ``segs`` in ``obj`` (``*`` each element); absent ones are skipped."""
    if not segs:
        return [obj]
    seg, rest = segs[0], segs[1:]
    if seg == "*":
        if not isinstance(obj, list):
            return []
        return [v for item in obj for v in _values_at(item, rest)]
    if isinstance(seg, int):
        if not isinstance(obj, list) or seg >= len(obj):
            return []
        return _values_at(obj[seg], rest)
    if not isinstance(obj, dict) or seg not in obj:
        return []
    return _values_at(obj[seg], rest)


def missing_args_message(tool_name: str, missing: Sequence[str]) -> str:
    """The tool error for required fields neither the agent input nor the model filled."""
    return (
        f"Invalid arguments for tool '{tool_name}': missing required "
        f"argument(s): {', '.join(missing)}. The agent input has no value for "
        "them; fill them in and call the tool again."
    )


def fixed_args_schema(
    schema: dict[str, Any], fixed_args: Mapping[str, FixedArg]
) -> dict[str, Any]:
    """The input schema the model sees.

    Static, text-builder and array-builder fields are removed at any depth
    (``x[*]`` removes the array ``x``); agent-input fields stay visible but
    optional, with a note, so the model can fill them when the agent input
    has no value. ``x[0]`` stays visible (its value is still merged).
    """
    if not fixed_args:
        return schema
    hidden: list[str] = []
    noted: dict[str, str] = {}
    for name, fixed in fixed_args.items():
        if fixed.from_input:
            noted[name] = str(fixed.path)
            continue
        segs = _segments(name)
        if len(segs) > 1 and segs[-1] == "*" and isinstance(segs[-2], str):
            hidden.append(_join(segs[:-1]))
        else:
            hidden.append(name)
    return _note_arguments(_hide_static(schema, hidden), noted)


__all__ = [
    "FixedArg",
    "dedupe_parents",
    "fixed_args_from_code",
    "fixed_args_from_properties",
    "fixed_args_schema",
    "input_name",
    "missing_args_message",
    "missing_from_input",
    "reachable",
    "with_fixed_args",
]
