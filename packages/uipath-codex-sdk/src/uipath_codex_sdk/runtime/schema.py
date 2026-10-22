"""Schema derivation for Codex agents (entrypoint input/output JSON schema)."""

from __future__ import annotations

import copy
import string
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from uipath.runtime.schema import (
    UiPathRuntimeEdge,
    UiPathRuntimeGraph,
    UiPathRuntimeNode,
    UiPathRuntimeSchema,
    transform_attachments,
    transform_nullable_types,
    transform_references,
)

from ..agent import UiPathCodexAgent
from ..attachments import ATTACHMENT_TITLE, JOB_ATTACHMENT_DEFINITION, RESOURCE_KIND

#: What the model writes for an attachment field of the agent output: the ID
#: of an attachment the run knows (the runtime fills in the rest).
_OUTPUT_ATTACHMENT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ID": {
            "type": "string",
            "description": (
                "ID of a JobAttachment from the input or a tool result; "
                "never a made-up one."
            ),
        }
    },
}


def _is_attachment_node(node: dict[str, Any]) -> bool:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.rsplit("/", 1)[-1] == JOB_ATTACHMENT_DEFINITION:
        return True
    return (
        node.get("title") == ATTACHMENT_TITLE
        or node.get("x-uipath-resource-kind") == RESOURCE_KIND
    )


def _output_attachments(node: Any) -> Any:
    """``node`` with each attachment schema replaced by :data:`_OUTPUT_ATTACHMENT`."""
    if isinstance(node, dict):
        if _is_attachment_node(node):
            return copy.deepcopy(_OUTPUT_ATTACHMENT)
        return {k: _output_attachments(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_output_attachments(v) for v in node]
    return node


def to_strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON schema satisfy OpenAI Responses structured-output strict mode.

    OpenAI/Codex ``response_format`` (``text.format``) requires every object to
    set ``additionalProperties: false`` and list all of its properties in
    ``required``. Pydantic's ``model_json_schema()`` does neither, so we walk
    the schema (including ``$defs`` and combinators) and enforce it.

    A job attachment (``UiPathAttachment``) becomes ``{"ID"}`` only: the
    model names an attachment the run knows, it never writes one. A ``$ref``
    with other keywords (a described model field) moves into an ``anyOf``.
    """
    schema = _output_attachments(copy.deepcopy(schema))

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "$ref" in node and len(node) > 1:
            # Strict mode refuses keywords next to ``$ref`` (pydantic puts a
            # field's ``description`` there): keep them on a one-branch anyOf.
            node["anyOf"] = [{"$ref": node.pop("$ref")}]
        props = node.get("properties")
        if isinstance(props, dict):
            node["additionalProperties"] = False
            node["required"] = list(props.keys())
            for value in props.values():
                walk(value)
        for defs_key in ("$defs", "definitions"):
            defs = node.get(defs_key)
            if isinstance(defs, dict):
                for value in defs.values():
                    walk(value)
        items = node.get("items")
        if items is not None:
            walk(items)
        for combinator in ("anyOf", "allOf", "oneOf"):
            branches = node.get(combinator)
            if isinstance(branches, list):
                for value in branches:
                    walk(value)

    walk(schema)
    return schema


_PERMISSIVE_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}
_DEFAULT_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"output": {"type": "string"}},
    "required": ["output"],
}


def _model_to_platform_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for the platform: resolve $ref/$defs and collapse nullable anyOf.

    Matches the convention used by the other UiPath integrations
    (uipath-claude-sdk / uipath-llamaindex). A ``UiPathAttachment`` field
    becomes ``$ref: #/definitions/job-attachment`` (the ``JobAttachment``
    resource kind), as in LangChain.
    """
    raw = model_type.model_json_schema()
    unpacked, _ = transform_references(raw)
    result: dict[str, Any] = {
        "type": "object",
        "properties": transform_nullable_types(unpacked.get("properties", {})),
        "required": unpacked.get("required", []),
    }
    for key in ("title", "description"):
        if key in unpacked:
            result[key] = unpacked[key]
    return transform_attachments(result)


def _parse_prompt_fields(prompt: str) -> dict[str, bool]:
    """Named ``str.format`` fields of ``prompt`` mapped to "used bare only".

    A name is bare when every use is a plain ``{name}``; attribute/index
    access (``{order.id}``, ``{items[0]}``) marks it as structured. Positional
    (``{}``, ``{0}``) fields are skipped, and a template ``str.format``
    cannot parse yields no fields.
    """
    fields: dict[str, bool] = {}

    def collect(template: str) -> None:
        for _, field_name, format_spec, _ in string.Formatter().parse(template):
            if field_name:
                base = field_name.split(".", 1)[0].split("[", 1)[0]
                if base and not base.isdigit():
                    bare = base == field_name
                    fields[base] = fields.get(base, True) and bare
            if format_spec:
                collect(format_spec)

    try:
        collect(prompt)
    except ValueError:
        return {}
    return fields


def prompt_fields(prompt: str) -> list[str]:
    """Named ``str.format`` fields of ``prompt``, in order of first use.

    Attribute/index access (``{order.id}``, ``{items[0]}``) contributes its
    base name; positional (``{}``, ``{0}``) fields are skipped, and so is a
    template ``str.format`` cannot parse.
    """
    return list(_parse_prompt_fields(prompt))


def _prompt_input_schema(prompt: str) -> dict[str, Any] | None:
    """Input schema derived from a prompt template's placeholders, if any.

    Bare ``{name}`` fields are strings. A name used with attribute/index
    access is left untyped: a string value would make ``str.format`` fail,
    and the shape it needs cannot be read off the template.
    """
    fields = _parse_prompt_fields(prompt)
    if not fields:
        return None
    return {
        "type": "object",
        "properties": {
            name: {"type": "string"} if bare else {} for name, bare in fields.items()
        },
        "required": list(fields),
    }


def _input_schema(agent: UiPathCodexAgent) -> dict[str, Any]:
    if agent.input_schema is not None:
        return _model_to_platform_schema(agent.input_schema)
    if agent.prompt:
        derived = _prompt_input_schema(agent.prompt)
        if derived is not None:
            return derived
    return dict(_PERMISSIVE_INPUT)


def _output_schema(agent: UiPathCodexAgent) -> dict[str, Any]:
    if agent.output_schema is not None:
        return _model_to_platform_schema(agent.output_schema)
    return dict(_DEFAULT_OUTPUT)


#: Codex built-in tools that exist in every sandbox mode: ``shell`` runs
#: (read-only) commands even under ``"read-only"``.
_ALWAYS_BUILTIN_TOOLS = ("shell",)
#: Built-in tools that only do anything when the sandbox allows writes.
_WRITE_BUILTIN_TOOLS = ("apply_patch",)


def _builtin_tools(agent: UiPathCodexAgent) -> list[str]:
    """Codex built-in tools the agent's sandbox makes usable.

    Codex always exposes ``shell`` (a ``"read-only"`` sandbox still runs
    read-only commands, and those calls surface as ``tools`` state events),
    so it is always listed; ``apply_patch`` is listed only when the sandbox
    allows writes.
    """
    tools = list(_ALWAYS_BUILTIN_TOOLS)
    if agent.sandbox != "read-only":
        tools.extend(_WRITE_BUILTIN_TOOLS)
    return tools


def get_agent_graph(agent: UiPathCodexAgent) -> UiPathRuntimeGraph:
    """Build a visualization graph for a Codex agent.

    The agent is a single model node; durable tools, MCP servers and Codex
    built-ins are aggregated into one ``tools`` node, mirroring
    uipath-claude-sdk. Node ids are fixed (``agent``/``tools``) because the
    runtime's state events reference them by id.
    """
    nodes: list[UiPathRuntimeNode] = [
        UiPathRuntimeNode(
            id="__start__", name="__start__", type="__start__", subgraph=None
        ),
        UiPathRuntimeNode(
            id="agent",
            name=agent.name or "agent",
            type="model",
            subgraph=None,
            metadata={"model_name": agent.model},
        ),
    ]
    edges: list[UiPathRuntimeEdge] = [
        UiPathRuntimeEdge(source="__start__", target="agent", label="input"),
    ]

    uipath_tools = [tool.name for tool in agent.tools or []]
    server_names = list(agent.mcp_servers or {})
    builtin_tools = _builtin_tools(agent)
    tool_names = [*uipath_tools, *server_names, *builtin_tools]

    if tool_names:
        nodes.append(
            UiPathRuntimeNode(
                id="tools",
                name="tools",
                type="tool",
                subgraph=None,
                metadata={
                    "tool_names": tool_names,
                    "tool_count": len(tool_names),
                    "mcp_servers": server_names,
                    "builtin_tools": builtin_tools,
                },
            )
        )
        edges.append(UiPathRuntimeEdge(source="agent", target="tools", label=None))
        edges.append(UiPathRuntimeEdge(source="tools", target="agent", label=None))

    nodes.append(
        UiPathRuntimeNode(id="__end__", name="__end__", type="__end__", subgraph=None)
    )
    edges.append(UiPathRuntimeEdge(source="agent", target="__end__", label="output"))
    return UiPathRuntimeGraph(nodes=nodes, edges=edges)


def get_entrypoint_schema(
    agent: UiPathCodexAgent, entrypoint: str
) -> UiPathRuntimeSchema:
    """Build the UiPath runtime schema for a Codex agent entrypoint.

    As in the other integrations, ``filePath`` is the entrypoint name (a
    deployed job passes it back to ``uipath run``, so it must be the
    ``codex.json`` key, not a path on the packing machine) and ``uniqueId``
    is a fresh UUID (Orchestrator rejects a package whose entry point id is
    not a GUID).

    ``metadata.settings.model`` is what ``uipath eval``/``run``/``debug``
    read as the agent model (LLM mocking, input generation and
    ``same-as-agent`` judges default to it). Those calls go through the
    gateway's chat-completions path rather than the Responses path Codex
    uses; the default Codex model (``gpt-5.6-terra``) was checked to answer
    there, structured output included. An agent on a Responses-only model
    should set an explicit model on its mocking strategies and judges.
    """
    return UiPathRuntimeSchema(
        filePath=entrypoint,
        uniqueId=str(uuid4()),
        type="agent",
        input=_input_schema(agent),
        output=_output_schema(agent),
        graph=get_agent_graph(agent),
        metadata={"settings": {"model": agent.model}},
    )


__all__ = [
    "get_agent_graph",
    "get_entrypoint_schema",
    "prompt_fields",
    "to_strict_output_schema",
]
