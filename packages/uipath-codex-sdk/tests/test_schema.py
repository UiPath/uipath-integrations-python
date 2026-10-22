"""Entrypoint schema: runtime graph, settings metadata, prompt-derived input."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel
from uipath._cli.cli_init import write_mermaid_files

from uipath_codex_sdk import StdioMcpServer, UiPathCodexAgent, uipath_process_tool
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime
from uipath_codex_sdk.runtime.schema import (
    get_agent_graph,
    get_entrypoint_schema,
    prompt_fields,
)

TEMPLATE = (
    Path(__file__).parents[1]
    / "src"
    / "uipath_codex_sdk"
    / "_cli"
    / "_templates"
    / "main.py.template"
)


def _runtime(agent: UiPathCodexAgent) -> UiPathCodexRuntime:
    return UiPathCodexRuntime(
        agent=agent, runtime_id="rt", entrypoint="agent", file_path="/tmp/main.py"
    )


def _edges(agent: UiPathCodexAgent) -> set[tuple[str, str, str | None]]:
    return {(e.source, e.target, e.label) for e in get_agent_graph(agent).edges}


def test_graph_read_only_agent_lists_shell_only():
    graph = get_agent_graph(UiPathCodexAgent(model="m", name="joker"))
    nodes = {n.id: n for n in graph.nodes}

    assert list(nodes) == ["__start__", "agent", "tools", "__end__"]
    assert nodes["agent"].type == "model"
    assert nodes["agent"].name == "joker"
    assert nodes["agent"].metadata == {"model_name": "m"}
    assert nodes["tools"].type == "tool"
    assert nodes["tools"].metadata == {
        "tool_names": ["shell"],
        "tool_count": 1,
        "mcp_servers": [],
        "builtin_tools": ["shell"],
    }
    assert _edges(UiPathCodexAgent(model="m")) == {
        ("__start__", "agent", "input"),
        ("agent", "tools", None),
        ("tools", "agent", None),
        ("agent", "__end__", "output"),
    }


def test_graph_tools_node_aggregates_durable_mcp_and_builtins():
    agent = UiPathCodexAgent(
        model="m",
        sandbox="full-access",
        tools=[uipath_process_tool("Invoice Approval", tool_name="approve")],
        mcp_servers={"orders": StdioMcpServer(command="python")},
    )
    tools = next(n for n in get_agent_graph(agent).nodes if n.id == "tools")

    assert tools.metadata == {
        "tool_names": ["approve", "orders", "shell", "apply_patch"],
        "tool_count": 4,
        "mcp_servers": ["orders"],
        "builtin_tools": ["shell", "apply_patch"],
    }


def test_schema_carries_graph_and_model_settings():
    schema = get_entrypoint_schema(UiPathCodexAgent(model="gpt-x"), "agent")

    assert schema.metadata == {"settings": {"model": "gpt-x"}}
    dumped = json.loads(schema.model_dump_json(by_alias=True))
    assert [n["id"] for n in dumped["graph"]["nodes"]] == [
        "__start__",
        "agent",
        "tools",
        "__end__",
    ]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Tell me a joke about {topic}.", ["topic"]),
        ("{a} and {b} and {a!r}", ["a", "b"]),
        ("{order.id} / {items[0]} / {order.total}", ["order", "items"]),
        ("{} {0} {{literal}}", []),
        ("{value:{width}}", ["value", "width"]),
        ("broken {", []),
        ("no placeholders", []),
    ],
)
def test_prompt_fields(prompt: str, expected: list[str]):
    assert prompt_fields(prompt) == expected


def test_input_schema_derived_from_prompt_placeholders():
    agent = UiPathCodexAgent(model="m", prompt="Summarize {doc} for {audience}.")
    schema = get_entrypoint_schema(agent, "agent")

    assert schema.input == {
        "type": "object",
        "properties": {"doc": {"type": "string"}, "audience": {"type": "string"}},
        "required": ["doc", "audience"],
    }
    rendered = _runtime(agent)._render_prompt({"doc": "d", "audience": "kids"})
    assert rendered == "Summarize d for kids."
    with pytest.raises(UiPathCodexRuntimeError):
        _runtime(agent)._render_prompt({"doc": "d"})


def test_structured_placeholders_are_not_typed_as_strings():
    agent = UiPathCodexAgent(
        model="m", prompt="Ship {order[id]} to {city}, gift: {order}."
    )
    schema = get_entrypoint_schema(agent, "agent")

    assert schema.input == {
        "type": "object",
        "properties": {"order": {}, "city": {"type": "string"}},
        "required": ["order", "city"],
    }
    runtime = _runtime(agent)
    order = {"id": "A1"}
    assert runtime._render_prompt({"order": order, "city": "Oslo"}) == (
        f"Ship A1 to Oslo, gift: {order}."
    )


@pytest.mark.parametrize(
    ("prompt", "fields"),
    [
        ("Ship {order[id]}", {"order": "A123"}),
        ("Ship to {order.city}", {"order": {"city": "Oslo"}}),
        ("First {items[3]}", {"items": ["a"]}),
    ],
)
def test_input_that_does_not_fit_the_template_is_input_invalid(
    prompt: str, fields: dict
):
    with pytest.raises(UiPathCodexRuntimeError) as err:
        _runtime(UiPathCodexAgent(model="m", prompt=prompt))._render_prompt(fields)
    assert "INPUT_INVALID" in str(err.value.error_info.code)


def test_input_schema_permissive_without_placeholders():
    for prompt in (None, "Say hi.", "positional {}"):
        schema = get_entrypoint_schema(UiPathCodexAgent(model="m", prompt=prompt), "a")
        assert schema.input == {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }


def test_declared_input_schema_wins_over_placeholders():
    class In(BaseModel):
        topic: str
        count: int = 1

    agent = UiPathCodexAgent(model="m", prompt="{topic}", input_schema=In)
    schema = get_entrypoint_schema(agent, "agent")

    assert schema.input["properties"]["count"]["type"] == "integer"
    assert schema.input["required"] == ["topic"]


def test_scaffold_template_declares_input_schema():
    namespace: dict[str, object] = {}
    exec(compile(TEMPLATE.read_text(), str(TEMPLATE), "exec"), namespace)
    agent = namespace["agent"]
    assert isinstance(agent, UiPathCodexAgent)

    schema = get_entrypoint_schema(agent, "agent")
    assert schema.input["required"] == ["topic"]
    assert schema.input["properties"]["topic"]["type"] == "string"
    assert _runtime(agent)._render_prompt({"topic": "robots"}).endswith("about robots.")


def test_uipath_init_writes_mermaid_for_graph(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = UiPathCodexAgent(model="m", tools=[uipath_process_tool("p")])

    paths = write_mermaid_files([get_entrypoint_schema(agent, "agent")])

    assert [os.path.basename(p) for p in paths] == ["agent.mermaid"]
    text = Path(paths[0]).read_text()
    for node_id in ("__start__", "agent", "tools", "__end__"):
        assert node_id in text
