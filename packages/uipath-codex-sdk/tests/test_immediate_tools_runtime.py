"""Runtime configuration of immediate (non-durable) UiPath tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from uipath_codex_sdk import (
    RemoteMcpServer,
    UiPathCodexAgent,
    uipath_integration_tool,
    uipath_process_tool,
)
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.runtime import (
    DURABLE_TOOL_TIMEOUT_SEC,
    UiPathCodexRuntime,
)
from uipath_codex_sdk.runtime.schema import get_agent_graph
from uipath_codex_sdk.runtime.tool_host import SERVER_NAME, DurableToolHost


def _is_tool(name: str = "web_search", timeout: float | None = 300.0):
    return uipath_integration_tool(
        name, connection_id="c", tool_path="/x", timeout_sec=timeout
    )


def _runtime(tools: list[Any]) -> UiPathCodexRuntime:
    agent = UiPathCodexAgent(model="m", prompt="p", tools=tools)
    return UiPathCodexRuntime(
        agent=agent, runtime_id="rt", entrypoint="agent", file_path="/tmp/x/main.py"
    )


def test_tools_split_into_durable_and_immediate():
    rt = _runtime([uipath_process_tool("a_plus_b"), _is_tool()])
    durable, immediate = rt._agent_tools()
    assert list(durable) == ["a_plus_b"] and list(immediate) == ["web_search"]
    assert list(rt._durable_tools()) == ["a_plus_b"]


def test_duplicate_names_across_kinds_are_rejected():
    rt = _runtime([uipath_process_tool("a_plus_b"), _is_tool("a_plus_b")])
    with pytest.raises(UiPathCodexRuntimeError) as info:
        rt._agent_tools()
    assert "CONFIG_INVALID" in str(info.value.error_info.code)
    assert "Two tools are named 'a_plus_b'" in info.value.error_info.detail


class Both:
    name = "both"
    description = "?"
    input_schema: dict[str, Any] = {"type": "object"}

    def validate_args(self, args):
        return args

    async def start(self, args, ctx):
        return None

    async def finish(self, args, value):
        return ""

    async def call(self, args, ctx):
        return ""


class Neither:
    name = "neither"
    description = "?"


@pytest.mark.parametrize("tool", [Both(), Neither()], ids=["both", "neither"])
def test_a_tool_must_be_exactly_one_kind(tool):
    with pytest.raises(UiPathCodexRuntimeError) as info:
        _runtime([tool])._agent_tools()
    assert "CONFIG_INVALID" in str(info.value.error_info.code)


def test_immediate_only_server_gets_a_short_tool_timeout():
    rt = _runtime([_is_tool(timeout=45), _is_tool("other", timeout=None)])
    _, immediate = rt._agent_tools()
    host = DurableToolHost(list(immediate.values()), store=None, runtime_id="rt")
    shim = SimpleNamespace(mcp_url="http://127.0.0.1:1/mcp")
    server = rt._mcp_servers(shim, host)[SERVER_NAME]  # type: ignore[arg-type]
    assert isinstance(server, RemoteMcpServer)
    assert server.tool_timeout_sec == 330.0  # 300 (default) + margin
    assert server.omit_tools_from == ["deferred", "code_mode"]
    assert rt._codex_home_dir({}) is None and rt._hermetic({}) is False


def test_durable_server_keeps_the_day_long_timeout():
    rt = _runtime([uipath_process_tool("a_plus_b"), _is_tool()])
    durable, immediate = rt._agent_tools()
    host = DurableToolHost(
        [*durable.values(), *immediate.values()],
        store=object(),  # type: ignore[arg-type]
        runtime_id="rt",
    )
    shim = SimpleNamespace(mcp_url="http://127.0.0.1:1/mcp")
    server = rt._mcp_servers(shim, host)[SERVER_NAME]  # type: ignore[arg-type]
    assert isinstance(server, RemoteMcpServer)
    assert server.tool_timeout_sec == DURABLE_TOOL_TIMEOUT_SEC


def test_graph_lists_immediate_tools():
    agent = UiPathCodexAgent(
        model="m", tools=[uipath_process_tool("a_plus_b"), _is_tool()]
    )
    tools = next(n for n in get_agent_graph(agent).nodes if n.id == "tools")
    assert tools.metadata is not None
    assert tools.metadata["tool_names"][:2] == ["a_plus_b", "web_search"]
