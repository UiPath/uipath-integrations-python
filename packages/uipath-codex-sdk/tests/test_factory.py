"""Factory wiring: state file, CODEX_HOME, resumable wrapping."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from aiohttp import ClientSession
from uipath.runtime import UiPathResumableRuntime, UiPathRuntimeContext

from uipath_codex_sdk import RemoteMcpServer, StdioMcpServer
from uipath_codex_sdk.mcp import build_mcp_config_overrides
from uipath_codex_sdk.runtime.factory import UiPathCodexRuntimeFactory
from uipath_codex_sdk.runtime.gateway import CodexGatewayShim
from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage
from uipath_codex_sdk.runtime.tool_host import DurableToolHost

from ._typing import sqlite_storage

MAIN = """
from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool
agent = UiPathCodexAgent(model="m", prompt="hi", tools=[uipath_process_tool("p")])
"""

PLAIN_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent
agent = UiPathCodexAgent(model="m", prompt="hi")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "codex.json").write_text('{"agents": {"agent": "main.py:agent"}}')
    (tmp_path / "main.py").write_text(MAIN)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed(project: Path) -> tuple[Path, Path]:
    rd = project / "__uipath"
    (rd / "codex_home" / "sessions").mkdir(parents=True)
    marker = rd / "codex_home" / "sessions" / "old.jsonl"
    marker.write_text("{}\n")
    db = rd / "state.db"
    db.write_bytes(b"")
    (rd / "state.db-wal").write_bytes(b"")
    return db, marker


@pytest.mark.parametrize(
    ("kwargs", "reset"),
    [
        ({}, True),
        ({"resume": True}, False),
        ({"job_id": "job-1"}, False),
        ({"keep_state_file": True}, False),
    ],
)
async def test_state_reset_rule(project, kwargs, reset):
    db, marker = _seed(project)
    ctx = UiPathRuntimeContext(runtime_dir=str(project / "__uipath"), **kwargs)
    factory = UiPathCodexRuntimeFactory(ctx)
    try:
        runtime = await factory.new_runtime("agent", "rt")
        assert isinstance(runtime, UiPathResumableRuntime)
        delegate = runtime.delegate
        assert isinstance(delegate, UiPathCodexRuntime)
        assert delegate._codex_home == str(
            (project / "__uipath" / "codex_home").resolve()
        )
        assert marker.exists() is (not reset)
        assert (project / "__uipath" / "state.db-wal").exists() is (not reset)
        storage = await factory.get_storage()
        assert isinstance(storage, SqliteResumableStorage)
        assert storage is runtime.storage is delegate.storage
        assert await factory.get_storage() is storage
    finally:
        await factory.dispose()


async def test_agent_without_durable_tools_keeps_its_codex_home(project):
    """No durable tools: no run-scoped CODEX_HOME, and nothing of it is wiped."""
    (project / "main.py").write_text(PLAIN_MAIN)
    _, marker = _seed(project)
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(runtime_dir=str(project / "__uipath"))
    )
    try:
        runtime = await factory.new_runtime("agent", "rt")
        assert isinstance(runtime, UiPathResumableRuntime)
        assert isinstance(runtime.delegate, UiPathCodexRuntime)
        assert runtime.delegate._codex_home is None
        assert runtime.delegate._codex_home_dir({}) is None
        assert marker.exists()
    finally:
        await factory.dispose()


async def test_explicit_state_file_path_is_never_reset(project):
    """langchain's rule: an explicit state_file_path is kept on a fresh run,
    and so is the run-scoped CODEX_HOME next to it."""
    db, marker = _seed(project)
    db.write_bytes(b"keep")
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=str(project / "__uipath"), state_file_path=str(db)
        )
    )
    try:
        assert factory._should_reset_state() is False
        await factory.new_runtime("agent", "rt")
        assert marker.exists()
        assert (project / "__uipath" / "state.db-wal").exists()
    finally:
        await factory.dispose()


async def test_explicit_state_file_path(project):
    path = project / "custom" / "s.db"
    factory = UiPathCodexRuntimeFactory(UiPathRuntimeContext(state_file_path=str(path)))
    try:
        storage = await sqlite_storage(factory)
        await storage.set_value("rt", "ns", "k", "v")
        assert path.exists()
    finally:
        await factory.dispose()


def test_parallel_flag_override():
    overrides = build_mcp_config_overrides(
        {
            "a": RemoteMcpServer(url="http://x/mcp", supports_parallel_tool_calls=True),
            "b": StdioMcpServer(command="python", supports_parallel_tool_calls=False),
            "c": RemoteMcpServer(url="http://y/mcp"),
        }
    )
    assert "mcp_servers.a.supports_parallel_tool_calls=true" in overrides
    assert "mcp_servers.b.supports_parallel_tool_calls=false" in overrides
    assert not any(o.startswith("mcp_servers.c.supports_parallel") for o in overrides)


async def test_gateway_mounts_tool_host_with_its_own_token(tmp_path):
    storage = SqliteResumableStorage(str(tmp_path / "s.db"))
    from uipath_codex_sdk.runtime.session_store import CodexSessionStore

    host = DurableToolHost([], store=CodexSessionStore(storage, "rt"), runtime_id="rt")
    shim = CodexGatewayShim(default_model="m", settings=object(), mcp_handler=host)  # type: ignore[arg-type]
    await shim.start()
    try:
        assert shim.mcp_url.endswith("/mcp") and shim.inflight == 0
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        async with ClientSession() as s:
            async with s.post(
                shim.mcp_url,
                json=msg,
                headers={"Authorization": f"Bearer {host.token}"},
            ) as r:
                assert r.status == 200
                assert (await r.json())["result"] == {"tools": []}
            # the gateway key does not open the tool host...
            async with s.post(
                shim.mcp_url,
                json=msg,
                headers={"Authorization": f"Bearer {shim.api_key}"},
            ) as r:
                assert r.status == 401
            # ...and the tool host token does not open the model routes
            async with s.get(
                shim.base_url + "/models",
                headers={"Authorization": f"Bearer {host.token}"},
            ) as r:
                assert r.status == 401
            async with s.get(
                shim.base_url + "/models",
                headers={"Authorization": f"Bearer {shim.api_key}"},
            ) as r:
                assert r.status == 200
    finally:
        await shim.stop()
        await storage.dispose()


async def test_schema_ids_are_deployable(project):
    # A deployed job hands filePath back to `uipath run`; a path from the
    # packing machine would not resolve there.
    factory = UiPathCodexRuntimeFactory(UiPathRuntimeContext())
    try:
        runtime = await factory.new_runtime("agent", "run-1")
        schema = await runtime.get_schema()
        assert schema.file_path == "agent"
        UUID(schema.unique_id)  # Orchestrator requires a GUID
    finally:
        await factory.dispose()


IMMEDIATE_MAIN = """
from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool
agent = UiPathCodexAgent(model="m", prompt="hi", tools=[
    uipath_integration_tool("web_search", connection_id="c", tool_path="/x")
])
"""


async def test_agent_with_only_immediate_tools_keeps_its_codex_home(project):
    (project / "main.py").write_text(IMMEDIATE_MAIN)
    _, marker = _seed(project)
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(runtime_dir=str(project / "__uipath"))
    )
    try:
        runtime = await factory.new_runtime("agent", "rt")
        assert isinstance(runtime, UiPathResumableRuntime)
        delegate = runtime.delegate
        assert isinstance(delegate, UiPathCodexRuntime)
        assert delegate._codex_home is None
        durable, immediate = delegate._agent_tools()
        assert durable == {} and list(immediate) == ["web_search"]
        assert delegate._codex_home_dir(durable) is None
        assert delegate._hermetic(durable) is False
        assert marker.exists()
    finally:
        await factory.dispose()


async def test_reports_the_framework(project):
    """Coded run telemetry and governance audit events tell frameworks apart."""
    factory = UiPathCodexRuntimeFactory(
        UiPathRuntimeContext(runtime_dir=str(project / "__uipath"))
    )
    try:
        settings = await factory.get_settings()
        assert settings is not None
        assert settings.agent_type == "uipath_coded"
        assert settings.agent_framework == "codex_sdk"
    finally:
        await factory.dispose()
