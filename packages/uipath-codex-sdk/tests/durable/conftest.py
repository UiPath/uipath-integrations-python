"""Fixtures for the offline durable-tool end-to-end tests.

Real bundled codex 0.161 binary + real gateway shim / tool host / runtime /
factory / ``UiPathResumableRuntime`` / ``UiPathResumeTriggerHandler``; only the
UiPath tenant is faked (Orchestrator jobs, and the LLM Gateway replaced by the
contract ``MockResponsesServer``).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ..contract.harness import BackgroundLoop, descendants, kill_pids
from ..contract.mock_responses import MockResponsesServer
from .fakes import FakeOrchestrator, mock_upstream_class

AGENT_MAIN = textwrap.dedent(
    """
    from pydantic import BaseModel

    from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool


    class AddIn(BaseModel):
        a: int
        b: int


    agent = UiPathCodexAgent(
        model="mock-model",
        prompt="PLEASE_CALL_DURABLE: add the numbers with a_plus_b.",
        tools=[
            uipath_process_tool(
                "a_plus_b",
                folder_path="Shared/Durable",
                description="Adds a and b.",
                input_schema=AddIn,
            )
        ],
        durable_settle_timeout=20.0,
    )
    """
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    monkeypatch.delenv("UIPATH_CODEX_BIN", raising=False)
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    for var in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    before = set(descendants(os.getpid()))
    yield
    kill_pids([p for p in descendants(os.getpid()) if p not in before])


@pytest.fixture
async def mock_model():
    server = MockResponsesServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
def world(monkeypatch) -> FakeOrchestrator:
    w = FakeOrchestrator()
    import uipath.platform.resume_triggers._protocol as protocol

    import uipath_codex_sdk.tools as tools

    monkeypatch.setattr(tools, "_uipath_client", w.client)
    monkeypatch.setattr(protocol, "UiPath", w.client)
    return w


@pytest.fixture
def upstream(monkeypatch, mock_model):
    import uipath_codex_sdk.runtime.gateway as gateway

    monkeypatch.setattr(
        gateway,
        "UpstreamRegistry",
        mock_upstream_class(f"{mock_model.base_url}/responses"),
    )
    monkeypatch.setattr(gateway, "get_default_client_settings", lambda: object())
    return mock_model


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "codex.json").write_text('{"agents": {"agent": "main.py:agent"}}')
    (proj / "main.py").write_text(AGENT_MAIN)
    monkeypatch.chdir(proj)
    return proj


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, world: FakeOrchestrator):
    """A project for the real ``uipath`` CLI, which calls ``asyncio.run``
    itself: the mock model server runs on a background loop."""
    loop = BackgroundLoop()
    server = MockResponsesServer()
    loop.run(server.start())
    import uipath_codex_sdk.runtime.gateway as gateway

    monkeypatch.setattr(
        gateway, "UpstreamRegistry", mock_upstream_class(f"{server.base_url}/responses")
    )
    monkeypatch.setattr(gateway, "get_default_client_settings", lambda: object())
    proj = tmp_path / "cli_project"
    proj.mkdir()
    (proj / "codex.json").write_text('{"agents": {"agent": "main.py:agent"}}')
    (proj / "main.py").write_text(AGENT_MAIN)
    (proj / "uipath.json").write_text("{}")
    (proj / "input.json").write_text("{}")
    monkeypatch.chdir(proj)
    for var in ("UIPATH_URL", "UIPATH_ACCESS_TOKEN", "UIPATH_TELEMETRY_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("UIPATH_TELEMETRY_ENABLED", "false")
    try:
        yield proj, server
    finally:
        loop.run(server.stop())
        loop.close()
