"""Tests for runtime/factory.py — UiPathGoogleADKRuntimeFactory."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents import LlmAgent
from google.adk.sessions.session import Session

from uipath_google_adk.runtime import factory as factory_mod
from uipath_google_adk.runtime.errors import UiPathGoogleADKRuntimeError
from uipath_google_adk.runtime.factory import UiPathGoogleADKRuntimeFactory
from uipath_google_adk.runtime.runtime import UiPathGoogleADKRuntime


@pytest.fixture(autouse=True)
def _no_instrumentation(monkeypatch):
    """Avoid real OpenTelemetry instrumentation side effects."""
    monkeypatch.setattr(factory_mod, "GoogleADKInstrumentor", lambda: MagicMock())


def _make_context(tmp_path, resume=False, job_id=None, keep=False):
    state_file = tmp_path / "state.db"
    return SimpleNamespace(
        resolved_state_file_path=str(state_file),
        resume=resume,
        job_id=job_id,
        keep_state_file=keep,
    )


def _write_config(tmp_path, agents):
    (tmp_path / "google_adk.json").write_text(json.dumps({"agents": agents}))


class TestDiscoverEntrypoints:
    def test_returns_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        assert factory.discover_entrypoints() == []

    def test_returns_agent_names(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"agent": "main.py:agent"})
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        assert factory.discover_entrypoints() == ["agent"]


class TestGetConnectionString:
    def test_deletes_stale_state_when_not_resuming(self, tmp_path):
        ctx = _make_context(tmp_path, resume=False, job_id=None, keep=False)
        stale = tmp_path / "state.db"
        stale.write_text("old")
        factory = UiPathGoogleADKRuntimeFactory(ctx)
        path = factory._get_connection_string()
        assert path == str(stale)
        assert not stale.exists()

    def test_keeps_state_when_resuming(self, tmp_path):
        ctx = _make_context(tmp_path, resume=True)
        keep = tmp_path / "state.db"
        keep.write_text("keep")
        factory = UiPathGoogleADKRuntimeFactory(ctx)
        factory._get_connection_string()
        assert keep.exists()


class TestLoadAgent:
    @pytest.mark.asyncio
    async def test_missing_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        with pytest.raises(UiPathGoogleADKRuntimeError, match="configuration"):
            await factory._load_agent("agent")

    @pytest.mark.asyncio
    async def test_unknown_entrypoint_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"agent": "main.py:agent"})
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        with pytest.raises(UiPathGoogleADKRuntimeError, match="not found"):
            await factory._load_agent("missing")


class TestResolveAgentCaches:
    @pytest.mark.asyncio
    async def test_second_resolve_uses_cache(self, tmp_path, monkeypatch):
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        agent = LlmAgent(name="asst", model="gemini-2.0-flash")
        load_agent = AsyncMock(return_value=agent)
        monkeypatch.setattr(factory, "_load_agent", load_agent)
        first = await factory._resolve_agent("agent")
        second = await factory._resolve_agent("agent")
        assert first is second
        load_agent.assert_awaited_once()


class TestNewRuntime:
    @pytest.mark.asyncio
    async def test_creates_runtime_with_session(self, tmp_path, monkeypatch):
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        agent = LlmAgent(name="asst", model="gemini-2.0-flash")
        monkeypatch.setattr(factory, "_resolve_agent", AsyncMock(return_value=agent))

        session = Session(id="rt-1", app_name="uipath", user_id="uipath-user", state={})
        session_service = MagicMock()
        session_service.get_session = AsyncMock(return_value=None)
        session_service.create_session = AsyncMock(return_value=session)
        monkeypatch.setattr(
            factory, "_get_session_service", AsyncMock(return_value=session_service)
        )

        runtime = await factory.new_runtime("agent", "rt-1")
        assert isinstance(runtime, UiPathGoogleADKRuntime)
        assert runtime.runtime_id == "rt-1"
        assert runtime.agent is agent
        session_service.create_session.assert_awaited_once()


class TestDispose:
    @pytest.mark.asyncio
    async def test_dispose_cleans_loaders_and_cache(self, tmp_path):
        factory = UiPathGoogleADKRuntimeFactory(_make_context(tmp_path))
        loader = MagicMock()
        loader.cleanup = AsyncMock()
        factory._agent_loaders = {"agent": loader}
        factory._agent_cache = {"agent": MagicMock()}
        await factory.dispose()
        loader.cleanup.assert_awaited_once()
        assert factory._agent_cache == {}
        assert factory._agent_loaders == {}
