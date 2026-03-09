"""Tests for resumable runtime wiring in factory."""

import os

import pytest
from agents import Agent, SQLiteSession
from uipath.runtime import UiPathResumableRuntime, UiPathRuntimeContext

from uipath_openai_agents.runtime.factory import UiPathOpenAIAgentRuntimeFactory
from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime
from uipath_openai_agents.runtime.storage import SqliteResumableStorage


@pytest.mark.asyncio
async def test_factory_wraps_runtime_with_resumable_runtime(tmp_path):
    context = UiPathRuntimeContext(
        runtime_dir=str(tmp_path / "runtime"),
        state_file="openai_state.db",
    )
    factory = UiPathOpenAIAgentRuntimeFactory(context=context)

    try:
        runtime = await factory._create_runtime_instance(
            agent=Agent(name="test-agent", instructions="test"),
            runtime_id="runtime-1",
            entrypoint="agent",
        )

        assert isinstance(runtime, UiPathResumableRuntime)
        assert isinstance(runtime.delegate, UiPathOpenAIAgentRuntime)
        assert isinstance(runtime.storage, SqliteResumableStorage)
        assert isinstance(runtime.delegate._session, SQLiteSession)
        assert runtime.delegate._session.session_id == "runtime-1"

        storage = await factory.get_storage()
        assert storage is not None
        assert runtime.storage is storage
    finally:
        await factory.dispose()


def test_get_connection_string_cleans_state_for_fresh_local_run(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "state.db"
    state_path.write_text("stale state")

    context = UiPathRuntimeContext(
        runtime_dir=str(runtime_dir),
        state_file="state.db",
        resume=False,
        job_id=None,
        keep_state_file=False,
    )
    factory = UiPathOpenAIAgentRuntimeFactory(context=context)

    resolved = factory._get_connection_string()

    assert resolved == str(state_path)
    assert not os.path.exists(state_path)
