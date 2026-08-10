"""Opt-in live coverage for Claude background-task lifecycle handling."""

from __future__ import annotations

import asyncio
import os

import pytest
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    tool,
)
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus
from uipath.runtime.events import UiPathRuntimeStateEvent

from tests.conftest import make_session_paths
from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("UIPATH_CLAUDE_LIVE_TEST") != "1",
        reason="set UIPATH_CLAUDE_LIVE_TEST=1 to call the live UiPath LLM Gateway",
    ),
]

MARKER = "BACKGROUND_WORKER_FINISHED"


@tool("slow_marker", "Wait briefly, then return the live-test marker.", {})
async def slow_marker(_args: dict[str, object]) -> dict[str, object]:
    await asyncio.sleep(3)
    return {"content": [{"type": "text", "text": MARKER}]}


async def test_runtime_waits_for_background_subagent_follow_up(tmp_path) -> None:
    """The first result frame must not close a still-running SDK-MCP worker."""
    server = create_sdk_mcp_server("live-background", tools=[slow_marker])
    agent = UiPathClaudeAgent(
        options=ClaudeAgentOptions(
            system_prompt=(
                "Delegate the request to the worker exactly once. The worker is "
                "configured to run in the background. Do not poll it and do not "
                "call any other tool. End the spawning turn immediately. When its "
                f"completion arrives, answer with exactly {MARKER}."
            ),
            allowed_tools=["Agent"],
            mcp_servers={"live-background": server},
            agents={
                "worker": AgentDefinition(
                    description="Runs the slow marker tool for the live test.",
                    prompt=(
                        "Call mcp__live-background__slow_marker exactly once, wait "
                        f"for its result, then return exactly {MARKER}."
                    ),
                    tools=["mcp__live-background__slow_marker"],
                    mcpServers=["live-background"],
                    model="inherit",
                    background=True,
                    maxTurns=4,
                    permissionMode="bypassPermissions",
                )
            },
            max_turns=8,
            permission_mode="bypassPermissions",
        ),
        uipath_llm=UiPathModel("claude-sonnet-4-5"),
    )
    storage = SqliteResumableStorage(str(tmp_path / "state.db"))
    runtime = UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=ClaudeSessionStore(storage, "live-background"),
        session_paths=make_session_paths(tmp_path, "live-background"),
        runtime_id="live-background",
    )

    try:
        async with asyncio.timeout(240):
            events = [
                event
                async for event in runtime.stream(
                    {"input": "Run the background worker."}
                )
            ]
    finally:
        await storage.dispose()

    result = events[-1]
    assert isinstance(result, UiPathRuntimeResult)
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert isinstance(result.output, dict)
    assert MARKER in result.output["result"]

    task_events = [
        event
        for event in events
        if isinstance(event, UiPathRuntimeStateEvent) and event.node_name == "task"
    ]
    assert task_events
    assert any(event.payload.get("status") == "completed" for event in task_events)
