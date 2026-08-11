"""Tests for durable suspension: defer, persist, resume, deliver, clear."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookContext,
    HookJSONOutput,
    HookMatcher,
    PreToolUseHookInput,
    ResultMessage,
    tool,
)
from claude_agent_sdk.types import DeferredToolUse
from mcp.types import CallToolRequest, CallToolRequestParams
from uipath.core.triggers import UiPathResumeTriggerType
from uipath.platform.common import CreateTask, WaitJob
from uipath.platform.orchestrator.job import Job
from uipath.platform.resume_triggers import UiPathResumeTriggerCreator
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus, UiPathStreamOptions

from tests.conftest import make_result_message, make_session_paths
from uipath_claude_sdk import ClaudeAgent, interrupt, uipath_tool_server
from uipath_claude_sdk.interrupts import PendingSuspend
from uipath_claude_sdk.runtime.errors import UiPathClaudeSDKRuntimeError
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore

SERVER_KEY = "tickets"
TOOL_NAME = "ask"
INTERRUPT_TOOL = f"mcp__{SERVER_KEY}__{TOOL_NAME}"
QUESTION = "Should I ship it?"
TOOL_USE_ID = "toolu_1"
ARGS = {"question": QUESTION}


@tool(TOOL_NAME, "Ask a human and wait for the answer.", {"question": str})
async def ask(args: dict[str, Any]) -> dict[str, Any]:
    answer = await interrupt(args["question"])
    return {"content": [{"type": "text", "text": str(answer)}]}


def hitl_options(**overrides: Any) -> ClaudeAgentOptions:
    servers: dict[str, Any] = {SERVER_KEY: uipath_tool_server(SERVER_KEY, tools=[ask])}
    servers.update(overrides.pop("extra_servers", {}))
    return ClaudeAgentOptions(mcp_servers=servers, **overrides)


def _permission_decision(output: HookJSONOutput) -> str:
    """The decision the CLI reads back from the hook, as JSON would carry it."""
    payload: dict[str, Any] = dict(output)
    if not payload:
        return "none"
    decision: str = payload["hookSpecificOutput"]["permissionDecision"]
    return decision


def _output(result: UiPathRuntimeResult) -> dict[str, Any]:
    """The result's output as the interrupt map the runtime yields."""
    assert isinstance(result.output, dict)
    return result.output


class InterruptClient:
    """Fake CLI that runs the real PreToolUse hook and honours its decision.

    On ``defer`` it reports a parked call the way the CLI does. On ``allow`` it
    invokes the real in-process MCP handler, so the resumed payload travels the
    same path it does in a live run.
    """

    last_options: ClaudeAgentOptions | None = None
    last_query: str | None = None
    tool_results: list[str] = []
    session_id: str = "session-1"

    def __init__(self, options: ClaudeAgentOptions) -> None:
        type(self).last_options = options
        self._options = options

    async def __aenter__(self) -> InterruptClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def query(self, message: str) -> None:
        type(self).last_query = message

    async def receive_response(self):
        """Stop at the result, as the SDK's own convenience method does."""
        async for message in self.receive_messages():
            yield message
            if isinstance(message, ResultMessage):
                return

    async def receive_messages(self):
        decision = await self._run_hook()
        if decision == "defer":
            yield make_result_message(
                result=None,
                session_id=type(self).session_id,
                deferred_tool_use=DeferredToolUse(
                    id=TOOL_USE_ID, name=INTERRUPT_TOOL, input=ARGS
                ),
            )
            return
        if decision == "allow":
            type(self).tool_results.append(await self._call_tool())
        yield make_result_message(result="done", session_id=type(self).session_id)

    async def _run_hook(self) -> str:
        matchers = (self._options.hooks or {}).get("PreToolUse")
        if not matchers:
            return "none"
        hook_input: PreToolUseHookInput = {
            "hook_event_name": "PreToolUse",
            "tool_name": INTERRUPT_TOOL,
            "tool_input": ARGS,
            "tool_use_id": TOOL_USE_ID,
            "session_id": type(self).session_id,
            "transcript_path": "",
            "cwd": "",
        }
        output = await matchers[0].hooks[0](
            hook_input,
            TOOL_USE_ID,
            HookContext(signal=None),
        )
        return _permission_decision(output)

    async def _call_tool(self) -> str:
        servers: Any = self._options.mcp_servers
        server = servers[SERVER_KEY]["instance"]
        for request_type, handler in server.request_handlers.items():
            if request_type.__name__ != "CallToolRequest":
                continue
            result = await handler(
                CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name=TOOL_NAME, arguments=ARGS),
                )
            )
            return result.root.content[0].text
        raise AssertionError("The tool server exposes no CallToolRequest handler.")


@pytest.fixture
def interrupt_client(monkeypatch: pytest.MonkeyPatch) -> type[InterruptClient]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "uipath_claude_sdk.runtime.runtime.ClaudeSDKClient", InterruptClient
    )
    InterruptClient.last_options = None
    InterruptClient.last_query = None
    InterruptClient.tool_results = []
    InterruptClient.session_id = "session-1"
    return InterruptClient


@pytest.fixture
def hitl_agent() -> ClaudeAgent:
    return ClaudeAgent(options=hitl_options())


def _make_runtime(
    agent: ClaudeAgent, tmp_path: Path, session_store: ClaudeSessionStore
) -> UiPathClaudeSDKRuntime:
    return UiPathClaudeSDKRuntime(
        agent=agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )


async def _run(runtime, input=None, resume=False) -> UiPathRuntimeResult:
    result: UiPathRuntimeResult | None = None
    async for event in runtime.stream(input, UiPathStreamOptions(resume=resume)):
        if isinstance(event, UiPathRuntimeResult):
            result = event
    assert result is not None
    return result


async def test_an_agent_without_uipath_tools_gets_the_options_it_wrote(
    plain_agent, interrupt_client, tmp_path, session_store
):
    """An agent is what its code says, so a plain one must be untouched.

    No tool, no hook and no suspend-safety environment: this runtime hands the
    SDK client exactly the options the developer wrote.

    The tracing instrumentor merges hooks of its own into
    ``ClaudeSDKClient.options`` on the way into ``query()``, after this runtime
    is done. Those are the instrumentor's and appear wherever it is installed,
    which ``test_an_agent_without_uipath_tools_contributes_no_hook_of_its_own``
    pins separately.
    """
    base = ClaudeAgentOptions(
        mcp_servers={"other": {"type": "stdio", "command": "noop"}},
        allowed_tools=["Bash"],
    )
    plain_agent.options = base
    runtime = _make_runtime(plain_agent, tmp_path, session_store)
    await _run(runtime, {"input": "hi"})

    options = interrupt_client.last_options
    assert options.mcp_servers == base.mcp_servers
    assert options.hooks == base.hooks
    assert options.allowed_tools == base.allowed_tools
    assert "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK" not in options.env


async def test_a_uipath_tool_adds_the_hook_and_nothing_else(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    async def developer_hook(hook_input, tool_use_id, context):
        return {}

    base = hitl_options(
        extra_servers={"other": {"type": "stdio", "command": "noop"}},
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[developer_hook])]},
        allowed_tools=["Bash"],
    )
    hitl_agent.options = base
    runtime = _make_runtime(hitl_agent, tmp_path, session_store)
    await _run(runtime, {"input": "ask me"})

    options = interrupt_client.last_options
    assert options.mcp_servers == base.mcp_servers
    assert options.allowed_tools == ["Bash"]
    assert len(options.hooks["PreToolUse"]) == 2
    assert options.hooks["PreToolUse"][1].hooks == [developer_hook]
    assert options.env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] == "1"

    paths = make_session_paths(tmp_path)
    assert options.cwd == paths.workspace
    assert options.env["CLAUDE_CONFIG_DIR"] == str(paths.config_dir)
    assert paths.workspace.is_dir()
    assert paths.config_dir.is_dir()


async def test_suspend_persists_the_parked_call(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    runtime = _make_runtime(hitl_agent, tmp_path, session_store)
    result = await _run(runtime, {"input": "ask me"})

    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert _output(result) == {TOOL_USE_ID: QUESTION}
    assert interrupt_client.tool_results == []

    pending = await session_store.get_pending_suspend()
    assert pending is not None
    assert pending.interrupt_id == TOOL_USE_ID
    assert pending.tool_name == INTERRUPT_TOOL
    assert pending.tool_use_id == TOOL_USE_ID
    assert await session_store.get_session_id() == "session-1"

    record = await session_store.get_entrypoint()
    assert record is not None
    assert record.entrypoint == "agent"


async def test_resume_delivers_the_payload_and_clears_the_record(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    await _run(_make_runtime(hitl_agent, tmp_path, session_store), {"input": "ask me"})

    # A fresh runtime stands in for the resumed process: the hook and the tool
    # server are rebuilt, only the store and the paths carry over.
    resumed = _make_runtime(hitl_agent, tmp_path, session_store)
    result = await _run(resumed, {TOOL_USE_ID: {"answer": "yes"}}, resume=True)

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert result.output == {"result": "done"}
    assert interrupt_client.tool_results == ["{'answer': 'yes'}"]
    assert interrupt_client.last_query == "Continue."
    assert interrupt_client.last_options.resume == "session-1"
    assert await session_store.get_pending_suspend() is None


async def test_resume_without_a_payload_leaves_the_call_parked(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    await _run(_make_runtime(hitl_agent, tmp_path, session_store), {"input": "ask me"})
    pending = await session_store.get_pending_suspend()

    resumed = _make_runtime(hitl_agent, tmp_path, session_store)
    result = await _run(resumed, None, resume=True)

    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert interrupt_client.tool_results == []
    still_pending = await session_store.get_pending_suspend()
    assert still_pending is not None
    assert still_pending.interrupt_id == pending.interrupt_id


async def test_resume_without_a_session_id_fails(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    runtime = _make_runtime(hitl_agent, tmp_path, session_store)
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="no Claude session id"):
        await _run(runtime, {"input": "hi"}, resume=True)


async def test_a_lost_transcript_fails_the_resume(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    await _run(_make_runtime(hitl_agent, tmp_path, session_store), {"input": "ask me"})
    interrupt_client.session_id = "session-restarted"

    resumed = _make_runtime(hitl_agent, tmp_path, session_store)
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="but the run answered from"):
        await _run(resumed, {TOOL_USE_ID: {"answer": "yes"}}, resume=True)


@pytest.mark.parametrize(
    ("value", "trigger_type"),
    [
        (
            CreateTask(title="Approve the refund", app_name="generic_escalation_app"),
            UiPathResumeTriggerType.TASK,
        ),
        (
            [WaitJob(job=Job(key="job-1", id=0)), WaitJob(job=Job(key="job-2", id=0))],
            UiPathResumeTriggerType.JOB,
        ),
    ],
)
async def test_a_reloaded_suspend_value_still_picks_the_same_trigger(
    session_store, value, trigger_type
):
    """Persisting a suspend value must not cost it its type.

    ``UiPathResumeTriggerCreator`` dispatches on the value's type and defaults
    to an API trigger for anything it does not recognise. A value that comes
    back from the store as a plain dict therefore re-parks on an API trigger
    the platform never fires, so the run waits on a resume nobody sends.
    """
    await session_store.set_pending_suspend(
        PendingSuspend(
            interrupt_id=TOOL_USE_ID,
            tool_name=INTERRUPT_TOOL,
            value=value,
            tool_use_id=TOOL_USE_ID,
        )
    )

    reloaded = await session_store.get_pending_suspend()

    assert reloaded is not None
    assert reloaded.value == value
    creator = UiPathResumeTriggerCreator()
    targets = reloaded.value if isinstance(reloaded.value, list) else [reloaded.value]
    assert targets
    for target in targets:
        assert creator._determine_trigger_type(target) is trigger_type


async def test_a_reloaded_question_stays_the_plain_string_it_was(session_store):
    """A bare string suspends on an API trigger, and text survives the store."""
    await session_store.set_pending_suspend(
        PendingSuspend(
            interrupt_id=TOOL_USE_ID,
            tool_name=INTERRUPT_TOOL,
            value=QUESTION,
            tool_use_id=TOOL_USE_ID,
        )
    )

    reloaded = await session_store.get_pending_suspend()

    assert reloaded is not None
    assert reloaded.value == QUESTION


async def test_resume_on_another_entrypoint_is_refused(
    hitl_agent, interrupt_client, tmp_path, session_store
):
    await _run(_make_runtime(hitl_agent, tmp_path, session_store), {"input": "ask me"})

    resumed = UiPathClaudeSDKRuntime(
        agent=hitl_agent,
        session_store=session_store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="other-agent",
    )
    with pytest.raises(UiPathClaudeSDKRuntimeError, match="cannot be\\s+resumed on"):
        await _run(resumed, {TOOL_USE_ID: {"answer": "yes"}}, resume=True)
