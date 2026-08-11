"""End-to-end proof that a run suspends on the Claude Code CLI defer protocol.

This is the regression test for the whole durable-suspend design. It drives the
real :class:`UiPathClaudeSDKRuntime` and the real Claude Code CLI binary against
a local HTTP server speaking canned Anthropic SSE, so it needs no tenant, no API
key and no network. A CLI upgrade that changes how a ``PreToolUse`` hook
returning ``permissionDecision: "defer"`` behaves fails here.

The agent deliberately declares no ``uipath_llm``: this test is about the
suspend protocol, not about the gateway, so the CLI talks to the stub directly.

The Claude Agent SDK instrumentor is installed throughout, as it is on every
deployed run, so the CLI is handed the merged hook list rather than the one this
package built on its own. An instrumentor release that stopped merging into
``PreToolUse`` would drop the interrupt hook and fail here.

It spawns a subprocess twice and is therefore slower than the rest of the
suite. Deselect it with ``-m "not defer_contract"``.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, tool
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from pydantic import BaseModel
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
)

from uipath_claude_sdk import ClaudeAgent, interrupt, uipath_tool_server
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime
from uipath_claude_sdk.runtime.session_paths import (
    ClaudeSessionPaths,
    encode_project_dir,
)
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore
from uipath_claude_sdk.runtime.storage import SqliteResumableStorage

pytestmark = pytest.mark.defer_contract

MODEL = "claude-sonnet-4-5"
RUNTIME_ID = "defer-contract"
SERVER_NAME = "approvals"
TOOL_NAME = "ask_approver"
INTERRUPT_TOOL = f"mcp__{SERVER_NAME}__{TOOL_NAME}"
TOOL_USE_ID = "toolu_defer_contract"

PROMPT = "Ship release 1.4.0."
QUESTION = "May I ship release 1.4.0?"
HUMAN_ANSWER = "yes, ship it"
ANSWER_PREFIX = "The human said: "

DEFER_MARKER = "hook_deferred_tool"

# Ambient Claude credentials would send the CLI somewhere other than the stub.
_UNSET_IN_SUBPROCESS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CONFIG_DIR",
)

_QUIET_CLI_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
}


# --- The stub upstream ------------------------------------------------------


def _sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


_MESSAGE_START = (
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_defer_contract",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
)


def _ask_for_approval() -> bytes:
    return _sse(
        _MESSAGE_START,
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": TOOL_USE_ID,
                    "name": INTERRUPT_TOOL,
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"question": QUESTION}),
                },
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def _say(text: str) -> bytes:
    return _sse(
        _MESSAGE_START,
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def _tool_result_blocks(body: str) -> list[dict[str, Any]]:
    """Every ``tool_result`` block carried by one Anthropic request body."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    blocks: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        blocks.extend(
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        )
    return blocks


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


class FakeAnthropic:
    """A local Anthropic stand-in replying with canned SSE on an ephemeral port.

    It asks for the interrupt tool until the conversation carries a
    ``tool_result``, then echoes that result back as its final text. The run's
    output therefore cannot carry the human answer unless the resumed payload
    really travelled to the model as a tool result.
    """

    def __init__(self) -> None:
        self.requests: list[str] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """The distinct tool results the conversation carried, oldest first.

        Every request repeats the whole conversation, so a block is counted
        once per tool_use id rather than once per request.
        """
        seen: dict[str, dict[str, Any]] = {}
        for body in self.requests:
            for block in _tool_result_blocks(body):
                seen.setdefault(str(block.get("tool_use_id")), block)
        return list(seen.values())

    def start(self) -> None:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                upstream.requests.append(body)
                blocks = _tool_result_blocks(body)
                if blocks:
                    answer = _say(f"{ANSWER_PREFIX}{_tool_result_text(blocks[-1])}")
                else:
                    answer = _ask_for_approval()
                self._reply(answer, "text/event-stream")

            def do_GET(self) -> None:
                self._reply(b'{"data":[]}', "application/json")

            def do_HEAD(self) -> None:
                self._reply(b"", "text/plain")

            def _reply(self, payload: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def instrumented() -> None:
    """Trace the run the way a deployed one is traced.

    The runtime factory installs this on every run, and with it the CLI receives
    the instrumentor's tracing hooks merged alongside the interrupt hook.
    ``conftest`` uninstruments once the test is over.
    """
    ClaudeAgentSDKInstrumentor().instrument()
    assert ClaudeAgentSDKInstrumentor().is_instrumented_by_opentelemetry


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAnthropic]:
    for name in _UNSET_IN_SUBPROCESS:
        monkeypatch.delenv(name, raising=False)
    server = FakeAnthropic()
    server.start()
    try:
        yield server
    finally:
        server.stop()


class ToolBodyLog:
    """What the developer's own tool body did, across both processes.

    Nothing is injected into the agent, so the only way to observe the protocol
    from the inside is the body the test itself wrote.
    """

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.returned: list[str] = []


BODY_LOG = ToolBodyLog()


@pytest.fixture
def tool_body() -> Iterator[ToolBodyLog]:
    BODY_LOG.started.clear()
    BODY_LOG.returned.clear()
    yield BODY_LOG


@tool(TOOL_NAME, "Ask a human a question and wait for the answer.", {"question": str})
async def ask_approver(args: dict[str, Any]) -> dict[str, Any]:
    BODY_LOG.started.append(dict(args))
    answer = str(await interrupt(args["question"]))
    BODY_LOG.returned.append(answer)
    return {"content": [{"type": "text", "text": answer}]}


class Approval(BaseModel):
    approved: bool


def _agent(base_url: str, output_schema: type[BaseModel] | None = None) -> ClaudeAgent:
    return ClaudeAgent(
        options=ClaudeAgentOptions(
            model=MODEL,
            system_prompt="Ask a human for approval before doing anything.",
            permission_mode="bypassPermissions",
            max_turns=4,
            mcp_servers={
                SERVER_NAME: uipath_tool_server(SERVER_NAME, tools=[ask_approver])
            },
            env={
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_API_KEY": "not-a-real-key",
                **_QUIET_CLI_ENV,
            },
        ),
        output_schema=output_schema,
        name="agent",
    )


@asynccontextmanager
async def _process(
    runtime_dir: Path,
    base_url: str,
    output_schema: type[BaseModel] | None = None,
    state_db: Path | None = None,
    workspace: Path | None = None,
) -> AsyncIterator[tuple[UiPathClaudeSDKRuntime, ClaudeSessionStore]]:
    """One simulated OS process over a runtime directory.

    Everything a process cannot inherit is rebuilt: the agent, the storage
    handle, the runtime, and with it the hooks and the in-process MCP server.

    ``state_db`` splits the two things the platform treats differently. The
    state database is carried across a suspension; the directory holding it is
    not, and a resumed job is handed a fresh one. Pass a path outside
    ``runtime_dir`` to model that.

    ``workspace`` overrides the working directory, which a managed workspace
    creates fresh per execution.
    """
    storage = SqliteResumableStorage(str(state_db or runtime_dir / "state.db"))
    store = ClaudeSessionStore(storage, RUNTIME_ID)
    paths = ClaudeSessionPaths.for_runtime(runtime_dir, RUNTIME_ID)
    if workspace is not None:
        paths = replace(paths, workspace=workspace)
    try:
        yield (
            UiPathClaudeSDKRuntime(
                agent=_agent(base_url, output_schema),
                session_store=store,
                session_paths=paths,
                runtime_id=RUNTIME_ID,
                entrypoint="agent",
            ),
            store,
        )
    finally:
        await storage.dispose()


def _output(result: UiPathRuntimeResult) -> dict[str, Any]:
    assert isinstance(result.output, dict)
    return result.output


def _transcript(paths: ClaudeSessionPaths, session_id: str) -> Path:
    matches = list((paths.config_dir / "projects").rglob(f"{session_id}.jsonl"))
    assert len(matches) == 1, f"Expected exactly one transcript, found {matches}."
    return matches[0]


# --- The contract -----------------------------------------------------------


async def test_a_deferred_call_suspends_and_a_fresh_process_finishes_it(
    tmp_path: Path, upstream: FakeAnthropic, tool_body: ToolBodyLog
) -> None:
    runtime_dir = tmp_path / "__uipath"
    paths = ClaudeSessionPaths.for_runtime(runtime_dir, RUNTIME_ID)

    async with _process(runtime_dir, upstream.base_url) as (runtime, store):
        suspended = await runtime.execute({"input": PROMPT})
        session_id = await store.get_session_id()
        pending = await store.get_pending_suspend()

    assert suspended.status == UiPathRuntimeStatus.SUSPENDED
    assert list(_output(suspended).values()) == [QUESTION]

    # The body runs up to interrupt() and no further, and the model is told
    # nothing: the call is parked, not answered.
    assert tool_body.started == [{"question": QUESTION}]
    assert tool_body.returned == []
    assert upstream.tool_results == []

    assert pending is not None
    assert pending.tool_name == INTERRUPT_TOOL
    assert pending.interrupt_id in _output(suspended)
    assert session_id is not None

    transcript = _transcript(paths, session_id)
    assert transcript.parent.name.endswith(paths.workspace.name)
    assert DEFER_MARKER in transcript.read_text()

    async with _process(runtime_dir, upstream.base_url) as (resumed, store):
        result = await resumed.execute(
            {pending.interrupt_id: HUMAN_ANSWER},
            UiPathExecuteOptions(resume=True),
        )
        assert await store.get_pending_suspend() is None
        assert await store.get_session_id() == session_id

    # The fresh process replays the body from the top with the original
    # arguments, and this time interrupt() returns instead of raising.
    assert tool_body.started == [{"question": QUESTION}, {"question": QUESTION}]
    assert tool_body.returned == [HUMAN_ANSWER]

    delivered = upstream.tool_results
    assert len(delivered) == 1
    assert delivered[0]["tool_use_id"] == pending.tool_use_id
    assert _tool_result_text(delivered[0]) == HUMAN_ANSWER

    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert _output(result) == {"result": f"{ANSWER_PREFIX}{HUMAN_ANSWER}"}


async def test_a_structured_output_agent_still_suspends(
    tmp_path: Path, upstream: FakeAnthropic, tool_body: ToolBodyLog
) -> None:
    """A deferred turn carries no structured output, and must not be validated.

    The suspend branch has to come before the output path: an agent declaring an
    output schema would otherwise report every suspension as a schema failure.
    """
    runtime_dir = tmp_path / "__uipath"

    async with _process(runtime_dir, upstream.base_url, Approval) as (runtime, store):
        suspended = await runtime.execute({"input": PROMPT})
        pending = await store.get_pending_suspend()

    assert suspended.status == UiPathRuntimeStatus.SUSPENDED
    assert list(_output(suspended).values()) == [QUESTION]
    assert pending is not None
    assert tool_body.returned == []


async def test_a_resumed_job_given_a_fresh_runtime_directory_still_continues(
    tmp_path: Path, upstream: FakeAnthropic, tool_body: ToolBodyLog
) -> None:
    """The deployed shape: state carried across, directory not.

    A real job suspended under ``.job-data/42d7392e-.../__uipath`` and resumed
    under ``.job-data/bad7abe7-.../__uipath``. The CLI's transcript lives under
    a directory named for the working directory, so a workspace inside the
    runtime directory encoded to a different name on the second pass and the
    session was reported as not found.
    """
    state_db = tmp_path / "carried" / "state.db"
    state_db.parent.mkdir(parents=True)
    suspend_dir = tmp_path / "job-data" / "42d7392e" / "__uipath"
    resume_dir = tmp_path / "job-data" / "bad7abe7" / "__uipath"

    async with _process(suspend_dir, upstream.base_url, state_db=state_db) as (
        runtime,
        store,
    ):
        suspended = await runtime.execute({"input": PROMPT})
        pending = await store.get_pending_suspend()
        session_id = await store.get_session_id()
        transcript = await store.get_transcript()

    assert suspended.status == UiPathRuntimeStatus.SUSPENDED
    assert pending is not None and session_id is not None
    assert transcript is not None, "The transcript was not carried into storage."
    assert transcript.session_id == session_id
    assert DEFER_MARKER in transcript.content

    # Nothing of the first pass survives on disk, exactly as on the platform.
    shutil.rmtree(suspend_dir)

    async with _process(resume_dir, upstream.base_url, state_db=state_db) as (
        resumed,
        store,
    ):
        result = await resumed.execute(
            {pending.interrupt_id: HUMAN_ANSWER},
            UiPathExecuteOptions(resume=True),
        )
        assert await store.get_session_id() == session_id, (
            "The CLI could not find the session and started a new one."
        )

    assert tool_body.returned == [HUMAN_ANSWER]
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert _output(result) == {"result": f"{ANSWER_PREFIX}{HUMAN_ANSWER}"}


async def test_a_resume_in_a_different_working_directory_finds_its_session(
    tmp_path: Path, upstream: FakeAnthropic, tool_body: ToolBodyLog
) -> None:
    """The managed-workspace shape: a fresh working directory per execution.

    ``uipath`` creates a managed workspace with ``tempfile.mkdtemp``, so a
    resumed execution gets a directory it has never seen. The CLI names its
    transcript directory after that path, so the name is derived rather than
    remembered. This is what pins that derivation: the real CLI, not our idea
    of its encoding, decides whether the session is found.
    """
    state_db = tmp_path / "carried" / "state.db"
    state_db.parent.mkdir(parents=True)
    first = tmp_path / "uipath-workspace-aaaa"
    second = tmp_path / "uipath-workspace-zzzz"

    async with _process(
        tmp_path / "run-1", upstream.base_url, state_db=state_db, workspace=first
    ) as (runtime, store):
        suspended = await runtime.execute({"input": PROMPT})
        pending = await store.get_pending_suspend()
        session_id = await store.get_session_id()
        record = await store.get_transcript()

    assert suspended.status == UiPathRuntimeStatus.SUSPENDED
    assert pending is not None and session_id is not None
    assert record is not None
    assert record.project_dir == encode_project_dir(first)

    async with _process(
        tmp_path / "run-2", upstream.base_url, state_db=state_db, workspace=second
    ) as (resumed, store):
        result = await resumed.execute(
            {pending.interrupt_id: HUMAN_ANSWER},
            UiPathExecuteOptions(resume=True),
        )
        assert await store.get_session_id() == session_id, (
            "The CLI could not find the session under the new working directory."
        )

    assert tool_body.returned == [HUMAN_ANSWER]
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert _output(result) == {"result": f"{ANSWER_PREFIX}{HUMAN_ANSWER}"}
