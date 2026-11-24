"""Resume input handling, re-suspension and session guards.

These paths never start codex: a resume that still has pending calls returns
SUSPENDED straight away. They run through the real ``UiPathResumableRuntime``
and ``UiPathResumeTriggerHandler`` with fake Orchestrator jobs.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from uipath.core.triggers import UiPathResumeTriggerName
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathResumableRuntime,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
)

import uipath_codex_sdk.runtime.runtime as runtime_mod
from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool
from uipath_codex_sdk.runtime.errors import UiPathCodexRuntimeError
from uipath_codex_sdk.runtime.runtime import UiPathCodexRuntime, normalize_resume_input
from uipath_codex_sdk.runtime.session_store import (
    STATE_STARTED,
    CodexSessionStore,
    DanglingCall,
    EntrypointRecord,
    PendingCall,
    ThreadRecord,
    encode_rollout,
)
from uipath_codex_sdk.runtime.storage import SqliteResumableStorage

from .durable.fakes import FakeOrchestrator

RT = "job-key-1"
THREAD = "01a11c0c-5728-7780-816f-157c0652c4b0"


def test_normalize_resume_input_shapes():
    one, two = ["c1"], ["c1", "c2"]
    assert normalize_resume_input(None, one) == {}
    assert normalize_resume_input({}, one) == {}
    assert normalize_resume_input({"c1": "x", "zz": 1}, two) == {"c1": "x"}
    assert normalize_resume_input({"c1": 1, "c2": 2}, two) == {"c1": 1, "c2": 2}
    # a raw payload answers the single pending call...
    assert normalize_resume_input({"sum": 3}, one) == {"c1": {"sum": 3}}
    assert normalize_resume_input("text", one) == {"c1": "text"}
    # ...and is ambiguous (ignored) with several
    assert normalize_resume_input({"sum": 3}, two) == {}
    assert normalize_resume_input({"sum": 3}, []) == {}


@pytest.fixture
def world(monkeypatch) -> FakeOrchestrator:
    w = FakeOrchestrator()
    import uipath.platform.resume_triggers._protocol as protocol

    import uipath_codex_sdk.tools as tools

    monkeypatch.setattr(tools, "_uipath_client", w.client)
    monkeypatch.setattr(protocol, "UiPath", w.client)
    monkeypatch.setattr(runtime_mod, "RESUSPEND_BACKOFF_SEC", 0.0)
    return w


@pytest.fixture
async def storage(tmp_path):
    s = SqliteResumableStorage(str(tmp_path / "state.db"))
    yield s
    await s.dispose()


def _agent() -> UiPathCodexAgent:
    return UiPathCodexAgent(
        model="m", prompt="p", tools=[uipath_process_tool("a_plus_b")]
    )


def _job(world: FakeOrchestrator, n: int) -> Job:
    key = f"job-{n}"
    job = Job.model_validate(
        {"Id": n, "Key": key, "State": "Running", "FolderKey": "fk"}
    )
    world.jobs[key] = job
    return job


async def _suspended_state(
    storage, world, call_ids: list[str], entrypoint: str = "agent"
) -> UiPathResumableRuntime:
    """State as left by a suspension: session records + JOB_RAW triggers."""
    store = CodexSessionStore(storage, RT)
    await store.set_entrypoint(EntrypointRecord(entrypoint))
    await store.set_thread(ThreadRecord(THREAD, "sessions/r.jsonl"))
    await store.set_rollout(encode_rollout("sessions/r.jsonl", b"{}\n"))
    output = {}
    for i, cid in enumerate(call_ids, start=1):
        value = WaitJobRaw(job=_job(world, i), process_folder_key="fk")
        await store.put_pending(
            PendingCall(
                cid, "a_plus_b", {"a": i}, STATE_STARTED, "WaitJobRaw", value, i
            )
        )
        output[cid] = value
    await store.set_dangling(
        [DanglingCall(c, "function_call", durable=True) for c in call_ids]
    )
    wrapper = _wrapper(storage)
    await wrapper._handle_suspension(
        UiPathRuntimeResult(output=output, status=UiPathRuntimeStatus.SUSPENDED)
    )
    return wrapper


def _wrapper(storage, agent: UiPathCodexAgent | None = None, entrypoint: str = "agent"):
    delegate = UiPathCodexRuntime(
        agent=agent or _agent(),
        runtime_id=RT,
        entrypoint=entrypoint,
        file_path="/tmp/main.py",
        storage=storage,
    )
    return UiPathResumableRuntime(
        delegate=delegate,
        storage=storage,
        trigger_manager=UiPathResumeTriggerHandler(),
        runtime_id=RT,
    )


async def test_resume_while_job_running_resuspends_without_duplicates(storage, world):
    await _suspended_state(storage, world, ["c1"])
    result = await _wrapper(storage).execute(None, UiPathExecuteOptions(resume=True))
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    triggers = await storage.get_triggers(RT)
    assert [t.interrupt_id for t in triggers] == ["c1"]


async def test_explicit_empty_resume_with_single_trigger(storage, world):
    """``--resume '{}'``: the wrapper deletes the only trigger and passes {};
    the re-suspension must mint a fresh *JOB_RAW* trigger (typed value)."""
    await _suspended_state(storage, world, ["c1"])
    result = await _wrapper(storage).execute({}, UiPathExecuteOptions(resume=True))
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    [trigger] = await storage.get_triggers(RT)
    assert trigger.interrupt_id == "c1"
    assert trigger.trigger_name == UiPathResumeTriggerName.JOB_RAW
    assert trigger.item_key == "job-1"
    assert isinstance(result.output["c1"], WaitJobRaw)


async def test_partial_resume_keeps_waiting_on_the_rest(storage, world):
    await _suspended_state(storage, world, ["c1", "c2"])
    world.finish("job-1", "Successful", {"sum": 1})
    result = await _wrapper(storage).execute(None, UiPathExecuteOptions(resume=True))
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert list(result.output) == ["c2"]
    assert [t.interrupt_id for t in await storage.get_triggers(RT)] == ["c2"]
    store = CodexSessionStore(storage, RT)
    assert await store.get_completed() == {"c1": json.dumps({"sum": 1})}
    assert list(await store.get_pending()) == ["c2"]


@pytest.mark.parametrize("state", ["Stopping", "Terminating"])
async def test_stopping_job_ends_the_resume_loop(storage, world, monkeypatch, state):
    """read_trigger reports Stopping/Terminating as fired. Through the real
    wrapper loop the call must be answered (not re-suspended), or the wrapper
    re-reads the fresh trigger and polls Orchestrator forever."""
    await _suspended_state(storage, world, ["c1"])
    world.finish("job-1", state)
    plans: list = []

    async def fake_run(self, prompt, tools, plan, **kwargs):
        plans.append(plan)
        yield UiPathRuntimeResult(
            output={"output": "done"}, status=UiPathRuntimeStatus.SUCCESSFUL
        )

    monkeypatch.setattr(UiPathCodexRuntime, "_run", fake_run)
    result = await asyncio.wait_for(
        _wrapper(storage).execute(None, UiPathExecuteOptions(resume=True)), 5
    )
    assert result.status == UiPathRuntimeStatus.SUCCESSFUL
    assert world.retrieve_calls <= 2
    [plan] = plans
    [item] = plan.items
    assert item["call_id"] == "c1"
    assert json.loads(item["output"]) == {"status": state.lower(), "job_key": "job-1"}


async def test_custom_pending_finish_resuspends(storage, world, monkeypatch):
    """A custom tool whose finish says "still pending" keeps the call waiting."""
    from uipath_codex_sdk import UiPathDurableToolPending

    await _suspended_state(storage, world, ["c1"])
    world.finish("job-1", "Successful", {"sum": 1})
    agent = _agent()
    tool = agent.tools[0]  # type: ignore[index]

    async def pending_finish(args, value):
        raise UiPathDurableToolPending("not yet")

    monkeypatch.setattr(tool, "finish", pending_finish)
    delegate = _wrapper(storage, agent).delegate
    result = await delegate.execute(
        {"c1": world.jobs["job-1"]}, UiPathExecuteOptions(resume=True)
    )
    assert result.status == UiPathRuntimeStatus.SUSPENDED
    assert list(result.output) == ["c1"]
    assert list(await CodexSessionStore(storage, RT).get_pending()) == ["c1"]


async def test_resume_without_session(storage, world):
    with pytest.raises(UiPathCodexRuntimeError) as exc:
        await _wrapper(storage).execute(None, UiPathExecuteOptions(resume=True))
    assert "SESSION_NOT_FOUND" in str(exc.value.error_info.code)


async def test_resume_on_other_entrypoint(storage, world):
    await _suspended_state(storage, world, ["c1"], entrypoint="other")
    with pytest.raises(UiPathCodexRuntimeError) as exc:
        await _wrapper(storage).execute(None, UiPathExecuteOptions(resume=True))
    assert "ENTRYPOINT_MISMATCH" in str(exc.value.error_info.code)


async def test_resume_with_unknown_tool(storage, world):
    await _suspended_state(storage, world, ["c1"])
    world.finish("job-1", "Successful", {"sum": 1})
    agent = UiPathCodexAgent(
        model="m", prompt="p", tools=[uipath_process_tool("other")]
    )
    with pytest.raises(UiPathCodexRuntimeError) as exc:
        await _wrapper(storage, agent).execute(None, UiPathExecuteOptions(resume=True))
    assert "TOOL_NOT_FOUND" in str(exc.value.error_info.code)


async def test_fresh_run_drops_stale_triggers(storage, world, monkeypatch):
    """A fresh run over a kept state file must not inherit old triggers."""
    await _suspended_state(storage, world, ["old_call"])
    assert await storage.get_triggers(RT)
    delegate = _wrapper(storage).delegate

    async def no_codex(self, prompt, tools, plan, **kwargs):
        yield None  # pragma: no cover - only the bookkeeping before it matters

    monkeypatch.setattr(UiPathCodexRuntime, "_run", no_codex)
    async for _ in delegate.stream({}, None):
        break
    assert await storage.get_triggers(RT) is None
    store = CodexSessionStore(storage, RT)
    assert await store.get_pending() == {}
    assert await store.get_thread() is None
    assert await store.get_entrypoint() == EntrypointRecord("agent")
