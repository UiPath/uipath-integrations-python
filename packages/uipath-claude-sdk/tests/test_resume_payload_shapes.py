"""The parked call must resolve from either resume payload shape.

``UiPathResumableRuntime`` hands the delegate one of two things. When it read a
fired trigger itself it passes the ``{interrupt_id: payload}`` map it built.
When the caller supplied input explicitly, as
``uipath run agent '{"answer": "Acme"}' --resume`` does, it deletes the trigger
and passes that input through unwrapped.

Accepting only the first shape looks correct in a test that builds the map by
hand, and fails in the field: the answer is dropped, the gate claims a fresh
interrupt id, and the run suspends again under a new trigger while the old one
is already gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import make_session_paths
from uipath_claude_sdk import ClaudeAgent
from uipath_claude_sdk.interrupts import PendingSuspend, SuspendChannel
from uipath_claude_sdk.runtime.runtime import UiPathClaudeSDKRuntime
from uipath_claude_sdk.runtime.session_store import ClaudeSessionStore

TOOL_USE_ID = "toolu_01"
INTERRUPT_ID = TOOL_USE_ID
ANSWER = {"answer": "Acme"}


def _pending() -> PendingSuspend:
    return PendingSuspend(
        interrupt_id=INTERRUPT_ID,
        tool_name="mcp__uipath__wait_for_input",
        value="Which vendor?",
        tool_use_id=TOOL_USE_ID,
    )


def _runtime(tmp_path: Path, store: ClaudeSessionStore) -> UiPathClaudeSDKRuntime:
    return UiPathClaudeSDKRuntime(
        agent=ClaudeAgent(),
        session_store=store,
        session_paths=make_session_paths(tmp_path),
        runtime_id="rt-1",
        entrypoint="agent",
    )


async def _restore(
    tmp_path: Path,
    store: ClaudeSessionStore,
    input: dict[str, Any] | None,
    resuming: bool = True,
) -> tuple[SuspendChannel, PendingSuspend | None]:
    await store.set_pending_suspend(_pending())
    runtime = _runtime(tmp_path, store)
    channel = SuspendChannel()
    restored = await runtime._restore_pending(channel, input, resuming)
    return channel, restored


async def test_keyed_resume_map_resolves_the_parked_call(
    tmp_path: Path, session_store: ClaudeSessionStore
):
    channel, restored = await _restore(tmp_path, session_store, {INTERRUPT_ID: ANSWER})

    assert restored is not None
    assert restored.interrupt_id == INTERRUPT_ID
    assert channel.resolved_for(INTERRUPT_ID) == ANSWER


async def test_bare_payload_resolves_the_parked_call(
    tmp_path: Path, session_store: ClaudeSessionStore
):
    """The shape a human types at the CLI, with no interrupt id in sight."""
    channel, restored = await _restore(tmp_path, session_store, ANSWER)

    assert restored is not None
    assert restored.interrupt_id == INTERRUPT_ID
    assert channel.resolved_for(INTERRUPT_ID) == ANSWER


async def test_empty_resume_reparks_under_the_same_interrupt_id(
    tmp_path: Path, session_store: ClaudeSessionStore
):
    """A resume with nothing to deliver must not mint a second trigger.

    The record is restored but left unresolved, so the gate re-parks the same
    call under its original interrupt id rather than claiming a fresh one and
    leaving the first trigger orphaned.
    """
    channel, restored = await _restore(tmp_path, session_store, None)

    assert restored is None
    assert channel.pending is not None
    assert channel.pending.interrupt_id == INTERRUPT_ID
    assert channel.resolved_for(INTERRUPT_ID) is None


async def test_a_fresh_run_never_consumes_a_pending_record(
    tmp_path: Path, session_store: ClaudeSessionStore
):
    """Without the resume flag the stored record must stay untouched."""
    channel, restored = await _restore(tmp_path, session_store, ANSWER, resuming=False)

    assert restored is None
    assert channel.pending is None


@pytest.mark.parametrize("payload", [ANSWER, {INTERRUPT_ID: ANSWER}])
async def test_resume_never_validates_against_the_input_schema(
    tmp_path: Path, session_store: ClaudeSessionStore, payload: dict[str, Any]
):
    """A resume must not be checked against the entrypoint input schema.

    The resume map is not entrypoint input, so validating it reports a missing
    required field when the real state is that an interrupt is being answered.
    """
    from uipath_claude_sdk.runtime.runtime import RESUME_PROMPT, RunContext

    await session_store.set_pending_suspend(_pending())
    runtime = _runtime(tmp_path, session_store)
    channel = SuspendChannel()
    restored = await runtime._restore_pending(channel, payload, True)
    context = RunContext(
        channel=channel,
        sdk_options=runtime.agent.options,
        session_id="session-1",
        pending=restored,
        resuming=True,
    )

    assert runtime._user_message_for_run(payload, context) == RESUME_PROMPT
