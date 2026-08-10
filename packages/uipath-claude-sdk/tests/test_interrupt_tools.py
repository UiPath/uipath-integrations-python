"""The developer owns the tool, and UiPath only lends it a suspension.

``uipath_tool_server`` must leave a tool's name, description and schema exactly
as written, and must not appear in an agent that never uses it. What it adds is
``interrupt``: the body runs inside the ``PreToolUse`` hook, suspends by raising
on the first pass and returns the resolved payload on the second, and the
handler delivers whatever the hook already computed instead of running twice.

The value handed to ``interrupt`` is never touched, so these tests push real
platform models through ``UiPathResumeTriggerCreator`` to prove the trigger kind
still comes from the model type and from nothing this package does.
"""

from __future__ import annotations

from typing import Any

import pytest
from claude_agent_sdk import (
    HookContext,
    HookJSONOutput,
    PreToolUseHookInput,
    tool,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
)
from uipath.core.triggers import UiPathResumeTriggerType
from uipath.platform.common import CreateTask, WaitJob
from uipath.platform.orchestrator.job import Job
from uipath.platform.resume_triggers import UiPathResumeTriggerCreator

from uipath_claude_sdk.interrupts import (
    InterruptOutsideRunError,
    SuspendChannel,
    active_channel,
    interrupt,
    uipath_tool_index,
    uipath_tool_server,
)
from uipath_claude_sdk.runtime.suspend import (
    TOOL_BODY_TIMEOUT_SECONDS,
    build_suspend_hooks,
)

SERVER_KEY = "tickets"
TOOL_NAME = "review"
QUALIFIED = f"mcp__{SERVER_KEY}__{TOOL_NAME}"
TOOL_USE_ID = "toolu_1"
ESCALATION = CreateTask(
    title="Action Required: Review classification",
    app_name="escalation_agent_app",
    app_folder_path="Shared",
    data={"AgentOutput": "Classified TCK-1 as billing"},
)


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def _review_tool(value: Any = ESCALATION, calls: list[str] | None = None):
    """The canonical developer tool: one interrupt, then the developer's own output."""

    @tool(TOOL_NAME, "Get a human to review the classification.", {"ticket_id": str})
    async def review(args: dict[str, Any]) -> dict[str, Any]:
        if calls is not None:
            calls.append(args["ticket_id"])
        action_data = await interrupt(value)
        return _text(f"Approved: {action_data['Answer']}")

    return review


async def _served(config: Any) -> list[Any]:
    server = config["instance"]
    for request_type, handler in server.request_handlers.items():
        if request_type.__name__ != "ListToolsRequest":
            continue
        result = await handler(ListToolsRequest(method="tools/list"))
        return list(result.root.tools)
    raise AssertionError("The server exposes no ListToolsRequest handler.")


async def _call(config: Any, args: dict[str, Any]) -> CallToolResult:
    server = config["instance"]
    for request_type, handler in server.request_handlers.items():
        if request_type.__name__ != "CallToolRequest":
            continue
        result = await handler(
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=TOOL_NAME, arguments=args),
            )
        )
        return result.root
    raise AssertionError("The server exposes no CallToolRequest handler.")


async def _gate(
    channel: SuspendChannel,
    config: Any,
    args: dict[str, Any],
    tool_name: str = QUALIFIED,
    tool_use_id: str | None = TOOL_USE_ID,
) -> HookJSONOutput:
    hooks = build_suspend_hooks(
        channel, uipath_tool_index({SERVER_KEY: config, "other": {"type": "stdio"}})
    )
    hook_input: PreToolUseHookInput = {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": args,
        "tool_use_id": tool_use_id or "",
        "session_id": "session-1",
        "transcript_path": "",
        "cwd": "",
    }
    return await hooks["PreToolUse"][0].hooks[0](
        hook_input, tool_use_id, HookContext(signal=None)
    )


def _decision(output: HookJSONOutput) -> tuple[str, str]:
    payload: dict[str, Any] = dict(output)
    if not payload:
        return "", ""
    specific = payload["hookSpecificOutput"]
    decision: str = specific["permissionDecision"]
    return decision, specific.get("permissionDecisionReason", "")


def _result_text(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if block.type == "text")


async def test_the_tool_reaches_the_model_exactly_as_it_was_written():
    """Wrapping is an implementation detail, so none of it may be visible."""
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    assert config["type"] == "sdk"
    assert config["name"] == SERVER_KEY
    served = await _served(config)
    assert [entry.name for entry in served] == [TOOL_NAME]
    assert served[0].description == "Get a human to review the classification."
    assert served[0].inputSchema["properties"]["ticket_id"] == {"type": "string"}


async def test_only_a_uipath_server_lands_in_the_index():
    """An agent that never called uipath_tool_server gets no UiPath wiring."""
    uipath_config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])
    plain = {"type": "stdio", "command": "noop"}

    assert set(uipath_tool_index({SERVER_KEY: uipath_config, "other": plain})) == {
        QUALIFIED
    }
    assert uipath_tool_index({"other": plain}) == {}
    assert uipath_tool_index(None) == {}


async def test_the_index_follows_the_key_the_developer_registered():
    """The CLI qualifies a tool by the mcp_servers key, not by the server's name."""
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    assert set(uipath_tool_index({"renamed": config})) == {"mcp__renamed__review"}


async def test_a_tool_this_package_did_not_build_is_left_alone():
    """The hook must never interfere with a developer's own tools."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    output = await _gate(channel, config, {"command": "ls"}, tool_name="Bash")

    assert output == {}
    assert channel.pending is None


async def test_the_hook_gets_a_timeout_long_enough_to_run_a_tool_body():
    """The body runs inside the hook, so the SDK's 60s default would cap it."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])
    hooks = build_suspend_hooks(channel, uipath_tool_index({SERVER_KEY: config}))

    assert hooks["PreToolUse"][0].timeout == TOOL_BODY_TIMEOUT_SECONDS
    assert TOOL_BODY_TIMEOUT_SECONDS > 60


async def test_the_suspend_pass_defers_and_keeps_the_value_untouched():
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    decision, _ = await _decision_for(channel, config)

    assert decision == "defer"
    assert channel.pending is not None
    assert channel.pending.value is ESCALATION
    assert channel.pending.tool_name == QUALIFIED
    assert channel.deferrals_requested == 1


async def _decision_for(channel: SuspendChannel, config: Any) -> tuple[str, str]:
    return _decision(await _gate(channel, config, {"ticket_id": "TCK-1"}))


async def test_the_interrupt_id_is_the_tool_use_id_the_cli_will_re_issue():
    """A uuid4 of our own could not be matched to the re-issued call."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    await _decision_for(channel, config)

    assert channel.pending is not None
    assert channel.pending.interrupt_id == TOOL_USE_ID
    assert channel.pending.tool_use_id == TOOL_USE_ID


async def test_the_resume_pass_runs_the_body_through_and_the_handler_delivers_it():
    """The handler must return the hook's value rather than run the body again."""
    calls: list[str] = []
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool(calls=calls)])
    args = {"ticket_id": "TCK-1"}

    await _gate(channel, config, args)
    channel.resolve(TOOL_USE_ID, {"Answer": True})
    decision, _ = _decision(await _gate(channel, config, args))

    assert decision == "allow"
    with active_channel(channel):
        result = await _call(config, args)

    assert _result_text(result) == "Approved: True"
    assert calls == ["TCK-1", "TCK-1"]
    assert channel.pending is None


async def test_a_body_that_never_interrupts_runs_once_and_allows():
    calls: list[str] = []

    @tool(TOOL_NAME, "Look a ticket up.", {"ticket_id": str})
    async def lookup(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args["ticket_id"])
        return _text(f"Ticket {args['ticket_id']} is open")

    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[lookup])
    args = {"ticket_id": "TCK-1"}

    decision, _ = _decision(await _gate(channel, config, args))

    assert decision == "allow"
    assert channel.pending is None
    with active_channel(channel):
        result = await _call(config, args)

    assert _result_text(result) == "Ticket TCK-1 is open"
    assert calls == ["TCK-1"]


async def test_two_identical_calls_in_one_turn_each_run_the_body_once():
    """The model may make the same call twice, and each has its own outcome.

    A handler is handed its arguments and no ``tool_use_id``, so the stash is
    keyed by the tool and its arguments. Holding one outcome per key would let
    the second call overwrite the first, and the handler that found nothing
    would run the body a second time: three executions for two calls, with
    whatever the body writes happening three times.
    """
    runs: list[str] = []

    @tool(TOOL_NAME, "Charge a ticket.", {"ticket_id": str})
    async def charge(args: dict[str, Any]) -> dict[str, Any]:
        runs.append(args["ticket_id"])
        return _text(f"charge {len(runs)}")

    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[charge])
    args = {"ticket_id": "TCK-1"}

    for tool_use_id in ("toolu_1", "toolu_2"):
        decision, _ = _decision(
            await _gate(channel, config, args, tool_use_id=tool_use_id)
        )
        assert decision == "allow"

    with active_channel(channel):
        delivered = {
            _result_text(await _call(config, args)),
            _result_text(await _call(config, args)),
        }

    assert runs == ["TCK-1", "TCK-1"]
    assert delivered == {"charge 1", "charge 2"}


async def test_a_body_that_raises_reaches_the_model_as_a_tool_error():
    """An exception escaping the hook would fault the job, not the turn."""

    @tool(TOOL_NAME, "Look a ticket up.", {"ticket_id": str})
    async def lookup(args: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("no such ticket")

    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[lookup])
    args = {"ticket_id": "TCK-1"}

    decision, _ = _decision(await _gate(channel, config, args))

    assert decision == "allow"
    with active_channel(channel):
        result = await _call(config, args)

    assert result.isError
    assert "no such ticket" in _result_text(result)


async def test_a_second_interrupt_is_denied_while_one_is_in_flight():
    """The CLI parks both and then erases one, so the second has to be refused."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    await _gate(channel, config, {"ticket_id": "TCK-1"})
    decision, reason = _decision(
        await _gate(channel, config, {"ticket_id": "TCK-2"}, tool_use_id="toolu_2")
    )

    assert decision == "deny"
    assert "already in flight" in reason
    assert channel.pending is not None
    assert channel.pending.interrupt_id == TOOL_USE_ID


async def test_a_call_with_no_tool_use_id_is_denied_rather_than_parked():
    """It could not be correlated with the re-issued call on resume."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    decision, reason = _decision(
        await _gate(channel, config, {"ticket_id": "TCK-1"}, tool_use_id=None)
    )

    assert decision == "deny"
    assert "tool_use id" in reason
    assert channel.pending is None


async def test_interrupt_outside_a_uipath_run_says_what_is_missing():
    """The likeliest mistake is forgetting uipath_tool_server, so name it."""
    with pytest.raises(InterruptOutsideRunError) as caught:
        await interrupt(ESCALATION)

    assert "uipath_tool_server" in str(caught.value)


async def test_a_tool_without_a_uipath_run_around_it_just_runs():
    """The same agent has to work under a plain Claude Agent SDK run."""
    calls: list[str] = []

    @tool(TOOL_NAME, "Look a ticket up.", {"ticket_id": str})
    async def lookup(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args["ticket_id"])
        return _text("open")

    config = uipath_tool_server(SERVER_KEY, tools=[lookup])

    result = await _call(config, {"ticket_id": "TCK-1"})

    assert _result_text(result) == "open"
    assert calls == ["TCK-1"]


async def test_a_handler_reached_without_a_stash_refuses_to_suspend_from_there():
    """A defer the CLI dropped must correct the model, not run the human step."""
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool()])

    with active_channel(channel):
        result = await _call(config, {"ticket_id": "TCK-1"})

    assert result.isError
    assert "never suspended" in _result_text(result)


@pytest.mark.parametrize(
    ("value", "trigger_type"),
    [
        (ESCALATION, UiPathResumeTriggerType.TASK),
        (
            [
                WaitJob(job=Job(key="job-1", id=0)),
                WaitJob(job=Job(key="job-2", id=0)),
            ],
            UiPathResumeTriggerType.JOB,
        ),
        ("Which vendor?", UiPathResumeTriggerType.API),
    ],
)
async def test_the_platform_decides_the_trigger_from_the_value_alone(
    value: Any, trigger_type: UiPathResumeTriggerType
):
    """Nothing here classifies a suspend value, so every model works untouched.

    A list is sibling triggers for one interrupt, resolved by whichever fires
    first, and anything the creator does not recognise degrades to an API
    trigger rather than failing.
    """
    channel = SuspendChannel()
    config = uipath_tool_server(SERVER_KEY, tools=[_review_tool(value=value)])

    await _gate(channel, config, {"ticket_id": "TCK-1"})

    assert channel.pending is not None
    creator = UiPathResumeTriggerCreator()
    targets = (
        channel.pending.value
        if isinstance(channel.pending.value, list)
        else [channel.pending.value]
    )
    for target in targets:
        assert creator._determine_trigger_type(target) is trigger_type


async def test_two_uipath_servers_do_not_pick_up_each_other_results():
    """Same tool name, same arguments, two servers, two different bodies."""
    first = uipath_tool_server("alpha", tools=[_named_echo("alpha")])
    second = uipath_tool_server("beta", tools=[_named_echo("beta")])
    channel = SuspendChannel()
    index = uipath_tool_index({"alpha": first, "beta": second})
    hooks = build_suspend_hooks(channel, index)
    args = {"ticket_id": "TCK-1"}

    for qualified in ("mcp__alpha__review", "mcp__beta__review"):
        hook_input: PreToolUseHookInput = {
            "hook_event_name": "PreToolUse",
            "tool_name": qualified,
            "tool_input": args,
            "tool_use_id": TOOL_USE_ID,
            "session_id": "session-1",
            "transcript_path": "",
            "cwd": "",
        }
        await hooks["PreToolUse"][0].hooks[0](
            hook_input, TOOL_USE_ID, HookContext(signal=None)
        )

    with active_channel(channel):
        assert _result_text(await _call(first, args)) == "alpha"
        assert _result_text(await _call(second, args)) == "beta"


def _named_echo(label: str):
    @tool(TOOL_NAME, "Echo which server answered.", {"ticket_id": str})
    async def echo(args: dict[str, Any]) -> dict[str, Any]:
        return _text(label)

    return echo
