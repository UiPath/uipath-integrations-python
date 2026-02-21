"""Runtime class for executing Agent Framework agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    Content,
    Message,
    WorkflowAgent,
    WorkflowRunResult,
    WorkflowRunState,
)
from pydantic import BaseModel
from uipath.core.chat import UiPathConversationToolCallConfirmationValue
from uipath.core.serialization import serialize_json
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode
from uipath.runtime.events import (
    UiPathRuntimeEvent,
    UiPathRuntimeMessageEvent,
    UiPathRuntimeStateEvent,
    UiPathRuntimeStatePhase,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from .breakpoints import (
    AgentInterruptException,
    create_breakpoint_result,
    inject_breakpoint_middleware,
    remove_breakpoint_middleware,
)
from .errors import UiPathAgentFrameworkErrorCode, UiPathAgentFrameworkRuntimeError
from .messages import AgentFrameworkChatMessagesMapper
from .resumable_storage import ScopedCheckpointStorage, SqliteResumableStorage
from .schema import get_agent_graph, get_entrypoints_schema


class UiPathAgentFrameworkRuntime:
    """A runtime class for executing Agent Framework agents within the UiPath framework."""

    def __init__(
        self,
        agent: WorkflowAgent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        checkpoint_storage: ScopedCheckpointStorage | None = None,
        resumable_storage: SqliteResumableStorage | None = None,
    ):
        self.agent: WorkflowAgent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self.chat = AgentFrameworkChatMessagesMapper()
        self._checkpoint_storage = checkpoint_storage
        self._resumable_storage = resumable_storage
        self._resume_responses: dict[str, Any] | None = None
        self._breakpoint_skip_nodes: dict[str, int] = {}
        self._last_breakpoint_node: str | None = None
        self._last_checkpoint_id: str | None = None
        self._resumed_from_checkpoint_id: str | None = None
        # Track tool nodes that emitted STARTED but not yet COMPLETED.
        # Persists across _stream_workflow() calls (same runtime instance
        # reused by UiPathChatRuntime's while loop), allowing us to emit
        # synthetic COMPLETED events on HITL resume when the framework
        # doesn't surface function_result in output/executor_completed.
        self._pending_tool_nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    async def _get_latest_checkpoint_id(self) -> str | None:
        """Get the latest checkpoint ID for this workflow."""
        if not self._checkpoint_storage:
            return None
        workflow_name = self.agent.workflow.name
        checkpoint = await self._checkpoint_storage.get_latest(
            workflow_name=workflow_name
        )
        return checkpoint.checkpoint_id if checkpoint else None

    async def _save_breakpoint_state(
        self, original_input: str, checkpoint_id: str | None = None
    ) -> None:
        """Persist breakpoint state to KV storage for resume.

        skip_nodes is a dict mapping executor_id → pass-through count.
        Each count records how many times the executor must be allowed
        to run before re-arming its breakpoint.  The count is incremented
        every time the same executor hits a breakpoint again (cyclic
        graphs, GroupChat orchestrators).

        checkpoint_id may be None when the breakpoint fired before any
        new checkpoint was created (e.g. the very first executor of a
        fresh turn).  In that case the resume path will replay from
        original_input with skip counts instead of restoring a checkpoint.
        """
        if not self._resumable_storage:
            return
        state = {
            "skip_nodes": dict(self._breakpoint_skip_nodes),
            "last_breakpoint_node": self._last_breakpoint_node,
            "checkpoint_id": checkpoint_id,
            "original_input": original_input,
        }
        await self._resumable_storage.set_value(
            self.runtime_id, "breakpoint", "state", state
        )

    async def _load_breakpoint_state(self) -> dict[str, Any] | None:
        """Load breakpoint state from KV storage."""
        if not self._resumable_storage:
            return None
        state = await self._resumable_storage.get_value(
            self.runtime_id, "breakpoint", "state"
        )
        if state and isinstance(state, dict):
            self._breakpoint_skip_nodes = dict(state.get("skip_nodes", {}))
            self._last_breakpoint_node = state.get("last_breakpoint_node")
            self._last_checkpoint_id = state.get("checkpoint_id")
            return state
        return None

    def _get_breakpoint_skip(self) -> dict[str, int]:
        """Get the skip_nodes dict for breakpoint injection.

        Returns accumulated skip counts.  The counts are reset whenever
        a checkpoint advancement is detected (see the breakpoint handlers
        in execute / _stream_workflow) so they stay correct regardless
        of whether checkpoints advance per-executor or stay at a coarser
        granularity.
        """
        return dict(self._breakpoint_skip_nodes)

    # ------------------------------------------------------------------
    # Session helpers (multi-turn conversation history)
    # ------------------------------------------------------------------

    async def _load_session(self) -> AgentSession:
        """Load or create an AgentSession for this runtime_id.

        Sessions maintain conversation history across turns. This is separate
        from checkpoints which handle workflow interruption/resume.
        """
        if self._resumable_storage:
            session_data = await self._resumable_storage.get_value(
                self.runtime_id, "session", "data"
            )
            if session_data is not None and isinstance(session_data, dict):
                return AgentSession.from_dict(session_data)

        return self.agent.create_session(session_id=self.runtime_id)

    async def _save_session(self, session: AgentSession) -> None:
        """Persist the session state after execution."""
        if self._resumable_storage:
            session_data = session.to_dict()
            await self._resumable_storage.set_value(
                self.runtime_id, "session", "data", session_data
            )

    def _apply_session_to_executors(self, session: AgentSession) -> None:
        """Propagate the loaded session to all AgentExecutors in the workflow.

        Each AgentExecutor uses a unique source_id key inside session.state,
        so sharing one session across all executors is safe and ensures
        conversation history is preserved across turns.
        """
        workflow = self.agent.workflow
        for executor in workflow.executors.values():
            if isinstance(executor, AgentExecutor):
                executor._session = session

    def _get_session_from_executors(self) -> AgentSession | None:
        """Extract the most complete session from AgentExecutors in the workflow.

        After checkpoint restore each executor receives its own independent
        session copy (unlike fresh runs where all executors share one object).
        Only the executor that processed the HITL/breakpoint response will
        have the updated conversation history. We return the session with the
        most messages to ensure the complete history is persisted.
        """
        workflow = self.agent.workflow
        best_session: AgentSession | None = None
        best_msg_count = -1
        for executor in workflow.executors.values():
            if isinstance(executor, AgentExecutor) and executor._session is not None:
                msg_count = self._count_session_messages(executor._session)
                if msg_count > best_msg_count:
                    best_msg_count = msg_count
                    best_session = executor._session
        return best_session

    @staticmethod
    def _count_session_messages(session: AgentSession) -> int:
        """Count total messages across all provider keys in a session's state."""
        count = 0
        for value in session.state.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                if isinstance(messages, list):
                    count += len(messages)
        return count

    # ------------------------------------------------------------------
    # HITL helpers (tool approval flow)
    # ------------------------------------------------------------------

    async def _save_hitl_state(self, request_info_map: dict[str, Any]) -> None:
        """Persist original function_approval_request Content objects for resume."""
        if not self._resumable_storage:
            return
        serialized = {}
        for request_id, content in request_info_map.items():
            if (
                isinstance(content, Content)
                and content.type == "function_approval_request"
            ):
                serialized[request_id] = content.to_dict()
        if serialized:
            await self._resumable_storage.set_value(
                self.runtime_id, "hitl", "state", serialized
            )

    async def _load_hitl_state(self) -> dict[str, Content] | None:
        """Load stored HITL state for converting resume responses."""
        if not self._resumable_storage:
            return None
        state = await self._resumable_storage.get_value(
            self.runtime_id, "hitl", "state"
        )
        if state and isinstance(state, dict):
            return {rid: Content.from_dict(data) for rid, data in state.items()}
        return None

    @staticmethod
    def _convert_to_hitl_output(request_info_map: dict[str, Any]) -> dict[str, Any]:
        """Convert request_info Content objects to UiPathConversationToolCallConfirmationValue."""
        output = {}
        for request_id, content in request_info_map.items():
            if (
                isinstance(content, Content)
                and content.type == "function_approval_request"
            ):
                fc = content.function_call  # nested Content(function_call)
                if fc is None:
                    continue
                args = fc.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                elif not isinstance(args, dict):
                    args = {}
                output[request_id] = UiPathConversationToolCallConfirmationValue(
                    tool_call_id=fc.call_id or "",
                    tool_name=fc.name or "",
                    input_schema={},
                    input_value=args,
                )
            else:
                output[request_id] = content  # pass through unknown types
        return output

    async def _convert_resume_responses(self, input: dict[str, Any]) -> dict[str, Any]:
        """Convert CAS approval responses to framework Content objects."""
        hitl_state = await self._load_hitl_state()
        if not hitl_state:
            return input  # no stored state, pass through

        converted = {}
        for request_id, response_data in input.items():
            original = hitl_state.get(request_id)
            if (
                original
                and isinstance(original, Content)
                and original.type == "function_approval_request"
            ):
                # Extract approval from CAS response format:
                # {"type": "uipath_cas_tool_call_confirmation", "value": {"approved": bool}}
                approved = False
                if isinstance(response_data, dict):
                    value = response_data.get("value", response_data)
                    if isinstance(value, dict):
                        approved = value.get("approved", False)
                converted[request_id] = original.to_function_approval_response(approved)
            else:
                converted[request_id] = response_data
        return converted

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input and return the result."""
        session = None
        try:
            is_resuming = bool(options and options.resume)

            workflow = self.agent.workflow

            # Capture the latest checkpoint BEFORE workflow.run() so we can
            # detect whether a NEW checkpoint was created during this execution.
            # Without this, breakpoints that fire before any new checkpoint
            # (e.g. the first executor of turn 2) would save a stale
            # checkpoint from the previous turn, causing the resume to
            # restore completed state instead of replaying from input.
            baseline_checkpoint_id = await self._get_latest_checkpoint_id()

            if is_resuming and input is not None:
                # HITL resume: checkpoint restores executor state (including session)
                self._resume_responses = await self._convert_resume_responses(input)

                # Inject breakpoints with accumulated skip counts so that
                # breakpoints don't re-fire on the same executor after HITL
                # approval (prevents breakpoint→HITL→breakpoint loop).
                if options and options.breakpoints:
                    await self._load_breakpoint_state()
                    inject_breakpoint_middleware(
                        self.agent,
                        options.breakpoints,
                        self._get_breakpoint_skip(),
                    )
                    # _load_breakpoint_state sets _last_checkpoint_id as a
                    # side effect. Clear it so it doesn't contaminate later runs.
                    self._last_checkpoint_id = None

                if self._resume_responses:
                    checkpoint_id = await self._get_latest_checkpoint_id()
                    self._resumed_from_checkpoint_id = checkpoint_id
                    result = await workflow.run(
                        responses=self._resume_responses,
                        checkpoint_id=checkpoint_id,
                        checkpoint_storage=self._checkpoint_storage,
                    )
                    self._resume_responses = None
                else:
                    self._resumed_from_checkpoint_id = None
                    result = await workflow.run(
                        message="",
                        checkpoint_storage=self._checkpoint_storage,
                    )
            elif is_resuming:
                # Breakpoint resume: restore from checkpoint
                bp_state = await self._load_breakpoint_state()
                checkpoint_id = self._last_checkpoint_id
                self._resumed_from_checkpoint_id = checkpoint_id
                original_input = bp_state.get("original_input", "") if bp_state else ""

                # Inject breakpoints with accumulated skip counts
                if options and options.breakpoints:
                    inject_breakpoint_middleware(
                        self.agent, options.breakpoints, self._get_breakpoint_skip()
                    )

                if checkpoint_id:
                    result = await workflow.run(
                        checkpoint_id=checkpoint_id,
                        checkpoint_storage=self._checkpoint_storage,
                    )
                else:
                    result = await workflow.run(
                        message=original_input,
                        checkpoint_storage=self._checkpoint_storage,
                    )
            else:
                # Fresh run: load session for multi-turn conversation history
                self._resumed_from_checkpoint_id = None
                session = await self._load_session()
                self._apply_session_to_executors(session)

                # Inject breakpoints for fresh runs
                if options and options.breakpoints:
                    inject_breakpoint_middleware(self.agent, options.breakpoints)

                user_input = self._prepare_input(input)
                result = await workflow.run(
                    message=user_input,
                    checkpoint_storage=self._checkpoint_storage,
                )

            # After resume paths the checkpoint restores the session into
            # executors directly, so the local ``session`` is still None.
            # Extract it so it can be persisted after completion.
            if session is None:
                session = self._get_session_from_executors()

            # Check for HITL suspension (framework's request_info mechanism)
            request_info_events = result.get_request_info_events()
            hitl_requests = {
                e.request_id: e.data
                for e in request_info_events
                if isinstance(e.data, Content)
                and e.data.type == "function_approval_request"
            }
            if hitl_requests:
                await self._save_hitl_state(hitl_requests)
                if session is not None:
                    await self._save_session(session)
                return UiPathRuntimeResult(
                    output=self._convert_to_hitl_output(hitl_requests),
                    status=UiPathRuntimeStatus.SUSPENDED,
                )

            if session is not None:
                await self._save_session(session)
            output = self._extract_workflow_output(result)
            return self._create_success_result(output)
        except AgentInterruptException as e:
            if session is not None:
                await self._save_session(session)
            if e.is_breakpoint:
                node_id = (
                    e.suspend_value.get("node_id", "")
                    if isinstance(e.suspend_value, dict)
                    else ""
                )
                # Detect checkpoint advancement and reset counts if needed
                latest_checkpoint = await self._get_latest_checkpoint_id()
                if (
                    latest_checkpoint
                    and self._resumed_from_checkpoint_id
                    and latest_checkpoint != self._resumed_from_checkpoint_id
                ):
                    self._breakpoint_skip_nodes = {}
                self._breakpoint_skip_nodes[node_id] = (
                    self._breakpoint_skip_nodes.get(node_id, 0) + 1
                )
                self._last_breakpoint_node = node_id
                original_input = self._prepare_input(input) if not is_resuming else ""
                # Only save checkpoint_id if it was created during THIS run.
                # If latest == baseline, no new checkpoint was created (e.g.
                # breakpoint on the first executor of a fresh turn) — save
                # the checkpoint we resumed from (if any) so we don't lose
                # it and replay from scratch on the next resume.
                # For fresh turns _resumed_from_checkpoint_id is None, which
                # correctly prevents using a stale checkpoint from the
                # previous turn.
                effective_checkpoint = (
                    latest_checkpoint
                    if latest_checkpoint != baseline_checkpoint_id
                    else self._resumed_from_checkpoint_id
                )
                await self._save_breakpoint_state(
                    original_input, checkpoint_id=effective_checkpoint
                )
                return create_breakpoint_result(e)
            return self._create_suspended_result(e)
        except Exception as e:
            raise self._create_runtime_error(e) from e
        finally:
            remove_breakpoint_middleware(self.agent)

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream workflow execution events in real-time."""
        try:
            is_resuming = bool(options and options.resume)
            session = None

            if is_resuming and input is not None:
                # HITL resume: input contains response data
                self._resume_responses = await self._convert_resume_responses(input)
                user_input = self._prepare_input(None)

                # Inject breakpoints with accumulated skip counts so that
                # breakpoints don't re-fire on the same executor after HITL
                # approval (prevents breakpoint→HITL→breakpoint loop).
                if options and options.breakpoints:
                    await self._load_breakpoint_state()
                    inject_breakpoint_middleware(
                        self.agent,
                        options.breakpoints,
                        self._get_breakpoint_skip(),
                    )
                    # _load_breakpoint_state sets _last_checkpoint_id as a
                    # side effect. Clear it so _stream_workflow doesn't
                    # mistake a subsequent fresh run for a breakpoint resume.
                    self._last_checkpoint_id = None

            elif is_resuming:
                # Breakpoint resume: restore original_input and session
                self._resume_responses = None
                bp_state = await self._load_breakpoint_state()
                user_input = bp_state.get("original_input", "") if bp_state else ""

                # Load session for context preservation across the breakpoint
                session = await self._load_session()
                self._apply_session_to_executors(session)

                # Inject breakpoints — skip strategy depends on resume mode
                if options and options.breakpoints:
                    inject_breakpoint_middleware(
                        self.agent, options.breakpoints, self._get_breakpoint_skip()
                    )

            else:
                # Fresh run — clear stale resume state from previous turns
                self._resume_responses = None
                self._last_checkpoint_id = None
                user_input = self._prepare_input(input)

                # Load session for multi-turn conversation history
                session = await self._load_session()
                self._apply_session_to_executors(session)

                # Inject breakpoints for fresh runs
                if options and options.breakpoints:
                    inject_breakpoint_middleware(self.agent, options.breakpoints)

            agent_name = self.agent.name or "agent"

            async for event in self._stream_workflow(
                user_input, agent_name, is_resuming, session
            ):
                yield event

        except Exception as e:
            raise self._create_runtime_error(e) from e
        finally:
            remove_breakpoint_middleware(self.agent)

    async def _stream_workflow(
        self,
        user_input: str,
        agent_name: str,
        is_resuming: bool = False,
        session: AgentSession | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream workflow execution with real-time executor lifecycle events."""
        assert isinstance(self.agent, WorkflowAgent)
        workflow = self.agent.workflow

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.STARTED,
        )

        # On HITL resume, emit COMPLETED for tool nodes that were left
        # pending when the previous stream suspended. The framework
        # doesn't surface function_result in output/executor_completed
        # for handoff workflows, so we synthesize these events here.
        if is_resuming and self._pending_tool_nodes:
            for tool_node in list(self._pending_tool_nodes):
                yield UiPathRuntimeStateEvent(
                    payload={},
                    node_name=tool_node,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            self._pending_tool_nodes.clear()

        # Capture the latest checkpoint BEFORE workflow.run() so we can
        # detect whether a NEW checkpoint was created during this execution.
        baseline_checkpoint_id = await self._get_latest_checkpoint_id()

        # Choose workflow.run() mode based on resume type
        if self._resume_responses:
            # HITL resume: pass responses to workflow with checkpoint
            checkpoint_id = await self._get_latest_checkpoint_id()
            self._resumed_from_checkpoint_id = checkpoint_id
            response_stream = workflow.run(
                responses=self._resume_responses,
                checkpoint_id=checkpoint_id,
                checkpoint_storage=self._checkpoint_storage,
                stream=True,
            )
            self._resume_responses = None
        elif self._last_checkpoint_id:
            # Breakpoint resume with checkpoint: restore and continue
            checkpoint_id = self._last_checkpoint_id
            self._resumed_from_checkpoint_id = checkpoint_id
            self._last_checkpoint_id = None
            response_stream = workflow.run(
                checkpoint_id=checkpoint_id,
                checkpoint_storage=self._checkpoint_storage,
                stream=True,
            )
        else:
            # Fresh run (or breakpoint resume without checkpoint — uses original_input)
            self._resumed_from_checkpoint_id = None
            response_stream = workflow.run(
                message=user_input,
                checkpoint_storage=self._checkpoint_storage,
                stream=True,
            )

        request_info_map: dict[str, Any] = {}
        is_suspended = False
        # Track which tool event phases were emitted per executor via output
        # events. When the workflow filters output events (e.g. GroupChat),
        # tool events are extracted from executor_completed data as a fallback.
        # Tracking phases (not just executor_ids) lets us handle HITL resume
        # where function_call (STARTED) is in output but function_result
        # (COMPLETED) is only in executor_completed.
        executor_tool_phases: dict[str, set[UiPathRuntimeStatePhase]] = {}

        # Emit an early STARTED event for the start executor so the graph
        # visualization shows it immediately rather than after it finishes.
        # The framework's _run_workflow_with_tracing awaits the entire start
        # executor before yielding any executor events, which means the real
        # executor_invoked arrives only after execution completes.
        pre_emitted_executor: str | None = None
        if not is_resuming:
            start_id = workflow.start_executor_id
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=start_id,
                phase=UiPathRuntimeStatePhase.STARTED,
            )
            pre_emitted_executor = start_id

        try:
            async for event in response_stream:
                if event.type == "request_info":
                    data = event.data
                    request_info_map[event.request_id] = data
                elif event.type == "executor_invoked":
                    # Skip the duplicate for the start executor we already emitted
                    if (
                        pre_emitted_executor
                        and event.executor_id == pre_emitted_executor
                    ):
                        pre_emitted_executor = None
                        continue
                    yield UiPathRuntimeStateEvent(
                        payload=self._serialize_event_data(event.data),
                        node_name=event.executor_id,
                        phase=UiPathRuntimeStatePhase.STARTED,
                    )
                elif event.type == "executor_completed":
                    # Extract tool state events from executor_completed data,
                    # skipping phases already emitted via output events.
                    # This handles three scenarios:
                    # 1. GroupChat (no output events): emit all from completed
                    # 2. Normal (both in output): skip all from completed
                    # 3. HITL resume (only STARTED in output): emit COMPLETED
                    if event.executor_id:
                        emitted_phases = executor_tool_phases.get(
                            event.executor_id, set()
                        )
                        for tool_event in self._extract_tool_state_events(
                            event.data, event.executor_id
                        ):
                            if tool_event.phase not in emitted_phases:
                                # Track pending tool nodes
                                if tool_event.node_name:
                                    if (
                                        tool_event.phase
                                        == UiPathRuntimeStatePhase.STARTED
                                    ):
                                        self._pending_tool_nodes.add(
                                            tool_event.node_name
                                        )
                                    elif (
                                        tool_event.phase
                                        == UiPathRuntimeStatePhase.COMPLETED
                                    ):
                                        self._pending_tool_nodes.discard(
                                            tool_event.node_name
                                        )
                                yield tool_event
                    yield UiPathRuntimeStateEvent(
                        payload=self._serialize_event_data(
                            self._filter_completed_data(event.data)
                        ),
                        node_name=event.executor_id,
                        phase=UiPathRuntimeStatePhase.COMPLETED,
                    )
                elif event.type == "output":
                    executor_id = getattr(event, "executor_id", None) or ""
                    tool_events = self._extract_tool_state_events(
                        event.data, executor_id
                    )
                    for tool_event in tool_events:
                        executor_tool_phases.setdefault(executor_id, set()).add(
                            tool_event.phase
                        )
                        # Track pending tool nodes across stream iterations
                        if tool_event.node_name:
                            if tool_event.phase == UiPathRuntimeStatePhase.STARTED:
                                self._pending_tool_nodes.add(tool_event.node_name)
                            elif tool_event.phase == UiPathRuntimeStatePhase.COMPLETED:
                                self._pending_tool_nodes.discard(tool_event.node_name)
                        yield tool_event
                    for msg_event in self._extract_workflow_messages(event.data):
                        yield UiPathRuntimeMessageEvent(payload=msg_event)

                # Detect workflow suspension via state
                if (
                    event.type == "status"
                    and event.state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
                ):
                    is_suspended = True
        except AgentInterruptException as e:
            # Breakpoint or HITL interrupt fired inside an inner agent
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )

            for msg_event in self.chat.close_message():
                yield UiPathRuntimeMessageEvent(payload=msg_event)

            # After resume paths the checkpoint restores the session into
            # executors directly, so the local ``session`` may still be None.
            if session is None:
                session = self._get_session_from_executors()
            if session is not None:
                await self._save_session(session)

            if e.is_breakpoint:
                node_id = (
                    e.suspend_value.get("node_id", "")
                    if isinstance(e.suspend_value, dict)
                    else ""
                )
                # Detect checkpoint advancement and reset counts if needed
                latest_checkpoint = await self._get_latest_checkpoint_id()
                if (
                    latest_checkpoint
                    and self._resumed_from_checkpoint_id
                    and latest_checkpoint != self._resumed_from_checkpoint_id
                ):
                    self._breakpoint_skip_nodes = {}
                self._breakpoint_skip_nodes[node_id] = (
                    self._breakpoint_skip_nodes.get(node_id, 0) + 1
                )
                self._last_breakpoint_node = node_id
                # Only save checkpoint_id if it was created during THIS run.
                # Fall back to the checkpoint we resumed from (if any) to
                # avoid replaying from scratch on the next resume.
                effective_checkpoint = (
                    latest_checkpoint
                    if latest_checkpoint != baseline_checkpoint_id
                    else self._resumed_from_checkpoint_id
                )
                await self._save_breakpoint_state(
                    user_input, checkpoint_id=effective_checkpoint
                )
                yield create_breakpoint_result(e)
            else:
                yield self._create_suspended_result(e)
            return

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.COMPLETED,
        )

        for msg_event in self.chat.close_message():
            yield UiPathRuntimeMessageEvent(payload=msg_event)

        # After resume paths the checkpoint restores the session into
        # executors directly, so the local ``session`` may still be None.
        if session is None:
            session = self._get_session_from_executors()
        if session is not None:
            await self._save_session(session)

        # Filter to only include HITL approval requests in SUSPENDED output.
        # Other request_info events (e.g. HandoffAgentUserRequest) are part of
        # the workflow's multi-turn mechanism and handled separately.
        hitl_requests = {
            rid: data
            for rid, data in request_info_map.items()
            if isinstance(data, Content) and data.type == "function_approval_request"
        }
        if is_suspended and hitl_requests:
            hitl_output = self._convert_to_hitl_output(hitl_requests)
            await self._save_hitl_state(hitl_requests)
            yield UiPathRuntimeResult(
                output=hitl_output,
                status=UiPathRuntimeStatus.SUSPENDED,
            )
        else:
            final_result = await response_stream.get_final_response()
            output = self._extract_workflow_output(final_result)
            yield self._create_success_result(output)

    @staticmethod
    def _filter_completed_data(data: Any) -> Any:
        """Strip streaming AgentResponseUpdate chunks from executor_completed data.

        The framework packs sent_messages + yielded_outputs into the
        executor_completed event. In streaming mode the yielded_outputs are
        individual AgentResponseUpdate token chunks which bloat the payload.
        Keep only the non-update items (e.g. AgentExecutorResponse).
        """
        if not isinstance(data, list):
            return data
        filtered = [item for item in data if not isinstance(item, AgentResponseUpdate)]
        return filtered if filtered else None

    @staticmethod
    def _serialize_event_data(data: Any) -> dict[str, Any]:
        """Serialize workflow event data into a JSON-safe payload."""
        if data is None:
            return {}
        try:
            safe = json.loads(serialize_json(data))
            if isinstance(safe, dict):
                return safe
            return {"data": safe}
        except Exception:
            return {"data": str(data)}

    @staticmethod
    def _extract_tool_state_events(
        data: Any, executor_id: str
    ) -> list[UiPathRuntimeStateEvent]:
        """Extract tool-node state events from output data containing function calls/results.

        Looks for Content objects with type 'function_call' (tool start) and
        'function_result' (tool end) and emits STARTED/COMPLETED StateEvents
        for the '{executor_id}_tools' node.
        """
        contents: list[Any] = []

        if isinstance(data, AgentExecutorResponse):
            return UiPathAgentFrameworkRuntime._extract_tool_state_events(
                data.agent_response, executor_id
            )
        elif isinstance(data, AgentResponseUpdate):
            contents = list(data.contents or [])
        elif isinstance(data, AgentResponse):
            for message in data.messages or []:
                contents.extend(message.contents or [])
        elif isinstance(data, Message):
            contents = list(data.contents or [])
        elif isinstance(data, list):
            events: list[UiPathRuntimeStateEvent] = []
            for item in data:
                events.extend(
                    UiPathAgentFrameworkRuntime._extract_tool_state_events(
                        item, executor_id
                    )
                )
            return events

        tool_node = f"{executor_id}_tools"
        tool_events: list[UiPathRuntimeStateEvent] = []
        for content in contents:
            if isinstance(content, Content):
                if content.type == "function_call" and content.name:
                    tool_events.append(
                        UiPathRuntimeStateEvent(
                            payload={"tool_name": content.name},
                            node_name=tool_node,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )
                    )
                elif content.type == "function_result":
                    tool_events.append(
                        UiPathRuntimeStateEvent(
                            payload={},
                            node_name=tool_node,
                            phase=UiPathRuntimeStatePhase.COMPLETED,
                        )
                    )
        return tool_events

    @staticmethod
    def _extract_contents(data: Any) -> list[Any]:
        """Extract Content objects from any workflow data type."""
        contents: list[Any] = []
        if isinstance(data, AgentExecutorResponse):
            return UiPathAgentFrameworkRuntime._extract_contents(data.agent_response)
        elif isinstance(data, AgentResponseUpdate):
            contents = list(data.contents or [])
        elif isinstance(data, AgentResponse):
            for message in data.messages or []:
                contents.extend(message.contents or [])
        elif isinstance(data, Message):
            contents = list(data.contents or [])
        elif isinstance(data, list):
            for item in data:
                contents.extend(UiPathAgentFrameworkRuntime._extract_contents(item))
        return contents

    def _extract_workflow_messages(self, data: Any) -> list[Any]:
        """Extract UiPath conversation message events from workflow output data."""
        events: list[Any] = []
        for content in self._extract_contents(data):
            if isinstance(content, Content):
                # Skip HITL approval requests — handled by the suspension mechanism
                if content.type == "function_approval_request":
                    continue
                events.extend(self.chat.map_streaming_content(content))
        return events

    def _extract_workflow_output(self, result: WorkflowRunResult) -> Any:
        """Extract output from WorkflowRunResult."""
        outputs = result.get_outputs()

        if not outputs:
            return ""

        texts: list[str] = []
        for data in outputs:
            text = self._extract_text_from_data(data)
            if text:
                texts.append(text)

        if texts:
            return "\n\n".join(texts)

        try:
            return json.loads(serialize_json(outputs[-1]))
        except Exception:
            return str(outputs[-1])

    @staticmethod
    def _extract_text_from_data(data: Any) -> str:
        """Extract text from any workflow data type."""
        if isinstance(data, (AgentResponseUpdate, AgentResponse)):
            return data.text or ""
        if isinstance(data, Message):
            return "".join(
                c.text for c in (data.contents or []) if hasattr(c, "text") and c.text
            )
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            parts: list[str] = []
            for item in data:
                if isinstance(item, Message):
                    text = "".join(
                        c.text
                        for c in (item.contents or [])
                        if hasattr(c, "text") and c.text
                    )
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, list):
                    for inner in item:
                        if isinstance(inner, Message) and inner.role == "assistant":
                            text = "".join(
                                c.text
                                for c in (inner.contents or [])
                                if hasattr(c, "text") and c.text
                            )
                            if text:
                                parts.append(text)
            return "\n\n".join(parts)
        return ""

    def _prepare_input(self, input: dict[str, Any] | None) -> str:
        """Prepare input string from UiPath input dictionary."""
        if not input:
            return ""

        if "messages" in input:
            return self.chat.map_messages_to_input(input["messages"])

        return json.dumps(input)

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        serialized_output = json.loads(serialize_json(output))

        if not isinstance(serialized_output, dict):
            serialized_output = {
                "messages": [
                    {
                        "role": "assistant",
                        "contentParts": [{"data": {"inline": serialized_output}}],
                    }
                ]
            }

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_suspended_result(
        self, exc: AgentInterruptException
    ) -> UiPathRuntimeResult:
        """Create a SUSPENDED result from an AgentInterruptException."""
        interrupt_value = exc.suspend_value
        if isinstance(interrupt_value, BaseModel):
            interrupt_value = interrupt_value.model_dump(by_alias=True)

        return UiPathRuntimeResult(
            output={exc.interrupt_id: interrupt_value},
            status=UiPathRuntimeStatus.SUSPENDED,
        )

    def _create_runtime_error(self, e: Exception) -> UiPathAgentFrameworkRuntimeError:
        """Handle execution errors and create appropriate runtime error."""
        if isinstance(e, UiPathAgentFrameworkRuntimeError):
            return e

        # Let AgentInterruptException propagate (handled by caller)
        if isinstance(e, AgentInterruptException):
            raise e

        detail = f"Error: {str(e)}"

        if isinstance(e, json.JSONDecodeError):
            return UiPathAgentFrameworkRuntimeError(
                UiPathErrorCode.INPUT_INVALID_JSON,
                "Invalid JSON input",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, TimeoutError):
            return UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.TIMEOUT_ERROR,
                "Agent execution timed out",
                detail,
                UiPathErrorCategory.USER,
            )

        return UiPathAgentFrameworkRuntimeError(
            UiPathAgentFrameworkErrorCode.AGENT_EXECUTION_FAILURE,
            "Agent execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this Agent Framework runtime."""
        entrypoints_schema = get_entrypoints_schema(self.agent)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint or "default",
            uniqueId=str(uuid4()),
            type="agent",
            input=entrypoints_schema.get("input", {}),
            output=entrypoints_schema.get("output", {}),
            graph=get_agent_graph(self.agent),
            metadata=None,
        )

    async def dispose(self) -> None:
        """Cleanup runtime resources."""
        pass
