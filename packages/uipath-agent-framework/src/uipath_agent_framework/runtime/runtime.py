"""Runtime class for executing Agent Framework agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from agent_framework import (
    AgentExecutor,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    Content,
    Message,
    WorkflowAgent,
    WorkflowRunResult,
)
from pydantic import BaseModel
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
    create_breakpoint_result,
    inject_breakpoint_middleware,
    remove_breakpoint_middleware,
)
from .errors import UiPathAgentFrameworkErrorCode, UiPathAgentFrameworkRuntimeError
from .interrupt import AgentInterruptException
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
        self._breakpoint_skip_nodes: set[str] = set()
        self._last_checkpoint_id: str | None = None

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

    async def _save_breakpoint_state(self, original_input: str) -> None:
        """Persist breakpoint state to KV storage for resume.

        The skip_nodes set accumulates across resumes so that concurrent
        executors breakpointed in the same superstep are all skipped on
        subsequent resumes (prevents the infinite-cycle bug).
        """
        if not self._resumable_storage:
            return
        checkpoint_id = await self._get_latest_checkpoint_id()
        state = {
            "skip_nodes": list(self._breakpoint_skip_nodes),
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
            self._breakpoint_skip_nodes = set(state.get("skip_nodes", []))
            self._last_checkpoint_id = state.get("checkpoint_id")
            return state
        return None

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

            if is_resuming and input is not None:
                # HITL resume: checkpoint restores executor state (including session)
                self._resume_responses = input

                # Inject breakpoints (no skip needed for HITL resume)
                if options and options.breakpoints:
                    inject_breakpoint_middleware(self.agent, options.breakpoints)

                if self._resume_responses:
                    checkpoint_id = await self._get_latest_checkpoint_id()
                    result = await workflow.run(
                        responses=self._resume_responses,
                        checkpoint_id=checkpoint_id,
                        checkpoint_storage=self._checkpoint_storage,
                    )
                    self._resume_responses = None
                else:
                    result = await workflow.run(
                        message="",
                        checkpoint_storage=self._checkpoint_storage,
                    )
            elif is_resuming:
                # Breakpoint resume: restore from checkpoint
                bp_state = await self._load_breakpoint_state()
                checkpoint_id = self._last_checkpoint_id
                original_input = bp_state.get("original_input", "") if bp_state else ""

                # Inject breakpoints, skipping all previously-resumed executors
                if options and options.breakpoints:
                    inject_breakpoint_middleware(
                        self.agent, options.breakpoints, self._breakpoint_skip_nodes
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
                self._breakpoint_skip_nodes.add(node_id)
                original_input = self._prepare_input(input) if not is_resuming else ""
                await self._save_breakpoint_state(original_input)
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
                self._resume_responses = input
                user_input = self._prepare_input(None)

                # Inject breakpoints (no skip needed for HITL resume)
                if options and options.breakpoints:
                    inject_breakpoint_middleware(self.agent, options.breakpoints)

            elif is_resuming:
                # Breakpoint resume: restore original_input and session
                self._resume_responses = None
                bp_state = await self._load_breakpoint_state()
                user_input = bp_state.get("original_input", "") if bp_state else ""

                # Load session for context preservation across the breakpoint
                session = await self._load_session()
                self._apply_session_to_executors(session)

                # Inject breakpoints, skipping all previously-resumed executors
                if options and options.breakpoints:
                    inject_breakpoint_middleware(
                        self.agent, options.breakpoints, self._breakpoint_skip_nodes
                    )

            else:
                # Fresh run
                self._resume_responses = None
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

        # Choose workflow.run() mode based on resume type
        if self._resume_responses:
            # HITL resume: pass responses to workflow with checkpoint
            checkpoint_id = await self._get_latest_checkpoint_id()
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
            self._last_checkpoint_id = None
            response_stream = workflow.run(
                checkpoint_id=checkpoint_id,
                checkpoint_storage=self._checkpoint_storage,
                stream=True,
            )
        else:
            # Fresh run (or breakpoint resume without checkpoint — uses original_input)
            response_stream = workflow.run(
                message=user_input,
                checkpoint_storage=self._checkpoint_storage,
                stream=True,
            )

        request_info_map: dict[str, Any] = {}
        is_suspended = False
        # Track executors whose tool events were emitted via output events.
        # When the workflow filters output events (e.g. GroupChat), tool events
        # are instead extracted from executor_completed data as a fallback.
        executors_with_tool_outputs: set[str] = set()

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
                    request_info_map[event.request_id] = event.data
                elif event.type == "executor_invoked":
                    # Skip the duplicate for the start executor we already emitted
                    if pre_emitted_executor and event.executor_id == pre_emitted_executor:
                        pre_emitted_executor = None
                        continue
                    yield UiPathRuntimeStateEvent(
                        payload=self._serialize_event_data(event.data),
                        node_name=event.executor_id,
                        phase=UiPathRuntimeStatePhase.STARTED,
                    )
                elif event.type == "executor_completed":
                    # When output events were filtered by the workflow (e.g.
                    # GroupChat where participants are not output executors),
                    # extract tool state events from the completed data instead.
                    if event.executor_id not in executors_with_tool_outputs:
                        for tool_event in self._extract_tool_state_events(
                            event.data, event.executor_id
                        ):
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
                    if tool_events:
                        executors_with_tool_outputs.add(executor_id)
                    for tool_event in tool_events:
                        yield tool_event
                    for msg_event in self._extract_workflow_messages(event.data):
                        yield UiPathRuntimeMessageEvent(payload=msg_event)

                # Detect workflow suspension via state
                if event.type == "status" and str(event.state) == "IDLE_WITH_PENDING_REQUESTS":
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

            if session is not None:
                await self._save_session(session)

            if e.is_breakpoint:
                node_id = (
                    e.suspend_value.get("node_id", "")
                    if isinstance(e.suspend_value, dict)
                    else ""
                )
                self._breakpoint_skip_nodes.add(node_id)
                await self._save_breakpoint_state(user_input)
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

        if session is not None:
            await self._save_session(session)

        if is_suspended and request_info_map:
            yield UiPathRuntimeResult(
                output=request_info_map,
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

        if isinstance(data, AgentResponseUpdate):
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

    def _extract_workflow_messages(self, data: Any) -> list[Any]:
        """Extract UiPath conversation message events from workflow output data."""
        events: list[Any] = []
        contents: list[Any] = []

        if isinstance(data, AgentResponseUpdate):
            contents = list(data.contents or [])
        elif isinstance(data, AgentResponse):
            for message in data.messages or []:
                contents.extend(message.contents or [])
        elif isinstance(data, Message):
            contents = list(data.contents or [])
        elif isinstance(data, list):
            for item in data:
                events.extend(self._extract_workflow_messages(item))
            return events

        for content in contents:
            if isinstance(content, Content):
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
