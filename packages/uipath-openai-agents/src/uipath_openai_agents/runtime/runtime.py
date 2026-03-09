"""Runtime class for executing OpenAI Agents within the UiPath framework."""

import inspect
import json
from collections.abc import Mapping
from typing import Any, AsyncGenerator
from uuid import uuid4

from agents import Agent, Runner, RunState, SQLiteSession
from agents.items import ToolApprovalItem
from pydantic import BaseModel
from uipath.core.serialization import serialize_json
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathRuntimeStorageProtocol,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode
from uipath.runtime.events import (
    UiPathRuntimeEvent,
    UiPathRuntimeMessageEvent,
    UiPathRuntimeStateEvent,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from .context import get_agent_context_type, parse_input_to_context
from .errors import UiPathOpenAIAgentsErrorCode, UiPathOpenAIAgentsRuntimeError
from .schema import get_agent_schema, get_entrypoints_schema

_RUN_STATE_NAMESPACE = "openai_agents"
_RUN_STATE_KEY = "run_state"


class UiPathOpenAIAgentRuntime:
    """A runtime class for executing OpenAI Agents within the UiPath framework."""

    def __init__(
        self,
        agent: Agent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        session: SQLiteSession | None = None,
        storage: UiPathRuntimeStorageProtocol | None = None,
    ):
        """Initialize the runtime.

        Args:
            agent: The OpenAI Agent to execute
            runtime_id: Unique identifier for this runtime instance
            entrypoint: Optional entrypoint name (for schema generation)
            session: Optional OpenAI Agents SDK session for persistent memory
            storage: Optional storage for persisting RunState across process restarts
        """
        self.agent: Agent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self._session = session
        self._storage = storage
        self._context_type: type[BaseModel] | None = get_agent_context_type(agent)

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input and configuration."""
        try:
            result: UiPathRuntimeResult | None = None
            async for event in self._run_agent(input, options, stream_events=False):
                if isinstance(event, UiPathRuntimeResult):
                    result = event

            if result is None:
                raise RuntimeError("Agent completed without returning a result")

            return result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events in real-time."""
        try:
            async for event in self._run_agent(input, options, stream_events=True):
                yield event
        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def _run_agent(
        self,
        input: dict[str, Any] | None,
        options: UiPathExecuteOptions | UiPathStreamOptions | None,
        stream_events: bool,
    ) -> AsyncGenerator[UiPathRuntimeEvent | UiPathRuntimeResult, None]:
        """Core agent execution logic used by both execute() and stream()."""
        agent_input, context = await self._prepare_agent_input_and_context(
            input, options
        )

        result = Runner.run_streamed(
            starting_agent=self.agent,
            input=agent_input,
            context=context,
            session=self._session,
        )

        async for event in result.stream_events():
            if stream_events:
                runtime_event = self._convert_stream_event_to_runtime_event(event)
                if runtime_event:
                    yield runtime_event

        interruptions = list(result.interruptions or [])
        if interruptions:
            if self._storage is not None:
                await self._save_run_state(result.to_state())

            yield UiPathRuntimeResult(
                status=UiPathRuntimeStatus.SUSPENDED,
                output=self._build_suspend_output(interruptions),
            )
            return

        yield self._create_success_result(result.final_output)

    # ------------------------------------------------------------------
    # Suspend / Resume
    # ------------------------------------------------------------------

    def _build_suspend_output(
        self, interruptions: list[ToolApprovalItem]
    ) -> dict[str, dict[str, Any]]:
        """Build suspend output map from OpenAI SDK interruptions."""
        suspend_output: dict[str, dict[str, Any]] = {}
        for index, interruption in enumerate(interruptions):
            interrupt_id = self._interrupt_id(interruption, index)
            raw_type = getattr(getattr(interruption, "raw_item", None), "type", None)

            if raw_type == "mcp_approval_request":
                payload: dict[str, Any] = {
                    "type": "mcp_approval_request",
                    "approval_request_id": interrupt_id,
                }
            else:
                payload = {"type": "tool_approval_request"}
            suspend_output[interrupt_id] = payload
        return suspend_output

    def _interrupt_id(self, interruption: ToolApprovalItem, index: int) -> str:
        """Return stable interrupt id for an SDK interruption item."""
        # ToolApprovalItem.call_id already falls through to raw_item.id
        call_id = interruption.call_id
        if isinstance(call_id, str) and call_id:
            return call_id
        return f"interrupt-{index + 1}"

    async def _save_run_state(self, run_state: RunState[Any]) -> None:
        """Persist OpenAI RunState JSON to runtime storage for resume."""
        if self._storage is None:
            return

        run_state_json = run_state.to_json(
            context_serializer=self._serialize_context_for_run_state
        )
        await self._storage.set_value(
            self.runtime_id,
            _RUN_STATE_NAMESPACE,
            _RUN_STATE_KEY,
            run_state_json,
        )

    async def _load_run_state(self) -> RunState[Any] | None:
        """Load persisted OpenAI RunState from runtime storage."""
        if self._storage is None:
            return None

        state_json = await self._storage.get_value(
            self.runtime_id,
            _RUN_STATE_NAMESPACE,
            _RUN_STATE_KEY,
        )
        if not isinstance(state_json, dict):
            return None

        context_deserializer = (
            self._deserialize_context_for_run_state
            if self._context_type is not None
            else None
        )
        maybe_run_state = RunState.from_json(
            initial_agent=self.agent,
            state_json=state_json,
            context_deserializer=context_deserializer,
        )
        if inspect.isawaitable(maybe_run_state):
            return await maybe_run_state
        return maybe_run_state

    def _apply_resume_decisions(
        self,
        run_state: RunState[Any],
        resume_input: dict[str, Any],
    ) -> None:
        """Apply UiPath resume decisions to OpenAI RunState interruptions."""
        interruptions = run_state.get_interruptions()
        if not interruptions:
            return

        interruption_by_id: dict[str, ToolApprovalItem] = {
            self._interrupt_id(interruption, index): interruption
            for index, interruption in enumerate(interruptions)
        }

        for interrupt_id, resume_value in resume_input.items():
            if interrupt_id == "messages":
                continue

            interruption = interruption_by_id.get(interrupt_id)
            if interruption is None:
                continue

            if self._is_approved(resume_value):
                run_state.approve(interruption)
            else:
                run_state.reject(interruption)

    @staticmethod
    def _is_approved(value: Any) -> bool:
        """Extract approval decision from a resume value.

        Accepts a plain bool or a dict with an "approve" key
        (e.g. MCP approval response payloads).
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_unset=True)
        if isinstance(value, dict):
            return bool(value.get("approve", False))
        return False

    # ------------------------------------------------------------------
    # Context serialization for RunState
    # ------------------------------------------------------------------

    def _serialize_context_for_run_state(self, context: Any) -> dict[str, Any]:
        """Serialize run context into a JSON-compatible mapping for RunState."""
        if context is None:
            return {}
        if isinstance(context, dict):
            return context
        if isinstance(context, BaseModel):
            return context.model_dump()
        try:
            result = json.loads(serialize_json(context))
            return result if isinstance(result, dict) else {"value": result}
        except Exception:
            return {"value": str(context)}

    def _deserialize_context_for_run_state(self, context: Mapping[str, Any]) -> Any:
        """Restore context object from RunState serialized payload."""
        context_dict = dict(context)
        if self._context_type is None:
            return context_dict
        try:
            return self._context_type.model_validate(context_dict)
        except Exception:
            return context_dict

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    async def _prepare_agent_input_and_context(
        self,
        input: dict[str, Any] | None,
        options: UiPathExecuteOptions | UiPathStreamOptions | None = None,
    ) -> tuple[str | list[Any] | RunState[Any], Any | None]:
        """Prepare agent input and context from UiPath input dictionary.

        On resume: loads persisted RunState, applies approval decisions, returns it.
        Otherwise: extracts messages and optional context from input dict.
        """
        if options and options.resume:
            run_state = await self._load_run_state()
            if run_state is None:
                raise RuntimeError(
                    f"Resume requested but no persisted run state found "
                    f"for runtime_id={self.runtime_id}"
                )
            if input:
                self._apply_resume_decisions(run_state, input)
            return run_state, None

        if not input:
            return "", None

        messages = input.get("messages", "")
        if not isinstance(messages, (str, list)):
            messages = ""

        context = None
        if self._context_type is not None:
            try:
                context = parse_input_to_context(input, self._context_type)
            except ValueError:
                pass

        return messages, context

    # ------------------------------------------------------------------
    # Event conversion
    # ------------------------------------------------------------------

    def _convert_stream_event_to_runtime_event(
        self,
        event: Any,
    ) -> UiPathRuntimeEvent | None:
        """Convert OpenAI streaming event to UiPath runtime event."""
        event_type = getattr(event, "type", None)
        event_name = getattr(event, "name", None)

        if event_type == "run_item_stream_event":
            event_item = getattr(event, "item", None)
            if event_item:
                if event_name in ["message_output_created", "reasoning_item_created"]:
                    return UiPathRuntimeMessageEvent(
                        payload=json.loads(serialize_json(event_item)),
                        metadata={"event_name": event_name},
                    )
                return UiPathRuntimeStateEvent(
                    payload=json.loads(serialize_json(event_item)),
                    metadata={"event_name": event_name},
                )

        if event_type == "agent_updated_stream_event":
            new_agent = getattr(event, "new_agent", None)
            if new_agent:
                return UiPathRuntimeStateEvent(
                    payload={"agent_name": getattr(new_agent, "name", "unknown")},
                    metadata={"event_type": "agent_updated"},
                )

        return None

    # ------------------------------------------------------------------
    # Result / error helpers
    # ------------------------------------------------------------------

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        serialized_output = json.loads(serialize_json(output))
        if not isinstance(serialized_output, dict):
            serialized_output = {"result": serialized_output}

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_runtime_error(self, e: Exception) -> UiPathOpenAIAgentsRuntimeError:
        """Handle execution errors and create appropriate runtime error."""
        if isinstance(e, UiPathOpenAIAgentsRuntimeError):
            return e

        detail = f"Error: {str(e)}"

        if isinstance(e, json.JSONDecodeError):
            return UiPathOpenAIAgentsRuntimeError(
                UiPathErrorCode.INPUT_INVALID_JSON,
                "Invalid JSON input",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, TimeoutError):
            return UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.TIMEOUT_ERROR,
                "Agent execution timed out",
                detail,
                UiPathErrorCategory.USER,
            )

        return UiPathOpenAIAgentsRuntimeError(
            UiPathOpenAIAgentsErrorCode.AGENT_EXECUTION_FAILURE,
            "Agent execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this OpenAI Agent runtime."""
        entrypoints_schema = get_entrypoints_schema(self.agent)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint,
            uniqueId=str(uuid4()),
            type="agent",
            input=entrypoints_schema.get("input", {}),
            output=entrypoints_schema.get("output", {}),
            graph=get_agent_schema(self.agent),
        )

    async def dispose(self) -> None:
        """Cleanup runtime resources."""
        if self._session is not None:
            self._session.close()
