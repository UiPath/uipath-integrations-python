"""Runtime class for executing OpenAI Agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from agents import Agent, Runner
from pydantic import BaseModel
from uipath.core.serialization import serialize_defaults
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
)
from uipath.runtime.schema import UiPathRuntimeSchema

from .context import get_agent_context_type, parse_input_to_context
from .errors import UiPathOpenAIAgentsErrorCode, UiPathOpenAIAgentsRuntimeError
from .schema import get_agent_schema, get_entrypoints_schema


class UiPathOpenAIAgentRuntime:
    """
    A runtime class for executing OpenAI Agents within the UiPath framework.
    """

    def __init__(
        self,
        agent: Agent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        """
        Initialize the runtime.

        Args:
            agent: The OpenAI Agent to execute
            runtime_id: Unique identifier for this runtime instance
            entrypoint: Optional entrypoint name (for schema generation)
        """
        self.agent: Agent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint

        # Detect context type from agent's generic parameter
        self._context_type: type[BaseModel] | None = get_agent_context_type(agent)

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """
        Execute the agent with the provided input and configuration.

        Args:
            input: Input dictionary containing the message for the agent
            options: Execution options

        Returns:
            UiPathRuntimeResult with the agent's output

        Raises:
            UiPathOpenAIAgentRuntimeError: If execution fails
        """
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
        """
        Stream agent execution events in real-time.

        Args:
            input: Input dictionary containing the message for the agent
            options: Stream options

        Yields:
            UiPathRuntimeEvent instances during execution,
            then the final UiPathRuntimeResult

        Raises:
            UiPathOpenAIAgentRuntimeError: If execution fails
        """
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
        """
        Core agent execution logic used by both execute() and stream().

        Args:
            input: Input dictionary
            options: Execution/stream options
            stream_events: Whether to stream events during execution

        Yields:
            Runtime events if stream_events=True, then final result
        """
        # Prepare input and context
        agent_input, context = self._prepare_agent_input_and_context(input)

        # Run the agent with streaming if events requested
        if stream_events:
            # Use streaming for events
            async for event_or_result in self._run_agent_streamed(
                agent_input, context, options, stream_events
            ):
                yield event_or_result
        else:
            # Use non-streaming for simple execution
            result = await Runner.run(
                starting_agent=self.agent,
                input=agent_input,
                context=context,
            )
            yield self._create_success_result(result.final_output)

    async def _run_agent_streamed(
        self,
        agent_input: str | list[Any],
        context: Any | None,
        options: UiPathExecuteOptions | UiPathStreamOptions | None,
        stream_events: bool,
    ) -> AsyncGenerator[UiPathRuntimeEvent | UiPathRuntimeResult, None]:
        """
        Run agent using streaming API to enable event streaming.

        Args:
            agent_input: Prepared agent input (string or list of messages)
            context: Optional context object (Pydantic model instance)
            options: Execution/stream options
            stream_events: Whether to yield streaming events to caller

        Yields:
            Runtime events if stream_events=True, then final result
        """

        # Use Runner.run_streamed() for streaming events (returns RunResultStreaming directly)
        result = Runner.run_streamed(
            starting_agent=self.agent,
            input=agent_input,
            context=context,
        )

        # Stream events from the agent
        async for event in result.stream_events():
            # Emit the event to caller if streaming is enabled
            if stream_events:
                runtime_event = self._convert_stream_event_to_runtime_event(event)
                if runtime_event:
                    yield runtime_event

        # Stream complete - yield final result
        yield self._create_success_result(result.final_output)

    def _convert_stream_event_to_runtime_event(
        self,
        event: Any,
    ) -> UiPathRuntimeEvent | None:
        """
        Convert OpenAI streaming event to UiPath runtime event.

        Args:
            event: Streaming event from Runner.run_streamed()

        Returns:
            UiPathRuntimeEvent or None if event should be filtered
        """

        event_type = getattr(event, "type", None)
        event_name = getattr(event, "name", None)

        # Handle run item events (messages, tool calls, etc.)
        if event_type == "run_item_stream_event":
            event_item = getattr(event, "item", None)
            if event_item:
                # Determine if this is a message or state event
                if event_name in ["message_output_created", "reasoning_item_created"]:
                    return UiPathRuntimeMessageEvent(
                        payload=serialize_defaults(event_item),
                        metadata={"event_name": event_name},
                    )
                else:
                    return UiPathRuntimeStateEvent(
                        payload=serialize_defaults(event_item),
                        metadata={"event_name": event_name},
                    )

        # Handle agent updated events
        if event_type == "agent_updated_stream_event":
            new_agent = getattr(event, "new_agent", None)
            if new_agent:
                return UiPathRuntimeStateEvent(
                    payload={"agent_name": getattr(new_agent, "name", "unknown")},
                    metadata={"event_type": "agent_updated"},
                )

        # Filter out raw response events (too granular)
        return None

    def _prepare_agent_input_and_context(
        self, input: dict[str, Any] | None
    ) -> tuple[str | list[Any], Any | None]:
        """
        Prepare agent input and context from UiPath input dictionary.

        - 'messages' field is always extracted as the LLM input
        - If agent has a context type, remaining fields are parsed into Pydantic model
        """
        if not input:
            return "", None

        # Extract messages (always goes to LLM)
        messages = input.get("messages", "")
        if not isinstance(messages, (str, list)):
            messages = ""

        # If agent has a context type, parse remaining fields into context
        context = None
        if self._context_type is not None:
            try:
                context = parse_input_to_context(input, self._context_type)
            except ValueError:
                pass  # Fallback to no context if parsing fails

        return messages, context

    def _serialize_message(self, message: Any) -> dict[str, Any]:
        """
        Serialize an agent message for event streaming.

        Args:
            message: Message object from the agent

        Returns:
            Dictionary representation of the message
        """
        serialized = serialize_defaults(message)

        # Ensure the result is a dictionary
        if isinstance(serialized, dict):
            return serialized

        # Fallback to wrapping in a content field
        return {"content": serialized}

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """
        Create result for successful completion.

        Args:
            output: The agent's output

        Returns:
            UiPathRuntimeResult with serialized output
        """
        # Serialize output
        serialized_output = serialize_defaults(output)

        # Ensure output is a dictionary
        if not isinstance(serialized_output, dict):
            serialized_output = {"result": serialized_output}

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_runtime_error(self, e: Exception) -> UiPathOpenAIAgentsRuntimeError:
        """
        Handle execution errors and create appropriate runtime error.

        Args:
            e: The exception that occurred

        Returns:
            UiPathOpenAIAgentsRuntimeError with appropriate error code
        """
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
        """
        Get schema for this OpenAI Agent runtime.

        Returns:
            UiPathRuntimeSchema with input/output schemas and graph structure
        """
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
        pass
