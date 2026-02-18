"""Runtime class for executing Google ADK agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types
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
    UiPathRuntimeStateEvent,
    UiPathRuntimeStatePhase,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from .errors import UiPathGoogleADKErrorCode, UiPathGoogleADKRuntimeError
from .schema import (
    get_agent_graph,
    get_entrypoints_schema,
    resolve_output_key,
    resolve_output_schema,
)

_TRANSFER_FN = "transfer_to_agent"


class UiPathGoogleADKRuntime:
    """
    A runtime class for executing Google ADK agents within the UiPath framework.
    """

    APP_NAME = "uipath"
    USER_ID = "uipath-user"

    def __init__(
        self,
        agent: BaseAgent,
        runner: InMemoryRunner,
        session: Session,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        self.agent: BaseAgent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self._runner: InMemoryRunner = runner
        self._session: Session = session

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input and return the result."""
        try:
            user_message = self._prepare_user_message(input)

            final_text = None
            async for event in self._runner.run_async(
                user_id=self.USER_ID,
                session_id=self._session.id,
                new_message=user_message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_text = part.text
                            break

            output = self._extract_output(final_text)
            return self._create_success_result(output)

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events in real-time.

        Emits lifecycle phase events (STARTED/COMPLETED) to drive the
        uipath-dev graph visualization. The frontend's setActiveNode is
        only triggered on phase="started", so we must explicitly emit
        STARTED every time the active agent changes (not just once).

        Tracks the currently active agent/tools nodes and emits proper
        STARTED/COMPLETED transitions when execution moves between agents.
        This ensures correct highlighting even when agents are revisited
        (e.g., coordinator → research_agent → coordinator → code_agent).

        Composite agents (SequentialAgent, ParallelAgent, LoopAgent) are
        transparent in Google ADK — they yield sub-agent events directly
        without emitting events with their own name as author. We synthesize
        STARTED/COMPLETED events for the root agent and __start__/__end__
        graph nodes so the full graph lights up during execution.
        """
        try:
            user_message = self._prepare_user_message(input)

            root_name = self.agent.name

            # __start__ and root agent kick off the graph visualization
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name="__start__",
                phase=UiPathRuntimeStatePhase.STARTED,
            )
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=root_name,
                phase=UiPathRuntimeStatePhase.STARTED,
            )

            final_text = None
            active_agent: str | None = None
            active_tools: str | None = None

            async for event in self._runner.run_async(
                user_id=self.USER_ID,
                session_id=self._session.id,
                new_message=user_message,
            ):
                author = event.author

                if author and author != "user":
                    # Agent changed — close out previous, start new
                    if active_agent != author:
                        if active_tools:
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=active_tools,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                            active_tools = None
                        if active_agent:
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=active_agent,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                        active_agent = author
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=author,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                    # Tools node: STARTED on real function calls
                    # (skip transfer_to_agent — it's agent delegation, not a tool)
                    tools_node = f"{author}_tools"
                    real_calls = [
                        fc
                        for fc in event.get_function_calls()
                        if fc.name != _TRANSFER_FN
                    ]
                    if real_calls and active_tools != tools_node:
                        if active_tools:
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=active_tools,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                        active_tools = tools_node
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=tools_node,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                    # Tools node: COMPLETED on real function responses,
                    # then re-STARTED for the agent to shift highlight back
                    real_responses = [
                        fr
                        for fr in event.get_function_responses()
                        if fr.name != _TRANSFER_FN
                    ]
                    if real_responses and active_tools:
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=active_tools,
                            phase=UiPathRuntimeStatePhase.COMPLETED,
                        )
                        active_tools = None
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=author,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                # Yield the regular UPDATED events (payload data)
                for runtime_event in self._convert_event(event):
                    yield runtime_event

                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_text = part.text
                            break

            # Close out any remaining active nodes
            if active_tools:
                yield UiPathRuntimeStateEvent(
                    payload={},
                    node_name=active_tools,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            if active_agent:
                yield UiPathRuntimeStateEvent(
                    payload={},
                    node_name=active_agent,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )

            # Root agent and __end__ close out the graph visualization
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=root_name,
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name="__end__",
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )

            output = self._extract_output(final_text)
            yield self._create_success_result(output)

        except Exception as e:
            raise self._create_runtime_error(e) from e

    def _extract_output(self, final_text: str | None) -> Any:
        """Extract structured output from the agent execution.

        Uses recursive resolution for composite agents (SequentialAgent, etc.):
        - resolve_output_key: checks last sub_agent's output_key
        - resolve_output_schema: checks last sub_agent's output_schema

        Priority:
        1. output_key → read from session.state (ADK stores parsed output
           there; if output_schema is also set, the value is already a dict)
        2. output_schema (without output_key) → parse the final JSON text
           ourselves, same as ADK does internally
        3. Raw text fallback
        """
        # output_key: ADK stores the result in session.state[output_key].
        # If output_schema is also set, the stored value is already parsed
        # via output_schema.model_validate_json().model_dump().
        output_key = resolve_output_key(self.agent)
        if output_key:
            value = self._session.state.get(output_key)
            if value is not None:
                return value

        # output_schema without output_key: the LLM was constrained to
        # produce JSON matching the schema (via response_mime_type), but
        # ADK didn't store it anywhere. Parse the raw text ourselves.
        # Note: resolve_output_schema returns None if the agent has
        # tools/sub_agents (invalid combo in Gemini API).
        output_schema = resolve_output_schema(self.agent)
        if output_schema and final_text:
            try:
                return output_schema.model_validate_json(final_text).model_dump(
                    exclude_none=True
                )
            except Exception:
                pass

        return final_text or ""

    def _prepare_user_message(self, input: dict[str, Any] | None) -> types.Content:
        """Prepare user message from UiPath input dictionary.

        If input contains 'messages', uses that as text (conversational).
        If input has no 'messages' key, serializes the entire input as JSON
        (for agents with typed input_schema).
        """
        if not input:
            text = ""
        elif "messages" in input:
            messages = input["messages"]
            if isinstance(messages, str):
                text = messages
            elif isinstance(messages, list):
                parts = []
                for msg in messages:
                    if isinstance(msg, str):
                        parts.append(msg)
                    elif isinstance(msg, dict):
                        parts.append(msg.get("content", str(msg)))
                    else:
                        parts.append(str(msg))
                text = "\n".join(parts)
            else:
                text = str(messages)
        else:
            # Typed input: serialize the entire input dict as JSON
            text = json.dumps(input)

        return types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

    def _convert_event(self, event: Event) -> list[UiPathRuntimeStateEvent]:
        """Convert a Google ADK Event to UiPath runtime state events.

        Uses event.author as node_name for agent nodes.
        Tool call/response events emit for BOTH the agent node and the tools node.
        """
        author = event.author

        if author == "user":
            return []

        tools_node = f"{author}_tools"
        events: list[UiPathRuntimeStateEvent] = []

        # Function calls — split transfers from real tool calls
        all_calls = event.get_function_calls()
        if all_calls:
            for fc in all_calls:
                is_transfer = fc.name == _TRANSFER_FN
                payload = {
                    "function_name": fc.name or "unknown",
                    "function_args": serialize_defaults(fc.args or {}),
                }
                event_type = "agent_transfer" if is_transfer else "function_call"
                # Agent node always gets the event
                events.append(
                    UiPathRuntimeStateEvent(
                        payload=payload,
                        node_name=author,
                        metadata={"event_type": event_type},
                    )
                )
                # Tools node only for real tool calls (not transfers)
                if not is_transfer:
                    events.append(
                        UiPathRuntimeStateEvent(
                            payload=payload,
                            node_name=tools_node,
                            metadata={"event_type": "function_call"},
                        )
                    )
            return events

        # Function responses → agent node only (the agent receives the result;
        # tools node was already COMPLETED by the stream() lifecycle tracker)
        all_responses = event.get_function_responses()
        if all_responses:
            for fr in all_responses:
                if fr.name == _TRANSFER_FN:
                    continue
                payload = {
                    "function_name": fr.name or "unknown",
                    "function_response": serialize_defaults(fr.response or {}),
                }
                events.append(
                    UiPathRuntimeStateEvent(
                        payload=payload,
                        node_name=author,
                        metadata={"event_type": "function_response"},
                    )
                )
            return events

        # Agent transfer (from actions, not function calls)
        if event.actions.transfer_to_agent:
            events.append(
                UiPathRuntimeStateEvent(
                    payload={"transfer_to_agent": event.actions.transfer_to_agent},
                    node_name=author,
                    metadata={"event_type": "agent_transfer"},
                )
            )
            return events

        # State delta
        if event.actions.state_delta:
            events.append(
                UiPathRuntimeStateEvent(
                    payload=serialize_defaults(event.actions.state_delta),
                    node_name=author,
                    metadata={"event_type": "state_delta"},
                )
            )
            return events

        # Partial streaming content
        if event.partial and event.content:
            events.append(
                UiPathRuntimeStateEvent(
                    payload=serialize_defaults(event.content),
                    node_name=author,
                    metadata={"event_type": "partial"},
                )
            )
            return events

        return events

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        serialized_output = serialize_defaults(output)

        if not isinstance(serialized_output, dict):
            serialized_output = {"result": serialized_output}

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_runtime_error(self, e: Exception) -> UiPathGoogleADKRuntimeError:
        """Handle execution errors and create appropriate runtime error."""
        if isinstance(e, UiPathGoogleADKRuntimeError):
            return e

        detail = f"Error: {str(e)}"

        if isinstance(e, json.JSONDecodeError):
            return UiPathGoogleADKRuntimeError(
                UiPathErrorCode.INPUT_INVALID_JSON,
                "Invalid JSON input",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, TimeoutError):
            return UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.TIMEOUT_ERROR,
                "Agent execution timed out",
                detail,
                UiPathErrorCategory.USER,
            )

        return UiPathGoogleADKRuntimeError(
            UiPathGoogleADKErrorCode.AGENT_EXECUTION_FAILURE,
            "Agent execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this Google ADK Agent runtime."""
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
