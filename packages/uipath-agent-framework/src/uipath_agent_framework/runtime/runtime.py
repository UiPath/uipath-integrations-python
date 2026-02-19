"""Runtime class for executing Agent Framework agents within the UiPath framework."""

import json
import logging
from typing import Any, AsyncGenerator
from uuid import uuid4

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    FunctionTool,
)
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
    UiPathRuntimeStatePhase,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from .errors import UiPathAgentFrameworkErrorCode, UiPathAgentFrameworkRuntimeError
from .messages import AgentFrameworkChatMessagesMapper
from .schema import (
    extract_agent_from_tool,
    get_agent_graph,
    get_agent_tools,
    get_entrypoints_schema,
)
from .storage import SqliteSessionStore

logger = logging.getLogger(__name__)


class UiPathAgentFrameworkRuntime:
    """A runtime class for executing Agent Framework agents within the UiPath framework."""

    def __init__(
        self,
        agent: BaseAgent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        session_store: SqliteSessionStore | None = None,
    ):
        self.agent: BaseAgent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self.chat = AgentFrameworkChatMessagesMapper()
        self._session_store = session_store

    @staticmethod
    def _build_agent_tool_names(agent: BaseAgent) -> set[str]:
        """Build a set of tool names that correspond to agent-as-tools.

        Inspects the agent's tools list and identifies which ones wrap sub-agents
        (created via BaseAgent.as_tool()). Returns the tool names (not agent names)
        so we can match them against function_call content.name during streaming.
        """
        agent_tool_names: set[str] = set()
        for tool in get_agent_tools(agent):
            inner_agent = extract_agent_from_tool(tool)
            if inner_agent is not None and isinstance(tool, FunctionTool):
                agent_tool_names.add(tool.name)
        return agent_tool_names

    @staticmethod
    def _build_tool_name_to_agent(agent: BaseAgent) -> dict[str, str]:
        """Build a mapping from tool name to sub-agent node name.

        When a function_call uses a tool name that wraps a sub-agent, we need
        to know which agent node to emit STARTED/COMPLETED for.
        """
        mapping: dict[str, str] = {}
        for tool in get_agent_tools(agent):
            inner_agent = extract_agent_from_tool(tool)
            if inner_agent is not None and isinstance(tool, FunctionTool):
                agent_name = inner_agent.name or "agent"
                mapping[tool.name] = agent_name
        return mapping

    async def _load_session(self) -> AgentSession:
        """Load or create an AgentSession for this runtime_id.

        If a session store is configured, loads the persisted session state.
        Otherwise creates a fresh session each time.
        """
        if self._session_store:
            session_data = await self._session_store.load_session(self.runtime_id)
            if session_data is not None:
                logger.debug(
                    "Restoring session from store for runtime_id=%s",
                    self.runtime_id,
                )
                return AgentSession.from_dict(session_data)  # type: ignore[attr-defined]

        return self.agent.create_session(session_id=self.runtime_id)  # type: ignore[attr-defined]

    async def _save_session(self, session: AgentSession) -> None:
        """Persist the session state after execution."""
        if self._session_store:
            session_data = session.to_dict()  # type: ignore[attr-defined]
            await self._session_store.save_session(self.runtime_id, session_data)
            logger.debug("Saved session to store for runtime_id=%s", self.runtime_id)

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input and return the result."""
        try:
            user_input = self._prepare_input(input)
            session = await self._load_session()
            response = await self.agent.run(user_input, session=session)  # type: ignore[attr-defined]
            await self._save_session(session)
            output = self._extract_output(response)
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
        uipath-dev graph visualization. Tracks the currently active
        agent/tools nodes and emits proper STARTED/COMPLETED transitions
        when execution moves between agents and tools.

        For multi-agent setups (via as_tool()), sub-agents get their own
        STARTED/COMPLETED events with node IDs matching the graph schema.
        """
        try:
            user_input = self._prepare_input(input)
            session = await self._load_session()
            agent_name = self.agent.name or "agent"

            # Pre-compute which tool names correspond to sub-agents
            agent_tool_names = self._build_agent_tool_names(self.agent)
            tool_name_to_agent = self._build_tool_name_to_agent(self.agent)

            # Emit root agent started
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.STARTED,
            )

            active_agent: str | None = agent_name
            active_tools: str | None = None
            final_text = ""

            response_stream = self.agent.run(user_input, stream=True, session=session)  # type: ignore[attr-defined]
            async for update in response_stream:
                if not isinstance(update, AgentResponseUpdate):
                    continue

                # Process contents from the streaming update
                contents = update.contents or []
                for content in contents:
                    if not isinstance(content, Content):
                        continue

                    # Track tool/agent node state transitions
                    if content.type == "function_call":
                        # During streaming, only the first chunk carries
                        # content.name. Subsequent chunks are partial
                        # argument fragments — skip them.
                        if not content.name:
                            continue

                        call_name = content.name
                        logger.debug(
                            "function_call: name=%s, call_id=%s, "
                            "is_agent_tool=%s, active_agent=%s",
                            call_name,
                            content.call_id,
                            call_name in agent_tool_names,
                            active_agent,
                        )

                        if call_name in agent_tool_names:
                            # This is a call to a sub-agent tool
                            sub_agent_name = tool_name_to_agent.get(
                                call_name, call_name
                            )

                            # Close any active tools node first
                            if active_tools:
                                yield UiPathRuntimeStateEvent(
                                    payload={},
                                    node_name=active_tools,
                                    phase=UiPathRuntimeStatePhase.COMPLETED,
                                )
                                active_tools = None

                            # Emit STARTED for the sub-agent node
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=sub_agent_name,
                                phase=UiPathRuntimeStatePhase.STARTED,
                            )
                            active_agent = sub_agent_name

                            # Emit state event for the agent tool call
                            payload: dict[str, Any] = {
                                "function_name": call_name,
                            }
                            if isinstance(content.arguments, dict):
                                payload["function_args"] = serialize_defaults(
                                    content.arguments
                                )
                            yield UiPathRuntimeStateEvent(
                                payload=payload,
                                node_name=sub_agent_name,
                                metadata={"event_type": "function_call"},
                            )
                        else:
                            # Regular tool call
                            tools_node = f"{active_agent}_tools"
                            if active_tools != tools_node:
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

                            # Emit state event for the tool call
                            payload = {
                                "function_name": call_name,
                            }
                            if isinstance(content.arguments, dict):
                                payload["function_args"] = serialize_defaults(
                                    content.arguments
                                )
                            yield UiPathRuntimeStateEvent(
                                payload=payload,
                                node_name=tools_node,
                                metadata={"event_type": "function_call"},
                            )

                    elif content.type == "function_result":
                        result_name = content.name or ""

                        if result_name in agent_tool_names:
                            # Sub-agent completed — emit COMPLETED, re-START parent
                            sub_agent_name = tool_name_to_agent.get(
                                result_name, result_name
                            )
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=sub_agent_name,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                            active_agent = agent_name
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=agent_name,
                                phase=UiPathRuntimeStatePhase.STARTED,
                            )
                        elif active_tools:
                            # Regular tool completed
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=active_tools,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                            active_tools = None
                            # Re-start agent node after tool completion
                            if active_agent:
                                yield UiPathRuntimeStateEvent(
                                    payload={},
                                    node_name=active_agent,
                                    phase=UiPathRuntimeStatePhase.STARTED,
                                )

                    # Yield conversation message events
                    for msg_event in self.chat.map_streaming_content(content):
                        yield UiPathRuntimeMessageEvent(payload=msg_event)

                # Accumulate text from the update
                if update.text:
                    final_text += update.text

            # Close any remaining active tools node
            if active_tools:
                yield UiPathRuntimeStateEvent(
                    payload={},
                    node_name=active_tools,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )

            # Complete agent node
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )

            # Close any open conversation message
            for msg_event in self.chat.close_message():
                yield UiPathRuntimeMessageEvent(payload=msg_event)

            # Persist session state after streaming completes
            await self._save_session(session)

            # Get final response
            final_response = await response_stream.get_final_response()
            output = self._extract_output(final_response)
            yield self._create_success_result(output)

        except Exception as e:
            raise self._create_runtime_error(e) from e

    def _prepare_input(self, input: dict[str, Any] | None) -> str:
        """Prepare input string from UiPath input dictionary."""
        if not input:
            return ""

        if "messages" in input:
            return self.chat.map_messages_to_input(input["messages"])

        # Fallback: serialize entire input as JSON
        return json.dumps(input)

    def _extract_output(self, response: AgentResponse) -> Any:
        """Extract output from agent response."""
        if response.text:
            return response.text
        return str(response) if response else ""

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        serialized_output = serialize_defaults(output)

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

    def _create_runtime_error(self, e: Exception) -> UiPathAgentFrameworkRuntimeError:
        """Handle execution errors and create appropriate runtime error."""
        if isinstance(e, UiPathAgentFrameworkRuntimeError):
            return e

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
