"""Runtime class for executing Agent Framework agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    FunctionTool,
    Message,
    WorkflowAgent,
)
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

from .errors import UiPathAgentFrameworkErrorCode, UiPathAgentFrameworkRuntimeError
from .messages import AgentFrameworkChatMessagesMapper
from .schema import (
    extract_agent_from_tool,
    get_agent_graph,
    get_agent_tools,
    get_entrypoints_schema,
)
from .storage import SqliteSessionStore


class _StreamState:
    """Mutable state tracker for agent streaming.

    Holds the sub-agent metadata (computed once) and the active node
    state that changes as function_call / function_result events arrive.
    """

    __slots__ = (
        "root_agent",
        "active_agent",
        "active_tools",
        "call_ids",
        "agent_tool_names",
        "tool_name_to_agent",
        "sub_agents_with_tools",
    )

    def __init__(
        self,
        root_agent: str,
        agent_tool_names: set[str],
        tool_name_to_agent: dict[str, str],
        sub_agents_with_tools: set[str],
    ) -> None:
        self.root_agent = root_agent
        self.active_agent: str = root_agent
        self.active_tools: str | None = None
        # call_id → sub-agent name (content.name on function_result
        # may be empty for as_tool() wrappers, so we match by call_id).
        self.call_ids: dict[str, str] = {}
        self.agent_tool_names = agent_tool_names
        self.tool_name_to_agent = tool_name_to_agent
        self.sub_agents_with_tools = sub_agents_with_tools


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

    # ------------------------------------------------------------------
    # Sub-agent introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sub_agent_info(
        agent: BaseAgent,
    ) -> tuple[set[str], dict[str, str], set[str]]:
        """Inspect the agent's tools once to extract all sub-agent metadata.

        Returns:
            agent_tool_names: tool names that wrap sub-agents
            tool_name_to_agent: mapping from tool name → sub-agent node name
            sub_agents_with_tools: sub-agent names that own tools
        """
        agent_tool_names: set[str] = set()
        tool_name_to_agent: dict[str, str] = {}
        sub_agents_with_tools: set[str] = set()

        for tool in get_agent_tools(agent):
            inner_agent = extract_agent_from_tool(tool)
            if inner_agent is None or not isinstance(tool, FunctionTool):
                continue

            agent_tool_names.add(tool.name)
            inner_name = inner_agent.name or "agent"
            tool_name_to_agent[tool.name] = inner_name

            if get_agent_tools(inner_agent):
                sub_agents_with_tools.add(inner_name)

        return agent_tool_names, tool_name_to_agent, sub_agents_with_tools

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _load_session(self) -> AgentSession:
        """Load or create an AgentSession for this runtime_id."""
        if self._session_store:
            session_data = await self._session_store.load_session(self.runtime_id)
            if session_data is not None:
                return AgentSession.from_dict(session_data)  # type: ignore[attr-defined]

        return self.agent.create_session(session_id=self.runtime_id)  # type: ignore[attr-defined]

    async def _save_session(self, session: AgentSession) -> None:
        """Persist the session state after execution."""
        if self._session_store:
            session_data = session.to_dict()  # type: ignore[attr-defined]
            await self._session_store.save_session(self.runtime_id, session_data)

    # ------------------------------------------------------------------
    # Execute (non-streaming)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stream (main entry)
    # ------------------------------------------------------------------

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events in real-time.

        Two streaming paths:
        - WorkflowAgent: raw workflow events (executor_invoked/completed).
        - Regular BaseAgent: function_call/function_result content tracking.
        """
        try:
            user_input = self._prepare_input(input)
            session = await self._load_session()
            agent_name = self.agent.name or "agent"

            if isinstance(self.agent, WorkflowAgent):
                async for event in self._stream_workflow(
                    user_input, session, agent_name
                ):
                    yield event
            else:
                async for event in self._stream_agent(user_input, session, agent_name):
                    yield event

        except Exception as e:
            raise self._create_runtime_error(e) from e

    # ------------------------------------------------------------------
    # Workflow streaming
    # ------------------------------------------------------------------

    async def _stream_workflow(
        self,
        user_input: str,
        session: AgentSession,
        agent_name: str,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream workflow execution with real-time executor lifecycle events."""
        assert isinstance(self.agent, WorkflowAgent)
        workflow = self.agent.workflow

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.STARTED,
        )

        response_stream = workflow.run(message=user_input, stream=True)

        async for event in response_stream:
            if event.type == "executor_invoked":
                yield UiPathRuntimeStateEvent(
                    payload=self._serialize_event_data(event.data),
                    node_name=event.executor_id,
                    phase=UiPathRuntimeStatePhase.STARTED,
                )
            elif event.type == "executor_completed":
                yield UiPathRuntimeStateEvent(
                    payload=self._serialize_event_data(event.data),
                    node_name=event.executor_id,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            elif event.type == "output":
                for msg_event in self._extract_workflow_messages(event.data):
                    yield UiPathRuntimeMessageEvent(payload=msg_event)

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.COMPLETED,
        )

        for msg_event in self.chat.close_message():
            yield UiPathRuntimeMessageEvent(payload=msg_event)

        await self._save_session(session)

        final_result = await response_stream.get_final_response()
        output = self._extract_workflow_output(final_result)
        yield self._create_success_result(output)

    # ------------------------------------------------------------------
    # Agent streaming
    # ------------------------------------------------------------------

    async def _stream_agent(
        self,
        user_input: str,
        session: AgentSession,
        agent_name: str,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream regular BaseAgent execution with tool/sub-agent tracking."""
        state = _StreamState(agent_name, *self._build_sub_agent_info(self.agent))

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.STARTED,
        )

        response_stream = self.agent.run(user_input, stream=True, session=session)  # type: ignore[attr-defined]
        async for update in response_stream:
            if not isinstance(update, AgentResponseUpdate):
                continue

            for content in update.contents or []:
                if not isinstance(content, Content):
                    continue

                for event in self._process_agent_content(state, content):
                    yield event

                for msg in self.chat.map_streaming_content(content):
                    yield UiPathRuntimeMessageEvent(payload=msg)

        # Teardown: close remaining nodes
        if state.active_tools:
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=state.active_tools,
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )

        yield UiPathRuntimeStateEvent(
            payload={},
            node_name=agent_name,
            phase=UiPathRuntimeStatePhase.COMPLETED,
        )

        for msg in self.chat.close_message():
            yield UiPathRuntimeMessageEvent(payload=msg)

        await self._save_session(session)

        final_response = await response_stream.get_final_response()
        yield self._create_success_result(self._extract_output(final_response))

    # ------------------------------------------------------------------
    # Agent content event handlers
    # ------------------------------------------------------------------

    def _process_agent_content(
        self, s: _StreamState, content: Content
    ) -> list[UiPathRuntimeStateEvent]:
        """Dispatch a streaming Content to the appropriate handler."""
        if content.type == "function_call":
            if not content.name:
                return []
            if content.name in s.agent_tool_names:
                return self._on_sub_agent_call(s, content)
            return self._on_tool_call(s, content)

        if content.type == "function_result":
            return self._on_function_result(s, content)

        return []

    def _on_sub_agent_call(
        self, s: _StreamState, content: Content
    ) -> list[UiPathRuntimeStateEvent]:
        """Handle a function_call that invokes a sub-agent via as_tool()."""
        call_name = content.name or ""
        sub_agent = s.tool_name_to_agent.get(call_name, call_name)
        events: list[UiPathRuntimeStateEvent] = []

        if content.call_id:
            s.call_ids[content.call_id] = sub_agent

        # Close any active tools node
        if s.active_tools:
            events.append(
                UiPathRuntimeStateEvent(
                    payload={},
                    node_name=s.active_tools,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            )
            s.active_tools = None

        payload = {"function_name": call_name}

        # Start sub-agent node
        events.append(
            UiPathRuntimeStateEvent(
                payload=payload,
                node_name=sub_agent,
                phase=UiPathRuntimeStatePhase.STARTED,
            )
        )
        s.active_agent = sub_agent

        # Sub-agent's internal tool calls are opaque in the as_tool()
        # stream — emit a synthetic STARTED on its tools node.
        if sub_agent in s.sub_agents_with_tools:
            tools_node = f"{sub_agent}_tools"
            events.append(
                UiPathRuntimeStateEvent(
                    payload=payload,
                    node_name=tools_node,
                    phase=UiPathRuntimeStatePhase.STARTED,
                )
            )
            s.active_tools = tools_node
        else:
            events.append(
                UiPathRuntimeStateEvent(
                    payload=payload,
                    node_name=sub_agent,
                    metadata={"event_type": "function_call"},
                )
            )

        return events

    def _on_tool_call(
        self, s: _StreamState, content: Content
    ) -> list[UiPathRuntimeStateEvent]:
        """Handle a regular (non-agent) function_call."""
        call_name = content.name or ""
        tools_node = f"{s.active_agent}_tools"
        events: list[UiPathRuntimeStateEvent] = []

        if s.active_tools != tools_node:
            if s.active_tools:
                events.append(
                    UiPathRuntimeStateEvent(
                        payload={},
                        node_name=s.active_tools,
                        phase=UiPathRuntimeStatePhase.COMPLETED,
                    )
                )
            s.active_tools = tools_node
            events.append(
                UiPathRuntimeStateEvent(
                    payload={},
                    node_name=tools_node,
                    phase=UiPathRuntimeStatePhase.STARTED,
                )
            )

        events.append(
            UiPathRuntimeStateEvent(
                payload={"function_name": call_name},
                node_name=tools_node,
                metadata={"event_type": "function_call"},
            )
        )
        return events

    def _on_function_result(
        self, s: _StreamState, content: Content
    ) -> list[UiPathRuntimeStateEvent]:
        """Handle a function_result for either a sub-agent or regular tool."""
        call_id = content.call_id or ""
        result_name = content.name or ""
        events: list[UiPathRuntimeStateEvent] = []

        # Match sub-agent by call_id first (reliable), fall back to name
        matched = s.call_ids.pop(call_id, None)
        if matched is None and result_name in s.agent_tool_names:
            matched = s.tool_name_to_agent.get(result_name, result_name)

        result_payload = self._build_result_payload(content)

        if matched:
            # Sub-agent completed — close tools, then agent, re-start root
            if s.active_tools and s.active_tools == f"{matched}_tools":
                events.append(
                    UiPathRuntimeStateEvent(
                        payload=result_payload,
                        node_name=s.active_tools,
                        phase=UiPathRuntimeStatePhase.COMPLETED,
                    )
                )
                s.active_tools = None

            events.append(
                UiPathRuntimeStateEvent(
                    payload=result_payload,
                    node_name=matched,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            )
            s.active_agent = s.root_agent
            events.append(
                UiPathRuntimeStateEvent(
                    payload={},
                    node_name=s.root_agent,
                    phase=UiPathRuntimeStatePhase.STARTED,
                )
            )
        elif s.active_tools:
            # Regular tool completed
            events.append(
                UiPathRuntimeStateEvent(
                    payload=result_payload,
                    node_name=s.active_tools,
                    phase=UiPathRuntimeStatePhase.COMPLETED,
                )
            )
            s.active_tools = None
            if s.active_agent:
                events.append(
                    UiPathRuntimeStateEvent(
                        payload={},
                        node_name=s.active_agent,
                        phase=UiPathRuntimeStatePhase.STARTED,
                    )
                )

        return events

    # ------------------------------------------------------------------
    # Payload / serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result_payload(content: Content) -> dict[str, Any]:
        """Build a payload dict from a function_result Content."""
        payload: dict[str, Any] = {}
        if content.name:
            payload["function_name"] = content.name
        if content.result is not None:
            try:
                payload["function_response"] = json.loads(
                    serialize_json(content.result)
                )
            except Exception:
                payload["function_response"] = str(content.result)
        return payload

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

    # ------------------------------------------------------------------
    # Workflow message / output extraction
    # ------------------------------------------------------------------

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

    def _extract_workflow_output(self, result: Any) -> Any:
        """Extract output from WorkflowRunResult."""
        outputs: list[Any] = []
        if hasattr(result, "get_outputs"):
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

    # ------------------------------------------------------------------
    # Input / output / result helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, input: dict[str, Any] | None) -> str:
        """Prepare input string from UiPath input dictionary."""
        if not input:
            return ""

        if "messages" in input:
            return self.chat.map_messages_to_input(input["messages"])

        return json.dumps(input)

    def _extract_output(self, response: AgentResponse) -> Any:
        """Extract output from agent response."""
        if response.text:
            return response.text
        return str(response) if response else ""

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

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

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
