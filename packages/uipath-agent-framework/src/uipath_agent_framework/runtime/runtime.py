"""Runtime class for executing Agent Framework agents within the UiPath framework."""

import json
import logging
from typing import Any, AsyncGenerator
from uuid import uuid4

from agent_framework import AgentResponse, AgentResponseUpdate, BaseAgent, Content
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
from .schema import get_agent_graph, get_entrypoints_schema

logger = logging.getLogger(__name__)


class UiPathAgentFrameworkRuntime:
    """A runtime class for executing Agent Framework agents within the UiPath framework."""

    def __init__(
        self,
        agent: BaseAgent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        self.agent: BaseAgent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self.chat = AgentFrameworkChatMessagesMapper()

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input and return the result."""
        try:
            user_input = self._prepare_input(input)
            response = await self.agent.run(user_input)
            output = self._extract_output(response)
            return self._create_success_result(output)
        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events in real-time."""
        try:
            user_input = self._prepare_input(input)
            agent_name = self.agent.name or "agent"

            # Emit root agent started
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.STARTED,
            )

            active_tools: str | None = None
            final_text = ""

            response_stream = self.agent.run(user_input, stream=True)
            async for update in response_stream:
                if not isinstance(update, AgentResponseUpdate):
                    continue

                # Process contents from the streaming update
                contents = update.contents or []
                for content in contents:
                    if not isinstance(content, Content):
                        continue

                    # Track tool node state transitions
                    if content.type == "function_call":
                        tools_node = f"{agent_name}_tools"
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
                            "function_name": content.name or "unknown",
                        }
                        if content.arguments:
                            payload["function_args"] = serialize_defaults(
                                content.arguments
                            )
                        yield UiPathRuntimeStateEvent(
                            payload=payload,
                            node_name=tools_node,
                            metadata={"event_type": "function_call"},
                        )

                    elif content.type == "function_result":
                        if active_tools:
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=active_tools,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                            active_tools = None
                            # Re-start agent node after tool completion
                            yield UiPathRuntimeStateEvent(
                                payload={},
                                node_name=agent_name,
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

    def _create_runtime_error(
        self, e: Exception
    ) -> UiPathAgentFrameworkRuntimeError:
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
