"""Runtime class for executing PydanticAI Agents within the UiPath framework."""

import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.messages import ToolReturnPart
from uipath.core.chat.content import (
    UiPathConversationContentPartChunkEvent,
    UiPathConversationContentPartEndEvent,
    UiPathConversationContentPartEvent,
    UiPathConversationContentPartStartEvent,
)
from uipath.core.chat.message import (
    UiPathConversationMessageEndEvent,
    UiPathConversationMessageEvent,
    UiPathConversationMessageStartEvent,
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

from .errors import UiPathPydanticAIErrorCode, UiPathPydanticAIRuntimeError
from .schema import (
    get_agent_schema,
    get_deps_type,
    get_entrypoints_schema,
    parse_input_to_deps,
)


class UiPathPydanticAIRuntime:
    """A runtime class for executing PydanticAI Agents within the UiPath framework."""

    def __init__(
        self,
        agent: Agent[Any, Any],
        runtime_id: str | None = None,
        entrypoint: str | None = None,
    ):
        self.agent: Agent[Any, Any] = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self._deps_type = get_deps_type(agent)

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent with the provided input."""
        try:
            user_prompt, deps = self._prepare_input(input)
            result = await self.agent.run(user_prompt, deps=deps)
            return self._create_success_result(result.output)
        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events in real-time."""
        from pydantic_graph import End

        try:
            user_prompt, deps = self._prepare_input(input)
            agent_name = self.agent.name or "agent"
            tools_node_name = f"{agent_name}_tools"
            has_tools = any(
                ts.tools
                for ts in self.agent.toolsets
                if isinstance(ts, FunctionToolset)
            )

            async with self.agent.iter(user_prompt, deps=deps) as agent_run:
                node = agent_run.next_node

                while not isinstance(node, End):
                    if Agent.is_model_request_node(node):
                        yield UiPathRuntimeStateEvent(
                            payload=self._model_request_payload(node),
                            node_name=agent_name,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                        model_node = node
                        message_id = str(uuid4())
                        content_part_id = f"chunk-{message_id}-0"
                        has_text = False

                        async with model_node.stream(agent_run.ctx) as stream:
                            async for text_chunk in stream.stream_text(
                                delta=True, debounce_by=None
                            ):
                                if not has_text:
                                    has_text = True
                                    yield UiPathRuntimeMessageEvent(
                                        payload=UiPathConversationMessageEvent(
                                            message_id=message_id,
                                            start=UiPathConversationMessageStartEvent(
                                                role="assistant",
                                                timestamp=self._get_timestamp(),
                                            ),
                                            content_part=UiPathConversationContentPartEvent(
                                                content_part_id=content_part_id,
                                                start=UiPathConversationContentPartStartEvent(
                                                    mime_type="text/plain",
                                                ),
                                            ),
                                        ),
                                    )

                                yield UiPathRuntimeMessageEvent(
                                    payload=UiPathConversationMessageEvent(
                                        message_id=message_id,
                                        content_part=UiPathConversationContentPartEvent(
                                            content_part_id=content_part_id,
                                            chunk=UiPathConversationContentPartChunkEvent(
                                                data=text_chunk,
                                            ),
                                        ),
                                    ),
                                )

                        next_node = await agent_run.next(model_node)

                        if has_text:
                            yield UiPathRuntimeMessageEvent(
                                payload=UiPathConversationMessageEvent(
                                    message_id=message_id,
                                    end=UiPathConversationMessageEndEvent(),
                                    content_part=UiPathConversationContentPartEvent(
                                        content_part_id=content_part_id,
                                        end=UiPathConversationContentPartEndEvent(),
                                    ),
                                ),
                            )

                        if Agent.is_call_tools_node(next_node):
                            yield UiPathRuntimeStateEvent(
                                payload=self._model_response_payload(next_node),
                                node_name=agent_name,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                        node = next_node

                    elif Agent.is_call_tools_node(node):
                        tool_calls = node.model_response.tool_calls if has_tools else []

                        if tool_calls:
                            yield UiPathRuntimeStateEvent(
                                payload={
                                    "tool_calls": [
                                        {"tool_name": tc.tool_name} for tc in tool_calls
                                    ],
                                },
                                node_name=tools_node_name,
                                phase=UiPathRuntimeStatePhase.STARTED,
                            )

                        next_node = await agent_run.next(node)

                        if tool_calls and Agent.is_model_request_node(next_node):
                            yield UiPathRuntimeStateEvent(
                                payload=self._tool_results_payload(next_node),
                                node_name=tools_node_name,
                                phase=UiPathRuntimeStatePhase.COMPLETED,
                            )
                        node = next_node

                    else:
                        node = await agent_run.next(node)

                assert agent_run.result is not None
                final_output = agent_run.result.output

            yield self._create_success_result(final_output)

        except Exception as e:
            raise self._create_runtime_error(e) from e

    @staticmethod
    def _get_timestamp() -> str:
        """Get current UTC timestamp in ISO 8601 format."""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    @staticmethod
    def _model_request_payload(node: Any) -> dict[str, Any]:
        """Build payload for a ModelRequestNode STARTED event."""
        payload: dict[str, Any] = {}
        try:
            parts = node.request.parts if node.request else []
            prompt_parts = [
                p.content
                for p in parts
                if hasattr(p, "content") and isinstance(p.content, str)
            ]
            if prompt_parts:
                payload["prompt"] = prompt_parts[-1]
        except Exception:
            pass
        return payload

    @staticmethod
    def _model_response_payload(next_node: Any) -> dict[str, Any]:
        """Build payload for a ModelRequestNode COMPLETED event.

        After agent_run.next() the returned node is the CallToolsNode
        which carries the model_response with usage data.
        """
        payload: dict[str, Any] = {}
        try:
            response = next_node.model_response
            if response.model_name:
                payload["model_name"] = response.model_name
            usage = response.usage
            if usage:
                payload["usage"] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
        except Exception:
            pass
        return payload

    @staticmethod
    def _tool_results_payload(next_node: Any) -> dict[str, Any]:
        """Build payload for a CallToolsNode COMPLETED event.

        After agent_run.next() the returned node is a ModelRequestNode
        whose request.parts contain ToolReturnPart objects with results.
        """
        payload: dict[str, Any] = {}
        try:
            parts = next_node.request.parts if next_node.request else []
            results = []
            for p in parts:
                if not isinstance(p, ToolReturnPart):
                    continue
                result: dict[str, Any] = {"tool_name": p.tool_name}
                content = p.content
                if isinstance(content, (str, dict, list)):
                    result["content"] = content
                else:
                    result["content"] = str(content)
                results.append(result)
            if results:
                payload["tool_results"] = results
        except Exception:
            pass
        return payload

    def _prepare_input(
        self, input: dict[str, Any] | None
    ) -> tuple[str, BaseModel | None]:
        """Prepare user prompt and deps from UiPath input dictionary.

        Two modes:
        - Structured (deps_type is a Pydantic model): entire input is parsed
          as deps. 'messages' field, if present in the model, is extracted
          as the user prompt.
        - Conversational (no deps_type): 'messages' field is the user prompt.

        Returns:
            (user_prompt, deps) tuple
        """
        if not input:
            return "", None

        if self._deps_type is not None:
            deps = parse_input_to_deps(input, self._deps_type)
            # If the deps model happens to have a 'messages' field, use it as prompt
            user_prompt = ""
            if hasattr(deps, "messages"):
                messages_val = deps.messages
                if isinstance(messages_val, str):
                    user_prompt = messages_val
            return user_prompt, deps

        # Conversational fallback: extract messages as prompt
        return self._extract_messages(input), None

    def _extract_messages(self, input: dict[str, Any]) -> str:
        """Extract a user prompt string from the 'messages' field.

        Expects UiPath conversation format:
        [{"role": "user", "contentParts": [{"data": {"inline": "..."}}]}]
        """
        messages = input.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""

        # Extract text from the last user message
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role and role != "user":
                continue
            text = self._extract_text_from_content_parts(msg)
            if text:
                return text

        # Fallback: extract from last message regardless of role
        if isinstance(messages[-1], dict):
            text = self._extract_text_from_content_parts(messages[-1])
            if text:
                return text

        return ""

    @staticmethod
    def _extract_text_from_content_parts(msg: dict[str, Any]) -> str:
        """Extract text from a message's contentParts in UiPath conversation format."""
        content_parts = msg.get("contentParts")
        if not isinstance(content_parts, list):
            return ""

        texts: list[str] = []
        for cp in content_parts:
            if not isinstance(cp, dict):
                continue
            data = cp.get("data")
            if isinstance(data, dict) and "inline" in data:
                inline = data["inline"]
                if isinstance(inline, str) and inline:
                    texts.append(inline)
        return "".join(texts)

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        if self.agent.output_type is str and isinstance(output, str):
            serialized_output = self._wrap_as_conversation_message(output)
        else:
            serialized_output = json.loads(serialize_json(output))
            if not isinstance(serialized_output, dict):
                serialized_output = {"result": serialized_output}

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    @staticmethod
    def _wrap_as_conversation_message(text: str) -> dict[str, Any]:
        """Wrap a plain string as a UiPath conversation message output."""
        return {
            "messages": [
                {
                    "role": "assistant",
                    "contentParts": [{"data": {"inline": text}}],
                }
            ]
        }

    def _create_runtime_error(self, e: Exception) -> UiPathPydanticAIRuntimeError:
        """Map exceptions to appropriate runtime errors."""
        if isinstance(e, UiPathPydanticAIRuntimeError):
            return e

        detail = f"Error: {str(e)}"

        if isinstance(e, json.JSONDecodeError):
            return UiPathPydanticAIRuntimeError(
                UiPathErrorCode.INPUT_INVALID_JSON,
                "Invalid JSON input",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, TimeoutError):
            return UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_TIMEOUT,
                "Agent execution timed out",
                detail,
                UiPathErrorCategory.USER,
            )

        return UiPathPydanticAIRuntimeError(
            UiPathPydanticAIErrorCode.AGENT_EXECUTION_ERROR,
            "Agent execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this PydanticAI Agent runtime."""
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
