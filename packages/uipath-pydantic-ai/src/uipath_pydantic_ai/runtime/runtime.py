"""Runtime class for executing PydanticAI Agents within the UiPath framework."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset
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
        """Stream agent execution events in real-time.

        Uses agent.iter() for node-level iteration, emitting
        UiPathRuntimeStateEvent with node_name and phase (STARTED/COMPLETED)
        to drive graph node highlighting in the UI.

        Node mapping to graph nodes (from schema.py):
        - ModelRequestNode -> agent node (STARTED/COMPLETED)
        - CallToolsNode   -> {agent}_tools node (STARTED/COMPLETED)
        """
        try:
            user_prompt, deps = self._prepare_input(input)
            agent_name = self.agent.name or "agent"
            tools_node = f"{agent_name}_tools"
            has_tools = any(
                ts.tools
                for ts in self.agent.toolsets
                if isinstance(ts, FunctionToolset)
            )

            # Signal agent node STARTED
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.STARTED,
            )

            async with self.agent.iter(user_prompt, deps=deps) as agent_run:
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        # LLM call: highlight agent node
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=agent_name,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                        yield UiPathRuntimeMessageEvent(
                            payload=json.loads(serialize_json(node.request)),
                            metadata={"event_name": "model_request"},
                        )

                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=agent_name,
                            phase=UiPathRuntimeStatePhase.COMPLETED,
                        )

                    elif Agent.is_call_tools_node(node) and has_tools:
                        # Tool execution: highlight tools node
                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=tools_node,
                            phase=UiPathRuntimeStatePhase.STARTED,
                        )

                        yield UiPathRuntimeStateEvent(
                            payload={},
                            node_name=tools_node,
                            phase=UiPathRuntimeStatePhase.COMPLETED,
                        )

                # Capture result inside the async with block
                assert agent_run.result is not None
                final_output = agent_run.result.output

            # Signal agent node COMPLETED
            yield UiPathRuntimeStateEvent(
                payload={},
                node_name=agent_name,
                phase=UiPathRuntimeStatePhase.COMPLETED,
            )

            yield self._create_success_result(final_output)

        except Exception as e:
            raise self._create_runtime_error(e) from e

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
        """Extract a user prompt string from the 'messages' field."""
        messages = input.get("messages", "")
        if not isinstance(messages, (str, list)):
            return ""

        if isinstance(messages, list):
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
            return " ".join(parts)

        return messages

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        serialized_output = json.loads(serialize_json(output))

        if not isinstance(serialized_output, dict):
            serialized_output = {"result": serialized_output}

        return UiPathRuntimeResult(
            output=serialized_output,
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

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
