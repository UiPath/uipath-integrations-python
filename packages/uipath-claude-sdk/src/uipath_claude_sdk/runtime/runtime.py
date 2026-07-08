"""Runtime class for executing Claude Agent SDK agents within the UiPath framework."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from pydantic import ValidationError
from uipath.core.serialization import serialize_json
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath.runtime.events import UiPathRuntimeEvent, UiPathRuntimeStateEvent
from uipath.runtime.schema import UiPathRuntimeSchema

from ..agent import ClaudeAgent
from .errors import UiPathClaudeSDKErrorCode, UiPathClaudeSDKRuntimeError
from .gateway_proxy import GatewayProxy
from .instrumentor import ClaudeSdkInstrumentor, ClaudeSdkSpanFactory
from .schema import get_agent_graph, get_entrypoints_schema

logger = logging.getLogger(__name__)

# The Claude SDK requires ANTHROPIC_API_KEY to be set; when routing through the
# gateway proxy, auth is handled by the proxy — the value is arbitrary.
_GATEWAY_PROXY_API_KEY = "gateway-proxy"


class UiPathClaudeSDKRuntime:
    """A runtime class for executing Claude Agent SDK agents within the UiPath framework.

    Owns the ``ClaudeSDKClient`` run loop and maps SDK messages to UiPath
    runtime events. LLM access defaults to the UiPath LLM Gateway via a local
    proxy; setting ``ANTHROPIC_API_KEY`` bypasses the proxy (BYO key).

    Args:
        agent: The loaded ClaudeAgent definition.
        runtime_id: Unique identifier for this runtime instance.
        entrypoint: Agent entrypoint name (for schema reporting).
        agenthub_config: AgentHub billing/consumption config header value.
    """

    def __init__(
        self,
        agent: ClaudeAgent,
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        agenthub_config: str = "agentsruntime",
    ):
        self.agent = agent
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self._agenthub_config = agenthub_config
        self._proxy: GatewayProxy | None = None
        # Injected by the host (duck-typed) to emit LLMOps spans.
        self.span_factory: ClaudeSdkSpanFactory | None = None

    # --- LLM access -------------------------------------------------------

    @property
    def _uses_byo_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def _build_llm_env(self) -> dict[str, str]:
        """Return the SDK subprocess env vars for LLM access.

        BYO key: empty overrides — the SDK inherits ANTHROPIC_API_KEY from the
        process environment. Otherwise a local gateway proxy is started and
        the SDK is pointed at it.
        """
        if self._uses_byo_key:
            return {}

        if self._proxy is None:
            uipath_url = os.environ.get("UIPATH_URL", "")
            access_token = os.environ.get("UIPATH_ACCESS_TOKEN", "")
            if not uipath_url or not access_token:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.GATEWAY_PROXY_ERROR,
                    "Missing LLM credentials",
                    "Set ANTHROPIC_API_KEY for direct Anthropic access, or "
                    "UIPATH_URL and UIPATH_ACCESS_TOKEN to route through the "
                    "UiPath LLM Gateway (run 'uipath auth').",
                    UiPathErrorCategory.DEPLOYMENT,
                )
            self._proxy = GatewayProxy(
                uipath_base_url=uipath_url,
                access_token=access_token,
                agenthub_config=self._agenthub_config,
            )
        if not self._proxy.port:
            await self._proxy.start()

        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self._proxy.port}",
            "ANTHROPIC_API_KEY": _GATEWAY_PROXY_API_KEY,
        }

    # --- Input/output mapping ---------------------------------------------

    def _get_user_message(self, input: dict[str, Any]) -> str:
        """Build the user message string for client.query().

        Renders the agent's prompt template from validated input fields when
        declared; otherwise uses the 'input' field or a JSON dump of the input.
        """
        validated: dict[str, Any] = input
        if self.agent.input_schema is not None:
            try:
                model = self.agent.input_schema.model_validate(input)
            except ValidationError as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.INPUT_VALIDATION_ERROR,
                    "Invalid input",
                    f"Input does not match the agent's input schema: {e}",
                    UiPathErrorCategory.USER,
                ) from e
            validated = model.model_dump()

        if self.agent.prompt:
            try:
                return self.agent.prompt.format(**validated)
            except (KeyError, IndexError) as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.INPUT_VALIDATION_ERROR,
                    "Invalid prompt template",
                    f"Failed to render prompt template from input: {e}",
                    UiPathErrorCategory.USER,
                ) from e

        if isinstance(validated.get("input"), str):
            return validated["input"]
        return json.dumps(validated)

    def _map_result_output(self, message: ResultMessage) -> dict[str, Any]:
        if self.agent.output_schema is not None:
            structured = message.structured_output
            try:
                validated = self.agent.output_schema.model_validate(structured or {})
            except ValidationError as e:
                raise UiPathClaudeSDKRuntimeError(
                    UiPathClaudeSDKErrorCode.OUTPUT_VALIDATION_ERROR,
                    "Invalid structured output",
                    f"Agent output does not match the declared output schema: {e}",
                    UiPathErrorCategory.USER,
                ) from e
            return json.loads(serialize_json(validated))
        # SDK-native structured output: options.output_format set by the user.
        if self.agent.options.output_format is not None and isinstance(
            message.structured_output, dict
        ):
            return message.structured_output
        return {"result": message.result or ""}

    def create_runtime_error(self, e: Exception) -> Exception:
        """Map a raw exception before it is re-raised from stream()."""
        if isinstance(e, UiPathClaudeSDKRuntimeError):
            return e
        if isinstance(e, TimeoutError):
            return UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.AGENT_TIMEOUT,
                "Agent execution timed out",
                f"Error: {e}",
                UiPathErrorCategory.USER,
            )
        return UiPathClaudeSDKRuntimeError(
            UiPathClaudeSDKErrorCode.AGENT_EXECUTION_ERROR,
            "Agent execution failed",
            f"Error: {e}",
            UiPathErrorCategory.USER,
        )

    # --- SDK options ------------------------------------------------------

    def _build_sdk_options(
        self,
        *,
        env: dict[str, str],
        workspace: Path,
        resume_session_id: str | None = None,
    ) -> ClaudeAgentOptions:
        """Derive execution options from the user's options.

        Injects execution-scoped fields (env, cwd, output_format, resume) while
        preserving everything the developer configured. ``setting_sources``
        defaults to [] to isolate from user/project/local Claude config.
        """
        base = self.agent.options
        overrides: dict[str, Any] = {
            "env": {**base.env, **env},
            "cwd": workspace,
        }
        if base.setting_sources is None:
            overrides["setting_sources"] = []
        if self.agent.output_schema is not None and base.output_format is None:
            overrides["output_format"] = {
                "type": "json_schema",
                "schema": self.agent.output_schema.model_json_schema(),
            }
        if resume_session_id is not None:
            overrides["resume"] = resume_session_id
        return replace(base, **overrides)

    # --- Message mapping ---------------------------------------------------

    @staticmethod
    def _map_assistant(message: AssistantMessage) -> list[UiPathRuntimeStateEvent]:
        events: list[UiPathRuntimeStateEvent] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="assistant",
                        payload={"text": block.text},
                    )
                )
            elif isinstance(block, ToolUseBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="tool_call",
                        payload={"tool": block.name, "input": block.input},
                    )
                )
            elif isinstance(block, ThinkingBlock):
                events.append(
                    UiPathRuntimeStateEvent(
                        node_name="thinking",
                        payload={"thinking": block.thinking},
                    )
                )
            else:
                logger.warning(
                    "Unhandled assistant block type: %s", type(block).__name__
                )
        return events

    @staticmethod
    def _map_tool_results(message: UserMessage) -> list[UiPathRuntimeStateEvent]:
        if isinstance(message.content, list):
            return [
                UiPathRuntimeStateEvent(
                    node_name="tool_result",
                    payload={
                        "tool_use_id": block.tool_use_id,
                        "content": str(block.content),
                    },
                )
                for block in message.content
                if isinstance(block, ToolResultBlock)
            ]
        return [
            UiPathRuntimeStateEvent(
                node_name="tool_result",
                payload={"content": str(message.content)},
            )
        ]

    @staticmethod
    def _map_system(message: SystemMessage) -> UiPathRuntimeStateEvent | None:
        if isinstance(message, (TaskStartedMessage, TaskProgressMessage)):
            return UiPathRuntimeStateEvent(
                node_name="task",
                payload={
                    "task_id": message.task_id,
                    "description": message.description,
                },
            )
        return None

    @staticmethod
    def _raise_result_error(message: ResultMessage) -> None:
        parts = [message.result or "Agent run failed"]
        if message.subtype:
            parts.append(f"subtype={message.subtype}")
        if message.stop_reason:
            parts.append(f"stop_reason={message.stop_reason}")
        if message.errors:
            parts.append(f"errors={message.errors}")
        if message.permission_denials:
            parts.append(f"permission_denials={message.permission_denials}")
        raise RuntimeError(" | ".join(parts))

    # --- Runtime protocol ---------------------------------------------------

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream agent execution events.

        Yields:
            UiPathRuntimeStateEvent for each intermediate step,
            then UiPathRuntimeResult as the final event.
        """
        try:
            mapper = (
                ClaudeSdkInstrumentor(self.span_factory, self.agent.options.model or "")
                if self.span_factory
                else None
            )

            user_message = self._get_user_message(input or {})
            env = await self._build_llm_env()
            # Isolated working directory for the Claude SDK subprocess.
            workspace = Path(tempfile.mkdtemp(prefix="uipath_claude_sdk_"))
            logger.info("Claude SDK agent workspace: %s", workspace)
            try:
                sdk_options = self._build_sdk_options(env=env, workspace=workspace)

                result_message: ResultMessage | None = None

                async with ClaudeSDKClient(options=sdk_options) as client:
                    await client.query(user_message)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            if mapper:
                                mapper.on_assistant_message(message)
                            for event in self._map_assistant(message):
                                yield event
                        elif isinstance(message, UserMessage):
                            if mapper:
                                mapper.on_user_message(message)
                            for event in self._map_tool_results(message):
                                yield event
                        elif isinstance(message, ResultMessage):
                            if message.is_error:
                                self._raise_result_error(message)
                            result_message = message
                        elif isinstance(message, SystemMessage):
                            mapped = self._map_system(message)
                            if mapped is not None:
                                yield mapped

                output = (
                    self._map_result_output(result_message) if result_message else {}
                )
                yield UiPathRuntimeResult(
                    output=output, status=UiPathRuntimeStatus.SUCCESSFUL
                )
            finally:
                if mapper:
                    mapper.cleanup()
                shutil.rmtree(workspace, ignore_errors=True)
        except Exception as e:
            raise self.create_runtime_error(e) from e

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent and return the final result."""
        stream_options = (
            UiPathStreamOptions(**options.model_dump()) if options else None
        )
        result: UiPathRuntimeResult | None = None
        async for event in self.stream(input, stream_options):
            if isinstance(event, UiPathRuntimeResult):
                result = event
        return result or UiPathRuntimeResult(output={})

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this Claude SDK agent runtime."""
        entrypoints_schema = get_entrypoints_schema(self.agent)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint or "",
            uniqueId=str(uuid4()),
            type="agent",
            input=entrypoints_schema.get("input", {}),
            output=entrypoints_schema.get("output", {}),
            graph=get_agent_graph(self.agent),
        )

    async def dispose(self) -> None:
        """Cleanup runtime resources."""
        if self._proxy is not None:
            await self._proxy.stop()
            self._proxy = None
