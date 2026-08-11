"""Factory for creating Claude SDK runtimes from claude.json configuration."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from uipath.core.feature_flags import FeatureFlags
from uipath.core.tracing import UiPathTraceManager
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from ..agent import ClaudeAgent
from ._telemetry import TRACER_NAME, AttributeStripper, GatewayCallTelemetry
from .config import ClaudeConfig
from .conversational_runtime import UiPathClaudeSDKConversationalRuntime
from .errors import (
    UiPathClaudeSDKErrorCode,
    UiPathClaudeSDKRuntimeError,
)
from .loader import ClaudeAgentLoader
from .runtime import UiPathClaudeSDKRuntime
from .session_paths import ClaudeSessionPaths
from .session_store import ClaudeSessionStore
from .storage import SqliteResumableStorage

logger = logging.getLogger(__name__)

_DEFAULT_RUNTIME_DIR = "__uipath"
_DEFAULT_STATE_FILE = "state.db"
_STATE_FILE_SIDECARS = ("-wal", "-shm")


TRACE_CONTEXT_FLAG = "EnableTraceContextHeaders"


def _let_the_gateway_record_model_spans() -> None:
    """Opt this package into gateway-recorded model spans.

    The flag is read by ``build_trace_context_headers``, and nothing populates
    it for a coded agent: the low-code runtime fetches its own flags, and the
    shared store otherwise answers False. Configuring it here only supplies a
    default, so a deployment that sets ``UIPATH_FEATURE_EnableTraceContextHeaders``
    still decides.
    """
    if FeatureFlags.get_flag(TRACE_CONTEXT_FLAG) is not None:
        return
    FeatureFlags.configure_flags({TRACE_CONTEXT_FLAG: True})


class UiPathClaudeSDKRuntimeFactory:
    """Factory for creating Claude SDK runtimes from claude.json configuration."""

    def __init__(
        self,
        context: UiPathRuntimeContext,
    ):
        """
        Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        self.context = context
        self._config: ClaudeConfig | None = None

        self._agent_cache: dict[str, ClaudeAgent] = {}
        self._agent_loaders: dict[str, ClaudeAgentLoader] = {}
        self._agent_lock = asyncio.Lock()

        self._storage_lock = asyncio.Lock()
        self._storage: SqliteResumableStorage | None = None

        _let_the_gateway_record_model_spans()
        self._tracer = self._setup_instrumentation(self.context.trace_manager)

    @staticmethod
    def _setup_instrumentation(trace_manager: UiPathTraceManager | None) -> Tracer:
        """Install the Claude Agent SDK instrumentor and the attribute stripper.

        The instrumentor emits the agent and tool spans but no LLM spans, since
        the bundled CLI makes the model's HTTP calls in its own process. The
        tracer returned here is the one the gateway shim's LLM spans are built
        with, and it is the only place a token count comes from.

        The stripper goes through the trace manager rather than onto the
        provider, so that it sits alongside the exporters and runs before the
        ones the CLI registers once the job is known.
        """
        provider = (
            trace_manager.tracer_provider
            if trace_manager is not None
            else trace.get_tracer_provider()
        )
        ClaudeAgentSDKInstrumentor().instrument(tracer_provider=provider)
        stripper = AttributeStripper()
        if trace_manager is not None:
            trace_manager.add_span_processor(stripper)
        elif isinstance(provider, TracerProvider):
            provider.add_span_processor(stripper)
        return provider.get_tracer(TRACER_NAME)

    def _should_reset_state_file(self) -> bool:
        """Whether a run starts from an empty state database.

        A resume reads its session id and its pending suspend record out of the
        state database, so the reset must be limited to a fresh local run: a
        resumed run either carries the resume flag or runs under a job id.
        """
        return (
            not self.context.resume
            and self.context.job_id is None
            and not self.context.keep_state_file
        )

    @staticmethod
    def _reset_state_file(path: str) -> None:
        """Delete the state database along with its write-ahead sidecars."""
        for candidate in (path, *(path + suffix for suffix in _STATE_FILE_SIDECARS)):
            if os.path.exists(candidate):
                os.remove(candidate)

    def _get_storage_path(self) -> str:
        """Get the storage path for agent state."""
        if self.context.state_file_path is not None:
            return self.context.state_file_path

        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if self._should_reset_state_file():
                self._reset_state_file(path)
            os.makedirs(self.context.runtime_dir, exist_ok=True)
            return path

        default_path = os.path.join(_DEFAULT_RUNTIME_DIR, _DEFAULT_STATE_FILE)
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return default_path

    async def _get_storage(self) -> SqliteResumableStorage:
        """Get or create the shared storage instance."""
        if self._storage is not None:
            return self._storage

        async with self._storage_lock:
            if self._storage is not None:
                return self._storage

            self._storage = SqliteResumableStorage(self._get_storage_path())
            return self._storage

    def _load_config(self) -> ClaudeConfig:
        """Load claude.json configuration."""
        if self._config is None:
            self._config = ClaudeConfig()
        return self._config

    async def _load_agent(self, entrypoint: str) -> ClaudeAgent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded ClaudeAgent

        Raises:
            UiPathClaudeSDKRuntimeError: If the agent cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load claude.json configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.agents:
            available = ", ".join(config.entrypoint)
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.AGENT_NOT_FOUND,
                "Agent not found",
                f"Agent '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.agents[entrypoint]
        agent_loader = ClaudeAgentLoader.from_path_string(entrypoint, path)
        self._agent_loaders[entrypoint] = agent_loader

        try:
            return await agent_loader.load()
        except UiPathClaudeSDKRuntimeError:
            raise
        except ImportError as e:
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.AGENT_IMPORT_ERROR,
                "Agent import failed",
                f"Failed to import agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathClaudeSDKRuntimeError(
                UiPathClaudeSDKErrorCode.AGENT_LOAD_FAILURE,
                "Failed to load agent",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _resolve_agent(self, entrypoint: str) -> ClaudeAgent:
        """
        Resolve an agent from configuration.
        Results are cached for reuse across multiple runtime instances.

        Args:
            entrypoint: Name of the agent to resolve

        Returns:
            The loaded ClaudeAgent ready for execution

        Raises:
            UiPathClaudeSDKRuntimeError: If resolution fails
        """
        async with self._agent_lock:
            if entrypoint in self._agent_cache:
                return self._agent_cache[entrypoint]

            loaded_agent = await self._load_agent(entrypoint)

            self._agent_cache[entrypoint] = loaded_agent

            return loaded_agent

    def _is_conversational(self) -> bool:
        """Conversational mode is defined by uipath.json (runtimeOptions.isConversational).

        At runtime the chat host also supplies a conversation_id, which takes
        precedence as the live signal.
        """
        if self.context.conversation_id is not None:
            return True
        config_path = self.context.config_path or "uipath.json"
        try:
            with open(config_path, "r") as f:
                uipath_config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        runtime_options = uipath_config.get("runtimeOptions", {})
        return bool(
            isinstance(runtime_options, dict)
            and runtime_options.get("isConversational")
        )

    def discover_entrypoints(self) -> list[str]:
        """
        Discover all agent entrypoints.

        Returns:
            List of agent names that can be used as entrypoints
        """
        config = self._load_config()
        if not config.exists:
            return []
        return config.entrypoint

    async def get_storage(self) -> UiPathRuntimeStorageProtocol | None:
        """
        Get the shared storage instance.
        """
        return await self._get_storage()

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        """
        Get the factory settings.

        Returns:
            Factory settings
        """
        return UiPathRuntimeFactorySettings(
            agent_type="uipath_coded",
            agent_framework="claude-agent-sdk",
        )

    def _create_delegate(
        self,
        agent: ClaudeAgent,
        runtime_id: str,
        entrypoint: str,
        storage: SqliteResumableStorage,
    ) -> UiPathRuntimeProtocol:
        """
        Create the runtime that owns the Claude SDK run loop.

        The runtime gets its own gateway telemetry, whose buffer of calls
        belongs to a single run and must not be shared with another run
        executing in the same process.

        Args:
            agent: The loaded agent definition
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name
            storage: Shared storage instance

        Returns:
            The conversational runtime when the project is conversational,
            otherwise the standard runtime
        """
        session_store = ClaudeSessionStore(storage, runtime_id)
        runtime: UiPathClaudeSDKRuntime

        if not self._is_conversational():
            runtime = UiPathClaudeSDKRuntime(
                agent=agent,
                session_store=session_store,
                session_paths=self._session_paths(runtime_id, entrypoint),
                runtime_id=runtime_id,
                entrypoint=entrypoint,
            )
        else:
            conversation_id = self.context.conversation_id or runtime_id
            runtime = UiPathClaudeSDKConversationalRuntime(
                agent=agent,
                session_store=session_store,
                session_paths=self._session_paths(conversation_id, entrypoint),
                runtime_id=runtime_id,
                entrypoint=entrypoint,
            )

        runtime.telemetry = GatewayCallTelemetry(self._tracer)
        return runtime

    def _session_paths(self, workspace_id: str, entrypoint: str) -> ClaudeSessionPaths:
        """Claude config and working directories for a run.

        The working directory goes beneath the directory the agent was loaded
        from, so the files packaged alongside it, ``.claude`` skills above all,
        are found by the CLI without being copied anywhere.
        """
        paths = ClaudeSessionPaths.for_runtime(
            self.context.runtime_dir, workspace_id, self._project_dir(entrypoint)
        )
        logger.debug(
            "Claude session paths: config_dir=%s workspace=%s",
            paths.config_dir,
            paths.workspace,
        )
        return paths

    def _project_dir(self, entrypoint: str) -> Path | None:
        """The directory the agent was loaded from, when it is known."""
        loader = self._agent_loaders.get(entrypoint)
        if loader is None:
            return None
        return Path(os.path.abspath(loader.file_path)).parent

    async def _create_runtime_instance(
        self,
        agent: ClaudeAgent,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from an agent.

        Args:
            agent: The loaded agent definition
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name

        Returns:
            Configured runtime instance
        """
        storage = await self._get_storage()

        base_runtime = self._create_delegate(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            storage=storage,
        )

        trigger_manager = UiPathResumeTriggerHandler()

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id=runtime_id,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """
        Create a new Claude SDK runtime instance.

        Every runtime is wrapped in UiPathResumableRuntime, so a suspension is
        persisted as a resume trigger and restored on the next resume.

        Args:
            entrypoint: Agent name from claude.json
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured runtime instance with agent
        """
        agent = await self._resolve_agent(entrypoint)

        return await self._create_runtime_instance(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        for loader in self._agent_loaders.values():
            await loader.cleanup()

        self._agent_loaders.clear()
        self._agent_cache.clear()

        if self._storage:
            await self._storage.dispose()
            self._storage = None


__all__ = ["UiPathClaudeSDKRuntimeFactory"]
