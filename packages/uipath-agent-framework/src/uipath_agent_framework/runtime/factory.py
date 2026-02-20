"""Factory for creating Agent Framework runtimes from agent_framework.json configuration."""

import asyncio
import os
from typing import Any

from agent_framework import WorkflowAgent
from agent_framework.observability import enable_instrumentation
from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_agent_framework.runtime.config import AgentFrameworkConfig
from uipath_agent_framework.runtime.errors import (
    UiPathAgentFrameworkErrorCode,
    UiPathAgentFrameworkRuntimeError,
)
from uipath_agent_framework.runtime.loader import AgentFrameworkAgentLoader
from uipath_agent_framework.runtime.resumable_storage import (
    ScopedCheckpointStorage,
    SqliteResumableStorage,
)
from uipath_agent_framework.runtime.runtime import UiPathAgentFrameworkRuntime


class UiPathAgentFrameworkRuntimeFactory:
    """Factory for creating Agent Framework runtimes from agent_framework.json configuration."""

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
        self._config: AgentFrameworkConfig | None = None

        self._agent_loaders: dict[str, AgentFrameworkAgentLoader] = {}

        self._storage: SqliteResumableStorage | None = None
        self._storage_lock = asyncio.Lock()

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation."""
        enable_instrumentation()

        tracer_provider = trace.get_tracer_provider()
        if isinstance(tracer_provider, TracerProvider):
            tracer_provider.add_span_processor(AgentFrameworkToOpenInferenceProcessor())

    def _load_config(self) -> AgentFrameworkConfig:
        """Load agent_framework.json configuration."""
        if self._config is None:
            self._config = AgentFrameworkConfig()
        return self._config

    def _get_db_path(self) -> str:
        """Get the database path for session persistence.

        Uses UiPathRuntimeContext to resolve the state file path.
        Cleans up stale state files when not resuming.
        """
        path = self.context.resolved_state_file_path
        # Delete previous state file if not resuming
        if (
            not self.context.resume
            and self.context.job_id is None
            and not self.context.keep_state_file
        ):
            if os.path.exists(path):
                os.remove(path)
        return path

    async def _get_storage(self) -> SqliteResumableStorage:
        """Get or create the shared resumable storage instance."""
        async with self._storage_lock:
            if self._storage is None:
                db_path = self._get_db_path()
                self._storage = SqliteResumableStorage(db_path)
                await self._storage.setup()
            return self._storage

    async def _load_agent(self, entrypoint: str) -> WorkflowAgent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded WorkflowAgent

        Raises:
            UiPathAgentFrameworkRuntimeError: If agent cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load agent_framework.json configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.agents:
            available = ", ".join(config.entrypoint)
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.AGENT_NOT_FOUND,
                "Agent not found",
                f"Agent '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.agents[entrypoint]
        agent_loader = AgentFrameworkAgentLoader.from_path_string(entrypoint, path)

        self._agent_loaders[entrypoint] = agent_loader

        try:
            return await agent_loader.load()
        except UiPathAgentFrameworkRuntimeError:
            raise
        except ImportError as e:
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.AGENT_IMPORT_ERROR,
                "Agent import failed",
                f"Failed to import agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.AGENT_TYPE_ERROR,
                "Invalid agent type",
                f"Agent '{entrypoint}' is not a valid Agent Framework agent: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.AGENT_VALUE_ERROR,
                "Invalid agent value",
                f"Invalid value in agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathAgentFrameworkRuntimeError(
                UiPathAgentFrameworkErrorCode.AGENT_LOAD_ERROR,
                "Failed to load agent",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _resolve_agent(self, entrypoint: str) -> WorkflowAgent:
        """Load a fresh agent instance for the given entrypoint.

        Agents are NOT cached — each runtime gets its own instance.
        WorkflowAgents hold internal mutable state (e.g. Workflow._is_running)
        that prevents concurrent executions on the same instance. Since the
        factory creates multiple runtimes in parallel (one per request),
        sharing an agent instance would cause "Workflow is already running" errors.
        """
        return await self._load_agent(entrypoint)

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
        """Get the shared storage instance."""
        return await self._get_storage()

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        """Get the factory settings."""
        return None

    async def _create_runtime_instance(
        self,
        agent: WorkflowAgent,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """Create a runtime instance from an agent.

        Creates the runtime with a shared SqliteResumableStorage for persistent
        conversation history and HITL trigger management. Wraps with
        UiPathResumableRuntime for resume trigger lifecycle handling.
        """
        storage = await self._get_storage()
        assert storage.checkpoint_storage is not None
        checkpoint_storage = ScopedCheckpointStorage(
            storage.checkpoint_storage, runtime_id
        )

        base_runtime = UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            checkpoint_storage=checkpoint_storage,
            resumable_storage=storage,
        )

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=UiPathResumeTriggerHandler(),
            runtime_id=runtime_id,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """
        Create a new Agent Framework runtime instance.

        Args:
            entrypoint: Agent name from agent_framework.json
            runtime_id: Unique identifier for the runtime instance
            **kwargs: Additional keyword arguments (unused)

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

        if self._storage:
            await self._storage.dispose()
            self._storage = None
