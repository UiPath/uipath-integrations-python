"""Factory for creating OpenAI Agents runtimes from openai_agents.json configuration."""

import asyncio
import os
from typing import Any

from agents import Agent, SQLiteSession
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from uipath.platform.resume_triggers import (
    UiPathResumeTriggerHandler,
)
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_openai_agents.runtime.agent import OpenAiAgentLoader
from uipath_openai_agents.runtime.config import OpenAiAgentsConfig
from uipath_openai_agents.runtime.errors import (
    UiPathOpenAIAgentsErrorCode,
    UiPathOpenAIAgentsRuntimeError,
)
from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime
from uipath_openai_agents.runtime.storage import SqliteResumableStorage


class UiPathOpenAIAgentRuntimeFactory:
    """Factory for creating OpenAI Agent runtimes from openai_agents.json configuration."""

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
        self._config: OpenAiAgentsConfig | None = None

        self._agent_cache: dict[str, Agent] = {}
        self._agent_loaders: dict[str, OpenAiAgentLoader] = {}
        self._agent_lock = asyncio.Lock()
        self._storage: SqliteResumableStorage | None = None
        self._storage_lock = asyncio.Lock()

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation."""
        OpenAIAgentsInstrumentor().instrument()

    def _load_config(self) -> OpenAiAgentsConfig:
        """Load openai_agents.json configuration."""
        if self._config is None:
            self._config = OpenAiAgentsConfig()
        return self._config

    def _get_connection_string(self) -> str:
        """Get the SQLite connection string used for resumable state."""
        if self.context.state_file_path is not None:
            return self.context.state_file_path

        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if (
                not self.context.resume
                and self.context.job_id is None
                and not self.context.keep_state_file
            ):
                # If not resuming and no job id, delete previous local state.
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            os.makedirs(self.context.runtime_dir, exist_ok=True)
            return path

        default_path = os.path.join("__uipath", "state.db")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return default_path

    async def _get_storage(self) -> SqliteResumableStorage:
        """Get or create the shared resumable storage instance."""
        async with self._storage_lock:
            if self._storage is None:
                self._storage = SqliteResumableStorage(self._get_connection_string())
        return self._storage

    async def _load_agent(self, entrypoint: str) -> Agent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded Agent

        Raises:
            UiPathOpenAIAgentRuntimeError: If agent cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load openai_agents.json configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.agents:
            available = ", ".join(config.entrypoint)
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.AGENT_NOT_FOUND,
                "Agent not found",
                f"Agent '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.agents[entrypoint]
        agent_loader = OpenAiAgentLoader.from_path_string(entrypoint, path)

        self._agent_loaders[entrypoint] = agent_loader

        try:
            return await agent_loader.load()
        except UiPathOpenAIAgentsRuntimeError:
            # Re-raise our own errors as-is
            raise
        except ImportError as e:
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.AGENT_IMPORT_ERROR,
                "Agent import failed",
                f"Failed to import agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.AGENT_TYPE_ERROR,
                "Invalid agent type",
                f"Agent '{entrypoint}' is not a valid OpenAI Agent: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.AGENT_VALUE_ERROR,
                "Invalid agent value",
                f"Invalid value in agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathOpenAIAgentsRuntimeError(
                UiPathOpenAIAgentsErrorCode.AGENT_LOAD_ERROR,
                "Failed to load agent",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _resolve_agent(self, entrypoint: str) -> Agent:
        """
        Resolve an agent from configuration.
        Results are cached for reuse across multiple runtime instances.

        Args:
            entrypoint: Name of the agent to resolve

        Returns:
            The loaded Agent ready for execution

        Raises:
            UiPathOpenAIAgentRuntimeError: If resolution fails
        """
        async with self._agent_lock:
            if entrypoint in self._agent_cache:
                return self._agent_cache[entrypoint]

            loaded_agent = await self._load_agent(entrypoint)
            self._agent_cache[entrypoint] = loaded_agent

            return loaded_agent

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

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        """
        Discover runtime instances for all entrypoints.

        Returns:
            List of OpenAI Agent runtime instances, one per entrypoint
        """
        entrypoints = self.discover_entrypoints()

        runtimes: list[UiPathRuntimeProtocol] = []
        for entrypoint in entrypoints:
            agent = await self._resolve_agent(entrypoint)

            runtime = await self._create_runtime_instance(
                agent=agent,
                runtime_id=entrypoint,
                entrypoint=entrypoint,
            )
            runtimes.append(runtime)

        return runtimes

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
        return None

    async def _create_runtime_instance(
        self,
        agent: Agent,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from an agent.

        Args:
            agent: The OpenAI Agent
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name

        Returns:
            Configured runtime instance
        """
        storage = await self._get_storage()
        trigger_manager = UiPathResumeTriggerHandler()
        session = SQLiteSession(session_id=runtime_id, db_path=storage.db_path)

        base_runtime = UiPathOpenAIAgentRuntime(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            session=session,
            storage=storage,
        )

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
        Create a new OpenAI Agent runtime instance.

        Args:
            entrypoint: Agent name from openai_agents.json
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

        if self._storage is not None:
            await self._storage.close()
            self._storage = None

        self._agent_loaders.clear()
        self._agent_cache.clear()
