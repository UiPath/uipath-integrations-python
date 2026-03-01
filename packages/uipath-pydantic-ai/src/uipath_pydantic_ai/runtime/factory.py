"""Factory for creating PydanticAI runtimes from pydantic_ai.json configuration."""

import asyncio
from typing import Any

from pydantic_ai import Agent
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_pydantic_ai.runtime.config import PydanticAiConfig
from uipath_pydantic_ai.runtime.errors import (
    UiPathPydanticAIErrorCode,
    UiPathPydanticAIRuntimeError,
)
from uipath_pydantic_ai.runtime.loader import PydanticAiAgentLoader
from uipath_pydantic_ai.runtime.runtime import UiPathPydanticAIRuntime


class UiPathPydanticAIRuntimeFactory:
    """Factory for creating PydanticAI runtimes from pydantic_ai.json configuration."""

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
        self._config: PydanticAiConfig | None = None

        self._agent_cache: dict[str, Agent] = {}
        self._agent_loaders: dict[str, PydanticAiAgentLoader] = {}
        self._agent_lock = asyncio.Lock()

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation.

        PydanticAI uses OpenInferenceSpanProcessor which is added to
        the TracerProvider via trace_manager.add_span_processor().
        """
        try:
            from openinference.instrumentation.pydantic_ai import (
                OpenInferenceSpanProcessor,
            )

            if self.context.trace_manager is not None:
                self.context.trace_manager.add_span_processor(
                    OpenInferenceSpanProcessor()
                )
                Agent.instrument_all()
        except Exception:
            pass

    def _load_config(self) -> PydanticAiConfig:
        """Load pydantic_ai.json configuration."""
        if self._config is None:
            self._config = PydanticAiConfig()
        return self._config

    async def _load_agent(self, entrypoint: str) -> Agent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded Agent

        Raises:
            UiPathPydanticAIRuntimeError: If agent cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load pydantic_ai.json configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.agents:
            available = ", ".join(config.entrypoint)
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_NOT_FOUND,
                "Agent not found",
                f"Agent '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.agents[entrypoint]
        agent_loader = PydanticAiAgentLoader.from_path_string(entrypoint, path)

        self._agent_loaders[entrypoint] = agent_loader

        try:
            return await agent_loader.load()
        except UiPathPydanticAIRuntimeError:
            # Re-raise our own errors as-is
            raise
        except ImportError as e:
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_IMPORT_ERROR,
                "Agent import failed",
                f"Failed to import agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_TYPE_ERROR,
                "Invalid agent type",
                f"Agent '{entrypoint}' is not a valid PydanticAI Agent: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_VALUE_ERROR,
                "Invalid agent value",
                f"Invalid value in agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathPydanticAIRuntimeError(
                UiPathPydanticAIErrorCode.AGENT_LOAD_FAILURE,
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
            UiPathPydanticAIRuntimeError: If resolution fails
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
            List of PydanticAI runtime instances, one per entrypoint
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
        return None

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
            agent: The PydanticAI Agent
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name

        Returns:
            Configured runtime instance
        """
        return UiPathPydanticAIRuntime(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """
        Create a new PydanticAI runtime instance.

        Args:
            entrypoint: Agent name from pydantic_ai.json
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
        self._agent_cache.clear()
