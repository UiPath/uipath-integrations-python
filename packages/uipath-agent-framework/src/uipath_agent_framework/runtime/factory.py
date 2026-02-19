"""Factory for creating Agent Framework runtimes from agent_framework.json configuration."""

import asyncio
from typing import Any

from agent_framework import BaseAgent
from agent_framework.observability import enable_instrumentation
from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from uipath.runtime import (
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

        self._agent_cache: dict[str, BaseAgent] = {}
        self._agent_loaders: dict[str, AgentFrameworkAgentLoader] = {}
        self._agent_lock = asyncio.Lock()

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation."""
        enable_instrumentation()

        # Add OpenInference span processor for Arize Phoenix compatibility
        tracer_provider = trace.get_tracer_provider()
        if isinstance(tracer_provider, TracerProvider):
            tracer_provider.add_span_processor(
                AgentFrameworkToOpenInferenceProcessor()
            )

    def _load_config(self) -> AgentFrameworkConfig:
        """Load agent_framework.json configuration."""
        if self._config is None:
            self._config = AgentFrameworkConfig()
        return self._config

    async def _load_agent(self, entrypoint: str) -> BaseAgent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded BaseAgent

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

    async def _resolve_agent(self, entrypoint: str) -> BaseAgent:
        """
        Resolve an agent from configuration.
        Results are cached for reuse across multiple runtime instances.

        Args:
            entrypoint: Name of the agent to resolve

        Returns:
            The loaded BaseAgent ready for execution

        Raises:
            UiPathAgentFrameworkRuntimeError: If resolution fails
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

    async def get_storage(self) -> UiPathRuntimeStorageProtocol | None:
        """Get the shared storage instance."""
        return None

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        """Get the factory settings."""
        return None

    async def _create_runtime_instance(
        self,
        agent: BaseAgent,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """Create a runtime instance from an agent."""
        return UiPathAgentFrameworkRuntime(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
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
        self._agent_cache.clear()
