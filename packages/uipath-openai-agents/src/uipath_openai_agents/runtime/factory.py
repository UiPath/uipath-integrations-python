"""Factory for creating OpenAI Agents runtimes from openai_agents.json configuration."""

import asyncio
from typing import Any

from agents import Agent
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from uipath.core.adapters import EvaluatorProtocol
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_openai_agents.governance import install_governance
from uipath_openai_agents.runtime.agent import OpenAiAgentLoader
from uipath_openai_agents.runtime.config import OpenAiAgentsConfig
from uipath_openai_agents.runtime.errors import (
    UiPathOpenAIAgentsErrorCode,
    UiPathOpenAIAgentsRuntimeError,
)
from uipath_openai_agents.runtime.runtime import UiPathOpenAIAgentRuntime


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

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation."""
        OpenAIAgentsInstrumentor().instrument()

    def _load_config(self) -> OpenAiAgentsConfig:
        """Load openai_agents.json configuration."""
        if self._config is None:
            self._config = OpenAiAgentsConfig()
        return self._config

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
        evaluator: EvaluatorProtocol | None = None,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from an agent.

        Args:
            agent: The OpenAI Agent
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name
            evaluator: When supplied, governance hooks are installed on the
                agent graph in place via :func:`install_governance`.

        Returns:
            Configured runtime instance
        """
        if evaluator is not None:
            install_governance(
                agent,
                evaluator,
                agent_name=entrypoint,
                session_id=runtime_id,
            )

        return UiPathOpenAIAgentRuntime(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """
        Create a new OpenAI Agent runtime instance.

        Args:
            entrypoint: Agent name from openai_agents.json
            runtime_id: Unique identifier for the runtime instance
            **kwargs: Forwarded factory kwargs. Recognized: ``evaluator``
                (``EvaluatorProtocol``) — when present, governance hooks are
                installed on the agent via :func:`install_governance`.

        Returns:
            Configured runtime instance with agent
        """
        agent = await self._resolve_agent(entrypoint)

        return await self._create_runtime_instance(
            agent=agent,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            evaluator=kwargs.get("evaluator"),
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        for loader in self._agent_loaders.values():
            await loader.cleanup()

        self._agent_loaders.clear()
        self._agent_cache.clear()
