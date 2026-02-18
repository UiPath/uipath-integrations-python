"""Factory for creating Google ADK runtimes from google_adk.json configuration."""

import asyncio
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import InMemoryRunner
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_google_adk.runtime.config import GoogleADKConfig
from uipath_google_adk.runtime.errors import (
    UiPathGoogleADKErrorCode,
    UiPathGoogleADKRuntimeError,
)
from uipath_google_adk.runtime.loader import GoogleADKAgentLoader
from uipath_google_adk.runtime.runtime import UiPathGoogleADKRuntime


class UiPathGoogleADKRuntimeFactory:
    """Factory for creating Google ADK runtimes from google_adk.json configuration."""

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
        self._config: GoogleADKConfig | None = None

        self._agent_cache: dict[str, BaseAgent] = {}
        self._agent_loaders: dict[str, GoogleADKAgentLoader] = {}
        self._agent_lock = asyncio.Lock()

        self._setup_instrumentation()

    def _setup_instrumentation(self) -> None:
        """Setup tracing and instrumentation."""
        GoogleADKInstrumentor().instrument()

    def _load_config(self) -> GoogleADKConfig:
        """Load google_adk.json configuration."""
        if self._config is None:
            self._config = GoogleADKConfig()
        return self._config

    async def _load_agent(self, entrypoint: str) -> BaseAgent:
        """
        Load an agent for the given entrypoint.

        Args:
            entrypoint: Name of the agent to load

        Returns:
            The loaded BaseAgent

        Raises:
            UiPathGoogleADKRuntimeError: If agent cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load google_adk.json configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.agents:
            available = ", ".join(config.entrypoint)
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.AGENT_NOT_FOUND,
                "Agent not found",
                f"Agent '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.agents[entrypoint]
        agent_loader = GoogleADKAgentLoader.from_path_string(entrypoint, path)

        self._agent_loaders[entrypoint] = agent_loader

        try:
            return await agent_loader.load()
        except UiPathGoogleADKRuntimeError:
            raise
        except ImportError as e:
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.AGENT_IMPORT_ERROR,
                "Agent import failed",
                f"Failed to import agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.AGENT_TYPE_ERROR,
                "Invalid agent type",
                f"Agent '{entrypoint}' is not a valid Google ADK Agent: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.AGENT_VALUE_ERROR,
                "Invalid agent value",
                f"Invalid value in agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathGoogleADKRuntimeError(
                UiPathGoogleADKErrorCode.AGENT_LOAD_ERROR,
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
            UiPathGoogleADKRuntimeError: If resolution fails
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
        agent: BaseAgent,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from an agent.

        Creates the InMemoryRunner and session eagerly so the runtime
        is fully initialized and ready to execute.
        """
        runner = InMemoryRunner(
            agent=agent,
            app_name=UiPathGoogleADKRuntime.APP_NAME,
        )
        session = await runner.session_service.create_session(
            app_name=UiPathGoogleADKRuntime.APP_NAME,
            user_id=UiPathGoogleADKRuntime.USER_ID,
        )

        return UiPathGoogleADKRuntime(
            agent=agent,
            runner=runner,
            session=session,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """
        Create a new Google ADK runtime instance.

        Args:
            entrypoint: Agent name from google_adk.json
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
