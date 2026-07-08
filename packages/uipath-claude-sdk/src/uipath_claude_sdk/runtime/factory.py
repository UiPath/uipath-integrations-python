"""Factory for creating Claude SDK runtimes from claude.json configuration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

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
from .config import ClaudeConfig
from .conversational_runtime import UiPathClaudeSDKConversationalRuntime
from .errors import (
    UiPathClaudeSDKErrorCode,
    UiPathClaudeSDKRuntimeError,
)
from .loader import ClaudeAgentLoader
from .runtime import UiPathClaudeSDKRuntime
from .session_store import ClaudeSessionStore
from .storage import SqliteResumableStorage

_DEFAULT_RUNTIME_DIR = "__uipath"


class UiPathClaudeSDKRuntimeFactory:
    """Factory for creating Claude SDK runtimes from claude.json configuration."""

    def __init__(
        self,
        context: UiPathRuntimeContext,
    ):
        self.context = context
        self._config: ClaudeConfig | None = None

        self._agent_cache: dict[str, ClaudeAgent] = {}
        self._agent_loaders: dict[str, ClaudeAgentLoader] = {}
        self._agent_lock = asyncio.Lock()
        self._storage: SqliteResumableStorage | None = None

    def _load_config(self) -> ClaudeConfig:
        if self._config is None:
            self._config = ClaudeConfig()
        return self._config

    def _runtime_dir(self) -> Path:
        return Path(self.context.runtime_dir or _DEFAULT_RUNTIME_DIR)

    def _get_sqlite_storage(self) -> SqliteResumableStorage:
        if self._storage is None:
            state_file = self.context.state_file_path or str(
                self._runtime_dir() / "state.db"
            )
            self._storage = SqliteResumableStorage(state_file)
        return self._storage

    async def _load_agent(self, entrypoint: str) -> ClaudeAgent:
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
        config = self._load_config()
        if not config.exists:
            return []
        return config.entrypoint

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        runtimes: list[UiPathRuntimeProtocol] = []
        for entrypoint in self.discover_entrypoints():
            runtime = await self.new_runtime(
                entrypoint=entrypoint, runtime_id=entrypoint
            )
            runtimes.append(runtime)
        return runtimes

    async def get_storage(self) -> UiPathRuntimeStorageProtocol | None:
        return self._get_sqlite_storage()

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        return UiPathRuntimeFactorySettings(
            agent_type="uipath_coded",
            agent_framework="claude-agent-sdk",
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """Create a new Claude SDK runtime instance.

        Standard agents return the base runtime. Conversational agents
        (uipath.json runtimeOptions.isConversational) are wrapped in
        UiPathResumableRuntime so each exchange suspends with an API resume
        trigger and resumes on the next user message.
        """
        agent = await self._resolve_agent(entrypoint)

        conversational_id = self.context.conversation_id or runtime_id

        if not self._is_conversational():
            return UiPathClaudeSDKRuntime(
                agent=agent,
                runtime_id=runtime_id,
                entrypoint=entrypoint,
            )

        storage = self._get_sqlite_storage()
        workspace_root = (
            self._runtime_dir() / "claude_workspaces" / _sanitize(conversational_id)
        )
        runtime = UiPathClaudeSDKConversationalRuntime(
            agent=agent,
            session_store=ClaudeSessionStore(storage, runtime_id),
            workspace_root=workspace_root,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )
        return UiPathResumableRuntime(
            delegate=runtime,
            storage=storage,
            trigger_manager=UiPathResumeTriggerHandler(),
            runtime_id=runtime_id,
        )

    async def dispose(self) -> None:
        for loader in self._agent_loaders.values():
            await loader.cleanup()

        self._agent_loaders.clear()
        self._agent_cache.clear()

        if self._storage is not None:
            await self._storage.dispose()
            self._storage = None


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


__all__ = ["UiPathClaudeSDKRuntimeFactory"]
