"""Dynamic loader for UiPathCodexAgent definitions (e.g. 'main.py:agent')."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Self

from uipath.runtime.errors import UiPathErrorCategory

from ..agent import UiPathCodexAgent
from .errors import UiPathCodexErrorCode, UiPathCodexRuntimeError

logger = logging.getLogger(__name__)


class CodexAgentLoader:
    """Load a UiPathCodexAgent from a 'file_path:variable_name' string."""

    def __init__(self, name: str, file_path: str, variable_name: str):
        self.name = name
        self.file_path = file_path
        self.variable_name = variable_name
        self._context_manager: Any = None

    @classmethod
    def from_path_string(cls, name: str, file_path: str) -> Self:
        if ":" not in file_path:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.CONFIG_INVALID,
                "Invalid agent path format",
                f"Invalid path format '{file_path}'. Expected 'file_path:variable_name'.",
                UiPathErrorCategory.USER,
            )
        file, variable = file_path.split(":", 1)
        return cls(name=name, file_path=file, variable_name=variable)

    async def load(self) -> UiPathCodexAgent:
        cwd = os.path.abspath(os.getcwd())
        abs_file_path = os.path.abspath(os.path.normpath(self.file_path))

        if not Path(abs_file_path).is_relative_to(Path(cwd)):
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_LOAD_FAILURE,
                "Invalid agent file path",
                f"Agent file path '{self.file_path}' must be within the working directory.",
                UiPathErrorCategory.USER,
            )
        if not os.path.exists(abs_file_path):
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_NOT_FOUND,
                "Agent file not found",
                f"Agent file '{self.file_path}' does not exist.",
                UiPathErrorCategory.USER,
            )

        self._setup_python_path(cwd)
        module = self._import_module(abs_file_path)

        agent_object = getattr(module, self.variable_name, None)
        if agent_object is None:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_NOT_FOUND,
                "Agent variable not found",
                f"'{self.variable_name}' not found in module '{self.file_path}'.",
                UiPathErrorCategory.USER,
            )

        agent = await self._resolve_agent(agent_object)

        if not isinstance(agent, UiPathCodexAgent):
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_TYPE_ERROR,
                "Invalid agent type",
                f"Expected uipath_codex_sdk.UiPathCodexAgent, got '{type(agent).__name__}'.",
                UiPathErrorCategory.USER,
            )
        return agent

    def _setup_python_path(self, cwd: str) -> None:
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        src_dir = os.path.join(cwd, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

    def _import_module(self, abs_file_path: str) -> Any:
        digest = hashlib.sha256(abs_file_path.encode()).hexdigest()[:12]
        module_name = f"{Path(abs_file_path).stem}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
        if not spec or not spec.loader:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_IMPORT_ERROR,
                "Failed to load agent module",
                f"Could not load module from: {abs_file_path}",
                UiPathErrorCategory.USER,
            )
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise UiPathCodexRuntimeError(
                UiPathCodexErrorCode.AGENT_LOAD_FAILURE,
                "Failed to execute agent module",
                f"Error loading module from {abs_file_path}: {e}",
                UiPathErrorCategory.USER,
            ) from e

    async def _resolve_agent(self, agent_object: Any) -> Any:
        """Resolve direct instances, (async) factory callables, and async context managers."""
        if callable(agent_object) and not isinstance(agent_object, UiPathCodexAgent):
            if inspect.iscoroutinefunction(agent_object):
                agent_instance = await agent_object()
            else:
                agent_instance = agent_object()
        else:
            agent_instance = agent_object

        if hasattr(agent_instance, "__aenter__") and callable(
            agent_instance.__aenter__
        ):
            self._context_manager = agent_instance
            return await agent_instance.__aenter__()
        return agent_instance

    async def cleanup(self) -> None:
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error during agent cleanup: %s", e)
            finally:
                self._context_manager = None


__all__ = ["CodexAgentLoader"]
