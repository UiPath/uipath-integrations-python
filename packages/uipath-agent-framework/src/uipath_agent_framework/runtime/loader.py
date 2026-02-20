import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Self

from agent_framework import BaseAgent, WorkflowAgent
from uipath.runtime.errors import UiPathErrorCategory

from .errors import (
    UiPathAgentFrameworkErrorCode,
    UiPathAgentFrameworkRuntimeError,
)


class AgentFrameworkAgentLoader:
    """Load agent from a Python file path (e.g.: 'main.py:agent')"""

    def __init__(self, name: str, file_path: str, variable_name: str):
        """Initialize the Agent Framework agent loader.
        Args:
            name: The name of the agent.
            file_path: The path to the Python file containing the agent.
            variable_name: The name of the variable that contains the agent instance.
        """
        self.name = name
        self.file_path = file_path
        self.variable_name = variable_name
        self._context_manager: Any = None

    @classmethod
    def from_path_string(cls, name: str, file_path: str) -> Self:
        """
        Create an AgentFrameworkAgentLoader from a path string.

        Args:
            name: Human-readable name of the agent.
            file_path: The path string in the format 'file_path:variable_name'.

        Returns:
            An instance of AgentFrameworkAgentLoader.
        """
        if ":" not in file_path:
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.CONFIG_INVALID,
                title="Invalid agent path format",
                detail=f"Invalid path format '{file_path}'. Expected format 'file_path:variable_name'.",
                category=UiPathErrorCategory.USER,
            )
        file, variable = file_path.split(":", 1)
        return cls(name=name, file_path=file, variable_name=variable)

    async def load(self) -> WorkflowAgent:
        """
        Load and return the agent.

        Returns:
            An instance of the loaded WorkflowAgent.

        Raises:
            UiPathAgentFrameworkRuntimeError: If loading fails
        """
        # Validate and normalize paths
        cwd = os.path.abspath(os.getcwd())
        abs_file_path = os.path.abspath(os.path.normpath(self.file_path))

        if not abs_file_path.startswith(cwd):
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_VALUE_ERROR,
                title="Invalid agent file path",
                detail=f"Agent file path '{self.file_path}' must be within the current working directory.",
                category=UiPathErrorCategory.USER,
            )

        if not os.path.exists(abs_file_path):
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_NOT_FOUND,
                title="Agent file not found",
                detail=f"Agent file '{self.file_path}' does not exist.",
                category=UiPathErrorCategory.USER,
            )
        # Ensure the current directory and src/ is in sys.path
        self._setup_python_path(cwd)

        # Import the module and retrieve the agent instance
        module = self._import_module(abs_file_path)

        # Get the agent instance from the module
        agent_object = getattr(module, self.variable_name, None)
        if agent_object is None:
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_NOT_FOUND,
                title="Agent variable not found",
                detail=f"'{self.variable_name}' not found in module '{self.file_path}'.",
                category=UiPathErrorCategory.USER,
            )

        agent = await self._resolve_agent(agent_object)
        if not isinstance(agent, WorkflowAgent):
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_TYPE_ERROR,
                title="Invalid agent type",
                detail=f"Expected WorkflowAgent, got '{type(agent).__name__}'. "
                "Use workflow.as_agent() to create a WorkflowAgent.",
                category=UiPathErrorCategory.USER,
            )

        return agent

    def _setup_python_path(self, cwd: str) -> None:
        """Add current directory and src/ to Python path if needed."""
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Support src-layout projects (mimics editable install)
        src_dir = os.path.join(cwd, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

    def _import_module(self, abs_file_path: str) -> Any:
        """Import a Python module from a file path."""
        module_name = Path(abs_file_path).stem
        spec = importlib.util.spec_from_file_location(module_name, abs_file_path)

        if not spec or not spec.loader:
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_IMPORT_ERROR,
                title="Failed to load agent module",
                detail=f"Could not load module from: {abs_file_path}",
                category=UiPathErrorCategory.USER,
            )

        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise UiPathAgentFrameworkRuntimeError(
                code=UiPathAgentFrameworkErrorCode.AGENT_LOAD_ERROR,
                title="Failed to execute agent module",
                detail=f"Error loading module from {abs_file_path}: {str(e)}",
                category=UiPathErrorCategory.USER,
            ) from e

    async def _resolve_agent(self, agent_object: Any) -> BaseAgent:
        """
        Resolve an agent object that might be:
        - A direct BaseAgent
        - A function that returns a BaseAgent
        - An async function that returns a BaseAgent
        - An async context manager that yields a BaseAgent
        """
        agent_instance = None
        # Handle callable (sync or async)
        if callable(agent_object):
            if inspect.iscoroutinefunction(agent_object):
                agent_instance = await agent_object()
            else:
                agent_instance = agent_object()
        else:
            agent_instance = agent_object

        # Handle async context manager
        if hasattr(agent_instance, "__aenter__") and callable(
            agent_instance.__aenter__
        ):
            self._context_manager = agent_instance
            return await agent_instance.__aenter__()

        return agent_instance

    async def cleanup(self) -> None:
        """Clean up resources (e.g., exit async context managers)."""
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error during agent cleanup: {e}")
            finally:
                self._context_manager = None
