"""Factory for creating LlamaIndex runtimes from llama_index.json configuration."""

import asyncio
import os

from openinference.instrumentation.llama_index import (
    LlamaIndexInstrumentor,
    get_current_span,
)
from uipath.core.tracing import UiPathSpanUtils, UiPathTraceManager
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory
from workflows import Workflow

from uipath_llamaindex.runtime._telemetry import (
    ToolCallAttributeNormalizer,
)
from uipath_llamaindex.runtime.config import LlamaIndexConfig
from uipath_llamaindex.runtime.errors import (
    UiPathLlamaIndexErrorCode,
    UiPathLlamaIndexRuntimeError,
)
from uipath_llamaindex.runtime.runtime import UiPathLlamaIndexRuntime
from uipath_llamaindex.runtime.storage import SqliteResumableStorage
from uipath_llamaindex.runtime.workflow import LlamaIndexWorkflowLoader


class UiPathLlamaIndexRuntimeFactory:
    """Factory for creating LlamaIndex runtimes from llama_index.json configuration."""

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
        self._config: LlamaIndexConfig | None = None

        self._workflow_cache: dict[str, Workflow] = {}
        self._workflow_loaders: dict[str, LlamaIndexWorkflowLoader] = {}
        self._workflow_lock = asyncio.Lock()

        self._storage_lock = asyncio.Lock()
        self._storage: SqliteResumableStorage | None = None

        self._setup_instrumentation(self.context.trace_manager)

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        LlamaIndexInstrumentor().instrument()
        UiPathSpanUtils.register_current_span_provider(get_current_span)

        if trace_manager:
            trace_manager.tracer_provider.add_span_processor(
                ToolCallAttributeNormalizer()
            )

    def _get_storage_path(self) -> str:
        """Get the storage path for workflow state."""
        if self.context.state_file_path is not None:
            return self.context.state_file_path

        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if (
                not self.context.resume
                and self.context.job_id is None
                and not self.context.keep_state_file
            ):
                # If not resuming and no job id, delete the previous state file
                if os.path.exists(path):
                    os.remove(path)
            os.makedirs(self.context.runtime_dir, exist_ok=True)
            return path

        default_path = os.path.join("__uipath", "state.db")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return default_path

    async def _get_storage(self) -> SqliteResumableStorage:
        """Get or create the shared storage instance."""
        if self._storage is not None:
            return self._storage

        async with self._storage_lock:
            if self._storage is not None:
                return self._storage

            storage_path = self._get_storage_path()
            self._storage = SqliteResumableStorage(storage_path)
            await self._storage.setup()
            return self._storage

    def _load_config(self) -> LlamaIndexConfig:
        """Load llama_index.json configuration."""
        if self._config is None:
            self._config = LlamaIndexConfig()
        return self._config

    async def _load_workflow(self, entrypoint: str) -> Workflow:
        """
        Load a workflow for the given entrypoint.

        Args:
            entrypoint: Name of the workflow to load

        Returns:
            The loaded Workflow

        Raises:
            LlamaIndexRuntimeError: If workflow cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.workflows:
            available = ", ".join(config.entrypoints)
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.WORKFLOW_NOT_FOUND,
                "Workflow not found",
                f"Workflow '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.workflows[entrypoint]
        workflow_loader = LlamaIndexWorkflowLoader.from_path_string(entrypoint, path)

        self._workflow_loaders[entrypoint] = workflow_loader

        try:
            return await workflow_loader.load()

        except ImportError as e:
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.WORKFLOW_IMPORT_ERROR,
                "Workflow import failed",
                f"Failed to import workflow '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.WORKFLOW_TYPE_ERROR,
                "Invalid workflow type",
                f"Workflow '{entrypoint}' is not a valid Workflow: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.WORKFLOW_VALUE_ERROR,
                "Invalid workflow value",
                f"Invalid value in workflow '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise UiPathLlamaIndexRuntimeError(
                UiPathLlamaIndexErrorCode.WORKFLOW_LOAD_ERROR,
                "Failed to load workflow",
                f"Unexpected error loading workflow '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _resolve_workflow(self, entrypoint: str) -> Workflow:
        """
        Resolve a workflow from configuration.
        Results are cached for reuse across multiple runtime instances.

        Args:
            entrypoint: Name of the workflow to resolve

        Returns:
            The loaded Workflow ready for execution

        Raises:
            LlamaIndexRuntimeError: If resolution fails
        """
        async with self._workflow_lock:
            if entrypoint in self._workflow_cache:
                return self._workflow_cache[entrypoint]

            loaded_workflow = await self._load_workflow(entrypoint)

            self._workflow_cache[entrypoint] = loaded_workflow

            return loaded_workflow

    def discover_entrypoints(self) -> list[str]:
        """
        Discover all workflow entrypoints.

        Returns:
            List of workflow names that can be used as entrypoints
        """
        config = self._load_config()
        if not config.exists:
            return []
        return config.entrypoints

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
        workflow: Workflow,
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from a workflow.

        Args:
            workflow: The workflow
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Workflow entrypoint name

        Returns:
            Configured runtime instance
        """

        storage = await self._get_storage()

        base_runtime = UiPathLlamaIndexRuntime(
            workflow=workflow,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            storage=storage,
            debug_mode=self.context.command in ("debug", "dev"),
        )

        trigger_manager = UiPathResumeTriggerHandler()

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id=runtime_id,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs
    ) -> UiPathRuntimeProtocol:
        """
        Create a new LlamaIndex runtime instance.

        Args:
            entrypoint: Workflow name from llama_index.json
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured runtime instance with workflow
        """
        workflow = await self._resolve_workflow(entrypoint)

        return await self._create_runtime_instance(
            workflow=workflow,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        for loader in self._workflow_loaders.values():
            await loader.cleanup()

        self._workflow_loaders.clear()
        self._workflow_cache.clear()

        if self._storage:
            await self._storage.dispose()
            self._storage = None
