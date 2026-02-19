from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from uipath_llamaindex.runtime.factory import UiPathLlamaIndexRuntimeFactory
from uipath_llamaindex.runtime.runtime import UiPathLlamaIndexRuntime
from uipath_llamaindex.runtime.schema import (
    get_entrypoints_schema,
    get_workflow_schema,
)


def register_runtime_factory() -> None:
    """Register the LlamaIndex factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathLlamaIndexRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register(
        "llamaindex", create_factory, "llama_index.json"
    )


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_workflow_schema",
    "UiPathLlamaIndexRuntimeFactory",
    "UiPathLlamaIndexRuntime",
]
