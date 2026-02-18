"""UiPath Google ADK Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .factory import UiPathGoogleADKRuntimeFactory
from .runtime import UiPathGoogleADKRuntime
from .schema import get_agent_graph, get_entrypoints_schema


def register_runtime_factory() -> None:
    """Register the Google ADK factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathGoogleADKRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register(
        "google-adk", create_factory, "google_adk.json"
    )


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_agent_graph",
    "UiPathGoogleADKRuntimeFactory",
    "UiPathGoogleADKRuntime",
]
