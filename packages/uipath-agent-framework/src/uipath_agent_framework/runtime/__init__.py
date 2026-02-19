"""UiPath Agent Framework Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .factory import UiPathAgentFrameworkRuntimeFactory
from .runtime import UiPathAgentFrameworkRuntime
from .schema import get_agent_graph, get_entrypoints_schema


def register_runtime_factory() -> None:
    """Register the Agent Framework factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathAgentFrameworkRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register(
        "agent-framework", create_factory, "agent_framework.json"
    )


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_agent_graph",
    "UiPathAgentFrameworkRuntimeFactory",
    "UiPathAgentFrameworkRuntime",
]
