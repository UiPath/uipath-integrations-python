"""UiPath Claude SDK Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .conversational_runtime import UiPathClaudeSDKConversationalRuntime
from .factory import UiPathClaudeSDKRuntimeFactory
from .runtime import UiPathClaudeSDKRuntime
from .schema import get_agent_graph, get_entrypoints_schema


def register_runtime_factory() -> None:
    """Register the Claude SDK factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathClaudeSDKRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register("claude", create_factory, "claude.json")


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_agent_graph",
    "UiPathClaudeSDKRuntimeFactory",
    "UiPathClaudeSDKRuntime",
    "UiPathClaudeSDKConversationalRuntime",
]
