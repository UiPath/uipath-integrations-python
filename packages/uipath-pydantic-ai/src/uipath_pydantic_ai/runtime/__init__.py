"""UiPath PydanticAI Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .factory import UiPathPydanticAIRuntimeFactory
from .runtime import UiPathPydanticAIRuntime
from .schema import get_agent_schema, get_entrypoints_schema


def register_runtime_factory() -> None:
    """Register the PydanticAI factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathPydanticAIRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register(
        "pydantic-ai", create_factory, "pydantic_ai.json"
    )


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_agent_schema",
    "UiPathPydanticAIRuntimeFactory",
    "UiPathPydanticAIRuntime",
]
