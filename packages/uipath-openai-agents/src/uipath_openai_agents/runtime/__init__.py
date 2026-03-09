"""UiPath OpenAI Agents Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .context import get_agent_context_type, parse_input_to_context
from .factory import UiPathOpenAIAgentRuntimeFactory
from .runtime import UiPathOpenAIAgentRuntime
from .schema import get_agent_schema, get_entrypoints_schema
from .storage import SqliteResumableStorage


def register_runtime_factory() -> None:
    """Register the OpenAI Agents factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathOpenAIAgentRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register(
        "openai-agents", create_factory, "openai_agents.json"
    )


__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_agent_schema",
    "UiPathOpenAIAgentRuntimeFactory",
    "UiPathOpenAIAgentRuntime",
    "SqliteResumableStorage",
    "get_agent_context_type",
    "parse_input_to_context",
]
