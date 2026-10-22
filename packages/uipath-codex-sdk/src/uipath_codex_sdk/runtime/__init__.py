"""UiPath Codex SDK Runtime."""

from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .factory import UiPathCodexRuntimeFactory
from .runtime import UiPathCodexRuntime


def register_runtime_factory() -> None:
    """Register the Codex SDK factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathCodexRuntimeFactory(context=context or UiPathRuntimeContext())

    UiPathRuntimeFactoryRegistry.register("codex", create_factory, "codex.json")


__all__ = [
    "register_runtime_factory",
    "UiPathCodexRuntimeFactory",
    "UiPathCodexRuntime",
]
