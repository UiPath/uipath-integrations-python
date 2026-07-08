"""UiPath Claude Agent SDK Integration."""

from .agent import ClaudeAgent

__all__ = ["ClaudeAgent"]


def __getattr__(name: str):
    if name == "UiPathClaudeSDKRuntimeFactory":
        from .runtime.factory import UiPathClaudeSDKRuntimeFactory

        return UiPathClaudeSDKRuntimeFactory
    if name == "UiPathClaudeSDKRuntime":
        from .runtime.runtime import UiPathClaudeSDKRuntime

        return UiPathClaudeSDKRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
