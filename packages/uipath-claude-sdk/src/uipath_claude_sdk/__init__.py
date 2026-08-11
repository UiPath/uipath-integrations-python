"""UiPath Claude Agent SDK Integration."""

from typing import Any

from .agent import ClaudeAgent, UiPathClaudeAgent
from .interrupts import (
    InterruptOutsideRunError,
    PendingSuspend,
    SuspendAlreadyPendingError,
    SuspendChannel,
    SuspendChannelError,
    UnknownInterruptError,
    interrupt,
    uipath_tool_server,
)
from .models import UiPathModel

__all__ = [
    "ClaudeAgent",
    "InterruptOutsideRunError",
    "PendingSuspend",
    "SuspendAlreadyPendingError",
    "SuspendChannel",
    "SuspendChannelError",
    "UiPathClaudeAgent",
    "UiPathModel",
    "UnknownInterruptError",
    "interrupt",
    "register_middleware",
    "uipath_tool_server",
]


def __getattr__(name: str) -> Any:
    if name == "register_middleware":
        from .middlewares import register_middleware

        return register_middleware
    if name == "UiPathClaudeSDKRuntimeFactory":
        from .runtime.factory import UiPathClaudeSDKRuntimeFactory

        return UiPathClaudeSDKRuntimeFactory
    if name == "UiPathClaudeSDKRuntime":
        from .runtime.runtime import UiPathClaudeSDKRuntime

        return UiPathClaudeSDKRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
