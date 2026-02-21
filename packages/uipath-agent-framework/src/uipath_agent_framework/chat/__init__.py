"""
UiPath Agent Framework Chat module.

Provides chat client implementations that route through UiPath's LLM Gateway
for use with Agent Framework agents.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (httpx, openai, anthropic SDK) at import time.
"""


def __getattr__(name):
    if name == "UiPathOpenAIChatClient":
        from .openai import UiPathOpenAIChatClient

        return UiPathOpenAIChatClient
    if name == "UiPathAnthropicClient":
        from .anthropic import UiPathAnthropicClient

        return UiPathAnthropicClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathOpenAIChatClient",
    "UiPathAnthropicClient",
]
