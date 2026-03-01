"""
UiPath PydanticAI Chat module.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (openai SDK, httpx) at import time. This significantly
improves CLI startup performance.

Do NOT add eager imports like:
    from .openai import UiPathChatOpenAI  # BAD - loads openai SDK immediately

Instead, all exports are loaded on-demand when first accessed.
"""


def __getattr__(name):
    if name == "UiPathChatOpenAI":
        from .openai import UiPathChatOpenAI

        return UiPathChatOpenAI
    if name in ("OpenAIModels", "GeminiModels", "BedrockModels"):
        from . import supported_models

        return getattr(supported_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathChatOpenAI",
    "OpenAIModels",
    "GeminiModels",
    "BedrockModels",
]
