"""
UiPath Google ADK Chat module.

Provides LLM implementations that route through UiPath's LLM Gateway (AgentHub)
for use with Google ADK agents.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (httpx, anthropic SDK) at import time.
"""


def __getattr__(name):
    if name == "UiPathOpenAI":
        from .openai import UiPathOpenAI

        return UiPathOpenAI
    if name == "UiPathGemini":
        from .gemini import UiPathGemini

        return UiPathGemini
    if name == "UiPathAnthropic":
        from .anthropic import UiPathAnthropic

        return UiPathAnthropic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathOpenAI",
    "UiPathGemini",
    "UiPathAnthropic",
]
