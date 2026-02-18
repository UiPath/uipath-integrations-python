"""UiPath Google ADK Runtime Integration."""


def __getattr__(name):
    if name in ("UiPathOpenAI", "UiPathGemini", "UiPathAnthropic"):
        from . import chat

        return getattr(chat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathOpenAI",
    "UiPathGemini",
    "UiPathAnthropic",
]
