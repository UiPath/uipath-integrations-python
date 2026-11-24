"""Quickstart Codex agent: tell me a joke, via the UiPath LLM Gateway."""

from uipath_codex_sdk import UiPathCodexAgent

agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Tell me a short, clean joke about {topic}.",
)
