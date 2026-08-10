"""Environment variables handed to the bundled Claude Code CLI.

``ANTHROPIC_BASE_URL`` is the only routing channel the CLI offers, and it also
picks its own auxiliary models for background work. Every one of those is
pinned to a catalog-resolved id so no traffic escapes to a default Anthropic
model name that the tenant does not host.
"""

from __future__ import annotations

from typing import Protocol

from .catalog import ResolvedModelSet

AUXILIARY_MODEL_VARS: dict[str, str] = {
    "ANTHROPIC_SMALL_FAST_MODEL": "haiku",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus",
    "CLAUDE_CODE_SUBAGENT_MODEL": "primary",
    "CLAUDE_CODE_BG_CLASSIFIER_MODEL": "haiku",
}

STATIC_VARS: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0",
}


class GatewayEndpoint(Protocol):
    """The two shim facts the CLI needs in order to reach the shim."""

    @property
    def base_url(self) -> str: ...

    @property
    def api_key(self) -> str: ...


def build_llm_env(shim: GatewayEndpoint, resolved: ResolvedModelSet) -> dict[str, str]:
    """Build the CLI subprocess environment for a running shim."""
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": shim.base_url,
        "ANTHROPIC_API_KEY": shim.api_key,
    }
    for name, family in AUXILIARY_MODEL_VARS.items():
        env[name] = resolved.for_family(family).wire_name
    env.update(STATIC_VARS)
    return env


__all__ = ["AUXILIARY_MODEL_VARS", "STATIC_VARS", "GatewayEndpoint", "build_llm_env"]
