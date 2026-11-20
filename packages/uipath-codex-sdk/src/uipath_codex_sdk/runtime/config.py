"""Configuration loader for the Codex SDK integration (codex.json)."""

from __future__ import annotations

import json
import os


class CodexConfig:
    """Loader for codex.json configuration.

    Format: ``{"agents": {"agent": "main.py:agent"}}``
    """

    def __init__(self, config_path: str = "codex.json"):
        self.config_path = config_path
        self._agents: dict[str, str] | None = None

    @property
    def exists(self) -> bool:
        return os.path.exists(self.config_path)

    @property
    def agents(self) -> dict[str, str]:
        if self._agents is None:
            self._agents = self._load_agents()
        return self._agents

    @property
    def entrypoint(self) -> list[str]:
        return list(self.agents.keys())

    def _load_agents(self) -> dict[str, str]:
        if not self.exists:
            raise FileNotFoundError(
                f"Codex configuration file not found at {self.config_path}"
            )

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{self.config_path}': {e}") from e

        agents = config.get("agents")
        if not isinstance(agents, dict) or not all(
            isinstance(v, str) for v in agents.values()
        ):
            raise ValueError(
                "Missing or invalid 'agents' key in codex.json configuration file. "
                'Expected {"agents": {"name": "file.py:variable"}}.'
            )
        return agents


__all__ = ["CodexConfig"]
