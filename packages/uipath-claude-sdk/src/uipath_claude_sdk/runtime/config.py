"""Configuration loader for the Claude SDK integration (claude.json)."""

from __future__ import annotations

import json
import os


class ClaudeConfig:
    """Loader for claude.json configuration.

    Format: ``{"agents": {"agent": "main.py:agent"}}``
    """

    def __init__(self, config_path: str = "claude.json"):
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
                f"Claude configuration file not found at {self.config_path}"
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
                "Missing or invalid 'agents' key in claude.json configuration file. "
                'Expected {"agents": {"name": "file.py:variable"}}.'
            )
        return agents
