import json
import os


class AgentFrameworkConfig:
    """Simple loader for Agent Framework configuration."""

    def __init__(self, config_path: str = "agent_framework.json"):
        self.config_path = config_path
        self._agents: dict[str, str] | None = None

    @property
    def exists(self) -> bool:
        """Check if the configuration file exists."""
        return os.path.exists(self.config_path)

    @property
    def agents(self) -> dict[str, str]:
        """Get agents names -> path mapping from config.

        Returns: A dictionary mapping agent names to their paths.
        """

        if self._agents is None:
            self._agents = self._load_agents()
        return self._agents

    @property
    def entrypoint(self) -> list[str]:
        """Get the entrypoint for the agents runtime.

        Returns: A list representing the entrypoint command.
        """
        return list(self.agents.keys())

    def _load_agents(self) -> dict[str, str]:
        """Load agents from the configuration file."""
        if not self.exists:
            raise FileNotFoundError(
                f"Agent Framework configuration file not found at {self.config_path}"
            )

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                if "agents" not in config:
                    raise ValueError(
                        "Missing 'agents' key in agent_framework.json configuration file."
                    )
                agents = config["agents"]
                if not isinstance(agents, dict):
                    raise ValueError("'agents' must be a dictionary.")
                return agents
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{self.config_path}': {e}") from e
