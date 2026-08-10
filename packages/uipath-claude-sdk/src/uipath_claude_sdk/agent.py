"""User-facing agent definition for the Claude Agent SDK integration.

Developers export a ``ClaudeAgent`` (or a bare ``ClaudeAgentOptions``) from
their entrypoint module. The UiPath runtime owns the run loop
(``ClaudeSDKClient.query()`` / ``receive_response()``) so it can map SDK
messages to UiPath runtime events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel


@dataclass
class ClaudeAgent:
    """Declarative agent definition executed by the UiPath Claude SDK runtime.

    Args:
        options: Claude Agent SDK options (model, system prompt, tools,
            MCP servers, permission mode, ...). The runtime injects
            execution-scoped fields (cwd, env, output_format, resume).
        input_schema: Optional Pydantic model describing structured input.
            When set, the runtime validates input against it and renders
            ``prompt`` from its fields.
        output_schema: Optional Pydantic model describing structured output.
            When set, the runtime requests native structured output from the
            SDK (``output_format=json_schema``) and returns the parsed result.
        prompt: Optional user-message template rendered with ``str.format``
            from the validated input fields (e.g. ``"Summarize {topic}"``).
        name: Display name used in the runtime graph.
    """

    options: ClaudeAgentOptions = field(default_factory=ClaudeAgentOptions)
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    prompt: str | None = None
    name: str = "agent"
