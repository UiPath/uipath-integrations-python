"""Autonomous multi-step Claude Agent SDK agent."""

from claude_agent_sdk import ClaudeAgentOptions

agent = ClaudeAgentOptions(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    system_prompt=(
        "You are an autonomous analyst. Work step by step: draft your notes "
        "in files, refine them, then produce the final answer."
    ),
    max_turns=25,
    permission_mode="bypassPermissions",
    tools=["Read", "Write", "Edit", "Glob", "Grep"],
    output_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "steps_taken": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "steps_taken"],
        },
    },
)
