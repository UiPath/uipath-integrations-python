"""Autonomous multi-step Claude Agent SDK agent."""

from claude_agent_sdk import ClaudeAgentOptions

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
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
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
)
