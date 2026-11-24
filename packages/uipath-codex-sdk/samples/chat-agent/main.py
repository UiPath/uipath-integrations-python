"""Meeting-room assistant: a conversational Codex agent.

One chat exchange per run. The conversation is one Codex thread, which the
runtime resumes for every new message, so the agent remembers what was said
earlier. Nothing in this file is chat-specific: `uipath.json` sets
`runtimeOptions.isConversational`, and the runtime does the rest.
"""

import json
from typing import Any

from uipath_codex_sdk import ToolContext, UiPathCodexAgent, uipath_analyze_files_tool

ROOMS = [
    {"name": "Aviary", "seats": 4, "floor": 1},
    {"name": "Beehive", "seats": 8, "floor": 2},
    {"name": "Cedar Hall", "seats": 20, "floor": 3},
]


class FindRoomsTool:
    """A local, immediate tool: no tenant resource behind it."""

    name = "find_rooms"
    description = "List the meeting rooms that seat at least the given number of people."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "people": {"type": "integer", "minimum": 1, "description": "Headcount"}
        },
        "required": ["people"],
    }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        people = args.get("people")
        if not isinstance(people, int) or people < 1:
            raise ValueError("people must be a positive integer")
        return {"people": people}

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        matches = [room for room in ROOMS if room["seats"] >= args["people"]]
        return json.dumps({"rooms": matches})


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    base_instructions=(
        "You help colleagues pick a meeting room. Call find_rooms to see what "
        "seats a given headcount, and suggest the smallest room that fits. You "
        "can only read the room list: you cannot book, hold or cancel anything, "
        "and you know nothing about who is using a room when. Say so plainly "
        "when you are asked for something you cannot do. When the user attaches "
        "a file (an agenda, a floor plan), read it with Analyze_Files. Answer in "
        "one or two sentences, and use what the person already told you earlier "
        "in the conversation rather than asking for it again."
    ),
    tools=[FindRoomsTool(), uipath_analyze_files_tool()],
)
