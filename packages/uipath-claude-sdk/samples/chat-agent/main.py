"""Meeting-room assistant that answers one message per invocation."""

from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

ROOMS = [
    {"name": "Aviary", "seats": 4, "floor": 1},
    {"name": "Beehive", "seats": 8, "floor": 2},
    {"name": "Cedar Hall", "seats": 20, "floor": 3},
]


@tool(
    "find_rooms",
    "List the meeting rooms that seat at least the given number of people.",
    {"people": int},
)
async def find_rooms(args: dict[str, Any]) -> dict[str, Any]:
    people = int(args["people"])
    matches = [room for room in ROOMS if int(room["seats"]) >= people]
    if not matches:
        return {
            "content": [
                {"type": "text", "text": f"No room seats {people} people."},
            ]
        }
    listing = "\n".join(
        f"{room['name']}: {room['seats']} seats, floor {room['floor']}"
        for room in matches
    )
    return {"content": [{"type": "text", "text": listing}]}


rooms_server = create_sdk_mcp_server(name="rooms", tools=[find_rooms])


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You help colleagues pick a meeting room. Call find_rooms to see what "
            "seats a given headcount, and suggest the smallest room that fits. "
            "You can only read the room list: you cannot book, hold or cancel "
            "anything, and you know nothing about who is using a room when. Say "
            "so plainly when you are asked for something you cannot do. Answer in "
            "one or two sentences, and use what the person already told you "
            "earlier in the conversation rather than asking for it again."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"rooms": rooms_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
)
