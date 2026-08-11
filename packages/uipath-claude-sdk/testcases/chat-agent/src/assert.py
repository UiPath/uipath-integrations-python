import json
import os
import sys

print("Checking conversational chat agent output...")

# Check NuGet package
uipath_dir = ".uipath"
if not os.path.exists(uipath_dir):
    print("NuGet package directory (.uipath) not found")
    sys.exit(1)

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith(".nupkg")]
if not nupkg_files:
    print("NuGet package file (.nupkg) not found in .uipath directory")
    sys.exit(1)

print(f"NuGet package found: {nupkg_files[0]}")

# Check agent output file
output_file = "__uipath/output.json"
if not os.path.isfile(output_file):
    print("Agent output file not found")
    sys.exit(1)

print("Agent output file found")

# Every exchange ends suspended on a fresh trigger waiting for the next message,
# so a conversational run never reports success. A terminal status here means
# the runtime stopped waiting and the conversation cannot be continued.
try:
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)
except Exception as e:
    print(f"Error reading output: {e}")
    sys.exit(1)

status = output_data.get("status")
if status != "suspended":
    print(f"The last exchange should have suspended for the next message: {status}")
    sys.exit(1)

print("Agent execution status: suspended")

output_content = output_data.get("output")
if not isinstance(output_content, dict) or len(output_content) != 1:
    print(f"Expected exactly one resume trigger in the output, got: {output_content}")
    sys.exit(1)

print(f"Next-message trigger: {next(iter(output_content))}")

# The conversation itself lives in the CLI's session transcript. Without it a
# later exchange starts a fresh session that has forgotten everything.
transcript_root = "__uipath/claude_home/projects"
if not os.path.isdir(transcript_root):
    print(f"Claude session transcripts not found under {transcript_root}")
    sys.exit(1)

transcripts = [
    os.path.join(root, name)
    for root, _, names in os.walk(transcript_root)
    for name in names
    if name.endswith(".jsonl")
]
if not transcripts:
    print("No Claude session transcript was persisted in the runtime directory")
    sys.exit(1)

session_ids = {os.path.splitext(os.path.basename(p))[0] for p in transcripts}
if len(session_ids) != 1:
    print(f"Three exchanges should share one Claude session, found: {session_ids}")
    sys.exit(1)

print(f"Single Claude session across all exchanges: {next(iter(session_ids))}")

# A working directory that resolves through a symlink is filed under two names,
# so read the fullest copy rather than assuming there is only one.
entries: list[dict] = []
for path in transcripts:
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if len(lines) > len(entries):
        entries = lines

user_messages: list[str] = []
assistant_texts: list[str] = []
tool_calls: list[dict] = []
for entry in entries:
    message = entry.get("message")
    if not isinstance(message, dict):
        continue
    content = message.get("content")
    if entry.get("type") == "user" and isinstance(content, str):
        user_messages.append(content)
    if entry.get("type") != "assistant" or not isinstance(content, list):
        continue
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            assistant_texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(block)

# The message each exchange carries has to reach the model. A resume that
# replaced it with a fixed continuation would leave "Continue." here instead,
# and the run would still look successful.
expected_messages = []
for name in ("input.json", "next_message.json", "third_message.json"):
    with open(name, "r", encoding="utf-8") as f:
        payload = json.load(f)
    parts = payload["messages"][-1]["contentParts"]
    expected_messages.append("".join(part["data"]["inline"] for part in parts))

if len(user_messages) != len(expected_messages):
    print(f"Expected {len(expected_messages)} user turns, found {len(user_messages)}:")
    for text in user_messages:
        print(f"  - {text[:80]}")
    sys.exit(1)

for index, (expected, actual) in enumerate(zip(expected_messages, user_messages), 1):
    if expected != actual:
        print(f"Turn {index} reached the model as {actual!r}, expected {expected!r}")
        sys.exit(1)
    print(f"Turn {index} delivered verbatim: {actual[:60]}...")

# The second turn asks for two facts only the first turn established, so an
# answer holding both proves the session was really resumed rather than
# restarted with the newest message alone.
recall = "\n".join(assistant_texts).lower()
for fact in ("beehive", "kingfisher"):
    if fact not in recall:
        print(f"The agent never recalled '{fact}', so earlier context was lost")
        sys.exit(1)

print("Agent recalled the room and the meeting name from the first turn")

# The third turn adds two people to a headcount only the first turn gave, and
# asks for a fresh lookup, so the tool has to be called again with the sum.
lookups = [call for call in tool_calls if call.get("name", "").endswith("find_rooms")]
if len(lookups) < 2:
    print(f"Expected the room list to be looked up again, got {len(lookups)} calls")
    sys.exit(1)

people = lookups[-1].get("input", {}).get("people")
try:
    people = int(people)
except (TypeError, ValueError):
    print(f"Last find_rooms call had no usable headcount: {lookups[-1].get('input')}")
    sys.exit(1)

if people <= 6:
    print(f"Last find_rooms call used {people}, so the two extra people were lost")
    sys.exit(1)

print(f"Room list looked up again for {people} people")
print("Conversational chat agent working correctly.")
