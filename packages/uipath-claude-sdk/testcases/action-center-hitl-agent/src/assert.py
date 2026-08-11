import json
import os
import sys

print("Checking Action Center HITL agent output...")

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

# Without the transcript a resume starts a conversation that forgot the action.
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

print(f"Claude session transcript found: {transcripts[0]}")

deferred = False
created_action = False
for path in transcripts:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "hook_deferred_tool" in content:
        deferred = True
    if "mcp__approvals__request_approval" in content:
        created_action = True

if not deferred:
    print("No deferred tool call in the transcript, the run never really suspended")
    sys.exit(1)

print("Deferred tool call recorded in the session transcript")

# The whole point of this testcase: the suspension carried a CreateTask, so it
# became a real Action Center task rather than a plain API trigger.
if not created_action:
    print("The agent never called mcp__approvals__request_approval")
    sys.exit(1)

print("Action Center interrupt tool call recorded in the session transcript")

# Check status and required fields
try:
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    status = output_data.get("status")
    if status != "successful":
        print(f"Agent execution failed with status: {status}")
        sys.exit(1)

    print("Agent execution status: successful")

    if "output" not in output_data:
        print("Missing 'output' field in agent response")
        sys.exit(1)

    output_content = output_data["output"]

    for field in ("approved", "approved_amount", "summary"):
        if field not in output_content:
            print(f"Missing '{field}' field in output")
            sys.exit(1)

    if output_content["approved"] is not True:
        print("The agent did not act on the approval delivered on resume")
        sys.exit(1)

    approved_amount = float(output_content["approved_amount"])
    expected_amount = 480.0
    if abs(approved_amount - expected_amount) > 0.01:
        print(f"Approved amount {approved_amount} is not {expected_amount}")
        sys.exit(1)

    summary = output_content["summary"]
    if not summary or not str(summary).strip():
        print("Summary field is empty")
        sys.exit(1)

    print(f"Approved: {output_content['approved']}")
    print(f"Approved amount: {approved_amount}")
    print(f"Summary: {summary}")

    print("Required fields validation passed")
    print("Action Center HITL agent working correctly.")

except Exception as e:
    print(f"Error checking output: {e}")
    sys.exit(1)
