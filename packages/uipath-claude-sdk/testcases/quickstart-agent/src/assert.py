import json
import os
import sys

print("Checking quickstart agent output...")

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

    for field in ("converted_amount", "rate_used", "explanation"):
        if field not in output_content:
            print(f"Missing '{field}' field in output")
            sys.exit(1)

    explanation = output_content["explanation"]
    if not explanation or not str(explanation).strip():
        print("Explanation field is empty")
        sys.exit(1)

    # The tool's fixed rates make 100 EUR worth 1.09 / 0.22 * 100 RON. The
    # tolerance absorbs the rounding a model may apply while still failing if
    # the tool was skipped and the number was invented.
    expected_rate = 1.09 / 0.22
    rate_used = float(output_content["rate_used"])
    if abs(rate_used - expected_rate) > 0.02 * expected_rate:
        print(f"Rate {rate_used} does not match the rate the tool returns")
        sys.exit(1)

    expected_amount = 100 * expected_rate
    converted_amount = float(output_content["converted_amount"])
    if abs(converted_amount - expected_amount) > 0.02 * expected_amount:
        print(f"Converted amount {converted_amount} is not close to {expected_amount}")
        sys.exit(1)

    print(f"Converted amount: {converted_amount}")
    print(f"Rate used: {rate_used}")
    print(f"Explanation: {explanation}")

    print("Required fields validation passed")
    print("Quickstart agent working correctly.")

except Exception as e:
    print(f"Error checking output: {e}")
    sys.exit(1)
