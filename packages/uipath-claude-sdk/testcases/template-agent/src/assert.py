import json
import os
import sys
import zipfile

print("Checking template agent evaluation...")

TOOL_NAME = "mcp__weather__get_weather"
JUDGE_ID = "evaluator-llm-judge-output"
TOOL_EVALUATOR_IDS = (
    "evaluator-tool-call-order",
    "evaluator-tool-call-count",
    "evaluator-tool-call-arguments",
)
EVALUATOR_ERROR_PREFIX = "Exception thrown by evaluator:"

# Check the package
uipath_dir = ".uipath"
if not os.path.exists(uipath_dir):
    print("NuGet package directory (.uipath) not found")
    sys.exit(1)

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith(".nupkg")]
if not nupkg_files:
    print("NuGet package file (.nupkg) not found in .uipath directory")
    sys.exit(1)

print(f"NuGet package found: {nupkg_files[0]}")

with zipfile.ZipFile(os.path.join(uipath_dir, nupkg_files[0])) as nupkg:
    packed_evaluations = [n for n in nupkg.namelist() if "evaluations/" in n]
if packed_evaluations:
    print(f"Evaluations were packed into the nupkg: {packed_evaluations}")
    sys.exit(1)

print("Evaluations are excluded from the nupkg")

# Check the agent's own run
output_file = "__uipath/output.json"
if not os.path.isfile(output_file):
    print("Agent output file not found")
    sys.exit(1)

print("Agent output file found")

try:
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    status = output_data.get("status")
    if status != "successful":
        print(f"Agent execution failed with status: {status}")
        sys.exit(1)

    print("Agent execution status: successful")

    output_content = output_data.get("output")
    if not output_content:
        print("Missing 'output' field in agent response")
        sys.exit(1)

    for field in ("city", "temperature_celsius", "summary"):
        if field not in output_content:
            print(f"Missing '{field}' field in output")
            sys.exit(1)

    # input.json asks for London, for which the tool's table returns 14.0. A
    # different number means the model answered without consulting the tool.
    temperature = float(output_content["temperature_celsius"])
    if abs(temperature - 14.0) > 0.01:
        print(f"Temperature {temperature} is not the 14.0 the tool returns")
        sys.exit(1)

    print(f"City: {output_content['city']}")
    print(f"Temperature: {temperature}")
    print(f"Summary: {output_content['summary']}")

except Exception as e:
    print(f"Error checking output: {e}")
    sys.exit(1)

# Check the evaluation results
eval_output_file = "eval_output.json"
if not os.path.isfile(eval_output_file):
    print("Evaluation output file not found")
    sys.exit(1)

print("Evaluation output file found")

with open(eval_output_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

set_results = eval_data.get("evaluationSetResults") or []
if not set_results:
    print("No evaluation results were produced")
    sys.exit(1)

failed = False
seen_ids = set()

for evaluation in set_results:
    name = evaluation.get("evaluationName")
    run_results = evaluation.get("evaluationRunResults") or []
    if not run_results:
        print(f"Evaluation '{name}' produced no evaluator results")
        sys.exit(1)

    for run_result in run_results:
        evaluator_id = run_result.get("evaluatorId")
        evaluator_name = run_result.get("evaluatorName")
        result = run_result.get("result") or {}
        score = result.get("score")
        details = result.get("details")
        seen_ids.add(evaluator_id)

        errored = isinstance(details, str) and details.startswith(
            EVALUATOR_ERROR_PREFIX
        )

        if evaluator_id in TOOL_EVALUATOR_IDS:
            if errored:
                print(f"{evaluator_name}: evaluator raised: {details[:400]}")
                failed = True
            elif score != 1.0:
                print(f"{evaluator_name}: scored {score}, expected 1.0")
                print(f"  details: {details}")
                failed = True
            else:
                print(f"{evaluator_name}: {score}")

        elif evaluator_id == JUDGE_ID:
            # The judge calls the UiPath LLM gateway, so whether it can run at
            # all depends on the tenant's governance policy for the model. A
            # refusal is not a defect in the agent, so it is reported and not
            # failed on. A score the judge actually computed is failed on.
            if errored:
                print(f"{evaluator_name}: SKIPPED, evaluator raised:")
                print(f"  {details[:400]}")
            elif score is None or score < 0.7:
                print(f"{evaluator_name}: scored {score}, expected at least 0.7")
                print(f"  details: {details}")
                failed = True
            else:
                print(f"{evaluator_name}: {score}")

for expected_id in TOOL_EVALUATOR_IDS + (JUDGE_ID,):
    if expected_id not in seen_ids:
        print(f"Evaluation set never ran '{expected_id}'")
        failed = True

# Check the spans the tool call evaluators read
traces_file = "traces.jsonl"
if not os.path.isfile(traces_file):
    print("Evaluation trace file not found")
    sys.exit(1)

tool_spans = []
with open(traces_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        span = json.loads(line)
        attributes = span.get("attributes")
        if not isinstance(attributes, dict):
            try:
                attributes = json.loads(span.get("Attributes", "{}"))
            except json.JSONDecodeError:
                attributes = {}
        if "tool.name" in attributes:
            tool_spans.append(attributes)

names = [attributes["tool.name"] for attributes in tool_spans]
if TOOL_NAME not in names:
    print(f"No '{TOOL_NAME}' span in the evaluation trace, found: {names}")
    failed = True
else:
    print(f"Tool span found: {TOOL_NAME}")

# tool.id carries a per-invocation tool_use_id, which the evaluators would
# match criteria against instead of the name, scoring every one of them 0.
with_tool_id = [
    attributes["tool.name"] for attributes in tool_spans if "tool.id" in attributes
]
if with_tool_id:
    print(f"Tool spans still carry tool.id: {with_tool_id}")
    failed = True
else:
    print("No tool span carries a tool.id")

if failed:
    print("Template agent evaluation failed.")
    sys.exit(1)

print("Template agent evaluation working correctly.")
