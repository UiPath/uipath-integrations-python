#!/bin/bash
set -e

# Runs the shipped template, unmodified, against the evaluation set it ships
# with. Everything the template owns is copied in rather than duplicated here,
# so the testcase fails when the template and its evaluators drift apart.

TEMPLATE_DIR="../../template"

echo "Copying the template's agent and evaluations..."
cp "$TEMPLATE_DIR/main.py" main.py
cp "$TEMPLATE_DIR/input.json" input.json
cp "$TEMPLATE_DIR/uipath.json" uipath.json
cp "$TEMPLATE_DIR/claude.json" claude.json
rm -rf evaluations
cp -r "$TEMPLATE_DIR/evaluations" evaluations

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Running the evaluation set..."
uv run uipath eval --no-report --output-file eval_output.json --trace-file traces.jsonl

echo "Running agent..."
echo "Input from input.json file"
uv run uipath run agent --file input.json
