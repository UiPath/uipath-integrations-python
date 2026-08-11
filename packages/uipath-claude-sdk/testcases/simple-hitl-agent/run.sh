#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Environment variables:"
echo "UIPATH_JOB_KEY: $UIPATH_JOB_KEY"

echo "Running agent..."
echo "Input from input.json file"
uv run uipath run agent --file input.json

echo "Resuming agent run with the human response..."
uv run uipath run agent --file human_response.json --resume
