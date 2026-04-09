#!/bin/bash
set -e

SAMPLE_DIR="../../samples/debug-agent"

echo "Copying agent files from debug-agent sample..."
cp "$SAMPLE_DIR/main.py" main.py
cp "$SAMPLE_DIR/llama_index.json" llama_index.json
cp "$SAMPLE_DIR/uipath.json" uipath.json

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

# Clear job key to force Console mode (not SignalR remote debugging)
export UIPATH_JOB_KEY=""

echo "=== Running debug breakpoint tests with pexpect ==="
uv run pytest src/test_debug.py -v -s
