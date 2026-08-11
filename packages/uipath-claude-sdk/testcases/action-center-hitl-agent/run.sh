#!/bin/bash
set -e

# Tenant prerequisite: this testcase raises a REAL Action Center action, because
# its tool passes a CreateTask to interrupt(). The tenant behind CLIENT_ID needs
# the "generic_escalation_app" action app reachable in the "Shared" folder, and
# the client needs permission to create actions there. Without it the run fails
# while creating the resume trigger ("appName or appKey is required", or an app
# lookup 404), not inside the agent. The other platform models need tenant
# resources this account cannot be assumed to own, so they are covered
# hermetically by tests/test_suspend_resume.py instead.
#
# The resume below answers the agent directly rather than completing the
# action, so every green run leaves one pending action behind in the tenant.

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

echo "Resuming agent run with the approver's answer..."
uv run uipath run agent --file human_response.json --resume
