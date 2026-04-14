#!/bin/bash
set -e

TEMPLATE_DIR="../../template"

echo "Copying template files..."
cp "$TEMPLATE_DIR/main.py" main.py
cp "$TEMPLATE_DIR/input.json" input.json
cp "$TEMPLATE_DIR/uipath.json" uipath.json
cp "$TEMPLATE_DIR/llama_index.json" llama_index.json
cp -r "$TEMPLATE_DIR/evaluations" evaluations

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

echo "Initializing the project..."
uv run uipath init

run_agent() {
    local extra_args="$1"
    if uv run uipath run agent --file input.json $extra_args 2>&1; then
        return 0
    else
        if uv run uipath run agent --file input.json $extra_args 2>&1 | grep -q "timed out"; then
            return 1
        fi
        # non-timeout error, fail immediately
        return 2
    fi
}

try_with_fallback() {
    local extra_args="$1"
    echo "Running agent with Bedrock provider..."
    if uv run uipath run agent --file input.json $extra_args; then
        return 0
    fi

    echo "⚠ Bedrock provider timed out or failed, switching to OpenAI fallback..."
    sed -i 's/^llm = UiPathChatBedrockConverse.*/# &/' main.py
    sed -i 's/^# llm = UiPathOpenAI/llm = UiPathOpenAI/' main.py
    uv run uipath init

    if uv run uipath run agent --file input.json $extra_args; then
        return 0
    fi

    echo "⚠ OpenAI provider also failed, trying Vertex fallback..."
    sed -i 's/^llm = UiPathOpenAI.*/# &/' main.py
    sed -i 's/^# llm = UiPathVertex/llm = UiPathVertex/' main.py
    uv run uipath init

    uv run uipath run agent --file input.json $extra_args
}

echo "Running agent..."
try_with_fallback ""

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
try_with_fallback "--trace-file .uipath/traces.jsonl" >> local_run_output.log

echo "Running evaluation..."
uv run uipath eval --no-report --output-file eval_output.json
