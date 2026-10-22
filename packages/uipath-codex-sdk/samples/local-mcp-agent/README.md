# Local MCP server (stdio)

A Codex agent that uses a tool from a local MCP server. `orders_server.py` is
a small stdio MCP server with one tool, `get_order_status`, whose answers only
it knows, so a correct report proves the agent called it.

```python
agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    prompt="Look up the status of order '{order_id}' using the orders tool and report it verbatim.",
    input_schema=OrderQuery,
    sandbox="full-access",
    mcp_servers={
        "orders": StdioMcpServer(command=sys.executable, args=["orders_server.py"]),
    },
)
```

Codex starts the server process itself. `sandbox="full-access"` lets it do
that on a robot, where Codex's own sandbox is not available (see the main
README's "Built-in tools and the sandbox").

## Run locally

```bash
cd samples/local-mcp-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json   # {"order_id": "A-1001"}
```

The output reports "Shipped, arriving 2026-09-10 via the Bucharest hub."

## Evaluate

`evaluations/eval-sets/evaluation-set-default.json` checks the reported
status and the tool call (count, order and arguments):

```bash
uv run uipath eval agent evaluations/eval-sets/evaluation-set-default.json --no-report
```

Tool-call evaluators use the plain tool name, `get_order_status`; the model
sees it as `mcp__orders__get_order_status`. MCP tools are called directly by
Codex, so they cannot be mocked in evals.

## Trace

```
Agent run - agent
├─ LLM call
├─ Tool call - get_order_status   tool.server orders
├─ LLM call
└─ Agent output
```
