# Remote MCP server (streamable HTTP)

A Codex agent that uses a tool from an MCP server reached over HTTP.
`orders_http_server.py` serves one tool, `get_order_status`, on
`http://127.0.0.1:8931/mcp`.

```python
mcp_servers={
    "orders": RemoteMcpServer(
        url="http://127.0.0.1:8931/mcp",
        bearer_token_env_var="ORDERS_TOKEN",
        bearer_token="demo-secret-token",
    )
}
```

Codex sends the token as `Authorization: Bearer`. It silently skips a remote
server whose `bearer_token_env_var` is not set; giving a literal
`bearer_token` makes the runtime set the variable for the Codex process. For
a UiPath-hosted MCP server, use `bearer_token_env_var="UIPATH_ACCESS_TOKEN"`
instead (see `samples/uipath-mcp-agent`).

## Run locally

```bash
cd samples/remote-mcp-agent
uv sync
uv run uipath auth
uv run python orders_http_server.py &        # in its own shell
uv run uipath run agent --file input.json   # {"order_id": "A-1003"}
```

The output reports "Delivered on 2026-09-05, signed for by J. Popescu."

The server runs on your machine, so this sample is for local runs; on
Orchestrator, point `url` at a server the robot can reach.
