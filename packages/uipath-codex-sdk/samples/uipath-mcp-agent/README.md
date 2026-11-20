# UiPath-hosted MCP server

A Codex agent that uses the tools of an MCP server hosted in your tenant
(Orchestrator → folder → **MCP Servers**). The sample points at an
`arithmetic-operations-mcp` server with tools such as `add`.

```python
mcp_servers={
    "uipath_mcp": RemoteMcpServer(
        url=MCP_URL,                               # the server's "Copy URL"
        bearer_token_env_var="UIPATH_ACCESS_TOKEN",
        startup_timeout_sec=60,
        tool_timeout_sec=900,
        default_tools_approval_mode="approve",
    )
}
```

* The server authenticates with the run's own UiPath token
  (`UIPATH_ACCESS_TOKEN`), which `uipath auth` writes locally and the robot
  sets for a job.
* UiPath MCP tools carry no read-only annotation, so Codex would ask for
  approval on every call, and `approval_mode="deny_all"` would reject it.
  `default_tools_approval_mode="approve"` pre-approves this trusted server's
  tools.
* `sandbox="read-only"` keeps the model from working around a missing tool
  with shell commands.

## Prerequisites

An MCP server in a tenant folder. Set `MCP_URL` in `main.py` (or the
`UIPATH_MCP_URL` environment variable) to its URL:
`https://<cloud>/<org>/<tenant>/agenthub_/mcp/<folder-key>/<slug>`.

## Run locally

```bash
cd samples/uipath-mcp-agent
uv sync
uv run uipath auth
uv run uipath run agent --file input.json   # {"task": "What is 17 plus 23? Use the MCP tool."}
```

`CODEX_MODEL` overrides the model.
