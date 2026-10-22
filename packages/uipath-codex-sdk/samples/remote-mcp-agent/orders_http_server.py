"""A streamable-HTTP MCP server (the 'remote' transport) exposing an order tool.

Run it in its own shell before the agent:
    uv run python orders_http_server.py
It listens on http://127.0.0.1:8931/mcp
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("orders")

_STATUS = {
    "a-1001": "Shipped, arriving 2026-09-10 via the Bucharest hub.",
    "a-1002": "Processing, awaiting stock (backordered 3 days).",
    "a-1003": "Delivered on 2026-09-05, signed for by J. Popescu.",
}


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Return the current fulfillment status for an order id."""
    return _STATUS.get(order_id.strip().lower(), "No such order.")


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8931, json_response=True)
