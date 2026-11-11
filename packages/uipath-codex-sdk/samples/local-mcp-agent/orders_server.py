"""A tiny stdio MCP server exposing one order-lookup tool.

The statuses are known only to this tool, so a correct report proves the
agent actually called it (rather than guessing).
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
    mcp.run()  # stdio transport by default
