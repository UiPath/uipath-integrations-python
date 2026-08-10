"""A tiny stdio MCP server keeping an in-memory expense ledger.

Started as a subprocess by the Claude Code CLI, it speaks MCP over stdin and
stdout for the lifetime of the run. It uses only ``mcp``, which the Claude
Agent SDK already depends on, so the sample needs no extra install.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("ledger")

ENTRIES: dict[str, float] = {}


@server.tool()
def add_entry(label: str, amount: float) -> str:
    """Record one expense in the ledger, replacing any entry with the same label."""
    ENTRIES[label] = amount
    return f"Recorded {label} at {amount:.2f}."


@server.tool()
def list_entries() -> dict[str, float]:
    """Return every recorded expense, keyed by label."""
    return dict(ENTRIES)


@server.tool()
def total() -> float:
    """Return the sum of every recorded expense."""
    return round(sum(ENTRIES.values()), 2)


@server.tool()
def largest_entry() -> str:
    """Return the label of the most expensive recorded entry."""
    if not ENTRIES:
        return "The ledger is empty."
    return max(ENTRIES, key=lambda label: ENTRIES[label])


if __name__ == "__main__":
    server.run()
