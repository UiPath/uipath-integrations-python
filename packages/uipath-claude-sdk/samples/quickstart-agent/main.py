"""Currency conversion agent with a custom in-process tool and structured output."""

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

_RATES_TO_USD = {"USD": 1.0, "EUR": 1.09, "GBP": 1.27, "RON": 0.22, "JPY": 0.0067}


@tool(
    "get_exchange_rate",
    "Get the exchange rate from one currency to another (ISO codes, e.g. EUR, USD).",
    {"from_currency": str, "to_currency": str},
)
async def get_exchange_rate(args: dict) -> dict:
    source = _RATES_TO_USD.get(args["from_currency"].upper())
    target = _RATES_TO_USD.get(args["to_currency"].upper())
    if source is None or target is None:
        return {"content": [{"type": "text", "text": "Unknown currency code."}]}
    return {"content": [{"type": "text", "text": str(source / target)}]}


currency_server = create_sdk_mcp_server(
    name="currency",
    tools=[get_exchange_rate],
)

agent = ClaudeAgentOptions(
    # Bedrock ARN-style model ID, routed through the UiPath LLM Gateway.
    # With ANTHROPIC_API_KEY set, use a direct model ID (e.g. "claude-sonnet-4-5").
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    system_prompt=(
        "You are a currency conversion assistant. "
        "Always use the get_exchange_rate tool to look up rates."
    ),
    max_turns=10,
    permission_mode="bypassPermissions",
    tools=[],
    mcp_servers={"currency": currency_server},
    output_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "converted_amount": {"type": "number"},
                "rate_used": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["converted_amount", "rate_used", "explanation"],
        },
    },
)
