"""Currency conversion agent with a deterministic in-process tool."""

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool
from pydantic import BaseModel, Field

from uipath_claude_sdk import UiPathClaudeAgent, UiPathModel

RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.09,
    "GBP": 1.27,
    "RON": 0.22,
    "JPY": 0.0067,
}


@tool(
    "get_exchange_rate",
    "Get the exchange rate from one currency to another (ISO codes, e.g. EUR, USD).",
    {"from_currency": str, "to_currency": str},
)
async def get_exchange_rate(args: dict) -> dict:
    source = RATES_TO_USD.get(args["from_currency"].upper())
    target = RATES_TO_USD.get(args["to_currency"].upper())
    if source is None or target is None:
        return {"content": [{"type": "text", "text": "Unknown currency code."}]}
    return {"content": [{"type": "text", "text": str(source / target)}]}


currency_server = create_sdk_mcp_server(name="currency", tools=[get_exchange_rate])


class ConversionRequest(BaseModel):
    amount: float = Field(description="Amount to convert.")
    from_currency: str = Field(description="ISO code to convert from, e.g. 'EUR'.")
    to_currency: str = Field(description="ISO code to convert to, e.g. 'RON'.")


class ConversionResult(BaseModel):
    converted_amount: float = Field(
        description="Amount expressed in the target currency."
    )
    rate_used: float = Field(description="Rate the tool returned.")
    explanation: str = Field(description="One sentence explaining the conversion.")


agent = UiPathClaudeAgent(
    options=ClaudeAgentOptions(
        system_prompt=(
            "You are a currency conversion assistant. "
            "Always call the get_exchange_rate tool to look up the rate, and "
            "compute the result from the value it returns."
        ),
        max_turns=10,
        permission_mode="bypassPermissions",
        tools=[],
        mcp_servers={"currency": currency_server},
    ),
    uipath_llm=UiPathModel("claude-sonnet-4-5"),
    input_schema=ConversionRequest,
    output_schema=ConversionResult,
    prompt="Convert {amount} {from_currency} to {to_currency}.",
)
