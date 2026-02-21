from agent_framework.orchestrations import HandoffBuilder

from uipath_agent_framework.chat import UiPathOpenAIChatClient
from uipath_agent_framework.chat.tools import requires_approval


@requires_approval
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts. Requires human approval.

    Args:
        from_account: Source account ID
        to_account: Destination account ID
        amount: Amount to transfer

    Returns:
        Confirmation message
    """
    return f"Transferred ${amount:.2f} from {from_account} to {to_account}"


@requires_approval
def get_customer_order(order_id: str) -> str:
    """Retrieve customer info and order details by order ID. Requires human approval.

    Args:
        order_id: The order ID to look up

    Returns:
        Customer and order information
    """
    return (
        f"Order {order_id}:\n"
        f"  Customer: Jane Doe (jane.doe@example.com)\n"
        f"  Product: Wireless Headphones (SKU-4521)\n"
        f"  Amount: $79.99\n"
        f"  Status: Delivered\n"
        f"  Date: 2026-02-15"
    )


@requires_approval
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. Requires human approval.

    Args:
        order_id: The order ID to refund
        amount: Refund amount
        reason: Reason for the refund

    Returns:
        Confirmation message
    """
    return f"Refund of ${amount:.2f} issued for order {order_id}: {reason}"


client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

triage = client.as_agent(
    name="triage",
    description="Routes customer requests to the right specialist.",
    instructions=(
        "You are a customer support triage agent. "
        "Route the customer to the right agent immediately without asking questions:\n"
        "- Order lookups and customer info -> orders_agent\n"
        "- Billing issues (payments, transfers) -> billing_agent\n"
        "- Returns and refunds -> returns_agent\n"
        "Hand off on the first message. Do not ask clarifying questions."
    ),
)

orders = client.as_agent(
    name="orders_agent",
    description="Looks up customer info and order details.",
    instructions=(
        "You are an order lookup specialist. "
        "When a customer asks about an order, immediately call get_customer_order "
        "with the order ID they provided. Do not ask for additional information."
    ),
    tools=[get_customer_order],
)

billing = client.as_agent(
    name="billing_agent",
    description="Handles billing, payments, and fund transfers.",
    instructions=(
        "You are a billing specialist. "
        "When a customer requests a payment or transfer, immediately call "
        "transfer_funds with the details provided. Do not ask for confirmation "
        "or additional information — the tool has built-in human approval."
    ),
    tools=[transfer_funds],
)

returns = client.as_agent(
    name="returns_agent",
    description="Handles product returns and refund requests.",
    instructions=(
        "You are a returns specialist. "
        "When a customer requests a refund, first call get_customer_order to look up "
        "the order details, then immediately call issue_refund with the order ID, "
        "the full order amount, and the reason the customer gave. "
        "Do not ask the customer for information you can look up. "
        "The tools have built-in human approval so just call them directly."
    ),
    tools=[get_customer_order, issue_refund],
)

workflow = (
    HandoffBuilder(
        name="customer_support",
        participants=[triage, orders, billing, returns],
    )
    .with_start_agent(triage)
    .add_handoff(triage, [orders, billing, returns])
    .add_handoff(orders, [billing, returns, triage])
    .add_handoff(billing, [orders, returns, triage])
    .add_handoff(returns, [orders, billing, triage])
    .build()
)

agent = workflow.as_agent(name="customer_support")
