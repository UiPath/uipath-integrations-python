from agent_framework.orchestrations import HandoffBuilder

from uipath_agent_framework.chat import UiPathOpenAIChatClient, requires_approval


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
        "You are a customer support triage agent. Determine what the "
        "customer needs help with and hand off to the right agent:\n"
        "- Billing issues (payments, transfers) -> billing_agent\n"
        "- Returns and refunds -> returns_agent\n"
    ),
)

billing = client.as_agent(
    name="billing_agent",
    description="Handles billing, payments, and fund transfers.",
    instructions=(
        "You are a billing specialist. Help customers with payments "
        "and transfers. Use the transfer_funds tool when needed — "
        "it will require human approval before executing."
    ),
    tools=[transfer_funds],
)

returns = client.as_agent(
    name="returns_agent",
    description="Handles product returns and refund requests.",
    instructions=(
        "You are a returns specialist. Help customers process returns "
        "and issue refunds. Use the issue_refund tool — it will "
        "require human approval before executing."
    ),
    tools=[issue_refund],
)

workflow = (
    HandoffBuilder(
        name="customer_support",
        participants=[triage, billing, returns],
    )
    .with_start_agent(triage)
    .add_handoff(triage, [billing, returns])
    .add_handoff(billing, [returns, triage])
    .add_handoff(returns, [billing, triage])
    .build()
)

agent = workflow.as_agent(name="customer_support")
