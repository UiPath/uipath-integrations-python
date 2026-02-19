from agent_framework.orchestrations import HandoffBuilder

from uipath_agent_framework.chat import UiPathOpenAIChatClient

client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

triage = client.as_agent(
    name="triage",
    description="Routes customer requests to the right specialist.",
    instructions=(
        "You are a customer support triage agent. Greet the customer and "
        "determine what they need help with, then hand off to the right agent:\n"
        "- Billing issues (invoices, payments, charges) -> billing_agent\n"
        "- Technical problems (bugs, errors, setup) -> tech_agent\n"
        "- Returns and refunds -> returns_agent\n\n"
        "Ask clarifying questions if needed before handing off."
    ),
)

billing = client.as_agent(
    name="billing_agent",
    description="Handles billing, invoices, and payment issues.",
    instructions=(
        "You are a billing specialist. Help customers with invoices, "
        "payment issues, and charges. If the issue turns out to be "
        "technical, hand off to tech_agent. If the customer wants to "
        "start a return, hand off to returns_agent."
    ),
)

tech = client.as_agent(
    name="tech_agent",
    description="Handles technical support and troubleshooting.",
    instructions=(
        "You are a technical support specialist. Help customers "
        "troubleshoot issues step by step. If the issue turns out to be "
        "billing-related, hand off to billing_agent."
    ),
)

returns = client.as_agent(
    name="returns_agent",
    description="Handles product returns and refund requests.",
    instructions=(
        "You are a returns specialist. Help customers process returns "
        "and initiate refunds. If they have questions about the refund "
        "amount or payment, hand off to billing_agent."
    ),
)

workflow = (
    HandoffBuilder(
        name="customer_support",
        participants=[triage, billing, tech, returns],
    )
    .with_start_agent(triage)
    .add_handoff(triage, [billing, tech, returns])
    .add_handoff(billing, [tech, returns, triage])
    .add_handoff(tech, [billing, triage])
    .add_handoff(returns, [billing, triage])
    .build()
)

agent = workflow.as_agent(name="customer_support")
