"""PydanticAI programmatic agent hand-off example.

Demonstrates app-orchestrated multi-agent flow where application code
decides which agent runs next (not the LLM). Three specialist agents
handle a customer support pipeline:

  1. classifier_agent  — classifies the request type
  2. billing_agent     — handles billing/payment questions
  3. technical_agent   — handles technical/product questions

Key PydanticAI patterns used:
- Union output types (ClassifiedRequest | Failed) for flow control
- message_history passing for conversational continuity
- Shared RunUsage tracking across all agents
- App code as the orchestrator (not the LLM)
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from uipath_pydantic_ai.chat import UiPathChatOpenAI

# --- Models ---


class ClassifiedRequest(BaseModel):
    """Result of classifying a customer request."""

    category: str = Field(description="'billing' or 'technical'")
    summary: str = Field(description="Brief summary of the request")


class Failed(BaseModel):
    """Unable to classify or handle the request."""

    reason: str = Field(description="Why the request could not be handled")


class SupportResponse(BaseModel):
    """Final response to the customer."""

    answer: str = Field(description="The response to the customer")
    category: str = Field(description="The category that handled the request")
    follow_up_needed: bool = Field(
        default=False, description="Whether follow-up is needed"
    )


# --- Agents ---

MODEL = "gpt-4.1-mini-2025-04-14"
uipath_client = UiPathChatOpenAI(model_name=MODEL)

# Step 1: Classify the request
classifier_agent = Agent(
    uipath_client.model,
    name="classifier",
    output_type=ClassifiedRequest | Failed,
    instructions=(
        "You are a customer support classifier. Analyze the customer's message "
        "and classify it as either 'billing' (payment, invoices, subscriptions, "
        "refunds) or 'technical' (bugs, features, how-to, product questions). "
        "If you cannot classify it, return a Failed response."
    ),
)

# Step 2a: Handle billing requests
billing_agent = Agent(
    uipath_client.model,
    name="billing_agent",
    output_type=SupportResponse,
    instructions=(
        "You are a billing support specialist. Help customers with payment, "
        "invoice, subscription, and refund questions. Be helpful and specific. "
        "Set follow_up_needed=true if the issue requires manual review."
    ),
)

# Step 2b: Handle technical requests
technical_agent = Agent(
    uipath_client.model,
    name="technical_agent",
    output_type=SupportResponse,
    instructions=(
        "You are a technical support specialist. Help customers with bugs, "
        "product features, how-to questions, and technical issues. Provide "
        "clear step-by-step solutions when possible. "
        "Set follow_up_needed=true if escalation to engineering is needed."
    ),
)


# --- Entrypoint: the coordinator is the classifier ---
# The runtime will call this agent. The programmatic hand-off logic
# is in the execute/stream methods of the runtime, but for this sample
# we expose the classifier as the entrypoint and implement the hand-off
# via a wrapper agent that delegates programmatically.

agent = Agent(
    uipath_client.model,
    name="support_coordinator",
    output_type=SupportResponse,
    instructions=(
        "You are a customer support coordinator. "
        "First classify the request, then route to the appropriate specialist:\n"
        "- For billing/payment issues, act as a billing specialist\n"
        "- For technical/product issues, act as a technical specialist\n"
        "Provide a helpful, specific response with the correct category."
    ),
)


@agent.tool
async def classify_and_route(ctx, customer_message: str) -> str:
    """Classify a customer request and route to the appropriate specialist.

    This tool demonstrates programmatic hand-off: the classifier agent runs first,
    then based on its output, the app code decides which specialist agent to invoke.

    Args:
        ctx: The agent context.
        customer_message: The customer's support request.

    Returns:
        The specialist agent's response.
    """
    # Step 1: Classify
    classification = await classifier_agent.run(
        customer_message,
        usage=ctx.usage,
    )

    if isinstance(classification.output, Failed):
        return f"Could not classify request: {classification.output.reason}"

    category = classification.output.category

    # Step 2: Route to specialist based on classification (app decides, not LLM)
    if category == "billing":
        specialist_result = await billing_agent.run(
            customer_message,
            usage=ctx.usage,
        )
    else:
        specialist_result = await technical_agent.run(
            customer_message,
            usage=ctx.usage,
        )

    response = specialist_result.output
    return (
        f"Category: {response.category}\n"
        f"Answer: {response.answer}\n"
        f"Follow-up needed: {response.follow_up_needed}"
    )
