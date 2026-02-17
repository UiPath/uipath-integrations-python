"""Company Multi-Agent System with Orchestrator and Specialized Agents."""

from agents import Agent
from hr_tools import (
    check_employee_benefits,
    check_leave_request_status,
    check_pto_balance,
    get_salary_info,
    schedule_hr_meeting,
    submit_leave_request,
)
from policy_tools import (
    check_compliance_status,
    get_company_policy,
    request_policy_clarification,
    search_policy_documents,
)
from procurement_tools import (
    check_budget_availability,
    create_purchase_order,
    get_vendor_information,
    request_budget_reallocation,
    search_preferred_vendors,
    track_order_status,
)
from pydantic import BaseModel, Field

# ============= REQUIRED UIPATH STRUCTURE =============


class Input(BaseModel):
    """Input model for the company agent."""

    messages: str = Field(..., description="User messages to send to the agent")


class Output(BaseModel):
    """Output model for the company agent."""

    result: str = Field(..., description="The response to the employee's request")


# ============= SPECIALIZED AGENTS =============

# HR Agent - Handles human resources queries
hr_agent = Agent(
    name="hr_agent",
    instructions="""You are an HR specialist assistant who helps employees with human resources matters.

    Your responsibilities:
    - Check PTO balances and process leave requests
    - Provide employee benefits information
    - Access salary and compensation details
    - Schedule HR meetings for complex issues
    - Answer HR policy questions

    IMPORTANT: You can look up employees by their name OR employee ID. When an employee introduces themselves by name, use their name directly in the tools.

    IMPORTANT WORKFLOW for leave requests:
    1. First, ALWAYS check the employee's PTO balance using check_pto_balance() with their name or ID
    2. Show them how many days they have remaining
    3. If they have sufficient balance, proceed with submit_leave_request()
    4. Provide them with the confirmation details and request ID
    5. Remind them this is pending manager approval

    Be professional, empathetic, and maintain confidentiality.
    When employees introduce themselves by name, acknowledge them personally.""",
    model="gpt-4o-mini",
    tools=[
        check_pto_balance,
        submit_leave_request,
        check_leave_request_status,
        check_employee_benefits,
        get_salary_info,
        schedule_hr_meeting,
    ],
)

# Procurement Agent - Handles purchasing and procurement
procurement_agent = Agent(
    name="procurement_agent",
    instructions="""You are a procurement specialist assistant who helps employees with purchasing and vendor management.

    Your responsibilities:
    - Check budget availability for purchases
    - Create and manage purchase orders
    - Track order status and shipments
    - Provide vendor information
    - Help with budget reallocation requests
    - Recommend preferred vendors

    IMPORTANT WORKFLOW for purchase orders:
    1. Understand the purchase requirement (item, quantity, vendor)
    2. ALWAYS check budget availability FIRST using check_budget_availability()
    3. If budget is approved, proceed to get vendor info if needed
    4. Create the purchase order with detailed information
    5. Provide PO number and approval workflow details

    For orders >$5,000: Remind them Director approval is required.
    For orders >$25,000: VP and Finance approval required.

    Always search for preferred vendors when employees ask about suppliers.
    Be helpful in navigating procurement processes.""",
    model="gpt-4o-mini",
    tools=[
        check_budget_availability,
        get_vendor_information,
        create_purchase_order,
        track_order_status,
        request_budget_reallocation,
        search_preferred_vendors,
    ],
)

# Policy Agent - Handles company policies and compliance
policy_agent = Agent(
    name="policy_agent",
    instructions="""You are a company policy and compliance specialist who helps employees understand company policies.

    Your responsibilities:
    - Provide detailed policy information
    - Check compliance status
    - Answer policy questions and interpretations
    - Help employees search policy documents
    - Submit policy clarification requests

    IMPORTANT APPROACH:
    1. When asked about a policy, use get_company_policy() to retrieve the full policy
    2. Extract and present the relevant sections clearly
    3. If the policy is complex, offer to submit a clarification request
    4. Always cite the policy number and effective date

    Common policies you handle:
    - remote_work: Remote work schedules and requirements
    - expense: Expense reimbursement limits and procedures
    - pto: PTO accrual, usage, and carryover rules
    - travel: Business travel booking and limits
    - security: Information security and data handling
    - code_of_conduct: Ethics and professional behavior

    Be clear, precise, and cite specific policy sections.
    If employees have complex scenarios, help them submit clarification requests.""",
    model="gpt-4o-mini",
    tools=[
        get_company_policy,
        check_compliance_status,
        request_policy_clarification,
        search_policy_documents,
    ],
)

# ============= ORCHESTRATOR AGENT =============

# Orchestrator Agent (Main Entry Point) - Routes to specialized agents
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="""You are a company assistant orchestrator that routes employee requests to the appropriate department specialist.

    Route requests as follows:
    - HR matters (benefits, leave, salary, PTO, employee relations) → handoff to hr_agent
    - Procurement matters (purchases, orders, budgets, vendors, supplies) → handoff to procurement_agent
    - Policy matters (company policies, compliance, regulations, guidelines) → handoff to policy_agent

    IMPORTANT: You MUST immediately analyze the user's request and handoff to the appropriate specialist.
    DO NOT ask clarifying questions or provide a menu of options.
    Read the user's message, determine which department should handle it, and handoff immediately.

    If a request spans multiple areas, choose the primary department that should handle it first.""",
    model="gpt-4o",
    handoffs=[hr_agent, procurement_agent, policy_agent],
)

# Main agent (entry point for UiPath)
agent = orchestrator_agent
