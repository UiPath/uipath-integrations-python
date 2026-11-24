"""Action Center escalation: the agent asks a manager to approve an expense.

``escalate_SimpleApprovalApp`` creates an Action Center task with the
``SimpleApprovalApp`` app and **suspends** the run until a manager completes
it. The run then resumes and the model gets the manager's outcome and
comment. A rejection ends the run (``outcome_mapping={"reject": "end"}``).
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_escalation_tool

#: The deployed Action Center app and its folder. They are also the key of
#: the app binding in bindings.json, so a deployment can rebind them.
APP_NAME = "SimpleApprovalApp"
APP_FOLDER = "Coded/SimpleApprovalAgent"
#: Who reviews the expenses (a user name or email in the tenant).
APPROVER = "alexandru.groza@uipath.com"
#: Expenses above this amount need a manager.
LIMIT = 1000


class Request(BaseModel):
    employee: str = Field(description="Who submitted the expense")
    amount: float = Field(description="Amount in EUR")
    reason: str = Field(description="What the expense is for")


class Output(BaseModel):
    approved: bool = Field(description="Whether the expense is approved")
    decided_by: str = Field(description="'agent' or the reviewer's email")
    comment: str = Field(description="The reviewer's comment, if any")


class Review(BaseModel):
    """The app's input: what the manager sees."""

    Content: str = Field(
        description="The expense to approve: employee, amount and reason, in one "
        "or two sentences."
    )


class Decision(BaseModel):
    """The app's output: what the manager fills in."""

    Comment: str = ""


approval = uipath_escalation_tool(
    APP_NAME,
    folder_path=APP_FOLDER,
    input_schema=Review,
    output_schema=Decision,
    title="Expense approval",
    assignee=APPROVER,
    priority="Medium",
    labels=["expenses"],
    outcome_mapping={"reject": "end"},
    description=(
        "Asks a manager to approve an expense in Action Center and waits for "
        "the decision. Use it for every expense above the limit."
    ),
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Expense from {employee}: {amount} EUR for '{reason}'.\n"
        f"Approve expenses up to {LIMIT} EUR yourself. Above that, ask a "
        "manager with escalate_SimpleApprovalApp and report their decision."
    ),
    tools=[approval],
)
