"""Integration Service tools: the agent creates a Jira issue, then comments on it.

Two curated activities of the Jira connector (``uipath-atlassian-jira``) on one
connection, both *immediate* tools (the run never suspends on them):

* ``create_jira_issue`` — **Create Issue** (``POST /curated_create_issue``).
  The project and issue type are fixed by this sample; the model writes the
  summary and description.
* ``add_jira_comment`` — **Add Comment** (``POST /curated_add_comment``), on
  the issue key the create call returned.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool

#: Id of your Jira connection (Integration Service → Connections → the
#: connection → its id in the URL). It is also the key of the connection
#: binding in bindings.json, so a deployment can rebind it.
CONNECTION_ID = "f06551fb-25c2-4a8e-9f48-d55958c12476"
JIRA = "uipath-atlassian-jira"
#: Jira project key (https://uipath.atlassian.net/jira/software/c/projects/PC).
PROJECT_KEY = "PC"
#: Issue type id of "Task" in that project (GET /project/PC/issuetypes).
TASK_ISSUE_TYPE_ID = "3"


class Request(BaseModel):
    summary: str = Field(description="Title of the ticket")
    details: str = Field(description="What the ticket is about")
    comment: str = Field(description="Comment to add once the ticket exists")


class Output(BaseModel):
    issue_key: str = Field(description="Key of the created issue, e.g. PC-1234")
    issue_url: str = Field(description="Browse URL of the created issue")
    comment_id: str = Field(description="Id of the added comment")


class IssueFields(BaseModel):
    summary: str = Field(description="One-line title of the issue.")
    description: str = Field(description="Description of the issue (plain text).")


class CreateIssueIn(BaseModel):
    fields: IssueFields


class AddCommentIn(BaseModel):
    issueIdOrKey: str = Field(description="Key of the issue to comment on, e.g. PC-1234.")
    body: str = Field(description="Text of the comment.")


create_issue = uipath_integration_tool(
    "create_jira_issue",
    connection_id=CONNECTION_ID,
    connector_key=JIRA,
    object_name="curated_create_issue",
    tool_path="/curated_create_issue",
    method="POST",
    input_schema=CreateIssueIn,  # fields.summary / fields.description → JSON body
    # Fixed by the sample, hidden from the model, merged into "fields".
    parameters=[
        {"name": "fields.project.key", "fieldLocation": "body", "fieldVariant": "static", "value": PROJECT_KEY},
        {"name": "fields.issuetype.id", "fieldLocation": "body", "fieldVariant": "static", "value": TASK_ISSUE_TYPE_ID},
    ],
    description=f"Creates a Task in the Jira project {PROJECT_KEY}. Returns the new issue's id and key.",
)

add_comment = uipath_integration_tool(
    "add_jira_comment",
    connection_id=CONNECTION_ID,
    connector_key=JIRA,
    object_name="curated_add_comment",
    tool_path="/curated_add_comment",
    method="POST",
    input_schema=AddCommentIn,
    parameters=[
        # The issue key travels in the query string; "body" is the JSON body.
        {"name": "issueIdOrKey", "fieldLocation": "query"},
        {"name": "body", "fieldLocation": "body"},
    ],
    description="Adds a comment to a Jira issue. Returns the created comment.",
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Create a Jira ticket with create_jira_issue: summary '{summary}', "
        "description '{details}'. Then add this comment to the ticket you "
        "created with add_jira_comment, using its key: '{comment}'. Report "
        "the issue key, its browse URL "
        "(https://uipath.atlassian.net/browse/<key>) and the comment id."
    ),
    tools=[create_issue, add_comment],
)
