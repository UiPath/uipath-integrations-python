"""Integration Service tools: the agent sends a Slack message as the bot.

Two activities of the Slack connector (``uipath-salesforce-slack``) on one
connection, both *immediate* tools (answered as soon as Integration Service
returns; the run never suspends on them):

* ``get_slack_user_by_email`` — **Get User by Email or User Id**
  (``GET /UsersByEmail/{usersByEmailId}``), to turn an email into the Slack
  user id the send activity needs;
* ``send_slack_message_to_user`` — **Send Message to User**
  (``POST /send_message_to_user_v2``), sent as the app bot.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool

#: Id of your Slack connection (Integration Service → Connections → the
#: connection → its id in the URL). It is also the key of the connection
#: binding in bindings.json, so a deployment can rebind it.
CONNECTION_ID = "8e1d0859-a9c0-4d11-b3e0-849a8abeb7b8"
SLACK = "uipath-salesforce-slack"


class Request(BaseModel):
    recipient_email: str = Field(description="Email of the Slack user to message")
    message: str = Field(description="What the message should say")


class Output(BaseModel):
    recipient_user_id: str = Field(description="Slack user id the message went to")
    message_ts: str = Field(description="Slack timestamp (id) of the sent message")
    text: str = Field(description="The text that was sent")


class UserByEmailIn(BaseModel):
    usersByEmailId: str = Field(description="Email address of the Slack user.")


class SendMessageIn(BaseModel):
    channel: str = Field(
        description="Slack user id of the recipient (e.g. 'U02AB3CD4EF'), "
        "as returned by get_slack_user_by_email."
    )
    messageToSend: str = Field(description="Text of the message (Slack mrkdwn).")


get_user = uipath_integration_tool(
    "get_slack_user_by_email",
    connection_id=CONNECTION_ID,
    connector_key=SLACK,
    object_name="UsersByEmail_GET",
    tool_path="/UsersByEmail/{usersByEmailId}",  # {usersByEmailId} → path
    method="GETBYID",
    input_schema=UserByEmailIn,
    # Look the user up by email (the activity also accepts a user id).
    parameters=[
        {"name": "By", "fieldLocation": "query", "fieldVariant": "static", "value": "Email"}
    ],
    description="Finds a Slack user by email. Returns the user, including its id.",
)

send_message = uipath_integration_tool(
    "send_slack_message_to_user",
    connection_id=CONNECTION_ID,
    connector_key=SLACK,
    object_name="send_message_to_user_v2",
    tool_path="/send_message_to_user_v2",
    method="POST",
    input_schema=SendMessageIn,  # channel, messageToSend → JSON body
    # Send as the app bot (the alternative is "user", the connection's user).
    parameters=[
        {"name": "send_as", "fieldLocation": "query", "fieldVariant": "static", "value": "bot"}
    ],
    description="Sends a direct message to one Slack user, as the bot.",
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Send this Slack message to {recipient_email}: {message}\n"
        "First look the user up by email with get_slack_user_by_email, then "
        "send the message exactly as given with send_slack_message_to_user, "
        "using the user's id. Report the recipient id, the message timestamp "
        "and the text that was sent."
    ),
    tools=[get_user, send_message],
)
