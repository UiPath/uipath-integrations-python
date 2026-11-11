"""Integration Service tool: the agent writes and sends an email with Outlook 365.

``send_email`` runs the **Send Email** activity of the Microsoft Outlook 365
connector (``uipath-microsoft-outlook365``,
``POST /hubs/productivity/send-mail-v2``) on your connection. The activity
takes ``multipart/form-data``: the message travels as JSON in the ``body``
part. It is an *immediate* tool: the run never suspends on it.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool

#: Id of your Outlook 365 connection (Integration Service → Connections → the
#: connection → its id in the URL). It is also the key of the connection
#: binding in bindings.json, so a deployment can rebind it.
CONNECTION_ID = "b172f2f8-e10f-4a88-b45e-2f7e62c550b2"


class Request(BaseModel):
    recipient_email: str = Field(description="Who to send the email to")
    topic: str = Field(description="What the email should be about")


class Output(BaseModel):
    to: str = Field(description="Recipient(s) the email was sent to")
    subject: str = Field(description="Subject of the sent email")


class EmailBody(BaseModel):
    content: str = Field(description="Body of the email, as plain text.")


class Message(BaseModel):
    toRecipients: str = Field(
        description="Recipient email addresses, separated by commas."
    )
    subject: str = Field(description="Subject of the email.")
    body: EmailBody


class SendEmailIn(BaseModel):
    message: Message


send_email = uipath_integration_tool(
    "send_email",
    connection_id=CONNECTION_ID,
    connector_key="uipath-microsoft-outlook365",
    object_name="send-mail-v2",
    tool_path="/hubs/productivity/send-mail-v2",
    method="POST",
    input_schema=SendEmailIn,
    # The message is sent as JSON in the "body" part of a multipart request.
    body_structure={"contentType": "multipart", "jsonBodySection": "body"},
    parameters=[
        # Send it for real: by default the activity only saves a draft.
        {"name": "saveAsDraft", "fieldLocation": "query", "fieldVariant": "static", "value": False},
        {"name": "saveToSentItems", "fieldLocation": "body", "fieldVariant": "static", "value": True},
        {
            "name": "message.body.contentType",
            "fieldLocation": "body",
            "fieldVariant": "static",
            "value": "Text",
        },
    ],
    description="Sends an email from the connected Outlook 365 mailbox.",
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Write a short, friendly email to {recipient_email} about {topic}, "
        "then send it with send_email. Sign it 'Your UiPath Codex agent'. "
        "Report who it was sent to and its subject."
    ),
    tools=[send_email],
)
