# Outlook email (Integration Service tool)

A Codex agent that writes an email and sends it through the **Send Email**
activity of the *Microsoft Outlook 365* connector
(`uipath-microsoft-outlook365`, `POST /hubs/productivity/send-mail-v2`),
called as an Integration Service (IS) tool.

| Model provides | Fixed (hidden from the model) |
| --- | --- |
| `message.toRecipients`, `message.subject`, `message.body.content` | `saveAsDraft=false` (query), `saveToSentItems=true`, `message.body.contentType="Text"` (body) |

The activity takes `multipart/form-data` with the message as JSON in the
`body` part, which is what `body_structure={"contentType": "multipart",
"jsonBodySection": "body"}` declares. Without `saveAsDraft=false` the activity
only saves a draft. See [`integration-tool-agent`](../integration-tool-agent)
for how IS tools work and how to rebind the connection.

**This sample really sends an email** from the connected mailbox. Change
`input.json` to email yourself.

## Prerequisites

* An **Outlook 365** connection. Put its id in `CONNECTION_ID` in `main.py`
  and as `key` / `defaultValue` in `bindings.json`.
* Access to the connection's folder for whoever runs the agent (locally your
  user; deployed, the robot account).

```bash
cd samples/outlook-email-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json
```

The output is structured: `{to, subject}`.
