# Slack message (Integration Service tools)

A Codex agent that sends a direct message on Slack **as the bot**, through two
activities of the *Slack* connector (`uipath-salesforce-slack`), called as
Integration Service (IS) tools:

| Tool | Activity | Model provides | Fixed (hidden from the model) |
| --- | --- | --- | --- |
| `get_slack_user_by_email` | Get User by Email or User Id — `GET /UsersByEmail/{usersByEmailId}` | the email | `By=Email` (query) |
| `send_slack_message_to_user` | Send Message to User — `POST /send_message_to_user_v2` | `channel` (user id), `messageToSend` | `send_as=bot` (query) |

The send activity needs a Slack **user id**, not an email, so the model looks
the user up first and passes its id. Both tools answer immediately; the run
never suspends. See [`integration-tool-agent`](../integration-tool-agent) for
how IS tools work, copying tools from a low-code `agent.json`, and rebinding.

**This sample really sends a Slack message.** Change `input.json` to message
yourself.

## Prerequisites

* A **Slack** connection whose app bot can message the recipient (the bot
  must be installed in the workspace). Put its id in `CONNECTION_ID` in
  `main.py` and as `key` / `defaultValue` in `bindings.json`.
* Access to the connection's folder for whoever runs the agent (locally your
  user; deployed, the robot account).

```bash
cd samples/slack-message-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json
```

The output is structured: `{recipient_user_id, message_ts, text}`.

## The trace

```
Agent run - agent
├── LLM call
├── Tool call - get_slack_user_by_email      (toolType Integration)
│   └── get_slack_user_by_email              (integrationTool) → activity_invoke …
├── LLM call
├── Tool call - send_slack_message_to_user   (toolType Integration)
│   └── send_slack_message_to_user           (integrationTool) → activity_invoke …
├── LLM call
└── Agent output
```
