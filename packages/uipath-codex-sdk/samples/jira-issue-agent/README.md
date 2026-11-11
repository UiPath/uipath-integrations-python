# Jira issue + comment (Integration Service tools)

A Codex agent that creates a Jira issue and then comments on it, through two
curated activities of the *Jira* connector (`uipath-atlassian-jira`), called
as Integration Service (IS) tools:

| Tool | Activity | Model provides | Fixed (hidden from the model) |
| --- | --- | --- | --- |
| `create_jira_issue` | Create Issue — `POST /curated_create_issue` | `fields.summary`, `fields.description` | `fields.project.key = "PC"`, `fields.issuetype.id = "3"` (Task) |
| `add_jira_comment` | Add Comment — `POST /curated_add_comment` | `issueIdOrKey` (query), `body` (comment text) | — |

The model chains the two calls: it passes the key that Create Issue returns to
Add Comment. Change `PROJECT_KEY` / `TASK_ISSUE_TYPE_ID` in `main.py` for
another project; the issue type ids of a project are listed by the connector
lookup `GET /project/<key>/issuetypes`. See
[`integration-tool-agent`](../integration-tool-agent) for how IS tools work
and how to rebind the connection.

**This sample really creates a Jira issue** (and a comment) in the project.

## Prerequisites

* A **Jira** connection with permission to create issues in the project. Put
  its id in `CONNECTION_ID` in `main.py` and as `key` / `defaultValue` in
  `bindings.json`.
* Access to the connection's folder for whoever runs the agent.

```bash
cd samples/jira-issue-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath run agent --file input.json
```

The output is structured: `{issue_key, issue_url, comment_id}`.
