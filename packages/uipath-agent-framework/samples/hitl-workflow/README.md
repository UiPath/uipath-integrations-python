# HITL Workflow

A customer support workflow with human-in-the-loop approval for sensitive operations. A triage agent routes requests to billing or returns specialists. Both `transfer_funds` and `issue_refund` tools require human approval before executing.

## Agent Graph

```mermaid
flowchart TB
  __start__(__start__)
  triage(triage)
  billing_agent(billing_agent)
  returns_agent(returns_agent)
  __end__(__end__)
  __start__ --> |input|triage
  triage --> billing_agent
  triage --> returns_agent
  billing_agent --> returns_agent
  billing_agent --> triage
  returns_agent --> billing_agent
  returns_agent --> triage
  billing_agent --> |output|__end__
  returns_agent --> |output|__end__
```

## Prerequisites

Authenticate with UiPath to configure your `.env` file:

```bash
uipath auth
```

## Run

```
uipath run agent '{"messages": [{"contentParts": [{"data": {"inline": "I need a refund for order #12345"}}], "role": "user"}]}'
```

## Debug

```
uipath dev web
```
