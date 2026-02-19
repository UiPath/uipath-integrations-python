# UiPath Agent Framework Integration

Python SDK that enables developers to build and deploy Microsoft Agent Framework agents to the UiPath Cloud Platform.

## Installation

```bash
pip install uipath-agent-framework
```

## Quick Start

1. Create an agent in `main.py`:

```python
from agent_framework.openai import OpenAIChatClient

agent = OpenAIChatClient(model_id="gpt-4o-mini").as_agent(
    name="my_agent",
    instructions="You are a helpful assistant.",
)
```

2. Create `agent_framework.json`:

```json
{
  "agents": {
    "agent": "main.py:agent"
  }
}
```

3. Run with UiPath CLI:

```bash
uipath run agent '{"messages": "Hello!"}'
```
