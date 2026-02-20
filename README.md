# UiPath Agents Runtime Integrations

A collection of Python SDKs that enable developers to build and deploy agents to the UiPath Cloud Platform using different agent frameworks. These packages provide programmatic interaction with UiPath Cloud Platform services and human-in-the-loop (HITL) semantics through Action Center integration.

All packages extend the [UiPath Python SDK](https://github.com/UiPath/uipath-python) and implement the [UiPath Runtime Protocol](https://github.com/UiPath/uipath-runtime-python).

## Integrations

| Framework | Version | Downloads | Links |
|---|---|---|---|
| [Google ADK](https://github.com/google/adk-python) | [![PyPI](https://img.shields.io/pypi/v/uipath-google-adk)](https://pypi.org/project/uipath-google-adk/) | [![Downloads](https://img.shields.io/pypi/dm/uipath-google-adk.svg)](https://pypi.org/project/uipath-google-adk/) | [README](packages/uipath-google-adk/README.md) · [Samples](packages/uipath-google-adk/samples/) |
| [LangChain](https://github.com/langchain-ai/langchain) | [![PyPI](https://img.shields.io/pypi/v/uipath-langchain)](https://pypi.org/project/uipath-langchain/) | [![Downloads](https://img.shields.io/pypi/dm/uipath-langchain.svg)](https://pypi.org/project/uipath-langchain/) | [README](https://github.com/UiPath/uipath-langchain-python#readme) · [Docs](https://uipath.github.io/uipath-python/langchain/quick_start/) · [Samples](https://github.com/UiPath/uipath-langchain-python/tree/main/samples) |
| [LlamaIndex](https://www.llamaindex.ai/) | [![PyPI](https://img.shields.io/pypi/v/uipath-llamaindex)](https://pypi.org/project/uipath-llamaindex/) | [![Downloads](https://img.shields.io/pypi/dm/uipath-llamaindex.svg)](https://pypi.org/project/uipath-llamaindex/) | [README](packages/uipath-llamaindex/README.md) · [Docs](https://uipath.github.io/uipath-python/llamaindex/quick_start/) · [Samples](packages/uipath-llamaindex/samples/) |
| [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | [![PyPI](https://img.shields.io/pypi/v/uipath-agent-framework)](https://pypi.org/project/uipath-agent-framework/) | [![Downloads](https://img.shields.io/pypi/dm/uipath-agent-framework.svg)](https://pypi.org/project/uipath-agent-framework/) | [README](packages/uipath-agent-framework/README.md) · [Samples](packages/uipath-agent-framework/samples/) |
| [OpenAI Agents](https://github.com/openai/openai-agents-python) | [![PyPI](https://img.shields.io/pypi/v/uipath-openai-agents)](https://pypi.org/project/uipath-openai-agents/) | [![Downloads](https://img.shields.io/pypi/dm/uipath-openai-agents.svg)](https://pypi.org/project/uipath-openai-agents/) | [README](packages/uipath-openai-agents/README.md) · [Docs](https://uipath.github.io/uipath-python/openai-agents/quick_start/) · [Samples](packages/uipath-openai-agents/samples/) |


## Structure

This repository is organized as a monorepo with multiple packages:

```
uipath-integrations-python/
└── packages/
    ├── uipath-llamaindex/      # LlamaIndex runtime
    ├── uipath-openai-agents/   # OpenAI Agents runtime
    ├── uipath-google-adk/      # Google ADK runtime
    └── uipath-agent-framework/ # Microsoft Agent Framework runtime
```

## Development

### Tools

Check out [uipath-dev](https://github.com/uipath/uipath-dev-python) - an interactive application for building, testing, and debugging UiPath Python runtimes, agents, and automation scripts.

### Contributions

Please read our [contribution guidelines](https://github.com/UiPath/uipath-integrations-python/blob/main/CONTRIBUTING.md) before submitting a pull request.


