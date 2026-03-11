# Literature Chat Agent

An AI assistant using Llamaindex and Tavily search for literature research and recommendations.

## Requirements

- Python 3.11+
- OpenAI API key
- Tavily API key

## Installation

```bash
uv venv -p 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Set your API keys as environment variables in .env

```bash
OPENAI_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
uipath run agent '{"messages": [{"contentParts": [{"mimeType" :"text/plain", "data": {"inline": "Tell me about 1984 by George Orwell ?"}}], "role": "user"}]}'
```
