# Generic RAG Agent

A flexible RAG (Retrieval Augmented Generation) agent that combines multiple knowledge bases with web search capabilities and MCP (Model Context Protocol) server integration.

## Features

- **Multiple Knowledge Base Support**: Query multiple context grounding indexes simultaneously
- **Web Search Integration**: Real-time information retrieval via Tavily API
- **MCP Server Integration**: Connect to external MCP servers to access additional tools dynamically
- **ReAct Agent**: Uses ReAct reasoning to plan and execute queries across multiple tools
- **Structured Output**: Generates formatted reports with summary, key findings, analysis, and recommendations

## Prerequisites

- Python 3.10 or higher
- UiPath Orchestrator account with Context Grounding service access
- OpenAI API key (configured in UiPath)
- (Optional) Tavily API key for web search functionality
- (Optional) MCP server URL for accessing additional tools

## Configuration

### Environment Variables

Create a `.env` file in the project directory with the following variables:

```env
# Required - UiPath Configuration
UIPATH_ACCESS_TOKEN=your_uipath_access_token
UIPATH_URL=https://cloud.uipath.com/your_account/your_tenant
UIPATH_TENANT_ID=your_tenant_id
UIPATH_ORGANIZATION_ID=your_organization_id

# Optional - Web Search
TAVILY_API_KEY=your_tavily_api_key

# Optional - MCP Server Integration
UIPATH_MCP_URL=https://your-mcp-server-url
```

**MCP Server Integration:**
When `UIPATH_MCP_URL` is configured, the agent will:
1. Connect to the MCP server at startup
2. Retrieve all available tools from the server
3. Add them to the agent's toolset alongside knowledge base and web search tools
4. Keep the MCP session alive during agent execution so tools can make remote calls

The agent gracefully handles MCP connection failures and continues with base tools only.

### Knowledge Base Configuration

The agent supports multiple knowledge bases configured in [main.py](main.py:21-34):

```python
INDEX_CONFIGS = {
    "primary_knowledge_base": {
        "index_name": "primary_index",
        "folder_path": "sample_data/primary_data",
        "tool_name": "primary_knowledge_base",
        "tool_description": "Primary knowledge base containing company policies, guidelines, and reference information"
    },
    "secondary_knowledge_base": {
        "index_name": "secondary_index",
        "folder_path": "sample_data/secondary_data",
        "tool_name": "secondary_knowledge_base",
        "tool_description": "Secondary knowledge base containing user preferences and personalization data"
    }
}
```

Customize these configurations to match your specific use case.

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

## Usage

### Input Parameters

The agent accepts the following input parameters in `input.json`:

```json
{
  "query": "Your question here",
  "add_data_to_index": false,
  "include_hitl": false
}
```

**Parameters:**
- `query` (string): The question or query to process
- `add_data_to_index` (boolean): Whether to index data before querying (default: false)
- `include_hitl` (boolean): Whether to include Human-in-the-Loop confirmation for the formatted answer (default: false)

### Running the Agent

The agent can be run in different modes:

#### 1. Query Mode (Default)

Query existing knowledge bases without re-indexing:

```json
{
  "query": "What is our remote work policy?",
  "add_data_to_index": false
}
```

#### 2. Index and Query Mode

Add data to indexes before querying:

```json
{
  "query": "What is our remote work policy?",
  "add_data_to_index": true
}
```

#### 3. Multi-Tool Query Mode

Ask questions that leverage multiple tools (knowledge bases, web search, and MCP tools):

```json
{
  "query": "What is the company policy on travel spending? What is the price of GPUs in 2025? What is 5+6+7+8+(6*10)?",
  "add_data_to_index": false
}
```

This query will cause the agent to:
- Query the knowledge base for company policy information
- Use web search (Tavily) for current GPU pricing
- Use MCP math tools (if configured) for calculation

#### 4. Human-in-the-Loop Mode

Enable human confirmation and feedback on the formatted answer:

```json
{
  "query": "What is our remote work policy?",
  "add_data_to_index": false,
  "include_hitl": true
}
```

When `include_hitl` is set to `true`:
- The agent will present the formatted answer and prompt for human confirmation
- You can respond with "yes" to approve the answer or provide feedback to request refinement
- The agent will re-query and reformat up to 3 times based on your feedback
- If `include_hitl` is `false` (default), the agent returns the formatted answer immediately without requesting confirmation

### Workflow Steps

The agent follows this workflow:

1. **Entry Point**: Receives query and routing parameters
2. **Data Indexing** (optional): Adds documents to Context Grounding indexes if `add_data_to_index` is true
3. **Wait for Ingestion**: Ensures indexes are ready before querying
4. **Process Query**:
   - Initializes base tools (knowledge bases and web search)
   - Connects to MCP server (if `UIPATH_MCP_URL` is configured) and retrieves MCP tools
   - Creates ReAct Agent with all available tools
   - Executes the agent to gather and synthesize information
   - Keeps MCP session alive during agent execution
5. **Format Final Answer**: Synthesizes and formats the response into a structured report
6. **Human Confirmation** (optional): If `include_hitl` is true, requests human confirmation and allows iterative refinement based on feedback

## Example Inputs

### Simple Query
```json
{
  "query": "What is our remote work policy?",
  "add_data_to_index": false
}
```

### Complex Multi-Tool Query
```json
{
  "query": "What is our remote work policy and what are the latest AI coding tools available in 2025? Also calculate 15% of 2500.",
  "add_data_to_index": false
}
```

This will leverage:
- **Knowledge bases** for company policy
- **Web search** for current AI tool information
- **MCP tools** (if configured) for percentage calculation

### Query with Human-in-the-Loop
```json
{
  "query": "What is our remote work policy and what are the latest AI coding tools available in 2025?",
  "add_data_to_index": false,
  "include_hitl": true
}
```

This will:
- Process the query using all available tools
- Format the answer into a structured report
- Present the answer and wait for human confirmation
- Allow you to provide feedback if refinement is needed

## Customization

### Adding Custom Tools

Add custom tools by modifying the `generate_tools` function in [main.py](main.py):

```python
def generate_tools(
    response_mode: ResponseMode,
) -> list[QueryEngineTool | FunctionTool]:
    # ... existing tools ...

    # Add your custom tools here
    custom_tool = FunctionTool.from_defaults(
        fn=your_custom_function,
        name="custom_tool_name",
        description="Description of what your tool does"
    )
    tools.append(custom_tool)

    return tools
```

### Connecting to Different MCP Servers

Simply update the `UIPATH_MCP_URL` in your `.env` file to point to a different MCP server. The agent will automatically connect and load all available tools from that server.

### Modifying the System Prompts

Update the agent behavior by modifying the system prompts in the `process_query` step in [main.py](main.py)

### Adjusting Knowledge Bases

Modify the `INDEX_CONFIGS` dictionary in [main.py](main.py) to add, remove, or modify knowledge bases.

## Troubleshooting

### MCP Connection Issues

If MCP tools are not loading:
- Verify `UIPATH_MCP_URL` is correctly set in your `.env` file
- Check that your `UIPATH_ACCESS_TOKEN` is valid and has not expired
- Ensure the MCP server is running and accessible
- Review the console output for MCP connection error messages
- The agent will continue with base tools only if MCP connection fails

### Index Ingestion Timeout

- Increase the retry count or wait time in the `wait_for_index_ingestion` step
- Check Context Grounding service status in UiPath Orchestrator
- Verify your documents are properly formatted

### Web Search Not Working

- Verify `TAVILY_API_KEY` is set in your environment or `.env` file
- Check that the Tavily API key has sufficient credits
- Review error messages in the agent logs

### Agent Not Using MCP Tools

The agent uses the ReAct reasoning pattern and decides which tools to use based on the query. If MCP tools aren't being used:
- Make sure your query requires functionality that the MCP tools provide
- Verify that MCP tool descriptions are clear and relevant
- Check the agent's reasoning output (verbose mode) to see why tools were or weren't selected

### Missing Dependencies

Run the installation command again:
```bash
uv sync
```

## Related Samples

- [simple-remote-mcp-agent](../simple-remote-mcp-agent/): Simple MCP server integration example
- [context-grounding-retriever-agent](../context-grounding-retriever-agent/): Basic context grounding implementation
