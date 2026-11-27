# Deep Research Agent with LlamaIndex, Web Search, and UiPath Context Grounding

A comprehensive deep research workflow implementation using LlamaIndex workflows that combines web search (via Tavily API) with optional UiPath context grounding services. This implementation follows the deep research pattern with structured planning, execution, and synthesis phases.

## Architecture

The workflow consists of three main agents:

1. **Research Planning Agent** - Creates structured research plans with prioritized questions
2. **Research Executor Agent** - Executes research using web search and optional UiPath context grounding
3. **Synthesis Agent** - Synthesizes findings into comprehensive reports

## Key Features

- **Multi-Agent Workflow**: Orchestrated using LlamaIndex workflows with event-driven architecture
- **Web Search Integration**: Primary research using Tavily API for comprehensive web search
- **Optional UiPath Integration**: Additional context grounding from UiPath enterprise knowledge bases
- **Structured Planning**: Creates detailed research plans with prioritized questions
- **Parallel Execution**: Executes research across multiple knowledge sources simultaneously
- **Source Diversity**: Combines public web information with private enterprise context
- **Comprehensive Synthesis**: Generates structured reports with executive summaries and detailed sections

## Installation

### Production Installation
```bash
uv sync
```

### Development Installation
```bash
# Install with development dependencies
uv sync --extra dev

# Or use make
make install
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for LLM
OPENAI_API_KEY=your_openai_api_key_here

# Required for web search (get free API key from https://tavily.com)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional - for UiPath context grounding
UIPATH_ORCHESTRATOR_URL=https://your-orchestrator.uipath.com
UIPATH_TENANT_NAME=your_tenant_name
UIPATH_CLIENT_ID=your_client_id
UIPATH_CLIENT_SECRET=your_client_secret
```

## Usage

### Command Line Interface

```bash
# Basic research with web search only
python main.py "Impact of AI automation on business processes"

# With additional context
python main.py "AI automation trends" --context "Focus on enterprise applications and ROI"

# Save report to file
python main.py "Machine learning in healthcare" --output report.txt
```

### Programmatic Usage

```python
from deep_research_workflow import DeepResearchWorkflow
from llama_index.core.llms import OpenAI

# Set up LLM
llm = OpenAI(model="gpt-4")

# Create workflow (UiPath integration is optional)
workflow = DeepResearchWorkflow(
    llm=llm,
    query_engines=None  # Will use web search only
)

# Run research
result = await workflow.run(
    topic="Impact of AI automation on business processes",
    context="Focus on enterprise automation and compliance"
)

print(f"Research complete: {result.topic}")
print(f"Executive Summary: {result.executive_summary}")
```

### With UiPath Context Grounding

```python
from main import create_uipath_query_engines, load_config

# Load configuration with UiPath settings
config = load_config()

# Create UiPath query engines (if configured)
query_engines = create_uipath_query_engines(config)

# Create workflow with both web search and UiPath
workflow = DeepResearchWorkflow(
    llm=llm,
    query_engines=query_engines  # Will combine web + UiPath sources
)
```

### Running the Example

```bash
python example_usage.py
```

## UiPath Integration

### Prerequisites

1. UiPath Orchestrator instance
2. UiPath Context Grounding Service
3. Storage buckets with document collections
4. Context grounding indexes

### Configuration

```python
config = {
    "uipath": {
        "orchestrator_url": "https://your-orchestrator.uipath.com",
        "tenant_name": "your_tenant",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "storage_buckets": {
            "company_policy": "bucket_id_1",
            "technical_docs": "bucket_id_2"
        },
        "context_grounding_indexes": {
            "company_policy": "index_id_1",
            "technical_docs": "index_id_2"
        }
    }
}
```

## Workflow Steps

1. **Planning Phase**: 
   - Analyzes research topic and context
   - Generates 5-7 specific research questions
   - Creates methodology and expected report structure
   - Prioritizes questions for execution

2. **Execution Phase**:
   - Executes research questions against UiPath context grounding
   - Retrieves relevant information from multiple knowledge sources
   - Calculates confidence scores for findings

3. **Synthesis Phase**:
   - Synthesizes all research findings
   - Generates executive summary
   - Creates detailed report sections
   - Compiles sources and references

## Output Structure

```python
FinalReport(
    topic="Research Topic",
    executive_summary="2-3 paragraph summary",
    sections={
        "Section 1": "Detailed content...",
        "Section 2": "Detailed content...",
        ...
    },
    sources=["source1", "source2", ...],
    generated_at=datetime.now()
)
```

## Development

```bash
make help       # Show all available commands
make install    # Install with dev dependencies
make format     # Format code with black and isort
make lint       # Run all linters (black, isort, flake8, mypy)
make test       # Run all tests
make clean      # Clean up cache files
make check      # Run format, lint, and test
```


## 📊 Architecture

The Deep Research Agent uses a multi-agent architecture built on LlamaIndex workflows:

```mermaid
graph TB
    %% User Input
    User[👤 User] --> Topic[📝 Research Topic & Context]

    %% Main Workflow
    Topic --> Workflow[🔄 DeepResearchWorkflow]

    %% Workflow Steps
    Workflow --> Plan[📋 Planning Step]
    Plan --> Research[🔍 Research Step]
    Research --> Synthesis[📊 Synthesis Step]

    %% Agents
    Plan --> Planner[🎯 ResearchPlannerAgent]
    Research --> Executor[⚡ ResearchExecutorAgent]
    Synthesis --> Synthesizer[📈 SynthesisAgent]

    %% Data Sources
    Executor --> WebSearch[🌐 Tavily Web Search]
    Executor --> UiPath[🏢 UiPath Context Grounding]

    %% UiPath Components
    UiPath --> Policy[📋 Company Policy Index]
    UiPath --> Tech[🔧 Technical Docs Index]
    UiPath --> KB[📚 Knowledge Base Index]
    UiPath --> Compliance[⚖️ Compliance Index]
    UiPath --> BP[⭐ Best Practices Index]

    %% LLM Integration
    Planner --> LLM[🧠 LLM<br/>OpenAI GPT-4]
    Executor --> LLM
    Synthesizer --> LLM

    %% Data Models
    Planner --> RPlan[📄 ResearchPlan]
    Executor --> RResults[📄 ResearchResult[]]
    Synthesizer --> FReport[📄 FinalReport]

    %% Output
    FReport --> Output[📤 Formatted Report]
    Output --> User

    %% Styling
    classDef userClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef workflowClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agentClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef dataClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef sourceClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef llmClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class User userClass
    class Workflow,Plan,Research,Synthesis workflowClass
    class Planner,Executor,Synthesizer agentClass
    class RPlan,RResults,FReport,Output dataClass
    class WebSearch,UiPath,Policy,Tech,KB,Compliance,BP sourceClass
    class LLM llmClass
```

