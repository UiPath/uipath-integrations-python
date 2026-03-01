"""Tests for PydanticAI schema extraction and graph building."""

from pydantic import BaseModel
from pydantic_ai import Agent

from uipath_pydantic_ai.runtime.schema import (
    get_agent_schema,
    get_deps_type,
    get_entrypoints_schema,
    parse_input_to_deps,
)

# ============= TEST MODELS =============


class TranslationOutput(BaseModel):
    """Test output model for translations."""

    original_text: str
    translated_text: str
    target_language: str


class TranslationInput(BaseModel):
    """Test input model for structured translation requests."""

    text: str
    source_language: str
    target_language: str


class ReviewInput(BaseModel):
    """Test input model with messages field."""

    messages: str
    review_type: str
    max_issues: int = 10


# ============= TEST AGENTS =============


# Conversational agent (no deps_type, no output_type)
agent_plain = Agent(
    "test",
    name="test_agent_plain",
)

# Structured output only
agent_with_output = Agent(
    "test",
    name="test_agent_with_output",
    output_type=TranslationOutput,
)

# Structured input only (deps_type)
agent_with_deps = Agent(
    "test",
    name="test_agent_with_deps",
    deps_type=TranslationInput,
)

# Structured input AND output
agent_structured = Agent(
    "test",
    name="test_agent_structured",
    deps_type=TranslationInput,
    output_type=TranslationOutput,
)

# Structured input with messages field
agent_with_messages_deps = Agent(
    "test",
    name="test_agent_messages_deps",
    deps_type=ReviewInput,
)


def get_weather(ctx, location: str) -> str:
    """Get the current weather for a location.

    Args:
        ctx: The agent context.
        location: The city and state.

    Returns:
        Weather information.
    """
    return f"Sunny in {location}"


def get_time(ctx, timezone: str) -> str:
    """Get the current time in a timezone.

    Args:
        ctx: The agent context.
        timezone: The timezone.

    Returns:
        Time information.
    """
    return f"12:00 PM in {timezone}"


# Agent with tools
agent_with_tools = Agent(
    "test",
    name="tools_agent",
    tools=[get_weather, get_time],
)


# ============= STRUCTURED INPUT TESTS =============


def test_deps_type_detection():
    """Test that deps_type is correctly detected."""
    assert get_deps_type(agent_plain) is None
    assert get_deps_type(agent_with_output) is None
    assert get_deps_type(agent_with_deps) is TranslationInput
    assert get_deps_type(agent_structured) is TranslationInput


def test_schema_structured_input():
    """Test that deps_type Pydantic model becomes the input schema."""
    schema = get_entrypoints_schema(agent_with_deps)

    input_schema = schema["input"]
    assert "text" in input_schema["properties"]
    assert "source_language" in input_schema["properties"]
    assert "target_language" in input_schema["properties"]

    # messages should NOT be in the schema — it's the deps model, not conversational
    assert "messages" not in input_schema["properties"]

    # All fields are required (no defaults)
    assert "text" in input_schema["required"]
    assert "source_language" in input_schema["required"]
    assert "target_language" in input_schema["required"]


def test_schema_structured_input_and_output():
    """Test agent with both deps_type and output_type."""
    schema = get_entrypoints_schema(agent_structured)

    # Input is the deps model
    assert "text" in schema["input"]["properties"]
    assert "source_language" in schema["input"]["properties"]
    assert "messages" not in schema["input"]["properties"]

    # Output is the output_type model
    assert "original_text" in schema["output"]["properties"]
    assert "translated_text" in schema["output"]["properties"]
    assert schema["output"].get("title") == "TranslationOutput"


def test_schema_deps_with_messages_field():
    """Test deps model that has a 'messages' field — it stays as a deps field."""
    schema = get_entrypoints_schema(agent_with_messages_deps)

    input_schema = schema["input"]
    # messages is part of the deps model, not the conversational fallback
    assert "messages" in input_schema["properties"]
    assert "review_type" in input_schema["properties"]
    assert "max_issues" in input_schema["properties"]

    assert input_schema.get("title") == "ReviewInput"


def test_parse_input_to_deps():
    """Test parsing a dict into a deps Pydantic model."""
    deps = parse_input_to_deps(
        {"text": "Hello", "source_language": "en", "target_language": "es"},
        TranslationInput,
    )
    assert isinstance(deps, TranslationInput)
    assert deps.text == "Hello"
    assert deps.source_language == "en"
    assert deps.target_language == "es"


def test_parse_input_to_deps_with_defaults():
    """Test deps parsing with default values."""
    deps = parse_input_to_deps(
        {"messages": "Review this", "review_type": "security"},
        ReviewInput,
    )
    assert isinstance(deps, ReviewInput)
    assert deps.messages == "Review this"
    assert deps.review_type == "security"
    assert deps.max_issues == 10


# ============= STRUCTURED OUTPUT TESTS =============


def test_schema_with_output_type():
    """Test that output schema is correctly inferred from agent's output_type."""
    schema = get_entrypoints_schema(agent_with_output)

    # Input should be conversational (no deps_type)
    assert "messages" in schema["input"]["properties"]

    # Output should be the model
    assert "original_text" in schema["output"]["properties"]
    assert "translated_text" in schema["output"]["properties"]
    assert "target_language" in schema["output"]["properties"]

    assert "original_text" in schema["output"]["required"]
    assert schema["output"].get("title") == "TranslationOutput"


def test_schema_fallback_without_output_type():
    """Test that schema falls back to defaults when no output_type."""
    schema = get_entrypoints_schema(agent_plain)

    assert "messages" in schema["input"]["properties"]
    assert "result" in schema["output"]["properties"]


def test_schema_with_plain_agent():
    """Test schema extraction with a plain agent (str output)."""
    schema = get_entrypoints_schema(agent_plain)

    assert "messages" in schema["input"]["properties"]
    assert "result" in schema["output"]["properties"]


# ============= GRAPH TESTS =============


def test_graph_basic_agent():
    """Test graph building for a simple agent without tools."""
    graph = get_agent_schema(agent_plain)

    node_ids = {node.id for node in graph.nodes}

    assert "__start__" in node_ids
    assert "__end__" in node_ids
    assert "test_agent_plain" in node_ids
    assert len(graph.nodes) == 3


def test_graph_agent_with_tools():
    """Test graph building for an agent with tools."""
    graph = get_agent_schema(agent_with_tools)

    node_ids = {node.id for node in graph.nodes}

    assert "__start__" in node_ids
    assert "__end__" in node_ids
    assert "tools_agent" in node_ids
    assert "tools_agent_tools" in node_ids
    assert len(graph.nodes) == 4


def test_graph_node_types():
    """Test that nodes have correct types."""
    graph = get_agent_schema(agent_with_tools)

    node_types = {node.id: node.type for node in graph.nodes}

    assert node_types["__start__"] == "__start__"
    assert node_types["__end__"] == "__end__"
    assert node_types["tools_agent"] == "node"
    assert node_types["tools_agent_tools"] == "tool"


def test_graph_control_edges():
    """Test that control flow edges are correctly created."""
    graph = get_agent_schema(agent_with_tools)

    edges = [(edge.source, edge.target, edge.label) for edge in graph.edges]

    assert ("__start__", "tools_agent", "input") in edges
    assert ("tools_agent", "__end__", "output") in edges


def test_graph_tool_edges():
    """Test that bidirectional tool edges exist."""
    graph = get_agent_schema(agent_with_tools)

    edges = [(edge.source, edge.target, edge.label) for edge in graph.edges]

    assert ("tools_agent", "tools_agent_tools", None) in edges
    assert ("tools_agent_tools", "tools_agent", None) in edges


def test_graph_tools_metadata():
    """Test that tools nodes have correct metadata."""
    graph = get_agent_schema(agent_with_tools)

    node_metadata = {node.id: node.metadata for node in graph.nodes}

    tools_metadata = node_metadata["tools_agent_tools"]
    assert tools_metadata is not None
    assert tools_metadata["tool_count"] == 2
    assert "get_weather" in tools_metadata["tool_names"]
    assert "get_time" in tools_metadata["tool_names"]


def test_graph_edge_count():
    """Test total number of edges for agent with tools."""
    graph = get_agent_schema(agent_with_tools)

    # 2 control edges + 2 tool edges = 4
    assert len(graph.edges) == 4


def test_graph_no_subgraphs():
    """Test that all nodes have None subgraph (flat structure)."""
    graph = get_agent_schema(agent_with_tools)

    for node in graph.nodes:
        assert node.subgraph is None
