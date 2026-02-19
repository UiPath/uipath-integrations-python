from agent_framework.orchestrations import ConcurrentBuilder

from uipath_agent_framework.chat import UiPathOpenAIChatClient

client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

sentiment_agent = client.as_agent(
    name="sentiment",
    description="Analyzes text sentiment.",
    instructions=(
        "You are a sentiment analyst. Analyze the sentiment of the given text. "
        "Return a short assessment: positive/negative/neutral with a confidence "
        "percentage and a one-sentence explanation."
    ),
)

topic_agent = client.as_agent(
    name="topic_extractor",
    description="Extracts main topics and entities from text.",
    instructions=(
        "You are a topic extraction specialist. Identify the main topics, "
        "entities (people, organizations, places), and key themes in the "
        "given text. Return a structured list."
    ),
)

summary_agent = client.as_agent(
    name="summarizer",
    description="Writes concise summaries.",
    instructions=(
        "You are a summarization expert. Write a concise one-paragraph "
        "summary of the given text, capturing the key points."
    ),
)

workflow = ConcurrentBuilder(
    participants=[sentiment_agent, topic_agent, summary_agent],
).build()

agent = workflow.as_agent(name="concurrent_analysis")
