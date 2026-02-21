from agent_framework.orchestrations import SequentialBuilder

from uipath_agent_framework.chat import UiPathOpenAIChatClient

client = UiPathOpenAIChatClient(model="gpt-5-mini-2025-08-07")

writer = client.as_agent(
    name="writer",
    description="Creates concise, engaging marketing copy.",
    instructions=(
        "You are a concise copywriter. Provide a single, punchy marketing "
        "paragraph based on the prompt. Focus on clarity and impact."
    ),
)

reviewer = client.as_agent(
    name="reviewer",
    description="Reviews and provides constructive feedback on content.",
    instructions=(
        "You are a thoughtful reviewer. Give brief, constructive feedback "
        "on the previous assistant message. Highlight strengths and suggest "
        "one specific improvement."
    ),
)

editor = client.as_agent(
    name="editor",
    description="Polishes content based on feedback into a final version.",
    instructions=(
        "You are a skilled editor. Based on the original content and the "
        "reviewer's feedback, produce a polished final version. Incorporate "
        "the suggested improvements while maintaining the original tone."
    ),
)

workflow = SequentialBuilder(
    participants=[writer, reviewer, editor],
).build()

agent = workflow.as_agent(name="sequential_workflow")
