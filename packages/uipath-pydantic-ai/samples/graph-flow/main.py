"""PydanticAI graph-based control flow example using pydantic-graph.

Demonstrates a typed state machine where agents are graph nodes with
compile-time validated edges. A blog post goes through a write-review
loop until the reviewer approves it.

Graph structure:
  WriteDraft → Review → (WriteDraft if needs revision | End if approved)

Key pydantic-graph patterns used:
- BaseNode[StateT] with typed return annotations defining legal edges
- GraphRunContext for shared state across nodes
- End[T] for typed graph termination
- Agent message history stored in state for multi-turn refinement
- Graph.mermaid_code() for automatic diagram generation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from uipath_pydantic_ai.chat import UiPathChatOpenAI

# --- Models ---


class BlogPost(BaseModel):
    """A blog post produced by the writer agent."""

    title: str = Field(description="Blog post title")
    content: str = Field(description="Blog post body text")
    tags: list[str] = Field(description="Relevant tags for the post")


class ReviewFeedback(BaseModel):
    """Reviewer's assessment of a blog post."""

    approved: bool = Field(description="Whether the post is ready to publish")
    feedback: str = Field(description="Feedback for the writer if not approved")
    score: int = Field(description="Quality score from 1-10", ge=1, le=10)


# --- Graph State ---


@dataclass
class BlogState:
    """Shared state across all graph nodes."""

    topic: str
    max_revisions: int = 3
    revision_count: int = 0
    writer_messages: list[ModelMessage] = field(default_factory=list)


# --- Agents ---

MODEL = "gpt-4.1-mini-2025-04-14"
uipath_client = UiPathChatOpenAI(model_name=MODEL)

writer_agent = Agent(
    uipath_client.model,
    name="writer",
    output_type=BlogPost,
    instructions=(
        "You are a skilled blog writer. Write engaging, well-structured blog "
        "posts on the given topic. If feedback is provided, revise your post "
        "to address the reviewer's comments while maintaining quality."
    ),
)

reviewer_agent = Agent(
    uipath_client.model,
    name="reviewer",
    output_type=ReviewFeedback,
    instructions=(
        "You are a strict blog editor. Review the blog post for quality, "
        "accuracy, clarity, and engagement. Score from 1-10. "
        "Only approve posts scoring 8 or higher. "
        "Provide specific, actionable feedback for improvement."
    ),
)


# --- Graph Nodes ---


@dataclass
class WriteDraft(BaseNode[BlogState]):
    """Write or revise a blog post draft."""

    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[BlogState]) -> Review:
        if self.feedback:
            prompt = (
                f"Revise the blog post about: {ctx.state.topic}\n"
                f"Reviewer feedback: {self.feedback}"
            )
        else:
            prompt = f"Write a blog post about: {ctx.state.topic}"

        result = await writer_agent.run(
            prompt,
            message_history=ctx.state.writer_messages,
        )
        ctx.state.writer_messages += result.new_messages()
        ctx.state.revision_count += 1

        return Review(draft=result.output)


@dataclass
class Review(BaseNode[BlogState, None, BlogPost]):
    """Review a blog post draft and decide whether to approve or revise."""

    draft: BlogPost

    async def run(self, ctx: GraphRunContext[BlogState]) -> WriteDraft | End[BlogPost]:
        prompt = format_as_xml({"topic": ctx.state.topic, "blog_post": self.draft})
        result = await reviewer_agent.run(prompt)
        review = result.output

        # Approve if score is high enough or max revisions reached
        if review.approved or ctx.state.revision_count >= ctx.state.max_revisions:
            return End(self.draft)

        # Send back for revision with feedback
        return WriteDraft(feedback=review.feedback)


# --- Build Graph ---

blog_graph = Graph(nodes=(WriteDraft, Review))


# --- Entrypoint Agent ---
# Wrap the graph in a PydanticAI agent so it can be used as a UiPath runtime
# entrypoint. The agent delegates to the graph via a tool.

agent = Agent(
    uipath_client.model,
    name="blog_pipeline",
    output_type=BlogPost,
    instructions=(
        "You are a blog post production pipeline. "
        "Use the write_blog tool to produce a polished blog post on any topic. "
        "Return the final result from the tool directly."
    ),
)


@agent.tool
async def write_blog(ctx, topic: str) -> str:
    """Run the write-review graph loop to produce a polished blog post.

    The graph iterates between a writer and reviewer agent until the post
    is approved or the maximum number of revisions is reached.

    Args:
        ctx: The agent context.
        topic: The topic to write about.

    Returns:
        The final blog post as formatted text.
    """
    state = BlogState(topic=topic)
    result = await blog_graph.run(WriteDraft(), state=state)
    post = result.output

    return (
        f"Title: {post.title}\n"
        f"Tags: {', '.join(post.tags)}\n"
        f"Revisions: {state.revision_count}\n\n"
        f"{post.content}"
    )
