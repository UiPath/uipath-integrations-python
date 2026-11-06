"""Batch Transform on a job attachment: the agent enriches a CSV of reviews.

The input carries a CSV job attachment. The model passes it to
``Enrich_Reviews``, a *durable* tool: the call starts a Batch Transform task
that adds ``sentiment`` and ``topic`` columns to every row, and the run
suspends. When the task ends the run resumes, the result CSV is uploaded as a
job attachment (``output.csv``) and the agent returns it.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_batch_transform_attachments_tool,
)


class Request(BaseModel):
    reviews: UiPathAttachment = Field(description="CSV file of customer reviews")
    task: str = Field(description="What to do with the reviews")


class Output(BaseModel):
    summary: str = Field(description="What the transform did")
    enriched: UiPathAttachment = Field(description="The CSV with the new columns")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "{task}\n"
        "Enrich the attached reviews with Enrich_Reviews (call it once, with a "
        "clear instruction for every row). Return the file it produces as "
        "'enriched' and summarize what was added."
    ),
    tools=[
        uipath_batch_transform_attachments_tool(
            "Enrich Reviews",
            description=(
                "Adds a sentiment and a topic column to every row of a CSV of "
                "customer reviews and returns the new CSV. Takes a few minutes."
            ),
            output_columns=[
                ("sentiment", "positive, neutral or negative"),
                ("topic", "the product area the review is about, in 1-3 words"),
            ],
        )
    ],
)
