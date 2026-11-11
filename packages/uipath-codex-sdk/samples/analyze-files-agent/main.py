"""Analyze Files: the agent answers a question about an attached document.

The input carries a job attachment (an Orchestrator file). The model sees it
as a JobAttachment object (``ID``, ``FullName``, ``MimeType``) and cannot open
it itself: it hands it to ``Analyze_Files``, which downloads the file and asks
the agent's model about it. The output names the document it answered from;
an output attachment field may only name an attachment the run knows.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_analyze_files_tool,
)


class Request(BaseModel):
    document: UiPathAttachment = Field(description="The document to read")
    question: str = Field(description="What to find out about the document")


class Output(BaseModel):
    answer: str = Field(description="The answer, from the document's contents")
    source: UiPathAttachment | None = Field(
        None, description="The document the answer comes from"
    )


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Answer this question about the attached document: {question}\n"
        "Read the document with Analyze_Files (one call, with a precise "
        "analysis task), answer from its result only, and return the document "
        "as the source."
    ),
    tools=[uipath_analyze_files_tool()],
)
