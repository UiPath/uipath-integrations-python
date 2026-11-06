"""DeepRAG on a job attachment: the agent researches an attached document.

The input carries a job attachment (a PDF or TXT file in Orchestrator). The
model passes it to ``DeepRAG``, a *durable* tool: the call starts a DeepRAG
task on the file (the service indexes it on the fly) and suspends the run;
when the task ends the run resumes and the model gets
``{"text": ..., "citations": [...], "deepRagId": ...}``.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import (
    UiPathAttachment,
    UiPathCodexAgent,
    uipath_deep_rag_attachments_tool,
)


class Request(BaseModel):
    document: UiPathAttachment = Field(description="The document to research (PDF or TXT)")
    question: str = Field(description="What to research in the document")


class Output(BaseModel):
    answer: str = Field(description="Answer synthesized by DeepRAG, with citations")
    sources: list[str] = Field(description="Sources cited in the answer")


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Research this question in the attached document with DeepRAG: "
        "{question}\n"
        "Call it once, with a clear research task. Answer from its result "
        "only, keep its citations, and list the cited sources."
    ),
    tools=[
        uipath_deep_rag_attachments_tool(
            "DeepRAG",
            description=(
                "Researches one attached document (up to 1,000 pages) and "
                "returns a synthesized answer with citations. Takes a few minutes."
            ),
        )
    ],
)
