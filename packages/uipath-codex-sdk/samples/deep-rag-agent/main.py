"""DeepRAG tool: the agent researches a finance index and cites its sources.

``research_finance`` runs a DeepRAG task on the Context Grounding index
``finance-resources`` (folder ``Coded``, finance PDFs such as "The Fed
Explained"). It is a *durable* tool: the call starts the
task and suspends the run; when the task ends the run resumes and the model
gets ``{"text": ..., "citations": [...], "deepRagId": ...}``.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_deep_rag_tool

#: The Context Grounding index and its folder. They are also the key of the
#: index binding in bindings.json, so a deployment can rebind them.
INDEX_NAME = "finance-resources"
INDEX_FOLDER = "Coded"


class Request(BaseModel):
    question: str = Field(description="Research question about the finance resources")


class Output(BaseModel):
    answer: str = Field(description="Answer synthesized by DeepRAG, with citations")
    sources: list[str] = Field(description="Source documents cited in the answer")


research_finance = uipath_deep_rag_tool(
    "Research Finance Resources",
    tool_name="research_finance",
    index_name=INDEX_NAME,
    folder_path=INDEX_FOLDER,
    citation_mode="Inline",
    description=(
        "Researches the finance resources (documents about the Federal "
        "Reserve, monetary policy, banking supervision and payments) across "
        "all documents and returns a synthesized answer with citations. Takes "
        "a few minutes."
    ),
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Research this question with research_finance: "
        "{question}\n"
        "Call it once, with a clear research task. Answer from its result "
        "only, keep its citations, and list the cited source documents."
    ),
    tools=[research_finance],
)
