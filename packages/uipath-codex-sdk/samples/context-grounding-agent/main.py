"""Context grounding tool: the agent answers questions from an invoices index.

``search_invoices`` runs a semantic search on the Context Grounding index
``invoices`` (folder ``Coded``), an *immediate* tool (the run never suspends
on it). The model writes the query; the tool returns the best matching chunks
as ``{"documents": [{"metadata": {...}, "page_content": "..."}]}``.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_context_tool

#: The Context Grounding index and its folder. They are also the key of the
#: index binding in bindings.json, so a deployment can rebind them.
INDEX_NAME = "invoices"
INDEX_FOLDER = "Coded"


class Request(BaseModel):
    question: str = Field(description="Question about the invoices")


class Output(BaseModel):
    answer: str = Field(description="Answer, based only on the invoices found")
    invoice_numbers: list[str] = Field(
        description="Numbers of the invoices the answer is based on"
    )
    sources: list[str] = Field(description="Source documents of those invoices")


search_invoices = uipath_context_tool(
    "Search Invoices",
    tool_name="search_invoices",
    index_name=INDEX_NAME,
    folder_path=INDEX_FOLDER,
    description=(
        "Searches the company's invoices (invoice number, customer, total "
        "amount, due date). Returns the most relevant invoice passages."
    ),
    number_of_results=5,
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Request,
    output_schema=Output,
    prompt=(
        "Answer this question about our invoices: {question}\n"
        "Look the invoices up with search_invoices and answer only from what "
        "it returns. Report the invoice numbers and the source documents of "
        "the invoices your answer is based on. If nothing relevant is found, "
        "say so and leave both lists empty."
    ),
    tools=[search_invoices],
)
