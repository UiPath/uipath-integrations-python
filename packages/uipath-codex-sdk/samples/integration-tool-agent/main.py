"""Integration Service tool: the agent searches the web through GenAI Activities.

``web_search`` runs the Integration Service activity **Web Search** of the
"UiPath GenAI Activities" connector (``uipath-uipath-airdk``,
``POST /v2/webSearch``) on your connection. It is an *immediate* tool: each
call is answered as soon as Integration Service returns, and the run never
suspends on it.
"""

from pydantic import BaseModel, Field

from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool

#: Id of your "UiPath GenAI Activities" connection (Integration Service →
#: Connections → the connection → its id in the URL). It is also the key of
#: the connection binding in bindings.json, so a deployment can rebind it.
CONNECTION_ID = "43a2061f-18f4-45ae-a411-349b6750c920"


class Topic(BaseModel):
    topic: str = Field(description="What to research")


class WebSearchIn(BaseModel):
    query: str = Field(description="Natural-language web search query.")
    num: int | None = Field(None, description="Number of results (default 10).")


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None


class WebSearchOut(BaseModel):
    """The activity's response. It also shapes a mocked result in evals."""

    formattedResults: str = Field(description="The results, formatted as text")
    results: list[WebResult] = Field(description="The results")


web_search = uipath_integration_tool(
    "web_search",
    connection_id=CONNECTION_ID,
    connector_key="uipath-uipath-airdk",
    object_name="v2::webSearch",
    tool_path="/v2/webSearch",
    method="POST",
    input_schema=WebSearchIn,  # sent as the JSON body (POST)
    output_schema=WebSearchOut,
    # Fixed values, hidden from the model and merged into every call.
    static_args={"provider": "GoogleCustomSearch"},
    parameters=[
        # The connector declares this header with a default the SDK does not
        # apply by itself.
        {
            "name": "X-Elements-Service-Profile",
            "fieldLocation": "header",
            "fieldVariant": "static",
            "value": "high-capacity",
        }
    ],
    description=(
        "Searches the public web. Returns formatted results plus a list of "
        "results with title, url and snippet."
    ),
)


agent = UiPathCodexAgent(
    model="gpt-5.6-terra",
    input_schema=Topic,
    prompt=(
        "Research {topic} with the web_search tool (one or two searches are "
        "enough) and write a short summary that cites the result URLs."
    ),
    tools=[web_search],
)
