"""Context Grounding tools: schemas, query resolution, the search, errors, spans.

No codex here: the tool is called directly, against a fake
``context_grounding`` service or the real SDK ``ContextGroundingService``
with its HTTP layer captured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import uipath_codex_sdk.context_tools as ctx_mod
import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    ToolContext,
    UiPathContextTool,
    UiPathToolError,
    context_tool_from_resource,
    uipath_context_tool,
)
from uipath_codex_sdk.context_tools import RESULT_CAP, _capped_json
from uipath_codex_sdk.tools import is_immediate_tool, tool_type_of

from .durable.fakes import enriched

CTX = ToolContext(call_id="call_1", runtime_id="rt", tool_name="Company_Policies")

#: An agent.json context resource (low-code shape), trimmed of UI-only fields.
POLICY_RESOURCE: dict[str, Any] = {
    "$resourceType": "context",
    "name": "Company Policies",
    "description": "HR and travel policies of the company.",
    "contextType": "index",
    "folderPath": "Shared",
    "indexName": "company-policies",
    "settings": {
        "resultCount": 3,
        "retrievalMode": "Semantic",
        "threshold": 0.5,
        "query": {"variant": "dynamic"},
        "folderPathPrefix": {"variant": "static", "value": "hr"},
        "fileExtension": {"value": "All"},
    },
}


def _result(*items: dict[str, Any], operation_id: str = "op-1") -> Any:
    from uipath.platform.context_grounding import UnifiedQueryResult

    return UnifiedQueryResult.model_validate(
        {
            "semanticResults": {
                "values": list(items),
                "metadata": {"operation_id": operation_id, "strategy": "s"},
            }
        }
    )


def _item(content: str = "Travel is booked via the portal.", **kw: Any) -> dict:
    return {
        "content": content,
        "source": kw.get("source", "travel.pdf"),
        "page_number": kw.get("page_number", 2),
        "score": kw.get("score", 0.83),
        "reference": kw.get("reference", "ref-1"),
    }


class FakeContextGrounding:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result: Any = _result(_item())
        self.error: BaseException | None = None

    async def unified_search_async(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeFolders:
    def __init__(self, keys: dict[str, str] | None = None) -> None:
        self.keys = keys or {}
        self.lookups: list[str] = []

    async def retrieve_key_async(self, *, folder_path: str) -> str | None:
        self.lookups.append(folder_path)
        return self.keys.get(folder_path)

    def retrieve_key(self, *, folder_path: str) -> str | None:
        raise AssertionError(f"blocking folder lookup of {folder_path!r}")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ctx_mod, "_folder_keys", {})
    monkeypatch.delenv("UIPATH_FOLDER_PATH", raising=False)
    monkeypatch.delenv("UIPATH_FOLDER_KEY", raising=False)


@pytest.fixture
def fake(monkeypatch) -> SimpleNamespace:
    client = SimpleNamespace(
        context_grounding=FakeContextGrounding(),
        folders=FakeFolders({"Shared": "fk-shared", "Jobs": "fk-jobs"}),
    )
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    return client


# --------------------------------------------------------------------------
# schemas and validation per query variant
# --------------------------------------------------------------------------
def test_dynamic_query_schema_and_validation():
    tool = uipath_context_tool("Company Policies", index_name="company-policies")
    assert tool.name == "Company_Policies"
    assert is_immediate_tool(tool)
    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search for in the knowledge base",
            }
        },
        "required": ["query"],
    }
    assert tool.validate_args({"query": "travel"}) == {"query": "travel"}
    with pytest.raises(ValueError):
        tool.validate_args({})
    with pytest.raises(ValueError, match="must not be empty"):
        tool.validate_args({"query": "  "})


def test_static_query_takes_no_arguments():
    tool = uipath_context_tool("p", index_name="i", query="travel policy")
    assert tool.input_schema == {"type": "object", "properties": {}}
    assert tool.validate_args({"query": "ignored"}) == {}
    assert tool.resolve_query({}, None) == "travel policy"


def test_argument_query_stays_visible_and_optional():
    tool = uipath_context_tool("p", index_name="i", query_from_input="{{topic}}")
    assert tool.query_from_input == "topic"
    prop = tool.input_schema["properties"]["query"]
    assert "required" not in tool.input_schema
    assert "agent input 'topic'" in prop["description"]
    assert tool.validate_args({}) == {}
    # The agent input wins; the model's query is the fallback.
    assert tool.resolve_query({"query": "model"}, {"topic": "input"}) == "input"
    assert tool.resolve_query({"query": "model"}, {"other": 1}) == "model"
    assert tool.resolve_query({}, {"topic": ""}) is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"index_name": ""}, "index_name is required"),
        ({"index_name": "i", "query": "q", "query_from_input": "t"}, "not both"),
        ({"index_name": "i", "query": " "}, "static query is empty"),
        ({"index_name": "i", "number_of_results": 0}, "at least 1"),
        ({"index_name": "i", "timeout_sec": 0}, "timeout_sec must be positive"),
    ],
)
def test_invalid_configurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        uipath_context_tool("p", **kwargs)


def test_span_attributes_and_tool_type():
    tool = uipath_context_tool("Company Policies", index_name="company-policies")
    assert tool.span_attributes == {
        "index_name": "company-policies",
        "context_retrieval_mode": "Semantic",
        "tool.display_name": "Company Policies",
    }
    assert tool_type_of(tool, tool.kind) == "ContextGrounding"
    assert tool_type_of(SimpleNamespace(kind="integration")) == "Integration"


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
async def test_call_searches_and_returns_langchain_documents(fake):
    tool = uipath_context_tool(
        "Company Policies",
        index_name="company-policies",
        folder_path="Shared",
        number_of_results=4,
        threshold=0.2,
        folder_path_prefix="hr",
        file_extension="pdf",
    )
    text = await tool.call(tool.validate_args({"query": "travel"}), CTX)
    assert json.loads(text) == {
        "documents": [
            {
                "metadata": {
                    "source": "travel.pdf",
                    "search_id": "op-1",
                    "reference": "ref-1",
                    "page_number": 2,
                    "score": 0.83,
                },
                "page_content": "Travel is booked via the portal.",
            }
        ]
    }
    [call] = fake.context_grounding.calls
    scope = call.pop("scope")
    assert (scope.folder, scope.extension) == ("hr", "pdf")
    assert call == {
        "name": "company-policies",
        "query": "travel",
        "search_mode": call["search_mode"],
        "number_of_results": 4,
        "threshold": 0.2,
        "folder_key": "fk-shared",
        "folder_path": None,
        "include_system_indexes": False,
    }
    assert call["search_mode"].value == "Semantic"


async def test_folder_defaults_to_the_job_folder_without_a_lookup(fake, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Jobs")
    monkeypatch.setenv("UIPATH_FOLDER_KEY", "fk-env")
    tool = uipath_context_tool("p", index_name="i")
    await tool.call({"query": "q"}, CTX)
    assert fake.context_grounding.calls[0]["folder_key"] == "fk-env"
    assert fake.folders.lookups == []


async def test_folder_key_is_looked_up_once(fake, monkeypatch):
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Jobs")  # no key in the env
    tool = uipath_context_tool("p", index_name="i")
    await tool.call({"query": "a"}, CTX)
    await tool.call({"query": "b"}, CTX)
    assert fake.folders.lookups == ["Jobs"]
    assert [c["folder_key"] for c in fake.context_grounding.calls] == ["fk-jobs"] * 2


async def test_unknown_folder_is_left_to_the_sdk(fake):
    tool = uipath_context_tool("p", index_name="i", folder_path="Nope")
    await tool.call({"query": "q"}, CTX)
    [call] = fake.context_grounding.calls
    assert (call["folder_key"], call["folder_path"]) == (None, "Nope")


async def test_no_scope_and_argument_values_from_the_agent_input(fake):
    tool = uipath_context_tool(
        "p",
        index_name="i",
        query_from_input="request.topic",
        folder_path_prefix_from_input="area",
        file_extension="All",
    )
    ctx = ToolContext(
        call_id="c",
        runtime_id="rt",
        tool_name="p",
        agent_input={"request": {"topic": "travel"}, "area": "finance"},
    )
    await tool.call({}, ctx)
    [call] = fake.context_grounding.calls
    assert call["query"] == "travel"
    assert call["scope"].folder == "finance" and call["scope"].extension is None
    tool2 = uipath_context_tool("p", index_name="i")
    await tool2.call({"query": "q"}, CTX)
    assert fake.context_grounding.calls[1]["scope"] is None


async def test_missing_query_is_a_tool_error(fake):
    tool = uipath_context_tool("p", index_name="i", query_from_input="topic")
    with pytest.raises(UiPathToolError, match="needs a query"):
        await tool.call({}, CTX)
    assert fake.context_grounding.calls == []


async def test_zero_results_is_not_an_error(fake):
    fake.context_grounding.result = _result()
    tool = uipath_context_tool("p", index_name="i")
    assert json.loads(await tool.call({"query": "q"}, CTX)) == {"documents": []}


def _ingesting() -> BaseException:
    from uipath.platform.errors import IngestionInProgressException

    return IngestionInProgressException(index_name="company-policies")


def _not_found() -> BaseException:
    from uipath.platform.errors import ContextGroundingIndexNotFoundError

    return ContextGroundingIndexNotFoundError("company-policies")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_not_found, "Context grounding index 'company-policies' was not found."),
        (
            _ingesting,
            "Context grounding index 'company-policies' is still being ingested. "
            "Try again later.",
        ),
        (
            lambda: enriched(400, message="bad scope"),
            "Context grounding returned an error for index 'company-policies': "
            "bad scope",
        ),
        (
            lambda: enriched(500, message="down"),
            "Failed to search context index 'company-policies' (HTTP 500): down",
        ),
        (
            lambda: httpx.ReadTimeout("slow"),
            "Context grounding search for tool 'Company_Policies' timed out. "
            "You may retry.",
        ),
    ],
)
async def test_errors_go_back_to_the_model(fake, error, expected):
    fake.context_grounding.error = error()
    tool = uipath_context_tool("Company Policies", index_name="company-policies")
    with pytest.raises(UiPathToolError) as info:
        await tool.call({"query": "q"}, CTX)
    assert str(info.value) == expected


async def test_unexpected_errors_propagate(fake):
    fake.context_grounding.error = RuntimeError("boom")
    tool = uipath_context_tool("p", index_name="i")
    with pytest.raises(RuntimeError):
        await tool.call({"query": "q"}, CTX)


def test_result_cap_drops_whole_documents():
    docs = [{"metadata": {"n": i}, "page_content": "x" * 10_000} for i in range(10)]
    text = _capped_json({"documents": docs})
    out = json.loads(text)
    assert len(text) <= RESULT_CAP
    assert out["truncated"] is True
    assert [d["metadata"]["n"] for d in out["documents"]] == list(
        range(len(out["documents"]))
    )
    huge = _capped_json({"documents": [{"page_content": "y" * 100_000}]})
    assert len(huge) <= RESULT_CAP and json.loads(huge)["truncated"] is True
    small = {"documents": [{"page_content": "z"}]}
    assert json.loads(_capped_json(small)) == small


async def test_mockable_passthrough(fake, monkeypatch):
    seen: list[Any] = []

    def fake_wrap(spec: Any, fn: Any) -> Any:
        seen.append(spec)
        return fn

    import uipath_codex_sdk.runtime.mocking as mocking

    monkeypatch.setattr(mocking, "wrap", fake_wrap)
    tool = uipath_context_tool("Company Policies", index_name="i")
    await tool.call({"query": "q"}, CTX)
    [spec] = seen
    assert spec.name == "Company Policies"
    assert spec.example_calls == []
    assert spec.output_schema["required"] == ["documents"]
    # Outside an eval mocking context the search runs for real.
    assert len(fake.context_grounding.calls) == 1


# --------------------------------------------------------------------------
# against the real SDK service (folder headers, resource overwrites)
# --------------------------------------------------------------------------
class CapturedContextGrounding:
    """A real ``ContextGroundingService`` whose HTTP calls are captured."""

    def __init__(self) -> None:
        from uipath.platform.common._config import UiPathApiConfig
        from uipath.platform.common._execution_context import UiPathExecutionContext
        from uipath.platform.context_grounding._context_grounding_service import (
            ContextGroundingService,
        )

        self.requests: list[dict[str, Any]] = []
        self.folders = FakeFolders({"Shared": "fk-shared", "Other": "fk-other"})
        self.context_grounding = ContextGroundingService(
            UiPathApiConfig(base_url="https://cloud.test/org/tenant", secret="s"),
            UiPathExecutionContext(),
            folders_service=self.folders,  # type: ignore[arg-type]
            buckets_service=None,  # type: ignore[arg-type]
        )
        self.context_grounding.request_async = self._request  # type: ignore[method-assign, assignment]

    async def _request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        url = str(url)
        self.requests.append({"method": method, "url": url, **kwargs})
        request = httpx.Request(method, "https://cloud.test" + url)
        if url.startswith("/ecs_/v2/indexes"):
            name = kwargs["params"]["$filter"].split("'")[1]
            body: Any = {
                "value": [
                    {
                        "id": f"id-{name}",
                        "name": name,
                        "lastIngestionStatus": "Successful",
                    }
                ]
            }
        else:
            body = {
                "semanticResults": {
                    "values": [_item("Ünïcode policy")],
                    "metadata": {"operation_id": "op-9", "strategy": "s"},
                }
            }
        return httpx.Response(200, json=body, request=request)


async def test_call_through_the_real_service(monkeypatch):
    sdk = CapturedContextGrounding()
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: sdk)
    tool = context_tool_from_resource(POLICY_RESOURCE)
    assert isinstance(tool, UiPathContextTool)
    tool.folder_path = "Shared"
    text = await tool.call({"query": "travel"}, CTX)
    assert "Ünïcode policy" in text  # not ASCII-escaped
    assert json.loads(text)["documents"][0]["metadata"]["search_id"] == "op-9"
    retrieve, search = sdk.requests
    assert retrieve["url"] == "/ecs_/v2/indexes"
    assert "fk-shared" in retrieve["headers"].values()
    assert search["url"] == "/ecs_/v1.2/search/id-company-policies"
    assert search["json"] == {
        "searchMode": "Semantic",
        "query": "travel",
        "semanticSearchOptions": {"numberOfResults": 3, "threshold": 0.5},
        "scope": {"folder": "hr"},
    }
    assert "fk-shared" in search["headers"].values()
    assert sdk.folders.lookups == ["Shared"]  # async, never the blocking one


async def test_resource_overwrite_rebinds_the_index(monkeypatch):
    from uipath.platform.common import ResourceOverwritesContext
    from uipath.platform.common._bindings import GenericResourceOverwrite

    sdk = CapturedContextGrounding()
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: sdk)
    tool = uipath_context_tool("p", index_name="company-policies", folder_path="Shared")

    async def overwrites():
        return {
            "index.company-policies.Shared": GenericResourceOverwrite.model_validate(
                {
                    "resource_type": "index",
                    "name": "prod-policies",
                    "folderPath": "Other",
                }
            )
        }

    async with ResourceOverwritesContext(overwrites):
        await tool.call({"query": "q"}, CTX)
    retrieve, search = sdk.requests
    assert "'prod-policies'" in retrieve["params"]["$filter"]
    assert "fk-other" in retrieve["headers"].values()
    assert search["url"] == "/ecs_/v1.2/search/id-prod-policies"
    assert sdk.folders.lookups == ["Other"]
    assert tool.index_name == "company-policies"  # never pre-resolved


# --------------------------------------------------------------------------
# agent.json resources
# --------------------------------------------------------------------------
def test_from_resource_maps_the_settings():
    tool = context_tool_from_resource(POLICY_RESOURCE)
    assert isinstance(tool, UiPathContextTool)
    assert tool.name == "Company_Policies"
    assert tool.display_name == "Company Policies"
    assert tool.description == "HR and travel policies of the company."
    assert (tool.index_name, tool.number_of_results, tool.threshold) == (
        "company-policies",
        3,
        0.5,
    )
    assert tool.folder_path is None  # folderPath is not used, as in langchain
    assert tool.folder_path_prefix == "hr" and tool.file_extension is None
    assert tool.query is None and tool.query_from_input is None
    assert tool.retrieval_mode == "Semantic"


def _with(**settings: Any) -> dict[str, Any]:
    r = json.loads(json.dumps(POLICY_RESOURCE))
    r["settings"].update(settings)
    return r


def test_from_resource_query_variants():
    static = context_tool_from_resource(
        _with(query={"variant": "static", "value": "leave policy"})
    )
    assert static.query == "leave policy" and static.input_schema["properties"] == {}
    argument = context_tool_from_resource(
        _with(query={"variant": "argument", "value": "{{topic}}"})
    )
    assert argument.query_from_input == "topic"
    no_query = json.loads(json.dumps(POLICY_RESOURCE))
    del no_query["settings"]["query"]
    assert context_tool_from_resource(no_query).input_schema["required"] == ["query"]
    builder = context_tool_from_resource(
        _with(query={"variant": "textBuilder", "value": "x"})
    )
    assert builder.query is None and builder.query_from_input is None
    prefix = context_tool_from_resource(
        _with(folderPathPrefix={"variant": "argument", "value": "{{area}}"})
    )
    assert prefix.folder_path_prefix is None
    assert prefix.folder_path_prefix_from_input == "area"
    structured = context_tool_from_resource(_with(retrievalMode="Structured"))
    assert structured.retrieval_mode == "Structured"
    assert structured.span_attributes["context_retrieval_mode"] == "Structured"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            {"settings": {**POLICY_RESOURCE["settings"], "retrievalMode": "DeepRAG"}},
            "DeepRAG needs 'settings.citationMode'",
        ),
        (
            {
                "settings": {
                    **POLICY_RESOURCE["settings"],
                    "retrievalMode": "BatchTransform",
                }
            },
            "not supported yet",
        ),
        (
            {"settings": {**POLICY_RESOURCE["settings"], "retrievalMode": "Unknown"}},
            "not supported yet",
        ),
        ({"contextType": "attachments"}, "context type 'attachments'"),
        ({"indexName": None}, "has no 'indexName'"),
        ({"settings": None}, "has no 'settings'"),
        (
            {"settings": {**POLICY_RESOURCE["settings"], "query": {"variant": None}}},
            "the query has no variant",
        ),
        (
            {
                "settings": {
                    **POLICY_RESOURCE["settings"],
                    "query": {"variant": "static"},
                }
            },
            "static query has no value",
        ),
        ({"isEnabled": False}, "is disabled"),
    ],
)
def test_from_resource_refusals(change, match):
    with pytest.raises(ValueError, match=match):
        context_tool_from_resource({**POLICY_RESOURCE, **change})


def test_from_resource_model_instance_and_overrides():
    from uipath.agent.models.agent import AgentContextResourceConfig

    model = AgentContextResourceConfig.model_validate(POLICY_RESOURCE)
    tool = context_tool_from_resource(model, tool_name="policies", timeout_sec=9)
    assert isinstance(tool, UiPathContextTool)
    assert tool.name == "policies" and tool.timeout_sec == 9


# --------------------------------------------------------------------------
# DeepRAG (durable)
# --------------------------------------------------------------------------
def _deep_rag(status: str = "Successful", **extra: Any) -> Any:
    from uipath.platform.context_grounding import DeepRagResponse

    data: dict[str, Any] = {
        "id": "dr-1",
        "name": "task-1",
        "createdDate": "2026-10-13T00:00:00Z",
        "lastDeepRagStatus": status,
        "content": None,
        **extra,
    }
    if status == "Successful" and data["content"] is None:
        data["content"] = {
            "text": "Owes $4,200 [1].",
            "citations": [
                {"ordinal": 1, "pageNumber": 2, "source": "a.pdf", "reference": "r"}
            ],
        }
    return DeepRagResponse.model_validate(data)


def test_deep_rag_tool_shape():
    from uipath_codex_sdk import UiPathDeepRagTool, uipath_deep_rag_tool
    from uipath_codex_sdk.tools import is_durable_tool

    tool = uipath_deep_rag_tool("Invoice Research", index_name="invoices")
    assert isinstance(tool, UiPathDeepRagTool) and is_durable_tool(tool)
    assert tool.name == "Invoice_Research"
    assert tool.citation_mode.value == "Inline"
    assert tool.input_schema["required"] == ["query"]
    assert (
        "what to research across documents"
        in (tool.input_schema["properties"]["query"]["description"])
    )
    assert tool_type_of(tool, tool.kind) == "ContextGrounding"
    assert tool.span_attributes["context_retrieval_mode"] == "DeepRAG"
    assert tool.validate_args({"query": "q"}) == {"query": "q"}
    with pytest.raises(ValueError, match="non-empty"):
        tool.validate_args({"query": " "})
    with pytest.raises(ValueError, match="longer than"):
        tool.validate_args({"query": "x" * 250_001})
    static = uipath_deep_rag_tool("r", index_name="i", query="Summarize")
    assert static.input_schema == {"type": "object", "properties": {}}
    assert static.validate_args({"query": "ignored"}) == {}
    skip = uipath_deep_rag_tool("r", index_name="i", citation_mode="Skip")
    assert skip.citation_mode.value == "Skip"
    with pytest.raises(ValueError, match="citation_mode"):
        uipath_deep_rag_tool("r", index_name="i", citation_mode="Footnotes")


async def test_deep_rag_finish_maps_every_outcome():
    from uipath_codex_sdk import UiPathDurableToolPending, uipath_deep_rag_tool

    tool = uipath_deep_rag_tool("r", index_name="i")
    ok = json.loads(await tool.finish({}, _deep_rag()))
    assert ok == {
        "text": "Owes $4,200 [1].",
        "citations": [
            {"ordinal": 1, "page_number": 2, "source": "a.pdf", "reference": "r"}
        ],
        "deepRagId": "dr-1",
    }
    # The serialized form (a resume value read back from storage) too.
    as_dict = _deep_rag().model_dump(by_alias=True)
    assert json.loads(await tool.finish({}, as_dict)) == ok
    failed = json.loads(
        await tool.finish({}, _deep_rag("Failed", failureReason="boom"))
    )
    assert failed == {"status": "failed", "deepRagId": "dr-1", "error": "boom"}
    for status in ("Queued", "InProgress"):
        with pytest.raises(UiPathDurableToolPending):
            await tool.finish({}, _deep_rag(status))
    assert await tool.finish({}, {"answer": "typed by hand"}) == json.dumps(
        {"answer": "typed by hand"}
    )
    assert await tool.finish({}, "plain") == "plain"


async def test_deep_rag_start_resolves_query_glob_and_folder(monkeypatch):
    from uipath.platform.common import WaitDeepRagRaw
    from uipath.platform.context_grounding import DeepRagCreationResponse

    from uipath_codex_sdk import DurableToolContext, uipath_deep_rag_tool

    starts: list[dict[str, Any]] = []

    class Cg:
        async def start_deep_rag_async(self, **kwargs: Any) -> Any:
            starts.append(kwargs)
            return DeepRagCreationResponse.model_validate(
                {"id": "dr-9", "lastDeepRagStatus": "Queued", "createdDate": "x"}
            )

    client = SimpleNamespace(
        context_grounding=Cg(), folders=FakeFolders({"Coded": "fk-c"})
    )
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    tool = uipath_deep_rag_tool(
        "r",
        index_name="invoices",
        folder_path="Coded",
        query_from_input="question",
        folder_path_prefix_from_input="area",
        file_extension="PDF",
    )
    ctx = DurableToolContext(
        call_id="c1",
        runtime_id="rt",
        tool_name="r",
        agent_input={"question": "From the input", "area": "/finance/"},
    )
    value = await tool.start({"query": "from the model"}, ctx)
    assert isinstance(value, WaitDeepRagRaw)
    assert (value.deep_rag.id, value.index_folder_key) == ("dr-9", "fk-c")
    assert value.index_folder_path is None
    [kw] = starts
    assert kw["prompt"] == "From the input"
    assert kw["glob_pattern"] == "finance/*.pdf"
    assert kw["citation_mode"].value == "Inline"
    assert (kw["folder_key"], kw["folder_path"]) == ("fk-c", None)
    assert tool.started_output(value) == {
        "status": "started",
        "deep_rag_id": "dr-9",
        "deep_rag_status": "Queued",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_not_found, "Context grounding index 'invoices' was not found."),
        (_ingesting, "is still being ingested"),
        (lambda: enriched(400, message="bad glob"), "(HTTP 400): bad glob"),
    ],
)
async def test_deep_rag_start_errors_are_durable_tool_errors(
    monkeypatch, error, expected
):
    from uipath_codex_sdk import (
        DurableToolContext,
        UiPathDurableToolError,
        uipath_deep_rag_tool,
    )

    class Cg:
        async def start_deep_rag_async(self, **kwargs: Any) -> Any:
            raise error()

    client = SimpleNamespace(context_grounding=Cg(), folders=FakeFolders())
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    tool = uipath_deep_rag_tool("r", index_name="invoices")
    with pytest.raises(UiPathDurableToolError) as info:
        await tool.start({"query": "q"}, DurableToolContext("c", "rt", "r"))
    assert expected in str(info.value)


@pytest.mark.parametrize(
    ("prefix", "ext", "glob"),
    [
        (None, None, "**/*"),
        (None, "pdf", "**/*.pdf"),
        ("hr", None, "hr/*"),
        ("/hr/2026/**", "TXT", "hr/2026/*.txt"),
        ("/", None, "**/*"),
    ],
)
def test_build_glob_pattern_matches_langchain(prefix, ext, glob):
    from uipath_codex_sdk.context_tools import build_glob_pattern

    assert build_glob_pattern(prefix, ext) == glob


def test_from_resource_builds_a_deep_rag_tool():
    from uipath_codex_sdk import UiPathDeepRagTool

    tool = context_tool_from_resource(
        _with(retrievalMode="DeepRAG", citationMode={"value": "Skip"})
    )
    assert isinstance(tool, UiPathDeepRagTool)
    assert tool.citation_mode.value == "Skip"
    assert tool.index_name == "company-policies"
    assert tool.folder_path_prefix == "hr"
    with pytest.raises(ValueError, match="needs 'settings.citationMode'"):
        context_tool_from_resource(_with(retrievalMode="DeepRAG"))


async def test_rebuilt_deep_rag_trigger_reads_without_registering_again(monkeypatch):
    import uipath.platform.resume_triggers._protocol as protocol
    from uipath.core.triggers import UiPathResumeTriggerName
    from uipath.platform.common import WaitDeepRagRaw
    from uipath.platform.context_grounding import DeepRagCreationResponse
    from uipath.platform.resume_triggers import UiPathResumeTriggerHandler

    from uipath_codex_sdk.runtime.runtime import _rebuilt_trigger
    from uipath_codex_sdk.runtime.session_store import rebuild_suspend_value

    value = WaitDeepRagRaw(
        deep_rag=DeepRagCreationResponse.model_validate(
            {"id": "dr-1", "lastDeepRagStatus": "Queued", "createdDate": "x"}
        ),
        index_folder_key="fk",
    )
    # The value survives the state store as its own type.
    stored = rebuild_suspend_value(
        "WaitDeepRagRaw", json.loads(value.model_dump_json())
    )
    assert isinstance(stored, WaitDeepRagRaw)

    class Cg:
        async def retrieve_deep_rag_async(self, id: str, **kwargs: Any) -> Any:
            assert id == "dr-1"
            return _deep_rag()

    monkeypatch.setattr(
        protocol, "UiPath", lambda: SimpleNamespace(context_grounding=Cg())
    )
    handler = UiPathResumeTriggerHandler()

    async def no_create(*args: Any, **kwargs: Any):
        raise AssertionError("create_trigger would register an external trigger")

    monkeypatch.setattr(handler, "create_trigger", no_create)
    trigger = await _rebuilt_trigger(handler, stored)
    assert trigger is not None
    assert trigger.trigger_name == UiPathResumeTriggerName.DEEP_RAG_RAW
    assert (trigger.item_key, trigger.folder_key) == ("dr-1", "fk")
    read = await handler.read_trigger(trigger)
    assert read is not None
    assert read.id == "dr-1" and read.content.text == "Owes $4,200 [1]."
    assert await _rebuilt_trigger(handler, {"not": "a model"}) is None


def test_hold_span_outside_a_durable_start_ends_the_span():
    from opentelemetry.sdk.trace import TracerProvider

    from uipath_codex_sdk.runtime.telemetry import (
        collect_held_spans,
        held_spans,
        hold_span,
    )

    tracer = TracerProvider().get_tracer("t")
    loose = tracer.start_span("loose")
    assert hold_span(loose) is False and not loose.is_recording()
    token = collect_held_spans()
    held = tracer.start_span("held")
    assert hold_span(held) is True and held.is_recording()
    [record] = held_spans(token)
    assert record["name"] == "held"
    held.end()
