"""Integration Service activity tools: mapping, schemas, invocation, errors.

No codex here: the tool is called directly, against a fake ``connections``
service or the real SDK ``ConnectionsService`` with its HTTP layer captured.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

import uipath_codex_sdk.tools as tools_mod
from uipath_codex_sdk import (
    ToolContext,
    UiPathIntegrationTool,
    UiPathTool,
    UiPathToolError,
    integration_tool_from_resource,
    uipath_integration_tool,
)
from uipath_codex_sdk.integration_tools import (
    _set_path,
    extract_top_level_field,
    fold_enums,
    remove_asterisk,
)
from uipath_codex_sdk.tools import is_durable_tool, is_immediate_tool

from .durable.fakes import FakeConnections, enriched

CONN = "11111111-aaaa-bbbb-cccc-000000000001"
CTX = ToolContext(call_id="call_1", runtime_id="rt", tool_name="web_search")

#: The web_researcher example's agent.json resource (uipath-agents-python),
#: trimmed of UI-only fields, with the connection id replaced.
WEB_SEARCH_RESOURCE: dict[str, Any] = {
    "$resourceType": "tool",
    "name": "Web Search",
    "description": "Web search executes a search of the public domain.",
    "type": "integration",
    "inputSchema": {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "The search engine to use",
                "enum": ["GoogleCustomSearch"],
                "oneOf": [
                    {"const": "GoogleCustomSearch", "title": "GoogleCustomSearch"}
                ],
            },
            "query": {"type": "string", "description": "The query"},
            "num": {"type": "integer", "description": "Number of results."},
        },
        "required": ["provider", "query"],
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "formattedResults": {"type": "string"},
            "results[*]": {
                "type": "array",
                "items": {"$ref": "#/definitions/results[*]"},
            },
        },
        "definitions": {
            "results[*]": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
            }
        },
    },
    "settings": {"timeout": 45},
    "properties": {
        "toolPath": "/v2/webSearch",
        "objectName": "v2::webSearch",
        "toolDisplayName": "Web Search",
        "toolDescription": "Web search.",
        "method": "POST",
        "connection": {
            "id": CONN,
            "name": "UiPath GenAI Activities",
            "elementInstanceId": 0,
            "connector": {"key": "uipath-uipath-airdk", "name": "GenAI"},
            "folder": {"key": "f867f9d2-365d-40ec-bdcd-ea0df740e58c"},
        },
        "parameters": [
            {
                "name": "provider",
                "type": "string",
                "fieldLocation": "body",
                "fieldVariant": "static",
                "value": "GoogleCustomSearch",
            },
            {
                "name": "query",
                "type": "string",
                "fieldLocation": "body",
                "fieldVariant": "dynamic",
                "value": "{{prompt}}",
            },
            {
                "name": "num",
                "type": "integer",
                "fieldLocation": "body",
                "fieldVariant": "dynamic",
            },
        ],
        "bodyStructure": {"contentType": "json"},
    },
    "isEnabled": True,
}


def _resource(**props: Any) -> dict[str, Any]:
    r = copy.deepcopy(WEB_SEARCH_RESOURCE)
    r["properties"].update(props)
    return r


def _info(tool: UiPathIntegrationTool) -> dict[str, list[str]]:
    i = tool.activity_metadata.parameter_location_info
    return {
        "query": i.query_params,
        "path": i.path_params,
        "header": i.header_params,
        "multipart": i.multipart_params,
        "body": i.body_fields,
    }


@pytest.fixture
def fake(monkeypatch) -> FakeConnections:
    conns = FakeConnections()
    client = type("Client", (), {"connections": conns})()
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: client)
    return conns


# --------------------------------------------------------------------------
# 1. ActivityMetadata mapping (langchain's convert_to_activity_metadata cases)
# --------------------------------------------------------------------------
def test_metadata_mapping_from_parameters():
    params = [
        {"name": "q", "fieldLocation": "query"},
        {"name": "id", "fieldLocation": "path"},
        {"name": "X-Trace", "fieldLocation": "header"},
        {"name": "meta.a.b", "fieldLocation": "body"},
        {"name": "meta.c", "fieldLocation": "body"},
        {"name": "attachments[*].name", "fieldLocation": "body"},
        {"name": "upload", "fieldLocation": "file"},
        {"name": "note", "fieldLocation": "multipart"},
        {"name": "body", "fieldLocation": "multipart"},  # the envelope marker
        {"name": "weird", "fieldLocation": "somewhere"},
    ]
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/items/{id}",
        method="GETBYID",
        parameters=params,
        body_structure={"contentType": "multipart", "jsonBodySection": "body"},
    )
    md = tool.activity_metadata
    assert md.method_name == "GET"
    assert md.object_path == "/items/{id}"
    assert md.content_type == "multipart/form-data"
    assert md.json_body_section == "body"
    assert _info(tool) == {
        "query": ["q"],
        "path": ["id"],
        "header": ["X-Trace"],
        "multipart": ["upload", "note"],
        "body": ["meta", "attachments", "weird"],
    }


def test_json_content_type_by_default():
    tool = uipath_integration_tool("t", connection_id=CONN, tool_path="/x")
    assert tool.activity_metadata.content_type == "application/json"
    assert tool.activity_metadata.json_body_section is None


@pytest.mark.parametrize(
    ("name", "top"),
    [
        ("metadata.field.test", "metadata"),
        ("attachments[*]", "attachments"),
        ("attachments[0].filename", "attachments"),
        ("simple", "simple"),
    ],
)
def test_extract_top_level_field(name, top):
    assert extract_top_level_field(name) == top


# --------------------------------------------------------------------------
# 2. coded location inference
# --------------------------------------------------------------------------
class SearchIn(BaseModel):
    query: str = Field(description="What to search for.")
    num: int | None = None


def test_post_schema_properties_go_to_the_body():
    tool = uipath_integration_tool(
        "web search",
        connection_id=CONN,
        tool_path="/v2/webSearch",
        input_schema=SearchIn,
        static_args={"provider": "GoogleCustomSearch"},
        parameters=[
            {
                "name": "X-Elements-Service-Profile",
                "fieldLocation": "header",
                "fieldVariant": "static",
                "value": "high-capacity",
            }
        ],
    )
    assert tool.name == "web_search"
    assert tool.display_name == "web search"
    assert _info(tool)["body"] == ["query", "num", "provider"]
    assert _info(tool)["header"] == ["X-Elements-Service-Profile"]
    assert set(tool.input_schema["properties"]) == {"query", "num"}
    assert isinstance(tool, UiPathTool)
    assert is_immediate_tool(tool) and not is_durable_tool(tool)


def test_get_schema_properties_go_to_the_query_and_placeholders_to_the_path():
    tool = uipath_integration_tool(
        "user",
        connection_id=CONN,
        tool_path="/UsersByEmail/{usersByEmailId}",
        method="GETBYID",
        input_schema={
            "type": "object",
            "properties": {
                "usersByEmailId": {"type": "string"},
                "fields": {"type": "string"},
            },
            "required": ["usersByEmailId"],
        },
    )
    assert tool.activity_metadata.method_name == "GET"
    assert _info(tool)["path"] == ["usersByEmailId"]
    assert _info(tool)["query"] == ["fields"]
    assert _info(tool)["body"] == []


def test_name_in_two_locations_is_rejected():
    with pytest.raises(ValueError, match="two locations"):
        uipath_integration_tool(
            "t",
            connection_id=CONN,
            tool_path="/x",
            parameters=[
                {"name": "a", "fieldLocation": "query"},
                {"name": "a", "fieldLocation": "header"},
            ],
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tool_path": " "}, "tool_path"),
        ({"connection_id": ""}, "connection_id"),
        ({"method": "TRACE"}, "unsupported method"),
    ],
)
def test_coded_constructor_checks(kwargs, match):
    base = {"connection_id": CONN, "tool_path": "/x"}
    with pytest.raises(ValueError, match=match):
        uipath_integration_tool("t", **{**base, **kwargs})


def test_schema_from_parameters_when_no_schema_is_given():
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        parameters=[
            {"name": "q", "type": "string", "fieldLocation": "query", "required": True},
            {
                "name": "k",
                "fieldLocation": "query",
                "fieldVariant": "static",
                "value": 1,
            },
        ],
    )
    assert tool.input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    assert tool.static_args == {"k": 1}


# --------------------------------------------------------------------------
# 3. static / argument parameters
# --------------------------------------------------------------------------
def test_static_fields_are_hidden_and_override_the_model():
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema={
            "type": "object",
            "properties": {"provider": {"type": "string"}, "query": {"type": "string"}},
            "required": ["provider", "query"],
        },
        static_args={"provider": "Google"},
    )
    assert set(tool.input_schema["properties"]) == {"query"}
    assert tool.input_schema["required"] == ["query"]
    args = tool.validate_args({"query": "q", "provider": "Bing"})
    assert tool.build_payload(args, None) == {"query": "q", "provider": "Google"}


def test_pydantic_model_requiring_a_static_field_still_validates():
    class In(BaseModel):
        provider: str
        query: str

    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema=In,
        static_args={"provider": "Google"},
    )
    assert "provider" not in tool.input_schema["properties"]
    assert tool.validate_args({"query": "q"}) == {"provider": "Google", "query": "q"}


def test_nested_static_paths():
    obj: dict[str, Any] = {"attachments": [{"text": "a"}, {"text": "b"}]}
    _set_path(obj, "attachments[*].actions.confirm", True)
    _set_path(obj, "meta.source.app", "codex")
    _set_path(obj, "absent[*].x", 1)  # no list, nothing created
    _set_path(obj, "items[1].id", 7)
    assert obj == {
        "attachments": [
            {"text": "a", "actions": {"confirm": True}},
            {"text": "b", "actions": {"confirm": True}},
        ],
        "meta": {"source": {"app": "codex"}},
        "items": [{}, {"id": 7}],
    }


def test_argument_parameters_resolve_from_the_agent_input():
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        parameters=[
            {
                "name": "channel",
                "fieldLocation": "body",
                "fieldVariant": "argument",
                "value": "{{team.channel}}",
            },
            {
                "name": "lang",
                "fieldLocation": "query",
                "fieldVariant": "argument",
                "value": "{{lang}}",
            },
        ],
    )
    assert set(tool.input_schema["properties"]) == {"q"}
    payload = tool.build_payload({"q": "x"}, {"team": {"channel": "#ops"}})
    assert payload == {"q": "x", "channel": "#ops"}  # lang missing: skipped


def test_malformed_argument_template():
    with pytest.raises(ValueError, match="malformed template"):
        uipath_integration_tool(
            "t",
            connection_id=CONN,
            tool_path="/x",
            parameters=[
                {
                    "name": "a",
                    "fieldLocation": "body",
                    "fieldVariant": "argument",
                    "value": "lang",
                }
            ],
        )


def test_unrouted_arguments_are_dropped_with_a_warning(caplog):
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert tool.validate_args({"q": "x", "extra": 1}) == {"q": "x"}
    assert "extra" in caplog.text
    with pytest.raises(ValueError):
        tool.validate_args({"q": 5})


# --------------------------------------------------------------------------
# 4. from an agent.json resource
# --------------------------------------------------------------------------
def test_from_resource_web_search():
    tool = integration_tool_from_resource(WEB_SEARCH_RESOURCE)
    assert tool.name == "Web_Search"
    assert tool.display_name == "Web Search"
    assert tool.connection_id == CONN
    assert tool.connector_key == "uipath-uipath-airdk"
    assert tool.object_name == "v2::webSearch"
    assert tool.timeout_sec == 45
    assert tool.static_args == {"provider": "GoogleCustomSearch"}
    # provider is static: hidden; the rest stays
    assert set(tool.input_schema["properties"]) == {"query", "num"}
    assert tool.input_schema["required"] == ["query"]
    assert _info(tool)["body"] == ["provider", "query", "num"]
    assert tool.activity_metadata.method_name == "POST"
    # [*] stripped from the output schema; the description mentions it
    out = tool.output_json_schema
    assert out is not None
    assert "results" in out["properties"] and "results" in out["definitions"]
    assert out["properties"]["results"]["items"]["$ref"] == "#/definitions/results"
    assert "Returns JSON matching" in tool.description
    assert tool.span_attributes == {
        "connector_key": "uipath-uipath-airdk",
        "connector_name": "UiPath GenAI Activities",
        "connection_id": CONN,
        "object_name": "v2::webSearch",
        "tool.display_name": "Web Search",
    }


def test_fold_enums():
    schema = WEB_SEARCH_RESOURCE["inputSchema"]
    folded = fold_enums(schema, ["provider", "query", "missing.path"])
    provider = folded["properties"]["provider"]
    assert "enum" not in provider
    assert provider["description"] == (
        "The search engine to use (Allowed value(s): GoogleCustomSearch.)"
    )
    assert "enum" in schema["properties"]["provider"]  # the input is not mutated
    templated = fold_enums(
        {"type": "object", "properties": {"x": {"type": "string", "enum": ["{{a}}"]}}},
        ["x"],
    )
    assert templated["properties"]["x"] == {"type": "string"}


def test_remove_asterisk():
    fixed = remove_asterisk(WEB_SEARCH_RESOURCE["outputSchema"])
    assert list(fixed["properties"]) == ["formattedResults", "results"]
    assert list(fixed["definitions"]) == ["results"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"isEnabled": False}, "disabled"),
        ({"properties": {"toolPath": ""}}, "toolPath"),
        (
            {"properties": {"connection": {"id": None, "elementInstanceId": 0}}},
            "connection.id",
        ),
    ],
)
def test_from_resource_refuses(change, match):
    r = copy.deepcopy(WEB_SEARCH_RESOURCE)
    for key, value in change.items():
        if isinstance(value, dict):
            r[key].update(value)
        else:
            r[key] = value
    with pytest.raises(ValueError, match=match):
        integration_tool_from_resource(r)


def test_from_resource_model_instance():
    from uipath.agent.models.agent import AgentIntegrationToolResourceConfig

    data = {**WEB_SEARCH_RESOURCE, "type": "Integration", "settings": None}
    model = AgentIntegrationToolResourceConfig.model_validate(data)
    tool = integration_tool_from_resource(model, tool_name="search", timeout_sec=9)
    assert tool.name == "search" and tool.timeout_sec == 9


# --------------------------------------------------------------------------
# 5./8. against the real SDK request builder (and resource overwrites)
# --------------------------------------------------------------------------
class CapturedSdk:
    """A real ``ConnectionsService`` whose HTTP calls are captured."""

    def __init__(self, folder_key: str = "fk-orig") -> None:
        from uipath.platform.common._config import UiPathApiConfig
        from uipath.platform.common._execution_context import UiPathExecutionContext
        from uipath.platform.connections._connections_service import (
            ConnectionsService,
        )

        self.requests: list[dict[str, Any]] = []
        self.folders = {CONN: folder_key}
        self.service = ConnectionsService(
            UiPathApiConfig(base_url="https://cloud.test/org/tenant", secret="s"),
            UiPathExecutionContext(),
            folders_service=None,  # type: ignore[arg-type]
        )
        self.service.request_async = self._request  # type: ignore[method-assign, assignment]
        self.connections = self.service

    async def _request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        url = str(url)
        self.requests.append({"method": method, "url": url, **kwargs})
        request = httpx.Request(method, "https://cloud.test" + url)
        if url.startswith("/connections_/api/v1/Connections/"):
            key = url.rsplit("/", 1)[1]
            body: Any = {
                "id": key,
                "name": "GenAI",
                "elementInstanceId": 7,
                "folder": {"key": self.folders.get(key, "fk-other")},
            }
        else:
            body = {"results": [{"title": "Ünïcode", "url": "https://u"}]}
        return httpx.Response(200, json=body, request=request)


async def test_call_routes_through_the_real_request_builder(monkeypatch):
    sdk = CapturedSdk()
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: sdk)
    tool = uipath_integration_tool(
        "web_search",
        connection_id=CONN,
        tool_path="/v2/{version}/webSearch",
        input_schema={
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "query": {"type": "string"},
                "num": {"type": "integer"},
            },
        },
        parameters=[
            {"name": "num", "fieldLocation": "query"},
            {
                "name": "X-Elements-Service-Profile",
                "fieldLocation": "header",
                "fieldVariant": "static",
                "value": "high-capacity",
            },
        ],
        static_args={"provider": "GoogleCustomSearch"},
    )
    args = tool.validate_args({"version": "v9", "query": "codex", "num": 3})
    text = await tool.call(args, CTX)
    assert json.loads(text) == {"results": [{"title": "Ünïcode", "url": "https://u"}]}
    assert "Ünïcode" in text  # not ASCII-escaped
    retrieve, invoke = sdk.requests
    assert retrieve["url"] == f"/connections_/api/v1/Connections/{CONN}"
    assert invoke["method"] == "POST"
    assert invoke["url"] == f"/elements_/v3/element/instances/{CONN}/v2/v9/webSearch"
    assert invoke["json"] == {"query": "codex", "provider": "GoogleCustomSearch"}
    assert invoke["params"] == {"num": "3"}
    headers = invoke["headers"]
    assert headers["X-Elements-Service-Profile"] == "high-capacity"
    assert "fk-orig" in headers.values()


async def test_resource_overwrite_rebinds_the_connection(monkeypatch):
    from uipath.platform.common import ResourceOverwritesContext
    from uipath.platform.common._bindings import ConnectionResourceOverwrite

    sdk = CapturedSdk()
    sdk.folders["new-conn"] = "fk-new"
    monkeypatch.setattr(tools_mod, "_uipath_client", lambda: sdk)
    tool = integration_tool_from_resource(WEB_SEARCH_RESOURCE)

    async def overwrites():
        return {
            f"connection.{CONN}": ConnectionResourceOverwrite.model_validate(
                {
                    "resource_type": "connection",
                    "connectionId": "new-conn",
                    "folderKey": "fk-new",
                }
            )
        }

    async with ResourceOverwritesContext(overwrites):
        await tool.call(tool.validate_args({"query": "q"}), CTX)
    retrieve, invoke = sdk.requests
    assert retrieve["url"] == "/connections_/api/v1/Connections/new-conn"
    assert invoke["url"] == "/elements_/v3/element/instances/new-conn/v2/webSearch"
    assert "fk-new" in invoke["headers"].values()
    assert tool.connection_id == CONN  # never pre-resolved


# --------------------------------------------------------------------------
# 6./7./9. errors, results, mockable passthrough
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            enriched(400, message="query is required"),
            "Integration service returned an error for tool 'web_search': query is required",
        ),
        (
            enriched(422, code="INVALID_DATA", message="bad num"),
            "Integration service returned an error for tool 'web_search': bad num",
        ),
        (
            enriched(422, code="OTHER", message="nope"),
            "Integration service call for tool 'web_search' failed (HTTP 422): nope",
        ),
        (
            enriched(500, message="down"),
            "Integration service call for tool 'web_search' failed (HTTP 500): down",
        ),
        (
            httpx.ReadTimeout("slow"),
            "Integration service call for tool 'web_search' timed out. You may retry.",
        ),
        (
            ValueError("Unsupported content type: text/csv"),
            "Tool 'web_search' is misconfigured: Unsupported content type: text/csv",
        ),
    ],
)
async def test_errors_become_model_facing_text(fake, error, expected):
    fake.fail = error
    tool = uipath_integration_tool("web_search", connection_id=CONN, tool_path="/x")
    with pytest.raises(UiPathToolError) as info:
        await tool.call({}, CTX)
    assert str(info.value) == expected


async def test_other_exceptions_propagate(fake):
    fake.fail = RuntimeError("kaput")
    tool = uipath_integration_tool("web_search", connection_id=CONN, tool_path="/x")
    with pytest.raises(RuntimeError):
        await tool.call({}, CTX)


async def test_list_result_and_single_real_call(fake):
    fake.responses["/users"] = [{"id": 1}, {"id": 2}]
    tool = uipath_integration_tool(
        "users", connection_id=CONN, tool_path="/users", method="GET"
    )
    assert json.loads(await tool.call({}, CTX)) == [{"id": 1}, {"id": 2}]
    # Outside an eval mocking context ``mockable`` (``runtime.mocking.wrap``)
    # calls through exactly once.
    assert len(fake.invocations) == 1
    inv = fake.invocations[0]
    assert inv.connection_id == CONN
    assert inv.activity_metadata.method_name == "GET"


# --------------------------------------------------------------------------
# review follow-ups
# --------------------------------------------------------------------------
class _AliasedIn(BaseModel):
    filter: str = Field(alias="$filter")
    api_key: str = Field(alias="X-Api-Key")
    top: int | None = Field(None, alias="$top")


def test_aliased_fields_are_validated_and_sent_by_alias(caplog):
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/items",
        method="GET",
        input_schema=_AliasedIn,
        parameters=[{"name": "X-Api-Key", "fieldLocation": "header"}],
    )
    info = tool.activity_metadata.parameter_location_info
    assert info.header_params == ["X-Api-Key"]
    assert set(info.query_params) == {"$filter", "$top"}
    assert set(tool.input_schema["properties"]) == {"$filter", "X-Api-Key", "$top"}
    args = tool.validate_args({"$filter": "a eq 1", "X-Api-Key": "k"})
    assert args == {"$filter": "a eq 1", "X-Api-Key": "k"}
    assert "no parameter location" not in caplog.text


class _LocaleIn(BaseModel):
    query: str
    locale: str = Field(description="Result language.")


def test_required_argument_field_of_a_pydantic_model():
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema=_LocaleIn,
        parameters=[
            {
                "name": "locale",
                "fieldLocation": "body",
                "fieldVariant": "argument",
                "value": "{{lang}}",
            }
        ],
    )
    # Visible but optional, with a note; the agent input wins when it has it.
    locale = tool.input_schema["properties"]["locale"]
    assert locale["description"].startswith("Result language. (Set from the agent")
    assert tool.input_schema["required"] == ["query"]
    args = tool.validate_args({"query": "q"})
    assert args == {"query": "q"}
    assert tool.build_payload(args, {"lang": "fr"}) == {"query": "q", "locale": "fr"}
    # The agent input has no value: the model's is sent.
    args = tool.validate_args({"query": "q", "locale": "de"})
    assert tool.build_payload(args, {}) == {"query": "q", "locale": "de"}
    assert tool.build_payload(args, {"lang": "fr"})["locale"] == "fr"


def test_argument_parameter_from_resource_stays_visible_to_the_model():
    resource = copy.deepcopy(WEB_SEARCH_RESOURCE)
    resource["properties"]["parameters"].append(
        {
            "name": "num",
            "type": "integer",
            "fieldLocation": "body",
            "fieldVariant": "argument",
            "value": "{{count}}",
        }
    )
    resource["inputSchema"]["required"].append("num")
    tool = integration_tool_from_resource(resource)
    assert "num" in tool.input_schema["properties"]
    assert "num" not in tool.input_schema["required"]
    assert (
        "agent input 'count'" in tool.input_schema["properties"]["num"]["description"]
    )
    args = tool.validate_args({"query": "q", "num": 3})
    assert tool.build_payload(args, None)["num"] == 3
    assert tool.build_payload(args, {"count": 9})["num"] == 9


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ({"items": [], "x": 1}, {"items": ["v"], "x": 1}),
        ({"x": 1}, {"x": 1, "items": ["v"]}),
        ({"items": ["a", "b"]}, {"items": ["v", "v"]}),
    ],
)
def test_static_star_on_empty_or_missing_array(start, expected):
    _set_path(start, "items[*]", "v")
    assert start == expected


def test_fold_enums_unwraps_a_nullable_leaf():
    schema = {
        "type": "object",
        "properties": {
            "color": {
                "anyOf": [{"type": "string", "enum": ["red", "blue"]}, {"type": "null"}]
            }
        },
    }
    folded = fold_enums(schema, ["color"])
    branch = folded["properties"]["color"]["anyOf"][0]
    assert "enum" not in branch
    assert branch["description"] == "Allowed value(s): red, blue."


class _Attachment(BaseModel):
    title: str
    text: str


class _PostIn(BaseModel):
    attachment: _Attachment


def test_nested_static_fields_are_hidden_and_prefilled():
    tool = uipath_integration_tool(
        "t",
        connection_id=CONN,
        tool_path="/x",
        input_schema=_PostIn,
        static_args={"attachment.title": "T"},
    )
    attachment = tool.input_schema["properties"]["attachment"]
    assert set(attachment["properties"]) == {"text"}
    assert attachment["required"] == ["text"]
    args = tool.validate_args({"attachment": {"text": "body"}})
    assert tool.build_payload(args, None) == {
        "attachment": {"title": "T", "text": "body"}
    }
    # A dict schema with a $ref'd nested static field.
    dict_tool = uipath_integration_tool(
        "d",
        connection_id=CONN,
        tool_path="/x",
        input_schema=_PostIn.model_json_schema(),
        static_args={"attachment.title": "T"},
    )
    nested = dict_tool.input_schema["properties"]["attachment"]
    assert "title" not in nested["properties"] and nested["required"] == ["text"]


def test_timeout_none_means_default_and_non_positive_is_refused():
    tool = uipath_integration_tool(
        "t", connection_id=CONN, tool_path="/x", timeout_sec=None
    )
    assert tool.timeout_sec == 300.0
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        uipath_integration_tool("t", connection_id=CONN, tool_path="/x", timeout_sec=0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Get User (v2)", "Get_User_v2"),
        ("Slack: Send message", "Slack_Send_message"),
        ("Send  Message", "Send_Message"),
        ("Web Search", "Web_Search"),
        ("!!!", "___"),
    ],
)
def test_resource_tool_names_match_low_code(name, expected):
    resource = copy.deepcopy(WEB_SEARCH_RESOURCE)
    resource["name"] = name
    assert integration_tool_from_resource(resource).name == expected
    assert integration_tool_from_resource(resource, tool_name="my tool").name == (
        "my_tool"
    )


@pytest.mark.parametrize(
    ("error", "level"),
    [
        (enriched(400, message="query is required"), "WARNING"),
        (enriched(422, code="INVALID_DATA", message="bad"), "WARNING"),
        (httpx.ReadTimeout("slow"), "WARNING"),
        (enriched(500, message="down"), "ERROR"),
    ],
)
async def test_is_errors_are_logged(fake, caplog, error, level):
    fake.fail = error
    tool = uipath_integration_tool("web_search", connection_id=CONN, tool_path="/x")
    with caplog.at_level("WARNING"), pytest.raises(UiPathToolError):
        await tool.call({}, CTX)
    records = [r for r in caplog.records if "web_search" in r.getMessage()]
    assert records and records[-1].levelname == level


def test_optional_field_keeps_its_description_in_the_tool_schema():
    class In(BaseModel):
        num: int | None = Field(None, description="Number of results.")

    tool = uipath_integration_tool(
        "t", connection_id=CONN, tool_path="/x", input_schema=In
    )
    assert tool.input_schema["properties"]["num"] == {
        "description": "Number of results.",
        "title": "Num",
        "type": "integer",
    }
