"""Integration Service activities as immediate tools.

An Integration Service (IS) activity is one HTTP operation of a connector
(e.g. GenAI Activities ``WebSearch``, ``POST /v2/webSearch``) run against a
connection. The runtime serves it to Codex on the ``uipath_tools`` MCP server
and answers each call as soon as
``UiPath().connections.invoke_activity_async(...)`` returns, the same call
low-code agents make (``uipath_langchain``'s ``create_integration_tool``). The
run never suspends on it.

Two ways to declare one, both ending in the same ``ActivityMetadata``:

* :func:`uipath_integration_tool`: coded, with fields named after the
  low-code resource (``tool_path``, ``method``, ``parameters``,
  ``body_structure``) and parameter locations inferred where unambiguous;
* :func:`integration_tool_from_resource`: an ``Integration`` resource copied
  from a low-code ``agent.json`` (or an ``AgentIntegrationToolResourceConfig``).

Typical use::

    from pydantic import BaseModel
    from uipath_codex_sdk import UiPathCodexAgent, uipath_integration_tool

    class WebSearchIn(BaseModel):
        query: str

    agent = UiPathCodexAgent(
        model="gpt-5.6-terra",
        prompt="Research {topic}.",
        tools=[
            uipath_integration_tool(
                "web_search",
                connection_id="<connection-id>",
                tool_path="/v2/webSearch",
                input_schema=WebSearchIn,
                static_args={"provider": "GoogleCustomSearch"},
            )
        ],
    )

The connection id is passed to the SDK unchanged on every call, so a
``connection.<id>`` resource overwrite (``bindings.json``) rebinds it.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError, create_model

from . import tools as _tools
from ._fixed_args import (  # noqa: F401  (re-exported: existing imports keep working)
    _SEGMENT_RE,
    _get_path,
    _hide_static,
    _navigate,
    _note_arguments,
    _parent_and_key,
    _resolve_ref,
    _segments,
    _set_path,
    _set_segments,
    _unwrap_any_of,
)
from .tools import (
    _PERMISSIVE_SCHEMA,
    ToolContext,
    UiPathToolError,
    _schema_from,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from uipath.agent.models.agent import (
        AgentIntegrationToolParameter,
        AgentIntegrationToolResourceConfig,
    )
    from uipath.platform.connections import ActivityMetadata

    from .runtime.mocking import MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "UiPathIntegrationTool",
    "integration_tool_from_resource",
    "uipath_integration_tool",
]

#: Default host-enforced timeout of an Integration Service call.
DEFAULT_TIMEOUT_SEC = 300.0
#: Methods an IS activity can have (``GETBYID`` is sent as ``GET``).
METHODS = frozenset({"GET", "GETBYID", "POST", "PUT", "PATCH", "DELETE"})
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
_LOCATIONS = ("query", "path", "header", "body", "multipart", "file")
_TEMPLATE_RE = re.compile(r"^\{\{(.+?)\}\}$")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
#: The output schema is added to the description only up to this size.
_OUTPUT_SCHEMA_DESCRIPTION_MAX = 4000


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------
@dataclass
class UiPathIntegrationTool:
    """An immediate tool that runs one Integration Service activity.

    Prefer :func:`uipath_integration_tool` or
    :func:`integration_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        connection_id: Connection the activity runs on. It is passed to the
            SDK as is, so a resource overwrite can rebind it.
        activity_metadata: Path, method, content type and parameter locations.
        input_json_schema: Schema shown to the model (static fields removed).
        input_model: Pydantic model the arguments are validated with, if any.
        static_args: Fixed values by IS parameter name (``a.b``, ``x[*].y``),
            merged over the model's arguments at call time.
        argument_args: ``{{x}}`` parameters: IS name → path in the agent input.
        output_json_schema: Output schema (description and eval mocks only).
        connector_key: Connector key (tracing only), e.g. ``uipath-uipath-airdk``.
        connection_name: Connection name (tracing only).
        object_name: IS object name (tracing only), e.g. ``v2::webSearch``.
        display_name: Name of the ``integrationTool`` span; defaults to ``name``.
        timeout_sec: Host-enforced timeout of one call, in seconds. None
            means the default (300); there is no way to turn it off.
        kind: Span ``toolType`` (``Integration``).
        example_calls: Example calls for eval mocking (``mockable``).
        mock_name: Name the activity is simulated by in evals; None means
            the display name (the resource name, e.g. ``Get User (v2)``).
    """

    #: The activity is mocked inside ``call`` (under its span), not by the host.
    mocks_in_call: ClassVar[bool] = True

    name: str
    description: str
    connection_id: str
    activity_metadata: ActivityMetadata
    input_json_schema: dict[str, Any] = field(
        default_factory=lambda: dict(_PERMISSIVE_SCHEMA)
    )
    input_model: type[BaseModel] | None = None
    static_args: dict[str, Any] = field(default_factory=dict)
    argument_args: dict[str, str] = field(default_factory=dict)
    output_json_schema: dict[str, Any] | None = None
    connector_key: str | None = None
    connection_name: str | None = None
    object_name: str | None = None
    display_name: str | None = None
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC
    kind: str = "integration"
    example_calls: list[Any] | None = None
    mock_name: str | None = None

    def __post_init__(self) -> None:
        self._invoker: Any = None
        self._validation_model: type[BaseModel] | None = None
        self.timeout_sec = _check_timeout(self.name, self.timeout_sec)

    # --- UiPathTool -------------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_json_schema

    @property
    def span_attributes(self) -> dict[str, Any]:
        attrs = {
            "connector_key": self.connector_key,
            "connector_name": self.connection_name,
            "connection_id": self.connection_id,
            "object_name": self.object_name,
            "tool.display_name": self.display_name,
        }
        return {k: v for k, v in attrs.items() if v is not None}

    def routed_names(self) -> set[str]:
        """Top-level input keys the SDK sends somewhere (others it drops)."""
        info = self.activity_metadata.parameter_location_info
        return {
            *info.query_params,
            *info.path_params,
            *info.header_params,
            *info.multipart_params,
            *info.body_fields,
        }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        if self.input_model is not None:
            # Static values are fixed and hidden from the model: pre-fill them
            # (at any depth) so a model that requires them validates. They
            # are set again over the arguments at call time.
            data = copy.deepcopy(args)
            for name, value in self.static_args.items():
                _set_path(data, name, copy.deepcopy(value))
            try:
                # by_alias: the advertised schema and the IS names use aliases.
                out = (
                    self._model_for_validation()
                    .model_validate(data)
                    .model_dump(mode="json", exclude_none=True, by_alias=True)
                )
            except ValidationError as e:
                raise ValueError(str(e)) from e
        else:
            _validate_against_schema(self.input_json_schema, args)
            out = dict(args)
        routed = self.routed_names()
        dropped = [k for k in out if k not in routed]
        if dropped:
            logger.warning(
                "Integration tool %s: argument(s) %s have no parameter location "
                "and are not sent.",
                self.name,
                dropped,
            )
            out = {k: v for k, v in out.items() if k in routed}
        return out

    def _model_for_validation(self) -> type[BaseModel]:
        """``input_model`` with top-level ``{{x}}`` fields made optional.

        Those are filled from the agent input at call time, so the model may
        leave them out.
        """
        model = self.input_model
        assert model is not None
        if self._validation_model is None:
            targets = {n for n in self.argument_args if extract_top_level_field(n) == n}
            overrides: dict[str, Any] = {}
            for fname, info in model.model_fields.items():
                if (info.alias or fname) in targets and info.is_required():
                    overrides[fname] = (
                        Optional[info.annotation],  # noqa: UP045 - runtime type
                        Field(None, alias=info.alias, description=info.description),
                    )
            self._validation_model = (
                create_model(model.__name__, __base__=model, **overrides)
                if overrides
                else model
            )
        return self._validation_model

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Run the activity; return its JSON result as text.

        Raises:
            UiPathToolError: The service refused the call, failed or timed
                out; the message is shown to the model.
        """
        from uipath.platform.errors import EnrichedException

        from .runtime.mocking import mark_simulated
        from .runtime.telemetry import (
            end_integration_tool_span,
            start_integration_tool_span,
        )

        payload = self.build_payload(args, ctx.agent_input)
        display = self.display_name or self.name
        span = start_integration_tool_span(display, arguments=payload)
        from opentelemetry import trace

        try:
            with trace.use_span(span, end_on_exit=False):
                result = await self._invoke()(**payload)
        except BaseException as e:
            end_integration_tool_span(span, error=e)
            if isinstance(e, EnrichedException):
                raise UiPathToolError(self._enriched_message(e)) from e
            if _is_httpx_timeout(e):
                logger.warning(
                    "Integration service call for tool %s timed out.", self.name
                )
                raise UiPathToolError(
                    f"Integration service call for tool '{self.name}' timed out. "
                    "You may retry."
                ) from e
            if type(e) is ValueError:
                raise UiPathToolError(
                    f"Tool '{self.name}' is misconfigured: {e}"
                ) from e
            raise
        mark_simulated(span)
        end_integration_tool_span(span, result=result)
        return json.dumps(result, ensure_ascii=False, default=str)

    # --- helpers ----------------------------------------------------------

    def build_payload(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The model's arguments with static and ``{{x}}`` values merged in."""
        payload = copy.deepcopy(args)
        for name, value in self.static_args.items():
            _set_path(payload, name, copy.deepcopy(value))
        for name, path in self.argument_args.items():
            found, value = _get_path(agent_input, path)
            if not found:
                logger.debug(
                    "Integration tool %s: {{%s}} is not in the agent input; "
                    "%s is not set.",
                    self.name,
                    path,
                    name,
                )
                continue
            _set_path(payload, name, copy.deepcopy(value))
        # Serialization sanitising (langchain's sanitize_dict_for_serialization).
        return json.loads(json.dumps(payload, default=str))

    def _invoke(self) -> Any:
        if self._invoker is not None:
            return self._invoker

        async def invoke_activity(**activity_input: Any) -> Any:
            # Looked up at call time (tests patch it).
            client = _tools._uipath_client()
            return await client.connections.invoke_activity_async(
                activity_metadata=self.activity_metadata,
                connection_id=self.connection_id,
                activity_input=activity_input,
            )

        from .runtime.mocking import wrap

        self._invoker = wrap(self.mock_spec, invoke_activity)
        return self._invoker

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's: by the resource name, with its output schema."""
        from .runtime.mocking import ANY_OBJECT_SCHEMA, MockSpec, mock_name_of

        return MockSpec(
            name=mock_name_of(self),
            description=self.description,
            input_schema=self.input_json_schema,
            output_schema=self.output_json_schema or dict(ANY_OBJECT_SCHEMA),
            example_calls=list(self.example_calls or []),
        )

    def _enriched_message(self, e: Any) -> str:
        """langchain's ``_INTEGRATION_ERRORS`` mapping, as model-facing text."""
        info = getattr(e, "error_info", None)
        code = getattr(info, "error_code", None) if info is not None else None
        message = (getattr(info, "message", None) if info is not None else None) or str(
            e
        ).strip()
        status = getattr(e, "status_code", None)
        if status == 400 or (status == 422 and code == "INVALID_DATA"):
            logger.warning(
                "Integration service rejected the arguments of tool %s (HTTP %s): %s",
                self.name,
                status,
                message,
            )
            return (
                f"Integration service returned an error for tool '{self.name}': "
                f"{message}"
            )
        logger.error(
            "Integration service call for tool %s failed (HTTP %s): %s",
            self.name,
            status,
            message,
        )
        return (
            f"Integration service call for tool '{self.name}' failed "
            f"(HTTP {status}): {message}"
        )


def _is_httpx_timeout(e: BaseException) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return False
    return isinstance(e, httpx.TimeoutException)


# --------------------------------------------------------------------------
# IS parameter names
# --------------------------------------------------------------------------
def extract_top_level_field(name: str) -> str:
    """Top-level field of an IS parameter name (``a.b`` → ``a``, ``x[*]`` → ``x``)."""
    return name.split(".")[0].split("[")[0]


# --------------------------------------------------------------------------
# ActivityMetadata (port of langchain's convert_to_activity_metadata)
# --------------------------------------------------------------------------
def _to_activity_metadata(
    tool_path: str,
    method: str,
    body_structure: Mapping[str, Any] | None,
    located: Sequence[tuple[str, str]],
) -> ActivityMetadata:
    """``ActivityMetadata`` from ``(IS name, fieldLocation)`` pairs.

    GETBYID is sent as GET. ``bodyStructure.contentType == "multipart"`` means
    ``multipart/form-data`` (with ``jsonBodySection``), anything else JSON.
    The multipart envelope marker (a multipart/file parameter named like the
    JSON body section) is skipped. query/path/header names map 1:1,
    multipart/file go to multipart, body and unknown locations to the body by
    top-level field.
    """
    from uipath.platform.connections import (
        ActivityMetadata,
        ActivityParameterLocationInfo,
    )

    http_method = method.upper()
    if http_method == "GETBYID":
        http_method = "GET"
    content_type = "application/json"
    json_body_section = None
    if body_structure is not None:
        if body_structure.get("contentType", "json") == "multipart":
            content_type = "multipart/form-data"
        json_body_section = body_structure.get("jsonBodySection")

    info = ActivityParameterLocationInfo()
    targets = {
        "query": info.query_params,
        "path": info.path_params,
        "header": info.header_params,
        "multipart": info.multipart_params,
        "file": info.multipart_params,
    }
    for name, location in located:
        if location in ("multipart", "file") and name == json_body_section:
            continue
        target = targets.get(location)
        key = name
        if target is None:
            target, key = info.body_fields, extract_top_level_field(name)
        if key not in target:
            target.append(key)
    return ActivityMetadata(
        object_path=tool_path,
        method_name=http_method,
        content_type=content_type,
        parameter_location_info=info,
        json_body_section=json_body_section,
    )


# --------------------------------------------------------------------------
# schema editing
# --------------------------------------------------------------------------
def fold_enums(schema: dict[str, Any], param_names: Sequence[str]) -> dict[str, Any]:
    """Fold each parameter's allowed values into its description; drop ``enum``.

    Port of langchain's ``strip_enums_from_schema`` / ``handle_enum``: the
    values come from ``enum`` or ``oneOf[].const``; ``{{…}}`` placeholders are
    not allowed values. Returns a modified deep copy.
    """
    schema = copy.deepcopy(schema)
    for name in param_names:
        node = _navigate(schema, _segments(name))
        if node is None:
            continue
        values = node.get("enum")
        if not values and isinstance(node.get("oneOf"), list):
            values = [
                b["const"]
                for b in node["oneOf"]
                if isinstance(b, dict) and "const" in b
            ]
        if values and not any(
            isinstance(v, str) and _TEMPLATE_RE.match(v) for v in values
        ):
            hint = f"Allowed value(s): {', '.join(str(v) for v in values)}."
            description = node.get("description")
            node["description"] = f"{description} ({hint})" if description else hint
        node.pop("enum", None)
    return schema


def remove_asterisk(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove the ``[*]`` IS puts in property names, definitions and ``$ref``.

    Port of langchain's ``remove_asterisk_from_properties`` (an IS bug
    workaround). Returns a modified deep copy.
    """
    schema = copy.deepcopy(schema)

    def fix(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if isinstance(node.get("$ref"), str):
            node["$ref"] = node["$ref"].replace("[*]", "")
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            fixed = {}
            for key, value in node["properties"].items():
                fixed[key.replace("[*]", "")] = value
                fix(value)
            node["properties"] = fixed
        if node.get("type") == "array":
            fix(node.get("items"))

    for key in ("$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            fixed = {}
            for name, value in schema[key].items():
                fixed[name.replace("[*]", "")] = value
                fix(value)
            schema[key] = fixed
    fix(schema)
    return schema


def _check_timeout(name: str, timeout_sec: float | None) -> float:
    """None means the default; a non-positive timeout is refused."""
    if timeout_sec is None:
        return DEFAULT_TIMEOUT_SEC
    if timeout_sec <= 0:
        raise ValueError(
            f"Integration tool '{name}': timeout_sec must be positive "
            f"(got {timeout_sec!r}); None means the {DEFAULT_TIMEOUT_SEC:g}s default."
        )
    return float(timeout_sec)


def _lowcode_tool_name(name: str) -> str:
    """langchain's ``sanitize_tool_name``: low-code agents name tools this way.

    Whitespace runs become one ``_``, other disallowed characters are dropped,
    then cut to 64 (``Get User (v2)`` → ``Get_User_v2``).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", "_".join(name.split()))[:64]
    return cleaned or sanitize_tool_name(name)


def _description(
    description: str | None,
    display: str,
    method: str,
    tool_path: str,
    output_schema: dict[str, Any] | None,
) -> str:
    text = description or (
        f"Runs the Integration Service activity '{display}' ({method} {tool_path})."
    )
    if output_schema and output_schema.get("properties"):
        compact = json.dumps(
            {"type": "object", "properties": output_schema["properties"]},
            separators=(",", ":"),
        )
        if len(compact) <= _OUTPUT_SCHEMA_DESCRIPTION_MAX:
            text += " Returns JSON matching: " + compact
    return text


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def _parameter(raw: Any) -> AgentIntegrationToolParameter:
    from uipath.agent.models.agent import AgentIntegrationToolParameter

    if isinstance(raw, AgentIntegrationToolParameter):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"parameter must be a mapping, got {type(raw).__name__}")
    data = dict(raw)
    data.setdefault("type", "string")
    try:
        return AgentIntegrationToolParameter.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"invalid parameter {data.get('name')!r}: {e}") from e


def _fixed_values(
    parameters: Sequence[AgentIntegrationToolParameter],
) -> tuple[dict[str, Any], dict[str, str]]:
    """``(static_args, argument_args)`` from ``static`` / ``argument`` parameters."""
    static: dict[str, Any] = {}
    arguments: dict[str, str] = {}
    for p in parameters:
        if p.field_variant == "static":
            static[p.name] = p.value
        elif p.field_variant == "argument":
            text = str(p.value) if p.value is not None else ""
            match = _TEMPLATE_RE.match(text)
            if not match:
                raise ValueError(
                    f"Argument parameter '{p.name}' has a malformed template: "
                    f"{p.value!r}. Expected '{{{{argName}}}}'."
                )
            arguments[p.name] = match.group(1).strip()
    return static, arguments


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def uipath_integration_tool(
    name: str,
    *,
    connection_id: str,
    tool_path: str,
    method: str = "POST",
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    parameters: Sequence[AgentIntegrationToolParameter | Mapping[str, Any]]
    | None = None,
    body_structure: Mapping[str, Any] | None = None,
    static_args: Mapping[str, Any] | None = None,
    connector_key: str | None = None,
    connection_name: str | None = None,
    object_name: str | None = None,
    tool_name: str | None = None,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
) -> UiPathIntegrationTool:
    """Declare an Integration Service activity as an immediate tool.

    Args:
        name: Tool name (sanitized into the model-facing name) and the
            display name of its ``integrationTool`` span.
        connection_id: Id of the IS connection. It is also the key of the
            ``connection`` binding in ``bindings.json``.
        tool_path: Activity path, e.g. ``/v2/webSearch`` or
            ``/UsersByEmail/{usersByEmailId}`` (``{x}`` becomes a path
            parameter).
        method: ``GET``, ``GETBYID``, ``POST``, ``PUT``, ``PATCH`` or ``DELETE``.
        description: What the tool does, for the model.
        input_schema: Pydantic model or JSON schema of the arguments the
            model fills in. Its top-level properties are sent in the body
            (POST/PUT/PATCH) or the query (GET/GETBYID/DELETE) unless
            ``parameters`` says otherwise.
        output_schema: Pydantic model or JSON schema of the result, added to
            the description.
        parameters: Low-code style parameters (``name``, ``fieldLocation``:
            query/path/header/body/multipart/file, optional ``fieldVariant``
            static/argument/dynamic and ``value``). ``static`` ones are fixed
            values hidden from the model; ``argument`` ones (``{{x}}``) come
            from the agent input when it has the value (the model sees them
            as optional fields and fills them otherwise).
        body_structure: ``{"contentType": "json"|"multipart", "jsonBodySection": ...}``.
        static_args: Fixed values by IS name (``a.b``, ``x[*].y`` work too),
            hidden from the model and merged over its arguments.
        connector_key: Connector key, e.g. ``uipath-uipath-airdk`` (tracing).
        connection_name: Connection name (tracing).
        object_name: IS object name, e.g. ``v2::webSearch`` (tracing).
        tool_name: Model-facing name; defaults to the sanitized ``name``.
        timeout_sec: Host-enforced timeout of one call, in seconds. None means
            the 300 s default; it cannot be turned off.

    Raises:
        ValueError: An empty ``tool_path`` or ``connection_id``, an unknown
            method, an invalid parameter, or a name in two locations.
    """
    if not tool_path or not tool_path.strip():
        raise ValueError(f"Integration tool '{name}' needs a non-empty tool_path.")
    if not connection_id or not str(connection_id).strip():
        raise ValueError(f"Integration tool '{name}' needs a connection_id.")
    upper = method.upper()
    if upper not in METHODS:
        raise ValueError(
            f"Integration tool '{name}': unsupported method {method!r} "
            f"(expected one of {', '.join(sorted(METHODS))})."
        )
    params = [_parameter(p) for p in parameters or []]
    static, arguments = _fixed_values(params)
    static.update(dict(static_args or {}))

    input_model = (
        input_schema
        if isinstance(input_schema, type) and issubclass(input_schema, BaseModel)
        else None
    )
    if input_schema is not None:
        schema = _schema_from(input_schema)
    elif params:
        schema = _schema_from_parameters(params)
    else:
        schema = dict(_PERMISSIVE_SCHEMA)

    # Locations: explicit parameters, then {x} in the path, then the rest by method.
    locations: dict[str, str] = {}
    located: list[tuple[str, str]] = []

    def locate(param_name: str, location: str, *, explicit: bool) -> None:
        location = location if location in _LOCATIONS else "body"
        key = extract_top_level_field(param_name) if location == "body" else param_name
        group = "multipart" if location == "file" else location
        known = locations.get(key)
        if known is not None and known != group:
            if not explicit:
                return
            raise ValueError(
                f"Integration tool '{name}': parameter {key!r} is in two locations "
                f"({known} and {group})."
            )
        locations[key] = group
        located.append((param_name, location))

    for p in params:
        locate(p.name, p.field_location, explicit=True)
    for placeholder in _PLACEHOLDER_RE.findall(tool_path):
        if placeholder not in locations:
            locate(placeholder, "path", explicit=False)
    default = "body" if upper in _BODY_METHODS else "query"
    for prop in (schema.get("properties") or {}).keys():
        if extract_top_level_field(prop) not in locations:
            locate(prop, default, explicit=False)
    for key in static:
        if extract_top_level_field(key) not in locations:
            locate(key, default, explicit=False)

    output_json = _schema_from(output_schema) if output_schema is not None else None
    return UiPathIntegrationTool(
        name=sanitize_tool_name(tool_name or name),
        description=_description(description, name, upper, tool_path, output_json),
        connection_id=str(connection_id),
        activity_metadata=_to_activity_metadata(
            tool_path, upper, body_structure, located
        ),
        input_json_schema=_note_arguments(
            _hide_static(schema, list(static)), arguments
        ),
        input_model=input_model,
        static_args=static,
        argument_args=arguments,
        output_json_schema=output_json,
        connector_key=connector_key,
        connection_name=connection_name,
        object_name=object_name,
        display_name=name,
        timeout_sec=timeout_sec,
    )


def _schema_from_parameters(
    params: Sequence[AgentIntegrationToolParameter],
) -> dict[str, Any]:
    """An input schema from the top-level parameters the model may fill."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        if p.field_variant == "static":
            continue
        if extract_top_level_field(p.name) != p.name:
            continue
        spec: dict[str, Any] = {"type": p.type or "string"}
        if p.description:
            spec["description"] = p.description
        props[p.name] = spec
        # ``{{x}}`` fields stay visible (the model fills them when the agent
        # input has no value) but are never required.
        if p.required and p.field_variant != "argument":
            required.append(p.name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def integration_tool_from_resource(
    resource: AgentIntegrationToolResourceConfig | Mapping[str, Any],
    *,
    tool_name: str | None = None,
    timeout_sec: float | None = None,
) -> UiPathIntegrationTool:
    """An immediate tool from a low-code ``Integration`` tool resource.

    Args:
        resource: An ``AgentIntegrationToolResourceConfig``, or the resource
            dict as found in a low-code ``agent.json`` (``resources[]`` entry
            with ``"type": "Integration"``).
        tool_name: Model-facing name; defaults to the resource name sanitized
            the way low-code agents do (``Get User (v2)`` → ``Get_User_v2``).
        timeout_sec: Host-enforced timeout; defaults to the resource's
            ``settings.timeout``, else 300 seconds.

    Raises:
        ValueError: The resource is disabled, has no ``toolPath`` or no
            ``connection.id``, or does not validate.
    """
    from uipath.agent.models.agent import AgentIntegrationToolResourceConfig

    if isinstance(resource, Mapping):
        data = dict(resource)
        if str(data.get("type", "")).lower() == "integration":
            data["type"] = "Integration"
        try:
            resource = AgentIntegrationToolResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid Integration tool resource: {e}") from e
    if resource.is_enabled is False:
        raise ValueError(
            f"Integration tool resource '{resource.name}' is disabled; filter it out "
            "before building tools."
        )
    props = resource.properties
    if not props.tool_path or not props.tool_path.strip():
        raise ValueError(
            f"Integration tool resource '{resource.name}' has an empty 'toolPath'."
        )
    connection = props.connection
    if not getattr(connection, "id", None):
        raise ValueError(
            f"Integration tool resource '{resource.name}' has no 'connection.id'."
        )
    method = props.method.upper()
    if method not in METHODS:
        raise ValueError(
            f"Integration tool resource '{resource.name}': unsupported method "
            f"{props.method!r}."
        )
    static, arguments = _fixed_values(props.parameters)
    schema = fold_enums(resource.input_schema, [p.name for p in props.parameters])
    output = remove_asterisk(resource.output_schema) if resource.output_schema else None
    connector = connection.connector if isinstance(connection.connector, dict) else {}
    if timeout_sec is None:
        settings_timeout = resource.settings.timeout if resource.settings else None
        timeout_sec = (
            float(settings_timeout) if settings_timeout else DEFAULT_TIMEOUT_SEC
        )
    description = resource.description or props.tool_description
    return UiPathIntegrationTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=_description(
            description, resource.name, method, props.tool_path, output
        ),
        connection_id=str(connection.id),
        activity_metadata=_to_activity_metadata(
            props.tool_path,
            method,
            props.body_structure,
            [(p.name, p.field_location) for p in props.parameters],
        ),
        input_json_schema=_note_arguments(
            _hide_static(schema, list(static)), arguments
        ),
        static_args=static,
        argument_args=arguments,
        output_json_schema=output,
        connector_key=connector.get("key"),
        connection_name=connection.name,
        object_name=props.object_name,
        display_name=resource.name,
        timeout_sec=timeout_sec,
        example_calls=list(props.example_calls) if props.example_calls else None,
        mock_name=resource.name,
    )
