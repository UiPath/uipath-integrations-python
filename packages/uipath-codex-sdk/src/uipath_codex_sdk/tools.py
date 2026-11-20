"""UiPath tools served to Codex by the runtime's own MCP endpoint.

Two kinds share the ``uipath_tools`` server:

* **durable** tools (:class:`UiPathDurableTool`) suspend the run on an
  Orchestrator job (below);
* **immediate** tools (:class:`UiPathTool`) answer the call as soon as their
  work is done, e.g. Integration Service activities
  (:func:`~uipath_codex_sdk.uipath_integration_tool`).

Durable tools: Codex tool calls that suspend the run on an Orchestrator job.

A durable tool is served to Codex by the runtime's own MCP endpoint. When the
model calls one, the runtime starts the job (:meth:`UiPathDurableTool.start`),
never answers the call, hard-stops Codex and returns SUSPENDED with the job as
the resume trigger. When the job finishes, the run resumes, possibly on
another robot, and :meth:`UiPathDurableTool.finish` turns the finished job into
the text the model receives as the result of its original call.

Typical use::

    from pydantic import BaseModel
    from uipath_codex_sdk import UiPathCodexAgent, uipath_process_tool

    class AddIn(BaseModel):
        a: int
        b: int

    agent = UiPathCodexAgent(
        model="gpt-5.6-terra",
        prompt="Use a_plus_b to add {a} and {b}.",
        tools=[uipath_process_tool("a_plus_b", input_schema=AddIn,
                                   description="Adds a and b.")],
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ValidationError

from ._fixed_args import (
    FixedArg,
    _segments,
    fixed_args_from_code,
    fixed_args_schema,
    missing_args_message,
    reachable,
    with_fixed_args,
)
from .attachments import AttachmentStore, attachment_paths, to_attachment, values_at
from .tool_results import ToolResultStore

if TYPE_CHECKING:
    from .runtime.mocking import MockSpan, MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "DurableToolContext",
    "ToolContext",
    "AnyUiPathTool",
    "UiPathDurableTool",
    "UiPathDurableToolError",
    "UiPathDurableToolPending",
    "UiPathProcessTool",
    "UiPathTool",
    "UiPathToolError",
    "is_durable_tool",
    "is_immediate_tool",
    "uipath_agent_tool",
    "uipath_process_tool",
]

#: Job states that are still on their way to a final state — the same set
#: ``UiPathResumeTriggerHandler.read_trigger`` keeps waiting on. Stopping and
#: Terminating are *not* in it: the handler reports them as fired, so treating
#: them as pending would make the in-process resume loop re-poll Orchestrator
#: until the child leaves that state (possibly never). They are answered as
#: final ``{"status": "stopping"|"terminating"}`` results instead.
_PENDING_STATES = frozenset({"pending", "running", "suspended", "resumed"})

_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_TOOL_NAME_MAX = 64

_PERMISSIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

_START_404_TEMPLATES: dict[str, str] = {
    "AssociatedProcessNotFound": (
        "Could not find process for tool '{tool}'. Please check if the process "
        "is deployed in the configured folder."
    ),
    "AttachmentNotFound": (
        "Could not find an attachment passed to tool '{tool}'. Please check that "
        "the attachments provided to the tool still exist."
    ),
}
_START_404_FALLBACK = (
    "Could not start process for tool '{tool}': an item required to start the "
    "job was not found. Server message: {message}"
)


class UiPathToolError(Exception):
    """A tool could not do its work; the message is shown to the model."""


class UiPathDurableToolError(UiPathToolError):
    """A durable tool could not do its work; the message is shown to the model."""


class UiPathDurableToolPending(Exception):
    """The resume value says the job has not reached a final state yet.

    Raised by :meth:`UiPathDurableTool.finish`; the runtime then suspends again
    on the same job instead of answering the call. Only raise it for states the
    resume trigger handler also treats as pending: a value the handler reports
    as fired is read again straight away by the in-process resume loop.
    """


@dataclass(frozen=True)
class DurableToolContext:
    """What a durable tool knows about the call it is starting.

    Attributes:
        call_id: The Codex/Responses ``call_id`` of the tool call. It is also
            the interrupt id of the resume trigger.
        runtime_id: The UiPath runtime id (the job key when deployed).
        tool_name: The name the model called the tool by.
        agent_input: The validated input of this run (a dict), used to resolve
            ``{{arg}}`` values; None when unknown.
        attachments: The job attachments the run knows. Attachment arguments
            arrive already resolved to full JobAttachment objects.
        model: The agent's model.
        results: Results durable tools keep for each other in the run (an
            IXP extraction's, for its Document Validation tool).
    """

    call_id: str
    runtime_id: str
    tool_name: str
    agent_input: dict[str, Any] | None = None
    attachments: AttachmentStore | None = None
    model: str | None = None
    results: ToolResultStore | None = None


@runtime_checkable
class UiPathDurableTool(Protocol):
    """A tool whose call suspends the run until an external operation ends.

    ``start`` runs once, when the model calls the tool. ``finish`` runs once,
    on resume, against the arguments saved at start; unlike an in-tool
    interrupt the tool body never runs twice.

    Optional attributes, read with ``getattr``: ``kind`` / ``tool_type`` (as
    for :class:`UiPathTool`), ``started_output(value) -> dict``, the
    ``Tool call`` span's output while the call is suspended (default: the
    job key and state), and ``span_args(args, ctx)``, the ``Tool call``
    span's input (default: the model's arguments). ``start`` may leave child
    spans open across the suspension with
    :func:`~uipath_codex_sdk.runtime.telemetry.hold_span`.

    Pre-filled arguments (optional, for a tool that merges values the model
    does not fill over its arguments): ``prefilled_args(args, agent_input)``,
    the model's arguments with those values merged in (unvalidated), and
    ``prefilled_schema``, the input schema they follow (pre-filled fields
    included). The host checks the job attachments of the merged arguments,
    as LangChain does once its static arguments are applied, so a model value
    a pre-filled one replaces is not checked; ``start`` / ``call`` still get
    the model's arguments. ``checks_prefilled_attachments = False`` has the
    host check only the model's own values the merge keeps, never a
    pre-filled one (an escalation, which LangChain never resolves).

    Eval mocking (all optional): ``mock_spec`` (a
    :class:`~uipath_codex_sdk.runtime.mocking.MockSpec`: the mock name,
    description, schemas and example calls; default from ``mock_name`` /
    ``display_name`` / ``name``, ``output_json_schema`` and
    ``example_calls``), ``mock_precheck(args, ctx)`` (raise
    :class:`UiPathToolError` to refuse a simulated call),
    ``start_mock_span(args, ctx)`` (the child span a simulated call runs
    under), ``mock_args(args, ctx)`` (the arguments the mocker sees; default:
    the model's) and ``async mocked_output(args, value, ctx) -> str`` (the
    mocked value as the model's text; default: the value as JSON).
    """

    #: Name the model calls the tool by (``[A-Za-z0-9_-]``, at most 64 chars).
    name: str
    #: Description shown to the model.
    description: str

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the tool arguments (an ``object`` schema)."""
        ...

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalise the model's arguments; raise ``ValueError``."""
        ...

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Start the operation and return the suspend value (e.g. ``WaitJobRaw``)."""
        ...

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Turn the resume value into the tool result text for the model."""
        ...


@dataclass(frozen=True)
class ToolContext:
    """What an immediate tool knows about the call it is answering.

    Attributes:
        call_id: The Codex/Responses ``call_id`` of the tool call, or None
            when the MCP request carried none.
        runtime_id: The UiPath runtime id (the job key when deployed).
        tool_name: The name the model called the tool by.
        agent_input: The validated input of this run (a dict), used to resolve
            ``{{arg}}`` parameters; None when unknown.
        attachments: The job attachments the run knows. Attachment arguments
            arrive already resolved to full JobAttachment objects.
        model: The agent's model (e.g. for a tool that calls the LLM itself).
    """

    call_id: str | None
    runtime_id: str
    tool_name: str
    agent_input: dict[str, Any] | None = None
    attachments: AttachmentStore | None = None
    model: str | None = None


@runtime_checkable
class UiPathTool(Protocol):
    """A tool the runtime answers as soon as :meth:`call` returns.

    Unlike a durable tool, the call is never held and the run never suspends
    on it. Optional attributes, read with ``getattr``:

    * ``kind`` (``str``, default ``"function"``): the span ``toolType``
      (capitalized) and the ``tool.kind`` attribute;
    * ``tool_type`` (``str``): the span ``toolType`` as is, instead of the
      capitalized ``kind`` (``ContextGrounding``);
    * ``timeout_sec`` (``float | None``): host-enforced timeout in seconds;
      None (or missing) means 300, there is no way to turn it off;
    * ``span_attributes`` (``dict``): extra attributes on the ``Tool call`` span;
    * the eval mocking hooks of :class:`UiPathDurableTool`, or
      ``mocks_in_call = True`` for a tool that mocks its own platform call
      inside ``call`` (the host then never simulates it);
    * ``prefilled_args`` / ``prefilled_schema``, as for
      :class:`UiPathDurableTool`.
    """

    #: Name the model calls the tool by (``[A-Za-z0-9_-]``, at most 64 chars).
    name: str
    #: Description shown to the model.
    description: str

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the tool arguments (an ``object`` schema)."""
        ...

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalise the model's arguments; raise ``ValueError``."""
        ...

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Do the work and return the result text; raise :class:`UiPathToolError`."""
        ...


#: What ``UiPathCodexAgent.tools`` holds: durable or immediate tools.
AnyUiPathTool = Union[UiPathDurableTool, UiPathTool]


def tool_type_of(tool: Any, kind: str | None = None) -> str:
    """``toolType`` of a tool's ``Tool call`` span.

    The tool's ``tool_type`` attribute when set (``ContextGrounding``), else
    its ``kind`` capitalized (``integration`` → ``Integration``).
    """
    explicit = getattr(tool, "tool_type", None)
    if explicit:
        return str(explicit)
    return str(kind or getattr(tool, "kind", None) or "function").capitalize()


def is_durable_tool(tool: Any) -> bool:
    """True for a tool that suspends the run (:class:`UiPathDurableTool`)."""
    return isinstance(tool, UiPathDurableTool)


def is_immediate_tool(tool: Any) -> bool:
    """True for a tool answered right away (:class:`UiPathTool`, not durable)."""
    return isinstance(tool, UiPathTool) and not is_durable_tool(tool)


def sanitize_tool_name(name: str) -> str:
    """Make ``name`` acceptable as a model-facing tool name."""
    cleaned = _TOOL_NAME_RE.sub("_", name.strip()) or "tool"
    return cleaned[:_TOOL_NAME_MAX]


def _schema_from(
    input_schema: type[BaseModel] | dict[str, Any] | None,
) -> dict[str, Any]:
    if input_schema is None:
        return dict(_PERMISSIVE_SCHEMA)
    if isinstance(input_schema, dict):
        return dict(input_schema)
    return _tool_schema_from_model(input_schema)


def _tool_schema_from_model(model_type: type[BaseModel]) -> dict[str, Any]:
    """A tool's argument schema: ``$ref`` resolved, nullable ``anyOf`` collapsed.

    Unlike the entrypoint schema (``_model_to_platform_schema``), collapsing
    ``anyOf: [X, null]`` keeps the field's own ``description``, ``title`` and
    non-null ``default`` (``num: int | None = Field(None, description=...)``), so the
    model still sees them. A ``UiPathAttachment`` field becomes the
    ``job-attachment`` reference, so the model only has to pass its ``ID``.
    """
    from uipath.runtime.schema import transform_attachments, transform_references

    raw = model_type.model_json_schema()
    unpacked, _ = transform_references(raw)
    result: dict[str, Any] = {
        "type": "object",
        "properties": _collapse_nullable(unpacked.get("properties", {})),
        "required": unpacked.get("required", []),
    }
    for key in ("title", "description"):
        if key in unpacked:
            result[key] = unpacked[key]
    return transform_attachments(result)


def _collapse_nullable(schema: Any) -> Any:
    if isinstance(schema, dict):
        branches = schema.get("anyOf")
        if isinstance(branches, list) and len(branches) == 2:
            non_null = [
                b
                for b in branches
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                merged = {k: v for k, v in schema.items() if k != "anyOf"}
                merged.update(non_null[0])
                for key in ("description", "title"):
                    if key in schema:
                        merged[key] = schema[key]
                if merged.get("default", 0) is None:
                    merged.pop("default")  # null is not a value of the type
                return _collapse_nullable(merged)
        return {k: _collapse_nullable(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_collapse_nullable(item) for item in schema]
    return schema


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _check_json_type(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        python_types = _JSON_TYPES.get(t)
        if python_types is None:
            return True  # unknown type keyword: do not reject
        if isinstance(value, bool) and t in ("integer", "number"):
            continue
        if isinstance(value, python_types):
            return True
    return False


def _validate_against_schema(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Shallow JSON-schema check: required keys and top-level property types."""
    missing = [k for k in schema.get("required") or [] if k not in args]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    props = schema.get("properties") or {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                raise ValueError(f"unexpected argument: {key}")
            continue
        if isinstance(spec, dict) and "type" in spec:
            if not _check_json_type(value, spec["type"]):
                raise ValueError(f"argument {key!r} must be of type {spec['type']}")


def _start_error_message(e: Exception, tool: str) -> str:
    """Model-facing text for a failed job start (langchain's mapping)."""
    status = getattr(e, "status_code", None)
    info = getattr(e, "error_info", None)
    code = getattr(info, "error_code", None) if info is not None else None
    message = (getattr(info, "message", None) if info is not None else None) or ""
    if status == 404 and str(code) == "1002":
        template = _START_404_TEMPLATES.get(message, _START_404_FALLBACK)
    elif status == 400 and str(code) == "1100":
        template = (
            "Could not find folder for tool '{tool}'. Please check if the folder "
            "exists and is accessible by the robot."
        )
    elif status == 409:
        template = "Cannot start process for tool '{tool}': {message}"
    else:
        template = "Failed to start the job for tool '{tool}': {message}"
        message = message or str(e)
    return template.format(tool=tool, message=message)


def _uipath_client() -> Any:
    """The UiPath platform client (a seam for tests)."""
    from uipath.platform import UiPath

    return UiPath()


def _current_span_id() -> str | None:
    """Hex id of the current span (the tool call span), or None."""
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.span_id, "016x")


def _job_from(value: Any) -> Any | None:
    """Return ``value`` as a platform ``Job`` when it is (or serializes) one."""
    from uipath.platform.orchestrator import Job

    if isinstance(value, Job):
        return value
    if isinstance(value, dict):
        keys = set(value)
        if {"Key", "State"} <= keys or {"key", "state", "id"} <= keys:
            try:
                return Job.model_validate(value)
            except ValidationError:
                return None
    return None


def _dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class UiPathProcessTool:
    """A durable tool that runs an Orchestrator process (or agent) as a job.

    Prefer the :func:`uipath_process_tool` / :func:`uipath_agent_tool`
    factories over constructing this directly.

    Attributes:
        process_name: Release name of the process in Orchestrator. It is also
            the binding key (``process.<process_name>``) resource overwrites
            are matched on.
        name: Model-facing tool name.
        description: Description shown to the model.
        folder_path: Orchestrator folder of the release. Falls back to
            ``UIPATH_FOLDER_PATH``, then to the SDK's configured folder.
        input_model: Pydantic model of the process input, when given.
        input_json_schema: JSON schema of the process input. Its attachment
            fields are linked to the child job as job attachments.
        output_json_schema: JSON schema of the process output: shown in the
            description, and the job attachments of its attachment fields
            become known to the run.
        run_as_me: Run the child job under the calling user's identity.
        kind: ``"process"`` or ``"agent"`` (agents are releases too).
        mock_name: Name the tool is simulated by in evals; None means
            ``process_name``.
        example_calls: Example calls for the eval LLM mocker.
        fixed_args: Pre-filled inputs by path (``a.b``, ``x[*].y``): static
            ones are hidden from the model, agent-input ones stay visible but
            optional; merged over the model's arguments when the job starts.
        tool_type: ``toolType`` of the ``Tool call`` span; None means the
            capitalized ``kind`` (``Process`` / ``Agent``).
        display_name: Name of a simulated call's ``processTool`` /
            ``agentTool`` span; None means ``process_name``.
        mock_child_span: Whether a simulated call has that span (low-code
            opens none for a Flow).
    """

    process_name: str
    name: str
    description: str
    folder_path: str | None = None
    input_model: type[BaseModel] | None = None
    input_json_schema: dict[str, Any] = field(
        default_factory=lambda: dict(_PERMISSIVE_SCHEMA)
    )
    output_json_schema: dict[str, Any] | None = None
    run_as_me: bool = False
    kind: Literal["process", "agent"] = "process"
    mock_name: str | None = None
    example_calls: list[Any] = field(default_factory=list)
    fixed_args: dict[str, FixedArg] = field(default_factory=dict)
    tool_type: str | None = None
    display_name: str | None = None
    mock_child_span: bool = True

    @property
    def input_schema(self) -> dict[str, Any]:
        """The process input as the model sees it (pre-filled fields hidden)."""
        return fixed_args_schema(self.input_json_schema, self.fixed_args)

    def merged_args(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The job's input: the model's arguments with the pre-filled values.

        Validated with ``input_model`` once merged, when there is one; else
        its required fields are checked (an agent-input field the input has
        no value for is left to the model, as in langchain).

        Raises:
            UiPathDurableToolError: The merged input does not validate.
        """
        if not self.fixed_args:
            return dict(args)
        merged = with_fixed_args(args, self.fixed_args, agent_input)
        if self.input_model is None:
            missing = [
                k
                for k in self.input_json_schema.get("required") or []
                if k not in merged
            ]
            if missing:
                raise UiPathDurableToolError(missing_args_message(self.name, missing))
            return merged
        try:
            return self.input_model.model_validate(merged).model_dump(
                mode="json", by_alias=True
            )
        except ValidationError as e:
            raise UiPathDurableToolError(
                f"Invalid arguments for tool '{self.name}': {e}"
            ) from e

    @property
    def prefilled_schema(self) -> dict[str, Any]:
        """The process input, pre-filled fields included (the host resolves its attachments)."""
        return self.input_json_schema

    def prefilled_args(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The model's arguments with the pre-filled values, unvalidated.

        The host checks the job attachments of these (LangChain's order).
        """
        if not self.fixed_args:
            return args
        return with_fixed_args(args, self.fixed_args, agent_input)

    def span_args(self, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """The ``Tool call`` span's input: the arguments with the pre-filled values.

        As low-code's, whose tool call carries them once merged (values
        marked sensitive included): the job input (:meth:`merged_args`), or
        the unvalidated merge when that input does not validate.
        """
        if not self.fixed_args:
            return args
        agent_input = getattr(ctx, "agent_input", None)
        try:
            return self.merged_args(args, agent_input)
        except UiPathToolError:
            return with_fixed_args(args, self.fixed_args, agent_input)

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as LangChain's process tool: the job's output schema."""
        from .runtime.mocking import ANY_OBJECT_SCHEMA, MockSpec

        return MockSpec(
            name=self.mock_name or self.process_name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_json_schema or dict(ANY_OBJECT_SCHEMA),
            example_calls=list(self.example_calls),
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan | None:
        """low-code's ``processTool`` / ``agentTool`` child of a simulated call.

        Its ``arguments`` are the job input the call would start with. None
        (no child) when ``mock_child_span`` is off.
        """
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_process_tool_span

        if not self.mock_child_span:
            return None
        span = start_process_tool_span(
            self.display_name or self.process_name,
            arguments=self.merged_args(args, getattr(ctx, "agent_input", None)),
            kind=self.kind,
        )
        return MockSpan(span)

    def mock_args(self, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """The model's arguments with the pre-filled values, as the mocker sees them.

        langchain's ``StaticArgsHandler`` merges them into the tool call
        before its mocked function runs, so mockito behaviors match on them.
        """
        return self.merged_args(args, getattr(ctx, "agent_input", None))

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated job output, as a successful job's (``finish``)."""
        return _dumps(value)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        if self.fixed_args:
            # The merged input is what the job gets (``merged_args``); here
            # the pre-filled fields only must not fail the model's arguments.
            if self.input_model is None:
                _validate_against_schema(self.input_schema, args)
            else:
                self._check_with_fixed(args)
            return dict(args)
        if self.input_model is not None:
            try:
                return self.input_model.model_validate(args).model_dump(
                    mode="json", by_alias=True
                )
            except ValidationError as e:
                raise ValueError(str(e)) from e
        _validate_against_schema(self.input_json_schema, args)
        return dict(args)

    def _check_with_fixed(self, args: dict[str, Any]) -> None:
        """Validate ``args`` with ``input_model``, the pre-filled values set.

        Agent-input fields are not known yet: a missing one is not an error,
        nor is an invalid model attachment in one (the host leaves an unknown
        one the agent input replaces unresolved); the agent input or, failing
        that, ``merged_args`` decides. Any other invalid model value is.
        """
        assert self.input_model is not None
        try:
            self.input_model.model_validate(
                with_fixed_args(args, self.fixed_args, None)
            )
        except ValidationError as e:
            pending = [
                _segments(name) for name, f in self.fixed_args.items() if f.from_input
            ]
            files: list[list[str | int]] = [
                list(p) for p in attachment_paths(self.input_json_schema)
            ]

            def excused(err: Any) -> bool:
                loc = tuple(err.get("loc", ()))
                if not any(_under(loc, segs) for segs in pending):
                    return False
                return err.get("type") == "missing" or any(
                    _under(loc, segs) for segs in files
                )

            if not all(excused(err) for err in e.errors()):
                raise ValueError(str(e)) from e

    def _folder_path(self) -> str | None:
        return self.folder_path or os.environ.get("UIPATH_FOLDER_PATH") or None

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Start the job and return ``WaitJobRaw`` for it.

        Only ``process_folder_key`` is set on the wait model (the folder path is
        a fallback for a job without a folder key): setting both makes every
        resume's job lookup fail.

        The job's ``ParentSpanId`` is the ``Tool call`` span the tool host opens
        around this call (passed explicitly: ``invoke_async`` would otherwise
        use its own ``processes_invoke`` span). The attachments in the
        arguments' attachment fields are linked to the job, as in LangChain.
        The job's input is :meth:`merged_args` (``args`` is not changed: the
        host keeps it for ``finish``). BTS correlation (``parent_operation_id``) is not passed: it comes from
        the low-code agent runtime, which coded Codex agents do not run under.
        """
        from uipath.platform.common import WaitJobRaw

        merged = self.merged_args(args, ctx.agent_input)
        attachments = [
            a
            for a in (
                to_attachment(v)
                for v in values_at(merged, attachment_paths(self.input_json_schema))
            )
            if a is not None
        ]
        extra: dict[str, Any] = {"attachments": attachments} if attachments else {}
        client = _uipath_client()
        try:
            job = await client.processes.invoke_async(
                name=self.process_name,
                input_arguments=merged,
                folder_path=self._folder_path(),
                run_as_me=True if self.run_as_me else None,
                parent_span_id=_current_span_id(),
                **extra,
            )
        except Exception as e:
            raise UiPathDurableToolError(_start_error_message(e, self.name)) from e
        if job is None or not getattr(job, "key", None):
            raise UiPathDurableToolError(
                f"Orchestrator did not return a job for tool '{self.name}'."
            )
        logger.info(
            "durable tool %s started %s job %s (call %s)",
            self.name,
            self.kind,
            job.key,
            ctx.call_id,
        )
        if job.folder_key:
            return WaitJobRaw(job=job, process_folder_key=job.folder_key)
        return WaitJobRaw(job=job, process_folder_path=self._folder_path())

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Map a finished job (or a raw resume payload) to the tool result text.

        * successful job → its output arguments (JSON text);
        * faulted job → ``{"status": "faulted", "error": <job info>, "job_key": ...}``
          (plus ``details`` with the structured job error, when there is one);
        * stopped, Stopping, Terminating or another final state →
          ``{"status": <state>, "job_key": ...}``;
        * still pending or running → :class:`UiPathDurableToolPending`;
        * anything else (a raw ``--resume`` payload) → JSON-dumped as is.
        """
        job = _job_from(resume_value)
        if job is None:
            return _dumps(resume_value)
        state = (job.state or "").lower()
        if state in _PENDING_STATES:
            raise UiPathDurableToolPending(f"job {job.key} is {state or 'unknown'}")
        if state == "successful":
            output = await _uipath_client().jobs.extract_output_async(job)
            if output is None or output == "":
                return _dumps({"status": "successful", "job_key": job.key})
            return _dumps(output)
        if state == "faulted":
            details = (
                job.job_error.model_dump(mode="json", exclude_none=True)
                if job.job_error is not None
                else None
            )
            fallback = (details or {}).get("detail") or (details or {}).get("title")
            result: dict[str, Any] = {
                "status": "faulted",
                "error": job.info or fallback or "Unknown error",
                "job_key": job.key,
            }
            if details:
                result["details"] = details
            return _dumps(result)
        return _dumps({"status": state or "unknown", "job_key": job.key})


def _under(loc: tuple[Any, ...], segs: list[str | int]) -> bool:
    """True when a validation error's ``loc`` is at or under path ``segs``."""
    if len(loc) < len(segs):
        return False
    return all(
        seg == part or (seg == "*" and isinstance(part, int))
        for seg, part in zip(segs, loc[: len(segs)], strict=True)
    )


def _description(
    description: str | None,
    process_name: str,
    kind: str,
    output_schema: dict[str, Any] | None,
) -> str:
    text = (
        description
        or f"Runs the UiPath {kind} '{process_name}' and returns its output."
    )
    if output_schema and output_schema.get("properties"):
        text += " Returns JSON matching: " + json.dumps(
            {"type": "object", "properties": output_schema["properties"]},
            separators=(",", ":"),
        )
    return text


def uipath_process_tool(
    name: str,
    *,
    folder_path: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    tool_name: str | None = None,
    run_as_me: bool = False,
    example_calls: list[Any] | None = None,
    static_args: Mapping[str, Any] | None = None,
    args_from_input: Mapping[str, str] | None = None,
) -> UiPathProcessTool:
    """Declare an Orchestrator process as a durable tool.

    Args:
        name: Release name of the process (e.g. ``"a_plus_b"``).
        folder_path: Orchestrator folder holding the release. Defaults to the
            run's folder (``UIPATH_FOLDER_PATH``).
        description: What the tool does, for the model.
        input_schema: Pydantic model or JSON schema of the process input
            arguments. Arguments are validated against it before the job starts.
        output_schema: Pydantic model or JSON schema of the process output,
            added to the description so the model knows what comes back.
        tool_name: Model-facing tool name; defaults to the sanitized ``name``.
        run_as_me: Run the child job under the calling user's identity.
        example_calls: Example calls (``uipath.eval.mocks.ExampleCall``) the
            eval LLM mocker is shown when it simulates the tool (mock name:
            ``name``).
        static_args: Fixed process inputs, hidden from the model. Keys are
            paths into the input (``a.b``, ``items[*].id``; JSONPath such as
            ``$['a']['b']`` too).
        args_from_input: Process inputs read from the agent input (field →
            path, ``{{a.b}}`` / ``a.b`` / ``$.a.b``). The field stays visible
            to the model (optional), whose value is used when the agent input
            has none.

    Raises:
        ValueError: A field in both ``static_args`` and ``args_from_input``,
            or one the input schema does not have.
    """
    return _build_tool(
        name,
        kind="process",
        folder_path=folder_path,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        tool_name=tool_name,
        run_as_me=run_as_me,
        example_calls=example_calls,
        static_args=static_args,
        args_from_input=args_from_input,
    )


def uipath_agent_tool(
    name: str,
    *,
    folder_path: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    tool_name: str | None = None,
    run_as_me: bool = False,
    example_calls: list[Any] | None = None,
    static_args: Mapping[str, Any] | None = None,
    args_from_input: Mapping[str, str] | None = None,
) -> UiPathProcessTool:
    """Declare a deployed UiPath agent as a durable tool.

    Agents are Orchestrator releases, so this is :func:`uipath_process_tool`
    with ``kind="agent"``; see it for the arguments.
    """
    return _build_tool(
        name,
        kind="agent",
        folder_path=folder_path,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        tool_name=tool_name,
        run_as_me=run_as_me,
        example_calls=example_calls,
        static_args=static_args,
        args_from_input=args_from_input,
    )


def _build_tool(
    name: str,
    *,
    kind: Literal["process", "agent"],
    folder_path: str | None,
    description: str | None,
    input_schema: type[BaseModel] | dict[str, Any] | None,
    output_schema: type[BaseModel] | dict[str, Any] | None,
    tool_name: str | None,
    run_as_me: bool,
    example_calls: list[Any] | None = None,
    static_args: Mapping[str, Any] | None = None,
    args_from_input: Mapping[str, str] | None = None,
) -> UiPathProcessTool:
    tool_label = sanitize_tool_name(tool_name or name)
    input_model = (
        input_schema
        if isinstance(input_schema, type) and issubclass(input_schema, BaseModel)
        else None
    )
    output_json = _schema_from(output_schema) if output_schema is not None else None
    input_json = _schema_from(input_schema)
    label = f"{kind.capitalize()} tool '{tool_label}'"
    fixed = fixed_args_from_code(static_args, args_from_input, label=label)
    if input_json.get("properties"):
        for field_name in fixed:
            if not reachable(input_json, field_name):
                raise ValueError(f"{label}: '{field_name}' is not in its input schema.")
    return UiPathProcessTool(
        process_name=name,
        name=tool_label,
        description=_description(description, name, kind, output_json),
        folder_path=folder_path,
        input_model=input_model,
        input_json_schema=input_json,
        output_json_schema=output_json,
        run_as_me=run_as_me,
        kind=kind,
        example_calls=list(example_calls or []),
        fixed_args=fixed,
    )
