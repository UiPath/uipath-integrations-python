"""Analyze Files: the low-code internal tool that reads job attachments.

The model passes JobAttachment objects (already resolved by the tool host
against the run's known attachments) and an ``analysisTask``. As in
LangChain, the tool downloads each file (30 MB at most, a larger one fails the
run), sends it with the task to the agent's model in one non-streaming
Responses call through the UiPath LLM Gateway, and answers
``{"analysisResult": <text>}``.

Spans, as in low-code agents: ``Tool call - <tool>`` (``toolType``
``Internal``) → ``<tool>`` (the internal tool) → ``LLM call`` with the files
sent as ``attachments``.

Typical use::

    from uipath_codex_sdk import UiPathAttachment, UiPathCodexAgent
    from uipath_codex_sdk import uipath_analyze_files_tool

    class Input(BaseModel):
        invoice: UiPathAttachment
        question: str

    agent = UiPathCodexAgent(
        input_schema=Input,
        prompt="Answer {question} about the attached invoice.",
        tools=[uipath_analyze_files_tool()],
    )
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError
from uipath.runtime.errors import UiPathErrorCategory

from ._fixed_args import (
    FixedArg,
    fixed_args_from_properties,
    fixed_args_schema,
    missing_args_message,
    missing_from_input,
    with_fixed_args,
)
from .attachments import (
    Attachment,
    dump_attachment,
    span_attachments,
    to_attachment,
)
from .integration_tools import _lowcode_tool_name
from .runtime.errors import UiPathCodexErrorCode, UiPathCodexRuntimeError
from .tools import (
    ToolContext,
    UiPathToolError,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from .runtime.mocking import MockSpec

logger = logging.getLogger(__name__)

DEFAULT_NAME = "Analyze Files"
DEFAULT_DESCRIPTION = (
    "Analyze one or more files with an LLM to extract, synthesize, or answer "
    "queries about their content."
)
#: LangChain's system message for the analysis call.
ANALYZE_FILES_SYSTEM_MESSAGE = (
    "Process the provided files to complete the given task. "
    "Analyze the files contents thoroughly to deliver an accurate response "
    "based on the extracted information."
)
NO_ATTACHMENTS_RESULT = "No attachments provided to analyze."
#: ``properties.toolType`` of the low-code resource.
ANALYZE_FILES_TOOL_TYPE = "analyze-attachments"
#: LangChain's per-file limit for LLM payloads.
MAX_FILE_BYTES = 30 * 1024 * 1024
IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
#: Types LangChain converts first (TIFF pages to PNG); not supported here yet.
UNSUPPORTED_MIME_TYPES = frozenset({"image/tiff", "image/tif"})
SPAN_LLM_CALL = "llmCall"
#: Low-code internal tool spans are ``toolCall`` spans named after the tool.
SPAN_INTERNAL_TOOL = "toolCall"


def analyze_files_input_schema() -> dict[str, Any]:
    """The canonical Analyze Files arguments (``attachments``, ``analysisTask``)."""
    return {
        "type": "object",
        "required": ["attachments", "analysisTask"],
        "properties": {
            "attachments": {
                "type": "array",
                "items": {"$ref": "#/definitions/job-attachment"},
                "description": (
                    "Array of files, documents, images, or other attachments to process"
                ),
            },
            "analysisTask": {
                "type": "string",
                "description": (
                    "The task, question, or instruction for processing the files"
                ),
            },
        },
        "definitions": {
            "job-attachment": {
                "type": "object",
                "required": ["ID"],
                "x-uipath-resource-kind": "JobAttachment",
                "properties": {
                    "ID": {
                        "type": "string",
                        "description": "Orchestrator attachment key",
                    },
                    "FullName": {"type": "string", "description": "File name"},
                    "MimeType": {
                        "type": "string",
                        "description": "The MIME type of the content, such as "
                        "'application/json' or 'image/png'",
                    },
                },
            }
        },
    }


def sanitize_filename(filename: str) -> str:
    """A file name the model providers accept (LangChain's rule)."""
    if not filename or filename.isspace():
        return "document"
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "-", filename)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip(" -")
    return cleaned or "document"


def _mb(size: int, decimals: int = 1) -> str:
    return f"{size / (1024 * 1024):.{decimals}f} MB"


def _too_large(detail: str) -> UiPathCodexRuntimeError:
    return UiPathCodexRuntimeError(
        UiPathCodexErrorCode.FILE_ERROR,
        "File attachment too large",
        detail,
        UiPathErrorCategory.USER,
    )


def _uipath_client() -> Any:
    """The UiPath platform client (a seam for tests)."""
    from uipath.platform import UiPath

    return UiPath()


def _http_client() -> Any:
    """httpx client for the blob download (a seam for tests)."""
    import httpx
    from uipath._utils._ssl_context import get_httpx_client_kwargs

    return httpx.AsyncClient(**get_httpx_client_kwargs())


async def _gateway_client(model: str) -> Any:
    """The LLM Gateway client of ``model`` (a seam for tests)."""
    import asyncio

    from uipath.llm_client import UiPathHttpxAsyncClient
    from uipath.llm_client.settings import get_default_client_settings

    from .runtime.gateway.upstream import UPSTREAM_TIMEOUT, build_api_config

    settings = await asyncio.to_thread(get_default_client_settings)
    return UiPathHttpxAsyncClient(
        model_name=model,
        client_settings=settings,
        api_config=build_api_config(),
        timeout=UPSTREAM_TIMEOUT,
        logger=logger,
    )


def _trace_headers() -> dict[str, str]:
    """Headers that let the LLM Gateway add its ``Model run`` span."""
    try:
        from uipath.platform.chat.llm_trace_context import (
            build_trace_context_headers,
        )

        from .runtime.telemetry import TRACE_SOURCE

        return build_trace_context_headers(extra_baggage=[TRACE_SOURCE])
    except Exception:
        logger.debug("could not build trace context headers", exc_info=True)
        return {}


def _output_text(response: Any) -> str:
    """The text of a Responses API answer."""
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _gateway_error(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return (getattr(response, "text", "") or "")[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return json.dumps(body)[:500]


@dataclass
class UiPathAnalyzeFilesTool:
    """An immediate tool that analyzes job attachments with the agent's model.

    Prefer :func:`uipath_analyze_files_tool` or
    :func:`analyze_files_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        input_json_schema: Arguments: ``attachments`` (JobAttachments) and
            ``analysisTask``.
        model: Model of the analysis call; None means the agent's model.
        timeout_sec: Host-enforced timeout; None means 300 seconds.
        fixed_args: Arguments pre-filled from the agent input or static
            values (agent.json ``argumentProperties``): static ones are
            hidden from the model, agent-input ones stay visible but
            optional (the model's value is used when the input has none).
        display_name: Name of the internal tool span (the resource name).
        max_file_bytes: Per-file size limit; a larger file fails the run.
        mock_name: Name the tool is simulated by in evals; None means the
            display name.
        mock_output_schema: Output schema of a simulated call (the
            resource's ``outputSchema``); None means ``{"analysisResult"}``.
    """

    #: The analysis is mocked inside ``call`` (under its span), not by the host.
    mocks_in_call: ClassVar[bool] = True

    name: str
    description: str
    input_json_schema: dict[str, Any] = field(
        default_factory=analyze_files_input_schema
    )
    model: str | None = None
    timeout_sec: float | None = None
    fixed_args: dict[str, FixedArg] = field(default_factory=dict)
    display_name: str | None = None
    max_file_bytes: int = MAX_FILE_BYTES
    kind: str = "internal"
    tool_type: str = "Internal"
    mock_name: str | None = None
    mock_output_schema: dict[str, Any] | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        """The arguments as the model sees them (pre-filled ones hidden)."""
        return fixed_args_schema(self.input_json_schema, self.fixed_args)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        _validate_against_schema(self.input_schema, args)
        return dict(args)

    @property
    def prefilled_schema(self) -> dict[str, Any]:
        """The arguments, pre-filled fields included (the host resolves their attachments)."""
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
        """The ``Tool call`` span's input: the arguments with the pre-filled values."""
        if not self.fixed_args:
            return args
        return with_fixed_args(args, self.fixed_args, getattr(ctx, "agent_input", None))

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        args = (
            with_fixed_args(args, self.fixed_args, ctx.agent_input)
            if self.fixed_args
            else copy.deepcopy(args)
        )
        # An agent-input field the input has no value for keeps the model's
        # value (langchain's StaticArgsHandler); a required one neither
        # filled fails the call before the mock, as langchain's schema check.
        missing = missing_from_input(self.input_json_schema, self.fixed_args, args)
        if missing:
            raise UiPathToolError(missing_args_message(self.name, missing))
        # Attachment validation runs before the mock (langchain's
        # job_attachment_wrapper); the argument checks and the no-files answer
        # are inside it (langchain's mocked tool_fn), so a simulated call is
        # always answered by the mocker.
        files = self._files(args.get("attachments"), ctx)

        from .runtime.mocking import mark_simulated, mocked_text, wrap
        from .runtime.telemetry import _json, fail_span, start_internal_tool_span

        async def analyze(**_kwargs: Any):  # the part a simulated call skips
            task = args.get("analysisTask")
            if not isinstance(task, str) or not task.strip():
                raise UiPathToolError("Argument 'analysisTask' is not available")
            if "attachments" not in args:
                raise UiPathToolError("Argument 'attachments' is not available")
            if not files:
                return {"analysisResult": NO_ATTACHMENTS_RESULT}
            parts = [await self._file_part(f) for f in files]
            text = await self._analyze(task, parts, files, ctx.model)
            return {"analysisResult": text}

        arguments = dict(args)
        if "attachments" in args:
            arguments["attachments"] = [dump_attachment(f) for f in files]
        span = start_internal_tool_span(
            self.display_name or self.name,
            arguments=arguments,
            tool_type=self.tool_type,
        )
        with trace.use_span(span, end_on_exit=False):
            try:
                result = await wrap(self.mock_spec, analyze)(**arguments)
            except BaseException as e:
                fail_span(span, e)
                span.end()
                raise
        mark_simulated(span)
        span.set_attribute("output", _json(result))
        span.set_attribute("output.mime_type", "application/json")
        span.set_status(Status(StatusCode.OK))
        span.end()
        return mocked_text(result)

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's: the resource's output schema, else ``analysisResult``."""
        from .runtime.mocking import MockSpec, mock_name_of

        return MockSpec(
            name=mock_name_of(self),
            description=self.description,
            input_schema=self.input_schema,
            output_schema=copy.deepcopy(self.mock_output_schema or {})
            or {
                "type": "object",
                "properties": {"analysisResult": {"type": "string"}},
                "required": ["analysisResult"],
            },
            example_calls=[],
        )

    def _files(self, value: Any, ctx: ToolContext) -> list[Attachment]:
        items = value if isinstance(value, list) else [value] if value else []
        files: list[Attachment] = []
        for item in items:
            attachment = to_attachment(item)
            if attachment is None and isinstance(item, dict) and ctx.attachments:
                # A pre-filled argument the host did not resolve.
                known = ctx.attachments.get(str(item.get("ID", item.get("id"))))
                attachment = known
            if attachment is None:
                raise UiPathToolError(
                    f"Tool '{self.name}' got an attachment that is not a valid "
                    "JobAttachment; pass the JobAttachment objects you were given."
                )
            files.append(attachment)
        return files

    async def _file_part(self, attachment: Attachment) -> dict[str, Any]:
        """A Responses content part with the file's bytes (base64)."""
        mime = (
            attachment.mime_type
            or mimetypes.guess_type(attachment.full_name)[0]
            or "application/octet-stream"
        ).lower()
        if mime in UNSUPPORTED_MIME_TYPES:
            raise UiPathToolError(
                f"Attachment '{attachment.full_name}' is a {mime} file, which "
                f"tool '{self.name}' cannot analyze yet."
            )
        data = await self._download(attachment)
        url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        if mime in IMAGE_MIME_TYPES:
            return {"type": "input_image", "image_url": url}
        return {
            "type": "input_file",
            "filename": sanitize_filename(attachment.full_name),
            "file_data": url,
        }

    async def _download(self, attachment: Attachment) -> bytes:
        try:
            info = await _uipath_client().attachments.get_blob_file_access_uri_async(
                key=attachment.id
            )
        except Exception as e:
            raise UiPathToolError(
                f"Could not access attachment '{attachment.full_name}' "
                f"({attachment.id}): {e}"
            ) from e
        cap = self.max_file_bytes
        async with _http_client() as http:
            async with http.stream("GET", info.uri) as response:
                if response.status_code >= 400:
                    raise UiPathToolError(
                        f"Could not download attachment '{attachment.full_name}' "
                        f"({attachment.id}): HTTP {response.status_code}"
                    )
                length = response.headers.get("content-length")
                if length and length.isdigit() and cap > 0 and int(length) > cap:
                    raise _too_large(
                        f"File '{attachment.full_name}' is {_mb(int(length))}, which "
                        f"exceeds the {_mb(cap, 0)} limit for Agent LLM payloads."
                    )
                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if cap > 0 and len(buffer) > cap:
                        raise _too_large(
                            f"File '{attachment.full_name}' exceeds the "
                            f"{_mb(cap, 0)} limit for LLM payloads (downloaded "
                            f"{_mb(len(buffer))} so far)."
                        )
        return bytes(buffer)

    async def _analyze(
        self,
        task: str,
        parts: list[dict[str, Any]],
        files: list[Attachment],
        agent_model: str | None,
    ) -> str:
        """One non-streaming Responses call under an ``LLM call`` span."""
        from .runtime.gateway.upstream import UPSTREAM_PATH
        from .runtime.telemetry import _json, get_tracer, span_attributes

        model = self.model or agent_model
        if not model:
            raise UiPathToolError(f"Tool '{self.name}' has no model to analyze with.")
        span = get_tracer().start_span(
            "LLM call",
            attributes=span_attributes(
                SPAN_LLM_CALL,
                input=task,
                model=model,
                attachments=span_attachments(files, "in"),
            ),
        )
        body = {
            "model": model,
            "instructions": ANALYZE_FILES_SYSTEM_MESSAGE,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": task}, *parts],
                }
            ],
            "stream": False,
            "store": False,
        }
        token = context_api.attach(trace.set_span_in_context(span))
        try:
            headers = {"content-type": "application/json", **_trace_headers()}
            client = await _gateway_client(model)
            try:
                response = await client.request(
                    "POST", UPSTREAM_PATH, headers=headers, content=json.dumps(body)
                )
            finally:
                await client.aclose()
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 400:
                raise UiPathToolError(
                    f"Tool '{self.name}' failed: the model returned "
                    f"{response.status_code}: {_gateway_error(response)}"
                )
            payload = response.json()
            text = _output_text(payload)
            span.set_attribute("output", _json(text))
            span.set_attribute("content", _json(text))
            usage = payload.get("usage") if isinstance(payload, dict) else None
            if usage:
                span.set_attribute("usage", _json(usage))
            span.set_status(Status(StatusCode.OK))
            return text
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e) or type(e).__name__))
            span.set_attribute("error", str(e) or type(e).__name__)
            raise
        finally:
            context_api.detach(token)
            span.end()


def uipath_analyze_files_tool(
    name: str = DEFAULT_NAME,
    *,
    description: str | None = None,
    model: str | None = None,
    tool_name: str | None = None,
    input_schema: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> UiPathAnalyzeFilesTool:
    """Declare the Analyze Files tool.

    Args:
        name: Display name (the internal tool span); the model-facing name is
            this sanitized unless ``tool_name`` is given.
        description: What the tool does, for the model.
        model: Model of the analysis call; defaults to the agent's model.
        tool_name: Model-facing tool name.
        input_schema: Argument schema; defaults to the canonical
            ``attachments`` + ``analysisTask``.
        timeout_sec: Host-enforced timeout; None means 300 seconds.
    """
    return UiPathAnalyzeFilesTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description or DEFAULT_DESCRIPTION,
        input_json_schema=dict(input_schema or analyze_files_input_schema()),
        model=model,
        timeout_sec=timeout_sec,
        display_name=name,
    )


def internal_tool_resource(resource: Any, tool_type: str) -> tuple[Any, str]:
    """An enabled ``AgentInternalToolResourceConfig`` of ``tool_type``, and its label.

    A dict is validated first; its ``type`` may be lowercase (``internal``, as
    Agent Builder writes it: only ``AgentDefinition`` normalises it).

    Raises:
        ValueError: The resource does not validate, is disabled, or is another
            internal tool.
    """
    from uipath.agent.models.agent import AgentInternalToolResourceConfig

    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "tool")
        if isinstance(data.get("type"), str) and data["type"].lower() == "internal":
            data["type"] = "Internal"
        try:
            resource = AgentInternalToolResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid internal tool resource: {e}") from e
    label = f"Tool resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    kind = getattr(getattr(resource, "properties", None), "tool_type", None)
    kind = getattr(kind, "value", kind)
    if kind != tool_type:
        raise ValueError(f"{label} is a '{kind}' internal tool, not '{tool_type}'.")
    return resource, label


def analyze_files_tool_from_resource(
    resource: Any,
    *,
    tool_name: str | None = None,
    model: str | None = None,
    timeout_sec: float | None = None,
) -> UiPathAnalyzeFilesTool:
    """The Analyze Files tool from a low-code internal ``tool`` resource.

    Args:
        resource: An ``AgentInternalToolResourceConfig``, or the resource dict
            as found in a low-code ``agent.json`` (``"$resourceType": "tool"``,
            ``"type": "Internal"``, ``properties.toolType``
            ``"analyze-attachments"``).
        tool_name: Model-facing name; defaults to the resource name sanitized.
        model: Model of the analysis call; defaults to the agent's model.
        timeout_sec: Host-enforced timeout; None means 300 seconds.

    Raises:
        ValueError: The resource is disabled, is another internal tool, or does
            not validate.
    """
    resource, label = internal_tool_resource(resource, ANALYZE_FILES_TOOL_TYPE)
    fixed = fixed_args_from_properties(label, resource.argument_properties)
    schema = resource.input_schema or analyze_files_input_schema()
    return UiPathAnalyzeFilesTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=resource.description or DEFAULT_DESCRIPTION,
        input_json_schema=dict(schema),
        model=model,
        timeout_sec=timeout_sec,
        fixed_args=fixed,
        display_name=resource.name,
        mock_name=resource.name,
        mock_output_schema=copy.deepcopy(resource.output_schema)
        if isinstance(resource.output_schema, dict)
        and resource.output_schema.get("properties")
        else None,
    )


__all__ = [
    "ANALYZE_FILES_SYSTEM_MESSAGE",
    "MAX_FILE_BYTES",
    "UiPathAnalyzeFilesTool",
    "analyze_files_input_schema",
    "analyze_files_tool_from_resource",
    "fixed_args_from_properties",
    "internal_tool_resource",
    "uipath_analyze_files_tool",
]
