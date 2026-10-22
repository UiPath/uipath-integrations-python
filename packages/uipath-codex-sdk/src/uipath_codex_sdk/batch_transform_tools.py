"""Batch Transform on a job attachment: the low-code internal tool that adds columns to a CSV.

The model passes a CSV attachment (a JobAttachment, resolved by the tool host)
and a ``query``. The tool is *durable*: one call to
``start_batch_transform_from_attachment_async`` (the service indexes the file
and starts the task) and the run suspends on it. When the task ends the run
resumes, the result CSV is downloaded, uploaded as a job attachment
(``output.csv``) and the model gets langchain's result
``{"result": {"ID", "FullName", "MimeType"}}``; the attachment is then known
to the run, so the model can pass it on or return it.

A failed task faults the run, as in langchain (the platform's trigger reader
raises on it).

Typical use::

    from uipath_codex_sdk import uipath_batch_transform_attachments_tool

    enrich = uipath_batch_transform_attachments_tool(
        "Enrich Reviews",
        output_columns=[("sentiment", "positive, neutral or negative")],
    )
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from uipath.platform.common import WaitBatchTransform

from . import tools as _tools
from ._fixed_args import (
    fixed_args_schema,
    missing_args_message,
    missing_from_input,
    with_fixed_args,
)
from .integration_tools import _is_httpx_timeout, _lowcode_tool_name
from .tools import (
    DurableToolContext,
    UiPathDurableToolError,
    UiPathToolError,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from .runtime.mocking import MockSpan, MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "BATCH_TRANSFORM_OUTPUT_SCHEMA",
    "UiPathBatchTransformAttachmentsTool",
    "WaitBatchTransformFile",
    "batch_transform_tool_from_resource",
    "uipath_batch_transform_attachments_tool",
]

#: The Agent Builder description of the Batch Transform internal tool.
DEFAULT_BATCH_TRANSFORM_DESCRIPTION = "Process and transform CSV files."
#: langchain's description of the ``query`` argument.
BATCH_TRANSFORM_QUERY_DESCRIPTION = (
    "Describe the task: what to research, what to synthesize."
)
#: ``properties.toolType`` of the low-code resource.
BATCH_TRANSFORM_TOOL_TYPE = "batch-transform"
#: The service's prompt limit.
BATCH_TRANSFORM_PROMPT_MAX = 250_000
#: Name of the uploaded result (langchain's).
DEFAULT_RESULT_NAME = "output.csv"
#: Where the trigger reader downloads results to (relative to the cwd).
RESULT_DIR = os.path.join(".uipath", "batch_transform")
#: How often an upload of the result is tried.
UPLOAD_ATTEMPTS = 3

_COMPLETED = re.compile(
    r"^Batch transform completed\. Modified file available at (.+)$"
)

#: langchain's ``BATCH_TRANSFORM_OUTPUT_SCHEMA``: the result attachment.
BATCH_TRANSFORM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "result": {
            "$ref": "#/definitions/job-attachment",
            "description": "The transformed result file as an attachment",
        }
    },
    "required": ["result"],
    "definitions": {
        "job-attachment": {
            "type": "object",
            "properties": {
                "ID": {"type": "string", "description": "Orchestrator attachment key"},
                "FullName": {"type": "string", "description": "File name"},
                "MimeType": {
                    "type": "string",
                    "description": "The MIME type of the content",
                },
            },
            "required": ["ID", "FullName", "MimeType"],
            "x-uipath-resource-kind": "JobAttachment",
        }
    },
}


class WaitBatchTransformFile(WaitBatchTransform):
    """``WaitBatchTransform`` read by codex's trigger reader, not the platform's.

    The platform's BatchRag reader downloads the result through the storage
    URL the service returns, unauthenticated and unchecked; a serverless
    robot cannot always reach that host. Its trigger payload carries
    ``reader: "codex"``, so :func:`read_batch_transform_trigger` only checks
    the task's status and ``finish`` downloads the result itself (through
    the authenticated endpoint first). ``destination_path`` keeps the payload readable
    by the platform's reader too.
    """

    destination_path: str
    reader: str = "codex"


def is_codex_batch_transform_trigger(trigger: Any) -> bool:
    """True for a BatchRag trigger of :class:`WaitBatchTransformFile`."""
    from uipath.core.triggers import UiPathResumeTriggerType

    payload = getattr(trigger, "payload", None)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return False
    return (
        getattr(trigger, "trigger_type", None) == UiPathResumeTriggerType.BATCH_RAG
        and isinstance(payload, dict)
        and payload.get("reader") == "codex"
    )


async def read_batch_transform_trigger(trigger: Any) -> dict[str, Any]:
    """The resume value of a finished task: ``{"batch_transform_id", "status"}``.

    Raises:
        UiPathPendingTriggerError: The task has not finished.
        UiPathFaultedTriggerError: The task failed (the run faults, as in
            langchain).
    """
    from uipath.core.errors import (
        ErrorCategory,
        UiPathFaultedTriggerError,
        UiPathPendingTriggerError,
    )

    task_id = str(trigger.item_key)
    task = (
        await _tools._uipath_client().context_grounding.retrieve_batch_transform_async(
            id=task_id
        )
    )
    status = str(
        getattr(task.last_batch_rag_status, "value", task.last_batch_rag_status)
    )
    if status.lower() == "failed":
        raise UiPathFaultedTriggerError(
            ErrorCategory.SYSTEM, f"Batch transform '{task_id}' failed."
        )
    if status.lower() != "successful":
        raise UiPathPendingTriggerError(
            ErrorCategory.SYSTEM,
            f"Batch transform '{task_id}' is not complete. Current status: {status}",
        )
    return {"batch_transform_id": task_id, "status": status}


def batch_transform_attachments_input_schema() -> dict[str, Any]:
    """The Agent Builder arguments: one CSV ``attachment`` (``query`` is added)."""
    from .attachment_tools import analyze_files_input_schema

    return {
        "type": "object",
        "properties": {
            "attachment": {
                "$ref": "#/definitions/job-attachment",
                "title": "Source",
                "description": "CSV file to transform",
            }
        },
        "required": ["attachment"],
        "definitions": copy.deepcopy(analyze_files_input_schema()["definitions"]),
    }


def _columns(label: str, columns: Sequence[Any]) -> list[Any]:
    """``BatchTransformOutputColumn``s; at least one, each named and described."""
    from pydantic import ValidationError
    from uipath.platform.context_grounding import BatchTransformOutputColumn

    out = []
    for column in columns:
        if isinstance(column, BatchTransformOutputColumn):
            out.append(column)
            continue
        if isinstance(column, Mapping):
            name, description = column.get("name"), column.get("description")
        elif isinstance(column, (tuple, list)) and len(column) == 2:
            name, description = column
        else:
            name, description = (
                getattr(column, "name", None),
                getattr(column, "description", None),
            )
        try:
            out.append(
                BatchTransformOutputColumn(
                    name=name or "", description=description or ""
                )
            )
        except ValidationError as e:
            raise ValueError(
                f"{label}: output column {name!r} needs a name (letters, digits, "
                "spaces, '.,!?-'; at most 500 characters) and a description."
            ) from e
    if not out:
        raise ValueError(f"{label} needs at least one output column.")
    return out


async def _folder_key(client: Any, folder_path: str | None) -> str | None:
    from .context_tools import _folder_key as folder_key

    return await folder_key(client, folder_path)


async def _job_folder_key(client: Any) -> str | None:
    """The job's folder: ``UIPATH_FOLDER_KEY``, else ``UIPATH_FOLDER_PATH`` looked up."""
    return os.environ.get("UIPATH_FOLDER_KEY") or await _folder_key(
        client, os.environ.get("UIPATH_FOLDER_PATH")
    )


@dataclass
class UiPathBatchTransformAttachmentsTool:
    """A durable tool that runs Batch Transform on a CSV job attachment.

    Prefer :func:`uipath_batch_transform_attachments_tool` or
    :func:`batch_transform_tool_from_resource` over constructing this
    directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        output_columns: The columns the task adds (``BatchTransformOutputColumn``).
        input_json_schema: Arguments without ``query``: an ``attachment``.
        query: Fixed task prompt; the model then passes no ``query``.
        query_description: Description of the ``query`` argument.
        web_search_grounding: Let the task search the web.
        fixed_args: Arguments pre-filled from the agent input or static
            values (agent.json ``argumentProperties``): static ones are
            hidden from the model, agent-input ones stay visible but
            optional (the model's value is used when the input has none).
        folder_path: Folder the task runs in; None means the job's folder.
        result_name: Name of the uploaded result attachment.
        display_name: Name of the ``contextGroundingTool`` span.
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span (low-code's).
        output_json_schema: The result's schema: its attachment becomes known.
        mock_name: Name the tool is simulated by in evals; None means the
            display name.
    """

    name: str
    description: str
    output_columns: list[Any]
    input_json_schema: dict[str, Any] = field(
        default_factory=batch_transform_attachments_input_schema
    )
    query: str | None = None
    query_description: str | None = None
    web_search_grounding: bool = False
    fixed_args: dict[str, Any] = field(default_factory=dict)
    folder_path: str | None = None
    result_name: str = DEFAULT_RESULT_NAME
    display_name: str | None = None
    kind: str = "context"
    tool_type: str = "ContextGrounding"
    output_json_schema: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(BATCH_TRANSFORM_OUTPUT_SCHEMA)
    )
    mock_name: str | None = None

    def __post_init__(self) -> None:
        from uipath.platform.context_grounding import ContextGroundingService

        label = f"Batch Transform tool '{self.name}'"
        if not hasattr(
            ContextGroundingService, "start_batch_transform_from_attachment_async"
        ):
            raise ValueError(
                f"{label}: the installed uipath-platform has no Batch Transform "
                "from attachments; upgrade it (>= 0.2.34)."
            )
        if self.query is not None and not self.query.strip():
            raise ValueError(f"{label}: the static query is empty.")
        self.output_columns = _columns(label, self.output_columns)
        props = self.input_json_schema.get("properties") or {}
        if "attachment" not in props and "attachment" not in self.fixed_args:
            raise ValueError(f"{label}: the input schema has no 'attachment' field.")

    # --- UiPathDurableTool ------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        """The arguments as the model sees them: ``query`` added, pre-filled ones hidden."""
        schema = copy.deepcopy(self.input_json_schema)
        props = schema.setdefault("properties", {})
        required = [r for r in schema.get("required") or [] if r != "query"]
        if self.query is None:
            props["query"] = {
                "type": "string",
                "description": self.query_description
                or BATCH_TRANSFORM_QUERY_DESCRIPTION,
            }
            required.append("query")
        else:
            props.pop("query", None)
        schema["required"] = required
        return fixed_args_schema(schema, self.fixed_args)

    def merged_args(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The model's arguments with the pre-filled values (they win).

        An agent-input field the input has no value for keeps the model's
        value, as in langchain.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled.
        """
        if not self.fixed_args:
            return copy.deepcopy(args)
        merged = with_fixed_args(args, self.fixed_args, agent_input)
        missing = missing_from_input(self.input_json_schema, self.fixed_args, merged)
        if missing:
            raise UiPathDurableToolError(missing_args_message(self.name, missing))
        return merged

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

    def mock_args(self, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """The arguments the mocker sees: :meth:`merged_args` (langchain's ``StaticArgsHandler``)."""
        return self.merged_args(args, getattr(ctx, "agent_input", None))

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {
            "context_retrieval_mode": "BatchTransform",
            "tool.display_name": self.display_name or self.name,
        }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        _validate_against_schema(self.input_schema, args)
        query = args.get("query")
        if isinstance(query, str) and len(query) > BATCH_TRANSFORM_PROMPT_MAX:
            raise ValueError(
                f"query is longer than {BATCH_TRANSFORM_PROMPT_MAX} characters"
            )
        return json.loads(json.dumps(args, default=str))

    def started_output(self, value: Any) -> dict[str, Any]:
        task = getattr(value, "batch_transform", None)
        return {
            "status": "started",
            "batch_transform_id": getattr(task, "id", None),
            "batch_transform_status": getattr(task, "last_batch_rag_status", None),
        }

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Start Batch Transform on the attachment; return ``WaitBatchTransformFile``.

        Raises:
            UiPathDurableToolError: No query or attachment, an unknown
                attachment, no folder, or the service refused the task.
        """
        from opentelemetry import trace
        from uipath.platform.errors import EnrichedException

        from .attachments import (
            dump_attachment,
            not_found_message,
            span_attachments,
            to_attachment,
        )
        from .runtime.telemetry import (
            end_context_grounding_span,
            hold_span,
            start_context_grounding_span,
        )

        args = self.merged_args(args, ctx.agent_input)
        query = self.query
        if query is None:
            raw_query = args.get("query")
            query = raw_query.strip() if isinstance(raw_query, str) else None
        if not query:
            raise UiPathDurableToolError("Query is required for Batch Transform tool")
        if len(query) > BATCH_TRANSFORM_PROMPT_MAX:
            raise UiPathDurableToolError(
                f"The query of tool '{self.name}' is longer than "
                f"{BATCH_TRANSFORM_PROMPT_MAX} characters."
            )
        if "attachment" not in args:
            raise UiPathDurableToolError("Argument 'attachment' is not available")
        raw = args["attachment"]
        if not raw:
            raise UiPathDurableToolError(
                "Attachment is required for Batch Transform tool"
            )
        attachment = to_attachment(raw)
        if attachment is None:
            raw_id = raw.get("ID", raw.get("id")) if isinstance(raw, dict) else None
            if not raw_id:
                raise UiPathDurableToolError("Attachment ID is required")
            attachment = ctx.attachments.get(str(raw_id)) if ctx.attachments else None
            if attachment is None:
                raise UiPathDurableToolError(not_found_message(str(raw_id)))
        args["attachment"] = dump_attachment(attachment)

        span = start_context_grounding_span(
            self.display_name or self.name,
            retrieval_mode="BatchTransform",
            query=query,
            output_columns=[c.model_dump() for c in self.output_columns],
            web_search_grounding=self.web_search_grounding,
            file_extension=attachment.mime_type,
            attachments=span_attachments([attachment], "in"),
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                client = _tools._uipath_client()
                if self.folder_path:
                    folder_key = await _folder_key(client, self.folder_path)
                    if not folder_key:
                        raise UiPathDurableToolError(
                            f"Folder '{self.folder_path}' was not found."
                        )
                else:
                    folder_key = await _job_folder_key(client)
                if not folder_key:
                    raise UiPathDurableToolError(
                        "Batch Transform on an attachment needs a folder: the run "
                        "has none (set UIPATH_FOLDER_PATH for a local run, or the "
                        "tool's folder_path)."
                    )
                started = await client.context_grounding.start_batch_transform_from_attachment_async(
                    name=f"task-{uuid.uuid4()}",
                    prompt=query,
                    output_columns=self.output_columns,
                    attachment=str(attachment.id),
                    enable_web_search_grounding=self.web_search_grounding,
                    folder_key=folder_key,
                    folder_path=None,
                )
        except BaseException as e:
            end_context_grounding_span(span, error=e)
            if isinstance(e, UiPathDurableToolError):
                raise
            if isinstance(e, EnrichedException):
                status = getattr(e, "status_code", None)
                info = getattr(e, "error_info", None)
                message = getattr(info, "message", None) or str(e).strip()
                raise UiPathDurableToolError(
                    f"Could not start Batch Transform on attachment "
                    f"'{attachment.full_name}' (HTTP {status}): {message}"
                ) from e
            if _is_httpx_timeout(e):
                raise UiPathDurableToolError(
                    f"Starting Batch Transform for tool '{self.name}' timed out; it "
                    "may have started. Retry only if needed."
                ) from e
            raise
        # Saved with the held span: set before hold_span.
        span.set_attribute("batch_transform_id", started.id)
        hold_span(span, result_attachment_key="result")
        logger.info(
            "Batch Transform tool %s started task %s on attachment %s (call %s)",
            self.name,
            started.id,
            attachment.id,
            ctx.call_id,
        )
        return WaitBatchTransformFile(
            batch_transform=started,
            index_folder_key=folder_key,
            # Deterministic: the reader may download more than once.
            destination_path=os.path.join(RESULT_DIR, f"{started.id}.csv"),
        )

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's: the whole ``{"result": <attachment>}``.

        Unlike langchain's attachment tool, the result is not wrapped in a
        second ``result``. The simulated attachment does not exist.
        """
        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.display_name or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=copy.deepcopy(BATCH_TRANSFORM_OUTPUT_SCHEMA),
            example_calls=[],
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The ``contextGroundingTool`` child, without a task.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled (before the child opens, as
                a real start fails before its span).
        """
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_context_grounding_span

        merged = self.merged_args(args, getattr(ctx, "agent_input", None))
        query = self.query
        if query is None:
            raw = merged.get("query")
            query = raw.strip() if isinstance(raw, str) else ""
        span = start_context_grounding_span(
            self.display_name or self.name,
            retrieval_mode="BatchTransform",
            query=query,
            output_columns=[c.model_dump() for c in self.output_columns],
            web_search_grounding=self.web_search_grounding,
        )
        return MockSpan(span)

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated result as is; the host makes its attachment known."""
        return json.dumps(value, default=str)

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Upload the downloaded result as a job attachment; langchain's result.

        The resume value names the finished task (codex's reader), which is
        downloaded here, or the file the platform's reader downloaded. The
        upload goes to the job's folder and is linked to the job (when there
        is one); a link failure keeps the uploaded attachment.
        """
        client = _tools._uipath_client()
        if isinstance(resume_value, dict) and resume_value.get("batch_transform_id"):
            task_id = str(resume_value["batch_transform_id"])
            path = Path(RESULT_DIR, f"{_safe_id(task_id)}.csv").resolve()
            await _download_result(client, task_id, path)
        else:
            path = _result_path(resume_value)
        folder_key = await _job_folder_key(client)
        attachment_id = await _upload(client, self.result_name, path, folder_key)
        job_key = os.environ.get("UIPATH_JOB_KEY")
        if job_key:
            try:
                await client.jobs.link_attachment_async(
                    attachment_key=attachment_id,
                    job_key=uuid.UUID(job_key),
                    folder_key=folder_key,
                )
            except Exception:
                logger.warning(
                    "Batch Transform result %s was uploaded but could not be linked "
                    "to job %s",
                    attachment_id,
                    job_key,
                    exc_info=True,
                )
        try:
            path.unlink()
        except OSError:
            pass
        return json.dumps(
            {
                "result": {
                    "ID": str(attachment_id),
                    "FullName": self.result_name,
                    "MimeType": "text/csv",
                }
            }
        )


def _safe_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]+", task_id):
        raise UiPathToolError(f"Unexpected Batch Transform id: {task_id[:80]}")
    return task_id


async def _download_result(client: Any, task_id: str, path: Path) -> None:
    """Download a finished task's result to ``path``.

    The service's authenticated download endpoint first: it goes through the
    UiPath host, which every robot reaches (a serverless robot cannot resolve
    the storage host of the read URL). The read URL (unauthenticated, as the
    platform downloads) is the fallback. Every response is checked.
    """
    import httpx
    from uipath._utils._ssl_context import get_httpx_client_kwargs
    from uipath.platform.context_grounding.context_grounding import (
        BatchTransformReadUriResponse,
    )

    cg = client.context_grounding
    try:
        spec = cg._batch_transform_download_blob_spec(id=task_id)
        blob = await cg.request_async(spec.method, spec.endpoint, headers=spec.headers)
        content = blob.content
    except Exception as e:
        logger.info(
            "Batch Transform result %s: the authenticated download failed (%s); "
            "trying its storage URL",
            task_id,
            e,
        )
        spec = cg._batch_transform_get_read_uri_spec(id=task_id)
        response = await cg.request_async(
            spec.method, spec.endpoint, headers=spec.headers
        )
        read_uri = BatchTransformReadUriResponse.model_validate(response.json())
        try:
            async with httpx.AsyncClient(**get_httpx_client_kwargs()) as http:
                stored = await http.get(read_uri.uri)
                stored.raise_for_status()
                content = stored.content
        except httpx.HTTPError:
            raise e from None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _result_path(resume_value: Any) -> Path:
    """The downloaded result named by the trigger reader's resume value."""
    match = _COMPLETED.match(resume_value) if isinstance(resume_value, str) else None
    if match is None:
        raise UiPathToolError(
            f"Unexpected Batch Transform resume value: {str(resume_value)[:200]}"
        )
    path = Path(match.group(1)).resolve()
    allowed = Path(RESULT_DIR).resolve()
    if path.parent != allowed or not path.is_file():
        raise UiPathToolError(f"The Batch Transform result {path} is not available.")
    return path


async def _upload(client: Any, name: str, path: Path, folder_key: str | None) -> Any:
    """``attachments.upload_async``, tried :data:`UPLOAD_ATTEMPTS` times."""
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            return await client.attachments.upload_async(
                name=name, source_path=str(path), folder_key=folder_key
            )
        except Exception:
            if attempt == UPLOAD_ATTEMPTS:
                raise
            logger.warning(
                "Uploading the Batch Transform result failed (attempt %s); retrying",
                attempt,
                exc_info=True,
            )
            await asyncio.sleep(attempt)
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def uipath_batch_transform_attachments_tool(
    name: str = "Batch Transform",
    *,
    output_columns: Sequence[Any],
    description: str | None = None,
    query: str | None = None,
    query_description: str | None = None,
    web_search_grounding: bool = False,
    attachment_from_input: str | None = None,
    folder_path: str | None = None,
    result_name: str = DEFAULT_RESULT_NAME,
    tool_name: str | None = None,
) -> UiPathBatchTransformAttachmentsTool:
    """A durable tool that runs Batch Transform on a CSV job attachment.

    The model passes an ``attachment`` and a ``query``; the run suspends
    until the task ends, and the model then gets the result file as a
    JobAttachment, ``{"result": {"ID", "FullName", "MimeType"}}``.

    Args:
        name: Display name; the model-facing name is it sanitized the way
            low-code agents do, unless ``tool_name`` is given.
        output_columns: The columns to add: ``(name, description)`` pairs,
            ``{"name", "description"}`` dicts or ``BatchTransformOutputColumn``.
        description: Shown to the model; defaults to Agent Builder's.
        query: A fixed task prompt: the model passes no ``query``.
        query_description: Description of the ``query`` argument.
        web_search_grounding: Let the task search the web.
        attachment_from_input: Path of the attachment in the agent input. Its
            value wins; ``attachment`` stays visible to the model but
            optional, and the model's is used when the input has none.
        folder_path: Folder the task runs in; defaults to the job's folder.
        result_name: Name of the uploaded result (default ``output.csv``).
        tool_name: Model-facing name override.

    Raises:
        ValueError: No or invalid output columns, an empty query, or an SDK
            without Batch Transform from attachments.
    """
    from .context_tools import _input_path
    from .escalation_tools import FixedArg

    fixed: dict[str, Any] = {}
    if attachment_from_input:
        fixed["attachment"] = FixedArg(path=_input_path(attachment_from_input))
    return UiPathBatchTransformAttachmentsTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description or DEFAULT_BATCH_TRANSFORM_DESCRIPTION,
        output_columns=list(output_columns),
        query=query,
        query_description=query_description,
        web_search_grounding=web_search_grounding,
        fixed_args=fixed,
        folder_path=folder_path,
        result_name=result_name,
        display_name=_lowcode_tool_name(name),
        mock_name=name,
    )


def batch_transform_tool_from_resource(
    resource: Any,
    *,
    tool_name: str | None = None,
) -> UiPathBatchTransformAttachmentsTool:
    """The Batch Transform tool from a low-code internal ``tool`` resource.

    Args:
        resource: An ``AgentInternalToolResourceConfig``, or the resource dict
            as found in a low-code ``agent.json`` (``"$resourceType": "tool"``,
            ``"type": "Internal"``, ``properties.toolType``
            ``"batch-transform"``). Its query (static, or written by the
            model), output columns, web search setting and static, argument
            and text-builder ``argumentProperties`` are used. ``fileExtension``
            and the resource's ``outputSchema`` are ignored, as in langchain;
            ``folderPathPrefix`` is ignored too (the attachment API has none).
        tool_name: Model-facing name; defaults to the resource name sanitized
            the way low-code agents do.

    Raises:
        ValueError: The resource is disabled, is another internal tool, has a
            static query without a value, no or invalid output columns, or
            does not validate.
    """
    from .attachment_tools import fixed_args_from_properties, internal_tool_resource
    from .context_tools import _value

    resource, label = internal_tool_resource(resource, BATCH_TRANSFORM_TOOL_TYPE)
    settings = resource.properties.settings
    setting = settings.query
    variant = str(_value(getattr(setting, "variant", "")) or "").lower()
    query: str | None = None
    if variant == "static":
        query = str(getattr(setting, "value", None) or "").strip() or None
        if query is None:
            raise ValueError(
                f"{label}: the static query has no value. Set one, or make the "
                "query dynamic so the model writes it."
            )
    prefix = getattr(settings.folder_path_prefix, "value", None)
    if prefix:
        logger.warning(
            "%s: folderPathPrefix %r is ignored (Batch Transform on an attachment "
            "has no folder prefix).",
            label,
            prefix,
        )
    web = getattr(settings.web_search_grounding, "value", None)
    display = _lowcode_tool_name(resource.name)
    return UiPathBatchTransformAttachmentsTool(
        name=sanitize_tool_name(tool_name) if tool_name else display,
        description=resource.description or DEFAULT_BATCH_TRANSFORM_DESCRIPTION,
        output_columns=[
            {"name": c.name, "description": c.description}
            for c in settings.output_columns
        ],
        input_json_schema=copy.deepcopy(
            resource.input_schema or batch_transform_attachments_input_schema()
        ),
        query=query,
        query_description=getattr(setting, "description", None),
        web_search_grounding=str(_value(web) or "") == "Enabled",
        fixed_args=fixed_args_from_properties(label, resource.argument_properties),
        display_name=display,
        mock_name=resource.name,
    )
