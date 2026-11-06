"""IXP extraction and its Document Validation escalation, as in low-code agents.

Two durable tools that work as a pair:

* :class:`UiPathIxpExtractionTool` (agent.json ``tool`` of ``type: "Ixp"``):
  the model passes a document (a JobAttachment); the tool downloads it,
  starts an IXP extraction on the project's tagged model version and
  suspends the run. On resume the model gets the extracted fields (the
  ``dataProjection``), and the full extraction response is kept for the
  validation tool.
* :class:`UiPathDocumentValidationTool` (agent.json ``escalation`` with
  ``escalationType: 1``): takes no arguments; it creates an Action Center
  Validation Station task for the last extraction of its IXP tool and
  suspends. On resume the model gets the validated fields, ``{"data": ...}``;
  a document marked as an exception (or a deleted task) ends the run with
  ``TERMINATION_ESCALATION_REJECTED``, as in langchain.

As in langchain, a call without a document, a download or start error and a
failed extraction or validation operation fail the run.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import ValidationError
from uipath.runtime.errors import UiPathErrorCategory

from . import tools as _tools
from .integration_tools import _get_path, _lowcode_tool_name
from .runtime.errors import (
    UiPathCodexErrorCode,
    UiPathCodexRuntimeError,
    UiPathRunTerminatedError,
)
from .tools import (
    DurableToolContext,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from .runtime.mocking import MockSpan, MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "UiPathDocumentValidationTool",
    "UiPathIxpExtractionTool",
    "ixp_extraction_tool_from_resource",
    "uipath_document_validation_tool",
    "uipath_ixp_extraction_tool",
    "vs_escalation_tool_from_resource",
]

DEFAULT_EXTRACTION_DESCRIPTION = (
    "Extracts structured data from a document with UiPath IXP."
)
DEFAULT_VALIDATION_DESCRIPTION = (
    "Sends the last extracted document to a person for validation in Action "
    "Center and returns the validated fields."
)
#: langchain's default Validation Station task title.
DEFAULT_VALIDATION_TITLE = "VS Escalation Task"
_VS_PENDING = frozenset({"pending", "unassigned"})


def ixp_extraction_input_schema() -> dict[str, Any]:
    """langchain's (and Agent Builder's) one-document argument schema."""
    from uipath.platform.attachments import Attachment

    attachment = Attachment.model_json_schema(by_alias=True)
    attachment["required"] = ["ID"]
    attachment["x-uipath-resource-kind"] = "JobAttachment"
    attachment.pop("title", None)
    return {
        "type": "object",
        "properties": {
            "attachment": {
                "description": "File to extract data from.",
                "$ref": "#/definitions/job-attachment",
            }
        },
        "required": ["attachment"],
        "definitions": {"job-attachment": attachment},
    }


def _fatal(title: str, detail: str) -> UiPathCodexRuntimeError:
    return UiPathCodexRuntimeError(
        UiPathCodexErrorCode.IXP_FAILED, title, detail, UiPathErrorCategory.USER
    )


def _value(enum_or_str: Any) -> Any:
    return getattr(enum_or_str, "value", enum_or_str)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
@dataclass
class UiPathIxpExtractionTool:
    """A durable tool that extracts a document's fields with an IXP project.

    Prefer :func:`uipath_ixp_extraction_tool` or
    :func:`ixp_extraction_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        project_name: The IXP project.
        version_tag: The tag of the project's published model version.
        input_json_schema: Arguments; the first attachment field is the
            document.
        display_name: Name of the ``ixpTool`` span and the key the
            validation tool finds the result under (the resource name).
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span (low-code's).
        mock_name: Name the tool is simulated by in evals; None means the
            display name (the resource name).
        example_calls: Example calls for the eval LLM mocker.
    """

    name: str
    description: str
    project_name: str
    version_tag: str
    input_json_schema: dict[str, Any] = field(
        default_factory=ixp_extraction_input_schema
    )
    display_name: str | None = None
    kind: str = "ixp"
    tool_type: str = "IxpExtraction"
    mock_name: str | None = None
    example_calls: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        from .attachments import attachment_paths

        label = f"IXP tool '{self.name}'"
        if not self.project_name:
            raise ValueError(f"{label}: project_name is required.")
        if not self.version_tag:
            raise ValueError(f"{label}: version_tag (the model's tag) is required.")
        if not attachment_paths(self.input_json_schema):
            self.input_json_schema = ixp_extraction_input_schema()

    @property
    def result_key(self) -> str:
        return self.display_name or self.name

    # --- UiPathDurableTool ------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_json_schema

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {"tool.display_name": self.result_key}

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        _validate_against_schema(self.input_schema, args)
        return json.loads(json.dumps(args, default=str))

    def started_output(self, value: Any) -> dict[str, Any]:
        extraction = getattr(value, "extraction", None)
        return {
            "status": "started",
            "extraction_id": getattr(extraction, "operation_id", None),
            "document_id": getattr(extraction, "document_id", None),
        }

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Download the document, start the extraction; ``WaitDocumentExtraction``.

        Raises:
            UiPathCodexRuntimeError: No document, or it could not be downloaded
                or extracted (the run fails, as in langchain).
        """
        from opentelemetry import trace
        from uipath.platform.common import WaitDocumentExtraction

        from .attachments import (
            dump_attachment,
            not_found_message,
            span_attachments,
            to_attachment,
            values_at,
        )
        from .runtime.telemetry import fail_span, hold_span, start_ixp_tool_span
        from .runtime.tool_host import input_paths

        attachment = None
        for value in values_at(args, input_paths(self)):
            attachment = to_attachment(value)
            if attachment is None and isinstance(value, dict) and ctx.attachments:
                raw_id = value.get("ID", value.get("id"))
                attachment = ctx.attachments.get(str(raw_id)) if raw_id else None
                if attachment is None and raw_id:
                    raise _fatal(
                        "Unknown job attachment", not_found_message(str(raw_id))
                    )
            if attachment is not None:
                break
        if attachment is None:
            raise _fatal(
                "Missing job attachment",
                f"Tool '{self.result_key}' was called without a job attachment to "
                "extract data from.",
            )

        span = start_ixp_tool_span(
            self.result_key,
            arguments={**args, "attachment": dump_attachment(attachment)},
            project_name=self.project_name,
            version_tag=self.version_tag,
            attachments=span_attachments([attachment], "in"),
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                started = await _start_extraction(
                    attachment, self.project_name, self.version_tag
                )
        except BaseException as e:
            fail_span(span, e)
            span.end()
            raise
        span.set_attribute("extractionId", started.operation_id)
        span.set_attribute("documentId", started.document_id)
        hold_span(span, output_attribute="result")
        if ctx.results is not None:
            ctx.results.mark_pending(self.result_key)
        logger.info(
            "IXP tool %s started extraction %s of %s (call %s)",
            self.name,
            started.operation_id,
            attachment.full_name,
            ctx.call_id,
        )
        return WaitDocumentExtraction(extraction=started)

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """The extracted fields (``dataProjection``) as JSON."""
        response = _extraction_response(resume_value)
        projection = response.model_dump(by_alias=True, mode="json").get(
            "dataProjection"
        )
        return json.dumps(
            projection if projection is not None else [], ensure_ascii=False
        )

    def stored_result(self, args: dict[str, Any], resume_value: Any) -> tuple[str, Any]:
        """The full extraction response, for the Document Validation tool."""
        response = _extraction_response(resume_value)
        return self.result_key, response.model_dump(by_alias=True, mode="json")

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's extraction tool: an ``ExtractionResponseIXP``."""
        import uipath.platform.common  # noqa: F401 - before documents (import cycle)
        from uipath.platform.documents import ExtractionResponseIXP

        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.result_key,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=ExtractionResponseIXP.model_json_schema(by_alias=True),
            example_calls=list(self.example_calls),
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The ``ixpTool`` child, without an extraction or document id."""
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_ixp_tool_span

        span = start_ixp_tool_span(
            self.result_key,
            arguments=args,
            project_name=self.project_name,
            version_tag=self.version_tag,
        )
        return MockSpan(span, "result")

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """Keep the simulated extraction for the validation tool; its ``dataProjection``.

        Raises:
            UiPathDurableToolError: The value is not an extraction response.
        """
        key, stored = self.stored_result(args, value)
        results = getattr(ctx, "results", None)
        if results is not None:
            results.put(key, stored)
        return await self.finish(args, value)


def _extraction_response(value: Any) -> Any:
    from uipath.platform.documents import ExtractionResponseIXP

    if isinstance(value, ExtractionResponseIXP):
        return value
    try:
        return ExtractionResponseIXP.model_validate(value)
    except ValidationError as e:
        raise UiPathDurableToolError(
            f"Unexpected IXP extraction result: {str(value)[:200]}"
        ) from e


async def _start_extraction(attachment: Any, project_name: str, tag: str) -> Any:
    """Download ``attachment`` to a temporary folder and start its extraction."""
    import uipath.platform.common  # noqa: F401 - before documents (import cycle)
    from uipath.platform.errors import EnrichedException

    client = _tools._uipath_client()
    name = Path(str(attachment.full_name).replace("\\", "/")).name or "document"
    with tempfile.TemporaryDirectory(prefix="codex-ixp-") as folder:
        path = os.path.join(folder, name)
        try:
            await client.attachments.download_async(
                key=attachment.id, destination_path=path
            )
        except Exception as e:
            raise _fatal(
                "Failed to download job attachment",
                f"Attachment '{attachment.full_name}' ({attachment.id}) could not "
                f"be downloaded: {e}",
            ) from e
        try:
            return await client.documents.start_ixp_extraction_async(
                project_name=project_name, tag=tag, file_path=path
            )
        except (EnrichedException, ValueError, Exception) as e:
            raise _fatal(
                "IXP extraction could not start",
                f"Extraction of '{attachment.full_name}' with IXP project "
                f"'{project_name}' (tag '{tag}') could not start: {e}",
            ) from e


# --------------------------------------------------------------------------
# document validation
# --------------------------------------------------------------------------
@dataclass
class UiPathDocumentValidationTool:
    """A durable tool that sends the last extraction to Validation Station.

    Prefer :func:`uipath_document_validation_tool` or
    :func:`vs_escalation_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        extraction_tool: The IXP tool whose last result is validated (its
            display name: the agent.json ``ixpToolId``).
        storage_bucket_name: Orchestrator storage bucket the document is
            stored in for the task.
        storage_bucket_directory_path: Directory inside the bucket.
        action_folder: Orchestrator folder of the bucket (and the task); None
            means the job's (``UIPATH_FOLDER_PATH``). DU looks the bucket up
            in this folder only.
        action_catalog: Action catalog of the task.
        priority: ``Low`` / ``Medium`` / ``High`` / ``Critical``.
        title: Fixed task title.
        title_from_input: Path of the title in the agent input.
        title_tokens: Text tokens of the title (agent.json text builder).
        display_name: Name of the ``vsEscalationTool`` span.
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span (low-code's).
        mock_name: Name the tool is simulated by in evals; None means the
            display name (the resource name).
    """

    name: str
    description: str
    extraction_tool: str
    storage_bucket_name: str | None = None
    storage_bucket_directory_path: str | None = None
    action_folder: str | None = None
    action_catalog: str | None = None
    priority: str | None = None
    title: str | None = None
    title_from_input: str | None = None
    title_tokens: list[Any] | None = None
    display_name: str | None = None
    kind: str = "escalation"
    tool_type: str = "Escalation"
    mock_name: str | None = None

    def __post_init__(self) -> None:
        if not self.extraction_tool:
            raise ValueError(
                f"Document validation tool '{self.name}': extraction_tool is required."
            )

    # --- UiPathDurableTool ------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {"tool.display_name": self.display_name or self.name}

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        return {}

    def started_output(self, value: Any) -> dict[str, Any]:
        validation = getattr(value, "extraction_validation", None)
        return {
            "status": "started",
            "operation_id": getattr(validation, "operation_id", None),
            "task_url": getattr(value, "task_url", None),
        }

    def resolve_title(self, agent_input: dict[str, Any] | None) -> str:
        if self.title_from_input:
            found, value = _get_path(agent_input, self.title_from_input)
            if found and value not in (None, ""):
                return str(value)
        if self.title_tokens is not None:
            from uipath.agent.utils.text_tokens import build_string_from_tokens

            built = build_string_from_tokens(self.title_tokens, agent_input or {})
            if built:
                return built
        return self.title or DEFAULT_VALIDATION_TITLE

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Create the Validation Station task; ``WaitDocumentExtractionValidation``.

        Raises:
            UiPathDurableToolError: The IXP tool has no result yet (the model
                must call it first and wait for it).
            UiPathCodexRuntimeError: The task could not be created (the run
                fails, as in langchain).
        """
        import uipath.platform.common  # noqa: F401 - before documents (import cycle)
        from opentelemetry import trace
        from uipath.platform.common import WaitDocumentExtractionValidation
        from uipath.platform.documents import (
            ActionPriority,
            ExtractionResponseIXP,
            StartExtractionValidationResponse,
        )

        from .runtime.telemetry import (
            fail_span,
            hold_span,
            start_vs_escalation_tool_span,
        )

        key = self.extraction_tool
        saved = self._require_extraction(ctx.results)
        extraction = ExtractionResponseIXP.model_validate(saved)
        span = start_vs_escalation_tool_span(
            self.display_name or self.name,
            ixp_tool_id=key,
            storage_bucket_name=self.storage_bucket_name,
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                client = _tools._uipath_client()
                try:
                    action = await client.documents.create_validate_extraction_action_async(
                        extraction_response=extraction,
                        action_title=self.resolve_title(ctx.agent_input),
                        action_priority=ActionPriority.from_str(self.priority),
                        action_catalog=self.action_catalog,
                        # DU finds the bucket (and puts the task) only in
                        # this folder: default to the job's.
                        action_folder=self.action_folder
                        or os.environ.get("UIPATH_FOLDER_PATH")
                        or None,
                        storage_bucket_name=self.storage_bucket_name,
                        storage_bucket_directory_path=self.storage_bucket_directory_path,
                    )
                except Exception as e:
                    raise _fatal(
                        "Document validation task could not be created",
                        f"The Validation Station task for '{key}' could not be "
                        f"created: {e}",
                    ) from e
        except BaseException as e:
            fail_span(span, e)
            span.end()
            raise
        task_url = (action.action_data or {}).get("taskUrl")
        span.set_attribute("operationId", action.operation_id)
        if task_url:
            span.set_attribute("taskUrl", str(task_url))
        hold_span(span, output_attribute="result")
        logger.info(
            "Document validation tool %s created task %s (operation %s, call %s)",
            self.name,
            task_url,
            action.operation_id,
            ctx.call_id,
        )
        return WaitDocumentExtractionValidation(
            extraction_validation=StartExtractionValidationResponse.model_validate(
                {
                    "operationId": action.operation_id,
                    "documentId": extraction.extraction_result.document_id,
                    "projectId": extraction.project_id,
                    "tag": extraction.tag,
                }
            ),
            task_url=task_url,
        )

    def _require_extraction(self, results: Any) -> Any:
        """The saved result of the extraction tool (real or simulated).

        Raises:
            UiPathDurableToolError: The extraction has not returned yet, or
                was never called.
        """
        key = self.extraction_tool
        if results is not None and results.is_pending(key):
            raise UiPathDurableToolError(
                f"The extraction of '{key}' has not finished yet. Wait for its "
                f"result, then call '{self.name}'."
            )
        saved = results.get(key) if results is not None else None
        if not saved:
            raise UiPathDurableToolError(
                f"Extraction result not found for {key} ixp extraction tool. "
                f"Call it first and wait for its result, then call '{self.name}'."
            )
        return saved

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's VS escalation: ``{"data": [field groups] | null}``."""
        import uipath.platform.common  # noqa: F401 - before documents (import cycle)
        from pydantic import create_model
        from uipath.platform.documents import FieldGroupValueProjection

        from .runtime.mocking import MockSpec

        output = create_model(
            "OutputSchema",
            data=(list[FieldGroupValueProjection] | None, None),
        )
        return MockSpec(
            name=self.mock_name or self.display_name or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=output.model_json_schema(),
            example_calls=[],
        )

    def mock_precheck(self, args: dict[str, Any], ctx: Any) -> None:
        """A simulated validation still needs an extraction, as in langchain."""
        self._require_extraction(getattr(ctx, "results", None))

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The ``vsEscalationTool`` child, without a task."""
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_vs_escalation_tool_span

        span = start_vs_escalation_tool_span(
            self.display_name or self.name,
            ixp_tool_id=self.extraction_tool,
            storage_bucket_name=self.storage_bucket_name,
        )
        return MockSpan(span, "result")

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated validated fields, ``{"data": ...}``."""
        data = value.get("data") if isinstance(value, dict) else None
        return json.dumps({"data": data}, ensure_ascii=False, default=str)

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """The validated fields, ``{"data": <dataProjection>}`` (langchain's).

        Raises:
            UiPathDurableToolPending: The task is still open.
            UiPathRunTerminatedError: The document was marked as an exception,
                or the task was deleted (the run ends).
        """
        data = _validation_data(resume_value)
        action_data = data.get("actionData") or {}
        status = str(data.get("actionStatus") or action_data.get("status") or "")
        if status.lower() in _VS_PENDING:
            raise UiPathDurableToolPending(f"validation task is {status}")
        if status.lower() == "deleted":
            raise UiPathRunTerminatedError(
                "VS Escalation Exception",
                "Agent run ended because the document validation task was deleted.",
                json.dumps({"status": "deleted"}),
            )
        rejection = (action_data.get("data") or {}).get("documentRejectionDetails")
        if rejection:
            reason = rejection.get("reason") if isinstance(rejection, dict) else None
            raise UiPathRunTerminatedError(
                "VS Escalation Exception",
                "Vs Escalation Exception: "
                + (reason or "The validation was marked as exception."),
                json.dumps({"status": "rejected", "reason": reason}),
            )
        return json.dumps({"data": data.get("dataProjection")}, ensure_ascii=False)


def _validation_data(value: Any) -> dict[str, Any]:
    from uipath.platform.documents import ValidateExtractionAction

    if isinstance(value, ValidateExtractionAction):
        return value.model_dump(by_alias=True, mode="json")
    if isinstance(value, dict):
        try:
            return ValidateExtractionAction.model_validate(value).model_dump(
                by_alias=True, mode="json"
            )
        except ValidationError:
            if "actionData" in value or "dataProjection" in value:
                return value  # a raw --resume answer
    raise UiPathDurableToolError(
        f"Unexpected document validation result: {str(value)[:200]}"
    )


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def uipath_ixp_extraction_tool(
    name: str,
    *,
    project_name: str,
    version_tag: str,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
    tool_name: str | None = None,
) -> UiPathIxpExtractionTool:
    """A durable tool that extracts a document's fields with an IXP project.

    Args:
        name: Display name (the ``ixpTool`` span, and what a
            :func:`uipath_document_validation_tool` names as its
            ``extraction_tool``); the model-facing name is it sanitized the
            way low-code agents do, unless ``tool_name`` is given.
        project_name: The IXP project.
        version_tag: The tag of its published model version (e.g. ``live``).
        description: Shown to the model.
        input_schema: Argument schema with an attachment field; defaults to
            one ``attachment``.
        tool_name: Model-facing name override.
    """
    return UiPathIxpExtractionTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description or DEFAULT_EXTRACTION_DESCRIPTION,
        project_name=project_name,
        version_tag=version_tag,
        input_json_schema=copy.deepcopy(input_schema or ixp_extraction_input_schema()),
        display_name=name,
    )


def uipath_document_validation_tool(
    name: str,
    *,
    extraction_tool: str | UiPathIxpExtractionTool,
    storage_bucket_name: str | None = None,
    storage_bucket_directory_path: str | None = None,
    action_folder: str | None = None,
    action_catalog: str | None = None,
    priority: str | None = None,
    title: str | None = None,
    title_from_input: str | None = None,
    description: str | None = None,
    tool_name: str | None = None,
) -> UiPathDocumentValidationTool:
    """A durable tool that sends the last extraction to Validation Station.

    Args:
        name: Display name; the model-facing name is it sanitized the way
            low-code agents do, unless ``tool_name`` is given.
        extraction_tool: The IXP tool (or its display name) to validate.
        storage_bucket_name: Storage bucket the document is kept in for the
            task.
        storage_bucket_directory_path: Directory inside the bucket.
        action_folder: Folder of the task and bucket; defaults to the job's.
        action_catalog: Action catalog of the task.
        priority: ``Low`` / ``Medium`` (default) / ``High`` / ``Critical``.
        title: Task title (default ``VS Escalation Task``).
        title_from_input: Path of the title in the agent input.
        description: Shown to the model.
        tool_name: Model-facing name override.
    """
    key = (
        extraction_tool.result_key
        if isinstance(extraction_tool, UiPathIxpExtractionTool)
        else extraction_tool
    )
    return UiPathDocumentValidationTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description
        or f"{DEFAULT_VALIDATION_DESCRIPTION} Call it only after '{key}' has "
        "returned its result.",
        extraction_tool=key,
        storage_bucket_name=storage_bucket_name,
        storage_bucket_directory_path=storage_bucket_directory_path,
        action_folder=action_folder,
        action_catalog=action_catalog,
        priority=priority,
        title=title,
        title_from_input=title_from_input,
        display_name=name,
    )


def ixp_extraction_tool_from_resource(
    resource: Any,
    *,
    tool_name: str | None = None,
) -> UiPathIxpExtractionTool:
    """The IXP extraction tool from a low-code ``tool`` resource (``type: "Ixp"``).

    Uses the project name, version tag, description and input schema (the
    first attachment field is the document). ``outputSchema`` and
    ``settings`` are ignored, as in langchain.

    Raises:
        ValueError: The resource is disabled, has no project or tag, or does
            not validate.
    """
    from uipath.agent.models.agent import AgentIxpExtractionResourceConfig

    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "tool")
        if isinstance(data.get("type"), str) and data["type"].lower() == "ixp":
            data["type"] = "Ixp"
        try:
            resource = AgentIxpExtractionResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid IXP tool resource: {e}") from e
    label = f"Tool resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    props = resource.properties
    if not props.project_name:
        raise ValueError(f"{label}: Project is required for IXP tool.")
    if not props.version_tag:
        raise ValueError(f"{label}: Model is required for IXP tool.")
    return UiPathIxpExtractionTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=resource.description or DEFAULT_EXTRACTION_DESCRIPTION,
        project_name=props.project_name,
        version_tag=props.version_tag,
        input_json_schema=copy.deepcopy(
            resource.input_schema or ixp_extraction_input_schema()
        ),
        display_name=resource.name,
        mock_name=resource.name,
        example_calls=list(props.example_calls or []),
    )


def vs_escalation_tool_from_resource(
    resource: Any,
    *,
    tool_name: str | None = None,
) -> UiPathDocumentValidationTool:
    """The Document Validation tool from a low-code ``escalation`` resource.

    ``escalationType: 1`` (Agent Builder's "Document Validation Task"): the
    first channel must be an Action Center channel; its task title (text,
    text builder or agent input) and priority are used, with the
    ``vsEscalationProperties`` (IXP tool, storage bucket, and the bucket's
    folder, which becomes the task's ``action_folder``). Recipients, app, labels and outcome mapping have no effect, as in
    langchain.

    Raises:
        ValueError: The resource is disabled, has no Action Center channel,
            no IXP tool or storage bucket, or does not validate.
    """
    from uipath.agent.models.agent import (
        AgentEscalationChannel,
        AgentIxpVsEscalationResourceConfig,
        DynamicTaskTitle,
        TextBuilderTaskTitle,
    )

    label = f"Escalation resource '{getattr(resource, 'name', None) or (resource.get('name') if isinstance(resource, Mapping) else '')}'"
    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "escalation")
        props = data.get("vsEscalationProperties") or {}
        if not (isinstance(props, dict) and props.get("ixpToolId")):
            raise ValueError(
                f"{label}: IXP tool is required for Document Validation Task."
            )
        if not props.get("storageBucketName"):
            raise ValueError(
                f"{label}: Storage bucket is required for Document Validation Task."
            )
        data["vsEscalationProperties"] = {
            **props,
            "storageBucketFolderPath": props.get("storageBucketFolderPath") or "",
        }
        try:
            resource = AgentIxpVsEscalationResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid document validation resource: {e}") from e
    label = f"Escalation resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    if not resource.channels:
        raise ValueError(f"{label} has no channels.")
    channel = resource.channels[0]
    if not isinstance(channel, AgentEscalationChannel):
        raise ValueError(
            f"VS escalation '{resource.name}' must use an Action Center channel but "
            f"received channel type '{_value(getattr(channel, 'type', None))}'."
        )
    vs = resource.vs_escalation_properties
    if not vs.ixp_tool_id:
        raise ValueError(f"{label}: IXP tool is required for Document Validation Task.")
    if not vs.storage_bucket_name:
        raise ValueError(
            f"{label}: Storage bucket is required for Document Validation Task."
        )
    title: str | None = None
    title_from_input: str | None = None
    title_tokens: list[Any] | None = None
    if isinstance(channel.task_title, TextBuilderTaskTitle):
        title_tokens = list(channel.task_title.tokens)
    elif isinstance(channel.task_title, DynamicTaskTitle):
        title_from_input = channel.task_title.argument_path
    elif isinstance(channel.task_title, str) and channel.task_title:
        title = channel.task_title
    ignored = [
        name
        for name, value in (
            ("recipients", getattr(channel, "recipients", None)),
            ("labels", getattr(channel, "labels", None)),
            ("outcomeMapping", getattr(channel, "outcome_mapping", None)),
        )
        if value
    ]
    if ignored:
        logger.info(
            "%s: %s have no effect on a Document Validation Task.",
            label,
            ", ".join(ignored),
        )
    display = getattr(channel.properties, "app_name", None) or _lowcode_tool_name(
        resource.name
    )
    return UiPathDocumentValidationTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=resource.description or DEFAULT_VALIDATION_DESCRIPTION,
        extraction_tool=vs.ixp_tool_id,
        storage_bucket_name=vs.storage_bucket_name,
        # Agent Builder stores the bucket's Orchestrator folder here (langchain
        # sends it as a directory inside the bucket, where DU does not find
        # the bucket).
        action_folder=vs.storage_bucket_folder_path or None,
        priority=_value(channel.priority) if channel.priority else None,
        title=title,
        title_from_input=title_from_input,
        title_tokens=title_tokens,
        display_name=display,
        mock_name=resource.name,
    )
