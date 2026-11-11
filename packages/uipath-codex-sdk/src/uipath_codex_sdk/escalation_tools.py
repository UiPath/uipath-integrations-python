"""Action Center escalations: durable tools that ask a human.

An escalation tool creates an Action Center task with an app (a form) and
**suspends** the run until a person completes it. The run then resumes,
possibly on another robot, and the model gets langchain's escalation result:
``{"outcome": <the button pressed>, "output": <the form's output fields>,
"task_id": ..., "assigned_to": ...}``.

Two ways to declare one:

* :func:`uipath_escalation_tool`: coded;
* :func:`escalation_tool_from_resource`: an ``escalation`` resource copied
  from a low-code ``agent.json`` (its first, ``actionCenter`` channel).

Typical use::

    from pydantic import BaseModel
    from uipath_codex_sdk import UiPathCodexAgent, uipath_escalation_tool

    class Review(BaseModel):
        amount: float
        reason: str

    agent = UiPathCodexAgent(
        model="gpt-5.6-terra",
        prompt="Approve the expense {expense}; ask a manager above 1000.",
        tools=[
            uipath_escalation_tool(
                "ExpenseApproval",                  # the deployed Action Center app
                folder_path="Shared",
                input_schema=Review,
                assignee="manager@example.com",
                outcome_mapping={"Reject": "end"},
            )
        ],
    )

The app is a bindable ``app`` resource (``app.<name>[.<folder>]``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping

from pydantic import BaseModel, ValidationError

from . import tools as _tools
from ._fixed_args import (  # noqa: F401  (FixedArg, _without_fields: re-exported)
    FixedArg,
    _get_path,
    _without_fields,
    fixed_args_from_code,
    fixed_args_from_properties,
    fixed_args_schema,
    missing_args_message,
    missing_from_input,
    with_fixed_args,
)
from .integration_tools import _lowcode_tool_name
from .runtime.errors import UiPathRunTerminatedError
from .tools import (
    _PERMISSIVE_SCHEMA,
    DurableToolContext,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    _schema_from,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentEscalationResourceConfig
    from uipath.platform.action_center.tasks import TaskRecipient

    from .runtime.mocking import MockSpan, MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "UiPathEscalationTool",
    "escalation_tool_from_resource",
    "uipath_escalation_tool",
]

DEFAULT_TITLE = "Escalation Task"
DELETED_OUTCOME = "The escalation task was deleted"
_PENDING_STATUSES = frozenset({0, 1, "0", "1", "unassigned", "pending"})
#: Keys that only a serialized ``Task`` has (a raw human answer has none).
_TASK_KEYS = frozenset({"taskDefinitionPropertiesId", "isDeleted", "is_deleted"})


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------
@dataclass
class UiPathEscalationTool:
    """A durable tool that creates an Action Center task and waits for it.

    Prefer :func:`uipath_escalation_tool` or
    :func:`escalation_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        app_name: Deployed Action Center app (a bindable ``app`` resource).
        folder_path: Folder of the app; None means the job's folder.
        folder_key: Folder key of the app, instead of ``folder_path``.
        app_key: App key; skips the deployed-app lookup.
        app_input_schema: JSON schema of the app's inputs (task data).
        output_json_schema: JSON schema of the app's outputs: the task data
            the model gets back is filtered to these fields.
        input_model: Pydantic model the task data is validated with.
        title: Fixed task title.
        title_from_input: Path of the title in the agent input.
        title_tokens: Text tokens of the title (agent.json text builder).
        recipient: Who the task is assigned to.
        recipient_config: An agent.json recipient, resolved at call time.
        assignee: User name or email the task is assigned to.
        priority: ``Low`` / ``Medium`` / ``High`` / ``Critical``.
        labels: Task labels.
        outcome_mapping: Outcome (button) → ``continue`` or ``end``; an
            ``end`` outcome ends the run.
        fixed_args: Pre-filled app fields by path (``a.b``, ``x[*].y``):
            static ones are hidden from the model, agent-input ones stay
            visible but optional; merged into the task data.
        is_actionable_message_enabled: Actionable notifications.
        actionable_message_metadata: Their metadata.
        display_name: Name of the ``escalationTool`` span; defaults to the
            app name.
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span.
        span_output_keys: Keys of the result the spans record (low-code's).
        mock_name: Name the tool is simulated by in evals; None means the
            model-facing name, lowercased (langchain's).
        example_calls: Example calls for the eval LLM mocker (the channel's).
    """

    name: str
    description: str
    app_name: str
    folder_path: str | None = None
    folder_key: str | None = None
    app_key: str | None = None
    app_input_schema: dict[str, Any] = field(
        default_factory=lambda: dict(_PERMISSIVE_SCHEMA)
    )
    output_json_schema: dict[str, Any] | None = None
    input_model: type[BaseModel] | None = None
    title: str | None = None
    title_from_input: str | None = None
    title_tokens: list[Any] | None = None
    recipient: TaskRecipient | None = None
    recipient_config: Any = None
    assignee: str | None = None
    priority: str | None = None
    labels: list[str] | None = None
    outcome_mapping: dict[str, str] | None = None
    fixed_args: dict[str, FixedArg] = field(default_factory=dict)
    is_actionable_message_enabled: bool | None = None
    actionable_message_metadata: dict[str, Any] | None = None
    display_name: str | None = None
    kind: str = "escalation"
    tool_type: str = "Escalation"
    span_output_keys: tuple[str, ...] = ("outcome", "output")
    mock_name: str | None = None
    example_calls: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.app_name:
            raise ValueError(f"Escalation tool '{self.name}': app_name is required.")
        if self.folder_path and self.folder_key:
            raise ValueError(
                f"Escalation tool '{self.name}': set folder_path or folder_key, "
                "not both."
            )
        if sum(x is not None for x in (self.title, self.title_from_input)) > 1:
            raise ValueError(
                f"Escalation tool '{self.name}': set title or title_from_input, "
                "not both."
            )
        for outcome, action in (self.outcome_mapping or {}).items():
            if str(action).lower() not in ("continue", "end"):
                raise ValueError(
                    f"Escalation tool '{self.name}': outcome '{outcome}' maps to "
                    f"{action!r}; use 'continue' or 'end'."
                )
        if self.title_from_input:
            self.title_from_input = self.title_from_input.strip().strip("{}").strip()
        if self.recipient is not None and not self.recipient.display_name:
            # assignedTo on the span; the SDK only fills it for id recipients.
            self.recipient.display_name = self.recipient.value

    # --- UiPathDurableTool ------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        """The app's inputs as the model sees them (pre-filled fields hidden)."""
        return fixed_args_schema(self.app_input_schema, self.fixed_args)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        _validate_against_schema(self.input_schema, args)
        return json.loads(json.dumps(args, default=str))

    def started_output(self, value: Any) -> dict[str, Any]:
        task = getattr(value, "action", None)
        task_id = getattr(task, "id", None)
        return {
            "status": "started",
            "task_id": task_id,
            "task_key": getattr(task, "key", None),
            "task_url": task_url(task_id),
        }

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Create the task; return ``WaitEscalation`` for it.

        The task is created here rather than by the resume trigger
        (``CreateEscalation``), as langchain does, so the span carries its id
        and URL and a rebuilt trigger never creates a second task. It is
        created unassigned and then assigned: an assignment failure is logged
        and the run still waits on the task (no duplicate on a retry).

        Raises:
            UiPathDurableToolError: The task data is invalid, the recipient
                cannot be resolved, or the task cannot be created.
        """
        from opentelemetry import trace
        from uipath.platform.common import WaitEscalation

        from .runtime.telemetry import (
            fail_span,
            hold_span,
            start_escalation_tool_span,
        )

        data = self.build_data(args, ctx.agent_input)
        title = self.resolve_title(ctx.agent_input)
        try:
            recipient = await self.resolve_recipient(ctx.agent_input)
        except ValueError as e:
            raise UiPathDurableToolError(
                f"Could not resolve who to assign the escalation of tool "
                f"'{self.name}' to: {e}"
            ) from e
        display = self.display_name or self.app_name
        span = start_escalation_tool_span(display, arguments=data)
        try:
            with trace.use_span(span, end_on_exit=False):
                task, app_name, folder_path, folder_key = await _create_task(
                    self.app_name,
                    self.folder_path
                    or (
                        None
                        if self.folder_key
                        else os.environ.get("UIPATH_FOLDER_PATH")
                    ),
                    requested_folder_path=self.folder_path,
                    folder_key=self.folder_key,
                    title=title,
                    data=data,
                    app_key=self.app_key,
                    priority=self.priority,
                    labels=self.labels or None,
                    is_actionable_message_enabled=self.is_actionable_message_enabled,
                    actionable_message_metadata=self.actionable_message_metadata,
                )
        except BaseException as e:
            fail_span(span, e)
            span.end()
            raise UiPathDurableToolError(self._create_error_message(e)) from e
        span.set_attribute("taskId", str(task.id))
        url = task_url(task.id)
        if url:
            span.set_attribute("taskUrl", url)
        if recipient is not None or self.assignee:
            assigned = await _assign_task(task, recipient, self.assignee)
            if assigned is None:
                shown = (recipient.display_name if recipient else None) or self.assignee
                if shown:
                    span.set_attribute("assignedTo", shown)
            else:
                span.set_attribute("assignmentError", assigned)
        hold_span(span, output_attribute="result", output_keys=self.span_output_keys)
        logger.info(
            "escalation tool %s created task %s (call %s)",
            self.name,
            task.id,
            ctx.call_id,
        )
        return WaitEscalation(
            action=task,
            app_name=app_name,
            recipient=recipient,
            app_folder_path=folder_path,
            app_folder_key=None if folder_path else folder_key,
        )

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Map the completed task (or a raw resume answer) to the result.

        * completed → ``{"status": "completed", "outcome", "output",
          "task_id", "assigned_to"}``; an outcome mapped to ``end`` ends the
          run (``TERMINATION_ESCALATION_REJECTED``), as in langchain;
        * deleted → ends the run too (langchain's ``end``);
        * still open → :class:`UiPathDurableToolPending`.

        A raw answer (``uipath run --resume '{...}'``) is the task's data, or
        ``{"action": ..., "data": {...}}``.
        """
        task = _task_from(resume_value)
        task_id = task.id
        if task.is_deleted:
            result = {
                "status": "deleted",
                "outcome": DELETED_OUTCOME,
                "output": None,
                "task_id": task_id,
            }
            raise UiPathRunTerminatedError(
                "Agent run ended because the escalation task was deleted",
                f"Escalation task {task_id} of tool '{self.name}' was deleted.",
                json.dumps(result),
            )
        status = task.status
        if (str(status).lower() if isinstance(status, str) else status) in (
            _PENDING_STATUSES
        ):
            raise UiPathDurableToolPending(f"task {task_id} is not completed yet")
        outcome = task.action
        output = filter_task_data(
            task.data or {}, self.app_input_schema, self.output_json_schema
        )
        result = {
            "status": "completed",
            "outcome": outcome,
            "output": output,
            "task_id": task_id,
            "assigned_to": _completed_by(task),
        }
        text = json.dumps(result, ensure_ascii=False, default=str)
        if self.outcome_action(outcome) == "end":
            raise UiPathRunTerminatedError(
                f"Agent run ended based on escalation outcome {outcome} with "
                "directive end",
                f"Escalation output: {json.dumps(output, default=str)}",
                text,
            )
        return text

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's ``EscalationToolOutput``: ``{action, data, is_deleted}``."""
        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.name.lower(),
            description=self.description,
            input_schema=self.input_schema,
            output_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["approve", "reject"]},
                    "data": copy.deepcopy(self.output_json_schema)
                    or {"type": "object"},
                    "is_deleted": {"type": "boolean", "default": False},
                },
                "required": ["action", "data"],
            },
            example_calls=list(self.example_calls),
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The ``escalationTool`` child, without a task.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled (before the child opens, as
                a real start fails before its span).
        """
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_escalation_tool_span

        span = start_escalation_tool_span(
            self.display_name or self.app_name,
            arguments=self._merged(args, getattr(ctx, "agent_input", None)),
        )
        return MockSpan(span, "result", self.span_output_keys)

    def mock_args(self, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """The model's arguments with the pre-filled fields, as the mocker sees them.

        langchain's ``StaticArgsHandler`` merges them into the tool call
        before its mocked function runs, so mockito behaviors match on them.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled.
        """
        return self._merged(args, getattr(ctx, "agent_input", None))

    #: LangChain never resolves an escalation's job attachments (its
    #: ``escalation_wrapper`` has no ``job_attachment_wrapper``): the host
    #: checks only the model's own attachment values the pre-filled fields
    #: leave in place, never a pre-filled one.
    checks_prefilled_attachments: ClassVar[bool] = False

    @property
    def prefilled_schema(self) -> dict[str, Any]:
        """The app's inputs, pre-filled fields included (listed on the span)."""
        return self.app_input_schema

    def prefilled_args(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The model's arguments with the pre-filled fields, unvalidated.

        A model attachment one of these replaces is not checked by the host
        (see ``checks_prefilled_attachments``).
        """
        if not self.fixed_args:
            return args
        return with_fixed_args(args, self.fixed_args, agent_input)

    def span_args(self, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """The ``Tool call`` span's input: the arguments with the pre-filled fields.

        As low-code's, whose tool call carries them once merged.
        """
        return self._with_fixed(args, getattr(ctx, "agent_input", None))

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated answer as a completed task, through :meth:`finish`.

        So the output filter, a deleted task and an outcome mapped to ``end``
        work as for a real task (the outcome is matched as given).

        Raises:
            UiPathDurableToolError: The simulated value is not an object.
            UiPathRunTerminatedError: The task is deleted, or its outcome ends
                the run.
        """
        from uipath.platform.action_center import Task

        if not isinstance(value, dict):
            raise UiPathDurableToolError(
                f"Unexpected simulated result of tool '{self.name}': {str(value)[:200]}"
            )
        data = value.get("data")
        task = Task.model_validate(
            {
                "status": 2,
                "action": value.get("action"),
                "data": data if isinstance(data, dict) else {},
                "isDeleted": bool(value.get("is_deleted")),
            }
        )
        return await self.finish(args, task)

    # --- helpers ----------------------------------------------------------

    def outcome_action(self, outcome: str | None) -> Literal["continue", "end"]:
        action = (self.outcome_mapping or {}).get(outcome or "")
        return "end" if str(action or "").lower() == "end" else "continue"

    def build_data(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The task data: the model's arguments with the pre-filled fields.

        An agent-input field the input has no value for keeps the model's
        value, as in langchain. Validated with ``input_model`` once merged,
        when there is one.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled, or the merged data does not
                validate.
        """
        data = self._merged(args, agent_input)
        if self.input_model is not None:
            try:
                data = self.input_model.model_validate(data).model_dump(mode="json")
            except ValidationError as e:
                raise UiPathDurableToolError(
                    f"Invalid task data for tool '{self.name}': {e}"
                ) from e
        return json.loads(json.dumps(data, default=str))

    def _with_fixed(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        return with_fixed_args(args, self.fixed_args, agent_input)

    def _merged(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> dict[str, Any]:
        """:meth:`_with_fixed`, refusing a required field still missing."""
        data = self._with_fixed(args, agent_input)
        missing = missing_from_input(self.app_input_schema, self.fixed_args, data)
        if missing:
            raise UiPathDurableToolError(missing_args_message(self.name, missing))
        return data

    def resolve_title(self, agent_input: dict[str, Any] | None) -> str:
        if self.title_from_input:
            found, value = _get_path(agent_input, self.title_from_input)
            if found and value not in (None, ""):
                return str(value)
        if self.title_tokens is not None:
            from uipath.agent.utils.text_tokens import build_string_from_tokens

            return build_string_from_tokens(self.title_tokens, agent_input or {})
        return self.title or DEFAULT_TITLE

    async def resolve_recipient(
        self, agent_input: dict[str, Any] | None
    ) -> TaskRecipient | None:
        if self.recipient is not None:
            return self.recipient.model_copy()
        if self.recipient_config is not None:
            return await resolve_recipient_config(self.recipient_config, agent_input)
        return None

    def _create_error_message(self, e: BaseException) -> str:
        info = getattr(e, "error_info", None)
        message = (getattr(info, "message", None) if info is not None else None) or str(
            e
        ).strip()
        status = getattr(e, "status_code", None)
        where = f" (HTTP {status})" if status else ""
        logger.error(
            "escalation tool %s could not create its task%s: %s",
            self.name,
            where,
            message,
        )
        return (
            f"Could not create the Action Center task of tool '{self.name}' "
            f"(app '{self.app_name}'){where}: {message}"
        )


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def _resource_override_app(func: Any) -> Any:
    try:
        from uipath.platform.common import resource_override
    except Exception:  # pragma: no cover - platform missing
        return func
    return resource_override(
        resource_type="app",
        resource_identifier="app_name",
        folder_identifier="app_folder_path",
    )(func)


@_resource_override_app
async def _create_task(
    app_name: str,
    app_folder_path: str | None = None,
    *,
    requested_folder_path: str | None,
    folder_key: str | None,
    title: str,
    data: dict[str, Any],
    app_key: str | None,
    priority: str | None,
    labels: list[str] | None,
    is_actionable_message_enabled: bool | None,
    actionable_message_metadata: dict[str, Any] | None,
) -> tuple[Any, str, str | None, str | None]:
    """``tasks.create_async``, unassigned; ``(task, app, folder path, key)``.

    ``app_name`` and ``app_folder_path`` go through the SDK's ``app`` resource
    override first (this function carries it), so the values the task was
    created with are the ones saved for the resume. Exactly one folder field
    is sent (the SDK refuses both): an explicit key, unless a binding moved
    the app to another folder path.
    """
    client = _tools._uipath_client()
    use_key = (
        folder_key if folder_key and app_folder_path == requested_folder_path else None
    )
    folder_path = None if use_key else app_folder_path
    task = await client.tasks.create_async(
        title,
        data,
        app_name=app_name,
        app_key=app_key,
        app_folder_path=folder_path,
        app_folder_key=use_key,
        priority=priority,
        labels=labels,
        is_actionable_message_enabled=is_actionable_message_enabled,
        actionable_message_metadata=actionable_message_metadata,
    )
    return task, app_name, folder_path, use_key


async def _assign_task(
    task: Any, recipient: TaskRecipient | None, assignee: str | None
) -> str | None:
    """Assign the task (the SDK's AssignTasks call); None, or the error text."""
    try:
        from uipath.platform.action_center._tasks_service import _assign_task_spec

        service = _tools._uipath_client().tasks
        spec = await _assign_task_spec(service, task.id, assignee, recipient)
        await service.request_async(
            spec.method, spec.endpoint, json=spec.json, content=spec.content
        )
        return None
    except Exception as e:
        message = str(e).strip() or type(e).__name__
        logger.warning(
            "Escalation task %s was created but could not be assigned; it stays "
            "unassigned in Action Center: %s",
            task.id,
            message,
        )
        return message


def task_url(task_id: Any) -> str | None:
    """Action Center URL of a task (``<UIPATH_URL>/actions_/tasks/<id>``)."""
    base = os.environ.get("UIPATH_URL")
    if not base or task_id is None:
        return None
    return f"{base.rstrip('/')}/actions_/tasks/{task_id}"


def _task_from(value: Any) -> Any:
    """``value`` as a ``Task``: one, its serialized form, or a raw answer.

    A serialized task is recognised by keys only a task has (or ``key`` and
    ``status`` together); ``{"action": ..., "data": {...}}`` is a typed answer;
    anything else is the answer's form data.
    """
    from uipath.platform.action_center import Task

    if isinstance(value, Task):
        return value
    if isinstance(value, dict):
        keys = set(value)
        if keys & _TASK_KEYS or {"key", "status"} <= keys:
            return Task.model_validate(value)
        if "data" in keys and keys <= {"action", "data", "status"}:
            return Task.model_validate({"status": 2, **value})
        return Task.model_validate({"status": 2, "data": value})
    return Task.model_validate({"status": 2, "data": {"value": value}})


def _completed_by(task: Any) -> str | None:
    user = task.completed_by_user or task.assigned_to_user
    if isinstance(user, dict):
        return user.get("emailAddress") or user.get("userName") or user.get("name")
    return None


def filter_task_data(
    data: dict[str, Any],
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """langchain's ``_parse_task_data``: the output fields of the task data.

    With an output schema, its fields; without one, every field that is not an
    input field.
    """
    if output_schema and output_schema.get("properties") is not None:
        keep = set(output_schema["properties"])
        return {k: v for k, v in data.items() if k in keep}
    inputs = set((input_schema or {}).get("properties") or {})
    return {k: v for k, v in data.items() if k not in inputs}


async def resolve_recipient_config(
    config: Any, agent_input: dict[str, Any] | None
) -> TaskRecipient | None:
    """langchain's ``resolve_recipient_value`` for an agent.json recipient.

    Raises:
        ValueError: An asset or agent-input value is missing.
    """
    from uipath.agent.models.agent import (
        AgentEscalationRecipientType as Kind,
    )
    from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType

    kind = str(getattr(getattr(config, "type", None), "value", config.type))

    def make(value: Any, type_: TaskRecipientType) -> TaskRecipient:
        return TaskRecipient(type=type_, value=str(value), displayName=str(value))

    if kind in (Kind.ASSET_USER_EMAIL.value, Kind.ASSET_GROUP_NAME.value):
        folder = getattr(config, "folder_path", None) or os.environ.get(
            "UIPATH_FOLDER_PATH"
        )
        try:
            asset = await _tools._uipath_client().assets.retrieve_async(
                name=config.asset_name, folder_path=folder
            )
        except Exception as e:
            raise ValueError(
                f"could not read asset '{config.asset_name}' in folder '{folder}': {e}"
            ) from e
        if not asset or not getattr(asset, "value", None):
            raise ValueError(f"asset '{config.asset_name}' has no value")
        return make(
            asset.value,
            TaskRecipientType.EMAIL
            if kind == Kind.ASSET_USER_EMAIL.value
            else TaskRecipientType.GROUP_NAME,
        )
    if kind in (Kind.ARGUMENT_EMAIL.value, Kind.ARGUMENT_GROUP_NAME.value):
        found, value = _get_path(agent_input, config.argument_path)
        if not found or value in (None, ""):
            raise ValueError(
                f"the agent input has no value for '{config.argument_path}'"
            )
        return make(
            value,
            TaskRecipientType.EMAIL
            if kind == Kind.ARGUMENT_EMAIL.value
            else TaskRecipientType.GROUP_NAME,
        )
    if kind == Kind.WORKLOAD.value:
        return make(config.display_name, TaskRecipientType.WORKLOAD)
    if kind == Kind.ROUND_ROBIN.value:
        return make(config.display_name, TaskRecipientType.ROUND_ROBIN)
    if kind in (
        Kind.USER_ID.value,
        Kind.GROUP_ID.value,
        Kind.USER_EMAIL.value,
        Kind.GROUP_NAME.value,
    ):
        type_ = (
            TaskRecipientType.EMAIL
            if kind == Kind.USER_EMAIL.value
            else TaskRecipientType(kind)
        )
        return TaskRecipient(
            type=type_,
            value=config.value,
            displayName=getattr(config, "display_name", None) or config.value,
        )
    logger.warning(
        "Escalation recipient type %r is not supported; the task is not assigned.",
        kind,
    )
    return None


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def _description(
    description: str | None, app_name: str, output_schema: dict[str, Any] | None
) -> str:
    text = description or (
        f"Asks a person to review this in UiPath Action Center (app '{app_name}') "
        "and waits for their decision. Returns the reviewer's outcome and the "
        "completed form fields."
    )
    if output_schema and output_schema.get("properties"):
        compact = json.dumps(
            {"type": "object", "properties": output_schema["properties"]},
            separators=(",", ":"),
        )
        if len(compact) <= 4000:
            text += " The form fields returned match: " + compact
    return text


def uipath_escalation_tool(
    app_name: str,
    *,
    folder_path: str | None = None,
    folder_key: str | None = None,
    app_key: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, Any] | None = None,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    title: str | None = None,
    title_from_input: str | None = None,
    recipient: TaskRecipient | None = None,
    assignee: str | None = None,
    priority: str | None = None,
    labels: list[str] | None = None,
    outcome_mapping: Mapping[str, str] | None = None,
    static_args: Mapping[str, Any] | None = None,
    args_from_input: Mapping[str, str] | None = None,
    is_actionable_message_enabled: bool | None = None,
    actionable_message_metadata: dict[str, Any] | None = None,
    display_name: str | None = None,
    tool_name: str | None = None,
) -> UiPathEscalationTool:
    """A durable tool that asks a person through an Action Center app.

    Args:
        app_name: The deployed Action Center app (a bindable ``app`` resource).
        folder_path: Folder of the app; defaults to the job's folder.
        folder_key: Folder key of the app, instead of ``folder_path``.
        app_key: App key; skips the deployed-app lookup.
        description: Shown to the model: say when to escalate.
        input_schema: The app's inputs (pydantic model or JSON schema); the
            model fills them, except the pre-filled ones.
        output_schema: The app's outputs: the form data the model gets back
            is filtered to these fields (default: every non-input field).
        title: Task title (default "Escalation Task").
        title_from_input: Path of the title in the agent input.
        recipient: A ``TaskRecipient`` (user/group by id, email or name,
            workload, round robin).
        assignee: User name or email, instead of ``recipient``.
        priority: ``Low`` / ``Medium`` / ``High`` / ``Critical``.
        labels: Task labels.
        outcome_mapping: Outcome (the app's button) → ``continue`` or ``end``;
            an ``end`` outcome ends the run.
        static_args: Fixed app inputs (``a.b``, ``x[*].y``), hidden from the
            model.
        args_from_input: App inputs read from the agent input (field → path).
            They stay visible to the model but optional: the agent input's
            value wins, and the model's is used when the input has none.
        is_actionable_message_enabled: Actionable notifications.
        actionable_message_metadata: Their metadata.
        display_name: Name of the ``escalationTool`` span; defaults to
            ``app_name``.
        tool_name: Model-facing name; defaults to ``escalate_<app_name>``.

    Raises:
        ValueError: An invalid combination or outcome mapping.
    """
    if recipient is not None and assignee:
        raise ValueError("set recipient or assignee, not both")
    model = (
        input_schema
        if isinstance(input_schema, type) and issubclass(input_schema, BaseModel)
        else None
    )
    output_json = _schema_from(output_schema) if output_schema is not None else None
    fixed = fixed_args_from_code(static_args, args_from_input)
    return UiPathEscalationTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else f"escalate_{_lowcode_tool_name(app_name)}",
        description=_description(description, app_name, output_json),
        app_name=app_name,
        folder_path=folder_path,
        folder_key=folder_key,
        app_key=app_key,
        app_input_schema=_schema_from(input_schema),
        output_json_schema=output_json,
        input_model=model,
        title=title,
        title_from_input=title_from_input,
        recipient=recipient,
        assignee=assignee,
        priority=priority,
        labels=list(labels) if labels else None,
        outcome_mapping=dict(outcome_mapping) if outcome_mapping else None,
        fixed_args=fixed,
        is_actionable_message_enabled=is_actionable_message_enabled,
        actionable_message_metadata=actionable_message_metadata,
        display_name=display_name or app_name,
    )


def escalation_tool_from_resource(
    resource: AgentEscalationResourceConfig | Mapping[str, Any],
    *,
    tool_name: str | None = None,
) -> UiPathEscalationTool:
    """A durable tool from a low-code ``escalation`` resource.

    Uses the resource's first channel, which must be an ``actionCenter`` app
    channel (QuickForm channels are not supported yet). It maps the app, its
    folder, input/output schemas, task title (text, text builder or agent
    input), first recipient, priority, labels, outcome mapping and the
    static / argument / text-builder / array-builder ``argumentProperties``
    (keys may be JSONPath, ``$['a']``). Escalation memory is not supported:
    a memory-enabled resource gets a warning.

    Raises:
        ValueError: The resource is disabled, not an Action Center app
            escalation, has no app name, or does not validate.
    """
    from uipath.agent.models.agent import (
        AgentEscalationChannel,
        AgentEscalationResourceConfig,
        AgentIxpVsEscalationResourceConfig,
        DynamicTaskTitle,
        TextBuilderTaskTitle,
    )

    # escalationType 1 (Document Validation Task) is another tool, as in
    # langchain's tool factory.
    if isinstance(resource, AgentIxpVsEscalationResourceConfig) or (
        isinstance(resource, Mapping)
        and (resource.get("escalationType") or resource.get("escalation_type")) == 1
    ):
        from .ixp_tools import vs_escalation_tool_from_resource

        return vs_escalation_tool_from_resource(resource, tool_name=tool_name)  # type: ignore[return-value]
    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "escalation")
        try:
            resource = AgentEscalationResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid escalation resource: {e}") from e
    label = f"Escalation resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    if not resource.channels:
        raise ValueError(f"{label} has no channels.")
    channel = resource.channels[0]
    if not isinstance(channel, AgentEscalationChannel):
        raise ValueError(
            f"{label}: channel type '{getattr(channel, 'type', None)}' is not "
            "supported yet (only Action Center app channels are)."
        )
    props = channel.properties
    if not props.app_name:
        raise ValueError(f"{label} has no 'appName'.")
    if resource.is_agent_memory_enabled:
        logger.warning(
            "%s has escalation memory enabled; it is not supported and is ignored.",
            label,
        )
    title: str | None = None
    title_from_input: str | None = None
    title_tokens: list[Any] | None = None
    if isinstance(channel.task_title, TextBuilderTaskTitle):
        title_tokens = list(channel.task_title.tokens)
    elif isinstance(channel.task_title, DynamicTaskTitle):
        title_from_input = channel.task_title.argument_path
    elif isinstance(channel.task_title, str):
        title = channel.task_title
    fixed = fixed_args_from_properties(label, channel.argument_properties)
    output_schema = channel.output_schema or None
    metadata = props.actionable_message_meta_data
    return UiPathEscalationTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else f"escalate_{_lowcode_tool_name(resource.name)}",
        description=_description(
            resource.description or channel.description, props.app_name, output_schema
        ),
        app_name=props.app_name,
        folder_path=props.folder_name,
        app_input_schema=dict(channel.input_schema or _PERMISSIVE_SCHEMA),
        output_json_schema=output_schema,
        title=title,
        title_from_input=title_from_input,
        title_tokens=title_tokens,
        recipient_config=channel.recipients[0] if channel.recipients else None,
        priority=channel.priority,
        labels=list(channel.labels) or None,
        outcome_mapping=dict(channel.outcome_mapping or {}) or None,
        fixed_args=fixed,
        is_actionable_message_enabled=props.is_actionable_message_enabled,
        actionable_message_metadata=metadata if isinstance(metadata, dict) else None,
        display_name=props.app_name or channel.name,
        example_calls=list(props.example_calls or []),
    )
