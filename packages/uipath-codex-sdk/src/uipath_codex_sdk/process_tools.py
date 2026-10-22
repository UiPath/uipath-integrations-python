"""Process-family tools from a low-code ``agent.json`` resource.

:func:`process_tool_from_resource` turns a ``tool`` resource of type
Process, Agent, Api, ProcessOrchestration, Flow or Function (every one of
them starts an Orchestrator job, as in langchain's ``create_process_tool``)
into a :class:`~uipath_codex_sdk.tools.UiPathProcessTool`, with its
``argumentProperties`` pre-filled.

Typical use::

    import json
    from uipath_codex_sdk import UiPathCodexAgent, process_tool_from_resource

    resources = json.load(open("agent.json"))["resources"]
    invoice = next(r for r in resources if r["name"] == "Create Invoice")
    agent = UiPathCodexAgent(
        model="gpt-5.6-terra",
        prompt="Create the invoice for {customer}.",
        tools=[process_tool_from_resource(invoice)],
    )
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import ValidationError

from ._fixed_args import fixed_args_from_properties
from .integration_tools import _lowcode_tool_name
from .tools import UiPathProcessTool, _description, sanitize_tool_name

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentProcessToolResourceConfig

logger = logging.getLogger(__name__)

__all__ = ["process_tool_from_resource"]

#: agent.json ``type`` (any case) → the canonical ``AgentToolType`` value.
_PROCESS_TYPES = {
    "agent": "Agent",
    "process": "Process",
    "api": "Api",
    "processorchestration": "ProcessOrchestration",
    "flow": "Flow",
    "function": "Function",
}
#: Resource type → ``toolType`` of the ``Tool call`` span (low-code's
#: ``get_tool_type_value``; a Flow falls through to its ``Integration``).
_TOOL_TYPES = {
    "Agent": "Agent",
    "Process": "Process",
    "Api": "Api",
    "ProcessOrchestration": "agenticProcess",
    "Flow": "Integration",
    "Function": "Function",
}


def process_tool_from_resource(
    resource: AgentProcessToolResourceConfig | Mapping[str, Any],
    *,
    tool_name: str | None = None,
    run_as_me: bool = False,
    folder_path: str | None = None,
) -> UiPathProcessTool:
    """A durable tool from a low-code process-family ``tool`` resource.

    Maps the process name (also the ``process.<name>`` binding key), the
    tool name (sanitized like low-code: ``Basic RPA Process`` becomes
    ``Basic_RPA_Process``), description, input/output schemas, example calls
    and ``argumentProperties`` (static, argument, text builder and array
    builder; JSONPath keys such as ``$['a']['b']``). The job runs in the
    run's folder (``UIPATH_FOLDER_PATH``) as in low-code: the resource's
    ``folderPath`` is ignored, and bindings redirect the process. Timeout,
    retry and conversational-confirmation settings are ignored.

    Args:
        resource: An ``AgentProcessToolResourceConfig``, or the resource dict
            as found in a low-code ``agent.json`` (``"$resourceType":
            "tool"``, ``"type"`` in any case).
        tool_name: Model-facing name; defaults to the resource name sanitized.
        run_as_me: Run the child job under the calling user's identity.
        folder_path: Folder of the release, instead of the run's folder.

    Raises:
        ValueError: The resource does not validate (another tool type, a
            ``dynamic`` argument property), is disabled, has no
            ``processName``, or has an object-builder argument property no
            parent path covers.
    """
    from uipath.agent.models.agent import (
        AgentProcessToolResourceConfig,
        AgentToolType,
    )

    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "tool")
        raw_type = data.get("type")
        data["type"] = _PROCESS_TYPES.get(str(raw_type or "").lower(), raw_type)
        try:
            resource = AgentProcessToolResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid process tool resource: {e}") from e
    if not isinstance(resource, AgentProcessToolResourceConfig):
        raise ValueError(f"Not a process tool resource: {type(resource).__name__}")
    label = f"Tool resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    props = resource.properties
    process_name = props.process_name
    if not process_name:
        raise ValueError(f"{label} has no 'processName'.")
    resource_type = str(getattr(resource.type, "value", resource.type))
    kind: Literal["process", "agent"] = (
        "agent" if resource.type == AgentToolType.AGENT else "process"
    )
    if props.folder_path and not folder_path:
        logger.debug(
            "%s: folderPath '%s' is ignored; the job runs in the run's folder.",
            label,
            props.folder_path,
        )
    settings = resource.settings
    if any(
        v is not None
        for v in (settings.timeout, settings.max_attempts, settings.retry_delay)
    ):
        logger.debug("%s: timeout and retry settings are ignored.", label)
    input_schema = copy.deepcopy(resource.input_schema)
    output_schema = (
        copy.deepcopy(resource.output_schema)
        if isinstance(resource.output_schema, dict)
        and resource.output_schema.get("properties")
        else None
    )
    return UiPathProcessTool(
        process_name=process_name,
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=_description(
            resource.description or None, process_name, kind, output_schema
        ),
        folder_path=folder_path,
        input_json_schema=input_schema,
        output_json_schema=output_schema,
        run_as_me=run_as_me,
        kind=kind,
        mock_name=resource.name,
        example_calls=list(props.example_calls or []),
        fixed_args=fixed_args_from_properties(
            label, resource.argument_properties, schema=input_schema, strict=True
        ),
        tool_type=_TOOL_TYPES.get(resource_type),
        display_name=process_name,
        mock_child_span=resource_type != "Flow",
    )
