"""Offline stand-ins for the UiPath tenant used by the durable-tool tests.

* :class:`FakeOrchestrator` – processes/jobs services with scripted job states
  (``processes.invoke_async``, ``jobs.retrieve_async``,
  ``jobs.extract_output_async``), shared by our tools and by the platform's
  ``UiPathResumeTriggerHandler``.
* :class:`FakeConnections` – the Integration Service ``connections`` service
  (``invoke_activity_async``) with recorded calls, canned responses, an
  injectable failure and a gate that holds calls in flight.
* :class:`MockUpstream` – replaces the gateway shim's ``UpstreamRegistry`` so
  model traffic goes to the contract ``MockResponsesServer`` instead of the
  UiPath LLM Gateway.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from uipath.platform.orchestrator import Job

from ..contract.mock_responses import (
    advertised_tools,
    find_tool,
    function_call_item,
    input_items,
    message_item,
)

TRIGGER = "PLEASE_CALL_DURABLE"
FOLDER_KEY = "11111111-2222-3333-4444-555555555555"


@dataclass
class Invocation:
    name: str
    input_arguments: dict[str, Any] | None
    folder_path: str | None
    run_as_me: bool | None
    job_key: str
    parent_span_id: str | None = None
    attachments: list[Any] | None = None


@dataclass
class FakeOrchestrator:
    jobs: dict[str, Job] = field(default_factory=dict)
    invocations: list[Invocation] = field(default_factory=list)
    retrieve_calls: int = 0
    fail_start: Exception | None = None
    connections: FakeConnections = field(default_factory=lambda: FakeConnections())
    context_grounding: FakeContextGrounding = field(
        default_factory=lambda: FakeContextGrounding()
    )
    tasks: FakeTasks = field(default_factory=lambda: FakeTasks())
    attachments: FakeAttachments = field(default_factory=lambda: FakeAttachments())
    api_client: FakeApiClient = field(default_factory=lambda: FakeApiClient())
    documents: FakeDocuments = field(default_factory=lambda: FakeDocuments())

    # --- control -----------------------------------------------------------
    def finish(
        self,
        key: str,
        state: str = "Successful",
        output: Any = None,
        info: str | None = None,
    ) -> Job:
        job = self.jobs[key]
        data = job.model_dump(by_alias=True)
        data["State"] = state
        data["Info"] = info
        data["OutputArguments"] = json.dumps(output) if output is not None else None
        data["EndTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.jobs[key] = Job.model_validate(data)
        return self.jobs[key]

    def finish_all(self, state: str = "Successful", output_fn=None) -> None:
        for inv in self.invocations:
            out = output_fn(inv) if output_fn else None
            self.finish(inv.job_key, state, out)

    # --- the client surface ------------------------------------------------
    def client(self) -> Any:
        return _FakeUiPath(self)


class _FakeProcesses:
    def __init__(self, world: FakeOrchestrator) -> None:
        self._w = world

    async def invoke_async(
        self,
        name: str,
        input_arguments: dict[str, Any] | None = None,
        *,
        folder_key: str | None = None,
        folder_path: str | None = None,
        run_as_me: bool | None = None,
        **kwargs: Any,
    ) -> Job:
        if self._w.fail_start is not None:
            raise self._w.fail_start
        n = len(self._w.invocations) + 1
        key = f"job-{n}-{uuid.uuid4().hex[:6]}"
        job = Job.model_validate(
            {
                "Id": n,
                "Key": key,
                "State": "Pending",
                "FolderKey": FOLDER_KEY,
                "InputArguments": json.dumps(input_arguments or {}),
            }
        )
        self._w.jobs[key] = job
        self._w.invocations.append(
            Invocation(
                name,
                input_arguments,
                folder_path,
                run_as_me,
                key,
                kwargs.get("parent_span_id"),
                kwargs.get("attachments"),
            )
        )
        return job


class _FakeJobs:
    def __init__(self, world: FakeOrchestrator) -> None:
        self._w = world

    async def retrieve_async(self, job_key: str, **kwargs: Any) -> Job:
        self._w.retrieve_calls += 1
        return self._w.jobs[job_key]

    async def extract_output_async(self, job: Job) -> str | None:
        return job.output_arguments

    async def link_attachment_async(self, **kwargs: Any) -> None:
        self._w.attachments.links.append(kwargs)


class _FakeFolders:
    async def retrieve_key_async(self, *, folder_path: str) -> str | None:
        return INDEX_FOLDER_KEY

    def retrieve_key(self, *, folder_path: str) -> str | None:
        raise AssertionError(f"blocking folder lookup of {folder_path!r}")


@dataclass
class FakeAttachments:
    """Uploaded attachments (``upload_async``) and job links."""

    uploads: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    fail_upload: int = 0

    async def download_async(
        self, *, key: Any, destination_path: str, **kwargs: Any
    ) -> str:
        Path(destination_path).write_text(f"document {key}")
        self.downloads.append(str(key))
        return destination_path

    async def upload_async(self, **kwargs: Any) -> uuid.UUID:
        if self.fail_upload:
            self.fail_upload -= 1
            raise RuntimeError("upload failed")
        content = Path(kwargs["source_path"]).read_text()
        key = uuid.uuid4()
        self.uploads.append({**kwargs, "content": content, "key": key})
        return key


@dataclass
@dataclass
class FakeDocuments:
    """IXP extractions and Validation Station tasks with scripted states."""

    PROJECT_ID = "ixp-project-1"
    extractions: dict[str, dict[str, Any]] = field(default_factory=dict)
    validations: dict[str, dict[str, Any]] = field(default_factory=dict)
    starts: list[dict[str, Any]] = field(default_factory=list)
    validation_starts: list[dict[str, Any]] = field(default_factory=list)
    fail_start: Exception | None = None

    @staticmethod
    def projection(total: str = "4200") -> list[dict[str, Any]]:
        return [
            {
                "fieldGroupName": "Invoice",
                "fieldValues": [
                    {
                        "id": "total",
                        "name": "Total",
                        "value": total,
                        "unformattedValue": total,
                        "confidence": 0.98,
                        "ocrConfidence": 0.99,
                        "type": "Number",
                    }
                ],
            }
        ]

    async def start_ixp_extraction_async(self, **kwargs: Any) -> Any:
        from uipath.platform.documents import StartExtractionResponse

        if self.fail_start is not None:
            raise self.fail_start
        op = f"ext-{len(self.starts) + 1}-{uuid.uuid4().hex[:6]}"
        content = Path(kwargs["file_path"]).read_text()
        self.starts.append({**kwargs, "content": content, "operation_id": op})
        self.extractions[op] = {"status": "Running", "tag": kwargs["tag"]}
        return StartExtractionResponse.model_validate(
            {
                "operationId": op,
                "documentId": f"doc-{op}",
                "projectId": self.PROJECT_ID,
                "tag": kwargs["tag"],
            }
        )

    def finish_extraction(self, op: str, status: str = "Succeeded") -> None:
        self.extractions[op]["status"] = status

    def extraction_response(self, op: str) -> dict[str, Any]:
        return {
            "extractionResult": {
                "DocumentId": f"doc-{op}",
                "ResultsVersion": 1,
                "ResultsDocument": {"Fields": [{"FieldId": "total", "Value": "4200"}]},
            },
            "projectId": self.PROJECT_ID,
            "projectType": "IXP",
            "tag": self.extractions[op]["tag"],
            "documentTypeId": "00000000-0000-0000-0000-000000000000",
            "dataProjection": self.projection(),
        }

    async def retrieve_ixp_extraction_result_async(
        self, project_id: str, tag: str, operation_id: str
    ) -> Any:
        from uipath.platform.documents import ExtractionResponseIXP
        from uipath.platform.errors import (
            OperationFailedException,
            OperationNotCompleteException,
        )

        status = self.extractions[operation_id]["status"]
        if status in ("NotStarted", "Running"):
            raise OperationNotCompleteException(
                operation_id=operation_id, status=status
            )
        if status != "Succeeded":
            raise OperationFailedException(
                operation_id=operation_id, status=status, error="extraction failed"
            )
        return ExtractionResponseIXP.model_validate(
            self.extraction_response(operation_id)
        )

    async def create_validate_extraction_action_async(self, **kwargs: Any) -> Any:
        from uipath.platform.documents import ValidateExtractionAction

        op = f"val-{len(self.validation_starts) + 1}-{uuid.uuid4().hex[:6]}"
        self.validation_starts.append({**kwargs, "operation_id": op})
        self.validations[op] = {"status": "Unassigned", "data": {}}
        extraction = kwargs["extraction_response"]
        return ValidateExtractionAction.model_validate(
            {
                "actionData": {
                    "status": "Unassigned",
                    "taskUrl": f"https://t/tasks/{op}",
                },
                "actionStatus": "Unassigned",
                "projectId": extraction.project_id,
                "projectType": "IXP",
                "tag": extraction.tag,
                "operationId": op,
                "extractorId": None,
                "documentTypeId": extraction.document_type_id,
            }
        )

    def complete_validation(
        self, op: str, total: str = "4250", rejection: str | None = None
    ) -> None:
        data: dict[str, Any] = {}
        if rejection is not None:
            data["documentRejectionDetails"] = {"reason": rejection}
        self.validations[op] = {"status": "Completed", "data": data, "total": total}

    async def retrieve_ixp_extraction_validation_result_async(
        self, project_id: str, tag: str, operation_id: str
    ) -> Any:
        from uipath.platform.documents import ValidateExtractionAction

        task = self.validations[operation_id]
        action_data: dict[str, Any] = {"data": task["data"]}
        if task["status"] != "Deleted":
            action_data["status"] = task["status"]
        return ValidateExtractionAction.model_validate(
            {
                "actionData": action_data,
                "actionStatus": task["status"],
                "projectId": project_id,
                "projectType": "IXP",
                "tag": tag,
                "operationId": operation_id,
                "extractorId": None,
                "documentTypeId": "00000000-0000-0000-0000-000000000000",
                "dataProjection": self.projection(task.get("total", "4200")),
            }
        )


@dataclass
class FakeApiClient:
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def request_async(self, method: str, url: str = "", **kwargs: Any) -> Any:
        self.requests.append({"method": method, "url": url, **kwargs})
        return httpx.Response(201, json={}, request=httpx.Request(method, "https://t"))


class _FakeUiPath:
    def __init__(self, world: FakeOrchestrator) -> None:
        self.processes = _FakeProcesses(world)
        self.jobs = _FakeJobs(world)
        self.connections = world.connections
        self.context_grounding = world.context_grounding
        self.folders = _FakeFolders()
        self.tasks = world.tasks
        self.attachments = world.attachments
        self.documents = world.documents
        self.api_client = world.api_client


INDEX_FOLDER_KEY = "99999999-8888-7777-6666-555555555555"


@dataclass
class DeepRagStart:
    name: str
    index_name: str | None
    prompt: str
    glob_pattern: str | None
    citation_mode: Any
    folder_key: str | None
    folder_path: str | None
    id: str
    attachments: list[str] | None = None


@dataclass
class FakeTasks:
    """Action Center tasks with scripted states (create / assign / retrieve)."""

    created: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieve_calls: list[dict[str, Any]] = field(default_factory=list)
    fail_create: Exception | None = None
    fail_assign: Exception | None = None

    def complete(
        self, key: str, action: str = "Approve", data: dict[str, Any] | None = None
    ) -> None:
        task = self.tasks[key]
        task["status"] = 2
        task["action"] = action
        task["data"] = {**(task.get("data") or {}), **(data or {})}
        task["completedByUser"] = {"emailAddress": "reviewer@example.com"}

    def delete(self, key: str) -> None:
        self.tasks[key]["isDeleted"] = True

    async def create_async(self, title: str, data: Any = None, **kwargs: Any) -> Any:
        from uipath.platform.action_center import Task

        if self.fail_create is not None:
            raise self.fail_create
        n = len(self.created) + 1
        key = f"task-key-{n}-{uuid.uuid4().hex[:6]}"
        self.created.append({"title": title, "data": data, **kwargs, "key": key})
        self.tasks[key] = {
            "id": 1000 + n,
            "key": key,
            "title": title,
            "status": 0,
            "data": dict(data or {}),
            "isDeleted": False,
        }
        return Task.model_validate(self.tasks[key])

    async def request_async(self, method: str, url: Any, **kwargs: Any) -> Any:
        if self.fail_assign is not None:
            raise self.fail_assign
        self.assignments.append({"method": method, "url": str(url), **kwargs})
        return httpx.Response(200, json={}, request=httpx.Request(method, "https://t"))

    async def retrieve_async(self, action_key: str, **kwargs: Any) -> Any:
        from uipath.platform.action_center import Task

        self.retrieve_calls.append({"key": action_key, **kwargs})
        return Task.model_validate(self.tasks[action_key])


@dataclass
class FakeContextGrounding:
    """DeepRAG tasks with scripted states (start / retrieve by id)."""

    starts: list[DeepRagStart] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    batch_starts: list[dict[str, Any]] = field(default_factory=list)
    batch_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    downloads: list[str] = field(default_factory=list)
    status_reads: list[str] = field(default_factory=list)
    result_requests: list[str] = field(default_factory=list)
    retrieve_calls: int = 0
    fail_start: Exception | None = None

    def finish(
        self,
        task_id: str,
        status: str = "Successful",
        text: str = "Acme Corp owes $4,200 [1].",
        failure_reason: str | None = None,
    ) -> None:
        task = self.tasks[task_id]
        task["lastDeepRagStatus"] = status
        if status == "Successful":
            task["content"] = {
                "text": text,
                "citations": [
                    {
                        "ordinal": 1,
                        "pageNumber": 1,
                        "source": "invoice.jpg",
                        "reference": "https://ref/1",
                    }
                ],
            }
        task["failureReason"] = failure_reason

    async def start_deep_rag_async(self, **kwargs: Any) -> Any:
        from uipath.platform.context_grounding import DeepRagCreationResponse

        if self.fail_start is not None:
            raise self.fail_start
        task_id = f"dr-{len(self.starts) + 1}-{uuid.uuid4().hex[:6]}"
        self.starts.append(
            DeepRagStart(
                name=kwargs["name"],
                index_name=kwargs.get("index_name"),
                prompt=kwargs["prompt"],
                glob_pattern=kwargs.get("glob_pattern"),
                citation_mode=kwargs["citation_mode"],
                folder_key=kwargs.get("folder_key"),
                folder_path=kwargs.get("folder_path"),
                id=task_id,
                attachments=kwargs.get("attachments"),
            )
        )
        self.tasks[task_id] = {
            "id": task_id,
            "name": kwargs["name"],
            "createdDate": "2026-10-13T00:00:00Z",
            "lastDeepRagStatus": "Queued",
            "content": None,
        }
        return DeepRagCreationResponse.model_validate(self.tasks[task_id])

    async def start_deep_rag_from_attachments_async(self, **kwargs: Any) -> Any:
        return await self.start_deep_rag_async(**kwargs)

    # --- Batch Transform ------------------------------------------------

    async def start_batch_transform_from_attachment_async(self, **kwargs: Any) -> Any:
        from uipath.platform.context_grounding import BatchTransformCreationResponse

        if self.fail_start is not None:
            raise self.fail_start
        task_id = f"bt-{len(self.batch_starts) + 1}-{uuid.uuid4().hex[:6]}"
        self.batch_starts.append({**kwargs, "id": task_id})
        self.batch_tasks[task_id] = {"status": "Queued", "csv": None}
        return BatchTransformCreationResponse.model_validate(
            {"id": task_id, "lastBatchRagStatus": "Queued"}
        )

    def finish_batch(
        self, task_id: str, status: str = "Successful", csv: str = "a,b\n1,2\n"
    ) -> None:
        self.batch_tasks[task_id] = {"status": status, "csv": csv}

    async def retrieve_batch_transform_async(self, id: str, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.status_reads.append(id)
        return SimpleNamespace(
            id=id, last_batch_rag_status=self.batch_tasks[id]["status"]
        )

    def _batch_transform_get_read_uri_spec(self, id: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(method="GET", endpoint=f"read-uri/{id}", headers={})

    def _batch_transform_download_blob_spec(self, id: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(method="GET", endpoint=f"download-blob/{id}", headers={})

    async def request_async(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        """The read-URI and authenticated download endpoints."""
        kind, task_id = endpoint.split("/", 1)
        self.result_requests.append(kind)
        request = httpx.Request(method, f"https://t/{endpoint}")
        if kind == "read-uri":
            # A storage host the test cannot reach (as a serverless robot).
            return httpx.Response(
                200,
                json={
                    "uri": "https://unreachable.blob.invalid/r.csv",
                    "isEncrypted": False,
                },
                request=request,
            )
        return httpx.Response(
            200, content=self.batch_tasks[task_id]["csv"].encode(), request=request
        )

    async def download_batch_transform_result_async(
        self, id: str, destination_path: str, **kwargs: Any
    ) -> None:
        from pathlib import Path

        from uipath.platform.errors import (
            BatchTransformFailedException,
            BatchTransformNotCompleteException,
        )

        self.downloads.append(destination_path)
        task = self.batch_tasks[id]
        if task["status"] == "Failed":
            raise BatchTransformFailedException(batch_transform_id=id)
        if task["status"] != "Successful":
            raise BatchTransformNotCompleteException(
                batch_transform_id=id, status=task["status"]
            )
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        Path(destination_path).write_text(task["csv"])

    async def retrieve_deep_rag_async(self, id: str, **kwargs: Any) -> Any:
        from uipath.platform.context_grounding import DeepRagResponse

        self.retrieve_calls += 1
        return DeepRagResponse.model_validate(self.tasks[id])


@dataclass
class ActivityInvocation:
    activity_metadata: Any
    connection_id: str
    activity_input: dict[str, Any]


class FakeConnections:
    """``connections.invoke_activity_async`` of the Integration Service."""

    def __init__(self) -> None:
        self.invocations: list[ActivityInvocation] = []
        #: Response by ``object_path``; ``{"ok": True}`` when not listed.
        self.responses: dict[str, Any] = {}
        self.fail: BaseException | None = None
        #: When set, calls wait on it (held in flight) before answering.
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def invoke_activity_async(
        self,
        activity_metadata: Any,
        connection_id: str,
        activity_input: dict[str, Any],
    ) -> Any:
        self.invocations.append(
            ActivityInvocation(activity_metadata, connection_id, activity_input)
        )
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        return self.responses.get(activity_metadata.object_path, {"ok": True})


def enriched(
    status: int, code: str | None = None, message: str = "bad request"
) -> Exception:
    """A real ``EnrichedException`` for an elements_ call, with a JSON body."""
    from uipath.platform.errors import EnrichedException

    body: dict[str, Any] = {"message": message}
    if code is not None:
        body["providerErrorCode"] = code
    request = httpx.Request(
        "POST", "https://cloud.test/org/tenant/elements_/v3/element/instances/c/x"
    )
    response = httpx.Response(
        status,
        request=request,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    error = httpx.HTTPStatusError("boom", request=request, response=response)
    return EnrichedException(error)


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------
class _RewriteToResponses(httpx.AsyncBaseTransport):
    """The shim posts to ``/``; the mock serves ``/v1/responses``."""

    def __init__(self, base: str) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self._base = httpx.URL(base)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.url = self._base
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def mock_upstream_class(responses_url: str) -> type:
    class MockUpstream:
        def __init__(self, settings: Any) -> None:
            self._client = httpx.AsyncClient(
                base_url=responses_url,
                transport=_RewriteToResponses(responses_url),
                timeout=60,
            )

        async def get(self, model: str) -> httpx.AsyncClient:
            return self._client

        async def aclose(self) -> None:
            await self._client.aclose()

    return MockUpstream


class AddScript:
    """Mock model: on the trigger prompt call ``a_plus_b`` ``n`` times in one
    response; afterwards answer with every tool output it was given."""

    def __init__(self, n_calls: int = 1, tool: str = "a_plus_b") -> None:
        self.call_ids = [f"call_{uuid.uuid4().hex[:24]}" for _ in range(n_calls)]
        self.tool = tool
        self.advertised: tuple[str | None, str | None] | None = None

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        user_text = json.dumps(
            [
                it
                for it in items
                if it.get("type") == "message" and it.get("role") == "user"
            ]
        )
        if TRIGGER in user_text and not (seen & set(self.call_ids)):
            name, ns = find_tool(advertised_tools(body), self.tool)
            self.advertised = (name, ns)
            if name:
                return [
                    function_call_item(cid, name, {"a": 17 + i, "b": 25}, ns)
                    for i, cid in enumerate(self.call_ids)
                ]
        outputs = {
            it["call_id"]: it.get("output")
            for it in items
            if it.get("type") == "function_call_output"
        }
        return [message_item("FINAL " + json.dumps(outputs, sort_keys=True), idx)]


class SiblingScript(AddScript):
    """One durable call plus a shell sibling (``exec_command``) in the same
    response, the sibling either after or before the durable call."""

    SIBLING_ID = "call_sibling00000000000000"

    def __init__(self, sibling_after: bool) -> None:
        super().__init__(n_calls=1)
        self.sibling_after = sibling_after

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_ids[0] in seen:
            return [message_item("FINAL", idx)]
        name, ns = find_tool(advertised_tools(body), self.tool)
        if not name:
            return None
        durable = function_call_item(self.call_ids[0], name, {"a": 1, "b": 2}, ns)
        sibling = function_call_item(
            self.SIBLING_ID, "exec_command", {"cmd": "echo sibling-ran"}
        )
        return [durable, sibling] if self.sibling_after else [sibling, durable]


class ChainScript(AddScript):
    """Call ``a_plus_b`` once, then again after the first result arrives."""

    def __init__(self) -> None:
        super().__init__(n_calls=2)

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        outputs = {
            it["call_id"]: it.get("output")
            for it in items
            if it.get("type") == "function_call_output"
        }
        name, ns = find_tool(advertised_tools(body), self.tool)
        for i, cid in enumerate(self.call_ids):
            if cid not in outputs:
                seen = {
                    it.get("call_id")
                    for it in items
                    if it.get("type") == "function_call"
                }
                if cid in seen or not name:
                    return None
                return [function_call_item(cid, name, {"a": 17 + i, "b": 25}, ns)]
        return [message_item("FINAL " + json.dumps(outputs, sort_keys=True), idx)]


IS_TOOL = "genai_web_search"
IS_CALL_ID = "call_genaiwebsearch0000000000"


def _outputs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        it["call_id"]: it.get("output")
        for it in items
        if it.get("type") == "function_call_output"
    }


class IntegrationScript:
    """Mock model: call the IS tool once, then answer with every tool output."""

    def __init__(self, tool: str = IS_TOOL, args: dict[str, Any] | None = None) -> None:
        self.tool = tool
        self.args = args if args is not None else {"query": "uipath codex"}
        self.call_id = IS_CALL_ID
        self.advertised: tuple[str | None, str | None] | None = None
        self.tool_specs: list[dict[str, Any]] = []

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_id not in seen:
            tools = advertised_tools(body)
            name, ns = find_tool(tools, self.tool)
            self.advertised = (name, ns)
            self.tool_specs = tools
            if name:
                return [function_call_item(self.call_id, name, self.args, ns)]
        return [
            message_item("FINAL " + json.dumps(_outputs(items), sort_keys=True), idx)
        ]


class MixedScript(AddScript):
    """A durable ``a_plus_b`` call and the IS call in one response."""

    def __init__(self, is_after: bool, is_args: dict[str, Any] | None = None) -> None:
        super().__init__(n_calls=1)
        self.is_after = is_after
        self.is_args = is_args if is_args is not None else {"query": "uipath codex"}

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_ids[0] in seen:
            return [
                message_item(
                    "FINAL " + json.dumps(_outputs(items), sort_keys=True), idx
                )
            ]
        tools = advertised_tools(body)
        name, ns = find_tool(tools, self.tool)
        is_name, is_ns = find_tool(tools, IS_TOOL)
        if not name or not is_name:
            return None
        durable = function_call_item(self.call_ids[0], name, {"a": 1, "b": 2}, ns)
        integration = function_call_item(IS_CALL_ID, is_name, self.is_args, is_ns)
        return [durable, integration] if self.is_after else [integration, durable]


class DeepRagScript(AddScript):
    """Mock model: call the DeepRAG tool once with a query, then answer with
    every tool output it was given."""

    QUERY = "How much does Acme Corp owe, and by when? Cite the invoices."

    def __init__(self, tool: str = "Invoice_Research") -> None:
        super().__init__(n_calls=1, tool=tool)

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_ids[0] not in seen:
            name, ns = find_tool(advertised_tools(body), self.tool)
            self.advertised = (name, ns)
            if name:
                return [
                    function_call_item(
                        self.call_ids[0], name, {"query": self.QUERY}, ns
                    )
                ]
        return [
            message_item("FINAL " + json.dumps(_outputs(items), sort_keys=True), idx)
        ]


class DeepRagAttachmentScript(AddScript):
    """Mock model: call the DeepRAG attachment tool once with ``attachment``
    (``{"ID": attachment_id}``) and a query, then answer with every output."""

    QUERY = "What does the contract say about termination? Cite the clauses."

    def __init__(self, attachment_id: str, tool: str = "DeepRAG") -> None:
        super().__init__(n_calls=1, tool=tool)
        self.attachment_id = attachment_id
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        self.bodies.append(body)
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_ids[0] not in seen:
            name, ns = find_tool(advertised_tools(body), self.tool)
            self.advertised = (name, ns)
            if name:
                args = {"query": self.QUERY, "attachment": {"ID": self.attachment_id}}
                return [function_call_item(self.call_ids[0], name, args, ns)]
        return [
            message_item("FINAL " + json.dumps(_outputs(items), sort_keys=True), idx)
        ]


class EscalationScript(AddScript):
    """Mock model: call the escalation tool once with an expense, then answer
    with every tool output it was given."""

    ARGS = {"amount": 1250.0, "reason": "Team offsite"}

    def __init__(self, tool: str = "escalate_ExpenseApproval") -> None:
        super().__init__(n_calls=1, tool=tool)

    def __call__(self, idx: int, body: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = input_items(body)
        seen = {it.get("call_id") for it in items if it.get("type") == "function_call"}
        if self.call_ids[0] not in seen:
            name, ns = find_tool(advertised_tools(body), self.tool)
            self.advertised = (name, ns)
            if name:
                return [function_call_item(self.call_ids[0], name, self.ARGS, ns)]
        return [
            message_item("FINAL " + json.dumps(_outputs(items), sort_keys=True), idx)
        ]
