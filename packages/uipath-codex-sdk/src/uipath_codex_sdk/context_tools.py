"""Context Grounding indexes as immediate (semantic search) tools.

A context tool searches one Context Grounding index and returns the best
matching chunks to the model. The runtime serves it on the ``uipath_tools``
MCP server and answers each call as soon as
``UiPath().context_grounding.unified_search_async(...)`` returns, the call
``uipath_langchain``'s context tool makes (``handle_semantic_search``). The run
never suspends on it.

Two ways to declare one:

* :func:`uipath_context_tool`: coded;
* :func:`context_tool_from_resource`: a ``context`` resource copied from a
  low-code ``agent.json`` (or an ``AgentContextResourceConfig``).

The query comes from the model (the default), is fixed (``query=``), or is
read from the agent input (``query_from_input=``, the low-code ``{{arg}}``
variant). The result is langchain's: ``{"documents": [{"metadata": {...},
"page_content": "..."}]}``.

Typical use::

    from uipath_codex_sdk import UiPathCodexAgent, uipath_context_tool

    agent = UiPathCodexAgent(
        model="gpt-5.6-terra",
        prompt="Answer from the company policies: {question}",
        tools=[
            uipath_context_tool(
                "Company Policies",
                index_name="company-policies",
                number_of_results=5,
            )
        ],
    )

The index name and folder are passed to the SDK through a function with the
SDK's ``index`` resource override, so an ``index.<name>[.<folder>]`` overwrite
(``bindings.json``) rebinds them.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from pydantic import ValidationError

from . import tools as _tools
from ._fixed_args import (
    fixed_args_schema,
    missing_args_message,
    missing_from_input,
    with_fixed_args,
)
from .integration_tools import (
    DEFAULT_TIMEOUT_SEC,
    _check_timeout,
    _get_path,
    _is_httpx_timeout,
    _lowcode_tool_name,
)
from .tools import (
    DurableToolContext,
    ToolContext,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    UiPathToolError,
    _validate_against_schema,
    sanitize_tool_name,
)

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentContextResourceConfig

    from .runtime.mocking import MockSpan, MockSpec

logger = logging.getLogger(__name__)

__all__ = [
    "UiPathContextTool",
    "UiPathDeepRagAttachmentsTool",
    "UiPathDeepRagTool",
    "context_tool_from_resource",
    "deep_rag_tool_from_resource",
    "uipath_context_tool",
    "uipath_deep_rag_attachments_tool",
    "uipath_deep_rag_tool",
]

#: langchain's description of the ``query`` argument.
QUERY_DESCRIPTION = "The query to search for in the knowledge base"
#: Retrieval modes answered by a semantic search. langchain sends Structured
#: as a semantic search too (``SearchMode`` has no other value).
SEARCH_MODES = ("Semantic", "Structured")
#: Cap of the JSON result, in characters (the span attribute cap): whole
#: documents are dropped from the end, so the JSON stays valid.
RESULT_CAP = 64_000


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------
@dataclass
class UiPathContextTool:
    """An immediate tool that runs a semantic search on one index.

    Prefer :func:`uipath_context_tool` or :func:`context_tool_from_resource`
    over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        index_name: The Context Grounding index searched. It is passed to the
            SDK as is, so a resource overwrite can rebind it.
        folder_path: Folder of the index; None means the job's folder
            (``UIPATH_FOLDER_PATH``), as low-code agents do.
        number_of_results: Maximum number of chunks returned.
        threshold: Minimum similarity score of a returned chunk.
        query: Fixed query; the model then passes no arguments.
        query_from_input: Path of the query in the agent input (``{{x}}``
            variant). The model still sees ``query``, marked optional, and its
            value is used only when the agent input has none.
        folder_path_prefix: Only search documents under this folder of the
            index's data source.
        folder_path_prefix_from_input: Path of the prefix in the agent input.
        file_extension: Only search documents with this extension (``pdf``).
        include_system_indexes: Fall back to tenant system indexes when the
            index is not found; None means only in Studio (debug) runs, as
            low-code agents do.
        retrieval_mode: The configured mode (``Semantic`` or ``Structured``),
            recorded on the span.
        display_name: Name of the ``contextGroundingTool`` span; defaults to
            ``name``.
        timeout_sec: Host-enforced timeout of one call, in seconds. None
            means the default (300).
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span.
        mock_name: Name the search is simulated by in evals; None means the
            display name (the resource name).
    """

    #: The search is mocked inside ``call`` (under its span), not by the host.
    mocks_in_call: ClassVar[bool] = True

    name: str
    description: str
    index_name: str
    folder_path: str | None = None
    number_of_results: int = 10
    threshold: float = 0.0
    query: str | None = None
    query_from_input: str | None = None
    folder_path_prefix: str | None = None
    folder_path_prefix_from_input: str | None = None
    file_extension: str | None = None
    include_system_indexes: bool | None = None
    retrieval_mode: str = "Semantic"
    display_name: str | None = None
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC
    kind: str = "context"
    tool_type: str = "ContextGrounding"
    mock_name: str | None = None

    def __post_init__(self) -> None:
        if not self.index_name:
            raise ValueError(f"Context tool '{self.name}': index_name is required.")
        if self.query is not None and self.query_from_input is not None:
            raise ValueError(
                f"Context tool '{self.name}': set query or query_from_input, not both."
            )
        if self.query is not None and not self.query.strip():
            raise ValueError(f"Context tool '{self.name}': the static query is empty.")
        if self.number_of_results < 1:
            raise ValueError(
                f"Context tool '{self.name}': number_of_results must be at least 1."
            )
        self.query_from_input = _input_path(self.query_from_input)
        self.folder_path_prefix_from_input = _input_path(
            self.folder_path_prefix_from_input
        )
        self.file_extension = _extension(self.file_extension)
        self.timeout_sec = _check_timeout(self.name, self.timeout_sec)

    # --- UiPathTool -------------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        if self.query is not None:
            return {"type": "object", "properties": {}}
        prop: dict[str, Any] = {"type": "string", "description": QUERY_DESCRIPTION}
        if self.query_from_input is not None:
            prop["description"] = (
                f"{QUERY_DESCRIPTION} (Set from the agent input "
                f"'{self.query_from_input}' when it has a value; fill it only if "
                "it does not.)"
            )
            return {"type": "object", "properties": {"query": prop}}
        return {"type": "object", "properties": {"query": prop}, "required": ["query"]}

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "context_retrieval_mode": self.retrieval_mode,
            "tool.display_name": self.display_name or self.name,
        }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        if self.query is not None:
            return {}
        _validate_against_schema(self.input_schema, args)
        query = args.get("query")
        if query is not None and not isinstance(query, str):
            raise ValueError("query must be a string")
        if self.query_from_input is None and not (query or "").strip():
            raise ValueError("query must not be empty")
        return {"query": query} if query is not None else {}

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Search the index; return the documents as JSON text.

        Raises:
            UiPathToolError: The index is missing or still ingesting, the
                service refused the search, or it timed out; the message is
                shown to the model.
        """
        from opentelemetry import trace
        from uipath.platform.errors import (
            ContextGroundingIndexNotFoundError,
            EnrichedException,
            IngestionInProgressException,
        )

        from .runtime.mocking import mark_simulated
        from .runtime.telemetry import (
            end_context_grounding_span,
            start_context_grounding_span,
        )

        query = self.resolve_query(args, ctx.agent_input)
        if not query:
            raise UiPathToolError(
                f"Tool '{self.name}' needs a query: pass one in 'query'."
            )
        prefix = self.resolve_folder_path_prefix(ctx.agent_input)
        span = start_context_grounding_span(
            self.display_name or self.name,
            retrieval_mode=self.retrieval_mode,
            query=query,
            threshold=self.threshold,
            number_of_results=self.number_of_results,
            folder_path_prefix=prefix,
            file_extension=self.file_extension,
            system_index=self._include_system_indexes(),
        )
        try:
            # Mocked under this span (the marker lands on it); the real
            # search runs under a retriever span of its own (``_invoke``).
            # The mocker sees langchain's arguments: the query the tool was
            # called with (the agent input's when set from it), none for a
            # static one; the prefix only reaches the real search.
            mock_args = {} if self.query is not None else {"query": query}
            with trace.use_span(span, end_on_exit=False):
                result = await self._invoke(query, prefix)(**mock_args)
        except BaseException as e:
            end_context_grounding_span(span, error=e)
            if isinstance(e, ContextGroundingIndexNotFoundError):
                logger.error(
                    "Context tool %s: index '%s' was not found.",
                    self.name,
                    self.index_name,
                )
                raise UiPathToolError(
                    f"Context grounding index '{self.index_name}' was not found."
                ) from e
            if isinstance(e, IngestionInProgressException):
                logger.warning(
                    "Context tool %s: index '%s' is still ingesting.",
                    self.name,
                    self.index_name,
                )
                raise UiPathToolError(
                    f"Context grounding index '{self.index_name}' is still being "
                    "ingested. Try again later."
                ) from e
            if isinstance(e, EnrichedException):
                raise UiPathToolError(self._enriched_message(e)) from e
            if _is_httpx_timeout(e):
                logger.warning("Context grounding search of %s timed out.", self.name)
                raise UiPathToolError(
                    f"Context grounding search for tool '{self.name}' timed out. "
                    "You may retry."
                ) from e
            raise
        text = _capped_json(result)
        mark_simulated(span)
        end_context_grounding_span(span, result=text)
        return text

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's search: ``{"documents": [...]}``."""
        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.display_name or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=DOCUMENTS_SCHEMA,
            example_calls=[],  # langchain: examples cannot be provided for context
        )

    # --- helpers ----------------------------------------------------------

    def resolve_query(
        self, args: dict[str, Any], agent_input: dict[str, Any] | None
    ) -> str | None:
        """The static query, else the agent input's, else the model's."""
        if self.query is not None:
            return self.query
        if self.query_from_input is not None:
            found, value = _get_path(agent_input, self.query_from_input)
            if found and value not in (None, ""):
                return str(value)
            logger.debug(
                "Context tool %s: {{%s}} is not in the agent input; using the "
                "model's query.",
                self.name,
                self.query_from_input,
            )
        query = args.get("query")
        return query.strip() or None if isinstance(query, str) else None

    def resolve_folder_path_prefix(
        self, agent_input: dict[str, Any] | None
    ) -> str | None:
        if self.folder_path_prefix:
            return self.folder_path_prefix
        if self.folder_path_prefix_from_input is not None:
            found, value = _get_path(agent_input, self.folder_path_prefix_from_input)
            if found and value not in (None, ""):
                return str(value)
        return None

    def _include_system_indexes(self) -> bool:
        if self.include_system_indexes is not None:
            return self.include_system_indexes
        try:
            from uipath.platform.common import UiPathConfig

            return bool(UiPathConfig.is_studio_project)
        except Exception:  # pragma: no cover - config unavailable
            return False

    def _invoke(self, query: str, folder_path_prefix: str | None) -> Any:
        """The search for ``query``, ``mockable`` with :attr:`mock_spec`.

        The real search runs under a ``ContextGroundingRetriever`` span; a
        simulated one has none (langchain's retriever is inside its mocked
        function too). The resolved query and prefix are bound here, so the
        mocked function's arguments are only the model-facing ones.
        """
        from opentelemetry import trace

        from .runtime.mocking import wrap
        from .runtime.telemetry import end_retriever_span, start_retriever_span

        async def search(**_kwargs: Any) -> Any:
            retriever = start_retriever_span(query)
            try:
                with trace.use_span(retriever, end_on_exit=False):
                    result = await _search_index(
                        self.index_name,
                        query,
                        folder_path=self.folder_path
                        or os.environ.get("UIPATH_FOLDER_PATH"),
                        number_of_results=self.number_of_results,
                        threshold=self.threshold,
                        folder_path_prefix=folder_path_prefix,
                        file_extension=self.file_extension,
                        include_system_indexes=self._include_system_indexes(),
                    )
            except BaseException as e:
                end_retriever_span(retriever, error=e)
                raise
            end_retriever_span(retriever, result=_capped_json(result))
            return result

        return wrap(self.mock_spec, search)

    def _enriched_message(self, e: Any) -> str:
        """langchain's ``_CONTEXT_GROUNDING_ERRORS`` mapping, as model-facing text."""
        info = getattr(e, "error_info", None)
        message = (getattr(info, "message", None) if info is not None else None) or str(
            e
        ).strip()
        status = getattr(e, "status_code", None)
        if status == 400:
            logger.warning(
                "Context grounding rejected the search of tool %s (HTTP 400): %s",
                self.name,
                message,
            )
            return (
                f"Context grounding returned an error for index "
                f"'{self.index_name}': {message}"
            )
        logger.error(
            "Context grounding search of tool %s failed (HTTP %s): %s",
            self.name,
            status,
            message,
        )
        return (
            f"Failed to search context index '{self.index_name}' "
            f"(HTTP {status}): {message}"
        )


#: Output schema of a context tool (eval mocks).
DOCUMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "description": "List of retrieved documents.",
            "items": {
                "type": "object",
                "properties": {
                    "metadata": {"type": "object"},
                    "page_content": {"type": "string"},
                },
                "required": ["page_content"],
            },
        }
    },
    "required": ["documents"],
}


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------
_folder_keys: dict[str, str | None] = {}


def _resource_override_index(func: Any) -> Any:
    try:
        from uipath.platform.common import resource_override
    except Exception:  # pragma: no cover - platform missing
        return func
    return resource_override(resource_type="index")(func)


@_resource_override_index
async def _search_index(
    name: str,
    query: str,
    folder_path: str | None = None,
    *,
    number_of_results: int,
    threshold: float,
    folder_path_prefix: str | None,
    file_extension: str | None,
    include_system_indexes: bool,
) -> dict[str, Any]:
    """``unified_search_async`` on ``name``, mapped to langchain's documents.

    ``name`` and ``folder_path`` go through the SDK's ``index`` resource
    override first (this function carries it). The folder key is then
    resolved asynchronously: given only a path, the SDK resolves it with a
    blocking call, which would stall the event loop the gateway proxy and the
    tool host share.
    """
    from uipath.platform.context_grounding import SearchMode, UnifiedSearchScope

    client = _tools._uipath_client()
    folder_key = await _folder_key(client, folder_path)
    scope = (
        UnifiedSearchScope(folder=folder_path_prefix, extension=file_extension)
        if folder_path_prefix or file_extension
        else None
    )
    result = await client.context_grounding.unified_search_async(
        name=name,
        query=query,
        search_mode=SearchMode.SEMANTIC,
        number_of_results=number_of_results,
        threshold=threshold,
        scope=scope,
        folder_key=folder_key,
        # Only without a key: the SDK then searches across folders, as it
        # does for a path it cannot resolve.
        folder_path=None if folder_key else folder_path,
        include_system_indexes=include_system_indexes,
    )
    return {"documents": documents_from_result(result)}


async def _folder_key(client: Any, folder_path: str | None) -> str | None:
    """Key of ``folder_path``: the job's own key for the job's folder, else looked up."""
    if not folder_path:
        return None
    if folder_path == os.environ.get("UIPATH_FOLDER_PATH") and os.environ.get(
        "UIPATH_FOLDER_KEY"
    ):
        return os.environ["UIPATH_FOLDER_KEY"]
    if folder_path not in _folder_keys:
        _folder_keys[folder_path] = await client.folders.retrieve_key_async(
            folder_path=folder_path
        )
    return _folder_keys[folder_path]


def documents_from_result(result: Any) -> list[dict[str, Any]]:
    """langchain's ``ContextGroundingRetriever`` documents of a unified search."""
    semantic = getattr(result, "semantic_results", None)
    values = getattr(semantic, "values", None) or []
    metadata = getattr(semantic, "metadata", None)
    search_id = getattr(metadata, "operation_id", None) if metadata else None
    return [
        {
            "metadata": {
                "source": item.source,
                "search_id": search_id,
                "reference": item.reference,
                "page_number": item.page_number,
                "score": item.score,
            },
            "page_content": item.content,
        }
        for item in values
    ]


def _capped_json(result: Any) -> str:
    """``result`` as JSON of at most :data:`RESULT_CAP` characters.

    Whole documents are dropped from the end (``"truncated": true`` is then
    added); a single document longer than the cap has its text cut.
    """
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= RESULT_CAP or not isinstance(result, dict):
        return text
    docs = list(result.get("documents") or [])
    while docs:
        out = {**result, "documents": docs, "truncated": True}
        text = json.dumps(out, ensure_ascii=False, default=str)
        if len(text) <= RESULT_CAP:
            return text
        if len(docs) == 1:
            doc = dict(docs[0])
            content = str(doc.get("page_content") or "")
            over = len(text) - RESULT_CAP
            doc["page_content"] = content[: max(0, len(content) - over - 1)] + "…"
            return json.dumps(
                {**out, "documents": [doc]}, ensure_ascii=False, default=str
            )
        docs.pop()
    return json.dumps({**result, "documents": [], "truncated": True})


# --------------------------------------------------------------------------
# DeepRAG (durable)
# --------------------------------------------------------------------------
#: langchain's description of the DeepRAG ``query`` argument.
DEEP_RAG_QUERY_DESCRIPTION = (
    "Describe the task: what to research across documents, what to synthesize, "
    "and how to cite sources"
)
#: The DeepRAG service's prompt limit.
DEEP_RAG_PROMPT_MAX = 250_000
_DEEP_RAG_PENDING = frozenset({"queued", "inprogress"})


@dataclass
class UiPathDeepRagTool:
    """A durable tool that runs a DeepRAG task on one index.

    The call starts the task and suspends the run; when the task ends the run
    resumes and the model gets langchain's DeepRAG result:
    ``{"text": ..., "citations": [...], "deepRagId": ...}``. Prefer
    :func:`uipath_deep_rag_tool` or :func:`context_tool_from_resource` over
    constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        index_name: The index the task researches (a bindable ``index``
            resource).
        citation_mode: ``Inline`` (citations in the text) or ``Skip``.
        folder_path: Folder of the index; None means the job's folder.
        query: Fixed task prompt; the model then passes no arguments.
        query_from_input: Path of the prompt in the agent input; the model's
            ``query`` is used only when the input has none.
        folder_path_prefix: Only research documents under this folder of the
            index's data source.
        folder_path_prefix_from_input: Path of that prefix in the agent input.
        file_extension: Only research documents with this extension.
        display_name: Name of the ``contextGroundingTool`` span.
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span.
        mock_name: Name the tool is simulated by in evals; None means the
            display name (the resource name).
    """

    name: str
    description: str
    index_name: str
    citation_mode: Any = "Inline"
    folder_path: str | None = None
    query: str | None = None
    query_from_input: str | None = None
    folder_path_prefix: str | None = None
    folder_path_prefix_from_input: str | None = None
    file_extension: str | None = None
    display_name: str | None = None
    kind: str = "context"
    tool_type: str = "ContextGrounding"
    retrieval_mode: str = "DeepRAG"
    mock_name: str | None = None

    def __post_init__(self) -> None:
        from uipath.platform.context_grounding import CitationMode

        if not self.index_name:
            raise ValueError(f"DeepRAG tool '{self.name}': index_name is required.")
        if self.query is not None and self.query_from_input is not None:
            raise ValueError(
                f"DeepRAG tool '{self.name}': set query or query_from_input, not both."
            )
        if self.query is not None and not self.query.strip():
            raise ValueError(f"DeepRAG tool '{self.name}': the static query is empty.")
        try:
            # The SDK reads citation_mode.value: a plain string would crash.
            self.citation_mode = CitationMode(_value(self.citation_mode))
        except ValueError as e:
            raise ValueError(
                f"DeepRAG tool '{self.name}': citation_mode must be 'Inline' or "
                f"'Skip', got {self.citation_mode!r}."
            ) from e
        self.query_from_input = _input_path(self.query_from_input)
        self.folder_path_prefix_from_input = _input_path(
            self.folder_path_prefix_from_input
        )
        self.file_extension = _extension(self.file_extension)

    # --- UiPathDurableTool ------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        if self.query is not None:
            return {"type": "object", "properties": {}}
        description = DEEP_RAG_QUERY_DESCRIPTION
        if self.query_from_input is not None:
            description += (
                f" (Set from the agent input '{self.query_from_input}' when it "
                "has a value.)"
            )
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": description}},
            "required": ["query"],
        }

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "context_retrieval_mode": self.retrieval_mode,
            "tool.display_name": self.display_name or self.name,
        }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        if self.query is not None:
            return {}
        _validate_against_schema(self.input_schema, args)
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > DEEP_RAG_PROMPT_MAX:
            raise ValueError(f"query is longer than {DEEP_RAG_PROMPT_MAX} characters")
        return {"query": query}

    def started_output(self, value: Any) -> dict[str, Any]:
        return _deep_rag_started_output(value)

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Start the DeepRAG task; return ``WaitDeepRagRaw`` for it.

        The task is started here rather than by the resume trigger (as
        langchain's ``CreateDeepRag`` does), so a start error comes back to
        the model as a tool error, and a failed task as a result instead of a
        faulted trigger. A ``contextGroundingTool`` span is left open until
        the task's result is used.

        Raises:
            UiPathDurableToolError: The index is missing or still ingesting,
                or the service refused the task.
        """
        from opentelemetry import trace
        from uipath.platform.common import WaitDeepRagRaw
        from uipath.platform.errors import (
            ContextGroundingIndexNotFoundError,
            EnrichedException,
            IngestionInProgressException,
        )

        from .runtime.telemetry import (
            end_context_grounding_span,
            hold_span,
            start_context_grounding_span,
        )

        query = _resolve(self.query, self.query_from_input, args, ctx.agent_input)
        if not query:
            raise UiPathDurableToolError(
                f"Tool '{self.name}' needs a query: pass one in 'query'."
            )
        if len(query) > DEEP_RAG_PROMPT_MAX:
            raise UiPathDurableToolError(
                f"The query of tool '{self.name}' is longer than "
                f"{DEEP_RAG_PROMPT_MAX} characters."
            )
        prefix = self.folder_path_prefix or _input_value(
            ctx.agent_input, self.folder_path_prefix_from_input
        )
        glob = build_glob_pattern(prefix, self.file_extension)
        span = start_context_grounding_span(
            self.display_name or self.name,
            retrieval_mode=self.retrieval_mode,
            query=query,
            citation_mode=_value(self.citation_mode),
            folder_path_prefix=prefix,
            file_extension=self.file_extension,
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                started, folder_key, folder_path = await _start_deep_rag(
                    self.index_name,
                    folder_path=self.folder_path
                    or os.environ.get("UIPATH_FOLDER_PATH"),
                    prompt=query,
                    glob_pattern=glob,
                    citation_mode=self.citation_mode,
                )
        except BaseException as e:
            end_context_grounding_span(span, error=e)
            if isinstance(e, ContextGroundingIndexNotFoundError):
                raise UiPathDurableToolError(
                    f"Context grounding index '{self.index_name}' was not found."
                ) from e
            if isinstance(e, IngestionInProgressException):
                raise UiPathDurableToolError(
                    f"Context grounding index '{self.index_name}' is still being "
                    "ingested. Try again later."
                ) from e
            if isinstance(e, EnrichedException):
                status = getattr(e, "status_code", None)
                info = getattr(e, "error_info", None)
                message = getattr(info, "message", None) or str(e).strip()
                raise UiPathDurableToolError(
                    f"Could not start DeepRAG on index '{self.index_name}' "
                    f"(HTTP {status}): {message}"
                ) from e
            if _is_httpx_timeout(e):
                raise UiPathDurableToolError(
                    f"Starting DeepRAG for tool '{self.name}' timed out. You may retry."
                ) from e
            raise
        span.set_attribute("deep_rag_id", started.id)
        hold_span(span)
        logger.info(
            "DeepRAG tool %s started task %s (call %s)",
            self.name,
            started.id,
            ctx.call_id,
        )
        return WaitDeepRagRaw(
            deep_rag=started,
            index_folder_key=folder_key,
            index_folder_path=None if folder_key else folder_path,
        )

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Map the finished task (or a raw resume payload) to the result text.

        * successful → ``{"text", "citations", "deepRagId"}`` (langchain's);
        * failed → ``{"status": "failed", "error": <reason>, "deepRagId"}``;
        * queued or in progress → :class:`UiPathDurableToolPending`;
        * anything else (a raw ``--resume`` payload) → JSON-dumped as is.
        """
        return _deep_rag_result(resume_value)

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated as langchain's DeepRAG: ``{"text", "citations", "deepRagId"}``."""
        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.display_name or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=deep_rag_output_schema(),
            example_calls=[],
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The ``contextGroundingTool`` child, without a task."""
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_context_grounding_span

        agent_input = getattr(ctx, "agent_input", None)
        span = start_context_grounding_span(
            self.display_name or self.name,
            retrieval_mode=self.retrieval_mode,
            query=_resolve(self.query, self.query_from_input, args, agent_input) or "",
            citation_mode=_value(self.citation_mode),
            folder_path_prefix=self.folder_path_prefix
            or _input_value(agent_input, self.folder_path_prefix_from_input),
            file_extension=self.file_extension,
        )
        return MockSpan(span)

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated result as is (a successful task's shape)."""
        return _dumps(value)


def deep_rag_output_schema() -> dict[str, Any]:
    """langchain's ``DeepRagOutputModel``: ``DeepRagContent`` plus ``deepRagId``."""
    from pydantic import Field, create_model
    from uipath.platform.context_grounding import DeepRagContent

    model = create_model(
        "DeepRagOutputModel",
        __base__=DeepRagContent,
        deep_rag_id=(str, Field(alias="deepRagId")),
    )
    return model.model_json_schema()


def _deep_rag_started_output(value: Any) -> dict[str, Any]:
    """The ``Tool call`` output while suspended on a DeepRAG task."""
    deep_rag = getattr(value, "deep_rag", None)
    return {
        "status": "started",
        "deep_rag_id": getattr(deep_rag, "id", None),
        "deep_rag_status": getattr(deep_rag, "last_deep_rag_status", None),
    }


def _deep_rag_result(resume_value: Any) -> str:
    """The model-facing result of a finished DeepRAG task (see ``finish``)."""
    deep_rag = _deep_rag_from(resume_value)
    if deep_rag is None:
        return _dumps(resume_value)
    status = str(_value(deep_rag.last_deep_rag_status) or "")
    if status.lower() in _DEEP_RAG_PENDING:
        raise UiPathDurableToolPending(f"DeepRAG {deep_rag.id} is {status}")
    if status.lower() == "successful" and deep_rag.content is not None:
        return json.dumps(
            {**deep_rag.content.model_dump(), "deepRagId": deep_rag.id},
            ensure_ascii=False,
            default=str,
        )
    result: dict[str, Any] = {
        "status": status.lower() or "unknown",
        "deepRagId": deep_rag.id,
    }
    if status.lower() != "successful":
        result["error"] = deep_rag.failure_reason or "DeepRAG did not succeed."
    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------
# DeepRAG on a job attachment (durable)
# --------------------------------------------------------------------------
#: langchain's description of the internal DeepRAG tool's ``query`` argument.
DEEP_RAG_ATTACHMENTS_QUERY_DESCRIPTION = (
    "Describe the task: what to research across documents, what to synthesize "
    "and how to cite sources."
)
#: The Agent Builder description of the DeepRAG internal tool.
DEFAULT_DEEP_RAG_ATTACHMENTS_DESCRIPTION = (
    "Get comprehensive synthesis across up to 1,000 pages with citations."
)
#: ``properties.toolType`` of the low-code resource.
DEEP_RAG_TOOL_TYPE = "deep-rag"


def deep_rag_attachments_input_schema() -> dict[str, Any]:
    """The Agent Builder DeepRAG arguments: one ``attachment`` (``query`` is added)."""
    from .attachment_tools import analyze_files_input_schema

    definitions = analyze_files_input_schema()["definitions"]
    return {
        "type": "object",
        "properties": {
            "attachment": {
                "$ref": "#/definitions/job-attachment",
                "title": "Source",
                "description": "File or document to analyze with DeepRAG",
            }
        },
        "required": ["attachment"],
        "definitions": copy.deepcopy(definitions),
    }


@dataclass
class UiPathDeepRagAttachmentsTool:
    """A durable tool that runs a DeepRAG task on a job attachment.

    One call to ``start_deep_rag_from_attachments_async``: the service builds
    a temporary index of the file and starts the task, and the run suspends
    on it (the same ``WaitDeepRagRaw`` as :class:`UiPathDeepRagTool`). The
    model gets ``{"text", "citations", "deepRagId"}``. DeepRAG reads PDF and
    TXT files. Prefer :func:`uipath_deep_rag_attachments_tool` or
    :func:`deep_rag_tool_from_resource` over constructing this directly.

    Attributes:
        name: Model-facing tool name.
        description: Description shown to the model.
        input_json_schema: Arguments without ``query``: an ``attachment``
            (a JobAttachment, resolved by the tool host).
        citation_mode: ``Inline`` (citations in the text) or ``Skip``.
        query: Fixed task prompt; the model then passes no ``query``.
        query_from_input: Path of the prompt in the agent input; the model's
            ``query`` is used only when the input has none.
        query_description: Description of the ``query`` argument.
        fixed_args: Arguments pre-filled from the agent input or static
            values (agent.json ``argumentProperties``): static ones are
            hidden from the model, agent-input ones stay visible but
            optional (the model's value is used when the input has none).
        folder_path: Folder the task runs in; None means the job's folder.
        display_name: Name of the internal tool span.
        kind: ``tool.kind`` of the ``Tool call`` span.
        tool_type: ``toolType`` of the ``Tool call`` span.
        mock_name: Name the tool is simulated by in evals; None means the
            display name.
        mock_output_schema: Output schema of a simulated call (the
            resource's ``outputSchema``); None means langchain's DeepRAG
            result.
    """

    name: str
    description: str
    input_json_schema: dict[str, Any] = field(
        default_factory=deep_rag_attachments_input_schema
    )
    citation_mode: Any = "Inline"
    query: str | None = None
    query_from_input: str | None = None
    query_description: str | None = None
    fixed_args: dict[str, Any] = field(default_factory=dict)
    folder_path: str | None = None
    display_name: str | None = None
    kind: str = "internal"
    tool_type: str = "Internal"
    mock_name: str | None = None
    mock_output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        from uipath.platform.context_grounding import CitationMode

        label = f"DeepRAG tool '{self.name}'"
        if self.query is not None and self.query_from_input is not None:
            raise ValueError(f"{label}: set query or query_from_input, not both.")
        if self.query is not None and not self.query.strip():
            raise ValueError(f"{label}: the static query is empty.")
        try:
            # The SDK reads citation_mode.value: a plain string would crash.
            self.citation_mode = CitationMode(_value(self.citation_mode))
        except ValueError as e:
            raise ValueError(
                f"{label}: citation_mode must be 'Inline' or 'Skip', got "
                f"{self.citation_mode!r}."
            ) from e
        self.query_from_input = _input_path(self.query_from_input)
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
            description = (
                self.query_description or DEEP_RAG_ATTACHMENTS_QUERY_DESCRIPTION
            )
            if self.query_from_input is not None:
                description += (
                    f" (Set from the agent input '{self.query_from_input}' when it "
                    "has a value.)"
                )
            props["query"] = {"type": "string", "description": description}
            required.append("query")
        else:
            props.pop("query", None)
        schema["required"] = required
        return fixed_args_schema(schema, self.fixed_args)

    @property
    def span_attributes(self) -> dict[str, Any]:
        return {"tool.display_name": self.display_name or self.name}

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

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        _validate_against_schema(self.input_schema, args)
        query = args.get("query")
        if isinstance(query, str) and len(query) > DEEP_RAG_PROMPT_MAX:
            raise ValueError(f"query is longer than {DEEP_RAG_PROMPT_MAX} characters")
        return json.loads(json.dumps(args, default=str))

    def started_output(self, value: Any) -> dict[str, Any]:
        return _deep_rag_started_output(value)

    async def start(self, args: dict[str, Any], ctx: DurableToolContext) -> Any:
        """Start DeepRAG on the attachment; return ``WaitDeepRagRaw`` for it.

        Raises:
            UiPathDurableToolError: No query or attachment, an unknown
                attachment, or the service refused the task (e.g. a file type
                DeepRAG cannot read).
        """
        from opentelemetry import trace
        from uipath.platform.common import WaitDeepRagRaw
        from uipath.platform.errors import EnrichedException

        from .attachments import (
            dump_attachment,
            not_found_message,
            span_attachments,
            to_attachment,
        )
        from .runtime.telemetry import fail_span, hold_span, start_internal_tool_span

        args = self.merged_args(args, ctx.agent_input)
        query = _resolve(self.query, self.query_from_input, args, ctx.agent_input)
        if not query:
            raise UiPathDurableToolError("Query is required for DeepRAG tool")
        if len(query) > DEEP_RAG_PROMPT_MAX:
            raise UiPathDurableToolError(
                f"The query of tool '{self.name}' is longer than "
                f"{DEEP_RAG_PROMPT_MAX} characters."
            )
        if "attachment" not in args:
            raise UiPathDurableToolError("Argument 'attachment' is not available")
        raw = args["attachment"]
        if not raw:
            raise UiPathDurableToolError("Attachment is required for DeepRAG tool")
        attachment = to_attachment(raw)
        if attachment is None:
            raw_id = raw.get("ID", raw.get("id")) if isinstance(raw, dict) else None
            if not raw_id:
                raise UiPathDurableToolError("Attachment ID is required")
            # A pre-filled argument the host did not resolve.
            attachment = ctx.attachments.get(str(raw_id)) if ctx.attachments else None
            if attachment is None:
                raise UiPathDurableToolError(not_found_message(str(raw_id)))
        args["attachment"] = dump_attachment(attachment)
        args["query"] = query

        span = start_internal_tool_span(
            self.display_name or self.name,
            arguments=args,
            tool_type=self.tool_type,
            citation_mode=_value(self.citation_mode),
            attachments=span_attachments([attachment], "in"),
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                started, folder_key = await _start_deep_rag_from_attachments(
                    prompt=query,
                    attachment_id=str(attachment.id),
                    citation_mode=self.citation_mode,
                    folder_path=self.folder_path,
                )
        except BaseException as e:
            fail_span(span, e)
            span.end()
            if isinstance(e, UiPathDurableToolError):
                raise
            if isinstance(e, EnrichedException):
                status = getattr(e, "status_code", None)
                info = getattr(e, "error_info", None)
                message = getattr(info, "message", None) or str(e).strip()
                raise UiPathDurableToolError(
                    f"Could not start DeepRAG on attachment "
                    f"'{attachment.full_name}' (HTTP {status}): {message}"
                ) from e
            if _is_httpx_timeout(e):
                raise UiPathDurableToolError(
                    f"Starting DeepRAG for tool '{self.name}' timed out. You may retry."
                ) from e
            raise
        # Saved with the held span: set before hold_span.
        span.set_attribute("deep_rag_id", started.id)
        span.set_attribute("output.mime_type", "application/json")
        hold_span(span)
        logger.info(
            "DeepRAG tool %s started task %s on attachment %s (call %s)",
            self.name,
            started.id,
            attachment.id,
            ctx.call_id,
        )
        return WaitDeepRagRaw(deep_rag=started, index_folder_key=folder_key)

    async def finish(self, args: dict[str, Any], resume_value: Any) -> str:
        """Map the finished task to the result text, as :class:`UiPathDeepRagTool`."""
        return _deep_rag_result(resume_value)

    # --- eval mocking -----------------------------------------------------

    @property
    def mock_spec(self) -> MockSpec:
        """Simulated with the resource's output schema (langchain's), else DeepRAG's."""
        from .runtime.mocking import MockSpec

        return MockSpec(
            name=self.mock_name or self.display_name or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=copy.deepcopy(self.mock_output_schema or {})
            or deep_rag_output_schema(),
            example_calls=[],
        )

    def start_mock_span(self, args: dict[str, Any], ctx: Any) -> MockSpan:
        """The internal tool child, without a task.

        Raises:
            UiPathDurableToolError: A required agent-input field neither the
                agent input nor the model filled (before the child opens, as
                a real start fails before its span).
        """
        from .runtime.mocking import MockSpan
        from .runtime.telemetry import start_internal_tool_span

        span = start_internal_tool_span(
            self.display_name or self.name,
            arguments=self.merged_args(args, getattr(ctx, "agent_input", None)),
            tool_type=self.tool_type,
            citation_mode=_value(self.citation_mode),
        )
        return MockSpan(span)

    async def mocked_output(self, args: dict[str, Any], value: Any, ctx: Any) -> str:
        """The simulated result as is."""
        return _dumps(value)


async def _start_deep_rag_from_attachments(
    *, prompt: str, attachment_id: str, citation_mode: Any, folder_path: str | None
) -> tuple[Any, str | None]:
    """``start_deep_rag_from_attachments_async``; ``(task, folder key)``.

    The folder is passed explicitly (the call sends none by itself): the
    tool's ``folder_path``, else the job's folder (``UIPATH_FOLDER_KEY``, else
    ``UIPATH_FOLDER_PATH`` looked up); the service requires one. Only a key is
    sent, so the SDK never looks a path up synchronously.
    """
    client = _tools._uipath_client()
    if folder_path:
        folder_key = await _folder_key(client, folder_path)
        if not folder_key:
            raise UiPathDurableToolError(f"Folder '{folder_path}' was not found.")
    else:
        folder_key = os.environ.get("UIPATH_FOLDER_KEY") or await _folder_key(
            client, os.environ.get("UIPATH_FOLDER_PATH")
        )
    if not folder_key:
        # The service refuses the call without a folder (HTTP 400).
        raise UiPathDurableToolError(
            "DeepRAG on an attachment needs a folder: the run has none (set "
            "UIPATH_FOLDER_PATH for a local run, or the tool's folder_path)."
        )
    started = await client.context_grounding.start_deep_rag_from_attachments_async(
        name=f"task-{uuid.uuid4()}",
        prompt=prompt,
        attachments=[attachment_id],
        citation_mode=citation_mode,
        folder_key=folder_key,
        folder_path=None,
    )
    return started, folder_key


def _deep_rag_from(value: Any) -> Any | None:
    """``value`` as a ``DeepRagResponse`` when it is (or serializes) one."""
    from uipath.platform.context_grounding import DeepRagResponse

    if isinstance(value, DeepRagResponse):
        return value
    if isinstance(value, dict) and (
        "lastDeepRagStatus" in value or "last_deep_rag_status" in value
    ):
        try:
            return DeepRagResponse.model_validate(value)
        except ValidationError:
            return None
    return None


def _dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _input_value(agent_input: dict[str, Any] | None, path: str | None) -> str | None:
    if path is None:
        return None
    found, value = _get_path(agent_input, path)
    return str(value) if found and value not in (None, "") else None


def _resolve(
    static: str | None,
    from_input: str | None,
    args: dict[str, Any],
    agent_input: dict[str, Any] | None,
) -> str | None:
    """The static value, else the agent input's, else the model's ``query``."""
    if static is not None:
        return static
    value = _input_value(agent_input, from_input)
    if value is not None:
        return value
    query = args.get("query")
    return query.strip() or None if isinstance(query, str) else None


def _resource_override_deep_rag(func: Any) -> Any:
    try:
        from uipath.platform.common import resource_override
    except Exception:  # pragma: no cover - platform missing
        return func
    return resource_override(resource_type="index", resource_identifier="index_name")(
        func
    )


@_resource_override_deep_rag
async def _start_deep_rag(
    index_name: str,
    folder_path: str | None = None,
    *,
    prompt: str,
    glob_pattern: str,
    citation_mode: Any,
) -> tuple[Any, str | None, str | None]:
    """``start_deep_rag_async`` on ``index_name``; ``(task, folder key, path)``.

    Like :func:`_search_index`: the index binding is applied first, then the
    folder key is resolved asynchronously.
    """
    client = _tools._uipath_client()
    folder_key = await _folder_key(client, folder_path)
    started = await client.context_grounding.start_deep_rag_async(
        name=f"task-{uuid.uuid4()}",
        index_name=index_name,
        prompt=prompt,
        glob_pattern=glob_pattern,
        citation_mode=citation_mode,
        folder_key=folder_key,
        folder_path=None if folder_key else folder_path,
    )
    return started, folder_key, folder_path


def build_glob_pattern(
    folder_path_prefix: str | None, file_extension: str | None
) -> str:
    """langchain's ``build_glob_pattern``: the DeepRAG file filter."""
    prefix = (folder_path_prefix or "").strip("/").rstrip("/*") or "**"
    extension = f"*.{file_extension.lower()}" if file_extension else "*"
    if prefix == "**":
        return "**/*" if extension == "*" else f"**/{extension}"
    return f"{prefix}/{extension}"


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def uipath_context_tool(
    name: str,
    *,
    index_name: str,
    description: str | None = None,
    folder_path: str | None = None,
    number_of_results: int = 10,
    threshold: float = 0.0,
    query: str | None = None,
    query_from_input: str | None = None,
    folder_path_prefix: str | None = None,
    folder_path_prefix_from_input: str | None = None,
    file_extension: str | None = None,
    include_system_indexes: bool | None = None,
    tool_name: str | None = None,
    timeout_sec: float | None = None,
) -> UiPathContextTool:
    """An immediate tool that runs a semantic search on a Context Grounding index.

    Args:
        name: Display name (``contextGroundingTool`` span); the model-facing
            name is it sanitized the way low-code agents do, unless
            ``tool_name`` is given.
        index_name: The index searched (a bindable ``index`` resource).
        description: Shown to the model; defaults to a sentence naming the
            index. Say what the index holds: the model picks tools by it.
        folder_path: Folder of the index; defaults to the job's folder.
        number_of_results: Maximum number of chunks returned (default 10).
        threshold: Minimum similarity score (default 0).
        query: A fixed query: the model calls the tool with no arguments.
        query_from_input: Path of the query in the agent input (``topic``,
            ``a.b`` or ``{{topic}}``); the model's ``query`` is the fallback.
        folder_path_prefix: Only search documents under this data-source
            folder.
        folder_path_prefix_from_input: Path of that prefix in the agent input.
        file_extension: Only search documents with this extension; ``All``
            means no filter.
        include_system_indexes: Fall back to tenant system indexes; defaults
            to only in Studio (debug) runs.
        tool_name: Model-facing name override.
        timeout_sec: Host-enforced timeout; None means 300 seconds.

    Raises:
        ValueError: An invalid combination (e.g. both ``query`` and
            ``query_from_input``), an empty index name or query.
    """
    return UiPathContextTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description
        or f"Searches the '{index_name}' knowledge base and returns the most "
        "relevant passages.",
        index_name=index_name,
        folder_path=folder_path,
        number_of_results=number_of_results,
        threshold=threshold,
        query=query,
        query_from_input=query_from_input,
        folder_path_prefix=folder_path_prefix,
        folder_path_prefix_from_input=folder_path_prefix_from_input,
        file_extension=file_extension,
        include_system_indexes=include_system_indexes,
        display_name=name,
        timeout_sec=timeout_sec,
    )


def uipath_deep_rag_tool(
    name: str,
    *,
    index_name: str,
    description: str | None = None,
    citation_mode: str = "Inline",
    folder_path: str | None = None,
    query: str | None = None,
    query_from_input: str | None = None,
    folder_path_prefix: str | None = None,
    folder_path_prefix_from_input: str | None = None,
    file_extension: str | None = None,
    tool_name: str | None = None,
) -> UiPathDeepRagTool:
    """A durable tool that runs a DeepRAG task on a Context Grounding index.

    The call suspends the run until the task ends (minutes, typically); the
    model then gets ``{"text", "citations", "deepRagId"}``.

    Args:
        name: Display name; the model-facing name is it sanitized the way
            low-code agents do, unless ``tool_name`` is given.
        index_name: The index researched (a bindable ``index`` resource).
        description: Shown to the model; say what the index holds.
        citation_mode: ``Inline`` (default) or ``Skip``.
        folder_path: Folder of the index; defaults to the job's folder.
        query: A fixed task prompt: the model calls the tool with no arguments.
        query_from_input: Path of the prompt in the agent input.
        folder_path_prefix: Only research documents under this folder.
        folder_path_prefix_from_input: Path of that prefix in the agent input.
        file_extension: Only research documents with this extension.
        tool_name: Model-facing name override.

    Raises:
        ValueError: An invalid combination, an empty index name or query, or
            an unknown citation mode.
    """
    return UiPathDeepRagTool(
        name=sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name),
        description=description
        or f"Researches the '{index_name}' knowledge base across documents and "
        "returns a synthesized answer with citations.",
        index_name=index_name,
        citation_mode=citation_mode,
        folder_path=folder_path,
        query=query,
        query_from_input=query_from_input,
        folder_path_prefix=folder_path_prefix,
        folder_path_prefix_from_input=folder_path_prefix_from_input,
        file_extension=file_extension,
        display_name=name,
    )


def uipath_deep_rag_attachments_tool(
    name: str = "DeepRAG",
    *,
    description: str | None = None,
    citation_mode: str = "Inline",
    query: str | None = None,
    query_from_input: str | None = None,
    query_description: str | None = None,
    attachment_from_input: str | None = None,
    folder_path: str | None = None,
    tool_name: str | None = None,
) -> UiPathDeepRagAttachmentsTool:
    """A durable tool that runs a DeepRAG task on a job attachment.

    The model passes an ``attachment`` (a JobAttachment it was given) and a
    ``query``; the call suspends the run until the task ends (minutes,
    typically), and the model then gets ``{"text", "citations",
    "deepRagId"}``. DeepRAG reads PDF and TXT files.

    Args:
        name: Display name; the model-facing name is it sanitized the way
            low-code agents do, unless ``tool_name`` is given.
        description: Shown to the model; defaults to Agent Builder's.
        citation_mode: ``Inline`` (default) or ``Skip``.
        query: A fixed task prompt: the model passes no ``query``.
        query_from_input: Path of the prompt in the agent input; the model's
            ``query`` is the fallback.
        query_description: Description of the ``query`` argument.
        attachment_from_input: Path of the attachment in the agent input. Its
            value wins; ``attachment`` stays visible to the model but
            optional, and the model's is used when the input has none.
        folder_path: Folder the task runs in; defaults to the job's folder.
        tool_name: Model-facing name override.

    Raises:
        ValueError: An invalid combination, an empty query or an unknown
            citation mode.
    """
    from .escalation_tools import FixedArg

    fixed: dict[str, Any] = {}
    if attachment_from_input:
        fixed["attachment"] = FixedArg(path=_input_path(attachment_from_input))
    tool_label = (
        sanitize_tool_name(tool_name) if tool_name else _lowcode_tool_name(name)
    )
    return UiPathDeepRagAttachmentsTool(
        name=tool_label,
        description=description or DEFAULT_DEEP_RAG_ATTACHMENTS_DESCRIPTION,
        citation_mode=citation_mode,
        query=query,
        query_from_input=query_from_input,
        query_description=query_description,
        fixed_args=fixed,
        folder_path=folder_path,
        display_name=_lowcode_tool_name(name),
        mock_name=name,
    )


def deep_rag_tool_from_resource(
    resource: Any,
    *,
    tool_name: str | None = None,
) -> UiPathDeepRagAttachmentsTool:
    """The DeepRAG tool from a low-code internal ``tool`` resource.

    Args:
        resource: An ``AgentInternalToolResourceConfig``, or the resource dict
            as found in a low-code ``agent.json`` (``"$resourceType": "tool"``,
            ``"type": "Internal"``, ``properties.toolType`` ``"deep-rag"``).
            Its query (static, dynamic or agent-input), citation mode and
            static, argument and text-builder ``argumentProperties`` are used;
            ``fileExtension`` and ``folderPathPrefix`` are ignored, as in
            langchain.
        tool_name: Model-facing name; defaults to the resource name sanitized
            the way low-code agents do.

    Raises:
        ValueError: The resource is disabled, is another internal tool, has a
            static query without a value, or does not validate.
    """
    from .attachment_tools import fixed_args_from_properties, internal_tool_resource

    resource, label = internal_tool_resource(resource, DEEP_RAG_TOOL_TYPE)
    settings = resource.properties.settings
    setting = settings.query
    if (
        setting is not None
        and str(_value(getattr(setting, "variant", "")) or "").lower() == "static"
        and not str(getattr(setting, "value", None) or "").strip()
    ):
        raise ValueError(
            f"{label}: the static query has no value. Set one, or make the query "
            "dynamic so the model writes it."
        )
    if setting is not None and not _value(getattr(setting, "variant", None)):
        query, query_from_input = None, None  # langchain: dynamic
    else:
        query, query_from_input = _query_setting(label, setting)
    schema = copy.deepcopy(resource.input_schema or deep_rag_attachments_input_schema())
    display = _lowcode_tool_name(resource.name)
    return UiPathDeepRagAttachmentsTool(
        name=sanitize_tool_name(tool_name) if tool_name else display,
        description=resource.description or DEFAULT_DEEP_RAG_ATTACHMENTS_DESCRIPTION,
        input_json_schema=schema,
        citation_mode=_value(settings.citation_mode.value),
        query=query,
        query_from_input=query_from_input,
        query_description=getattr(setting, "description", None) if setting else None,
        fixed_args=fixed_args_from_properties(label, resource.argument_properties),
        display_name=display,
        mock_name=resource.name,
        mock_output_schema=_mock_output_schema(resource),
    )


def context_tool_from_resource(
    resource: AgentContextResourceConfig | Mapping[str, Any],
    *,
    tool_name: str | None = None,
    timeout_sec: float | None = None,
) -> UiPathContextTool | UiPathDeepRagTool:
    """A tool from a low-code ``context`` resource.

    Retrieval mode ``Semantic`` or ``Structured`` gives an immediate search
    tool (langchain answers both with a semantic search); ``DeepRAG`` gives a
    durable :class:`UiPathDeepRagTool`, which needs ``settings.citationMode``.

    Args:
        resource: An ``AgentContextResourceConfig``, or the resource dict as
            found in a low-code ``agent.json`` (``resources[]`` entry with
            ``"$resourceType": "context"``).
        tool_name: Model-facing name; defaults to the resource name sanitized
            the way low-code agents do.
        timeout_sec: Host-enforced timeout; None means 300 seconds.

    Raises:
        ValueError: The resource is disabled, is not an index resource, has
            no index name, uses a retrieval mode not supported yet
            (BatchTransform, DataFabric), is DeepRAG without a citation mode,
            or does not validate.
    """
    from uipath.agent.models.agent import (
        AgentContextResourceConfig,
        AgentContextType,
    )

    if isinstance(resource, Mapping):
        data = dict(resource)
        data.setdefault("$resourceType", "context")
        try:
            resource = AgentContextResourceConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Not a valid context resource: {e}") from e
    label = f"Context resource '{resource.name}'"
    if resource.is_enabled is False:
        raise ValueError(f"{label} is disabled; filter it out before building tools.")
    if resource.context_type == AgentContextType.ATTACHMENTS:
        raise ValueError(
            f"{label}: context type 'attachments' is not supported; use an "
            "internal 'deep-rag' tool (deep_rag_tool_from_resource) or "
            "'batch-transform' tool (batch_transform_tool_from_resource) on job "
            "attachments."
        )
    if resource.context_type not in (None, AgentContextType.INDEX):
        raise ValueError(
            f"{label}: context type '{_value(resource.context_type)}' is not "
            "supported yet (only index resources are)."
        )
    if not resource.index_name:
        raise ValueError(f"{label} has no 'indexName'.")
    settings = resource.settings
    if settings is None:
        raise ValueError(f"{label} has no 'settings'.")
    mode = str(_value(settings.retrieval_mode))
    supported = (*SEARCH_MODES, "DeepRAG")
    if mode.lower() not in {m.lower() for m in supported}:
        raise ValueError(
            f"{label}: retrieval mode '{mode}' is not supported yet "
            f"(supported: {', '.join(supported)})."
        )
    query, query_from_input = _query_setting(label, settings.query)
    prefix, prefix_from_input = _prefix_setting(settings.folder_path_prefix)
    extension = settings.file_extension.value if settings.file_extension else None
    if mode.lower() == "deeprag":
        citation = settings.citation_mode.value if settings.citation_mode else None
        if not citation:
            raise ValueError(f"{label}: DeepRAG needs 'settings.citationMode'.")
        return UiPathDeepRagTool(
            name=sanitize_tool_name(tool_name)
            if tool_name
            else _lowcode_tool_name(resource.name),
            description=resource.description
            or f"Researches the '{resource.index_name}' knowledge base.",
            index_name=resource.index_name,
            citation_mode=citation,
            query=query,
            query_from_input=query_from_input,
            folder_path_prefix=prefix,
            folder_path_prefix_from_input=prefix_from_input,
            file_extension=str(extension) if extension else None,
            retrieval_mode="DeepRAG",
            display_name=resource.name,
            mock_name=resource.name,
        )
    return UiPathContextTool(
        name=sanitize_tool_name(tool_name)
        if tool_name
        else _lowcode_tool_name(resource.name),
        description=resource.description
        or f"Searches the '{resource.index_name}' knowledge base and returns the "
        "most relevant passages.",
        index_name=resource.index_name,
        number_of_results=settings.result_count,
        threshold=settings.threshold,
        query=query,
        query_from_input=query_from_input,
        folder_path_prefix=prefix,
        folder_path_prefix_from_input=prefix_from_input,
        file_extension=str(extension) if extension else None,
        retrieval_mode=mode,
        display_name=resource.name,
        timeout_sec=timeout_sec,
        mock_name=resource.name,
    )


def _mock_output_schema(resource: Any) -> dict[str, Any] | None:
    """The resource's ``outputSchema`` when it declares fields (eval mocks)."""
    schema = getattr(resource, "output_schema", None)
    if isinstance(schema, dict) and schema.get("properties"):
        return copy.deepcopy(schema)
    return None


def _query_setting(label: str, setting: Any) -> tuple[str | None, str | None]:
    """``(static query, agent-input path)`` of a ``settings.query``.

    Like langchain: a missing setting is dynamic; a setting without a variant
    or a static one without a value is refused; any other variant
    (``textBuilder``, ``arrayBuilder``) is dynamic.
    """
    if setting is None:
        return None, None
    variant = _value(getattr(setting, "variant", None))
    if not variant:
        raise ValueError(f"{label}: the query has no variant.")
    variant = str(variant).lower()
    value = getattr(setting, "value", None)
    if variant == "static":
        if not value or not str(value).strip():
            raise ValueError(f"{label}: the static query has no value.")
        return str(value), None
    if variant == "argument" and value:
        return None, str(value)
    return None, None


def _prefix_setting(setting: Any) -> tuple[str | None, str | None]:
    if setting is None or not getattr(setting, "value", None):
        return None, None
    variant = str(_value(getattr(setting, "variant", None)) or "").lower()
    if variant == "static":
        return str(setting.value), None
    if variant == "argument":
        return None, str(setting.value)
    return None, None


def _input_path(path: str | None) -> str | None:
    """``{{a.b}}`` / ``a.b`` → ``a.b``; empty → None."""
    if path is None:
        return None
    stripped = path.strip().strip("{}").strip()
    return stripped or None


def _extension(ext: str | None) -> str | None:
    """``All`` (any case) and empty mean no filter."""
    if not ext or not ext.strip() or ext.strip().lower() == "all":
        return None
    return ext.strip().lstrip(".")


def _value(enum_or_str: Any) -> Any:
    return getattr(enum_or_str, "value", enum_or_str)
