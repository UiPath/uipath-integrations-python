"""UiPath Codex SDK Integration."""

from typing import Any

from .agent import CodexAgent, UiPathCodexAgent
from .attachment_tools import (
    UiPathAnalyzeFilesTool,
    analyze_files_tool_from_resource,
    uipath_analyze_files_tool,
)
from .attachments import AttachmentStore, UiPathAttachment
from .batch_transform_tools import (
    UiPathBatchTransformAttachmentsTool,
    batch_transform_tool_from_resource,
    uipath_batch_transform_attachments_tool,
)
from .context_tools import (
    UiPathContextTool,
    UiPathDeepRagAttachmentsTool,
    UiPathDeepRagTool,
    context_tool_from_resource,
    deep_rag_tool_from_resource,
    uipath_context_tool,
    uipath_deep_rag_attachments_tool,
    uipath_deep_rag_tool,
)
from .escalation_tools import (
    UiPathEscalationTool,
    escalation_tool_from_resource,
    uipath_escalation_tool,
)
from .integration_tools import (
    UiPathIntegrationTool,
    integration_tool_from_resource,
    uipath_integration_tool,
)
from .ixp_tools import (
    UiPathDocumentValidationTool,
    UiPathIxpExtractionTool,
    ixp_extraction_tool_from_resource,
    uipath_document_validation_tool,
    uipath_ixp_extraction_tool,
    vs_escalation_tool_from_resource,
)
from .mcp import RemoteMcpServer, StdioMcpServer
from .process_tools import process_tool_from_resource
from .tools import (
    AnyUiPathTool,
    DurableToolContext,
    ToolContext,
    UiPathDurableTool,
    UiPathDurableToolError,
    UiPathDurableToolPending,
    UiPathProcessTool,
    UiPathTool,
    UiPathToolError,
    uipath_agent_tool,
    uipath_process_tool,
)

__all__ = [
    "AnyUiPathTool",
    "AttachmentStore",
    "CodexAgent",
    "DurableToolContext",
    "RemoteMcpServer",
    "StdioMcpServer",
    "ToolContext",
    "UiPathAnalyzeFilesTool",
    "UiPathAttachment",
    "UiPathBatchTransformAttachmentsTool",
    "UiPathCodexAgent",
    "UiPathContextTool",
    "UiPathDeepRagAttachmentsTool",
    "UiPathDocumentValidationTool",
    "UiPathDeepRagTool",
    "UiPathDurableTool",
    "UiPathDurableToolError",
    "UiPathDurableToolPending",
    "UiPathEscalationTool",
    "UiPathIntegrationTool",
    "UiPathIxpExtractionTool",
    "UiPathProcessTool",
    "UiPathTool",
    "UiPathToolError",
    "analyze_files_tool_from_resource",
    "batch_transform_tool_from_resource",
    "context_tool_from_resource",
    "deep_rag_tool_from_resource",
    "escalation_tool_from_resource",
    "integration_tool_from_resource",
    "ixp_extraction_tool_from_resource",
    "process_tool_from_resource",
    "register_middleware",
    "uipath_agent_tool",
    "uipath_analyze_files_tool",
    "uipath_batch_transform_attachments_tool",
    "uipath_context_tool",
    "uipath_deep_rag_attachments_tool",
    "uipath_deep_rag_tool",
    "uipath_document_validation_tool",
    "uipath_escalation_tool",
    "uipath_integration_tool",
    "uipath_ixp_extraction_tool",
    "uipath_process_tool",
    "vs_escalation_tool_from_resource",
]


def __getattr__(name: str) -> Any:
    if name == "register_middleware":
        from .middlewares import register_middleware

        return register_middleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
