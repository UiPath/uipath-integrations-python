"""User-facing agent definition for the Codex SDK integration.

Developers export a :class:`UiPathCodexAgent` from their entrypoint module
(``main.py:agent``). The UiPath runtime owns the Codex run loop
(``AsyncCodex`` → ``thread_start`` → ``run``) so it can point Codex at the
UiPath LLM Gateway and map the result to a UiPath runtime result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from .mcp import McpServer
from .tools import AnyUiPathTool

__all__ = ["UiPathCodexAgent", "CodexAgent"]


@dataclass
class UiPathCodexAgent:
    """Declarative Codex agent executed by the UiPath Codex SDK runtime.

    Args:
        model: Model id as listed by the tenant's discovery endpoint, e.g.
            ``"gpt-5.6-terra"`` or ``"gpt-4.1-2025-04-14"``. The runtime starts
            a local gateway shim and routes this model through the UiPath LLM
            Gateway.
        prompt: User-message template rendered with ``str.format`` from the
            validated input fields (e.g. ``"Tell me a joke about {topic}"``).
            When the input has no fields to interpolate, it is used verbatim.
        input_schema: Optional Pydantic model describing structured input. When
            set, input is validated against it before the prompt is rendered.
        output_schema: Optional Pydantic model describing structured output.
            When set, the runtime requests native structured output from Codex.
        base_instructions: Optional system/base instructions for the thread.
        sandbox: Codex sandbox preset name (``"read-only"``,
            ``"workspace-write"``, ``"full-access"``). Defaults to read-only —
            a plain LLM agent never needs to touch the filesystem. Note that
            ``"workspace-write"`` relies on Codex's OS-level sandbox (Linux
            namespaces/landlock); in restricted containers that lack those
            capabilities, use ``"full-access"`` (the run is already contained
            by the robot/host).
        approval_mode: How Codex handles escalated permission requests:
            ``"deny_all"`` (default here — never escalates, no reviewer call)
            or ``"auto_review"`` (Codex's own default — an automated reviewer
            decides, which issues an extra LLM call using a ``codex-auto-review``
            model that the UiPath gateway may not route). We default to
            ``"deny_all"``; the public async SDK exposes no approval-handler
            hook, and the codex process auto-accepts command/file approvals, so
            ``"deny_all"`` (with a ``"full-access"`` sandbox for tool/coding
            agents) is the reliable posture on the gateway.
        mcp_servers: MCP servers to expose to Codex as tools, keyed by server
            name. Values are ``StdioMcpServer`` (local subprocess) or
            ``RemoteMcpServer`` (streamable-HTTP endpoint).
        codex_bin: Path to a specific ``codex`` binary to run instead of the
            one bundled with ``openai-codex``. Falls back to the
            ``UIPATH_CODEX_BIN`` env var, then the bundled binary. Use this to
            target a newer stable CLI (e.g. rust-v0.153.4).
        name: Display name used in the runtime graph.
        tools: UiPath tools, served to Codex as the ``uipath_tools`` MCP
            server. **Durable** tools
            (:func:`~uipath_codex_sdk.uipath_process_tool`,
            :func:`~uipath_codex_sdk.uipath_agent_tool`) start an Orchestrator
            job and suspend the run until the job finishes; the job output
            then becomes the result of the model's original call.
            **Immediate** tools
            (:func:`~uipath_codex_sdk.uipath_integration_tool`, or any
            :class:`~uipath_codex_sdk.UiPathTool`) are answered as soon as
            their work is done and never suspend the run.
        codex_home: ``CODEX_HOME`` for the codex process. With durable tools
            (and for a conversational agent) it defaults to a run-scoped
            directory under the runtime dir
            (``__uipath/codex_home``), reset together with the state database
            on a fresh local run; ``~/.codex`` (config.toml, AGENTS.md) is then
            not read. Without durable tools it defaults to codex's own
            resolution (``$CODEX_HOME``, else ``~/.codex``), as before.
        hermetic: Pass ``--config`` overrides that stop codex from calling the
            ChatGPT backend (``features.plugins=false`` and an unreachable
            ``chatgpt_base_url``). ``None`` (default) turns them on only for
            agents with durable tools and conversational agents.
        durable_tool_instructions: Tag durable tool descriptions with
            ``[long-running]`` and add one developer instruction about them
            (call independent ones together; never retry one that is pending).
        durable_settle_timeout: Seconds a suspension waits for non-durable
            sibling calls of the same model response to finish before Codex is
            stopped anyway.
    """

    model: str
    prompt: str | None = None
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    base_instructions: str | None = None
    sandbox: str = "read-only"
    approval_mode: str = "deny_all"
    mcp_servers: dict[str, McpServer] | None = field(default=None)
    codex_bin: str | None = None
    name: str = "agent"
    tools: list[AnyUiPathTool] | None = field(default=None)
    codex_home: str | None = None
    hermetic: bool | None = None
    durable_tool_instructions: bool = True
    durable_settle_timeout: float = 120.0


CodexAgent = UiPathCodexAgent
