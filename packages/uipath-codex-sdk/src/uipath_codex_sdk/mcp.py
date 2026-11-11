"""MCP server declarations for Codex agents.

Codex loads MCP servers from its config. We pass them as ``--config`` overrides
(the same channel used for the UiPath model_provider), which Codex parses as
TOML — e.g. a stdio server becomes::

    [mcp_servers.<name>]
    command = "python3"
    args = ["server.py"]
    [mcp_servers.<name>.env]
    KEY = "value"
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "StdioMcpServer",
    "RemoteMcpServer",
    "McpServer",
    "build_mcp_config_overrides",
    "build_mcp_env",
    "missing_bearer_token_env_vars",
]


@dataclass
class StdioMcpServer:
    """A local MCP server Codex launches as a stdio subprocess.

    Args:
        command: Executable to launch (e.g. ``sys.executable`` or ``"npx"``).
        args: Arguments passed to the command.
        env: Extra environment variables for the server process.
        default_tools_approval_mode: Codex approval mode for this server's
            tools: ``"auto"`` (Codex default: approval required unless the tool
            is annotated ``readOnlyHint``), ``"prompt"``, ``"writes"`` or
            ``"approve"`` (never ask). With the runtime's ``approval_mode=
            "deny_all"`` any tool that needs approval is rejected, so trusted
            servers whose tools carry no annotations (e.g. UiPath MCP) need
            ``"approve"``.
        tool_approval_modes: Per-tool override of the above, ``{tool: mode}``.
        omit_tools_from: Model-facing surfaces this server's tools are kept
            off: ``"direct"`` (initial tool list), ``"deferred"`` (found via
            tool search) and ``"code_mode"`` (nested calls from ``exec``
            scripts). E.g. ``["deferred", "code_mode"]`` makes the tools always
            directly callable.
        supports_parallel_tool_calls: Whether Codex may run several of this
            server's tool calls from one model response at the same time.
            Codex serializes MCP calls by default; ``True`` lets them overlap.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    default_tools_approval_mode: str | None = None
    tool_approval_modes: dict[str, str] | None = None
    supports_parallel_tool_calls: bool | None = None
    omit_tools_from: list[str] | None = None


@dataclass
class RemoteMcpServer:
    """A remote (streamable-HTTP) MCP server Codex connects to over the network.

    Args:
        url: Streamable-HTTP endpoint of the MCP server (e.g.
            ``"https://host/mcp"``).
        bearer_token_env_var: Name of an environment variable Codex reads and
            sends as ``Authorization: Bearer <value>``. For a UiPath
            AgentHub-hosted MCP server this is typically
            ``"UIPATH_ACCESS_TOKEN"`` (already present in the environment). NOTE:
            if this names a variable that is NOT set in the codex process
            environment, codex silently skips the server (no tools). Either
            ensure the variable is exported, or pass ``bearer_token`` below.
        bearer_token: Optional literal token value. When set (together with
            ``bearer_token_env_var``), the runtime injects it into the codex
            process environment under that variable name, so the server is
            reachable without relying on an ambient env var.
        http_headers: Additional static HTTP headers to send on every request.
        startup_timeout_sec: How long codex waits for the server to initialize
            and list its tools.
        tool_timeout_sec: Per tool-call timeout (codex default: 300s). UiPath
            MCP tools that start an Orchestrator job can take longer.
        default_tools_approval_mode: Codex approval mode for this server's
            tools: ``"auto"`` (Codex default: approval required unless the tool
            is annotated ``readOnlyHint``), ``"prompt"``, ``"writes"`` or
            ``"approve"`` (never ask). With the runtime's ``approval_mode=
            "deny_all"`` any tool that needs approval is rejected, so trusted
            servers whose tools carry no annotations (e.g. UiPath MCP) need
            ``"approve"``.
        tool_approval_modes: Per-tool override of the above, ``{tool: mode}``.
        omit_tools_from: Model-facing surfaces this server's tools are kept
            off: ``"direct"`` (initial tool list), ``"deferred"`` (found via
            tool search) and ``"code_mode"`` (nested calls from ``exec``
            scripts). E.g. ``["deferred", "code_mode"]`` makes the tools always
            directly callable.
        supports_parallel_tool_calls: Whether Codex may run several of this
            server's tool calls from one model response at the same time.
            Codex serializes MCP calls by default; ``True`` lets them overlap.
    """

    url: str
    bearer_token_env_var: str | None = None
    bearer_token: str | None = None
    http_headers: dict[str, str] | None = None
    startup_timeout_sec: float | None = None
    tool_timeout_sec: float | None = None
    default_tools_approval_mode: str | None = None
    tool_approval_modes: dict[str, str] | None = None
    supports_parallel_tool_calls: bool | None = None
    omit_tools_from: list[str] | None = None


#: Any MCP server declaration accepted by ``UiPathCodexAgent.mcp_servers``.
McpServer = StdioMcpServer | RemoteMcpServer


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_str_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_inline_table(mapping: dict[str, str]) -> str:
    items = ", ".join(f"{_toml_str(k)} = {_toml_str(v)}" for k, v in mapping.items())
    return "{" + items + "}"


def build_mcp_config_overrides(
    servers: dict[str, McpServer] | None,
) -> tuple[str, ...]:
    """Build Codex ``--config`` override strings defining the given MCP servers."""
    if not servers:
        return ()
    overrides: list[str] = []
    for name, server in servers.items():
        prefix = f"mcp_servers.{name}"
        if isinstance(server, RemoteMcpServer):
            overrides.append(f"{prefix}.url={_toml_str(server.url)}")
            if server.bearer_token_env_var:
                overrides.append(
                    f"{prefix}.bearer_token_env_var={_toml_str(server.bearer_token_env_var)}"
                )
            if server.http_headers:
                overrides.append(
                    f"{prefix}.http_headers={_toml_inline_table(server.http_headers)}"
                )
            if server.startup_timeout_sec is not None:
                overrides.append(
                    f"{prefix}.startup_timeout_sec={float(server.startup_timeout_sec)}"
                )
            if server.tool_timeout_sec is not None:
                overrides.append(
                    f"{prefix}.tool_timeout_sec={float(server.tool_timeout_sec)}"
                )
        else:  # StdioMcpServer
            overrides.append(f"{prefix}.command={_toml_str(server.command)}")
            if server.args:
                overrides.append(f"{prefix}.args={_toml_str_list(server.args)}")
            if server.env:
                overrides.append(f"{prefix}.env={_toml_inline_table(server.env)}")
        overrides.extend(_approval_overrides(prefix, server))
        if server.supports_parallel_tool_calls is not None:
            flag = "true" if server.supports_parallel_tool_calls else "false"
            overrides.append(f"{prefix}.supports_parallel_tool_calls={flag}")
        if server.omit_tools_from is not None:
            for surface in server.omit_tools_from:
                if surface not in _SURFACES:
                    raise ValueError(
                        f"invalid omit_tools_from surface {surface!r}; expected one of {_SURFACES}"
                    )
            overrides.append(
                f"{prefix}.omit_tools_from={_toml_str_list(server.omit_tools_from)}"
            )
    return tuple(overrides)


_APPROVAL_MODES = ("auto", "prompt", "writes", "approve")
_SURFACES = ("direct", "deferred", "code_mode")


def _check_mode(mode: str) -> str:
    if mode not in _APPROVAL_MODES:
        raise ValueError(
            f"invalid MCP approval mode {mode!r}; expected one of {_APPROVAL_MODES}"
        )
    return mode


def _approval_overrides(prefix: str, server: McpServer) -> list[str]:
    out: list[str] = []
    if server.default_tools_approval_mode is not None:
        mode = _check_mode(server.default_tools_approval_mode)
        out.append(f"{prefix}.default_tools_approval_mode={_toml_str(mode)}")
    for tool, mode in (server.tool_approval_modes or {}).items():
        out.append(
            f"{prefix}.tools.{tool}.approval_mode={_toml_str(_check_mode(mode))}"
        )
    return out


def build_mcp_env(servers: dict[str, McpServer] | None) -> dict[str, str]:
    """Env vars to inject into the codex process for remote MCP bearer tokens.

    For each :class:`RemoteMcpServer` that carries a literal ``bearer_token``
    plus a ``bearer_token_env_var``, map the variable name to the value so
    codex can send it as ``Authorization: Bearer``.
    """
    env: dict[str, str] = {}
    for server in (servers or {}).values():
        if (
            isinstance(server, RemoteMcpServer)
            and server.bearer_token is not None
            and server.bearer_token_env_var
        ):
            env[server.bearer_token_env_var] = server.bearer_token
    return env


def missing_bearer_token_env_vars(
    servers: dict[str, McpServer] | None, environ: dict[str, str]
) -> dict[str, str]:
    """Map server name → unresolved bearer_token_env_var.

    A remote server whose ``bearer_token_env_var`` is neither given as a literal
    ``bearer_token`` nor present in ``environ`` will be silently skipped by
    codex; callers should warn about these.
    """
    missing: dict[str, str] = {}
    for name, server in (servers or {}).items():
        if not isinstance(server, RemoteMcpServer):
            continue
        var = server.bearer_token_env_var
        if not var:
            continue
        if server.bearer_token is None and not environ.get(var):
            missing[name] = var
    return missing
