"""Helpers for pointing a Claude agent at MCP servers.

MCP support comes from the Claude Agent SDK itself: anything placed in
``ClaudeAgentOptions.mcp_servers`` is started and connected by the Claude Code
CLI. This module only removes the boilerplate around the two remote transports
(``http`` and ``sse``) and around resolving a UiPath-hosted MCP server to the
URL and bearer token it expects. Stdio and in-process SDK servers need no
helper: pass their config dictionaries straight through.
"""

from __future__ import annotations

import os
from typing import Literal

from claude_agent_sdk.types import (
    McpHttpServerConfig,
    McpServerConfig,
    McpSSEServerConfig,
)

from .interrupts import UIPATH_MCP_SERVER_NAME

McpTransport = Literal["http", "sse"]
RemoteMcpServerConfig = McpHttpServerConfig | McpSSEServerConfig

_ACCESS_TOKEN_ENV = "UIPATH_ACCESS_TOKEN"

__all__ = [
    "UIPATH_MCP_SERVER_NAME",
    "McpTransport",
    "RemoteMcpServerConfig",
    "validate_mcp_servers",
    "remote_mcp_server",
    "uipath_mcp_server",
]


def remote_mcp_server(
    url: str,
    *,
    access_token: str | None = None,
    transport: McpTransport = "http",
    headers: dict[str, str] | None = None,
) -> RemoteMcpServerConfig:
    """Build the config for an MCP server reached over HTTP or SSE.

    Args:
        url: Endpoint of the MCP server. UiPath-hosted servers expose a
            streamable HTTP endpoint, so ``transport="http"`` is the default.
        access_token: Bearer token to authenticate with. Defaults to the
            ``UIPATH_ACCESS_TOKEN`` environment variable, which the UiPath
            runtime populates. No ``Authorization`` header is sent when
            neither is available, which is what a public MCP server wants.
        transport: ``"http"`` for streamable HTTP, ``"sse"`` for server-sent
            events.
        headers: Extra headers merged into the request. An ``Authorization``
            entry here wins over ``access_token``.

    Returns:
        A config to place under a key of ``ClaudeAgentOptions.mcp_servers``.
        The key must not be ``uipath``: see :func:`validate_mcp_servers`.

    Raises:
        ValueError: If ``url`` is empty or ``transport`` is not a remote one.
    """
    if not url:
        raise ValueError("A remote MCP server needs a non-empty url.")

    resolved = dict(headers or {})
    token = (
        access_token if access_token is not None else os.environ.get(_ACCESS_TOKEN_ENV)
    )
    if token and "Authorization" not in resolved:
        resolved["Authorization"] = f"Bearer {token}"

    if transport == "http":
        http_config: McpHttpServerConfig = {"type": "http", "url": url}
        if resolved:
            http_config["headers"] = resolved
        return http_config

    if transport == "sse":
        sse_config: McpSSEServerConfig = {"type": "sse", "url": url}
        if resolved:
            sse_config["headers"] = resolved
        return sse_config

    raise ValueError(
        f"Unsupported MCP transport {transport!r}. Use 'http' or 'sse', or pass "
        "a stdio or SDK server config directly."
    )


def uipath_mcp_server(
    slug: str | None = None,
    *,
    name: str | None = None,
    folder_path: str | None = None,
    access_token: str | None = None,
    transport: McpTransport = "http",
    headers: dict[str, str] | None = None,
) -> RemoteMcpServerConfig:
    """Resolve a UiPath-hosted MCP server and build its config.

    The server is looked up through the UiPath SDK, which returns the endpoint
    the tenant serves it on. The lookup is a network call made when this
    function runs, so calling it while building a module-level agent means
    ``uipath init`` and ``uipath run`` both need an authenticated tenant that
    already hosts the server. Pass the endpoint to :func:`remote_mcp_server`
    instead when that is not wanted.

    Args:
        slug: Legacy slug of the server.
        name: Display name of the server. Give either this or ``slug``.
        folder_path: Folder the server lives in. Defaults to the folder of the
            current execution context.
        access_token: Bearer token to authenticate with. Defaults to the
            credential the UiPath SDK resolved.
        transport: ``"http"`` for streamable HTTP, ``"sse"`` for server-sent
            events.
        headers: Extra headers merged into the request.

    Returns:
        A config to place under a key of ``ClaudeAgentOptions.mcp_servers``.
        The key must not be ``uipath``: see :func:`validate_mcp_servers`.

    Raises:
        ValueError: If the server has no endpoint configured.
    """
    from uipath.platform import UiPath

    sdk = UiPath()
    server = sdk.mcp.retrieve(slug, name=name, folder_path=folder_path)
    if not server.mcp_url:
        raise ValueError(
            f"UiPath MCP server {name or slug!r} has no URL configured, so the "
            "agent cannot connect to it."
        )

    return remote_mcp_server(
        server.mcp_url,
        access_token=access_token or sdk._config.secret,
        transport=transport,
        headers=headers,
    )


def validate_mcp_servers(
    servers: dict[str, McpServerConfig],
) -> dict[str, McpServerConfig]:
    """Validate an ``mcp_servers`` mapping while the agent is being defined.

    The runtime registers its own in-process server under the ``uipath`` key,
    which is where the interrupt tools that suspend a run live. A developer
    server under that key would hide them, so the runtime refuses to start.
    Passing the mapping through here surfaces the collision when the module is
    imported rather than on the first run.

    Args:
        servers: Mapping of server name to config, as
            ``ClaudeAgentOptions.mcp_servers`` accepts it.

    Returns:
        The same mapping, copied.

    Raises:
        ValueError: If the mapping uses the reserved ``uipath`` key.
    """
    if UIPATH_MCP_SERVER_NAME in servers:
        raise ValueError(
            f"MCP server name {UIPATH_MCP_SERVER_NAME!r} is reserved for the "
            "UiPath interrupt tools. Rename that server so its tools stay "
            "reachable."
        )
    return dict(servers)
