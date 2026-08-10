"""Tests for the MCP server helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from uipath_claude_sdk.interrupts import UIPATH_MCP_SERVER_NAME
from uipath_claude_sdk.mcp import (
    remote_mcp_server,
    uipath_mcp_server,
    validate_mcp_servers,
)

URL = "https://cloud.uipath.com/org/tenant/agenthub_/mcp/demo/mcp"


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)


class TestRemoteMcpServer:
    def test_http_is_the_default_transport(self) -> None:
        assert remote_mcp_server(URL) == {"type": "http", "url": URL}

    def test_sse_transport_shape(self) -> None:
        assert remote_mcp_server(URL, transport="sse") == {
            "type": "sse",
            "url": URL,
        }

    def test_access_token_becomes_a_bearer_header(self) -> None:
        config = remote_mcp_server(URL, access_token="tok")

        assert config["headers"] == {"Authorization": "Bearer tok"}

    def test_token_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "from-env")

        config = remote_mcp_server(URL)

        assert config["headers"] == {"Authorization": "Bearer from-env"}

    def test_explicit_token_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "from-env")

        config = remote_mcp_server(URL, access_token="explicit")

        assert config["headers"] == {"Authorization": "Bearer explicit"}

    def test_no_token_leaves_the_request_unauthenticated(self) -> None:
        assert "headers" not in remote_mcp_server(URL)

    def test_empty_token_leaves_the_request_unauthenticated(self) -> None:
        assert "headers" not in remote_mcp_server(URL, access_token="")

    def test_extra_headers_are_merged_with_the_bearer_header(self) -> None:
        config = remote_mcp_server(
            URL, access_token="tok", headers={"X-UIPATH-FolderKey": "folder-1"}
        )

        assert config["headers"] == {
            "X-UIPATH-FolderKey": "folder-1",
            "Authorization": "Bearer tok",
        }

    def test_caller_authorization_header_wins(self) -> None:
        config = remote_mcp_server(
            URL, access_token="tok", headers={"Authorization": "Basic abc"}
        )

        assert config["headers"] == {"Authorization": "Basic abc"}

    def test_caller_headers_are_copied(self) -> None:
        headers = {"X-Trace": "1"}

        remote_mcp_server(URL, access_token="tok", headers=headers)

        assert headers == {"X-Trace": "1"}

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty url"):
            remote_mcp_server("")

    def test_unsupported_transport_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            remote_mcp_server(URL, transport="stdio")  # type: ignore[arg-type]


@dataclass
class FakeMcpServer:
    mcp_url: str | None = URL


@dataclass
class FakeMcpService:
    server: FakeMcpServer
    calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)

    def retrieve(self, slug: Any = None, **kwargs: Any) -> FakeMcpServer:
        self.calls.append((slug, kwargs))
        return self.server


@dataclass
class FakeConfig:
    secret: str = "sdk-secret"


class FakeUiPath:
    def __init__(self, server: FakeMcpServer | None = None) -> None:
        self.mcp = FakeMcpService(server or FakeMcpServer())
        self._config = FakeConfig()


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> list[FakeUiPath]:
    created: list[FakeUiPath] = []
    server = FakeMcpServer()

    def factory() -> FakeUiPath:
        sdk = FakeUiPath(server)
        created.append(sdk)
        return sdk

    monkeypatch.setattr("uipath.platform.UiPath", factory)
    return created


class TestUiPathMcpServer:
    def test_resolves_the_tenant_url_and_credential(
        self, fake_sdk: list[FakeUiPath]
    ) -> None:
        config = uipath_mcp_server("demo-server", folder_path="Shared")

        assert config == {
            "type": "http",
            "url": URL,
            "headers": {"Authorization": "Bearer sdk-secret"},
        }
        assert fake_sdk[0].mcp.calls == [
            ("demo-server", {"name": None, "folder_path": "Shared"})
        ]

    def test_looks_the_server_up_by_display_name(
        self, fake_sdk: list[FakeUiPath]
    ) -> None:
        uipath_mcp_server(name="My Server")

        assert fake_sdk[0].mcp.calls == [
            (None, {"name": "My Server", "folder_path": None})
        ]

    def test_sse_transport_is_selectable(self, fake_sdk: list[FakeUiPath]) -> None:
        config = uipath_mcp_server("demo-server", transport="sse")

        assert config["type"] == "sse"

    def test_explicit_token_wins_over_the_sdk_credential(
        self, fake_sdk: list[FakeUiPath]
    ) -> None:
        config = uipath_mcp_server("demo-server", access_token="explicit")

        assert config["headers"] == {"Authorization": "Bearer explicit"}

    def test_server_without_a_url_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "uipath.platform.UiPath", lambda: FakeUiPath(FakeMcpServer(mcp_url=None))
        )

        with pytest.raises(ValueError, match="has no URL configured"):
            uipath_mcp_server("demo-server")


class TestMcpServers:
    def test_returns_a_copy_of_the_mapping(self) -> None:
        servers: dict[str, Any] = {"demo": remote_mcp_server(URL)}

        validated = validate_mcp_servers(servers)

        assert validated == servers
        assert validated is not servers

    def test_reserved_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=UIPATH_MCP_SERVER_NAME):
            validate_mcp_servers({UIPATH_MCP_SERVER_NAME: remote_mcp_server(URL)})

    def test_empty_mapping_is_allowed(self) -> None:
        assert validate_mcp_servers({}) == {}
