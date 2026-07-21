"""Tests for the UiPath PydanticAI chat client and lazy chat module exports."""

import httpx
import pytest

# ============= LAZY MODULE EXPORT TESTS =============


def test_chat_module_lazy_exports_client():
    """__getattr__ lazily resolves UiPathChatOpenAI to the real class."""
    import uipath_pydantic_ai.chat as chat
    from uipath_pydantic_ai.chat.openai import UiPathChatOpenAI

    assert chat.UiPathChatOpenAI is UiPathChatOpenAI


def test_chat_module_lazy_exports_models():
    """__getattr__ lazily resolves the supported-model enums."""
    import uipath_pydantic_ai.chat as chat
    from uipath_pydantic_ai.chat.supported_models import OpenAIModels

    assert chat.OpenAIModels is OpenAIModels


def test_chat_module_unknown_attribute_raises():
    """Accessing an unknown attribute raises AttributeError."""
    import uipath_pydantic_ai.chat as chat

    with pytest.raises(AttributeError):
        _ = chat.DoesNotExist


# ============= SUPPORTED MODEL ENUM TESTS =============


def test_openai_model_enum_is_string_value():
    """OpenAIModels is a StrEnum whose members equal their wire string."""
    from uipath_pydantic_ai.chat.supported_models import OpenAIModels

    assert OpenAIModels.gpt_4o_2024_11_20 == "gpt-4o-2024-11-20"


# ============= URL REWRITE TESTS =============


def test_rewrite_url_responses_endpoint():
    """A /responses path is rewritten to /completions."""
    from uipath_pydantic_ai.chat.openai import _rewrite_openai_url

    result = _rewrite_openai_url(
        "https://gw.uipath.com/llm/openai/responses", httpx.QueryParams()
    )
    assert str(result) == "https://gw.uipath.com/llm/openai/completions"


def test_rewrite_url_chat_completions_endpoint():
    """A /chat/completions path is collapsed to /completions."""
    from uipath_pydantic_ai.chat.openai import _rewrite_openai_url

    result = _rewrite_openai_url(
        "https://gw.uipath.com/llm/openai/chat/completions", httpx.QueryParams()
    )
    assert str(result) == "https://gw.uipath.com/llm/openai/completions"


def test_rewrite_url_preserves_query_params():
    """Query params are carried onto the rewritten URL."""
    from uipath_pydantic_ai.chat.openai import _rewrite_openai_url

    params = httpx.QueryParams({"api-version": "2024-12-01-preview"})
    result = _rewrite_openai_url("https://gw.uipath.com/llm/openai/completions", params)
    assert result is not None
    assert result.params["api-version"] == "2024-12-01-preview"


def test_rewrite_url_base_only_strips_query():
    """A bare base URL (no known path) gets /completions appended."""
    from uipath_pydantic_ai.chat.openai import _rewrite_openai_url

    result = _rewrite_openai_url(
        "https://gw.uipath.com/llm/openai?foo=bar", httpx.QueryParams()
    )
    assert str(result) == "https://gw.uipath.com/llm/openai/completions"


# ============= CLIENT CONSTRUCTION TESTS =============


def test_client_requires_org_id(monkeypatch):
    """Missing organization id raises a descriptive ValueError."""
    from uipath_pydantic_ai.chat.openai import UiPathChatOpenAI

    monkeypatch.delenv("UIPATH_ORGANIZATION_ID", raising=False)
    with pytest.raises(ValueError, match="UIPATH_ORGANIZATION_ID"):
        UiPathChatOpenAI(tenant_id="t", token="tok")


def test_client_build_headers_and_base_url(monkeypatch):
    """A fully configured client builds gateway headers and a base URL from UIPATH_URL."""
    from uipath_pydantic_ai.chat.openai import UiPathChatOpenAI

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant/")
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    monkeypatch.delenv("UIPATH_PROCESS_KEY", raising=False)

    client = UiPathChatOpenAI(
        token="my-token",
        org_id="org",
        tenant_id="tenant",
        agenthub_config="cfg",
        byo_connection_id="conn-1",
    )

    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer my-token"
    assert headers["X-UiPath-LlmGateway-ApiFlavor"] == "chat-completions"
    assert headers["X-UiPath-AgentHub-Config"] == "cfg"
    assert headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "conn-1"

    # endpoint has the /completions suffix stripped; base url is prefixed by UIPATH_URL
    assert not client.endpoint.endswith("/completions")
    assert client._build_base_url() == (
        f"https://cloud.uipath.com/org/tenant/{client.endpoint}"
    )
    assert client.model_name == client._model_name


def test_client_requires_uipath_url(monkeypatch):
    """Building the base URL without UIPATH_URL set raises ValueError."""
    from uipath_pydantic_ai.chat.openai import UiPathChatOpenAI

    monkeypatch.delenv("UIPATH_URL", raising=False)
    with pytest.raises(ValueError, match="UIPATH_URL"):
        UiPathChatOpenAI(token="tok", org_id="org", tenant_id="tenant")


def test_client_extra_headers_override_defaults(monkeypatch):
    """extra_headers take precedence over the built-in gateway defaults."""
    from uipath_pydantic_ai.chat.openai import UiPathChatOpenAI

    monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")

    client = UiPathChatOpenAI(
        token="tok",
        org_id="org",
        tenant_id="tenant",
        extra_headers={"X-UiPath-LlmGateway-ApiFlavor": "responses"},
    )
    assert client._build_headers()["X-UiPath-LlmGateway-ApiFlavor"] == "responses"
