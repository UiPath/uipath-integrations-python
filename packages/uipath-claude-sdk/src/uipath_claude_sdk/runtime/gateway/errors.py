"""Anthropic-shaped error rendering for the gateway shim.

The bundled Claude Code CLI only understands Anthropic's error envelope, so
every failure the shim reports has to be translated into that shape before it
reaches the client.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

ANTHROPIC_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",
}

_MESSAGE_KEYS = ("message", "Message", "detail", "title", "error_description")
_CORRELATION_KEYS = ("correlationId", "traceId", "requestId", "request_id")
_MAX_MESSAGE_CHARS = 2000


class GatewayShimError(Exception):
    """Base class for failures the shim can express as an Anthropic error."""

    status: int = 500
    error_type: str = "api_error"


class ModelNotInCatalogError(GatewayShimError):
    """The requested model does not resolve to any id in the tenant catalog."""

    status = 404
    error_type = "not_found_error"

    def __init__(self, requested: str, available: Sequence[str]) -> None:
        self.requested = requested
        self.available = list(available)
        super().__init__(
            f"Model '{requested}' is not available on this UiPath tenant. "
            f"Available models: {', '.join(self.available) or '(none reported by discovery)'}."
        )


class UnsupportedApiFlavorError(GatewayShimError):
    """Discovery resolved the model to a wire format the shim cannot route."""

    status = 400
    error_type = "invalid_request_error"

    def __init__(self, model_id: str, vendor_type: str, api_flavor: str | None) -> None:
        self.model_id = model_id
        self.vendor_type = vendor_type
        self.api_flavor = api_flavor
        super().__init__(
            f"Model '{model_id}' resolves to vendor '{vendor_type}' with api flavor "
            f"'{api_flavor}', which the UiPath gateway shim cannot route. Supported "
            f"flavors are AnthropicMessages and invoke."
        )


def anthropic_error_type(status: int) -> str:
    """Map an HTTP status onto the Anthropic error type name."""
    if status in ANTHROPIC_ERROR_TYPES:
        return ANTHROPIC_ERROR_TYPES[status]
    if 400 <= status < 500:
        return "invalid_request_error"
    return "api_error"


def error_payload(
    status: int,
    message: str,
    *,
    error_type: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build an Anthropic error envelope for a known message."""
    payload: dict[str, Any] = {
        "type": "error",
        "error": {
            "type": error_type or anthropic_error_type(status),
            "message": message[:_MAX_MESSAGE_CHARS],
        },
    }
    if correlation_id:
        payload["request_id"] = correlation_id
    return payload


def to_anthropic_error(
    status: int,
    body: bytes | str | dict[str, Any] | None,
    *,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Translate a gateway failure into the Anthropic error envelope.

    Preserves the gateway's own message and correlation id when the body
    carries them, and falls back to the decoded body text otherwise.
    """
    parsed = _parse_body(body)
    message = _extract_message(parsed)
    if message is None:
        message = _fallback_message(body, status)
    return error_payload(
        status,
        message,
        correlation_id=correlation_id or _extract_correlation_id(parsed),
    )


def _parse_body(body: bytes | str | dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(body, dict):
        return body
    if not body:
        return None
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    try:
        loaded = json.loads(text)
    except (ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _extract_message(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    inner = parsed.get("error")
    if isinstance(inner, dict):
        nested = _first_string(inner, _MESSAGE_KEYS)
        if nested is not None:
            return nested
    elif isinstance(inner, str) and inner.strip():
        return inner
    return _first_string(parsed, _MESSAGE_KEYS)


def _extract_correlation_id(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    inner = parsed.get("error")
    if isinstance(inner, dict):
        nested = _first_string(inner, _CORRELATION_KEYS)
        if nested is not None:
            return nested
    return _first_string(parsed, _CORRELATION_KEYS)


def _first_string(source: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _fallback_message(body: bytes | str | dict[str, Any] | None, status: int) -> str:
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace").strip()
    elif isinstance(body, str):
        text = body.strip()
    else:
        text = ""
    return text or f"UiPath LLM Gateway returned HTTP {status}."


__all__ = [
    "ANTHROPIC_ERROR_TYPES",
    "GatewayShimError",
    "ModelNotInCatalogError",
    "UnsupportedApiFlavorError",
    "anthropic_error_type",
    "error_payload",
    "to_anthropic_error",
]
