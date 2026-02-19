"""Shared utilities for UiPath Agent Framework chat providers."""

import os
from typing import Optional

from uipath.utils import EndpointManager


def get_uipath_config() -> tuple[str, str]:
    """Read UiPath configuration from environment variables.

    Returns:
        Tuple of (uipath_url, access_token).

    Raises:
        ValueError: If required environment variables are not set.
    """
    uipath_url = os.getenv("UIPATH_URL")
    if not uipath_url:
        raise ValueError("UIPATH_URL environment variable is required")

    access_token = os.getenv("UIPATH_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("UIPATH_ACCESS_TOKEN environment variable is required")

    return uipath_url, access_token


def get_uipath_headers(token: str) -> dict[str, str]:
    """Build HTTP headers for UiPath Gateway requests.

    Args:
        token: UiPath access token.

    Returns:
        Dictionary of HTTP headers.
    """
    headers = {
        "Authorization": f"Bearer {token}",
    }
    if job_key := os.getenv("UIPATH_JOB_KEY"):
        headers["X-UiPath-JobKey"] = job_key
    if process_key := os.getenv("UIPATH_PROCESS_KEY"):
        headers["X-UiPath-ProcessKey"] = process_key
    return headers


def build_gateway_url(
    vendor: str,
    model: str,
    uipath_url: Optional[str] = None,
) -> str:
    """Build the full URL for the UiPath LLM Gateway.

    Args:
        vendor: The LLM vendor (e.g., "openai", "awsbedrock").
        model: The model name.
        uipath_url: Optional UiPath URL. If not provided, reads from UIPATH_URL env var.

    Returns:
        The full gateway URL.

    Raises:
        ValueError: If UIPATH_URL is not set.
    """
    if not uipath_url:
        uipath_url = os.getenv("UIPATH_URL")
    if not uipath_url:
        raise ValueError("UIPATH_URL environment variable is required")

    vendor_endpoint = EndpointManager.get_vendor_endpoint()
    formatted_endpoint = vendor_endpoint.format(
        vendor=vendor,
        model=model,
    )
    return f"{uipath_url.rstrip('/')}/{formatted_endpoint}"
