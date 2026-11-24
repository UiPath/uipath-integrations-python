"""Fixtures for the Codex durable-tools (design A-prime) contract tests.

These tests run the *real bundled* codex binary from ``openai-codex-cli-bin``
against a local mock Responses API and an in-process MCP host. No network,
no UiPath tenant. Run with ``uv run pytest -m contract tests/contract -rA``.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from .harness import BackgroundLoop, ContractEnv, Evidence, descendants, kill_pids

EXPECTED_CODEX_MINOR = "0.161."


@pytest.fixture(scope="session", autouse=True)
def _bundled_codex_only():
    """Force the pinned bundled runtime: UIPATH_CODEX_BIN must not leak in."""
    saved = os.environ.pop("UIPATH_CODEX_BIN", None)
    from codex_cli_bin import bundled_codex_path  # type: ignore[import-untyped]

    path = bundled_codex_path()
    version = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=30
    ).stdout.strip()
    print(f"\n[contract] bundled codex: {path} -> {version!r}")
    assert EXPECTED_CODEX_MINOR in version, (
        f"expected bundled codex {EXPECTED_CODEX_MINOR}x, got {version!r}"
    )
    yield {"path": str(path), "version": version}
    if saved is not None:
        os.environ["UIPATH_CODEX_BIN"] = saved


@pytest.fixture(scope="session")
def bg_loop():
    loop = BackgroundLoop()
    yield loop
    loop.close()


@pytest.fixture
def cenv(bg_loop, tmp_path):
    before = set(descendants(os.getpid()))
    env = ContractEnv(bg_loop, tmp_path)
    try:
        yield env
    finally:
        env.close()
        # belt and braces: nothing spawned by this test may survive it
        leftovers = [p for p in descendants(os.getpid()) if p not in before]
        kill_pids(leftovers)


@pytest.fixture
def evidence(request):
    ev = Evidence(request.node.name)
    yield ev
    print("\n" + ev.dump())
