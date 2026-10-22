"""Glue for the durable-tools contract tests.

* ``BackgroundLoop`` runs the aiohttp servers (mock Responses API + durable
  MCP host) on an asyncio loop in a daemon thread, so tests can drive the
  synchronous ``openai_codex.Codex`` client from the main thread.
* ``CodexApp`` launches the *bundled* codex app-server (never a
  ``UIPATH_CODEX_BIN`` override) with a per-test ``CODEX_HOME`` and the
  provider / MCP config overrides of design A-prime.
* rollout helpers locate/parse ``CODEX_HOME/sessions/**/rollout-*.jsonl``.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

from .durable_mcp_host import SERVER_NAME

T = TypeVar("T")
JsonObj = dict[str, Any]

MOCK_PROVIDER = "mock"
MOCK_MODEL = "mock-model"
MOCK_KEY_ENV = "MOCK_RESPONSES_KEY"
TOOLS_TOKEN_ENV = "UIPATH_TOOLS_TOKEN"


# --------------------------------------------------------------------------
# background asyncio loop
# --------------------------------------------------------------------------
class BackgroundLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="contract-loop"
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Awaitable[T], timeout: float = 30.0) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)  # type: ignore[arg-type]

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def descendants(pid: int) -> list[int]:
    """All descendant pids of ``pid`` (Linux /proc walk)."""
    children: dict[int, list[int]] = {}
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(stat) as f:
                data = f.read()
            # pid (comm) state ppid ...
            rest = data.rsplit(")", 1)[1].split()
            ppid = int(rest[1])
            children.setdefault(ppid, []).append(int(stat.split("/")[2]))
        except Exception:
            continue
    out: list[int] = []
    stack = [pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            out.append(c)
            stack.append(c)
    return out


def pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def kill_pids(pids: list[int]) -> None:
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


# --------------------------------------------------------------------------
# Codex app-server wrapper
# --------------------------------------------------------------------------
def build_overrides(
    mock_base_url: str,
    mcp_url: str | None,
    *,
    parallel: bool = True,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    ov = [
        f'model_providers.{MOCK_PROVIDER}.name="Mock"',
        f'model_providers.{MOCK_PROVIDER}.base_url="{mock_base_url}"',
        f'model_providers.{MOCK_PROVIDER}.wire_api="responses"',
        f'model_providers.{MOCK_PROVIDER}.env_key="{MOCK_KEY_ENV}"',
        f"model_providers.{MOCK_PROVIDER}.request_max_retries=0",
        f"model_providers.{MOCK_PROVIDER}.stream_max_retries=0",
        # keep the run hermetic: no remote plugin / chatgpt backend traffic
        "features.plugins=false",
        'chatgpt_base_url="http://127.0.0.1:9/backend-api"',
    ]
    if mcp_url is not None:
        s = f"mcp_servers.{SERVER_NAME}"
        ov += [
            f'{s}.url="{mcp_url}"',
            f'{s}.bearer_token_env_var="{TOOLS_TOKEN_ENV}"',
            f'{s}.default_tools_approval_mode="approve"',
            f"{s}.tool_timeout_sec=3600",
            f"{s}.startup_timeout_sec=30",
            f"{s}.supports_parallel_tool_calls={'true' if parallel else 'false'}",
        ]
    ov += list(extra)
    return tuple(ov)


class CodexApp:
    """One codex app-server process (bundled binary) bound to a CODEX_HOME."""

    def __init__(
        self, codex_home: Path, cwd: Path, overrides: tuple[str, ...], token: str
    ) -> None:
        self.codex_home = codex_home
        self.cwd = cwd
        env = {
            "CODEX_HOME": str(codex_home),
            MOCK_KEY_ENV: "mock-secret",
            TOOLS_TOKEN_ENV: token,
            "RUST_LOG": os.environ.get("CONTRACT_RUST_LOG", "warn"),
            # Hermetic: any non-loopback HTTP(S) goes to a dead proxy.
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
            "https_proxy": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "all_proxy": "http://127.0.0.1:9",
            "no_proxy": "127.0.0.1,localhost",
        }
        cfg = CodexConfig(config_overrides=overrides, cwd=str(cwd), env=env)
        assert cfg.codex_bin is None  # always the bundled runtime
        self.codex = Codex(cfg)
        self.metadata = self.codex.metadata
        proc = self.codex._client._proc
        assert proc is not None
        self.pid = proc.pid
        self._known_children: list[int] = []
        self.killed = False

    # thread helpers ------------------------------------------------------
    def thread_start(self, cwd: Path | None = None):
        return self.codex.thread_start(
            model=MOCK_MODEL,
            model_provider=MOCK_PROVIDER,
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.deny_all,
            cwd=str(cwd or self.cwd),
        )

    def thread_resume(self, thread_id: str, cwd: Path | None = None):
        return self.codex.thread_resume(
            thread_id,
            model=MOCK_MODEL,
            model_provider=MOCK_PROVIDER,
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.deny_all,
            cwd=str(cwd) if cwd else None,
        )

    def raw(self, method: str, params: JsonObj) -> Any:
        return self.codex._client._request_raw(method, params)

    def stderr_tail(self, n: int = 60) -> str:
        return self.codex._client._stderr_tail(n)

    # termination ---------------------------------------------------------
    def sigkill(self) -> list[int]:
        """Design A-prime suspend: SIGKILL the app-server process itself."""
        kids = descendants(self.pid)
        self._known_children = kids
        proc = self.codex._client._proc
        assert proc is not None
        proc.kill()  # == codex._client._proc.kill(); SIGKILL, no graceful shutdown
        proc.wait(timeout=10)
        self.killed = True
        return kids

    def cleanup(self) -> None:
        """Always-safe teardown: kill process tree, close client threads."""
        try:
            kids = set(self._known_children) | set(descendants(self.pid))
        except Exception:
            kids = set(self._known_children)
        proc = self.codex._client._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
        kill_pids(sorted(kids))
        try:
            self.codex.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# rollout helpers
# --------------------------------------------------------------------------
def find_rollouts(codex_home: Path) -> list[Path]:
    return sorted(
        Path(p)
        for p in glob.glob(
            str(codex_home / "sessions" / "**" / "rollout-*.jsonl"), recursive=True
        )
    )


def rollout_for_thread(codex_home: Path, thread_id: str) -> Path | None:
    for p in find_rollouts(codex_home):
        if thread_id in p.name:
            return p
    return None


def read_rollout(path: Path) -> list[JsonObj]:
    out: list[JsonObj] = []
    try:
        text = path.read_text()
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"_unparsable": line[:200]})
    return out


def response_items(lines: list[JsonObj]) -> list[JsonObj]:
    return [
        ln.get("payload") or {} for ln in lines if ln.get("type") == "response_item"
    ]


def rollout_function_calls(
    lines: list[JsonObj], call_id: str | None = None
) -> list[JsonObj]:
    return [
        p
        for p in response_items(lines)
        if p.get("type") == "function_call"
        and (call_id is None or p.get("call_id") == call_id)
    ]


def rollout_function_outputs(
    lines: list[JsonObj], call_id: str | None = None
) -> list[JsonObj]:
    return [
        p
        for p in response_items(lines)
        if p.get("type")
        in ("function_call_output", "custom_tool_call_output", "mcp_tool_call_output")
        and (call_id is None or p.get("call_id") == call_id)
    ]


def turn_aborted_lines(lines: list[JsonObj]) -> list[JsonObj]:
    return [ln for ln in lines if "turn_aborted" in json.dumps(ln)]


def compact_line(ln: JsonObj, limit: int = 300) -> str:
    s = json.dumps({"type": ln.get("type"), "payload": ln.get("payload")})
    return s if len(s) <= limit else s[:limit] + "..."


def restore_rollout(src_home: Path, rollout: Path, dst_home: Path) -> Path:
    """Copy only the rollout jsonl into a fresh CODEX_HOME at the same relative path."""
    rel = rollout.relative_to(src_home)
    dst = dst_home / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rollout, dst)
    return dst


def wait_until(
    pred: Callable[[], bool], timeout: float = 20.0, interval: float = 0.05
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def dir_listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# --------------------------------------------------------------------------
# per-test environment + scripted durable scenario
# --------------------------------------------------------------------------
TRIGGER = "PLEASE_CALL_DURABLE"


class DurableScript:
    """Mock-model policy.

    * a user turn containing TRIGGER, whose input has not yet seen our calls
      -> one response with one function_call per id in ``call_ids``
      (using the exact name/namespace Codex advertised for the MCP tool);
    * anything else -> a final assistant message.
    """

    def __init__(self, call_ids: list[str], tool_needle: str = "start_job") -> None:
        self.call_ids = call_ids
        self.tool_needle = tool_needle
        self.advertised: tuple[str | None, str | None] | None = None

    def __call__(self, idx: int, body: JsonObj) -> list[JsonObj] | None:
        from .mock_responses import (
            advertised_tools,
            find_tool,
            function_call_item,
            input_items,
            last_user_text,
        )

        if TRIGGER not in last_user_text(body):
            return None
        seen = {
            it.get("call_id")
            for it in input_items(body)
            if it.get("type") == "function_call"
        }
        if seen & set(self.call_ids):
            return None
        name, ns = find_tool(advertised_tools(body), self.tool_needle)
        self.advertised = (name, ns)
        if not name:
            return None  # tool not advertised -> final message; tests will fail with evidence
        return [
            function_call_item(cid, name, {"job": f"job-{i}"}, ns)
            for i, cid in enumerate(self.call_ids)
        ]


class ContractEnv:
    """Owns servers, codex processes and temp dirs for one test; always cleans up."""

    def __init__(self, loop: BackgroundLoop, root: Path) -> None:
        from .mock_responses import MockResponsesServer

        self.loop = loop
        self.root = root
        self.script = DurableScript([])
        self.mock = MockResponsesServer(self.script)
        loop.run(self.mock.start())
        self.hosts: list[Any] = []
        self.apps: list[CodexApp] = []

    def dir(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def new_host(self, token: str, on_call: Callable[[JsonObj], Any] | None = None):
        from .durable_mcp_host import DurableMcpHost

        host = DurableMcpHost(token, on_call=on_call)
        self.loop.run(host.start())
        self.hosts.append(host)
        return host

    def new_app(
        self, home: Path, cwd: Path, host: Any, *, parallel: bool = True
    ) -> CodexApp:
        app = CodexApp(
            home,
            cwd,
            build_overrides(self.mock.base_url, host.url, parallel=parallel),
            host.token,
        )
        self.apps.append(app)
        return app

    def close(self) -> None:
        for app in self.apps:
            app.cleanup()
        for host in self.hosts:
            try:
                self.loop.run(host.stop(), timeout=10)
            except Exception:
                pass
        try:
            self.loop.run(self.mock.stop(), timeout=10)
        except Exception:
            pass


class Evidence:
    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: list[tuple[str, str]] = []

    def add(self, label: str, value: Any) -> None:
        if not isinstance(value, str):
            try:
                value = json.dumps(value, indent=1, default=str)
            except Exception:
                value = repr(value)
        self.entries.append((label, value))

    def dump(self, limit: int = 20000) -> str:
        out = [f"===== EVIDENCE [{self.name}] ====="]
        for label, value in self.entries:
            out.append(f"--- {label}:\n{value}")
        s = "\n".join(out)
        return s if len(s) <= limit else s[:limit] + "\n...<truncated>"
