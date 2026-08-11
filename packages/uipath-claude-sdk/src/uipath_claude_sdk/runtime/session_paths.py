"""Filesystem locations that let a Claude CLI session survive a suspension.

The Claude CLI keeps its own session transcript at
``$CLAUDE_CONFIG_DIR/projects/<encoded cwd>/<session id>.jsonl``. That middle
component is the working directory's whole absolute path with every character
outside ``[A-Za-z0-9]`` replaced by a dash, so the CLI finds a session again
only when the working directory string is byte for byte what it was.

That string used to have to be stable, and getting it wrong cost a deployed
job: it suspended under ``/tmp/home/.job-data/42d7392e-.../__uipath`` and
resumed under ``/tmp/home/.job-data/bad7abe7-.../__uipath``, because the
platform carries the state database across and not the directory holding it.
The working directory encoded to a different name and the CLI silently started
a fresh session.

It no longer has to be stable. The transcript is restored under the name
:func:`encode_project_dir` derives from whatever working directory this process
is using, so a run whose working directory moved still finds its session.

That frees the working directory to be useful rather than merely constant, so
it sits at ``<project>/.runs/<runtime id>``: beneath the directory the agent
was loaded from, which is where its packaged files are. The CLI searches
upwards for ``.claude``, so skills and project settings that shipped with the
agent are found with nothing copied, while each run still writes into its own
directory rather than into the package.

A transcript must never exist in two project directories at once: the CLI then
reports the session as not found. Restore it under one name rather than copying
it around.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_RUNTIME_DIR = "__uipath"
_CONFIG_DIR_NAME = "claude_home"
_PROJECTS_DIR_NAME = "projects"
_RUNS_DIR_NAME = ".runs"
_FALLBACK_COMPONENT = "default"
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")

CLAUDE_CONFIG_DIR_ENV_VAR = "CLAUDE_CONFIG_DIR"


@dataclass(frozen=True)
class ClaudeSessionPaths:
    """Directories for one Claude SDK session.

    Attributes:
        config_dir: Value for ``CLAUDE_CONFIG_DIR``. Holds the CLI's own
            session transcripts. Its own path may move between processes: only
            the ``projects`` subdirectory name has to be reproduced, and that
            is derived from the workspace rather than from here.
        workspace: Working directory handed to the SDK subprocess as ``cwd``.
            The one path that must be identical across a suspension.
    """

    config_dir: Path
    workspace: Path

    @classmethod
    def for_runtime(
        cls,
        runtime_dir: str | Path | None,
        runtime_id: str,
        project_dir: str | Path | None = None,
    ) -> ClaudeSessionPaths:
        """Derive the session paths for a runtime instance.

        Args:
            runtime_dir: The persisted runtime directory. Falls back to
                ``__uipath`` when unset, matching the storage path precedence.
            runtime_id: Identifier the workspace is keyed on. The job key on
                the platform, and the conversation id for conversational runs
                so every exchange of one conversation shares a workspace.
            project_dir: Directory the agent was loaded from, which is also
                where its packaged files live. The working directory is placed
                beneath it so the CLI, which searches upwards for ``.claude``,
                finds the skills and settings that shipped with the agent
                without anything being copied. Falls back to the runtime
                directory when the caller cannot say.

        Returns:
            Absolute paths, so they stay valid inside the SDK subprocess whose
            own working directory is the workspace.
        """
        base = Path(os.path.abspath(runtime_dir or _DEFAULT_RUNTIME_DIR))
        root = Path(os.path.abspath(project_dir)) if project_dir else base
        return cls(
            config_dir=base / _CONFIG_DIR_NAME,
            workspace=(root / _RUNS_DIR_NAME / sanitize_path_component(runtime_id)),
        )

    @property
    def projects_dir(self) -> Path:
        """Where the CLI keeps one directory per working directory."""
        return self.config_dir / _PROJECTS_DIR_NAME

    @property
    def project_dir_name(self) -> str:
        """The directory name the CLI files this workspace's sessions under."""
        return encode_project_dir(self.workspace)

    def find_transcript(self, session_id: str) -> Path | None:
        """Locate a session's transcript, wherever the CLI filed it."""
        if not self.projects_dir.is_dir():
            return None
        matches = sorted(self.projects_dir.glob(f"*/{session_id}.jsonl"))
        if not matches:
            return None
        if len(matches) > 1:
            found = ", ".join(str(m.parent.name) for m in matches)
            raise ValueError(
                f"Session {session_id} has a transcript in more than one project "
                f"directory ({found}), which the CLI reports as a missing session."
            )
        return matches[0]

    def transcript_path(self, session_id: str) -> Path:
        """Where this workspace's transcript for a session belongs."""
        return self.projects_dir / self.project_dir_name / f"{session_id}.jsonl"

    def ensure(self) -> None:
        """Create both directories. Idempotent, safe to call on every run."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def env_overrides(self) -> dict[str, str]:
        """Environment variables pointing the CLI at the config directory."""
        return {CLAUDE_CONFIG_DIR_ENV_VAR: str(self.config_dir)}


def encode_project_dir(workspace: Path) -> str:
    """Name the CLI files a working directory's sessions under.

    Every character outside ``[A-Za-z0-9]`` becomes a dash, so
    ``/tmp/uipath-claude-workspaces/job-1`` becomes
    ``-tmp-uipath-claude-workspaces-job-1``. Verified against a directory the
    CLI wrote, and pinned by the defer contract, which resumes a real session
    under a *different* working directory: get this wrong and the CLI reports
    the session as missing rather than failing quietly here.

    Deriving the name means the working directory is free to move between
    processes, which is what lets the agent run in a managed workspace.
    """
    return _NON_ALNUM.sub("-", str(workspace))


def sanitize_path_component(value: str) -> str:
    """Reduce an identifier to a safe single path component."""
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return sanitized or _FALLBACK_COMPONENT


__all__ = [
    "CLAUDE_CONFIG_DIR_ENV_VAR",
    "ClaudeSessionPaths",
    "encode_project_dir",
    "sanitize_path_component",
]
