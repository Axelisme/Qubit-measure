#!/usr/bin/env python
"""MCP server for the agent coordination taskboard.

A stdio JSON-RPC 2.0 server (launched per ``.mcp.json``).  Unlike the GUI bridges
there is no live process to forward to: tools dispatch *in-process* to a
``TaskboardStore`` that does atomic file I/O over ``task_plans/taskboard.json``.

NOTE: The store uses ``fcntl.flock`` — Linux/macOS only (POSIX advisory lock).

Usage (per ``.mcp.json``):
    uv run --extra client python lib/zcu_tools/mcp/taskboard/server.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

# Launched standalone so add the repo ``lib`` dir to sys.path for absolute imports.
# lib/zcu_tools/mcp/taskboard/server.py -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpServerConfig,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)
from zcu_tools.mcp.taskboard.method_specs import METHOD_SPECS  # noqa: E402
from zcu_tools.mcp.taskboard.store import TaskboardStore  # noqa: E402

logger = logging.getLogger("zcu_tools.mcp.taskboard.server")

_SERVER_INSTRUCTIONS = """\
Multi-agent path-coordination taskboard (ADR-0022).

Use these tools to coordinate parallel agents working on the same checkout so
they do not clobber each other's in-progress edits.

Workflow:
  1. taskboard_check(paths, mode='write') — dry-run: see if any granted claim
     overlaps.  Zero side effects.
  2. taskboard_claim(owner, paths, task, mode='write') — reserve your scope.
     - If status='granted': you hold the lock, proceed with your edits.
     - If status='pending': you are queued behind conflicting claims.
       Option A (short wait, <=30s): taskboard_wait(claim_id, timeout_s=...).
       Option B (longer wait): ScheduleWakeup and poll taskboard_list until
       your claim_id appears in the 'active' list.
     - Same-session overlaps are granted and returned under warnings; only a
       different top-level session blocks.
  3. Do your edits / commits.
  4. taskboard_release(claim_id) — ONLY after your changes are committed to git.
     This auto-promotes the next pending claim(s).

Heartbeat: for long operations call taskboard_touch(claim_id) periodically
(TTL default: 2 hours) to prevent stale-claim reclaim.

Path syntax:
  - Repo-relative file or directory: 'lib/zcu_tools/gui/', 'README.md'
  - Glob pattern: 'lib/zcu_tools/gui/**/*.py'
  - @-prefixed resource token (non-file singletons):
    '@hw/zcu216', '@gui/measure', '@port/8767', '@gui/fluxdep'

Read-only work (no writes, no ambiguous scope) does not require a claim.
"""

_SESSION_ID_ENV_VARS = (
    "CLAUDE_CODE_SESSION_ID",
    "CODEX_THREAD_ID",
    "AGENT_SESSION_ID",
)
_SESSION_ID_ENV_VAR_BYTES = tuple(name.encode() for name in _SESSION_ID_ENV_VARS)
_PROC_ANCESTOR_LIMIT = 16
_PROCESS_SESSION_ID = f"taskboard-process-{uuid4().hex}"

_CONFIG = McpServerConfig(
    tool_prefix="taskboard_",
    server_display_name="taskboard",
    server_instructions=_SERVER_INSTRUCTIONS,
)


def _session_identity() -> str | None:
    """The conflict identity for claims/checks from this server process.

    Agent hosts inject a stable session/thread id into stdio MCP subprocesses
    (``CLAUDE_CODE_SESSION_ID`` for Claude Code, ``CODEX_THREAD_ID`` for Codex).
    It is the same value for a top-level session and all its sub-agents, and
    differs across top-level sessions.  Reading it from the process env (MCP tool
    calls carry no per-request session context) is what lets an orchestrator and
    its sub-agents share one coordination identity (ADR-0022).

    Some hosts expose the id to the top-level agent process but do not forward it
    into MCP subprocesses.  On Linux, fall back to reading only the allowlisted
    session-id names from ancestor ``/proc/*/environ`` entries.  As a final
    fallback, use a process-local identity: stdio MCP servers are scoped to one
    client session, so calls handled by the same taskboard process should not
    self-block even when the host exposes no session env at all.
    """
    return _session_identity_with_source()[0]


def _session_identity_with_source() -> tuple[str, str]:
    found = _find_session_identity_in_env(os.environ)
    if found is not None:
        value, name = found
        return value, f"env:{name}"

    found = _find_session_identity_in_ancestor_env()
    if found is not None:
        value, source = found
        return value, source

    return _PROCESS_SESSION_ID, "process-local"


def _session_identity_from_env(env: Mapping[str, str]) -> str | None:
    found = _find_session_identity_in_env(env)
    return found[0] if found is not None else None


def _find_session_identity_in_env(env: Mapping[str, str]) -> tuple[str, str] | None:
    for name in _SESSION_ID_ENV_VARS:
        value = env.get(name)
        if value:
            return value, name
    return None


def _session_identity_from_ancestor_env(
    *,
    start_pid: int | None = None,
    proc_root: Path = Path("/proc"),
    max_depth: int = _PROC_ANCESTOR_LIMIT,
) -> str | None:
    found = _find_session_identity_in_ancestor_env(
        start_pid=start_pid,
        proc_root=proc_root,
        max_depth=max_depth,
    )
    return found[0] if found is not None else None


def _find_session_identity_in_ancestor_env(
    *,
    start_pid: int | None = None,
    proc_root: Path = Path("/proc"),
    max_depth: int = _PROC_ANCESTOR_LIMIT,
) -> tuple[str, str] | None:
    """Best-effort Linux fallback for hosts that do not pass MCP env vars.

    Only the allowlisted session-id variable names are inspected.  Any failure
    (non-Linux, hidden ``/proc``, permissions, malformed stat data) degrades to
    ``None`` so claim semantics fall back to per-owner coordination.
    """
    pid = start_pid if start_pid is not None else os.getppid()
    visited: set[int] = set()

    for _ in range(max_depth):
        if pid <= 0 or pid in visited:
            return None
        visited.add(pid)
        proc_dir = proc_root / str(pid)

        try:
            found = _find_session_identity_in_environ_bytes(
                (proc_dir / "environ").read_bytes()
            )
        except OSError:
            found = None
        if found is not None:
            value, name = found
            return value, f"ancestor-env:pid={pid}:{name}"

        try:
            next_pid = _parent_pid_from_proc_stat((proc_dir / "stat").read_text())
        except (OSError, ValueError):
            return None
        if next_pid == pid:
            return None
        pid = next_pid

    return None


def _session_identity_from_environ_bytes(raw: bytes) -> str | None:
    found = _find_session_identity_in_environ_bytes(raw)
    return found[0] if found is not None else None


def _find_session_identity_in_environ_bytes(raw: bytes) -> tuple[str, str] | None:
    values: dict[bytes, bytes] = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep and key in _SESSION_ID_ENV_VAR_BYTES and value:
            values[key] = value

    for name in _SESSION_ID_ENV_VAR_BYTES:
        value = values.get(name)
        if value:
            return value.decode(errors="replace"), name.decode()
    return None


def _parent_pid_from_proc_stat(stat_text: str) -> int:
    """Parse PPID from Linux ``/proc/<pid>/stat``.

    The comm field is wrapped in parentheses and may contain spaces, so split
    after the final ``") "`` before reading the state/ppid fields.
    """
    try:
        rest = stat_text.rsplit(") ", 1)[1]
        fields = rest.split()
        return int(fields[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("malformed proc stat") from exc


def build_dispatch(
    store: TaskboardStore,
) -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Map each wire method to a ``TaskboardStore`` call. Keys must match METHOD_SPECS.

    ``claim`` and ``check`` are augmented with a server-derived ``identity``: the
    agent session id when present, else (for claim only — check carries no owner)
    the caller's ``owner``.  Identity is never a wire parameter; callers do not
    pass it.
    """

    def _claim(p: dict[str, Any]) -> Any:
        identity, source = _session_identity_with_source()
        logger.debug(
            "claim owner=%r paths=%r mode=%r identity=%r source=%s",
            p.get("owner"),
            p.get("paths"),
            p.get("mode", "write"),
            identity,
            source,
        )
        return store.claim(**p, identity=identity)

    def _check(p: dict[str, Any]) -> Any:
        # check has no owner argument; identity is the session id when available,
        # else None (store reports all overlaps for an anonymous dry run).
        identity, source = _session_identity_with_source()
        logger.debug(
            "check paths=%r mode=%r identity=%r source=%s",
            p.get("paths"),
            p.get("mode", "write"),
            identity,
            source,
        )
        return store.check(**p, identity=identity)

    return {
        "claim": _claim,
        "release": lambda p: store.release(**p),
        "check": _check,
        "list": lambda p: store.list_claims(**p),
        "wait": lambda p: store.wait(**p),
        "touch": lambda p: store.touch(**p),
        "force_release": lambda p: store.force_release(**p),
    }


def build_tools(store: TaskboardStore) -> dict[str, dict[str, Any]]:
    """Generate the MCP tool table, dispatching in-process to ``store``."""
    dispatch = build_dispatch(store)

    def local_send(
        method: str, params: dict[str, Any], timeout_seconds: float = 30.0
    ) -> Any:
        del timeout_seconds  # no transport; calls are local + synchronous
        handler = dispatch.get(method)
        if handler is None:
            raise RuntimeError(f"unknown taskboard method {method!r}")
        return handler(params)

    return assemble_tools(
        generate_tools(_CONFIG, METHOD_SPECS, frozenset(), local_send), {}, frozenset()
    )


def _taskboard_root() -> Path:
    """Where the JSON store lives: ``$ZCU_TASKBOARD_DIR`` or ``<repo>/task_plans``."""
    env = os.environ.get("ZCU_TASKBOARD_DIR")
    if env:
        return Path(env)
    # lib/zcu_tools/mcp/taskboard/server.py -> repo root
    return Path(__file__).resolve().parents[4] / "task_plans"


def main() -> None:
    def _setup_logging() -> None:
        from zcu_tools.gui.logging_setup import setup_gui_logging

        setup_gui_logging(
            app_name="taskboard",
            log_root=Path(__file__).resolve().parents[4],
            group="mcp",
            extra_namespaces=("zcu_tools.mcp.taskboard",),
        )
        logger.debug(
            "taskboard MCP startup pid=%s ppid=%s cwd=%s argv=%r process_identity=%s",
            os.getpid(),
            os.getppid(),
            os.getcwd(),
            sys.argv,
            _PROCESS_SESSION_ID,
        )

    root = _taskboard_root()
    store = TaskboardStore(
        json_path=root / "taskboard.json",
        md_path=root / "taskboard.md",
    )
    run_stdio_loop(_CONFIG, build_tools(store), on_start=_setup_logging)


if __name__ == "__main__":
    main()
