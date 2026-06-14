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

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

_CONFIG = McpServerConfig(
    tool_prefix="taskboard_",
    server_display_name="taskboard",
    server_instructions=_SERVER_INSTRUCTIONS,
)


def _session_identity() -> str | None:
    """The conflict identity for claims/checks from this server process.

    Claude Code injects ``CLAUDE_CODE_SESSION_ID`` into every stdio MCP subprocess;
    it is the *same* value for a top-level session and all its sub-agents, and
    differs across top-level sessions.  Reading it from the process env (MCP tool
    calls carry no per-request session context) is what lets an orchestrator and
    its sub-agents share one coordination identity (ADR-0022).  Returns ``None``
    when unset, so the store falls back to the per-call ``owner`` label.
    """
    return os.environ.get("CLAUDE_CODE_SESSION_ID")


def build_dispatch(
    store: TaskboardStore,
) -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Map each wire method to a ``TaskboardStore`` call. Keys must match METHOD_SPECS.

    ``claim`` and ``check`` are augmented with a server-derived ``identity``: the CC
    session id when present, else (for claim only — check carries no owner) the
    caller's ``owner``.  Identity is never a wire parameter; callers do not pass it.
    """
    session_id = _session_identity()

    def _claim(p: dict[str, Any]) -> Any:
        identity = session_id if session_id is not None else p["owner"]
        return store.claim(**p, identity=identity)

    def _check(p: dict[str, Any]) -> Any:
        # check has no owner argument; identity is the session id when available,
        # else None (store reports all overlaps for an anonymous dry run).
        return store.check(**p, identity=session_id)

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
    root = _taskboard_root()
    store = TaskboardStore(
        json_path=root / "taskboard.json",
        md_path=root / "taskboard.md",
    )
    run_stdio_loop(_CONFIG, build_tools(store))


if __name__ == "__main__":
    main()
