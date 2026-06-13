"""Per-app GUI session discovery: a single-file advertisement per running GUI.

A user-launched GUI may bind its agreed-upon control port, or — when that port is
already taken by an unrelated process — fall back to an OS-assigned ephemeral
port (see :class:`ControlOptions.allow_port_fallback`). Either way the agent needs
to find the *actual* port. This module is that rendezvous: the GUI writes a single
JSON file per app at socket-open and deletes it at close; the agent reads it when
its ``connect`` tool is called without an explicit port.

Design (mirrors the session-registry evaluation):
  - **One file per app** (``<app>.json``), not a multi-entry registry: the user
    runs one instance per app, so there is never more than one live session to
    advertise. Independent files mean no cross-app lock is needed.
  - **Stable fields only**: ``app / port / pid / host / started / wire_version``.
    Volatile state (project, SoC, context) is NOT stored — the agent queries it
    over the live socket after connecting, so this file never needs refresh hooks.
  - **Stale self-healing**: a read verifies the advertised process is alive (pid
    probe) and its socket is reachable (connect probe). A stale file (GUI crashed
    without clearing it, or pid recycled) is treated as "no session" and deleted
    on the spot, so the next read starts clean.

The pid/port probes are re-implemented here (not imported from
``zcu_tools.mcp.core.bridge``) because ``gui.remote`` must not depend on the
``mcp`` layer — the dependency runs the other way. They are small pure functions.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# The four apps that advertise a session. The slug is the discovery key shared by
# the GUI writer and the MCP reader; it is independent of the bridge's internal
# ``app_name`` (measure's is "gui") so both ends agree on a stable identifier.
SessionApp = str


class SessionEntry(TypedDict):
    """The stable, discovery-necessary fields of a running GUI session."""

    app: str
    port: int
    pid: int
    host: str
    started: str
    wire_version: int


def session_dir() -> Path:
    """The per-user directory holding one ``<app>.json`` per running GUI.

    ``~/.cache/zcu-tools/sessions/``. This is a per-machine/per-user runtime fact
    (the GUI and the MCP server run as the same user on the same host), so it
    lives under the user cache, not inside any repo checkout. On Windows
    ``Path.home()/".cache"`` is not the conventional cache location
    (``%LOCALAPPDATA%`` is) — Windows support for this discovery path is
    unverified; the repo's client runs on Linux.
    """
    return Path.home() / ".cache" / "zcu-tools" / "sessions"


def _session_path(app: SessionApp) -> Path:
    return session_dir() / f"{app}.json"


def _port_is_open(host: str, port: int) -> bool:
    """True if a TCP listener accepts a connection at ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` currently exists."""
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(code)
            ):
                return False
            return code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # EPERM: the process exists but we may not signal it — still alive.
        return True
    return True


def write_session(
    app: SessionApp,
    port: int,
    *,
    pid: int,
    host: str,
    wire_version: int,
    started: str,
) -> None:
    """Advertise a running GUI by overwriting ``<app>.json`` (single file).

    Best-effort: a write failure is logged and swallowed, since discovery is an
    optimisation over the agreed-upon-port fallback, not a hard dependency.
    """
    entry: SessionEntry = {
        "app": app,
        "port": port,
        "pid": pid,
        "host": host,
        "started": started,
        "wire_version": wire_version,
    }
    path = _session_path(app)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry), encoding="utf-8")
    except OSError:
        logger.warning("failed to write session discovery file %s", path, exc_info=True)


def clear_session(app: SessionApp) -> None:
    """Remove this app's discovery file (GUI close). Idempotent, best-effort."""
    try:
        _session_path(app).unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "failed to clear session discovery file for %r", app, exc_info=True
        )


def read_session(app: SessionApp) -> SessionEntry | None:
    """Return this app's live session entry, or ``None`` if none is live.

    A missing or malformed file -> ``None`` (no session). A present file whose
    advertised process is dead OR whose socket is unreachable is stale: it is
    deleted and ``None`` returned, so a later read starts clean.
    """
    path = _session_path(app)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("failed to read session discovery file %s", path, exc_info=True)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # A corrupt file is as good as no session; drop it so it stops shadowing
        # the agreed-upon-port fallback.
        logger.warning("corrupt session discovery file %s; removing", path)
        clear_session(app)
        return None

    entry = _coerce_entry(data)
    if entry is None:
        logger.warning("malformed session discovery file %s; removing", path)
        clear_session(app)
        return None

    if not _pid_alive(entry["pid"]) or not _port_is_open(entry["host"], entry["port"]):
        logger.debug("stale session discovery file %s; removing", path)
        clear_session(app)
        return None
    return entry


def _coerce_entry(data: object) -> SessionEntry | None:
    """Validate the parsed JSON has the required stable fields with right types."""
    if not isinstance(data, dict):
        return None
    try:
        return {
            "app": str(data["app"]),
            "port": int(data["port"]),
            "pid": int(data["pid"]),
            "host": str(data["host"]),
            "started": str(data["started"]),
            "wire_version": int(data["wire_version"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "SessionApp",
    "SessionEntry",
    "clear_session",
    "read_session",
    "session_dir",
    "write_session",
]
