"""agent_session_registry — file-based registry for independent agent sessions.

Each live or recently-stopped session is represented by one JSON file under
``~/.cache/zcu-tools/agent_sessions/<session_id>.json``.  The registry is
stateless: callers write, read, and remove individual records via pure functions.

Design principles (mirrors session_discovery.py; ADR-0024 B1b):
  - One file per session (not a monolithic registry): concurrent GUI writers are
    independent, no cross-session locking needed.
  - Atomic writes via tmp+os.replace — POSIX-atomic; best-effort on Windows
    (Windows-verify: os.replace on Windows fails if the target is open, but this
    is unlikely for short-lived registry files).
  - No flock / fifo (both Unix-only); the only IPC is file I/O.
  - Stale-running self-healing: ``read_record`` detects a ``status=running``
    record whose PID is dead and transitions it to ``status=stopped`` in-place
    (decision D from the B1b-2 spec).

Session IDs
-----------
  ``session_id``      : short 8-hex UUID, owned by this module (registry key).
  ``claude_session_id``: the ``session_id`` from the ``system/init`` stream-json
                         frame; empty string until populated by the supervisor
                         (B1b-4 ``--resume``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class AgentSessionRecord(TypedDict):
    """One registry entry for an independent agent session.

    Fields
    ------
    session_id       : Registry-local 8-hex short identifier (not claude's).
    claude_session_id: The ``session_id`` from the stream-json ``system/init``
                       frame.  Empty string until the supervisor writes it back
                       (B1b-4 ``--resume``).
    pid              : Supervisor process PID.
    status           : ``"running"`` or ``"stopped"``.
    log_path         : Absolute path to ``log.ndjson`` for this session.
    spool_dir        : Absolute path to the spool directory for this session.
    created          : ISO-8601 UTC string (``datetime.utcnow().isoformat()``).
    title            : First N chars of the initial task string (display only).
    """

    session_id: str
    claude_session_id: str
    pid: int
    status: str  # "running" | "stopped"
    log_path: str
    spool_dir: str
    created: str
    title: str


# ---------------------------------------------------------------------------
# Registry directory
# ---------------------------------------------------------------------------


def registry_dir() -> Path:
    """Per-user directory holding one ``<session_id>.json`` per agent session.

    ``~/.cache/zcu-tools/agent_sessions/``.  On Windows ``Path.home()/.cache``
    is not conventional (``%LOCALAPPDATA%`` would be), but we mirror the
    existing ``session_discovery.py`` convention for consistency.
    Windows-verify: path resolution untested on Windows.
    """
    return Path.home() / ".cache" / "zcu-tools" / "agent_sessions"


def _record_path(session_id: str) -> Path:
    return registry_dir() / f"{session_id}.json"


# ---------------------------------------------------------------------------
# PID liveness probe (duplicated here so services/ has no dep on gui/remote)
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` refers to a live process.

    Duplicated from ``independent_agent_session._supervisor_alive`` intentionally:
    the registry must not depend on ``independent_agent_session`` (circular would
    form if that module ever imports registry helpers).  Pure function — no Qt.

    POSIX:   ``os.kill(pid, 0)`` — existence probe, no actual signal.
    Windows: ctypes OpenProcess + GetExitCodeProcess.
    Windows-verify: the Windows branch is written but untested on Linux.
    """
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    return _pid_alive_posix(pid)


def _pid_alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — treat as alive.
        return True
    except OSError:
        return False


def _pid_alive_windows(pid: int) -> bool:  # pragma: no cover  # Windows-verify
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return False
    exit_code = ctypes.c_ulong(0)
    ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
        handle, ctypes.byref(exit_code)
    )
    ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    STILL_ACTIVE = 259
    return exit_code.value == STILL_ACTIVE


# ---------------------------------------------------------------------------
# Short session_id generation
# ---------------------------------------------------------------------------


def new_session_id() -> str:
    """Generate a new 8-hex registry session ID (distinct from claude_session_id)."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_record(record: AgentSessionRecord) -> None:
    """Write (or overwrite) a session record atomically.

    Uses tmp-file + ``os.replace`` for POSIX atomicity.  The registry directory
    is created on first write.
    """
    directory = registry_dir()
    directory.mkdir(parents=True, exist_ok=True)

    target = _record_path(record["session_id"])
    tmp = target.with_suffix(".json.tmp")
    raw = json.dumps(record, indent=2, ensure_ascii=False)
    tmp.write_text(raw, encoding="utf-8")
    # os.replace is atomic on POSIX; on Windows it may fail if target is held
    # open, but agent session files are only opened for short reads — acceptable.
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_record(session_id: str) -> AgentSessionRecord | None:
    """Read one session record by session_id.

    Returns ``None`` if the file does not exist.

    Stale-running self-heal (decision D): if the stored ``status`` is ``"running"``
    but the PID is dead, the record is updated to ``"stopped"`` and re-written
    before returning.  This keeps the history (log_path / title / created) intact
    while reflecting reality to callers.
    """
    path = _record_path(session_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(
            "agent_session_registry: read_record %s failed: %s", session_id, exc
        )
        return None

    try:
        obj: AgentSessionRecord = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "agent_session_registry: record %s is malformed JSON: %s", session_id, exc
        )
        return None

    # Self-heal: running + dead pid → stopped.
    if obj.get("status") == "running" and not _pid_alive(obj.get("pid", 0)):
        logger.info(
            "agent_session_registry: session %s pid=%s dead — marking stopped",
            session_id,
            obj.get("pid"),
        )
        obj["status"] = "stopped"
        try:
            write_record(obj)
        except OSError as exc:
            logger.warning(
                "agent_session_registry: could not persist self-heal for %s: %s",
                session_id,
                exc,
            )

    return obj


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_records() -> list[AgentSessionRecord]:
    """Return all session records sorted by ``created`` (oldest first).

    Applies the same stale-running self-heal as ``read_record`` for each entry.
    Files that cannot be parsed are silently skipped.
    """
    directory = registry_dir()
    if not directory.exists():
        return []

    records: list[AgentSessionRecord] = []
    try:
        for path in directory.iterdir():
            if path.suffix != ".json":
                continue
            session_id = path.stem
            rec = read_record(session_id)
            if rec is not None:
                records.append(rec)
    except OSError as exc:
        logger.warning("agent_session_registry: list_records failed: %s", exc)

    records.sort(key=lambda r: r.get("created", ""))
    return records


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def remove_record(session_id: str) -> None:
    """Delete the session record file.  Idempotent — no error if already gone."""
    path = _record_path(session_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "agent_session_registry: remove_record %s failed: %s", session_id, exc
        )
