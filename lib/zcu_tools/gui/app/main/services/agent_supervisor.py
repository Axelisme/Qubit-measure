"""agent_supervisor — detached Python wrapper that drives a ``claude`` subprocess.

IPC protocol (portable file-based, no fifo / flock)
----------------------------------------------------
session_dir/
    log.ndjson      — supervisor appends one JSON line per stdout line from claude;
                      viewers poll-tail this file using a byte-offset cursor.
    spool/          — command spool directory; each file is one user message:
                        <timestamp>_<rand>.json  →  {"type":"user","message":{...}}
                      supervisor polls the directory, consumes files in name-sorted
                      order (read → write to claude stdin → delete), then loops.

Log line format: the raw stream-json line emitted by claude, verbatim.
Spool file format: a JSON object containing the raw stdin envelope produced by
  ``build_stdin_message``.  The spool consumer verifies the file is complete
  (valid JSON) before feeding it, so a partially-written file is skipped and
  retried on the next poll.

Detached spawn (cross-platform)
---------------------------------
POSIX : ``Popen(..., start_new_session=True)``
Windows: ``Popen(..., creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)``
         Windows-verify — logic written but not runnable on Linux.

Stop (cross-platform)
----------------------
POSIX : ``os.kill(pid, signal.SIGINT)``
Windows: ``os.kill(pid, signal.CTRL_BREAK_EVENT)`` with fallback to
          ``subprocess.run(['taskkill', '/F', '/PID', str(pid)])``
         Windows-verify — logic written but not runnable on Linux.

Standalone usage
-----------------
  python -m zcu_tools.gui.app.main.services.agent_supervisor \\
      --session-dir /tmp/zcu_agent/my_session \\
      --task "run onetone" \\
      --repo-root /path/to/repo

The supervisor exits when the claude subprocess exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Import pure helpers from agent_runner; no Qt dependency here.
from zcu_tools.gui.app.main.services.agent_runner import (
    build_claude_argv,
    build_loopback_mcp_config,
    build_stdin_message,
)

# Registry helpers — imported lazily in run_supervisor_loop to keep the
# supervisor importable even when called without a session_id (B1b-1 compat).
from zcu_tools.gui.app.main.services.agent_session_registry import (
    AgentSessionRecord,
    write_record,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FILENAME = "log.ndjson"
_SPOOL_DIRNAME = "spool"

# Supervisor poll interval for the spool directory (seconds).
_SPOOL_POLL_INTERVAL = 0.10

# ---------------------------------------------------------------------------
# SupervisorHandle — returned to the GUI layer after detached spawn
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorHandle:
    """Opaque reference to a detached supervisor process.

    ``pid`` is the supervisor *Python* process PID (not the claude PID).
    The GUI-side ``IndependentAgentSession`` uses this to send stop signals.
    """

    pid: int
    log_path: Path
    spool_dir: Path


# ---------------------------------------------------------------------------
# Portable spool helpers (pure, unit-testable without spawning)
# ---------------------------------------------------------------------------


def _spool_filename(ts: float) -> str:
    """Generate a sortable unique filename for a spool entry.

    Format: ``<13-digit-ms-timestamp>_<8-rand-alnum>.json``
    Sorting by name gives FIFO order across concurrent writers.
    """
    ms = int(ts * 1000)
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{ms:013d}_{rand}.json"


def write_spool_message(spool_dir: Path, text: str) -> Path:
    """Write one user message as a spool file.

    The content is the raw stream-json stdin envelope so the supervisor can
    write it verbatim to claude's stdin.  Writing is atomic on POSIX
    (rename after write) and best-effort on Windows.

    Returns the path of the created spool file.
    """
    envelope_bytes = build_stdin_message(text)
    filename = _spool_filename(time.time())
    target = spool_dir / filename
    # Write to a temp sibling then rename → atomic on POSIX; Windows-verify.
    tmp = spool_dir / (filename + ".tmp")
    tmp.write_bytes(envelope_bytes)
    tmp.rename(target)
    return target


def consume_spool_entries(spool_dir: Path) -> list[Path]:
    """Return spool files sorted by name (FIFO order), skipping temp files.

    Only files ending in ``.json`` (not ``.tmp``) that contain valid JSON
    are returned.  Partially-written files (``.tmp``) are silently skipped.
    """
    entries: list[Path] = []
    try:
        for p in sorted(spool_dir.iterdir()):
            if p.suffix != ".json":
                continue
            # Fast-fail guard: skip files that are not yet valid JSON
            # (e.g. the writer was interrupted between write and rename).
            try:
                json.loads(p.read_bytes())
            except (json.JSONDecodeError, OSError):
                logger.debug("supervisor: spool skip incomplete file %s", p.name)
                continue
            entries.append(p)
    except OSError:
        logger.debug("supervisor: could not list spool dir %s", spool_dir)
    return entries


# ---------------------------------------------------------------------------
# Log-append helper (pure, unit-testable)
# ---------------------------------------------------------------------------


def append_log_line(log_path: Path, raw_line: str) -> None:
    """Append one stream-json line to the log file.

    The line is written as-is from claude's stdout (already a JSON object).
    A trailing newline is ensured so viewers can split on ``\\n``.
    """
    line = raw_line.rstrip("\n") + "\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


# ---------------------------------------------------------------------------
# Main supervisor loop (blocking; pure-testable inner logic)
# ---------------------------------------------------------------------------


def run_supervisor_loop(
    session_dir: Path,
    task: str,
    repo_root: str,
    *,
    session_id: str | None = None,
    _spawn_claude: bool = True,
) -> int:
    """Blocking supervisor main loop.

    Spawns claude, routes stdout → log file, polls spool dir → claude stdin.
    Returns the claude exit code (or -1 if spawn failed).

    If ``session_id`` is provided, a registry record is written (status=running)
    at startup and updated to status=stopped in the ``finally`` block.

    ``_spawn_claude=False`` is a test seam: skips the real Popen but exercises
    log/spool routing logic with a fake process object injected externally.
    Used only by unit tests.
    """
    import datetime

    session_dir = Path(session_dir)
    log_path = session_dir / _LOG_FILENAME
    spool_dir = session_dir / _SPOOL_DIRNAME
    spool_dir.mkdir(parents=True, exist_ok=True)

    # Self-log to a file: the supervisor is detached with stdout/stderr → DEVNULL,
    # so without this its own warnings/errors (spool failures, stdout-pump crashes)
    # are invisible. ``supervisor.log`` sits beside the claude log for debugging.
    try:
        _fh = logging.FileHandler(session_dir / "supervisor.log", encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_fh)
        logger.setLevel(logging.DEBUG)
    except OSError:
        pass

    # Write registry record before spawning claude, so the GUI can attach
    # even if the spawn fails (it will self-heal to stopped on next read).
    if session_id is not None:
        record: AgentSessionRecord = {
            "session_id": session_id,
            "claude_session_id": "",  # populated in B1b-4 via --resume
            "pid": os.getpid(),
            "status": "running",
            "log_path": str(log_path),
            "spool_dir": str(spool_dir),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "title": task[:60],  # first 60 chars as display title
        }
        try:
            write_record(record)
            logger.info(
                "supervisor: wrote registry record session_id=%s pid=%s",
                session_id,
                os.getpid(),
            )
        except OSError as exc:
            logger.warning("supervisor: could not write registry record: %s", exc)

    mcp_config_path = build_loopback_mcp_config(repo_root)
    argv = build_claude_argv(task, mcp_config_path)

    if not _spawn_claude:
        # Test path: caller sets up a fake proc externally; we just return.
        logger.debug("supervisor: _spawn_claude=False, returning without spawn")
        if session_id is not None:
            _update_registry_stopped(session_id, log_path, spool_dir)
        return 0

    env = os.environ.copy()
    # Remove API key so claude uses subscription auth (same as AgentRunner).
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            # cwd defaults to the current directory; the supervisor inherits
            # the GUI's cwd which is the repo root.
        )
    except OSError as exc:
        logger.error("supervisor: failed to spawn claude: %s", exc)
        if session_id is not None:
            _update_registry_stopped(session_id, log_path, spool_dir)
        return -1

    assert proc.stdin is not None
    assert proc.stdout is not None

    # Seed the first user turn via stdin (same rationale as AgentRunner).
    first_msg = build_stdin_message(task)
    proc.stdin.write(first_msg)
    proc.stdin.flush()

    logger.info("supervisor: claude pid=%s log=%s", proc.pid, log_path)

    try:
        _io_loop(proc, log_path, spool_dir)
    finally:
        # Flush any remaining stdout that arrived after loop exit.
        if proc.stdout:
            remaining = proc.stdout.read()
            for raw in remaining.decode("utf-8", errors="replace").splitlines():
                if raw.strip():
                    append_log_line(log_path, raw)
        logger.info("supervisor: done, exit_code=%s", proc.returncode)
        # Update registry to stopped now that the session has ended.
        if session_id is not None:
            _update_registry_stopped(session_id, log_path, spool_dir)

    return proc.returncode if proc.returncode is not None else -1


def _update_registry_stopped(session_id: str, log_path: Path, spool_dir: Path) -> None:
    """Write a stopped-status registry record for this session.

    Called from ``run_supervisor_loop``'s finally block; silently logs on
    failure (the GUI self-heals via pid-probe on the next ``list_records``).
    """
    from zcu_tools.gui.app.main.services.agent_session_registry import read_record

    try:
        existing = read_record(session_id)
        if existing is not None:
            existing["status"] = "stopped"
            write_record(existing)
        else:
            # Record was removed (e.g. user deleted it); don't recreate.
            logger.debug(
                "supervisor: registry record %s already gone at shutdown", session_id
            )
    except OSError as exc:
        logger.warning(
            "supervisor: could not update registry to stopped for %s: %s",
            session_id,
            exc,
        )


def _io_loop(
    proc: subprocess.Popen,  # type: ignore[type-arg]
    log_path: Path,
    spool_dir: Path,
) -> None:
    """Core I/O: read claude stdout → log (reader thread); poll spool → claude
    stdin (main loop). Runs until claude exits.

    stdout reading and spool delivery MUST be on separate threads: claude stays
    alive between turns (``--input-format stream-json``) and produces no stdout
    while waiting for the next stdin message, so a blocking ``read1`` on the main
    loop would starve the spool poll and the next user turn would never reach
    claude. The reader thread owns the (blocking) stdout read; the main loop owns
    the spool→stdin delivery. Single reader / single writer → no shared-state lock
    needed. ``read1`` blocks portably (no ``select``, which does not work on
    Windows pipes).
    """
    assert proc.stdin is not None
    assert proc.stdout is not None
    stdout = proc.stdout

    def _pump_stdout() -> None:
        """Blocking-read claude stdout → append complete lines to the log."""
        buf = bytearray()
        try:
            while True:
                chunk = stdout.read1(4096)  # type: ignore[union-attr]
                if not chunk:
                    break  # EOF — claude exited
                buf.extend(chunk)
                while True:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        break
                    line = buf[:nl].decode("utf-8", errors="replace")
                    del buf[: nl + 1]
                    if line.strip():
                        append_log_line(log_path, line)
        except Exception:
            logger.exception("supervisor: stdout pump error")

    reader = threading.Thread(
        target=_pump_stdout, name="claude-stdout-pump", daemon=True
    )
    reader.start()

    while True:
        # Poll spool directory and feed pending messages to claude stdin.
        for spool_file in consume_spool_entries(spool_dir):
            try:
                raw_bytes = spool_file.read_bytes()
                proc.stdin.write(raw_bytes)
                proc.stdin.flush()
                spool_file.unlink()
                logger.debug("supervisor: consumed spool %s", spool_file.name)
            except OSError as exc:
                logger.warning(
                    "supervisor: spool consume error %s: %s", spool_file.name, exc
                )

        # Check if claude has exited.
        if proc.poll() is not None:
            break

        time.sleep(_SPOOL_POLL_INTERVAL)

    reader.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Detached supervisor spawn (GUI entry point)
# ---------------------------------------------------------------------------


def spawn_supervisor_detached(
    session_dir: Path,
    task: str,
    repo_root: str,
    *,
    session_id: str | None = None,
) -> SupervisorHandle:
    """Spawn the supervisor as a detached process and return a handle.

    The supervisor is launched as ``python -m <this_module>`` with the
    session directory, task, repo root, and optional session_id as CLI
    arguments.  It is detached from the calling GUI process's lifetime:

    POSIX:   ``start_new_session=True`` (new session, no controlling tty,
             SIGHUP-independent).
    Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` creationflags.
             Windows-verify — not runnable on Linux.

    The caller is responsible for creating ``session_dir`` before calling
    this function.  If ``session_id`` is provided, the detached supervisor
    writes a registry record at startup and marks it stopped on exit.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable,
        "-m",
        "zcu_tools.gui.app.main.services.agent_supervisor",
        "--session-dir",
        str(session_dir),
        "--task",
        task,
        "--repo-root",
        repo_root,
    ]
    if session_id is not None:
        argv.extend(["--session-id", session_id])

    if sys.platform == "win32":
        # Windows-verify: DETACHED_PROCESS keeps the console hidden;
        # CREATE_NEW_PROCESS_GROUP is required for CTRL_BREAK_EVENT stop.
        creationflags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    else:
        # POSIX: start_new_session=True creates a new process group and
        # session so SIGHUP from the parent GUI does not reach the supervisor.
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    log_path = session_dir / _LOG_FILENAME
    spool_dir = session_dir / _SPOOL_DIRNAME

    logger.info(
        "spawn_supervisor_detached: pid=%s session_dir=%s", proc.pid, session_dir
    )
    return SupervisorHandle(pid=proc.pid, log_path=log_path, spool_dir=spool_dir)


# ---------------------------------------------------------------------------
# Stop helper (cross-platform, used by IndependentAgentSession)
# ---------------------------------------------------------------------------


def stop_supervisor(pid: int) -> None:
    """Send a stop signal to a running supervisor process.

    POSIX:   SIGINT → graceful shutdown of claude and supervisor.
    Windows: CTRL_BREAK_EVENT (requires CREATE_NEW_PROCESS_GROUP) with
             fallback to ``taskkill /F``.
             Windows-verify — CTRL_BREAK_EVENT path untested on Linux.
    """
    if sys.platform == "win32":
        _stop_windows(pid)
    else:
        _stop_posix(pid)


def _stop_posix(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGINT)
        logger.info("stop_supervisor: sent SIGINT to pid=%s", pid)
    except ProcessLookupError:
        logger.debug("stop_supervisor: pid=%s already gone", pid)
    except OSError as exc:
        logger.warning("stop_supervisor: SIGINT failed for pid=%s: %s", pid, exc)


def _stop_windows(pid: int) -> None:  # pragma: no cover  # Windows-verify
    """Windows stop: CTRL_BREAK_EVENT then taskkill fallback."""
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        logger.info("stop_supervisor: sent CTRL_BREAK_EVENT to pid=%s", pid)
        return
    except OSError as exc:
        logger.warning(
            "stop_supervisor: CTRL_BREAK_EVENT failed for pid=%s: %s; "
            "falling back to taskkill",
            pid,
            exc,
        )
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            check=False,
            capture_output=True,
        )
        logger.info("stop_supervisor: taskkill called for pid=%s", pid)
    except OSError as exc2:
        logger.error("stop_supervisor: taskkill also failed for pid=%s: %s", pid, exc2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detached supervisor that wraps a claude subprocess."
    )
    parser.add_argument("--session-dir", required=True, help="Session directory path.")
    parser.add_argument("--task", required=True, help="Initial user task string.")
    parser.add_argument("--repo-root", required=True, help="Repository root path.")
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Registry session_id (8-hex).  When provided, the supervisor writes "
            "a running record to the agent-session registry at startup and "
            "a stopped record on exit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m zcu_tools…agent_supervisor``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [supervisor] %(message)s",
    )
    args = _parse_args(argv)
    exit_code = run_supervisor_loop(
        session_dir=Path(args.session_dir),
        task=args.task,
        repo_root=args.repo_root,
        session_id=args.session_id,
    )
    sys.exit(exit_code if exit_code >= 0 else 1)


if __name__ == "__main__":
    main()
