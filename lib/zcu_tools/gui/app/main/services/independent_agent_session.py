"""IndependentAgentSession — AgentSessionPort backed by a detached supervisor.

This is the B1b-1/B1b-2 implementation of ``AgentSessionPort``.  A detached
Python supervisor process owns the ``claude`` subprocess; this class tails the
supervisor's NDJSON log file from the GUI main thread via a ``QTimer`` poll
and writes user messages into the command spool directory.

IPC summary
-----------
  log  (read):  ``session_dir/log.ndjson`` — one stream-json line per row;
                ``_on_tick`` reads new bytes starting from ``_log_offset``,
                splits on ``\\n``, feeds each complete line to ``StreamJsonParser``.
  spool (write): ``session_dir/spool/<ts>_<rand>.json`` — one file per user
                 message, written by ``send_user_message`` via
                 ``write_spool_message``; the supervisor consumes and deletes.

State tracking
--------------
  ``AgentRunState`` is reused exactly as in ``AgentRunner`` (same state machine).
  Supervisor liveness is inferred from a pid-exists check (``_supervisor_alive``).
  ``is_running()`` returns True while the supervisor process is alive and the
  run-state is active (working or waiting).

Threading model
---------------
  All methods must be called from the Qt main thread.  The ``QTimer`` callback
  ``_on_tick`` is also main-thread (Qt guarantee), so no locking is needed.

B1b-2 additions over B1b-1
---------------------------
  - ``start()`` uses ``registry_dir()/<session_id>/`` instead of a tmpdir;
    the ``session_id`` is passed to ``spawn_supervisor_detached`` which forwards
    it to the detached supervisor so it writes the registry record.
  - ``attach(record)`` rebuilds the tail state from an existing registry record
    without spawning a new supervisor.  The poll-tail runs from offset=0 so the
    full transcript history is replayed.
  - ``detach()`` stops the poll timer and clears ``_handle`` **without** calling
    ``stop_supervisor`` — the session keeps running and can be re-attached later.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QTimer  # type: ignore[attr-defined]

from .agent_runner import (
    AgentRunState,
    AssistantTextUpdate,
    ResultUpdate,
    StreamJsonParser,
    ToolUseUpdate,
    TranscriptUpdate,
)
from .agent_session_registry import (
    AgentSessionRecord,
    new_session_id,
    registry_dir,
)
from .agent_supervisor import (
    SupervisorHandle,
    spawn_supervisor_detached,
    stop_supervisor,
    write_spool_message,
)
from .ports import AgentSessionPort, AgentState  # noqa: F401 (re-export for type alias)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Poll interval for log-tail (milliseconds).  ~5 polls per second balances
# latency against main-thread CPU; ADR-0024 says "~200ms".
_LOG_POLL_INTERVAL_MS = 200

# Guard against runaway log growth eating the read buffer on a single tick.
_MAX_BYTES_PER_TICK = 512 * 1024  # 512 KiB


class IndependentAgentSession(QObject):
    """AgentSessionPort backed by a detached supervisor + file-based IPC.

    Lifecycle (new session):
      1. ``start(task, repo_root)`` — create session_dir under registry_dir,
         spawn supervisor (which writes the registry record), start QTimer poll-tail.
      2. ``send_user_message(text)`` — write a spool file.
      3. ``stop()`` — send platform stop signal to supervisor pid.
      4. Poll-tail reads log lines → StreamJsonParser → callbacks → state machine.
      5. When the log shows a ``result`` frame (or supervisor disappears),
         the session ends.

    Lifecycle (attach to existing session):
      1. ``attach(record)`` — set up ``_handle`` from record's log/spool/pid;
         reset parser + offset to 0; start poll-tail from beginning.
      2. The full log history is replayed through callbacks (transcript rebuild).
      3. If the record is stopped, the session reads to EOF and transitions to
         stopped naturally; ``send_user_message`` is a no-op.
      4. ``detach()`` — stop the poll timer, clear ``_handle``, do NOT kill supervisor.

    ``callbacks`` mirrors the ``_RunnerCallbacks`` dataclass from AgentRunner
    so the wiring in ``Controller`` can share the same callback-construction
    code for both backends.
    """

    def __init__(
        self,
        *,
        on_update: Callable[[list[TranscriptUpdate]], None],
        on_state_changed: Callable[[AgentState], None],
        on_process_error: Callable[[str], None],
        has_pending_wait: Callable[[], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_update = on_update
        self._on_state_changed = on_state_changed
        self._on_process_error = on_process_error
        self._run_state = AgentRunState(has_pending_wait)
        self._parser = StreamJsonParser()

        self._handle: SupervisorHandle | None = None
        self._log_offset: int = 0  # byte offset into log.ndjson
        self._line_buf: bytearray = bytearray()  # incomplete line accumulator

        # QTimer for log poll-tail (main thread).
        self._timer = QTimer(self)
        self._timer.setInterval(_LOG_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        # AgentSessionPort listeners registered via add_state_listener.
        self._state_listeners: list[Callable[[AgentState], None]] = []

    # ------------------------------------------------------------------
    # AgentSessionPort
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._run_state.state

    def is_running(self) -> bool:
        """True while the detached supervisor process is alive.

        Process-liveness, NOT run-state: an idle session *between turns* (after a
        ``result`` frame set state to ``idle``) still has a live supervisor whose
        claude keeps reading stdin, so a follow-up Send must route to
        ``send_user_message`` (not a fresh ``start``). Mirrors the CLI backend's
        ``is_running`` (QProcess alive) — see AgentSessionPort contract.
        """
        if self._handle is None:
            return False
        return _supervisor_alive(self._handle.pid)

    def start(self, task: str, repo_root: str) -> None:
        """Create session dir under registry_dir, spawn detached supervisor, begin poll-tail.

        B1b-2: the session dir is ``registry_dir()/<session_id>/`` instead of
        a tmpdir so the registry and log/spool dirs share the same root.  The
        short ``session_id`` is passed to ``spawn_supervisor_detached`` which
        forwards ``--session-id`` to the supervisor CLI; the supervisor writes
        the registry record with its own PID.

        Fast-fails if a supervisor is already running.
        """
        if self._handle is not None and _supervisor_alive(self._handle.pid):
            logger.warning(
                "IndependentAgentSession.start() called while supervisor still running;"
                " ignored"
            )
            return

        # Reset parser and state for a fresh session.
        self._parser = StreamJsonParser()
        self._log_offset = 0
        self._line_buf = bytearray()
        self._run_state.on_start()

        # B1b-2: use a named dir under registry_dir so registry record and
        # log/spool are co-located.  The supervisor will write the record.
        session_id = new_session_id()
        session_dir = registry_dir() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._handle = spawn_supervisor_detached(
                session_dir, task, repo_root, session_id=session_id
            )
        except OSError as exc:
            logger.error("IndependentAgentSession.start(): spawn failed: %s", exc)
            self._run_state.on_stop()
            self._emit_state()
            self._on_process_error(f"supervisor spawn failed: {exc}")
            return

        self._timer.start()
        self._emit_state()
        logger.info(
            "IndependentAgentSession: supervisor pid=%s session_dir=%s",
            self._handle.pid,
            session_dir,
        )

    def send_user_message(self, text: str) -> None:
        """Write a user message into the spool directory.

        No-op if no supervisor is running.
        """
        if self._handle is None:
            logger.warning(
                "IndependentAgentSession.send_user_message(): no supervisor; ignored"
            )
            return
        if not _supervisor_alive(self._handle.pid):
            logger.warning(
                "IndependentAgentSession.send_user_message(): supervisor gone; ignored"
            )
            return
        try:
            write_spool_message(self._handle.spool_dir, text)
        except OSError as exc:
            logger.error(
                "IndependentAgentSession.send_user_message(): spool write failed: %s",
                exc,
            )
            return
        self._run_state.on_stdin_sent()
        self._emit_state()

    def stop(self) -> None:
        """Send a platform stop signal to the supervisor."""
        if self._handle is None:
            return
        stop_supervisor(self._handle.pid)
        self._run_state.on_stop()
        self._timer.stop()
        self._emit_state()

    def session_id(self) -> str:
        """Return the session_id from the last ``system/init`` frame."""
        return self._parser.session_id

    def add_state_listener(self, cb: Callable[[AgentState], None]) -> None:
        """Register a callback invoked on every state transition (main-thread).

        Duplicate registrations are silently ignored.  Exceptions raised by
        ``cb`` are swallowed and logged so they cannot interrupt the state
        machine or other listeners.
        """
        if cb not in self._state_listeners:
            self._state_listeners.append(cb)

    # ------------------------------------------------------------------
    # B1b-2: attach / detach
    # ------------------------------------------------------------------

    def attach(self, record: AgentSessionRecord) -> None:
        """Attach to an existing session without spawning a new supervisor.

        Rebuilds the poll-tail state from ``record``'s log_path / spool_dir /
        pid.  The log is tailed from offset=0 so the full transcript history
        is replayed through the callbacks, letting the dialog's AgentChatService
        rebuild its transcript display.

        If the record's ``status`` is ``"stopped"`` the session is started in
        ``"working"`` state (so the ticker can run and reach EOF naturally);
        once the result frame arrives (or the log is fully consumed with no
        result), the state machine transitions to stopped.  If the record status
        is ``"running"`` the normal live-tail behaviour applies.

        Any previous handle is detached first (timer stopped, handle cleared)
        without sending a stop signal.
        """
        if self._handle is not None:
            self.detach()

        log_path = Path(record["log_path"])
        spool_dir = Path(record["spool_dir"])
        pid = record["pid"]

        self._handle = SupervisorHandle(pid=pid, log_path=log_path, spool_dir=spool_dir)

        # Fresh parser and offset=0: replay the full log history.
        self._parser = StreamJsonParser()
        self._log_offset = 0
        self._line_buf = bytearray()

        # Transition to working so the ticker runs.  The state machine will
        # converge to stopped when the result frame is reached (or pid dies).
        self._run_state.on_start()

        self._timer.start()
        self._emit_state()
        logger.info(
            "IndependentAgentSession.attach: session_id=%s pid=%s status=%s",
            record.get("session_id"),
            pid,
            record.get("status"),
        )

    def detach(self) -> None:
        """Detach from the current session without stopping the supervisor.

        Stops the poll timer and clears the handle so this object is idle.
        The remote supervisor process is left running; a future ``attach()``
        call can re-connect to the same session.

        This is the Close button's action in the Picker UI (decision H):
        CLI AgentRunner's ``detach()`` calls ``stop()``; Independent's
        ``detach()`` only stops the tail.
        """
        self._timer.stop()
        self._handle = None
        self._run_state.on_stop()
        self._emit_state()
        logger.debug("IndependentAgentSession.detach(): timer stopped, handle cleared")

    # ------------------------------------------------------------------
    # Log poll-tail (QTimer callback, main thread)
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        """Called by QTimer on the main thread.

        Reads new bytes from the log file starting at ``_log_offset``, splits
        on ``\\n``, and feeds complete lines to ``StreamJsonParser``.
        """
        if self._handle is None:
            return

        log_path = self._handle.log_path

        try:
            new_bytes = _read_log_tail(log_path, self._log_offset, _MAX_BYTES_PER_TICK)
        except OSError:
            # Log not yet created (supervisor just started) — try again next tick.
            return

        if not new_bytes:
            # No new data; if the supervisor has exited (session truly ended),
            # mark stopped and stop tailing — regardless of run-state (an idle
            # between-turns session whose supervisor then dies must also stop).
            if (
                not _supervisor_alive(self._handle.pid)
                and self._run_state.state != "stopped"
            ):
                # working/waiting at death = died mid-turn (unexpected crash);
                # idle = normal between-turns session end (no error).
                was_active = self._run_state.is_active()
                logger.info(
                    "IndependentAgentSession: supervisor pid=%s exited; stopping tail",
                    self._handle.pid,
                )
                self._run_state.on_stop()
                self._timer.stop()
                self._emit_state()
                if was_active:
                    self._on_process_error(
                        "supervisor process disappeared unexpectedly"
                    )
            return

        self._log_offset += len(new_bytes)
        self._line_buf.extend(new_bytes)

        # Extract and process complete lines.
        while True:
            nl = self._line_buf.find(b"\n")
            if nl == -1:
                break
            line_bytes = self._line_buf[:nl]
            del self._line_buf[: nl + 1]
            line = line_bytes.decode("utf-8", errors="replace")
            updates = self._parser.feed_line(line)
            if updates:
                self._route_updates(updates)

    # ------------------------------------------------------------------
    # Internal helpers (testable by calling directly, bypassing QTimer)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Expose ``_on_tick`` for testing without a live QTimer.

        Tests can feed log data and call ``tick()`` to drive the state machine
        without starting the Qt event loop.
        """
        self._on_tick()

    def _route_updates(self, updates: list[TranscriptUpdate]) -> None:
        """Route parsed updates to callbacks and update the run-state machine."""
        self._on_update(updates)
        has_assistant = any(
            isinstance(u, (AssistantTextUpdate, ToolUseUpdate)) for u in updates
        )
        if has_assistant:
            self._run_state.on_assistant_chunk_received()
        for update in updates:
            if isinstance(update, ResultUpdate):
                self._run_state.on_result(update.is_error)
                # A result frame ends a *turn*, not the session: the detached
                # supervisor stays alive between turns, so keep poll-tailing for
                # the next turn's output. The timer stops only on supervisor
                # death (_on_tick) or detach()/stop().
        self._emit_state()

    def _emit_state(self) -> None:
        """Notify all listeners of the current state."""
        current = self._run_state.state
        self._on_state_changed(current)
        for cb in self._state_listeners:
            try:
                cb(current)
            except Exception:
                logger.exception(
                    "IndependentAgentSession: state listener %r raised; swallowed", cb
                )


# ---------------------------------------------------------------------------
# Platform-portable process liveness check
# ---------------------------------------------------------------------------


def _supervisor_alive(pid: int) -> bool:
    """Return True if ``pid`` refers to a live process.

    Uses ``os.kill(pid, 0)`` on POSIX (signal 0 = existence probe, no effect).
    On Windows, a Process handle is opened briefly.
    Windows-verify — the ctypes branch is not tested on Linux.
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
        # Process exists but we lack permission to signal it.
        # This can happen if the process is owned by another user.
        return True
    except OSError:
        return False


def _pid_alive_windows(pid: int) -> bool:  # pragma: no cover  # Windows-verify
    """Windows-verify: open process handle and check exit code."""
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
# Log-tail reader (pure, unit-testable)
# ---------------------------------------------------------------------------


def _read_log_tail(log_path: Path, offset: int, max_bytes: int) -> bytes:
    """Read up to ``max_bytes`` from ``log_path`` starting at ``offset``.

    Raises ``OSError`` if the file does not exist yet (caller should retry).
    Returns ``b""`` if no new data is available.
    """
    with log_path.open("rb") as fh:
        fh.seek(offset)
        return fh.read(max_bytes)
