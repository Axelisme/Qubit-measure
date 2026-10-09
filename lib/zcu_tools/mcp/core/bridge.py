"""McpBridge — the shared transport for the GUI apps' MCP servers.

Each GUI app ships a small ``mcp_server.py`` launched standalone (``python
.../mcp_server.py`` per ``.mcp.json``, stdio transport). They all need the same
plumbing to bridge an MCP host (Claude / Gemini / VS Code) to a live GUI's
``NdjsonRpcEndpoint`` over a single persistent TCP socket. This module owns that
plumbing; it knows nothing of any app's method set, version guard, operations,
or diagnostics.

  - :class:`McpBridge` holds one process's socket state (the socket, the
    reader thread, the RID condition + pending-reply map, the launched GUI
    subprocess + its pid/log files) as *instance* attributes — no module
    globals. It exposes ``send_rpc_raw`` (low-level request/reply, no policy),
    ``connect`` / ``disconnect`` / ``launch`` / ``stop`` (lifecycle), and
    ``wire_version_note`` (the handshake probe). An optional ``on_event`` hook
    receives event-push lines (the reader otherwise drops them).
  - :class:`MCPBridgeConfig` extends
    :class:`~zcu_tools.mcp.core.stdio_server.McpServerConfig` with the GUI-bridge
    launch knobs (name / port / versions / pid+log file names / run-script name).
    Tool generation and the MCP stdio loop live in
    :mod:`zcu_tools.mcp.core.stdio_server`.

App-specific policy stays with each app: the read-only apps wrap
``send_rpc_raw`` in a thin error-raising ``send_gui_rpc`` and drop events;
measure-gui's session composes ``send_rpc_raw`` with its optimistic-concurrency
guard, operation tracking, and hand-written tools. It does not subscribe to push
events; operation request/reply supplies Stop feedback.

Threading:
  - Main (stdio) thread: reads MCP request lines, dispatches into tool handlers,
    writes MCP response lines back.
  - Reader thread (per :class:`McpBridge`): the only reader of the GUI socket;
    parses NDJSON lines into RPC replies (delivered to the matching waiter) or
    event pushes (handed to ``on_event``, or dropped if None).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES, encode_line
from zcu_tools.mcp.core.stdio_server import McpServerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPBridgeConfig(McpServerConfig):
    """A :class:`McpServerConfig` plus the knobs an :class:`McpBridge` needs to
    fork and talk to a live GUI subprocess: the control port, the wire/mcp version
    for the handshake, and the pid / log / run-script paths.
    """

    app_name: str
    default_port: int
    mcp_version: int
    wire_version: int
    pid_file: Path
    log_file: Path
    run_script_name: str
    # The session-discovery slug the GUI advertises under (e.g. "measure"); shared
    # with the GUI writer. Distinct from app_name (measure's app_name is "gui").
    app_slug: str = ""

    @property
    def stderr_file(self) -> Path:
        """File that receives the launched GUI's stderr, next to ``log_file``.

        Each launch truncates it. A file never fills up the way an unread pipe
        does, so warnings written after startup cannot block the GUI.
        """
        return self.log_file.with_name(self.log_file.name + ".stderr")


def resolve_connect_port(config: MCPBridgeConfig, requested: int | None) -> int:
    """Pick the port a ``connect`` tool should attach to.

    An explicit ``requested`` port always wins. Otherwise consult session
    discovery for a live GUI advertised under ``config.app_slug`` (covers the
    ephemeral-fallback case where the GUI is not on its agreed-upon port); if none
    is live, fall back to the agreed-upon ``default_port`` (backward compatible).
    """
    if requested is not None:
        return requested
    if config.app_slug:
        from zcu_tools.gui.remote.session_discovery import read_session

        entry = read_session(config.app_slug)
        if entry is not None:
            return entry["port"]
    return config.default_port


def port_is_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(code)
            ):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


class GuiAuthenticationError(RuntimeError):
    """The GUI rejected a control-token authentication request."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"GUI auth failed ({code}): {message}")


class GuiTransportTimeoutError(TimeoutError):
    """The GUI socket did not return an RPC reply before the transport deadline."""

    def __init__(self, method: str, timeout_seconds: float) -> None:
        self.method = method
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"GUI RPC {method!r} did not complete within {timeout_seconds:g}s"
        )


class GuiMessageTooLargeError(ValueError):
    """A request was not sent, or an oversized incoming frame closed the socket."""

    def __init__(self, direction: Literal["request", "response"]) -> None:
        self.direction = direction
        self.reason = "message_too_large"
        detail = (
            "The request was not sent."
            if direction == "request"
            else "The request may have executed; inspect state before retrying."
        )
        super().__init__(f"GUI {direction} exceeds {MAX_LINE_BYTES} bytes. {detail}")


# A line arrived from the GUI: route it (reply keyed by id / event push).
DeliverFn = Callable[[dict[str, Any]], None]
# A terminal transport failure wakes pending RPCs with its cause, without replay.
OnClosedFn = Callable[[Exception | None], None]


class Transport(Protocol):
    """The seam between :class:`McpBridge` and how lines reach the GUI.

    The bridge owns the RPC bookkeeping (pending map, RID condition) and routing
    logic; a transport owns only *moving bytes*. The bridge injects its routing
    callbacks once via :meth:`attach`; the transport calls them when a line
    arrives. The real transport is a TCP socket + reader thread
    (:class:`SocketTransport`); tests inject a synchronous fake.
    """

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Hand the transport the bridge's routing callbacks (once)."""
        ...

    @property
    def is_open(self) -> bool: ...

    def send_line(self, payload: dict[str, Any]) -> None:
        """Serialise + send one NDJSON line toward the GUI."""
        ...

    def close(self) -> None:
        """Tear down (idempotent)."""
        ...


class SocketTransport:
    """The real transport: a TCP socket + a dedicated NDJSON reader thread.

    Owns the socket, the sole-reader thread (recv + frame + route via the
    attached callbacks), and the line writer (lock-guarded sendall). The bridge's
    pending map / RID condition stay in the bridge — on socket drop the reader
    calls the attached ``on_closed`` so the bridge can wake its waiters.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        self._sock_lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._deliver_reply: DeliverFn | None = None
        self._deliver_event: DeliverFn | None = None
        self._on_closed: OnClosedFn | None = None

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        self._deliver_reply = deliver_reply
        self._deliver_event = deliver_event
        self._on_closed = on_closed

    @property
    def is_open(self) -> bool:
        with self._sock_lock:
            return self._sock is not None

    def open(self, port: int) -> None:
        """Connect to 127.0.0.1:port and start the reader thread.

        Raises OSError if nothing is listening (the caller maps it to an
        actionable message).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            sock.close()
            raise
        # Short blocking timeout so the reader loop wakes to observe the stop flag.
        sock.settimeout(1.0)
        with self._sock_lock:
            self._sock = sock
            self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name=f"mcp-{self._app_name}-reader", daemon=True
        )
        self._reader_thread.start()

    def send_line(self, payload: dict[str, Any]) -> None:
        try:
            data = encode_line(payload)
        except RemoteError as exc:
            raise GuiMessageTooLargeError("request") from exc
        with self._sock_lock:
            if self._sock is None:
                raise RuntimeError("transport not open")
            self._sock.sendall(data)

    def close(self) -> None:
        with self._sock_lock:
            self._reader_stop.set()
            sock, self._sock = self._sock, None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        t = self._reader_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._reader_thread = None

    def _reader_loop(self) -> None:
        """Sole reader of the GUI socket; routes replies, hands events to hook."""
        buf = bytearray()
        sock: socket.socket | None = None
        failure: Exception | None = None
        while not self._reader_stop.is_set() and failure is None:
            sock = self._sock
            if sock is None:
                return
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                frame_size = len(buf) if nl < 0 else nl
                if frame_size > MAX_LINE_BYTES:
                    failure = GuiMessageTooLargeError("response")
                    break
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if not line:
                    continue
                self._route_line(line)
        # Unexpected EOF invalidates liveness before waking callers. A deliberate
        # close already retired the socket and must not affect a new connection.
        with self._sock_lock:
            if self._reader_stop.is_set() or self._sock is not sock or sock is None:
                return
            self._sock = None
        sock.close()
        if self._on_closed is not None:
            self._on_closed(failure)

    def _route_line(self, line: bytes) -> None:
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError):
            logger.debug("skipping unparseable GUI socket line", exc_info=True)
            return
        if isinstance(msg, dict) and "id" in msg:
            if self._deliver_reply is not None:
                self._deliver_reply(msg)
        elif (
            isinstance(msg, dict) and "event" in msg and self._deliver_event is not None
        ):
            self._deliver_event(msg)


class McpBridge:
    """One MCP server process's bridge to a live GUI over a TCP socket.

    Holds all socket + subprocess state as instance attributes. Construct with
    the app :class:`MCPBridgeConfig`; optionally pass ``on_event`` to receive
    event-push lines (else they are dropped). Inert until :meth:`connect` /
    :meth:`launch`.
    """

    def __init__(
        self,
        config: MCPBridgeConfig,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self._rid_cond = threading.Condition()
        self._rid_counter = 0
        self._pending: dict[str, dict[str, Any]] = {}
        self._proc: subprocess.Popen[bytes] | None = None
        # The wire transport. None until connect()/launch() builds a real
        # SocketTransport, or a test injects a fake via set_transport / the ctor.
        self._transport: Transport | None = None
        if transport is not None:
            self.set_transport(transport)

    def set_transport(self, transport: Transport | None) -> None:
        """Swap the wire transport (the seam for tests + connect()).

        Attaching wires the bridge's routing callbacks into the transport; the
        bridge keeps the pending map / RID condition. Replacing or detaching
        fails the previous transport's pending RPCs but does not close that
        externally managed wire. Use disconnect() for owned connection teardown.
        """
        with self._rid_cond:
            previous = self._transport
            if previous is transport:
                return
            if previous is not None:
                self._retire_transport(previous)
            self._transport = transport
            if transport is not None:
                transport.attach(
                    self._deliver_reply,
                    self._deliver_event,
                    lambda failure: self._on_socket_closed(transport, failure),
                )

    def _deliver_event(self, msg: dict[str, Any]) -> None:
        # Preserve the drop-if-None semantics: read-only apps wire no on_event.
        if self._on_event is not None:
            self._on_event(msg)

    def _on_socket_closed(
        self, transport: Transport, failure: Exception | None
    ) -> None:
        self._retire_transport(
            transport, failure or ConnectionError("GUI socket closed unexpectedly.")
        )

    def _retire_transport(
        self, transport: Transport, failure: Exception | None = None
    ) -> bool:
        """Atomically end admission and settle waiters, but do not join the reader."""
        with self._rid_cond:
            if self._transport is not transport:
                return False
            self._transport = None
            failure = failure or RuntimeError(
                "Disconnected from GUI. Pending RPC outcomes are unknown."
            )
            for holder in self._pending.values():
                holder["error"] = failure
                holder["done"] = True
            self._pending.clear()
            self._rid_cond.notify_all()
            return True

    @property
    def is_connected(self) -> bool:
        with self._rid_cond:
            return self._transport is not None and self._transport.is_open

    # ------------------------------------------------------------------
    # pid file
    # ------------------------------------------------------------------

    def _write_pid_file(self, pid: int) -> None:
        with suppress(OSError):
            self.config.pid_file.write_text(str(pid))

    def _read_pid_file(self) -> int | None:
        try:
            return int(self.config.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def _clear_pid_file(self) -> None:
        self.config.pid_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Socket I/O
    # ------------------------------------------------------------------

    def _next_rid(self) -> str:
        with self._rid_cond:
            self._rid_counter += 1
            return f"mcp-{self._rid_counter}"

    def _deliver_reply(self, msg: dict[str, Any]) -> None:
        rid = msg.get("id")
        if not isinstance(rid, str):
            return
        with self._rid_cond:
            holder = self._pending.pop(rid, None)
            if holder is None:
                return
            holder["message"] = msg
            holder["done"] = True
            self._rid_cond.notify_all()

    def _close_failed_transport(self, transport: Transport, failure: Exception) -> None:
        """Wake peers before closing a stream whose outcomes are uncertain."""
        if self._retire_transport(
            transport,
            ConnectionError(f"GUI disconnected after transport failure: {failure}"),
        ):
            transport.close()

    def send_rpc_raw(
        self, method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Issue one RPC and wait for its reply (no policy). Raises on timeout.

        Returns the raw wire response dict (``{ok, result}`` or ``{ok:False,
        error}``). App layers wrap this to raise on ``ok:False`` and add policy.
        Disconnection interrupts the wait, not the remote operation. An admitted
        request may have been sent; the bridge never replays it.
        """
        holder: dict[str, Any] = {"done": False}
        with self._rid_cond:
            transport = self._transport
            if transport is None or not transport.is_open:
                raise RuntimeError(
                    f"GUI not connected. Call {self.config.tool_prefix}connect first."
                )
            rid = self._next_rid()
            self._pending[rid] = holder
        try:
            transport.send_line({"id": rid, "method": method, "params": params})
        except GuiMessageTooLargeError:
            # Local preflight failure: no bytes were sent, so this connection is safe.
            with self._rid_cond:
                self._pending.pop(rid, None)
            raise
        except Exception as exc:
            with self._rid_cond:
                self._pending.pop(rid, None)
            self._close_failed_transport(transport, exc)
            raise

        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        with self._rid_cond:
            while not holder["done"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending.pop(rid, None)
                    timed_out = True
                    break
                self._rid_cond.wait(timeout=remaining)
        if timed_out:
            failure = GuiTransportTimeoutError(method, timeout_seconds)
            self._close_failed_transport(transport, failure)
            raise failure
        if "error" in holder and "message" not in holder:
            raise holder["error"]
        return holder["message"]

    def wire_version_note(self) -> str:
        """Probe ``wire.version`` and report a human-readable version note."""
        cfg = self.config
        try:
            resp = self.send_rpc_raw("wire.version", {}, 5.0)
            result = resp.get("result", {})
            wire_ver = result.get("wire_version")
            gui_ver = result.get("gui_version", "?")
        except Exception as exc:  # noqa: BLE001 — probe is best-effort
            return (
                f" versions: mcp wire=v{cfg.wire_version} mcp=v{cfg.mcp_version}, "
                f"gui=unknown ({exc})"
            )
        if wire_ver != cfg.wire_version:
            return (
                f" WIRE VERSION MISMATCH: mcp wire=v{cfg.wire_version}, "
                f"gui wire=v{wire_ver} — the two processes speak different "
                "protocols; restart the stale one."
            )
        return (
            f" wire v{cfg.wire_version} (mcp==gui); gui code v{gui_ver}, "
            f"mcp code v{cfg.mcp_version}."
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, port: int, token: str | None = None) -> str:
        """Attach to an already-running GUI's control port. Returns a note.

        Tears down any existing connection first. With a token, authenticates
        via the ``auth`` RPC. The caller's ``send_gui_rpc`` is used for auth so
        app-level error formatting applies — but auth is a plain RPC, so we use
        ``send_rpc_raw`` + a minimal check here to stay app-agnostic.
        """
        cfg = self.config
        self.disconnect()

        transport = SocketTransport(cfg.app_name)
        self.set_transport(transport)
        try:
            transport.open(port)
        except OSError as exc:
            self.set_transport(None)
            raise RuntimeError(
                f"No GUI is listening on 127.0.0.1:{port} ({exc}). "
                f"{cfg.tool_prefix}connect attaches to an already-running GUI; "
                f"start one with {cfg.tool_prefix}launch."
            ) from exc

        if token:
            resp = self.send_rpc_raw("auth", {"token": token}, 30.0)
            if not resp.get("ok", False):
                err = resp.get("error")
                if not isinstance(err, dict):
                    err = {}
                code = str(err.get("code", "unauthorized"))
                message = str(err.get("message", "authentication rejected"))
                self.disconnect()
                raise GuiAuthenticationError(code, message)
            return (
                f"Connected to {cfg.server_display_name} on 127.0.0.1:{port} "
                f"with token auth." + self.wire_version_note()
            )
        return (
            f"Connected to {cfg.server_display_name} on 127.0.0.1:{port}."
            + self.wire_version_note()
        )

    def disconnect(self) -> str:
        """Detach and wake pending RPCs without cancelling remote operations."""
        with self._rid_cond:
            transport = self._transport
            if transport is None:
                return "Not connected."
            was_open = transport.is_open
            self._retire_transport(transport)
        # close may join the reader, whose final callback needs the condition.
        transport.close()
        return "Disconnected from GUI." if was_open else "Not connected."

    def launch(
        self,
        repo_root: Path,
        port: int,
        token: str | None = None,
        *,
        auto_connect: bool = True,
        extra_args: list[str] | None = None,
    ) -> str:
        """Fork the GUI subprocess on ``port``, wait until ready, maybe connect.

        ``repo_root`` anchors ``scripts/<run_script_name>``. ``extra_args`` are
        appended to the launch command (apps that need extra flags). The GUI's
        stderr goes to ``config.stderr_file``, truncated on each launch; an
        ``OSError`` from opening that file aborts the launch.
        """
        cfg = self.config
        if self._proc is not None and self._proc.poll() is None:
            return f"GUI already running (pid={self._proc.pid})."

        python = sys.executable
        run_gui = repo_root / "scripts" / cfg.run_script_name
        if not run_gui.exists():
            raise FileNotFoundError(f"{cfg.run_script_name} not found at {run_gui}")

        if port_is_open(port):
            raise RuntimeError(
                f"Port {port} is already in use — a GUI is likely already running "
                f"there. Use {cfg.tool_prefix}connect to attach to it, or launch "
                f"on a different port."
            )

        cmd = [
            str(python),
            str(run_gui),
            "--control-port",
            str(port),
            "--log-file",
            str(cfg.log_file),
        ]
        if token:
            cmd += ["--control-token", token]
        if extra_args:
            cmd += list(extra_args)

        # The child keeps its own copy of the handle; the parent closes its copy.
        with cfg.stderr_file.open("wb") as stderr:
            if os.name == "nt":
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    start_new_session=True,
                )
        self._write_pid_file(self._proc.pid)

        ready = self._wait_for_launch(self._proc, port)
        pid = self._proc.pid
        if not ready:
            return (
                f"GUI launched (pid={pid}) but port {port} not yet reachable — "
                f"call {cfg.tool_prefix}connect manually when ready."
            )

        log_note = f" DEBUG log: {cfg.log_file}; stderr: {cfg.stderr_file}"
        if auto_connect:
            self.connect(port, token)
            return (
                f"GUI launched (pid={pid}), listening on port {port}, and connected."
                + self.wire_version_note()
                + log_note
            )
        return f"GUI launched (pid={pid}) and listening on port {port}." + log_note

    def _wait_for_launch(self, proc: subprocess.Popen[bytes], port: int) -> bool:
        """Wait for readiness, reporting an early process exit with its stderr tail."""
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                stderr = self.config.stderr_file.read_bytes()
                tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
                self._proc = None
                raise RuntimeError(
                    f"GUI process exited during startup (returncode={rc}) before "
                    f"port {port} was ready. Last stderr "
                    f"({self.config.stderr_file}):\n" + "\n".join(tail)
                )
            if port_is_open(port):
                return True
            time.sleep(0.3)
        return False

    @property
    def launched_gui(self) -> bool:
        """True if THIS bridge launched a GUI subprocess that is still running.

        Cleanup-on-exit guards on this so an *attach-only* server (lazy
        auto-connect to a GUI another process launched) never stops that shared
        GUI. Without the guard, the pid-file fallback in ``stop()`` lets any
        attached server kill a GUI it merely connected to — e.g. the
        external-terminal agent's loopback MCP server shutting down the user's
        GUI when the agent window closes. ``stop()`` keeps the pid-file fallback
        for the *explicit* ``gui_stop`` tool (a deliberate action); only the
        automatic exit cleanup is restricted to launched-by-us GUIs.
        """
        proc = self._proc
        return proc is not None and proc.poll() is None

    def wait_for_gui_exit(self, pid: int, timeout: float = 5.0) -> bool:
        """Wait for the responding GUI process without terminating it.

        The PID comes from that GUI's shutdown reply, never a shared PID file.
        A timeout leaves the process and connection alone.
        """
        if isinstance(pid, bool) or pid <= 0 or timeout < 0:
            raise ValueError("expected a positive GUI PID and nonnegative timeout")
        proc = self._proc
        if proc is not None and (proc.pid != pid or proc.poll() is not None):
            proc = None
        return self._await_exit(pid, proc, timeout)

    def _pid_for_stop(self) -> tuple[int | None, subprocess.Popen[bytes] | None]:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            return proc.pid, proc
        return self._read_pid_file(), None

    def _await_exit(
        self, pid: int, proc: subprocess.Popen[bytes] | None, timeout: float
    ) -> bool:
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.2)
        return False

    def stop(
        self,
        timeout: float = 10.0,
        timeout_kill: bool = True,
        shutdown_rpc: str | None = None,
    ) -> dict[str, Any]:
        """Stop the GUI subprocess this bridge launched (own-cleanup use).

        ``shutdown_rpc`` (e.g. ``"app.shutdown"``) is tried first for a clean
        in-app quit if the app exposes one; else an OS signal closes it. Apps
        that observe a user-driven GUI do NOT expose stop as an agent tool —
        this exists for the MCP server's own exit cleanup so a server-launched
        GUI is not orphaned.

        Returns a machine-readable outcome ``{"exited": bool, "note": str}`` so
        callers branch on ``exited`` instead of string-matching the prose
        (Fast-Fail). ``exited`` is true when the process is gone (graceful exit
        or force-kill), false when a graceful close timed out and was left
        running.
        """
        pid, proc = self._pid_for_stop()
        if pid is None:
            self._proc = None
            self.disconnect()
            # No managed process: nothing to exit, so there is no live GUI left.
            return {
                "exited": True,
                "note": "No GUI process managed by this MCP server.",
            }

        if shutdown_rpc is not None and self.is_connected:
            # RPC failure must not prevent the OS-signal cleanup below.
            with suppress(Exception):
                self.send_rpc_raw(shutdown_rpc, {}, 5.0)

        try:
            if proc is not None:
                proc.terminate()
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        exited = self._await_exit(pid, proc, timeout)
        self.disconnect()

        if not exited and timeout_kill:
            try:
                if proc is not None:
                    proc.kill()
                    proc.wait(timeout=5.0)
                else:
                    sig = (
                        signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
                    )
                    os.kill(pid, sig)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                pass
            self._proc = None
            self._clear_pid_file()
            return {
                "exited": True,
                "note": (
                    f"GUI process (pid={pid}) force-killed after graceful close "
                    f"timed out."
                ),
            }

        if not exited:
            return {
                "exited": False,
                "note": (
                    f"SIGTERM sent but GUI (pid={pid}) has not exited within "
                    f"{timeout:.0f}s. Re-run stop, or pass timeout_kill=true to "
                    f"force-kill."
                ),
            }

        self._proc = None
        self._clear_pid_file()
        return {"exited": True, "note": f"GUI process (pid={pid}) closed."}


__all__ = [
    "GuiAuthenticationError",
    "GuiMessageTooLargeError",
    "GuiTransportTimeoutError",
    "McpBridge",
    "MCPBridgeConfig",
    "port_is_open",
    "resolve_connect_port",
]
