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
  - :class:`McpServerConfig` carries the prefix + display name + instructions the
    stdio loop / tool-gen need; :class:`MCPBridgeConfig` adds the GUI-bridge launch
    knobs (name / port / versions / pid+log file names / run-script name).
  - Module helpers build the MCP tool surface from a method-spec table
    (``coerce_arg`` / ``make_forwarder`` / ``generate_tools``) and run the MCP
    stdio protocol loop (``build_initialize_result`` / ``run_stdio_loop``).

App-specific policy stays in each ``mcp_server.py``: the read-only apps wrap
``send_rpc_raw`` in a thin error-raising ``send_gui_rpc`` and drop events;
measure-gui composes ``send_rpc_raw`` with its optimistic-concurrency guard,
operation tracking, the diagnostic queue (via ``on_event``), and its hand-written
tools.

Threading:
  - Main (stdio) thread: reads MCP request lines, dispatches into tool handlers,
    writes MCP response lines back.
  - Reader thread (per :class:`McpBridge`): the only reader of the GUI socket;
    parses NDJSON lines into RPC replies (delivered to the matching waiter) or
    event pushes (handed to ``on_event``, or dropped if None).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from zcu_tools.gui.remote.param_spec import JsonType, build_input_schema

# The type of a generated/override MCP tool entry.
Tool = dict[str, Any]
ToolTable = dict[str, Tool]
# A function issuing one GUI RPC and returning the result dict (raises on error).
# Read-only apps pass a thin wrapper over McpBridge.send_rpc_raw; measure-gui
# passes its guarded send_gui_rpc.
SendFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class McpServerConfig:
    """The minimal config the stdio loop + tool generation need.

    ``tool_prefix`` is the wire-method -> tool-name prefix (e.g. ``fluxdep_``). A
    no-subprocess server (e.g. agent-memory, dispatching in-process) needs only
    this; the GUI bridges add the launch fields via :class:`MCPBridgeConfig`.
    """

    tool_prefix: str
    server_display_name: str
    server_instructions: str


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


def _port_is_open(port: int) -> bool:
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
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(code)
            ):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


# A line arrived from the GUI: route it (reply keyed by id / event push).
DeliverFn = Callable[[dict[str, Any]], None]
# The transport closed (real socket dropped): wake any pending RPC waiters.
OnClosedFn = Callable[[], None]


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
        self._sock = sock
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name=f"mcp-{self._app_name}-reader", daemon=True
        )
        self._reader_thread.start()

    def send_line(self, payload: dict[str, Any]) -> None:
        sock = self._sock
        if sock is None:
            raise RuntimeError("transport not open")
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with self._sock_lock:
            sock.sendall(data)

    def close(self) -> None:
        sock = self._sock
        if sock is None:
            return
        self._reader_stop.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        self._sock = None
        t = self._reader_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._reader_thread = None

    def _reader_loop(self) -> None:
        """Sole reader of the GUI socket; routes replies, hands events to hook."""
        buf = bytearray()
        while not self._reader_stop.is_set():
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
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(msg, dict) and "id" in msg:
                    if self._deliver_reply is not None:
                        self._deliver_reply(msg)
                elif isinstance(msg, dict) and "event" in msg:
                    if self._deliver_event is not None:
                        self._deliver_event(msg)
                    # else: event pushes are dropped (read-only apps).
        # Socket dropped: let the bridge wake any pending RPC waiters.
        if self._on_closed is not None:
            self._on_closed()


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
        self._proc: subprocess.Popen | None = None
        # The wire transport. None until connect()/launch() builds a real
        # SocketTransport, or a test injects a fake via set_transport / the ctor.
        self._transport: Transport | None = None
        if transport is not None:
            self.set_transport(transport)

    def set_transport(self, transport: Transport | None) -> None:
        """Swap the wire transport (the seam for tests + connect()).

        Attaching wires the bridge's routing callbacks into the transport; the
        bridge keeps the pending map / RID condition. Passing None detaches.
        """
        self._transport = transport
        if transport is not None:
            transport.attach(
                self._deliver_reply, self._deliver_event, self._on_socket_closed
            )

    def _deliver_event(self, msg: dict[str, Any]) -> None:
        # Preserve the drop-if-None semantics: read-only apps wire no on_event.
        if self._on_event is not None:
            self._on_event(msg)

    def _on_socket_closed(self) -> None:
        # The reader thread saw the socket drop: wake every pending RPC waiter so
        # callers see "disconnected" instead of blocking to their timeout.
        with self._rid_cond:
            for holder in self._pending.values():
                holder["error"] = "GUI socket closed unexpectedly."
                holder["done"] = True
            self._rid_cond.notify_all()

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_open

    # ------------------------------------------------------------------
    # pid file
    # ------------------------------------------------------------------

    def _write_pid_file(self, pid: int) -> None:
        try:
            self.config.pid_file.write_text(str(pid))
        except OSError:
            pass

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

    def send_rpc_raw(
        self, method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Issue one RPC and wait for its reply (no policy). Raises on timeout.

        Returns the raw wire response dict (``{ok, result}`` or ``{ok:False,
        error}``). App layers wrap this to raise on ``ok:False`` and add policy.
        """
        transport = self._transport
        if transport is None or not transport.is_open:
            raise RuntimeError(
                f"GUI not connected. Call {self.config.tool_prefix}connect first."
            )
        rid = self._next_rid()
        holder: dict[str, Any] = {"done": False}
        with self._rid_cond:
            self._pending[rid] = holder
        try:
            transport.send_line({"id": rid, "method": method, "params": params})
        except Exception:
            with self._rid_cond:
                self._pending.pop(rid, None)
            raise

        deadline = time.monotonic() + timeout_seconds
        with self._rid_cond:
            while not holder["done"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending.pop(rid, None)
                    raise TimeoutError(
                        f"GUI RPC {method!r} did not complete within {timeout_seconds}s"
                    )
                self._rid_cond.wait(timeout=remaining)
        if "error" in holder and "message" not in holder:
            raise ConnectionError(holder["error"])
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
                err = resp.get("error", {})
                raise RuntimeError(
                    f"GUI auth failed ({err.get('code')}): {err.get('message')}"
                )
            return (
                f"Connected to {cfg.server_display_name} on 127.0.0.1:{port} "
                f"with token auth." + self.wire_version_note()
            )
        return (
            f"Connected to {cfg.server_display_name} on 127.0.0.1:{port}."
            + self.wire_version_note()
        )

    def disconnect(self) -> str:
        transport = self._transport
        if transport is None or not transport.is_open:
            return "Not connected."
        transport.close()
        self._transport = None
        return "Disconnected from GUI."

    def launch(
        self,
        repo_root: Path,
        port: int,
        token: str | None = None,
        auto_connect: bool = True,
        extra_args: list | None = None,
    ) -> str:
        """Fork the GUI subprocess on ``port``, wait until ready, maybe connect.

        ``repo_root`` anchors ``script/<run_script_name>``. ``extra_args`` are
        appended to the launch command (apps that need extra flags).
        """
        cfg = self.config
        if self._proc is not None and self._proc.poll() is None:
            return f"GUI already running (pid={self._proc.pid})."

        python = sys.executable
        run_gui = repo_root / "script" / cfg.run_script_name
        if not run_gui.exists():
            raise FileNotFoundError(f"{cfg.run_script_name} not found at {run_gui}")

        if _port_is_open(port):
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

        if os.name == "nt":
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
            )
        else:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        self._write_pid_file(self._proc.pid)

        deadline = time.monotonic() + 15.0
        ready = False
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                stderr = b""
                if self._proc.stderr is not None:
                    stderr = self._proc.stderr.read() or b""
                tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
                self._proc = None
                raise RuntimeError(
                    f"GUI process exited during startup (returncode={rc}) before "
                    f"port {port} was ready. Last stderr:\n" + "\n".join(tail)
                )
            if _port_is_open(port):
                ready = True
                break
            time.sleep(0.3)

        pid = self._proc.pid
        if not ready:
            return (
                f"GUI launched (pid={pid}) but port {port} not yet reachable — "
                f"call {cfg.tool_prefix}connect manually when ready."
            )

        log_note = f" DEBUG log: {cfg.log_file}"
        if auto_connect:
            self.connect(port, token)
            return (
                f"GUI launched (pid={pid}), listening on port {port}, and connected."
                + self.wire_version_note()
                + log_note
            )
        return f"GUI launched (pid={pid}) and listening on port {port}." + log_note

    def _pid_for_stop(self) -> tuple[int | None, subprocess.Popen | None]:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            return proc.pid, proc
        return self._read_pid_file(), None

    def _await_exit(
        self, pid: int, proc: subprocess.Popen | None, timeout: float
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
    ) -> str:
        """Stop the GUI subprocess this bridge launched (own-cleanup use).

        ``shutdown_rpc`` (e.g. ``"app.shutdown"``) is tried first for a clean
        in-app quit if the app exposes one; else an OS signal closes it. Apps
        that observe a user-driven GUI do NOT expose stop as an agent tool —
        this exists for the MCP server's own exit cleanup so a server-launched
        GUI is not orphaned.
        """
        pid, proc = self._pid_for_stop()
        if pid is None:
            self._proc = None
            self.disconnect()
            return "No GUI process managed by this MCP server."

        if shutdown_rpc is not None and self.is_connected:
            try:
                self.send_rpc_raw(shutdown_rpc, {}, 5.0)
            except Exception:  # noqa: BLE001 — fall through to signal
                pass

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
            return (
                f"GUI process (pid={pid}) force-killed after graceful close timed out."
            )

        if not exited:
            return (
                f"SIGTERM sent but GUI (pid={pid}) has not exited within "
                f"{timeout:.0f}s. Re-run stop, or pass timeout_kill=true to "
                f"force-kill."
            )

        self._proc = None
        self._clear_pid_file()
        return f"GUI process (pid={pid}) closed."


# ---------------------------------------------------------------------------
# Tool generation from a method-spec table (the wire SSOT)
# ---------------------------------------------------------------------------


def coerce_arg(value: object, json_type: JsonType) -> object:
    if value is None:
        return None
    if json_type is JsonType.STRING:
        return str(value)
    if json_type is JsonType.INTEGER:
        return int(value)  # type: ignore[arg-type]
    if json_type is JsonType.NUMBER:
        return float(value)  # type: ignore[arg-type]
    if json_type is JsonType.BOOLEAN:
        return bool(value)
    if json_type is JsonType.OBJECT:
        return dict(value)  # type: ignore[call-overload]
    return value  # JSON: pass through


def make_forwarder(method: str, spec, send_fn: SendFn):
    """Build an MCP forwarder that projects arguments into RPC params per spec.

    ``send_fn`` issues the RPC: read-only apps pass a thin error-raising wrapper
    over :meth:`McpBridge.send_rpc_raw`; measure-gui passes its guarded
    ``send_gui_rpc``.
    """
    rpc_timeout = max(float(spec.timeout_seconds), 30.0)

    def _forwarder(arguments: dict[str, Any]) -> dict[str, Any]:
        rpc_params: dict[str, Any] = {}
        for p in spec.params:
            if p.required:
                if p.name not in arguments or arguments[p.name] is None:
                    raise ValueError(f"missing {p.name!r}")
                rpc_params[p.name] = coerce_arg(arguments[p.name], p.json_type)
            elif arguments.get(p.name) is not None:
                rpc_params[p.name] = coerce_arg(arguments[p.name], p.json_type)
        return send_fn(method, rpc_params, timeout_seconds=rpc_timeout)

    return _forwarder


def generate_tools(
    config: McpServerConfig,
    method_specs: dict[str, Any],
    non_generated: frozenset[str],
    send_fn: SendFn,
) -> ToolTable:
    """Generate one MCP tool per method spec (skipping ``non_generated``)."""
    out: ToolTable = {}
    for method, spec in method_specs.items():
        if method in non_generated:
            continue
        tool_name = spec.tool_name or config.tool_prefix + method.replace(".", "_")
        out[tool_name] = {
            "handler": make_forwarder(method, spec, send_fn),
            "description": spec.description or method,
            "inputSchema": build_input_schema(spec.params),
        }
    return out


def assemble_tools(
    generated: ToolTable, overrides: ToolTable, override_names: frozenset[str]
) -> ToolTable:
    """Merge generated + selected override tools; fail-fast on name collision."""
    selected = {
        name: spec for name, spec in overrides.items() if name in override_names
    }
    collisions = set(generated) & set(selected)
    if collisions:
        raise RuntimeError(f"override/generated tool collision: {sorted(collisions)}")
    return {**generated, **selected}


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def build_initialize_result(
    config: McpServerConfig, server_version: str = "1.0.0"
) -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": config.server_display_name, "version": server_version},
        "instructions": config.server_instructions,
    }


def run_stdio_loop(
    config: McpServerConfig,
    tools: ToolTable,
    *,
    on_cleanup: Callable[[], None] | None = None,
    on_each_reply: Callable[[], list[dict[str, Any]]] | None = None,
    on_start: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    server_version: str = "1.0.0",
) -> None:
    """Run the MCP stdio JSON-RPC loop until stdin closes.

    Hooks (all optional; defaults give the bare read-only behaviour):
      - ``on_start`` runs once after stdin/stdout are reconfigured to UTF-8, before
        the loop (measure-gui attaches its per-session file logging here).
      - ``on_cleanup`` runs once when stdin closes (e.g. stop a server-launched GUI).
      - ``on_each_reply`` (measure-gui) returns ready-made content blocks to
        piggyback on every successful tool reply (e.g. drained diagnostics): a list
        of ``{"type": "text", "text": ...}`` dicts, each appended after the tool's
        own content. The hook owns the wording (returns ``[]`` for nothing).
      - ``on_error`` is called from within each ``except`` block with a
        preformatted context message (measure-gui passes ``logger.exception``) so
        the active exception is logged with its traceback.

    ``server_version`` is the ``serverInfo.version`` reported on ``initialize``.

    A ``RuntimeError`` carrying a ``reason`` attribute (set from the GUI wire error
    envelope) has its tag appended to the tool-error text, so an agent can branch
    on the machine-readable reason without parsing the prose.
    """
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if on_start is not None:
        on_start()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                if on_cleanup is not None:
                    on_cleanup()
                break
            line = line.strip()
            if not line:
                continue

            req = json.loads(line)
            method = req.get("method")
            rid = req.get("id")

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": build_initialize_result(config, server_version),
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

            elif method == "notifications/initialized":
                continue

            elif method == "tools/list":
                tools_list = [
                    {
                        "name": name,
                        "description": info["description"],
                        "inputSchema": info["inputSchema"],
                    }
                    for name, info in tools.items()
                ]
                resp = {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools_list}}
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})

                tool = tools.get(name)
                if not tool:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {name}",
                        },
                    }
                else:
                    try:
                        handler: Callable[[dict[str, Any]], Any] = tool["handler"]
                        res = handler(arguments)
                        text = (
                            res if isinstance(res, str) else json.dumps(res, indent=2)
                        )
                        content = [{"type": "text", "text": text}]
                        if on_each_reply is not None:
                            content.extend(on_each_reply())
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {"content": content},
                        }
                    except Exception as e:
                        if on_error is not None:
                            on_error(f"MCP tool {name!r} dispatch failed")
                        # GUI-side business errors (RuntimeError with an already-
                        # clear message) carry no useful Python stack for the agent
                        # — the traceback is always the same forwarder frames, pure
                        # noise. Strip it for those; keep the full traceback only for
                        # unexpected bridge-side failures, where the stack is the
                        # actual debugging signal.
                        if isinstance(e, RuntimeError):
                            text = f"Error executing tool {name!r}: {e}"
                            # Surface the machine-readable reason tag (e.g.
                            # no_run_result / no_project) when the wire carried one,
                            # so the agent can branch on it without parsing prose.
                            reason = getattr(e, "reason", None)
                            if reason:
                                text += f"\nreason: {reason}"
                        else:
                            text = (
                                f"Error executing tool {name!r}: {e}\n"
                                f"{traceback.format_exc()}"
                            )
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {
                                "isError": True,
                                "content": [{"type": "text", "text": text}],
                            },
                        }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            else:
                if rid is not None:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}",
                        },
                    }
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
        except Exception as e:
            if on_error is not None:
                on_error("MCP loop exception")
            sys.stderr.write(f"MCP Loop Exception: {e}\n{traceback.format_exc()}\n")
            sys.stderr.flush()


__all__ = [
    "McpBridge",
    "MCPBridgeConfig",
    "McpServerConfig",
    "assemble_tools",
    "build_initialize_result",
    "coerce_arg",
    "generate_tools",
    "make_forwarder",
    "run_stdio_loop",
]
