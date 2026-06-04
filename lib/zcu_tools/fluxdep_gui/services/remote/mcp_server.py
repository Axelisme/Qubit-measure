#!/usr/bin/env python
"""MCP server bridge for the fluxdep-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live fluxdep GUI's ``RemoteControlAdapter`` over a
single persistent TCP socket. The exposed tools are READ-ONLY: every analysis
method tool is generated 1:1 from the wire-method contract table (``METHOD_SPECS``,
all pure queries — the user drives the GUI); the agent-facing lifecycle tools
(``fluxdep_launch`` / ``fluxdep_connect`` / ``fluxdep_disconnect``) are
hand-written and fork ``script/run_fluxdep_gui.py``. ``tool_fluxdep_stop`` exists
only for the server's own exit cleanup — it is NOT exposed as an agent tool.

Threading:
  - Main (stdio) thread: reads MCP request lines, dispatches into tool handlers,
    writes MCP response lines back.
  - Reader thread: the only reader of the GUI socket; parses NDJSON lines into
    RPC replies (delivered to the matching waiter) or event pushes (dropped —
    the agent uses request/reply, not event subscription).
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Dict, Optional, Tuple

# This bridge is launched standalone (``python .../mcp_server.py``), so the repo
# ``lib`` dir is not on sys.path by default. Add it so the wire-contract modules
# import cleanly.
_LIB_DIR = Path(__file__).resolve().parents[4]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: importing the wire-contract module below pulls qtpy in
# transitively only via the service layer — but method_specs/param_spec/wire are
# Qt-free, so importing just those is safe. We still guard qtpy for the GUI fork.
for _gui_dep in ("qtpy",):
    if importlib.util.find_spec(_gui_dep) is None:
        sys.stderr.write(
            "fluxdep-gui MCP server requires the 'gui' extra (qtpy); "
            f"'{_gui_dep}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n"
        )
        raise SystemExit(1)

from zcu_tools.fluxdep_gui.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.fluxdep_gui.services.remote.param_spec import (  # noqa: E402
    JsonType,
    build_input_schema,
)
from zcu_tools.fluxdep_gui.services.remote.wire import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# This MCP server's own code revision — reported (not compared) in the version
# note so an agent can confirm a reconnect picked up bridge-side edits.
MCP_VERSION = 1

_SERVER_INSTRUCTIONS = """\
Observe a live fluxdep-gui (fluxonium flux-dependence analysis) over a TCP socket.

This bridge is READ-ONLY: the USER drives the analysis in the GUI (load spectra,
pick half/integer flux lines, select spectral points, cross-spectrum filter, run
the database fit, export). The agent's job is to watch and report current state —
there are no load / align / point-pick / select / fit / export tools, because
point-picking and axis-orientation judgement need the user's eye on the preview.

Getting started:
  1. fluxdep_launch opens a GUI subprocess for the user (auto-connects the bridge).
     Or fluxdep_connect to attach to a GUI the user already started.
  2. The user does the analysis in the GUI; you observe it with the read tools.
  fluxdep_disconnect detaches the bridge without stopping the GUI. There is no
  stop tool — the agent never closes the user's GUI.

Read tools (all pure queries):
  - fluxdep_state_check → {has_project, spectrum_count, has_active}.
  - fluxdep_project_info → {chip_name, qub_name, result_dir, database_path}.
  - fluxdep_spectrum_list → each loaded spectrum's {name, spec_type, aligned,
    points_selected} (i.e. how far the user has taken each spectrum).
  - fluxdep_selection_pointcloud → the joint {fluxs, freqs} cloud assembled from
    every spectrum's selected points (freqs in GHz).
  - fluxdep_fit_result → {has_result, params:{EJ,EC,EL} or null, database_path,
    EJb, ECb, ELb, transitions, r_f, sample_f} — the user's fit inputs + result.

A failed call always raises; the read tools are idempotent, so retrying a read is
safe.
"""

# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

_GUI_SOCK_LOCK = threading.Lock()
_GUI_SOCK: Optional[socket.socket] = None

_RID_COND = threading.Condition()
_RID_COUNTER = 0
_PENDING: Dict[str, Dict[str, Any]] = {}

_READER_THREAD: Optional[threading.Thread] = None
_READER_STOP = threading.Event()

_GUI_PROC: Optional[subprocess.Popen] = None
_GUI_PID_FILE = Path(gettempdir()) / "zcu_tools_fluxdep_gui.pid"
_GUI_LOG_FILE = Path(gettempdir()) / "zcu_tools_fluxdep_gui_debug.log"


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


def _write_pid_file(pid: int) -> None:
    try:
        _GUI_PID_FILE.write_text(str(pid))
    except OSError:
        pass


def _read_pid_file() -> Optional[int]:
    try:
        return int(_GUI_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _clear_pid_file() -> None:
    _GUI_PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Socket I/O
# ---------------------------------------------------------------------------


def _next_rid() -> str:
    global _RID_COUNTER
    with _RID_COND:
        _RID_COUNTER += 1
        return f"mcp-{_RID_COUNTER}"


def _send_line(payload: Dict[str, Any]) -> None:
    if _GUI_SOCK is None:
        raise RuntimeError("GUI not connected. Call fluxdep_connect first.")
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with _GUI_SOCK_LOCK:
        _GUI_SOCK.sendall(data)


def _reader_loop() -> None:
    """Sole reader of the GUI socket; routes replies (events are dropped)."""
    buf = bytearray()
    while not _READER_STOP.is_set():
        if _GUI_SOCK is None:
            return
        try:
            chunk = _GUI_SOCK.recv(4096)
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
                _deliver_reply(msg)
            # Event pushes are dropped: the agent drives via request/reply.
    with _RID_COND:
        for holder in _PENDING.values():
            holder["error"] = "GUI socket closed unexpectedly."
            holder["done"] = True
        _RID_COND.notify_all()


def _deliver_reply(msg: Dict[str, Any]) -> None:
    rid = msg.get("id")
    if not isinstance(rid, str):
        return
    with _RID_COND:
        holder = _PENDING.pop(rid, None)
        if holder is None:
            return
        holder["message"] = msg
        holder["done"] = True
        _RID_COND.notify_all()


def _send_gui_rpc_raw(
    method: str, params: Dict[str, Any], timeout_seconds: float
) -> Dict[str, Any]:
    if _GUI_SOCK is None:
        raise RuntimeError("GUI not connected. Call fluxdep_connect first.")
    rid = _next_rid()
    holder: Dict[str, Any] = {"done": False}
    with _RID_COND:
        _PENDING[rid] = holder
    try:
        _send_line({"id": rid, "method": method, "params": params})
    except Exception:
        with _RID_COND:
            _PENDING.pop(rid, None)
        raise

    deadline = time.monotonic() + timeout_seconds
    with _RID_COND:
        while not holder["done"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _PENDING.pop(rid, None)
                raise TimeoutError(
                    f"GUI RPC {method!r} did not complete within {timeout_seconds}s"
                )
            _RID_COND.wait(timeout=remaining)
    if "error" in holder and "message" not in holder:
        raise ConnectionError(holder["error"])
    return holder["message"]


def send_gui_rpc(
    method: str, params: Dict[str, Any], timeout_seconds: float = 30.0
) -> Dict[str, Any]:
    """Issue one RPC against the GUI; raises on error or timeout."""
    resp = _send_gui_rpc_raw(method, params, timeout_seconds)
    if not resp.get("ok", False):
        err = resp.get("error", {})
        msg = f"GUI Error ({err.get('code')}): {err.get('message')}"
        reason = err.get("reason")
        if reason:
            msg += f" (reason: {reason})"
        raise RuntimeError(msg)
    return dict(resp.get("result", {}))


def _wire_version_note() -> str:
    try:
        resp = _send_gui_rpc_raw("wire.version", {}, 5.0)
        result = resp.get("result", {})
        wire_ver = result.get("wire_version")
        gui_ver = result.get("gui_version", "?")
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        return (
            f" versions: mcp wire=v{MCP_WIRE_VERSION} mcp=v{MCP_VERSION}, "
            f"gui=unknown ({exc})"
        )
    if wire_ver != MCP_WIRE_VERSION:
        return (
            f" WIRE VERSION MISMATCH: mcp wire=v{MCP_WIRE_VERSION}, gui wire=v{wire_ver}"
            " — the two processes speak different protocols; restart the stale one."
        )
    return (
        f" wire v{MCP_WIRE_VERSION} (mcp==gui); gui code v{gui_ver}, "
        f"mcp code v{MCP_VERSION}."
    )


# ---------------------------------------------------------------------------
# Connection lifecycle tools
# ---------------------------------------------------------------------------


def tool_fluxdep_connect(arguments: Dict[str, Any]) -> str:
    global _GUI_SOCK, _READER_THREAD
    port = arguments.get("port", 8766)
    if not isinstance(port, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    token = arguments.get("token")

    tool_fluxdep_disconnect({})

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError as exc:
        sock.close()
        raise RuntimeError(
            f"No GUI is listening on 127.0.0.1:{port} ({exc}). fluxdep_connect "
            f"attaches to an already-running GUI; start one with fluxdep_launch."
        ) from exc
    sock.settimeout(1.0)
    _GUI_SOCK = sock
    _READER_STOP.clear()
    _READER_THREAD = threading.Thread(
        target=_reader_loop, name="mcp-fluxdep-reader", daemon=True
    )
    _READER_THREAD.start()

    if token:
        send_gui_rpc("auth", {"token": token})
        return (
            f"Connected to fluxdep-gui on 127.0.0.1:{port} with token auth."
            + _wire_version_note()
        )
    return f"Connected to fluxdep-gui on 127.0.0.1:{port}." + _wire_version_note()


def tool_fluxdep_disconnect(arguments: Dict[str, Any]) -> str:
    global _GUI_SOCK, _READER_THREAD
    del arguments
    if _GUI_SOCK is None:
        return "Not connected."
    _READER_STOP.set()
    try:
        _GUI_SOCK.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        _GUI_SOCK.close()
    except OSError:
        pass
    _GUI_SOCK = None
    t = _READER_THREAD
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _READER_THREAD = None
    return "Disconnected from GUI."


def _port_is_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def tool_fluxdep_launch(arguments: Dict[str, Any]) -> str:
    global _GUI_PROC
    if _GUI_PROC is not None and _GUI_PROC.poll() is None:
        return f"GUI already running (pid={_GUI_PROC.pid})."

    port = int(arguments.get("port", 8766))
    token: Optional[str] = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    repo_root = Path(__file__).parents[5]
    python = sys.executable
    run_gui = repo_root / "script" / "run_fluxdep_gui.py"

    if not run_gui.exists():
        raise FileNotFoundError(f"run_fluxdep_gui.py not found at {run_gui}")

    if _port_is_open(port):
        raise RuntimeError(
            f"Port {port} is already in use — a GUI is likely already running "
            f"there. Use fluxdep_connect to attach to it, or launch on a "
            f"different port."
        )

    cmd = [
        str(python),
        str(run_gui),
        "--control-port",
        str(port),
        "--log-file",
        str(_GUI_LOG_FILE),
    ]
    if token:
        cmd += ["--control-token", token]

    if os.name == "nt":
        _GUI_PROC = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
        )
    else:
        _GUI_PROC = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    _write_pid_file(_GUI_PROC.pid)

    deadline = time.monotonic() + 15.0
    ready = False
    while time.monotonic() < deadline:
        rc = _GUI_PROC.poll()
        if rc is not None:
            stderr = b""
            if _GUI_PROC.stderr is not None:
                stderr = _GUI_PROC.stderr.read() or b""
            tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
            _GUI_PROC = None
            raise RuntimeError(
                f"GUI process exited during startup (returncode={rc}) before "
                f"port {port} was ready. Last stderr:\n" + "\n".join(tail)
            )
        if _port_is_open(port):
            ready = True
            break
        time.sleep(0.3)

    pid = _GUI_PROC.pid
    if not ready:
        return (
            f"GUI launched (pid={pid}) but port {port} not yet reachable — "
            "call fluxdep_connect manually when ready."
        )

    log_note = f" DEBUG log: {_GUI_LOG_FILE}"
    if auto_connect:
        tool_fluxdep_connect(
            {"port": port, "token": token} if token else {"port": port}
        )
        return (
            f"GUI launched (pid={pid}), listening on port {port}, and connected."
            + _wire_version_note()
            + log_note
        )
    return f"GUI launched (pid={pid}) and listening on port {port}." + log_note


def _pid_for_stop() -> Tuple[Optional[int], Optional[subprocess.Popen]]:
    proc = _GUI_PROC
    if proc is not None and proc.poll() is None:
        return proc.pid, proc
    return _read_pid_file(), None


def _await_exit(pid: int, proc: Optional[subprocess.Popen], timeout: float) -> bool:
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


def tool_fluxdep_stop(arguments: Dict[str, Any]) -> str:
    global _GUI_PROC
    pid, proc = _pid_for_stop()
    if pid is None:
        _GUI_PROC = None
        tool_fluxdep_disconnect({})
        return "No GUI process managed by this MCP server."

    timeout = float(arguments.get("timeout", 10.0))
    timeout_kill = bool(arguments.get("timeout_kill", True))

    # No app.shutdown RPC for fluxdep; close via OS signal (process group). Send
    # SIGTERM first for a clean Qt quit (aboutToQuit stops the adapter).
    try:
        if proc is not None:
            proc.terminate()
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass

    exited = _await_exit(pid, proc, timeout)
    tool_fluxdep_disconnect({})

    if not exited and timeout_kill:
        try:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=5.0)
            else:
                sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
                os.kill(pid, sig)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
        _GUI_PROC = None
        _clear_pid_file()
        return f"GUI process (pid={pid}) force-killed after graceful close timed out."

    if not exited:
        return (
            f"SIGTERM sent but GUI (pid={pid}) has not exited within {timeout:.0f}s. "
            f"Re-run fluxdep_stop, or pass timeout_kill=true to force-kill."
        )

    _GUI_PROC = None
    _clear_pid_file()
    return f"GUI process (pid={pid}) closed."


# ---------------------------------------------------------------------------
# Generated tools — derived from METHOD_SPECS (the wire SSOT)
# ---------------------------------------------------------------------------

# Methods that must NOT be auto-generated as agent tools.
_NON_GENERATED_METHODS = frozenset(
    {
        # mcp<->RPC bookkeeping only; version numbers must not surface to the agent.
        "resources.versions",
    }
)


def _tool_name_for(method: str, spec) -> str:
    return spec.tool_name or "fluxdep_" + method.replace(".", "_")


def _coerce_arg(value: object, json_type: "JsonType") -> object:
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


def _make_forwarder(method: str, spec):
    """Build an MCP forwarder that projects arguments into RPC params per spec."""
    rpc_timeout = max(float(spec.timeout_seconds), 30.0)

    def _forwarder(arguments: Dict[str, Any]) -> Dict[str, Any]:
        rpc_params: Dict[str, Any] = {}
        for p in spec.params:
            if p.required:
                if p.name not in arguments or arguments[p.name] is None:
                    raise ValueError(f"missing {p.name!r}")
                rpc_params[p.name] = _coerce_arg(arguments[p.name], p.json_type)
            elif arguments.get(p.name) is not None:
                rpc_params[p.name] = _coerce_arg(arguments[p.name], p.json_type)
        return send_gui_rpc(method, rpc_params, timeout_seconds=rpc_timeout)

    return _forwarder


def _generate_tools() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for method, spec in METHOD_SPECS.items():
        if method in _NON_GENERATED_METHODS:
            continue
        tool_name = _tool_name_for(method, spec)
        out[tool_name] = {
            "handler": _make_forwarder(method, spec),
            "description": spec.description or method,
            "inputSchema": build_input_schema(spec.params),
        }
    return out


# ---------------------------------------------------------------------------
# Hand-written lifecycle tools
# ---------------------------------------------------------------------------


_OVERRIDE_TOOLS: Dict[str, Dict[str, Any]] = {
    "fluxdep_connect": {
        "handler": tool_fluxdep_connect,
        "description": (
            "Connect the MCP bridge to an ALREADY-RUNNING fluxdep-gui's TCP "
            "control port (default 8766). Errors if no GUI is listening there — "
            "use fluxdep_launch to start one. Skip this if you used fluxdep_launch "
            "with auto_connect=true (default)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP port of a running GUI control service (default 8766)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
        },
    },
    "fluxdep_disconnect": {
        "handler": tool_fluxdep_disconnect,
        "description": (
            "Disconnect the MCP bridge from the GUI control port. Does NOT stop "
            "the GUI process — it keeps running for the user to drive."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "fluxdep_launch": {
        "handler": tool_fluxdep_launch,
        "description": (
            "Launch the fluxdep-gui as a NEW subprocess on a TCP control port "
            "(default 8766), wait until ready, and optionally connect. Use as the "
            "first step. Errors if the port is already in use (a stale GUI). By "
            "default auto_connect=true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP control port for the GUI (default 8766)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional shared auth token",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Call fluxdep_connect automatically once ready (default true)",
                },
            },
        },
    },
    # NOTE: there is deliberately no "fluxdep_stop" entry — see _OVERRIDE_NAMES.
}

# fluxdep_stop is intentionally NOT exposed as an agent tool: the agent observes a
# GUI the user drives and must not kill the user's GUI. tool_fluxdep_stop is kept
# as a function for the MCP server's own _cleanup_on_exit (so a server-launched
# GUI is not orphaned when the server process ends), but it is not registered here.
_OVERRIDE_NAMES = frozenset({"fluxdep_connect", "fluxdep_disconnect", "fluxdep_launch"})


def _assemble_tools() -> Dict[str, Dict[str, Any]]:
    generated = _generate_tools()
    overrides = {
        name: spec for name, spec in _OVERRIDE_TOOLS.items() if name in _OVERRIDE_NAMES
    }
    collisions = set(generated) & set(overrides)
    if collisions:
        raise RuntimeError(f"override/generated tool collision: {sorted(collisions)}")
    return {**generated, **overrides}


TOOLS: Dict[str, Dict[str, Any]] = _assemble_tools()


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def _cleanup_on_exit() -> None:
    try:
        tool_fluxdep_stop({"timeout_kill": True})
    except Exception:
        pass


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                _cleanup_on_exit()
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
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "fluxdep-gui-control",
                            "version": "1.0.0",
                        },
                        "instructions": _SERVER_INSTRUCTIONS,
                    },
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
                    for name, info in TOOLS.items()
                ]
                resp = {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools_list}}
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})

                tool = TOOLS.get(name)
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
                        handler: Callable[[Dict[str, Any]], Any] = tool["handler"]
                        res = handler(arguments)
                        text = (
                            res if isinstance(res, str) else json.dumps(res, indent=2)
                        )
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {"content": [{"type": "text", "text": text}]},
                        }
                    except Exception as e:
                        if isinstance(e, RuntimeError):
                            text = f"Error executing tool {name!r}: {e}"
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
            sys.stderr.write(f"MCP Loop Exception: {e}\n{traceback.format_exc()}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
