#!/usr/bin/env python
"""MCP server bridge for ``RemoteControlService``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live GUI's ``RemoteControlService`` over a
single persistent TCP socket. Event push from the GUI is received by a
dedicated reader thread, parked in an internal queue and exposed to the LLM
via the ``gui_events_poll`` polling tool.

Threading:
  - Main (stdio) thread: reads MCP request lines, dispatches into tool
    handlers, writes MCP response lines back. Concurrency with the reader
    thread is mediated by a single ``threading.Lock`` covering all writes
    and request-id state.
  - Reader thread: the **only** reader of the GUI socket. Parses NDJSON
    lines into either RPC replies (delivered to the matching waiter via a
    ``threading.Condition``) or event pushes (appended to an in-memory
    queue capped at 1024 entries).
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
from collections import deque
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# Connection state (module-level so tools are thin wrappers)
# ---------------------------------------------------------------------------

_GUI_SOCK_LOCK = threading.Lock()
_GUI_SOCK: Optional[socket.socket] = None

# RPC bookkeeping — ``_PENDING`` maps rid -> result holder; the reader thread
# unblocks the matching waiter via ``_RID_COND``.
_RID_COND = threading.Condition()
_RID_COUNTER = 0
_PENDING: Dict[str, Dict[str, Any]] = {}

# Event push queue (FIFO, drop-oldest when full).
_EVENT_QUEUE_MAX = 1024
_EVENT_QUEUE: Deque[Dict[str, Any]] = deque(maxlen=_EVENT_QUEUE_MAX)
_EVENT_COND = threading.Condition()

_READER_THREAD: Optional[threading.Thread] = None
_READER_STOP = threading.Event()

# GUI subprocess (when launched via gui_launch tool).
_GUI_PROC: Optional[subprocess.Popen] = None

# PID file for cross-session GUI process tracking.
_GUI_PID_FILE = Path(gettempdir()) / "zcu_tools_gui.pid"


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
    """Send a single NDJSON line on the GUI socket (lock-guarded)."""
    if _GUI_SOCK is None:
        raise RuntimeError("GUI not connected. Call gui_connect first.")
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with _GUI_SOCK_LOCK:
        _GUI_SOCK.sendall(data)


def _reader_loop() -> None:
    """Sole reader of the GUI socket; routes replies vs event pushes."""
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
            elif isinstance(msg, dict) and "event" in msg:
                _deliver_event(msg)
            # Anything else is silently dropped.
    # Wake everyone so callers see "disconnected".
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


def _deliver_event(msg: Dict[str, Any]) -> None:
    with _EVENT_COND:
        _EVENT_QUEUE.append(msg)
        _EVENT_COND.notify_all()


def send_gui_rpc(
    method: str,
    params: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Issue one RPC against the GUI; raises on error or timeout."""
    if _GUI_SOCK is None:
        raise RuntimeError("GUI not connected. Call gui_connect first.")
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
    resp = holder["message"]
    if not resp.get("ok", False):
        err = resp.get("error", {})
        raise RuntimeError(f"GUI Error ({err.get('code')}): {err.get('message')}")
    return resp.get("result", {})


# ---------------------------------------------------------------------------
# Connection lifecycle tools
# ---------------------------------------------------------------------------


def tool_gui_connect(arguments: Dict[str, Any]) -> str:
    global _GUI_SOCK, _READER_THREAD
    port = arguments.get("port")
    if port is None or not isinstance(port, int):
        raise ValueError("Missing or invalid 'port' argument (must be integer)")
    token = arguments.get("token")

    # Tear down any existing connection / reader before replacing it.
    tool_gui_disconnect({})

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(1.0)  # short blocking timeout for reader loop responsiveness
    _GUI_SOCK = sock
    _READER_STOP.clear()
    _READER_THREAD = threading.Thread(
        target=_reader_loop, name="mcp-gui-reader", daemon=True
    )
    _READER_THREAD.start()

    if token:
        send_gui_rpc("auth", {"token": token})
        return f"Connected to GUI on 127.0.0.1:{port} with token authentication."
    return f"Connected to GUI on 127.0.0.1:{port}."


def tool_gui_disconnect(arguments: Dict[str, Any]) -> str:
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
    with _EVENT_COND:
        _EVENT_QUEUE.clear()
    return "Disconnected from GUI."


def tool_gui_launch(arguments: Dict[str, Any]) -> str:
    global _GUI_PROC
    if _GUI_PROC is not None and _GUI_PROC.poll() is None:
        return f"GUI already running (pid={_GUI_PROC.pid})."

    port = int(arguments.get("port", 8765))
    token: Optional[str] = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    repo_root = Path(__file__).parents[
        5
    ]  # lib/zcu_tools/gui/services/remote -> repo root
    python = repo_root / ".venv" / "bin" / "python"
    run_gui = repo_root / "run_gui.py"

    if not run_gui.exists():
        raise FileNotFoundError(f"run_gui.py not found at {run_gui}")

    cmd = [str(python), str(run_gui), "--control-port", str(port), "--no-log"]
    if token:
        cmd += ["--control-token", token]

    _GUI_PROC = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_pid_file(_GUI_PROC.pid)

    # Wait until the port is listening (up to 15 s).
    deadline = time.monotonic() + 15.0
    ready = False
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close()
            ready = True
            break
        except OSError:
            time.sleep(0.3)

    pid = _GUI_PROC.pid
    if not ready:
        return (
            f"GUI launched (pid={pid}) but port {port} not yet reachable — "
            "call gui_connect manually when ready."
        )

    if auto_connect:
        tool_gui_connect({"port": port, "token": token} if token else {"port": port})
        return f"GUI launched (pid={pid}), listening on port {port}, and connected."

    return f"GUI launched (pid={pid}) and listening on port {port}."


def tool_gui_stop(arguments: Dict[str, Any]) -> str:
    global _GUI_PROC
    tool_gui_disconnect({})

    proc = _GUI_PROC
    pid: Optional[int] = None

    if proc is not None and proc.poll() is None:
        pid = proc.pid
    else:
        # Fallback: recover pid from file (handles cross-session restarts).
        pid = _read_pid_file()
        if pid is None:
            _GUI_PROC = None
            return "No GUI process managed by this MCP server."
        proc = None  # no Popen object available

    force = bool(arguments.get("force", False))
    sig = signal.SIGKILL if force else (
        signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT
    )
    try:
        os.kill(pid, sig)
        if proc is not None:
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.3)
                except ProcessLookupError:
                    break
    except ProcessLookupError:
        pass

    _GUI_PROC = None
    _clear_pid_file()
    return f"GUI process (pid={pid}) stopped."


def tool_gui_connect_mock(arguments: Dict[str, Any]) -> str:
    del arguments
    # 1. connect.start
    send_gui_rpc("connect.start", {"kind": "mock"})

    # 2. Apply startup project parameters (same as driver.py connect-mock)
    repo_root = Path.cwd()
    chip = "Q1_Chip"
    qub = "Q1"
    res = "R1"
    result_dir = str(repo_root / "result")
    db_path = str(repo_root / "Database")

    send_gui_rpc(
        "startup.apply",
        {
            "chip_name": chip,
            "qub_name": qub,
            "res_name": res,
            "result_dir": result_dir,
            "database_path": db_path,
        },
    )

    # 3. Wait for has_soc to become true
    connected = False
    for _ in range(50):
        time.sleep(0.1)
        check = send_gui_rpc("state.has_soc", {})
        if check.get("value", False):
            connected = True
            break

    if not connected:
        raise RuntimeError("Mock SOC failed to connect within timeout.")

    # 4. Check context labels and use first one, or create new
    labels_res = send_gui_rpc("context.labels", {})
    labels = labels_res.get("labels", [])
    if labels:
        send_gui_rpc("context.use", {"label": labels[0]})
        active_label = labels[0]
    else:
        send_gui_rpc("context.new", {"value": 1.0, "unit": "A"})
        active_res = send_gui_rpc("context.active", {})
        active_label = active_res.get("label", "new")

    return f"Mock SOC connected and startup applied. Active context set to: {active_label!r}"


# ---------------------------------------------------------------------------
# Workflow tools (thin pass-through wrappers)
# ---------------------------------------------------------------------------


def tool_gui_startup_apply(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params = {
        "chip_name": str(arguments["chip_name"]),
        "qub_name": str(arguments["qub_name"]),
        "res_name": str(arguments["res_name"]),
        "result_dir": str(arguments["result_dir"]),
        "database_path": str(arguments["database_path"]),
    }
    return send_gui_rpc("startup.apply", params)


def tool_gui_adapter_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("adapter.list", {})


def tool_gui_connect_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"kind": str(arguments["kind"])}
    if params["kind"] == "remote":
        params["ip"] = str(arguments["ip"])
        params["port"] = int(arguments["port"])
    return send_gui_rpc("connect.start", params)


def tool_gui_tab_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("tab.list", {})


def tool_gui_tab_new(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("tab.new", {"adapter_name": str(arguments["adapter_name"])})


def tool_gui_tab_close(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("tab.close", {"tab_id": str(arguments["tab_id"])})


def tool_gui_tab_set_active(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("tab.set_active", {"tab_id": str(arguments["tab_id"])})


def tool_gui_tab_snapshot(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("tab.snapshot", {"tab_id": str(arguments["tab_id"])})


def tool_gui_tab_get_cfg(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("tab.get_cfg", {"tab_id": str(arguments["tab_id"])})


def tool_gui_tab_update_cfg(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc(
        "tab.update_cfg",
        {"tab_id": str(arguments["tab_id"]), "raw": dict(arguments["raw"])},
    )


def tool_gui_run_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("run.start", {"tab_id": str(arguments["tab_id"])})


def tool_gui_run_cancel(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("run.cancel", {})


def tool_gui_run_running_tab(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("run.running_tab", {})


def tool_gui_save_both(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"tab_id": str(arguments["tab_id"])}
    if arguments.get("data_path") is not None:
        params["data_path"] = str(arguments["data_path"])
    if arguments.get("image_path") is not None:
        params["image_path"] = str(arguments["image_path"])
    if arguments.get("comment") is not None:
        params["comment"] = str(arguments["comment"])
    return send_gui_rpc("save.both", params)


def tool_gui_save_data(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"tab_id": str(arguments["tab_id"])}
    if arguments.get("data_path") is not None:
        params["data_path"] = str(arguments["data_path"])
    if arguments.get("comment") is not None:
        params["comment"] = str(arguments["comment"])
    return send_gui_rpc("save.data", params)


def tool_gui_save_image(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"tab_id": str(arguments["tab_id"])}
    if arguments.get("image_path") is not None:
        params["image_path"] = str(arguments["image_path"])
    return send_gui_rpc("save.image", params)


def tool_gui_state_check(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    has_proj = send_gui_rpc("state.has_project", {}).get("value", False)
    has_ctx = send_gui_rpc("state.has_context", {}).get("value", False)
    has_act = send_gui_rpc("state.has_active_context", {}).get("value", False)
    has_soc = send_gui_rpc("state.has_soc", {}).get("value", False)
    return {
        "has_project": has_proj,
        "has_context": has_ctx,
        "has_active_context": has_act,
        "has_soc": has_soc,
    }


def tool_gui_context_labels(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("context.labels", {})


def tool_gui_context_active(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("context.active", {})


def tool_gui_context_use(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("context.use", {"label": str(arguments["label"])})


def tool_gui_context_new(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if "value" in arguments and arguments["value"] is not None:
        params["value"] = float(arguments["value"])
    if "unit" in arguments and arguments["unit"] is not None:
        params["unit"] = str(arguments["unit"])
    if "clone_from_current" in arguments:
        params["clone_from_current"] = bool(arguments["clone_from_current"])
    return send_gui_rpc("context.new", params)


def tool_gui_session_persist(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("session.persist", {})


def tool_gui_session_restore(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("session.restore", {})


# ---------------------------------------------------------------------------
# Phase 81a tools — events / dialog / view
# ---------------------------------------------------------------------------


def _coerce_str_list(value: object, *, field: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{field!r} must be a list of strings")
    return list(value)


def tool_gui_events_subscribe(arguments: Dict[str, Any]) -> Dict[str, Any]:
    events = _coerce_str_list(arguments.get("events", []), field="events")
    return send_gui_rpc("events.subscribe", {"events": events})


def tool_gui_events_unsubscribe(arguments: Dict[str, Any]) -> Dict[str, Any]:
    events = _coerce_str_list(arguments.get("events", []), field="events")
    return send_gui_rpc("events.unsubscribe", {"events": events})


def tool_gui_events_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("events.list", {})


def tool_gui_events_poll(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Drain up to ``max_events`` queued event pushes, blocking briefly."""
    timeout_seconds = float(arguments.get("timeout_seconds", 5.0))
    max_events = int(arguments.get("max_events", 16))
    if max_events < 1:
        max_events = 1
    out: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    with _EVENT_COND:
        while len(out) < max_events:
            while _EVENT_QUEUE and len(out) < max_events:
                out.append(_EVENT_QUEUE.popleft())
            if out:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _EVENT_COND.wait(timeout=remaining)
    return {"events": out}


def tool_gui_dialog_open(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("dialog.open", {"name": str(arguments["name"])})


def tool_gui_dialog_close(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("dialog.close", {"name": str(arguments["name"])})


def tool_gui_dialog_list_open(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("dialog.list_open", {})


def tool_gui_view_snapshot(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("view.snapshot", {})


def tool_gui_view_screenshot(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Capture window/tab as PNG; optionally write to ``out_path`` and strip b64."""
    params: Dict[str, Any] = {}
    if "tab_id" in arguments and arguments["tab_id"] is not None:
        params["tab_id"] = str(arguments["tab_id"])
    res = send_gui_rpc("view.screenshot", params)
    out_path = arguments.get("out_path")
    if out_path:
        import base64

        png = base64.b64decode(res["png_b64"])
        Path(out_path).write_bytes(png)
        res = {
            "bytes": res.get("bytes", len(png)),
            "saved_to": str(out_path),
        }
    return res


# ---------------------------------------------------------------------------
# Phase 81b tools — cfg.set_field / context queries / device queries
# ---------------------------------------------------------------------------


def tool_gui_cfg_set_field(arguments: Dict[str, Any]) -> Dict[str, Any]:
    if "value" not in arguments:
        raise ValueError("missing 'value'")
    return send_gui_rpc(
        "cfg.set_field",
        {
            "tab_id": str(arguments["tab_id"]),
            "path": str(arguments["path"]),
            "value": arguments["value"],
        },
    )


def tool_gui_context_get_md(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("context.get_md", {})


def tool_gui_context_get_md_attr(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("context.get_md_attr", {"key": str(arguments["key"])})


def tool_gui_context_get_ml(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("context.get_ml", {})


def tool_gui_context_set_md_attr(arguments: Dict[str, Any]) -> Dict[str, Any]:
    if "value" not in arguments:
        raise ValueError("missing 'value'")
    return send_gui_rpc(
        "context.set_md_attr",
        {"key": str(arguments["key"]), "value": arguments["value"]},
    )


def tool_gui_context_del_md_attr(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("context.del_md_attr", {"key": str(arguments["key"])})


def tool_gui_context_set_ml_module(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc(
        "context.set_ml_module",
        {"name": str(arguments["name"]), "raw": dict(arguments["raw"])},
    )


def tool_gui_context_del_ml_module(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("context.del_ml_module", {"name": str(arguments["name"])})


def tool_gui_context_set_ml_waveform(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc(
        "context.set_ml_waveform",
        {"name": str(arguments["name"]), "raw": dict(arguments["raw"])},
    )


def tool_gui_context_del_ml_waveform(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("context.del_ml_waveform", {"name": str(arguments["name"])})


def tool_gui_device_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("device.list", {})


def tool_gui_device_snapshot(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("device.snapshot", {"name": str(arguments["name"])})


def tool_gui_device_connect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "type_name": str(arguments["type_name"]),
        "name": str(arguments["name"]),
        "address": str(arguments["address"]),
    }
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    return send_gui_rpc("device.connect", params)


def tool_gui_device_disconnect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"name": str(arguments["name"])}
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    return send_gui_rpc("device.disconnect", params)


def tool_gui_device_reconnect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("device.reconnect", {"name": str(arguments["name"])})


def tool_gui_device_forget(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("device.forget", {"name": str(arguments["name"])})


def tool_gui_device_set_value(arguments: Dict[str, Any]) -> Dict[str, Any]:
    if "value" not in arguments:
        raise ValueError("missing 'value'")
    return send_gui_rpc(
        "device.set_value",
        {"name": str(arguments["name"]), "value": arguments["value"]},
    )


def tool_gui_device_setup(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc(
        "device.setup",
        {"name": str(arguments["name"]), "updates": dict(arguments["updates"])},
    )


def tool_gui_device_cancel_operation(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return send_gui_rpc("device.cancel_operation", {"name": str(arguments["name"])})


def tool_gui_device_active_setup(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("device.active_setup", {})


def tool_gui_device_active_operation(arguments: Dict[str, Any]) -> Dict[str, Any]:
    del arguments
    return send_gui_rpc("device.active_operation", {})


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


TOOLS: Dict[str, Dict[str, Any]] = {
    "gui_connect": {
        "handler": tool_gui_connect,
        "description": (
            "Connect the MCP bridge to an already-running GUI's TCP control port. "
            "Skip this if you used gui_launch with auto_connect=true (default)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP port of the GUI control service",
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
            "required": ["port"],
        },
    },
    "gui_disconnect": {
        "handler": tool_gui_disconnect,
        "description": (
            "Disconnect the MCP bridge from the GUI control port. "
            "Does NOT stop the GUI process — use gui_stop for that."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_launch": {
        "handler": tool_gui_launch,
        "description": (
            "Launch the qubit-measure GUI as a subprocess, wait until its TCP "
            "control port is ready, and optionally connect immediately. "
            "Use this as the first step to start a session. "
            "By default auto_connect=true so gui_connect is called automatically."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP control port for the GUI (default 8765)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional shared auth token (also passed to gui_connect if auto_connect=true)",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Call gui_connect automatically once port is ready (default true)",
                },
            },
        },
    },
    "gui_stop": {
        "handler": tool_gui_stop,
        "description": (
            "Stop the GUI subprocess that was started by gui_launch, "
            "and disconnect the MCP socket."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Send SIGKILL instead of SIGTERM (default false)",
                }
            },
        },
    },
    "gui_connect_mock": {
        "handler": tool_gui_connect_mock,
        "description": (
            "One-shot setup for testing/offline use: starts a Mock FPGA SoC, "
            "applies default project startup parameters (chip=Q1_Chip, qub=Q1, res=R1), "
            "waits for SoC to be ready, then activates the first existing context "
            "(or creates one at 1.0 A). Requires gui_connect first."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_startup_apply": {
        "handler": tool_gui_startup_apply,
        "description": (
            "Apply project startup settings: chip/qubit/resonator names, result directory, "
            "and database path. Must be called once after SoC connects before running experiments. "
            "Use gui_connect_mock instead for mock/testing workflows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chip_name": {"type": "string"},
                "qub_name": {"type": "string"},
                "res_name": {"type": "string"},
                "result_dir": {"type": "string"},
                "database_path": {"type": "string"},
            },
            "required": [
                "chip_name",
                "qub_name",
                "res_name",
                "result_dir",
                "database_path",
            ],
        },
    },
    "gui_adapter_list": {
        "handler": tool_gui_adapter_list,
        "description": "List available experiment adapter names.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_connect_start": {
        "handler": tool_gui_connect_start,
        "description": (
            "Start an FPGA SoC hardware connection. kind='mock' uses a software-simulated SoC "
            "(no real hardware needed); kind='remote' connects to a real ZCU216 board at ip:port. "
            "This controls the SoC/FPGA link, not the MCP socket — see gui_connect for that."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "'mock' or 'remote'"},
                "ip": {
                    "type": "string",
                    "description": "Board IP address (required when kind='remote')",
                },
                "port": {
                    "type": "integer",
                    "description": "Board port (required when kind='remote')",
                },
            },
            "required": ["kind"],
        },
    },
    "gui_tab_list": {
        "handler": tool_gui_tab_list,
        "description": (
            "List all open experiment tabs with their tab_id and adapter_name. "
            "Use tab_id from this response in all other gui_tab_* and gui_run_* calls."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_tab_new": {
        "handler": tool_gui_tab_new,
        "description": (
            "Open a new experiment tab for the given adapter. "
            "Get valid adapter names from gui_adapter_list first. "
            "Returns the new tab_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Adapter name from gui_adapter_list, e.g. 'onetone/fake_freq'",
                }
            },
            "required": ["adapter_name"],
        },
    },
    "gui_tab_close": {
        "handler": tool_gui_tab_close,
        "description": "Close an experiment tab. Any unsaved data will be lost.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_set_active": {
        "handler": tool_gui_tab_set_active,
        "description": "Bring an experiment tab to the foreground in the GUI view.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_snapshot": {
        "handler": tool_gui_tab_snapshot,
        "description": (
            "Get a scalar summary of a tab's current state: run status, "
            "has_run_result, has_analyze_result, has_figure, save_paths, adapter_name. "
            "If tab_id is omitted, returns all tabs as a list under 'tabs'. "
            "If tab_id is given, returns a single tab snapshot directly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Omit to get all tabs",
                }
            },
        },
    },
    "gui_tab_get_cfg": {
        "handler": tool_gui_tab_get_cfg,
        "description": (
            "Read the full configuration dict of a tab. "
            "Use the returned keys and paths with gui_cfg_set_field for targeted edits, "
            "or pass the modified dict to gui_tab_update_cfg."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_update_cfg": {
        "handler": tool_gui_tab_update_cfg,
        "description": (
            "Merge a partial config dict into a tab's configuration. "
            "Only the keys present in 'raw' are updated; omitted keys keep their current values. "
            "Call gui_tab_get_cfg first to see the full structure. "
            "For single-field edits, prefer gui_cfg_set_field instead. "
            "Fails with PRECONDITION_FAILED if the tab is currently running."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "raw": {
                    "type": "object",
                    "description": "Partial config dict — only provided keys are updated",
                },
            },
            "required": ["tab_id", "raw"],
        },
    },
    "gui_run_start": {
        "handler": tool_gui_run_start,
        "description": (
            "Start running the experiment in the given tab. "
            "Returns immediately (async); the run continues in the background. "
            "Poll gui_run_running_tab or subscribe to 'run_lock_changed' to detect completion. "
            "Editing cfg (gui_cfg_set_field / gui_tab_update_cfg) is rejected while the tab is running."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_run_cancel": {
        "handler": tool_gui_run_cancel,
        "description": "Cancel the currently running experiment. No-op if nothing is running.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_run_running_tab": {
        "handler": tool_gui_run_running_tab,
        "description": "Return the tab_id of the currently running experiment, or null if idle.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_save_set_paths": {
        "handler": lambda args: send_gui_rpc("save.set_paths", args),
        "description": (
            "Set the save path overrides for a tab (mirrors the UI path fields). "
            "After calling this, gui_save_data / gui_save_image / gui_save_both "
            "can be called without explicit paths and will use these values. "
            "The paths are also visible in gui_tab_snapshot under 'save_paths'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "data_path": {
                    "type": "string",
                    "description": "Absolute path for HDF5 data file",
                },
                "image_path": {
                    "type": "string",
                    "description": "Absolute path for PNG image file",
                },
            },
            "required": ["tab_id", "data_path", "image_path"],
        },
    },
    "gui_save_both": {
        "handler": tool_gui_save_both,
        "description": (
            "Save experiment data and analysis plot image in one call. "
            "Paths are optional — if omitted, uses the tab's configured save_paths "
            "(set via gui_save_set_paths or the UI path fields; visible in gui_tab_snapshot). "
            "Optional comment is stored in the HDF5 file metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "data_path": {
                    "type": "string",
                    "description": "Override data path (optional; uses tab save_paths if omitted)",
                },
                "image_path": {
                    "type": "string",
                    "description": "Override image path (optional; uses tab save_paths if omitted)",
                },
                "comment": {
                    "type": "string",
                    "description": "Optional comment stored in HDF5 metadata",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_save_data": {
        "handler": tool_gui_save_data,
        "description": (
            "Save only the experiment data (HDF5) for a tab. "
            "data_path is optional — if omitted, uses the tab's configured save_paths "
            "(set via gui_save_set_paths or the UI path fields; visible in gui_tab_snapshot). "
            "Optional comment is stored in the HDF5 file metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "data_path": {
                    "type": "string",
                    "description": "Override data path (optional; uses tab save_paths if omitted)",
                },
                "comment": {
                    "type": "string",
                    "description": "Optional comment stored in HDF5 metadata",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_save_image": {
        "handler": tool_gui_save_image,
        "description": (
            "Save only the analysis plot image (PNG) for a tab. "
            "image_path is optional — if omitted, uses the tab's configured save_paths "
            "(set via gui_save_set_paths or the UI path fields; visible in gui_tab_snapshot)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "image_path": {
                    "type": "string",
                    "description": "Override image path (optional; uses tab save_paths if omitted)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_state_check": {
        "handler": tool_gui_state_check,
        "description": (
            "Return all four GUI readiness flags at once: has_project, has_context, "
            "has_active_context, has_soc. Call this to verify the GUI is ready before "
            "running experiments. All four should be true for a normal workflow."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_labels": {
        "handler": tool_gui_context_labels,
        "description": (
            "List all context labels in the current project. "
            "Each label encodes the flux bias point, e.g. '052621_1.000A' means "
            "date 05/26/21 at 1.000 A. Use gui_context_use to switch between them."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_active": {
        "handler": tool_gui_context_active,
        "description": "Return the currently active context label (flux bias point).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_use": {
        "handler": tool_gui_context_use,
        "description": (
            "Switch the active context to an existing label. "
            "Get available labels from gui_context_labels first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Exact label string from gui_context_labels",
                }
            },
            "required": ["label"],
        },
    },
    "gui_context_new": {
        "handler": tool_gui_context_new,
        "description": (
            "Create a new flux-bias context at the given current value and unit "
            "(e.g. value=1.5, unit='A'). Optionally clone settings from the current context. "
            "The new context becomes active automatically."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Flux bias current value"},
                "unit": {
                    "type": "string",
                    "description": "Unit string, e.g. 'A' or 'mA'",
                },
                "clone_from_current": {
                    "type": "boolean",
                    "description": "Copy MetaDict and ModuleLibrary from active context",
                },
            },
        },
    },
    "gui_session_persist": {
        "handler": tool_gui_session_persist,
        "description": "Save the current set of open tabs to disk so they can be restored after a restart.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_session_restore": {
        "handler": tool_gui_session_restore,
        "description": "Reopen tabs from the last persisted session (saved by gui_session_persist).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_events_subscribe": {
        "handler": tool_gui_events_subscribe,
        "description": (
            "Subscribe to one or more GUI event push streams. "
            "Subscribed events are delivered to gui_events_poll. "
            "Use gui_events_list to see available event names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Event names to subscribe to, e.g. ['run_lock_changed', 'tab_content_changed']",
                }
            },
            "required": ["events"],
        },
    },
    "gui_events_unsubscribe": {
        "handler": tool_gui_events_unsubscribe,
        "description": "Unsubscribe from one or more GUI event push streams.",
        "inputSchema": {
            "type": "object",
            "properties": {"events": {"type": "array", "items": {"type": "string"}}},
            "required": ["events"],
        },
    },
    "gui_events_list": {
        "handler": tool_gui_events_list,
        "description": (
            "List all supported event names and which ones are currently subscribed. "
            "Call this before gui_events_subscribe to discover available events."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_events_poll": {
        "handler": tool_gui_events_poll,
        "description": (
            "Drain up to max_events queued event pushes from subscribed streams. "
            "Blocks for up to timeout_seconds (default 5.0) if the queue is empty. "
            "Returns immediately if events are available. "
            "Must call gui_events_subscribe first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max wait time in seconds (default 5.0)",
                },
                "max_events": {
                    "type": "integer",
                    "description": "Max events to return in one call (default 16)",
                },
            },
        },
    },
    "gui_dialog_open": {
        "handler": tool_gui_dialog_open,
        "description": (
            "Open a named dialog panel in the GUI. "
            "Valid names: 'setup', 'device', 'predictor', 'inspect', 'startup'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "One of: setup, device, predictor, inspect, startup",
                }
            },
            "required": ["name"],
        },
    },
    "gui_dialog_close": {
        "handler": tool_gui_dialog_close,
        "description": "Close a named dialog if it is currently open. No-op if already closed.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_dialog_list_open": {
        "handler": tool_gui_dialog_list_open,
        "description": "List the names of all dialogs currently open in the GUI.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_view_snapshot": {
        "handler": tool_gui_view_snapshot,
        "description": (
            "Return a JSON summary of the current GUI window state: "
            "active tab, context label, run status, and open dialogs."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_view_screenshot": {
        "handler": tool_gui_view_screenshot,
        "description": (
            "Capture the main window or a specific tab as a PNG image. "
            "If out_path (absolute path) is given, the image is written to disk and "
            "the base64 payload is omitted from the reply. "
            "If tab_id is omitted, captures the full main window."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Capture a specific tab instead of the full window",
                },
                "out_path": {
                    "type": "string",
                    "description": "Absolute path to save PNG (omits base64 from reply)",
                },
            },
        },
    },
    "gui_cfg_set_field": {
        "handler": tool_gui_cfg_set_field,
        "description": (
            "Set a single cfg field on a tab by dotted path. "
            "Use gui_tab_get_cfg to discover valid paths first. "
            "Examples: 'reps' (integer), 'sweep.expts' (integer), 'qubit_pulse.value.freq' (float). "
            "Prefer this over gui_tab_update_cfg for single-field edits. "
            "Fails with PRECONDITION_FAILED if the tab is currently running."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "Dotted path from gui_tab_get_cfg, e.g. 'sweep.expts'",
                },
                "value": {
                    "type": ["number", "string", "boolean", "object", "array", "null"],
                    "description": "New value (number / string / bool)",
                },
            },
            "required": ["tab_id", "path", "value"],
        },
    },
    "gui_context_get_md": {
        "handler": tool_gui_context_get_md,
        "description": (
            "List all MetaDict attribute keys in the active context. "
            "MetaDict stores qubit experiment parameters (e.g. resonator frequency 'r_f', "
            "waveform width 'rf_w'). Use gui_context_get_md_attr to read individual values."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_get_md_attr": {
        "handler": tool_gui_context_get_md_attr,
        "description": (
            "Read one MetaDict attribute value by key. "
            "Get available keys from gui_context_get_md first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "MetaDict key from gui_context_get_md",
                }
            },
            "required": ["key"],
        },
    },
    "gui_context_set_md_attr": {
        "handler": tool_gui_context_set_md_attr,
        "description": (
            "Set one MetaDict attribute to a new value. "
            "Changes are persisted to the context's JSON file immediately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {
                    "type": ["number", "string", "boolean", "object", "array", "null"]
                },
            },
            "required": ["key", "value"],
        },
    },
    "gui_context_del_md_attr": {
        "handler": tool_gui_context_del_md_attr,
        "description": "Delete one MetaDict attribute from the active context.",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    "gui_context_get_ml": {
        "handler": tool_gui_context_get_ml,
        "description": (
            "List all module and waveform names in the active context's ModuleLibrary. "
            "ModuleLibrary stores QICK pulse definitions (waveforms, reset sequences, etc.)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_set_ml_module": {
        "handler": tool_gui_context_set_ml_module,
        "description": "Add or replace one ModuleLibrary module entry from a raw config dict.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Module name"},
                "raw": {"type": "object", "description": "Module configuration dict"},
            },
            "required": ["name", "raw"],
        },
    },
    "gui_context_del_ml_module": {
        "handler": tool_gui_context_del_ml_module,
        "description": "Delete one module from the active context's ModuleLibrary.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_context_set_ml_waveform": {
        "handler": tool_gui_context_set_ml_waveform,
        "description": "Add or replace one waveform in the active context's ModuleLibrary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Waveform name"},
                "raw": {"type": "object", "description": "Waveform configuration dict"},
            },
            "required": ["name", "raw"],
        },
    },
    "gui_context_del_ml_waveform": {
        "handler": tool_gui_context_del_ml_waveform,
        "description": "Delete one waveform from the active context's ModuleLibrary.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_device_list": {
        "handler": tool_gui_device_list,
        "description": (
            "List all registered hardware devices with their name, type, and connection status. "
            "Devices include signal generators, flux bias sources, etc. "
            "Use device name in all other gui_device_* calls."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_device_snapshot": {
        "handler": tool_gui_device_snapshot,
        "description": "Read the cached state snapshot (settings, last known value) for one device by name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Device name from gui_device_list",
                }
            },
            "required": ["name"],
        },
    },
    "gui_device_connect": {
        "handler": tool_gui_device_connect,
        "description": (
            "Register and start connecting a hardware device (async). "
            "Returns immediately; connection completes in the background. "
            "Poll gui_device_active_operation or subscribe to 'device_changed' to detect completion. "
            "type_name identifies the driver class (e.g. 'YOKOGS200', 'SGS100A'); "
            "address is the VISA/GPIB/IP address. "
            "Set remember=true to persist the device across sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type_name": {
                    "type": "string",
                    "description": "Driver class name, e.g. 'YOKOGS200'",
                },
                "name": {
                    "type": "string",
                    "description": "Friendly name for this device",
                },
                "address": {"type": "string", "description": "VISA or IP address"},
                "remember": {
                    "type": "boolean",
                    "description": "Persist device across sessions (default false)",
                },
            },
            "required": ["type_name", "name", "address"],
        },
    },
    "gui_device_disconnect": {
        "handler": tool_gui_device_disconnect,
        "description": (
            "Start disconnecting a device (async). "
            "Returns immediately; disconnection completes in the background. "
            "Poll gui_device_active_operation or subscribe to 'device_changed' to detect completion. "
            "Set remember=false to also remove it from persistent storage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "remember": {
                    "type": "boolean",
                    "description": "Keep device in persistent storage (default true)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_device_reconnect": {
        "handler": tool_gui_device_reconnect,
        "description": "Reconnect a previously remembered device using its stored address.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_device_forget": {
        "handler": tool_gui_device_forget,
        "description": "Remove a device from persistent storage. It will not appear after restart.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_device_set_value": {
        "handler": tool_gui_device_set_value,
        "description": (
            "Set the output value on a connected value-type device (e.g. flux bias current in A). "
            "The device must already be connected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {
                    "type": ["number", "string", "boolean", "object", "array", "null"],
                    "description": "New output value (type depends on device, typically a float)",
                },
            },
            "required": ["name", "value"],
        },
    },
    "gui_device_setup": {
        "handler": tool_gui_device_setup,
        "description": (
            "Update device configuration fields and apply them (async). "
            "Returns immediately; the setup continues in the background. "
            "Poll gui_device_active_setup to track progress. "
            "'updates' is a partial dict of field names to new values, merged into the current device info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": "Partial device info dict to apply",
                },
            },
            "required": ["name", "updates"],
        },
    },
    "gui_device_cancel_operation": {
        "handler": tool_gui_device_cancel_operation,
        "description": "Cancel an in-progress device connect/setup operation.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_device_active_setup": {
        "handler": tool_gui_device_active_setup,
        "description": "Poll the progress of an ongoing device setup operation started by gui_device_setup.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_device_active_operation": {
        "handler": tool_gui_device_active_operation,
        "description": "Poll the progress of an ongoing device connect/disconnect operation.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_run_progress": {
        "handler": lambda args: send_gui_rpc("run.progress", {}),
        "description": (
            "Read the current run progress bar state. "
            "Returns active=false and bars=[] if no run is in progress. "
            "When active=true, 'bars' is a list of progress bar snapshots — one per "
            "active nesting level (e.g. outer rounds + inner averages). "
            "Each entry has: desc (label), n (steps done), total (total steps or null), "
            "elapsed (seconds), remaining (estimated seconds left or null), "
            "format (human-readable string like 'Rounds 23/100 [0:25<1:15]')."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_tab_figure_screenshot": {
        "handler": lambda args: send_gui_rpc("tab.figure_screenshot", args),
        "description": (
            "Capture only the figure/plot area of a tab as PNG. "
            "More focused than gui_view_screenshot — excludes config panel and progress bar. "
            "Fails with PRECONDITION_FAILED if the tab has no figure yet "
            "(run has not completed or no analysis result). "
            "If out_path is given, the PNG is saved to disk and png_b64 is omitted from the reply."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "out_path": {
                    "type": "string",
                    "description": "Optional file path to save the PNG",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_predictor_load": {
        "handler": lambda args: send_gui_rpc("predictor.load", args),
        "description": (
            "Load a FluxoniumPredictor from a params.json file. "
            "path must be an absolute path to the params.json file. "
            "flux_bias is the DC flux bias offset in Amperes (default 0.0). "
            "Fails with PRECONDITION_FAILED if the file cannot be loaded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to params.json"},
                "flux_bias": {
                    "type": "number",
                    "default": 0.0,
                    "description": "DC flux bias offset in Amperes",
                },
            },
            "required": ["path"],
        },
    },
    "gui_predictor_clear": {
        "handler": lambda args: send_gui_rpc("predictor.clear", {}),
        "description": "Clear the currently loaded FluxoniumPredictor.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_predictor_predict": {
        "handler": lambda args: send_gui_rpc("predictor.predict", args),
        "description": (
            "Predict a qubit transition frequency at a given flux value. "
            "value is the flux in Amperes. "
            "from_lvl and to_lvl select the energy level transition (default 0→1). "
            "Returns freq_mhz (float). "
            "Fails with PRECONDITION_FAILED if no predictor is loaded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Flux value in Amperes"},
                "from_lvl": {
                    "type": "integer",
                    "default": 0,
                    "description": "Lower energy level (default 0)",
                },
                "to_lvl": {
                    "type": "integer",
                    "default": 1,
                    "description": "Upper energy level (default 1)",
                },
            },
            "required": ["value"],
        },
    },
    "gui_predictor_info": {
        "handler": lambda args: send_gui_rpc("predictor.info", {}),
        "description": (
            "Get the currently loaded predictor info. "
            "Returns info={path, flux_bias} if loaded, or info=null if not loaded."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_tab_get_analyze_result": {
        "handler": lambda args: send_gui_rpc("tab.get_analyze_result", args),
        "description": (
            "Read the scalar summary of a tab's latest analyze result. "
            "Returns summary=null if no analyze result is available yet. "
            "Fields vary by adapter (e.g. onetone: freq_mhz, fwhm_mhz; T1: t1_us, t1_err_us). "
            "Call after gui_analyze_start completes (wait for tab_interaction_changed event)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_get_analyze_params": {
        "handler": lambda args: send_gui_rpc("tab.get_analyze_params", args),
        "description": (
            "Read the current analyze parameters for a tab as a flat dict. "
            "Returns analyze_params=null if no run result exists yet (analyze params are "
            "initialized from the run result). Use the returned keys/values as the "
            "base for gui_analyze_start updates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_analyze_start": {
        "handler": lambda args: send_gui_rpc("analyze.start", args),
        "description": (
            "Start analyzing the run result in a tab (fire-and-forget, async). "
            "Returns immediately; analysis continues in the background. "
            "The tab must have a run result (has_run_result=true in gui_tab_snapshot). "
            "Optional 'updates' is a partial dict of analyze param overrides merged into "
            "the current analyze params (get current params via gui_tab_get_analyze_params). "
            "Subscribe to 'tab_interaction_changed' or poll gui_tab_snapshot to detect "
            "completion (has_analyze_result=true)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": "Partial analyze param overrides (optional)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_get_cfg_summary": {
        "handler": lambda args: send_gui_rpc("tab.get_cfg_summary", args),
        "description": (
            "Read the tab's current cfg as a clean, human-readable dict. "
            "Unlike gui_tab_get_cfg, this strips all internal tags (__kind, is_unset) "
            "and returns only scalar values: numbers, strings, null (for unset fields), "
            "eval expressions as strings, sweep ranges as {start,stop,expts,step} dicts, "
            "and module/waveform refs as {chosen, value} dicts. "
            "Use this to understand what parameters the tab is currently configured with."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def _cleanup_on_exit() -> None:
    """Stop the GUI process when the MCP host disconnects (stdin EOF)."""
    try:
        tool_gui_stop({"force": False})
    except Exception:
        pass


def main() -> None:
    # Set stdin/stdout to UTF-8 encoded mode.
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
                            "name": "qubit-measure-control",
                            "version": "1.1.0",
                        },
                    },
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

            elif method == "notifications/initialized":
                continue

            elif method == "tools/list":
                tools_list = []
                for name, info in TOOLS.items():
                    tools_list.append(
                        {
                            "name": name,
                            "description": info["description"],
                            "inputSchema": info["inputSchema"],
                        }
                    )
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
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {
                                "isError": True,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Error executing tool {name!r}: {e}\n{traceback.format_exc()}",
                                    }
                                ],
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
