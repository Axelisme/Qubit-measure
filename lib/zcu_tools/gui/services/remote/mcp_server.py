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
import socket
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
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
    return send_gui_rpc(
        "save.both",
        {
            "tab_id": str(arguments["tab_id"]),
            "data_path": str(arguments["data_path"]),
            "image_path": str(arguments["image_path"]),
        },
    )


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
# Tool registry
# ---------------------------------------------------------------------------


TOOLS: Dict[str, Dict[str, Any]] = {
    "gui_connect": {
        "handler": tool_gui_connect,
        "description": "Connect to the qubit-measure GUI's TCP control port.",
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
        "description": "Disconnect from the GUI control port.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_connect_mock": {
        "handler": tool_gui_connect_mock,
        "description": "Initiate connection to Mock SOC and automatically set up active context.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_startup_apply": {
        "handler": tool_gui_startup_apply,
        "description": "Apply project startup settings to load parameters.",
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
    "gui_tab_list": {
        "handler": tool_gui_tab_list,
        "description": "List all open experiment tabs.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_tab_new": {
        "handler": tool_gui_tab_new,
        "description": "Open a new experiment tab for the given adapter name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "e.g., 'onetone/fake_freq'",
                }
            },
            "required": ["adapter_name"],
        },
    },
    "gui_tab_close": {
        "handler": tool_gui_tab_close,
        "description": "Close an experiment tab by its tab_id.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_set_active": {
        "handler": tool_gui_tab_set_active,
        "description": "Make an experiment tab active in the view.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_snapshot": {
        "handler": tool_gui_tab_snapshot,
        "description": "Get summary snapshot of tab run/analyze state.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_get_cfg": {
        "handler": tool_gui_tab_get_cfg,
        "description": "Read raw config dictionary of a tab.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_tab_update_cfg": {
        "handler": tool_gui_tab_update_cfg,
        "description": "Write/replace config dictionary of a tab (full replace).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "raw": {
                    "type": "object",
                    "description": "Fully replaced configuration dict",
                },
            },
            "required": ["tab_id", "raw"],
        },
    },
    "gui_run_start": {
        "handler": tool_gui_run_start,
        "description": "Start running experiment (fire-and-forget).",
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_run_cancel": {
        "handler": tool_gui_run_cancel,
        "description": "Cancel currently running experiment.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_run_running_tab": {
        "handler": tool_gui_run_running_tab,
        "description": "Get the running tab_id or null.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_save_both": {
        "handler": tool_gui_save_both,
        "description": "Save both data and analysis plot image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "data_path": {"type": "string"},
                "image_path": {"type": "string"},
            },
            "required": ["tab_id", "data_path", "image_path"],
        },
    },
    "gui_state_check": {
        "handler": tool_gui_state_check,
        "description": "Quick check of GUI connection status flags.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # ---- Phase 81a — events / dialog / view -----------------------------
    "gui_events_subscribe": {
        "handler": tool_gui_events_subscribe,
        "description": "Subscribe to GUI event push streams.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of event names (see gui_events_list)",
                }
            },
            "required": ["events"],
        },
    },
    "gui_events_unsubscribe": {
        "handler": tool_gui_events_unsubscribe,
        "description": "Unsubscribe from GUI event push streams.",
        "inputSchema": {
            "type": "object",
            "properties": {"events": {"type": "array", "items": {"type": "string"}}},
            "required": ["events"],
        },
    },
    "gui_events_list": {
        "handler": tool_gui_events_list,
        "description": "List all supported and currently subscribed event names.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_events_poll": {
        "handler": tool_gui_events_poll,
        "description": (
            "Drain up to ``max_events`` queued event pushes; blocks for up to "
            "``timeout_seconds`` if the queue is empty."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_seconds": {"type": "number", "default": 5.0},
                "max_events": {"type": "integer", "default": 16},
            },
        },
    },
    "gui_dialog_open": {
        "handler": tool_gui_dialog_open,
        "description": "Open a named dialog (setup / device / predictor / inspect / startup).",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_dialog_close": {
        "handler": tool_gui_dialog_close,
        "description": "Close a named dialog if open.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    "gui_dialog_list_open": {
        "handler": tool_gui_dialog_list_open,
        "description": "List dialogs currently open in the GUI.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_view_snapshot": {
        "handler": tool_gui_view_snapshot,
        "description": "Capture a JSON-friendly summary of visible window state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_view_screenshot": {
        "handler": tool_gui_view_screenshot,
        "description": (
            "Capture the main window (or one active tab) as PNG. If "
            "``out_path`` is given the image is written to disk and the "
            "base64 payload is omitted from the reply."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "out_path": {"type": "string"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def main() -> None:
    # Set stdin/stdout to UTF-8 encoded mode.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
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
