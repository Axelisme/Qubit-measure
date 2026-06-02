#!/usr/bin/env python
"""MCP server bridge for ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live GUI's ``RemoteControlAdapter`` over a
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
from collections import deque
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

# This bridge is launched standalone (``python .../mcp_server.py``), so the repo
# ``lib`` dir is not on sys.path by default. Add it so the wire-contract modules
# import cleanly. Importing under ``zcu_tools.gui`` runs ``gui/__init__`` which
# eagerly loads Qt; the bridge tolerates this (it never builds a QApplication),
# trading a heavier import for a single MethodSpec source of truth shared with
# the dispatcher.
_LIB_DIR = Path(__file__).resolve().parents[4]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: importing ``zcu_tools.gui`` below eagerly loads Qt. If the
# 'gui' extra was not installed (e.g. a fresh checkout on another machine), the
# import chain would otherwise die with a bare ``ModuleNotFoundError`` deep
# inside the package. Surface an actionable instruction on stderr instead
# (stdout is the JSON-RPC channel and must stay clean).
for _gui_dep in ("qtpy", "PyQt6"):
    if importlib.util.find_spec(_gui_dep) is None:
        sys.stderr.write(
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            f"'{_gui_dep}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n"
        )
        raise SystemExit(1)

from zcu_tools.gui.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.services.remote.param_spec import (  # noqa: E402
    JsonType,
    build_input_schema,
)
from zcu_tools.gui.services.remote.wire import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# This MCP server's own code revision — reported (not compared) in the version
# note so an agent can confirm a reconnect picked up its bridge-side edits. Bump
# on a meaningful mcp_server change you want to be able to spot a reload of.
# v2: piggyback/diagnostic-split + default-subscribe now includes run_started /
#     run_finished (was run_lock_changed).
# v3: default-subscribe device_setup_started/finished (was device_setup_changed).
# v4: gui_run_start / gui_connect_start now short-wait degrade like device ops
#     (a fast run/connect returns its product, slow degrades to a handle); new
#     gui_run_wait / gui_connect_wait semantic waits.
# v7: batch convenience tools gui_editor_set_fields / gui_context_set_md_attrs —
#     bridge-side fan-out over the existing editor.set_field / context.set_md_attr
#     RPCs (fail-fast, set_fields returns the whole draft cfg). No wire change, so
#     WIRE_VERSION stays put.
# v8: adapter.guide generated tool (gui_adapter_guide) rides the WIRE 12 bump —
#     read an adapter's orientation guide before running it.
# v9: removed gui_connect_mock — agent now walks the same startup path as a GUI
#     user (gui_connect_start kind=mock + gui_startup_apply + gui_context_use),
#     eliminating the mock-only shortcut that diverged (caused a save-path bug).
#     Pure mcp-side tool removal; no wire change (WIRE stays 12).
# v10: startup.apply omitting result_dir/database_path now fills the default
#      per-qubit roots (<cwd>/result|Database/<chip>/<qub>) so the project is
#      runnable, instead of leaving a DRAFT context. Handler-side default only;
#      wire contract unchanged.
# v11: gui_launch uses sys.executable (cross-platform, was hardcoded
#      .venv/bin/python) + Windows process-group flag; gui_stop closes via the
#      app.shutdown RPC (graceful, no OS kill) and only force-kills when
#      timeout_kill=true (replaces the old 'force' SIGKILL param). gui_app_shutdown
#      tool rides WIRE 13.
MCP_VERSION = 11

# ---------------------------------------------------------------------------
# Server usage instructions (returned in the MCP `initialize` result)
# ---------------------------------------------------------------------------

_SERVER_INSTRUCTIONS = """\
Drive a live qubit-measure GUI over a TCP control socket.

Getting started (same path a GUI user takes — no mock shortcut):
  1. gui_launch (auto-connects).
  2. gui_connect_start(kind='mock') for offline/testing (or kind='remote',
     ip, port for hardware) — the same connect the user picks via the setup
     dialog's "Use MockSoc" checkbox. It short-wait degrades; gui_connect_wait
     if it returns pending.
  3. gui_startup_apply(chip_name, qub_name, res_name[, result_dir,
     database_path]) — applies the project; omitting result_dir/database_path
     scopes them under chip/qub per the notebook layout.
  4. gui_context_use(label) to activate an existing context, or
     gui_context_new(value, unit) to create one.
  5. gui_state_check — all four flags (has_project / has_context /
     has_active_context / has_soc) should be true before running experiments.

Typical experiment loop:
  - gui_adapter_list -> gui_tab_new(adapter_name) -> note the returned tab_id.
  - Inspect/edit config: gui_tab_get_cfg, then edit single fields via the tab's
    cfg-editor session — take editor_id from gui_tab_snapshot and call
    gui_editor_set_field(editor_id, path, value). Paths are dotted and must match
    gui_tab_list_paths, e.g. 'reps', 'sweep.gain.expts',
    'modules.qub_pulse.value.freq'. Nested module fields need the 'modules.'
    prefix; an unknown path fails with invalid_params rather than silently
    no-op'ing. (This is the same draft the GUI form shows — edits are WYSIWYG.)
  - gui_run_start(tab_id) waits briefly (wait_seconds, default 1.0): a fast run
    returns {status:'finished', tab:{...}}; a slow one returns {status:'pending'}
    — then gui_run_wait(tab_id) or watch run_finished. (gui_device_* and
    gui_connect_start degrade the same way.)
  - gui_analyze(tab_id) after a run — synchronous: on return the analysis is
    done (read gui_tab_get_analyze_result, no wait step). Then gui_save_data /
    gui_save_image / gui_save_both to persist.

Detecting completion — prefer events over polling:
  - run_started{tab_id} and run_finished{tab_id, outcome, error_message} are
    auto-subscribed on connect; gui_events_poll (blocks up to timeout_seconds)
    receives pushes. (gui_events_subscribe to add more, e.g.
    'tab_content_changed'.)
  - run_finished.outcome ∈ 'finished'|'failed'|'cancelled' (error_message set
    on failure) tells success from failure from cancellation.
  - tab_content_changed fires when a run/analyze result becomes available.
  - gui_run_progress gives in-flight bar snapshots (token/format/maximum/value/
    percent) but is a fallback; do not busy-poll gui_run_running_tab in a sleep
    loop.
  - 'diagnostic' is an UNSOLICITED push (no subscription needed) carrying
    {severity: 'error'|'info', title, message} — the same user-facing feedback
    the GUI shows in a dialog/status bar. It arrives via gui_events_poll like
    any event; watch for severity=='error' to learn about failures the GUI
    surfaced (including ones not tied to a call you made).

Preconditions are enforced server-side and identical to the GUI buttons:
  - Run/save require an active file-backed context; save/analyze require an
    existing run result. Violations return precondition_failed with a message.
  - Editing cfg while a tab is running returns precondition_failed.

Call contract — read before issuing defensive/duplicate calls:
  - A failed call always raises an error; it never returns stale or partial
    data. One call is therefore enough — never fire a backup copy of the same
    tool in the same turn 'in case the first did not go through'.
  - Query tools (gui_*_list / _get* / _snapshot / _check / _active* /
    _progress, e.g. gui_tab_list, gui_tab_get_cfg, gui_state_check) are
    read-only and side-effect-free. Safe to retry across turns, but duplicating
    within a turn is pure waste — the result cannot change.
  - Mutating tools DO have side effects and must be sent exactly once: gui_run_start
    (fire-and-forget — a duplicate starts a SECOND run), gui_editor_set_field,
    gui_tab_new / gui_tab_close, gui_save_*, gui_device_connect / _disconnect / _setup,
    gui_context_set_* / _del_* / _rename_*, gui_editor_commit. Issue once and
    read the response rather than re-sending.
"""

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

# Event push queue (FIFO, drop-oldest when full). Diagnostics are split into
# their own queue (ADR-0013): the GUI sends them on the same socket as events,
# but they are user-facing error/info feedback delivered unconditionally (no
# subscription), so this side keeps them separate from the subscribed-event
# stream — both surface to the agent (poll + piggyback), tagged distinctly.
_EVENT_QUEUE_MAX = 1024
_EVENT_QUEUE: Deque[Dict[str, Any]] = deque(maxlen=_EVENT_QUEUE_MAX)
_DIAGNOSTIC_QUEUE: Deque[Dict[str, Any]] = deque(maxlen=_EVENT_QUEUE_MAX)
_EVENT_COND = threading.Condition()

# Events the experiment workflow assumes the agent always wants to hear about
# (run finished/failed, device setup lifecycle, SoC connect/drop). The agent
# can subscribe to more; these are auto-subscribed right after connect because
# "what an agent must not miss" is experiment policy, owned by this mcp layer —
# not a wire mechanism the GUI server should bake in.
_DEFAULT_SUBSCRIBE = (
    "run_started",
    "run_finished",
    "device_setup_started",
    "device_setup_finished",
    "soc_changed",
)

# --- Optimistic-concurrency bookkeeping (policy lives here, mcp side) --------
#
# The agent never sees version numbers; they are bookkeeping between this mcp
# layer and the RPC server. ``_LAST_SEEN`` tracks the versions we last read via
# ``resources.versions``. Guarded ops (run/save/commit) attach the subset of
# versions they depend on as ``expected_versions``; the server compares them
# atomically and rejects with PRECONDITION_FAILED if any moved (a concurrent —
# possibly human — edit). On rejection we re-read the table so the next attempt
# carries fresh baselines.
_LAST_SEEN: Dict[str, int] = {}

# Dependency map (the single place that knows what each guarded op depends on).
# Patterns use {tab_id}/{editor_id} placeholders and a literal ``device:*`` that
# expands to every current device:* key. save.* does NOT depend on cfg — the
# saved content comes from the run result's own cfg_snapshot. writeback.apply
# depends on the run+analyze results it recomputes from, plus context (it writes
# md/ml). Note: md/ml content edits bump the ``context`` version, so any op
# depending on ``context`` (run.start / editor.commit / writeback.apply) detects
# a concurrent md/ml change.
_GUARD_DEPS: Dict[str, tuple[str, ...]] = {
    # ``device:*`` guards mutations of *existing* devices; ``devices:__set__``
    # guards *set membership* (a device added/removed since the agent last read
    # versions) which the per-member glob cannot reveal.
    "run.start": (
        "tab:{tab_id}:cfg",
        "tab:{tab_id}",
        "soc",
        "context",
        "device:*",
        "devices:__set__",
    ),
    "save.data": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "save.image": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "save.both": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    # writeback.set / writeback.apply edit + apply the persistent draft (computed
    # from run+analyze results, write md/ml). A concurrent rerun/reanalyze or
    # context edit must invalidate them.
    "writeback.set": ("tab:{tab_id}:result", "tab:{tab_id}:analyze", "context"),
    "writeback.apply": ("tab:{tab_id}:result", "tab:{tab_id}:analyze", "context"),
    "editor.commit": ("editor:{editor_id}", "context"),
}

# --- Async-operation handle bookkeeping (operation_id <-> semantic name) ------
#
# Start ops (device.setup / run.start / connect.start) return an ``operation_id``
# the agent never sees (mcp/RPC bookkeeping, like version numbers). The agent
# refers to an in-flight operation by a name it understands; mcp maps that
# semantic key to the latest operation_id for it. ``operation.await`` then
# blocks on that id. "Latest wins": starting overwrites the key, since the agent
# semantically means "the current operation for this resource".
_OP_BY_KEY: Dict[str, int] = {}

# Which semantic key a start RPC's operation_id belongs to (param -> key).
# Device connect/disconnect/setup all key on the device name: "latest wins" means
# the most recent operation for that device is the one a wait tool awaits.
_OP_KEY_OF: Dict[str, "Callable[[Dict[str, Any]], str]"] = {
    "device.connect": lambda p: f"device:{p.get('name', '')}",
    "device.disconnect": lambda p: f"device:{p.get('name', '')}",
    "device.setup": lambda p: f"device:{p.get('name', '')}",
    "run.start": lambda p: f"tab:{p.get('tab_id', '')}",
    "connect.start": lambda p: "soc",  # noqa: ARG005 — uniform signature
}

_READER_THREAD: Optional[threading.Thread] = None
_READER_STOP = threading.Event()

# GUI subprocess (when launched via gui_launch tool).
_GUI_PROC: Optional[subprocess.Popen] = None

# PID file for cross-session GUI process tracking.
_GUI_PID_FILE = Path(gettempdir()) / "zcu_tools_gui.pid"

# DEBUG log for GUIs we launch (OS temp dir, not the repo). gui_launch points
# run_gui.py here so an agent can read the server-side event flow for debugging.
_GUI_LOG_FILE = Path(gettempdir()) / "zcu_tools_gui_debug.log"


def _pid_alive(pid: int) -> bool:
    """Cross-platform 'is this pid still running?'.

    POSIX: ``os.kill(pid, 0)`` raises ProcessLookupError when gone. Windows has
    no signal 0, so probe via the process handle (OpenProcess + exit code)."""
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
        return True  # exists but not signalable (e.g. permissions)


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
        if msg.get("event") == "diagnostic":
            _DIAGNOSTIC_QUEUE.append(msg)
        else:
            _EVENT_QUEUE.append(msg)
        _EVENT_COND.notify_all()


def _drain_pending() -> Dict[str, list]:
    """Take everything buffered since the last drain — for piggyback on any tool
    result. Returns separate ``events`` / ``diagnostics`` lists (possibly empty).
    The agent gets background notifications without a dedicated poll call."""
    with _EVENT_COND:
        events = [m for m in _EVENT_QUEUE]
        diagnostics = [m for m in _DIAGNOSTIC_QUEUE]
        _EVENT_QUEUE.clear()
        _DIAGNOSTIC_QUEUE.clear()
    return {"events": events, "diagnostics": diagnostics}


def _send_gui_rpc_raw(
    method: str,
    params: Dict[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Issue one RPC and return its parsed reply envelope (no guard logic)."""
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
    return holder["message"]


def _wire_version_note() -> str:
    """Build a version note for connect/launch replies showing all three
    versions, so an agent can confirm a reload took effect by eyeballing the
    numbers (no need for this layer to judge "stale").

    ``wire.version`` is a no-auth probe, so this works right after connect.
      - WIRE_VERSION is the only *compared* axis (mcp pins it): a mismatch means
        the two sides speak different protocols → hard MISMATCH.
      - GUI_VERSION and MCP_VERSION are *reported* (code revisions of each
        process). They are shown, not compared — the agent reads them to verify
        its own edits reloaded.
    """
    try:
        resp = _send_gui_rpc_raw("wire.version", {}, 5.0)
        result = resp.get("result", {})
        wire_ver = result.get("wire_version")
        gui_ver = result.get("gui_version", "?")  # absent on a pre-split GUI
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        return f" versions: mcp wire=v{MCP_WIRE_VERSION} mcp=v{MCP_VERSION}, gui=unknown ({exc})"
    if wire_ver != MCP_WIRE_VERSION:
        return (
            f" WIRE VERSION MISMATCH: mcp wire=v{MCP_WIRE_VERSION}, gui wire=v{wire_ver}"
            " — the two processes speak different protocols; restart the stale one."
        )
    return (
        f" wire v{MCP_WIRE_VERSION} (mcp==gui); gui code v{gui_ver}, "
        f"mcp code v{MCP_VERSION}."
    )


def _refresh_versions() -> None:
    """Re-read the full resource version table into ``_LAST_SEEN``.

    Pure read via ``resources.versions`` (the single read entry point). Called
    on connect and after a stale rejection so the next guarded op carries fresh
    baselines. Failures are swallowed — a missing table just means no guard.
    """
    try:
        resp = _send_gui_rpc_raw("resources.versions", {}, 5.0)
    except Exception:  # pragma: no cover — best-effort resync
        return
    if not resp.get("ok", False):
        return
    versions = resp.get("result", {}).get("versions")
    if isinstance(versions, dict):
        _LAST_SEEN.clear()
        _LAST_SEEN.update(versions)


def _build_expected_versions(method: str, params: Dict[str, Any]) -> Dict[str, int]:
    """Resolve a guarded method's dependency patterns into expected versions.

    Policy lives here: expand {tab_id}/{editor_id} placeholders and the literal
    ``device:*`` (every current device:* key) against ``_LAST_SEEN``. Returns the
    subset of versions the op depends on; the server compares only these.
    """
    deps = _GUARD_DEPS.get(method)
    if not deps:
        return {}
    expected: Dict[str, int] = {}
    for pattern in deps:
        if pattern == "device:*":
            for key in _LAST_SEEN:
                if key.startswith("device:"):
                    expected[key] = _LAST_SEEN[key]
            continue
        key = pattern.format(
            tab_id=params.get("tab_id", ""),
            editor_id=params.get("editor_id", ""),
        )
        expected[key] = _LAST_SEEN.get(key, 0)
    return expected


def send_gui_rpc(
    method: str,
    params: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Issue one RPC against the GUI; raises on error or timeout.

    For guarded methods (run/save/commit) attaches ``expected_versions`` from
    the mcp-side bookkeeping so the server can reject stale operations. On a
    stale rejection the version table is re-read so the agent's retry is fresh.
    """
    send_params = params
    if method in _GUARD_DEPS:
        send_params = dict(params)
        send_params["expected_versions"] = _build_expected_versions(method, params)

    resp = _send_gui_rpc_raw(method, send_params, timeout_seconds)
    if not resp.get("ok", False):
        err = resp.get("error", {})
        if err.get("reason") == "stale_version":
            # A dependency moved since we last read it; resync so the retry is
            # against the current table, and surface a plain-language hint.
            _refresh_versions()
            raise RuntimeError(
                "GUI Error (PRECONDITION_FAILED): a resource you depend on was "
                "changed in the GUI since you last saw it; review then retry"
            )
        msg = f"GUI Error ({err.get('code')}): {err.get('message')}"
        raise RuntimeError(msg)
    # Every successful RPC is a round-trip in which the agent "observed" the GUI;
    # refresh the baseline so its own reads/writes are not later seen as stale by
    # its own guarded ops. A concurrent (human) change between two RPCs lands
    # after this refresh and so is correctly caught by the next guard.
    _refresh_versions()
    result = dict(resp.get("result", {}))
    # Capture a start op's operation_id under its semantic key (latest wins), so
    # an agent can later await it by name — then strip it from the result, since
    # the raw id is mcp<->RPC bookkeeping that must not surface to the agent.
    key_of = _OP_KEY_OF.get(method)
    if key_of is not None and "operation_id" in result:
        _OP_BY_KEY[key_of(params)] = int(result.pop("operation_id"))
    return result


# ---------------------------------------------------------------------------
# Connection lifecycle tools
# ---------------------------------------------------------------------------


def tool_gui_connect(arguments: Dict[str, Any]) -> str:
    global _GUI_SOCK, _READER_THREAD
    # Default port 8765, symmetric with gui_launch — but opposite expectation:
    # connect attaches to a GUI that is ALREADY running there (launch starts a
    # new one on a free port). So a missing GUI is the error case here.
    port = arguments.get("port", 8765)
    if not isinstance(port, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    token = arguments.get("token")

    # Tear down any existing connection / reader before replacing it.
    tool_gui_disconnect({})

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError as exc:
        sock.close()
        raise RuntimeError(
            f"No GUI is listening on 127.0.0.1:{port} ({exc}). gui_connect "
            f"attaches to an already-running GUI; start one first with "
            f"gui_launch, or pass the port of a running GUI."
        ) from exc
    sock.settimeout(1.0)  # short blocking timeout for reader loop responsiveness
    _GUI_SOCK = sock
    _READER_STOP.clear()
    _READER_THREAD = threading.Thread(
        target=_reader_loop, name="mcp-gui-reader", daemon=True
    )
    _READER_THREAD.start()

    if token:
        send_gui_rpc("auth", {"token": token})
        _auto_subscribe_defaults()
        return (
            f"Connected to GUI on 127.0.0.1:{port} with token authentication."
            + _wire_version_note()
        )
    _auto_subscribe_defaults()
    return f"Connected to GUI on 127.0.0.1:{port}." + _wire_version_note()


def _auto_subscribe_defaults() -> None:
    """After connect, subscribe the events an experiment agent should never miss
    (run/device/SoC lifecycle). Experiment policy lives here, not in the GUI
    server. Best-effort: a subscribe failure must not fail the connect."""
    try:
        send_gui_rpc("events.subscribe", {"events": list(_DEFAULT_SUBSCRIBE)})
    except Exception:
        pass


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
        _DIAGNOSTIC_QUEUE.clear()
    return "Disconnected from GUI."


def _port_is_open(port: int) -> bool:
    """True if something is already accepting TCP connections on the port."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


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
    # Use this MCP server's own interpreter to spawn the GUI: it is the venv
    # Python (with the 'gui' extra) that 'uv run' launched the bridge under, so
    # the GUI child has the same dependencies. This is also cross-platform —
    # avoids the hardcoded '.venv/bin/python' that does not exist on Windows
    # (which uses '.venv\Scripts\python.exe').
    python = sys.executable
    run_gui = repo_root / "run_gui.py"

    if not run_gui.exists():
        raise FileNotFoundError(f"run_gui.py not found at {run_gui}")

    # Pre-flight: if the port is ALREADY accepting connections, someone else
    # (a stale GUI from a previous session that wasn't stopped, or a foreign
    # process) owns it. Spawning now would crash with 'Address already in use'
    # and we'd silently connect to the OTHER process instead — reporting a fresh
    # pid while talking to old code. Fail fast with an actionable message.
    if _port_is_open(port):
        raise RuntimeError(
            f"Port {port} is already in use — a previous GUI is likely still "
            f"running (its socket is open). This MCP server does not manage it, "
            f"so launching here would either fail or connect to that stale "
            f"process (old code, despite a fresh pid). Stop the old GUI first "
            f"(gui_stop, or kill the run_gui.py holding port {port}), or launch "
            f"on a different port."
        )

    # stderr to a file so a startup crash (e.g. bind failure) is reportable.
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

    # Detach the GUI into its own process group/session so a signal to the MCP
    # bridge does not propagate to it. start_new_session is POSIX-only; Windows
    # uses CREATE_NEW_PROCESS_GROUP.
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

    # Wait until the port is listening (up to 15 s). Each iteration first checks
    # the child is still alive: if it died (e.g. the port got taken between the
    # pre-flight check and bind), report the crash instead of connecting to
    # whatever now answers on the port.
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
            "call gui_connect manually when ready."
        )

    log_note = f" DEBUG log: {_GUI_LOG_FILE}"
    if auto_connect:
        tool_gui_connect({"port": port, "token": token} if token else {"port": port})
        return (
            f"GUI launched (pid={pid}), listening on port {port}, and connected."
            + _wire_version_note()
            + log_note
        )

    return f"GUI launched (pid={pid}) and listening on port {port}." + log_note


def _pid_for_stop() -> Tuple[Optional[int], Optional[subprocess.Popen]]:
    """The GUI pid (+ Popen if this process owns it) to stop, or (None, None)."""
    proc = _GUI_PROC
    if proc is not None and proc.poll() is None:
        return proc.pid, proc
    # Fallback: recover pid from file (handles cross-session restarts).
    return _read_pid_file(), None


def _await_exit(pid: int, proc: Optional[subprocess.Popen], timeout: float) -> bool:
    """Block up to ``timeout`` for the GUI to exit; True if it did."""
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


def tool_gui_stop(arguments: Dict[str, Any]) -> str:
    global _GUI_PROC
    pid, proc = _pid_for_stop()
    if pid is None:
        _GUI_PROC = None
        tool_gui_disconnect({})
        return "No GUI process managed by this MCP server."

    timeout = float(arguments.get("timeout", 10.0))
    timeout_kill = bool(arguments.get("timeout_kill", False))

    # Graceful close over the existing RPC channel — runs the GUI's normal
    # window-close path (persist session, disconnect devices, cleanup) on its
    # main thread, with no OS signal. This is cross-platform (no SIGKILL /
    # process-group handling). Send BEFORE disconnecting the socket the RPC
    # travels on.
    try:
        send_gui_rpc("app.shutdown", {})
    except Exception as exc:
        # stderr is safe (stdout is the JSON-RPC channel); proceed to await/kill.
        sys.stderr.write(f"gui_stop: app.shutdown RPC failed: {exc!r}\n")

    exited = _await_exit(pid, proc, timeout)
    tool_gui_disconnect({})

    if not exited and timeout_kill:
        # Opt-in fallback: the GUI did not close gracefully in time. terminate()
        # / kill() are cross-platform (POSIX SIGTERM/SIGKILL, Windows
        # TerminateProcess). pid-only path uses os.kill (SIGTERM both platforms;
        # SIGKILL POSIX-only).
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
        # Left running on purpose (no timeout_kill). Do not clear our handle —
        # a later gui_stop can retry.
        return (
            f"app.shutdown sent but GUI (pid={pid}) has not exited within "
            f"{timeout:.0f}s. It may be finishing cleanup, or stuck. Re-run "
            f"gui_stop, or pass timeout_kill=true to force-kill on timeout."
        )

    _GUI_PROC = None
    _clear_pid_file()
    return f"GUI process (pid={pid}) closed gracefully."


# ---------------------------------------------------------------------------
# Workflow tools (thin pass-through wrappers)
# ---------------------------------------------------------------------------


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


def tool_gui_editor_subscribe(arguments: Dict[str, Any]) -> Dict[str, Any]:
    editor_id = str(arguments["editor_id"])
    return send_gui_rpc("editor.subscribe", {"editor_id": editor_id})


def tool_gui_editor_unsubscribe(arguments: Dict[str, Any]) -> Dict[str, Any]:
    editor_id = str(arguments["editor_id"])
    return send_gui_rpc("editor.unsubscribe", {"editor_id": editor_id})


def _coerce_pairs(
    value: object, *, field: str, keys: "tuple[str, str]"
) -> List[Dict[str, Any]]:
    """Validate a batch list of {k0, k1} dicts, fail-fast on shape errors.

    Validation happens up front (before any RPC) so a malformed item never lets
    a partial batch fire — keeping the failure boundary at 'nothing applied'
    rather than 'some applied'.
    """
    k0, k1 = keys
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field!r} must be a non-empty list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or k0 not in item or k1 not in item:
            raise ValueError(f"{field}[{i}] must be an object with {k0!r} and {k1!r}")
        out.append(item)
    return out


def tool_gui_editor_set_fields(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Apply several editor.set_field edits to ONE editor, fail-fast in order.

    Convenience fan-out (a for-loop over the single-field RPC) — there is no
    atomicity: edits before the failing one stay applied and are NOT rolled
    back. On the first error this raises, reporting how many succeeded and which
    path failed so the agent can reconcile. On success the WHOLE resulting draft
    cfg is returned (via editor.get) rather than per-edit deltas, so the agent
    sees the full final state in one shot.
    """
    editor_id = str(arguments["editor_id"])
    edits = _coerce_pairs(arguments.get("edits"), field="edits", keys=("path", "value"))
    for i, edit in enumerate(edits):
        try:
            send_gui_rpc(
                "editor.set_field",
                {
                    "editor_id": editor_id,
                    "path": str(edit["path"]),
                    "value": edit["value"],
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"batch set_field failed at edits[{i}] (path={edit['path']!r}); "
                f"{i} edit(s) already applied and NOT rolled back: {exc}"
            ) from exc
    cfg = send_gui_rpc("editor.get", {"editor_id": editor_id})
    return {"applied": len(edits), "cfg": cfg}


def tool_gui_context_set_md_attrs(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Set several MetaDict attributes, fail-fast in order.

    Convenience fan-out over context.set_md_attr (no atomicity: attrs before the
    failing one stay set, NOT rolled back). On the first error this raises with
    the failing key and the count already applied.
    """
    attrs = _coerce_pairs(arguments.get("attrs"), field="attrs", keys=("key", "value"))
    for i, attr in enumerate(attrs):
        try:
            send_gui_rpc(
                "context.set_md_attr",
                {"key": str(attr["key"]), "value": attr["value"]},
            )
        except Exception as exc:
            raise RuntimeError(
                f"batch set_md_attr failed at attrs[{i}] (key={attr['key']!r}); "
                f"{i} attr(s) already set and NOT rolled back: {exc}"
            ) from exc
    return {"applied": len(attrs)}


def _await_operation_by_key(key: str, what: str, timeout: float) -> Dict[str, Any]:
    """Block until the latest operation for ``key`` settles; semantic result.

    Translates the agent's semantic name to the internal operation_id, awaits it
    via the off-main operation.await RPC, and returns a plain-language status.
    operation.await raises (PRECONDITION_FAILED) on failed/cancelled, surfaced to
    the agent as an error; here we shape the success message.
    """
    operation_id = _OP_BY_KEY.get(key)
    if operation_id is None:
        return {
            "status": "no_operation",
            "message": f"No in-flight operation for {what}.",
        }
    res = send_gui_rpc(
        "operation.await", {"operation_id": operation_id, "timeout": timeout}, timeout
    )
    return {"status": res.get("status", "finished"), "message": f"{what} completed."}


def _is_timeout_error(exc: RuntimeError) -> bool:
    """True when a send_gui_rpc RuntimeError carries the wire TIMEOUT code.

    send_gui_rpc formats failures as ``GUI Error (<code>): ...`` where <code> is
    the lowercase ErrorCode value (ErrorCode.TIMEOUT == 'timeout'). The literal is
    matched here to keep the bridge free of the errors-module import.
    """
    return "(timeout)" in str(exc)


def _start_op_with_short_wait(
    key: str,
    what: str,
    wait_seconds: float,
    product: "Callable[[], Dict[str, Any]]",
    pending_hint: str,
) -> Dict[str, Any]:
    """Wait briefly for a just-started async op, degrading to a handle on timeout.

    The start RPC must already have run (its operation_id captured into _OP_BY_KEY
    under ``key`` by send_gui_rpc). Awaits up to ``wait_seconds``:
    - settles in time -> ``{status:'finished', **product()}`` so the caller sees
      the op's resulting state immediately (device snapshot / tab snapshot / soc);
    - still running -> ``{status:'pending', message:<hint>}`` so the caller can
      use the matching wait tool. operation.await still raises on failure/cancel.

    Shared by device connect/disconnect/setup, run.start, and connect.start —
    every op that has both a fast and a slow mode gets the same degrade.
    """
    operation_id = _OP_BY_KEY.get(key)
    if operation_id is None:
        # No handle captured (op already settled synchronously) — report product.
        return {"status": "finished", **product()}
    try:
        send_gui_rpc(
            "operation.await",
            {"operation_id": operation_id, "timeout": wait_seconds},
            wait_seconds + 5.0,
        )
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            return {
                "status": "pending",
                "message": f"{what} still in progress after {wait_seconds}s; {pending_hint}",
            }
        raise  # genuine failure/cancellation surfaces as an error
    return {"status": "finished", **product()}


def _device_snapshot(name: str) -> Any:
    """Fetch one device's snapshot (now including its live ``info`` params)."""
    return send_gui_rpc("device.snapshot", {"name": name}).get("snapshot")


def tool_gui_device_wait_operation(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Block until the named device's current operation completes (semantic).

    Covers connect / disconnect / setup — whichever is the latest operation for
    the device. Returns status='finished' on success; raises on failure/
    cancellation; status='no_operation' if nothing is in flight for that device.
    """
    name = str(arguments["name"])
    timeout = float(arguments.get("timeout", 120.0))
    return _await_operation_by_key(
        f"device:{name}", f"Device {name!r} operation", timeout
    )


def tool_gui_run_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Start a run, waiting briefly for a fast (small reps/rounds) run to finish.

    A run has both modes — a tiny sweep finishes in well under a second, a big
    one takes minutes — so it degrades like device ops: settles in time ->
    {status:'finished', tab:<tab.snapshot>} (has_run_result reflects the result);
    still running -> {status:'pending'} (await with gui_run_wait or watch
    run_finished). send_gui_rpc attaches the version guard + captures the
    operation_id under tab:<tab_id>.
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("run.start", {"tab_id": tab_id})
    return _start_op_with_short_wait(
        f"tab:{tab_id}",
        f"Run on tab {tab_id!r}",
        wait_seconds,
        lambda: {"tab": send_gui_rpc("tab.snapshot", {"tab_id": tab_id})},
        f"await it with gui_run_wait(tab_id={tab_id!r}) or watch 'run_finished'.",
    )


def tool_gui_run_wait(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Block until the run on ``tab_id`` completes (semantic wait, mirrors
    gui_device_wait_operation). Raises on failure/cancellation."""
    tab_id = str(arguments["tab_id"])
    timeout = float(arguments.get("timeout", 600.0))
    return _await_operation_by_key(f"tab:{tab_id}", f"Run on tab {tab_id!r}", timeout)


def tool_gui_connect_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Connect the SoC, waiting briefly for the (usually fast) connect to land.

    Degrades like device ops: settles -> {status:'finished', view:<view.snapshot>};
    still running -> {status:'pending'} (await with gui_connect_wait). kind='mock'
    or kind='remote' with ip+port.
    """
    params: Dict[str, Any] = {"kind": str(arguments["kind"])}
    if "ip" in arguments:
        params["ip"] = str(arguments["ip"])
    if "port" in arguments:
        params["port"] = int(arguments["port"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("connect.start", params)
    # On a settled connect, fold in the SoC hardware summary so the agent sees
    # the board (channels / sample rates / freq ranges) in the same reply,
    # without a separate gui_soc_info round-trip.
    return _start_op_with_short_wait(
        "soc",
        "SoC connect",
        wait_seconds,
        lambda: {
            "view": send_gui_rpc("view.snapshot", {}),
            "soc": send_gui_rpc("soc.info", {}),
        },
        "await it with gui_connect_wait() or watch 'soc_changed'.",
    )


def tool_gui_connect_wait(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Block until the SoC connect completes (semantic wait). Raises on failure.

    On success also returns the SoC hardware summary (same as gui_soc_info)."""
    timeout = float(arguments.get("timeout", 120.0))
    result = _await_operation_by_key("soc", "SoC connect", timeout)
    if result.get("status") == "finished":
        result["soc"] = send_gui_rpc("soc.info", {})
    return result


def tool_gui_events_poll(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Drain up to ``max_events`` queued event pushes, blocking briefly. Also
    returns any buffered diagnostics so an idle agent that only polls still sees
    GUI error/info feedback."""
    timeout_seconds = float(arguments.get("timeout_seconds", 5.0))
    max_events = int(arguments.get("max_events", 16))
    if max_events < 1:
        max_events = 1
    out: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    with _EVENT_COND:
        while len(out) < max_events:
            while _DIAGNOSTIC_QUEUE:
                diagnostics.append(_DIAGNOSTIC_QUEUE.popleft())
            while _EVENT_QUEUE and len(out) < max_events:
                out.append(_EVENT_QUEUE.popleft())
            if out or diagnostics:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _EVENT_COND.wait(timeout=remaining)
    # Receiving an event is the point at which the agent "observed" a GUI change
    # (including async terminals like run/device/connect completion, whose version
    # bumps happen outside any RPC the bridge issues). Resync the baseline now —
    # done outside the _EVENT_COND lock since it makes a synchronous RPC — so a
    # following guarded op isn't blocked by the agent's own just-finished work.
    if out:
        _refresh_versions()
    return {"events": out, "diagnostics": diagnostics}


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


def tool_gui_dialog_screenshot(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Capture a currently-open dialog as PNG; optionally write to out_path."""
    params: Dict[str, Any] = {"dialog_name": str(arguments["dialog_name"])}
    res = send_gui_rpc("dialog.screenshot", params)
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
# Phase 81b tools — context queries / device queries
# ---------------------------------------------------------------------------


def tool_gui_device_connect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    name = str(arguments["name"])
    params: Dict[str, Any] = {
        "type_name": str(arguments["type_name"]),
        "name": name,
        "address": str(arguments["address"]),
    }
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.connect", params)  # operation_id captured into _OP_BY_KEY
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} connect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        f"await it with gui_device_wait_operation(name={name!r}) or watch 'device_changed'.",
    )


def tool_gui_device_disconnect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    name = str(arguments["name"])
    params: Dict[str, Any] = {"name": name}
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.disconnect", params)
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} disconnect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        f"await it with gui_device_wait_operation(name={name!r}) or watch 'device_changed'.",
    )


def tool_gui_device_setup(arguments: Dict[str, Any]) -> Dict[str, Any]:
    name = str(arguments["name"])
    updates = arguments.get("updates", {})
    if not isinstance(updates, dict):
        raise ValueError("'updates' must be an object")
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.setup", {"name": name, "updates": dict(updates)})
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} setup",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        f"await it with gui_device_wait_operation(name={name!r}) or watch 'device_setup_finished'.",
    )


# ---------------------------------------------------------------------------
# Generated tools — derived from dispatch.METHOD_REGISTRY (the wire SSOT)
# ---------------------------------------------------------------------------

# Methods that must NOT be auto-generated: they need extra client-side work
# (file writes, fan-out, MCP-side queues) or multi-field coercion, and are
# hand-written in _OVERRIDE_TOOLS below. Lifecycle tools (gui_connect/launch/
# stop/disconnect) have no RPC method and are hand-written too.
_NON_GENERATED_METHODS = frozenset(
    {
        # coerce_* → frozen request (multi-field) + mcp-side short-wait degrade
        # (await the returned operation_id briefly, then return snapshot or handle).
        "device.connect",
        "device.disconnect",
        "device.setup",
        # client-side file write of base64 PNG
        "view.screenshot",
        "dialog.screenshot",
        "tab.figure_screenshot",
        # fan-out / MCP-side queue (handled at the service, not the registry)
        "state.has_project",
        "state.has_context",
        "state.has_active_context",
        "state.has_soc",
        "events.subscribe",
        "events.unsubscribe",
        "events.list",
        # mcp<->RPC bookkeeping only; never an agent-facing tool (version numbers
        # must not surface to the agent — used internally by _refresh_versions).
        "resources.versions",
        # operation handle await: agent drives it via semantic wait tools (e.g.
        # gui_device_wait_operation), which translate name -> operation_id; the raw
        # by-id RPC is never an agent tool.
        "operation.await",
        # hand-written short-wait degrade (like device ops): a fast run / connect
        # returns its product, a slow one degrades to a handle (gui_run_wait /
        # gui_connect_wait).
        "run.start",
        "connect.start",
    }
)


def _tool_name_for(method: str, spec) -> str:
    return spec.tool_name or "gui_" + method.replace(".", "_")


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
    """Build an MCP forwarder that projects arguments into RPC params per spec.

    Required params are coerced and always sent. Optional params are coerced and
    sent only when present and non-None, matching the legacy hand-written
    forwarders (which used ``if arguments.get(k) is not None``).
    """
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
# Hand-written tools — lifecycle + overrides that the generator cannot express
# ---------------------------------------------------------------------------


_OVERRIDE_TOOLS: Dict[str, Dict[str, Any]] = {
    "gui_connect": {
        "handler": tool_gui_connect,
        "description": (
            "Connect the MCP bridge to an ALREADY-RUNNING GUI's TCP control port "
            "(default 8765). Errors if no GUI is listening there — use gui_launch "
            "to start one. Skip this if you used gui_launch with auto_connect=true "
            "(default)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": (
                        "TCP port of a running GUI control service (default 8765)"
                    ),
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
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
            "Launch the qubit-measure GUI as a NEW subprocess on a free TCP "
            "control port (default 8765), wait until it is ready, and optionally "
            "connect. Use this as the first step to start a session. Errors if "
            "the port is already in use (a stale GUI still running) — stop it "
            "first (gui_stop) or pass a different port; this avoids silently "
            "attaching to old code. By default auto_connect=true so gui_connect "
            "is called automatically."
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
            "Stop the GUI started by gui_launch, then disconnect the MCP socket. "
            "Closes gracefully via the app.shutdown RPC (the GUI's normal "
            "window-close: persist session, disconnect devices, cleanup) — no OS "
            "kill, cross-platform. Waits up to 'timeout' s for it to exit. If it "
            "does not and timeout_kill=false (default), reports it still running "
            "(re-run to retry); timeout_kill=true force-kills on timeout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait for graceful exit (default 10)",
                },
                "timeout_kill": {
                    "type": "boolean",
                    "description": (
                        "Force-kill the process if it has not exited within "
                        "'timeout' (default false — leave it running and report)"
                    ),
                },
            },
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
                    "description": "Event names to subscribe to, e.g. ['run_finished', 'tab_content_changed']",
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
    "gui_editor_subscribe": {
        "handler": tool_gui_editor_subscribe,
        "description": (
            "Subscribe to a cfg-editor session's change stream. After this, "
            "editor_changed (any field edit, by you OR a GUI user) and "
            "editor_closed (session ended: committed/discarded/tab_closed/"
            "evicted/disconnected) pushes for this editor_id are delivered to "
            "gui_events_poll. Get a tab's editor_id from gui_tab_snapshot. "
            "Essential when editing a tab cfg an interactive user may also touch: "
            "it lets you notice your value being overwritten."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_tab_snapshot or gui_editor_open)",
                }
            },
            "required": ["editor_id"],
        },
    },
    "gui_editor_unsubscribe": {
        "handler": tool_gui_editor_unsubscribe,
        "description": "Stop receiving a cfg-editor session's change/close pushes.",
        "inputSchema": {
            "type": "object",
            "properties": {"editor_id": {"type": "string"}},
            "required": ["editor_id"],
        },
    },
    "gui_editor_set_fields": {
        "handler": tool_gui_editor_set_fields,
        "description": (
            "Batch-apply several field edits to ONE cfg-editor session in order. "
            "Convenience fan-out over gui_editor_set_field — NOT atomic: it stops "
            "at the first failure (fail-fast) and edits applied before it are NOT "
            "rolled back; the error names the failing path and how many already "
            "applied. On success returns {applied, cfg} where 'cfg' is the WHOLE "
            "resulting draft (settable paths + current values, same as "
            "gui_editor_get) so you see the final state without per-edit deltas. "
            "Each edit's 'path' is dotted (see gui_tab_list_paths) and 'value' is "
            "a JSON scalar or an md-ref {__kind:eval, expr}, exactly as "
            "gui_editor_set_field."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_tab_snapshot or gui_editor_open)",
                },
                "edits": {
                    "type": "array",
                    "description": "Edits applied in order; each {path, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Dotted field path",
                            },
                            "value": {
                                "description": "JSON scalar or {__kind:eval, expr}"
                            },
                        },
                        "required": ["path", "value"],
                    },
                },
            },
            "required": ["editor_id", "edits"],
        },
    },
    "gui_context_set_md_attrs": {
        "handler": tool_gui_context_set_md_attrs,
        "description": (
            "Batch-set several MetaDict attributes in order. Convenience fan-out "
            "over gui_context_set_md_attr — NOT atomic: stops at the first failure "
            "(fail-fast), attrs set before it are NOT rolled back, and the error "
            "names the failing key plus how many already applied. Returns "
            "{applied} on success."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "attrs": {
                    "type": "array",
                    "description": "Attributes set in order; each {key, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "MetaDict key"},
                            "value": {"description": "JSON-safe value"},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            "required": ["attrs"],
        },
    },
    "gui_device_wait_operation": {
        "handler": tool_gui_device_wait_operation,
        "description": (
            "Block until the named device's current operation (connect / disconnect "
            "/ setup — whichever was started last) completes. Returns "
            "status='finished' on success; raises on failure/cancellation; "
            "status='no_operation' if nothing is in flight for that device. Use "
            "this after a gui_device_* tool returned status='pending'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device name"},
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 120)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_run_start": {
        "handler": tool_gui_run_start,
        "description": (
            "Start a run. Waits up to wait_seconds (default 1.0): a fast run "
            "(small reps/rounds) finishes in time -> {status:'finished', tab:{...}} "
            "(tab snapshot, has_run_result set); a slow run -> {status:'pending'} "
            "(await with gui_run_wait or watch 'run_finished'). Track in-flight "
            "progress with gui_run_progress."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_run_wait": {
        "handler": tool_gui_run_wait,
        "description": (
            "Block until the run on tab_id completes. status='finished' on "
            "success; raises on failure/cancellation; status='no_operation' if "
            "no run is in flight. Use after gui_run_start returned status='pending'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 600)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_connect_start": {
        "handler": tool_gui_connect_start,
        "description": (
            "Connect the SoC. kind='mock' (offline) or kind='remote' with ip+port. "
            "Waits up to wait_seconds (default 1.0): connects in time -> "
            "{status:'finished', view:{...}}; else {status:'pending'} (await with "
            "gui_connect_wait or watch 'soc_changed')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "'mock' or 'remote'"},
                "ip": {"type": "string", "description": "Board IP (remote)"},
                "port": {"type": "integer", "description": "Board port (remote)"},
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["kind"],
        },
    },
    "gui_connect_wait": {
        "handler": tool_gui_connect_wait,
        "description": (
            "Block until the SoC connect completes. status='finished' on success; "
            "raises on failure; status='no_operation' if none in flight. Use after "
            "gui_connect_start returned status='pending'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 120)",
                },
            },
        },
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
    "gui_dialog_screenshot": {
        "handler": tool_gui_dialog_screenshot,
        "description": (
            "Capture a currently-open dialog as a PNG image. "
            "dialog_name must be one of: setup, device, predictor, inspect, startup. "
            "Fails with PRECONDITION_FAILED if the named dialog is not currently open. "
            "If out_path (absolute path) is given, the image is written to disk and "
            "the base64 payload is omitted from the reply."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dialog_name": {
                    "type": "string",
                    "description": "One of: setup, device, predictor, inspect, startup",
                },
                "out_path": {
                    "type": "string",
                    "description": "Absolute path to save PNG (omits base64 from reply)",
                },
            },
            "required": ["dialog_name"],
        },
    },
    "gui_device_connect": {
        "handler": tool_gui_device_connect,
        "description": (
            "Register and connect a hardware device. Waits up to wait_seconds "
            "(default 1.0) for the connection: if it lands in time, returns "
            "{status:'finished', snapshot:{...}} (snapshot includes the device's "
            "live info params); otherwise {status:'pending'} — await it with "
            "gui_device_wait_operation or watch 'device_changed'. type_name is the "
            "driver class (e.g. 'YOKOGS200', 'SGS100A'); address is the VISA/GPIB/IP "
            "address. Set remember=true to persist the device across sessions."
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
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["type_name", "name", "address"],
        },
    },
    "gui_device_disconnect": {
        "handler": tool_gui_device_disconnect,
        "description": (
            "Disconnect a device. Waits up to wait_seconds (default 1.0): returns "
            "{status:'finished', snapshot:{...}} if it lands in time, else "
            "{status:'pending'} (await with gui_device_wait_operation or watch "
            "'device_changed'). Set remember=false to also remove it from "
            "persistent storage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "remember": {
                    "type": "boolean",
                    "description": "Keep device in persistent storage (default true)",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_device_setup": {
        "handler": tool_gui_device_setup,
        "description": (
            "Apply a device setup: patch the device's info fields via 'updates' "
            "(e.g. {'value': 0.5} to ramp a source's output value — this is the way "
            "to set an output value, ramped/cancellable, no separate set_value). "
            "Waits up to wait_seconds (default 1.0): returns {status:'finished', "
            "snapshot:{...}} if it lands in time, else {status:'pending'} (await "
            "with gui_device_wait_operation, or read progress via "
            "gui_device_active_setup). The device must already be connected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device name"},
                "updates": {
                    "type": "object",
                    "description": "Device info field updates (e.g. {'value': 0.5})",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name", "updates"],
        },
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
}


# Tool names served by the hand-written overrides rather than the generator:
# lifecycle tools (no RPC method) + the convenience/coercion/file-write tools.
_OVERRIDE_NAMES = frozenset(
    {
        "gui_connect",
        "gui_disconnect",
        "gui_launch",
        "gui_stop",
        "gui_device_connect",
        "gui_device_disconnect",
        "gui_device_setup",
        "gui_view_screenshot",
        "gui_dialog_screenshot",
        "gui_tab_figure_screenshot",
        "gui_state_check",
        "gui_events_subscribe",
        "gui_events_unsubscribe",
        "gui_events_list",
        "gui_events_poll",
        "gui_editor_subscribe",
        "gui_editor_unsubscribe",
        "gui_editor_set_fields",
        "gui_context_set_md_attrs",
        "gui_device_wait_operation",
        "gui_run_start",
        "gui_run_wait",
        "gui_connect_start",
        "gui_connect_wait",
    }
)


def _assemble_tools() -> Dict[str, Dict[str, Any]]:
    """Generated tools overlaid with the hand-written override subset.

    The generator owns every 1:1 RPC tool (schema from the ParamSpec SSOT);
    overrides own lifecycle / fan-out / file-write / coercion tools. The two
    sets are disjoint by name — a collision means an override leaked into the
    generatable set and is a programming error.
    """
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
    """Stop the GUI process when the MCP host disconnects (stdin EOF)."""
    try:
        # Best-effort graceful close on host disconnect; force-kill on timeout so
        # we don't leak a GUI process when the bridge goes away.
        tool_gui_stop({"timeout_kill": True})
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
                        "instructions": _SERVER_INSTRUCTIONS,
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
                        content = [{"type": "text", "text": text}]
                        # Piggyback (ADR-0013): drain background notifications
                        # buffered since the last tool call onto this result, so
                        # the agent learns about run/device/SoC changes + GUI
                        # diagnostics without a dedicated poll. The poll tool owns
                        # the event drain itself — don't double-drain there.
                        if name != "gui_events_poll":
                            pending = _drain_pending()
                            if pending["events"] or pending["diagnostics"]:
                                content.append(
                                    {
                                        "type": "text",
                                        "text": "notifications since last call:\n"
                                        + json.dumps(pending, indent=2),
                                    }
                                )
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {"content": content},
                        }
                    except Exception as e:
                        # GUI-side business errors (send_gui_rpc raises
                        # RuntimeError with an already-clear "GUI Error (code):
                        # message") carry no useful Python stack for the agent —
                        # the traceback is always the same forwarder→send_gui_rpc
                        # frames, pure noise. Strip it for those; keep the full
                        # traceback only for unexpected bridge-side failures,
                        # where the stack is the actual debugging signal.
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
