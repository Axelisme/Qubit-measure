#!/usr/bin/env python
"""MCP server bridge for ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live GUI's ``RemoteControlAdapter`` over a
single persistent TCP socket. Event push from the GUI is received by a
dedicated reader thread, parked in an internal queue and exposed to the LLM
piggybacked on tool replies (diagnostics) — the agent is not exposed to
resource-change events; it waits/polls operation handles instead.

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
# import cleanly. Importing under ``zcu_tools.gui.app.main`` runs that package's
# ``__init__`` which eagerly loads Qt; the bridge tolerates this (it never builds
# a QApplication), trading a heavier import for a single MethodSpec source of
# truth shared with the dispatcher.
# lib/zcu_tools/gui/app/main/services/remote -> lib
_LIB_DIR = Path(__file__).resolve().parents[6]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: importing ``zcu_tools.gui.app.main`` below eagerly loads Qt. If the
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

# NOTE: absolute imports (NOT relative) — this module is launched as a script
# (``python .../mcp_server.py`` per .mcp.json), so it has no parent package and a
# relative import would fail with "attempted relative import with no known
# parent package". The sys.path insert above makes the absolute path resolvable.
from zcu_tools.gui.app.main.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.main.services.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.gui.remote.param_spec import (  # noqa: E402
    JsonType,
    build_input_schema,
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
# v12: cfg introspection slimming (WIRE 14, Phase 120b). gui_tab_list_paths /
#      gui_editor_get gain under (sub-tree scope) + verbosity (compact default =
#      {path,kind,choices?}; full adds value/type; paths = bare list[str]).
#      gui_editor_set_fields returns {applied, valid} (no longer the whole cfg);
#      gui_analyze (was gui_analyze_start, v? sync) unchanged here. cfg_spec is
#      ref-only. Token-noise + WYSIWYG read split: edit returns success, read via
#      list_paths.
# v13: run/analyze replies fold in figure_path (Phase 120c-⑧). On a finished
#      run / completed analyze that produced a figure, the reply carries
#      figure_path — a PNG rendered to the cross-platform temp dir (gettempdir,
#      keyed by tab_id), NOT base64 — so the agent opens the file instead of a
#      separate gui_tab_figure_screenshot call. gui_analyze becomes a hand-written
#      override (was a generated forwarder) to attach it.
# v14: gui_analyze awaits the analyze operation (WIRE 15, Phase 120c). analyze is
#      an async worker; gui_analyze starts it (analyze.start now returns an
#      operation_id, captured under analyze:<tab_id>) then awaits via
#      _await_operation_by_key, so it stays synchronous to the agent AND the
#      figure is rendered (has_figure true) by the time figure_path is attached.
# v15: non-blocking per-domain poll (Phase 120c-1) — gui_run_poll / gui_device_poll
#      / gui_connect_poll map a zero-timeout operation.await onto finished /
#      running / failed / no_operation, keyed on the semantic name (tab_id /
#      device name / soc), no operation_id exposed (ADR-0005 kept). Lets an agent
#      check a slow op without blocking, replacing the run_finished event watch.
# v16: agent not exposed to events (Phase 120c-2). The GUI still emits its full
#      EventBus stream on the wire (RPC-side registration unchanged), but the
#      bridge drops every non-diagnostic event in _deliver_event and removes the
#      agent tools gui_events_subscribe/poll/list/unsubscribe +
#      gui_editor_subscribe/unsubscribe. Only diagnostics (GUI error/info push)
#      still piggyback tool replies. Resource-change awareness = the version
#      guard; async completion = gui_*_poll / gui_*_wait. No wire change.
# v17: stale-guard error names changed resources (WIRE 16, Phase 120c-3). The
#      version-guard PRECONDITION_FAILED carries data.stale (resource keys);
#      _describe_stale_keys translates them into agent language and folds them
#      into the error message ("… changed (the active context, this tab's cfg) …")
#      so the agent knows what to re-read. Version numbers stay mcp bookkeeping.
# v18: wait returns structured timeout (Phase 120c-4). gui_run_wait /
#      gui_device_wait_operation / gui_connect_wait no longer raise on timeout (a
#      bounded wait elapsing is expected, not a crash) — they return
#      {status:'finished'|'timed_out'|'no_operation', waited_seconds}. Both
#      timeout flavors (bridge socket TimeoutError, GUI-side "(timeout)") map to
#      timed_out; a genuine failed/cancelled still raises. No wire change.
# MCP 21: gui_launch gains a 'clean' arg (passes --clean to run_measure_gui → skip
#      restoring the persisted session at startup); tab.get_cfg_summary
#      description now warns its key shape (the '.value.' nesting) is read-only
#      and differs from the editable list_paths shape. No wire change.
# MCP 22: poll replies fold in progress + distinguish cancel (WIRE 20, Phase 129).
#      A 'running' gui_*_poll reply now carries the live bars (operation.progress
#      folded in) — gui_run_progress / gui_device_setup_progress tools are gone.
#      gui_*_poll reports user cancel as status:'cancelled' (was 'failed'), read
#      structurally from the wire reason via GuiRpcError.
# MCP 23: tool errors append the machine-readable reason tag (Phase 130) — a
#      precondition failure (no_run_result / no_project / …) now ends with
#      "reason: <tag>" so the agent can branch without parsing the prose. No wire
#      change (reason already on the wire envelope; only surfaced at mcp now).
MCP_VERSION = 23

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
     gui_context_new([bind_device][, clone_from]) to create one. The flux
     value/unit are NOT supplied directly: bind_device names a connected flux
     device (FakeDevice->unit none, YOKOGS200->unit A) and its *current* value
     is read (never set) to name the context; omit it for an unbound context.
     clone_from is the label of an existing context to clone ml/md from.
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
    — then gui_run_wait(tab_id) or gui_run_poll(tab_id). (gui_device_* and
    gui_connect_start degrade the same way.)
  - gui_analyze(tab_id) after a run — synchronous: on return the analysis is
    done (read gui_tab_get_analyze_result, no wait step). Then gui_save_data /
    gui_save_image / gui_save_result to persist.

Detecting completion — no events; wait or poll a handle:
  - A slow gui_run_start returns {status:'pending'}; then either gui_run_wait
    (blocks until done, raises on failure/cancellation) or gui_run_poll (returns
    immediately: 'finished'|'running'|'cancelled'|'failed'|'no_operation').
    'cancelled' is a user/agent cancel, distinct from 'failed'. A finished run/
    poll with a figure folds in figure_path. Same shape for devices
    (gui_device_wait_operation / gui_device_poll) and connect
    (gui_connect_wait / gui_connect_poll). gui_analyze is synchronous to you
    (it awaits internally) and returns figure_path.
  - While 'running', the poll reply carries the live progress bars (active,
    bars[token/format/maximum/value/percent/n/total]) — no separate progress
    tool. Don't busy-poll gui_run_running_tab in a sleep loop; use gui_run_poll.
  - GUI diagnostics still reach you UNSOLICITED: every tool reply piggybacks any
    {severity:'error'|'info', title, message} the GUI surfaced since your last
    call (e.g. "Data saved to …", a run-failure reason) under "notifications
    since last call". Watch severity=='error' for failures the GUI raised,
    including ones not tied to the call you just made. (Resource-change events
    are NOT exposed — stale detection is the version guard's job; it rejects a
    run/save/commit whose dependencies a GUI user changed under you.)

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

# Diagnostic push queue (FIFO, drop-oldest when full). The GUI still emits its
# full EventBus stream + diagnostics on the wire, but the agent is exposed only
# to diagnostics — user-facing error/info feedback ("Data saved to …", a run
# failure reason) that no version-guard or poll could surface. Resource-change
# events are dropped in _deliver_event (Phase 120c-2). Diagnostics piggyback on
# the next tool reply; _DIAGNOSTIC_COND guards the queue (notified on append).
_DIAGNOSTIC_QUEUE_MAX = 1024
_DIAGNOSTIC_QUEUE: Deque[Dict[str, Any]] = deque(maxlen=_DIAGNOSTIC_QUEUE_MAX)
_DIAGNOSTIC_COND = threading.Condition()

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
    "save.result": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
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
    "analyze.start": lambda p: f"analyze:{p.get('tab_id', '')}",
}

_READER_THREAD: Optional[threading.Thread] = None
_READER_STOP = threading.Event()

# GUI subprocess (when launched via gui_launch tool).
_GUI_PROC: Optional[subprocess.Popen] = None

# PID file for cross-session GUI process tracking.
_GUI_PID_FILE = Path(gettempdir()) / "zcu_tools_gui.pid"

# DEBUG log for GUIs we launch (OS temp dir, not the repo). gui_launch points
# run_measure_gui.py here so an agent can read the server-side event flow for debugging.
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
    # The GUI still emits its full EventBus stream over the wire (RPC-side
    # registration unchanged), but the agent is NOT exposed to resource-change
    # events: stale detection is the version-guard's job, and async completion is
    # polled via gui_run_poll / gui_*_poll. Only diagnostics (the GUI's own
    # error/info push — "Data saved to …", run-failure reason) reach the agent,
    # piggybacked on the next tool reply. Everything else is dropped here.
    if msg.get("event") != "diagnostic":
        return
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.append(msg)
        _DIAGNOSTIC_COND.notify_all()


def _drain_pending() -> Dict[str, list]:
    """Take the diagnostics buffered since the last drain — for piggyback on any
    tool result. The agent gets GUI error/info feedback without a dedicated poll
    call (resource-change events are not exposed; see _deliver_event)."""
    with _DIAGNOSTIC_COND:
        diagnostics = [m for m in _DIAGNOSTIC_QUEUE]
        _DIAGNOSTIC_QUEUE.clear()
    return {"diagnostics": diagnostics}


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


def _describe_stale_keys(keys: "list") -> "list[str]":
    """Translate stale resource keys into agent language (no version numbers).

    The server names which resource identities moved (e.g. 'tab:<uuid>:cfg',
    'context', 'device:flux'); the agent should not see the raw keyspace, so map
    them to phrases it can act on. Unknown shapes pass through verbatim (forward-
    compatible) rather than being dropped.
    """
    out: list[str] = []
    for raw in keys:
        key = str(raw)
        if key == "context":
            out.append("the active context (md/ml)")
        elif key == "soc":
            out.append("the SoC connection")
        elif key == "devices:__set__":
            out.append("the set of devices (one added/removed)")
        elif key.startswith("device:"):
            out.append(f"device {key[len('device:') :]!r}")
        elif key.startswith("editor:"):
            out.append("the cfg-editor draft")
        elif key.startswith("tab:"):
            # tab:<uuid>[:facet] — surface the facet, not the opaque uuid.
            facet = key.split(":", 2)[2] if key.count(":") >= 2 else ""
            label = {
                "cfg": "this tab's cfg",
                "result": "this tab's run result",
                "analyze": "this tab's analysis",
                "save_path": "this tab's save path",
            }.get(facet, "this tab")
            if label not in out:
                out.append(label)
        else:
            out.append(key)
    return out


class GuiRpcError(RuntimeError):
    """A wire-level GUI error, carrying the structured ``reason`` / ``code``
    tags alongside the human message. Subclasses RuntimeError so existing
    ``except RuntimeError`` sites keep working; callers that care (e.g. poll
    distinguishing cancelled vs failed) read ``.reason`` instead of the text."""

    def __init__(
        self, message: str, *, reason: Optional[str] = None, code: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.code = code


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
            # against the current table, and name (in agent language) which
            # resources changed so the agent knows what to re-read.
            _refresh_versions()
            data = err.get("data") or {}
            stale = _describe_stale_keys(data.get("stale", []))
            detail = f" ({', '.join(stale)})" if stale else ""
            raise RuntimeError(
                "GUI Error (PRECONDITION_FAILED): a resource you depend on was "
                f"changed in the GUI since you last saw it{detail}; review then "
                "retry"
            )
        msg = f"GUI Error ({err.get('code')}): {err.get('message')}"
        raise GuiRpcError(msg, reason=err.get("reason"), code=err.get("code"))
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
        return (
            f"Connected to GUI on 127.0.0.1:{port} with token authentication."
            + _wire_version_note()
        )
    return f"Connected to GUI on 127.0.0.1:{port}." + _wire_version_note()


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
    with _DIAGNOSTIC_COND:
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
    clean = bool(arguments.get("clean", False))
    repo_root = Path(__file__).parents[
        7
    ]  # lib/zcu_tools/gui/app/main/services/remote -> repo root
    # Use this MCP server's own interpreter to spawn the GUI: it is the venv
    # Python (with the 'gui' extra) that 'uv run' launched the bridge under, so
    # the GUI child has the same dependencies. This is also cross-platform —
    # avoids the hardcoded '.venv/bin/python' that does not exist on Windows
    # (which uses '.venv\Scripts\python.exe').
    python = sys.executable
    run_gui = repo_root / "script" / "run_measure_gui.py"

    if not run_gui.exists():
        raise FileNotFoundError(f"run_measure_gui.py not found at {run_gui}")

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
            f"(gui_stop, or kill the run_measure_gui.py holding port {port}), or launch "
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
    if clean:
        cmd += ["--clean"]

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
# Batch / dialog / view tools
# ---------------------------------------------------------------------------


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
    path failed so the agent can reconcile. On success returns
    ``{applied, valid}`` — the count applied and whether the resulting draft is
    valid. It does NOT echo cfg content (that would force a lowering pass which
    eagerly evaluates EvalValue); read the cfg with gui_tab_list_paths if needed.
    """
    editor_id = str(arguments["editor_id"])
    edits = _coerce_pairs(arguments.get("edits"), field="edits", keys=("path", "value"))
    valid = True
    for i, edit in enumerate(edits):
        try:
            res = send_gui_rpc(
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
        valid = bool(res.get("valid", True))
    return {"applied": len(edits), "valid": valid}


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
    """Block until the latest operation for ``key`` settles, or ``timeout`` s
    elapse; semantic result.

    Returns ``{status, waited_seconds[, message]}``: 'finished' (settled OK),
    'timed_out' (still running after the bounded wait — NOT a crash, no raise),
    or 'no_operation' (nothing tracked). ``waited_seconds`` is how long the wait
    actually blocked. A genuine failed/cancelled outcome still raises (the agent
    must see it as an error), distinct from a timeout.
    """
    operation_id = _OP_BY_KEY.get(key)
    if operation_id is None:
        return {
            "status": "no_operation",
            "message": f"No in-flight operation for {what}.",
        }
    start = time.monotonic()
    try:
        # Allow the bridge RPC a little slack beyond the op timeout so the
        # GUI-side timeout (a clean 'still running' signal) is what fires first,
        # not the socket round-trip ceiling.
        res = send_gui_rpc(
            "operation.await",
            {"operation_id": operation_id, "timeout": timeout},
            timeout + 5.0,
        )
    except TimeoutError:
        # Bridge socket round-trip ceiling hit — the op is still running. This is
        # an expected outcome of a bounded wait, not a crash: report it (no raise,
        # no traceback) with how long we actually waited so the agent can decide.
        return {
            "status": "timed_out",
            "waited_seconds": round(time.monotonic() - start, 3),
            "message": f"{what} still in progress after {timeout}s.",
        }
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            return {
                "status": "timed_out",
                "waited_seconds": round(time.monotonic() - start, 3),
                "message": f"{what} still in progress after {timeout}s.",
            }
        raise  # genuine failure/cancellation — surfaces to the agent as an error
    return {
        "status": res.get("status", "finished"),
        "waited_seconds": round(time.monotonic() - start, 3),
        "message": f"{what} completed.",
    }


def _poll_operation_by_key(key: str, what: str) -> Dict[str, Any]:
    """Non-blocking status of the latest operation for ``key`` (no event needed).

    Maps a zero-timeout await onto a plain status: 'finished' (settled OK),
    'running' (still in flight), 'failed'/'cancelled' (terminal error — does NOT
    raise here, unlike the blocking wait; poll reports it as a status), or
    'no_operation' (nothing tracked / already reaped). Lets an agent that
    started a slow op go do other work and check back without blocking.
    """
    operation_id = _OP_BY_KEY.get(key)
    if operation_id is None:
        return {"status": "no_operation", "message": f"No operation for {what}."}
    try:
        send_gui_rpc("operation.await", {"operation_id": operation_id, "timeout": 0.0})
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            # Still running — fold the live progress bars into the poll reply so
            # the agent watches progress without a separate tool call.
            progress = send_gui_rpc(
                "operation.progress", {"operation_id": operation_id}
            )
            return {
                "status": "running",
                "message": f"{what} still in progress.",
                **progress,
            }
        # terminal error — report as status rather than raising (poll is a query,
        # not an await). A user-initiated cancel is a distinct, non-failure
        # outcome: surface it as 'cancelled' so the agent need not parse the
        # message to tell "it crashed" from "I cancelled it" (the wire carries
        # reason='cancelled', read structurally via GuiRpcError.reason).
        reason = getattr(exc, "reason", None)
        if reason == "cancelled":
            return {"status": "cancelled", "message": f"{what} was cancelled."}
        return {"status": "failed", "message": f"{what}: {exc}"}
    return {"status": "finished", "message": f"{what} completed."}


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


def _figure_path_if_any(tab_id: str) -> "Optional[str]":
    """If the tab has a figure, render it to a temp PNG and return its path.

    Saves to the cross-platform temp dir (tempfile.gettempdir(), keyed by
    tab_id) and returns the path — NOT base64, so a run/analyze reply stays
    small. The agent opens the file (e.g. via Read) to see the plot, without a
    separate gui_tab_figure_screenshot call. Returns None when the tab has no
    figure yet (a data-only run), so the caller simply omits figure_path.
    """
    try:
        snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})
    except RuntimeError:
        return None
    interaction = snap.get("interaction", {}) if isinstance(snap, dict) else {}
    if not interaction.get("has_figure"):
        return None
    out_path = str(Path(gettempdir()) / f"zcu_tools_figure_{tab_id}.png")
    try:
        send_gui_rpc("tab.figure_screenshot", {"tab_id": tab_id, "out_path": out_path})
    except RuntimeError:
        return None  # figure raced away / render failed — just omit it
    return out_path


def _run_tab_summary(tab_id: str) -> Dict[str, Any]:
    """A run-finished tab summary: only {tab_id, interaction}. The full
    tab.snapshot also carries adapter_name / editor_id / save_paths, none of
    which change across a run — re-sending them every run is wasted tokens
    (the agent already has them from gui_tab_snapshot). figure_path is folded
    separately by _with_figure."""
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})
    interaction = snap.get("interaction", {}) if isinstance(snap, dict) else {}
    return {"tab_id": tab_id, "interaction": interaction}


def _with_figure(tab_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach figure_path to a finished run/analyze reply when a figure exists."""
    if result.get("status") == "finished":
        path = _figure_path_if_any(tab_id)
        if path is not None:
            result["figure_path"] = path
    return result


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


def tool_gui_device_poll(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Non-blocking status of the named device's latest operation (connect /
    disconnect / setup): finished / running / failed / no_operation."""
    name = str(arguments["name"])
    return _poll_operation_by_key(f"device:{name}", f"Device {name!r} operation")


def tool_gui_connect_poll(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Non-blocking status of the SoC connect: finished / running / failed /
    no_operation. On finished, also returns the SoC hardware summary."""
    del arguments
    result = _poll_operation_by_key("soc", "SoC connect")
    if result.get("status") == "finished":
        result["soc"] = _connect_soc_summary()
    return result


def tool_gui_run_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Start a run, waiting briefly for a fast (small reps/rounds) run to finish.

    A run has both modes — a tiny sweep finishes in well under a second, a big
    one takes minutes — so it degrades like device ops: settles in time ->
    {status:'finished', tab:<tab.snapshot>} (has_run_result reflects the result);
    still running -> {status:'pending'} (await with gui_run_wait or poll with
    gui_run_poll). send_gui_rpc attaches the version guard + captures the
    operation_id under tab:<tab_id>.
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("run.start", {"tab_id": tab_id})
    result = _start_op_with_short_wait(
        f"tab:{tab_id}",
        f"Run on tab {tab_id!r}",
        wait_seconds,
        lambda: {"tab": _run_tab_summary(tab_id)},
        f"await it with gui_run_wait(tab_id={tab_id!r}).",
    )
    # On a finished run that produced a figure, fold in figure_path so the agent
    # need not call gui_tab_figure_screenshot separately.
    return _with_figure(tab_id, result)


def tool_gui_run_wait(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Block until the run on ``tab_id`` completes (semantic wait, mirrors
    gui_device_wait_operation). Raises on failure/cancellation. On a finished
    run that produced a figure, the reply includes figure_path (a temp PNG)."""
    tab_id = str(arguments["tab_id"])
    timeout = float(arguments.get("timeout", 600.0))
    result = _await_operation_by_key(f"tab:{tab_id}", f"Run on tab {tab_id!r}", timeout)
    return _with_figure(tab_id, result)


def tool_gui_run_poll(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Non-blocking status of the run on ``tab_id``: finished / running / failed
    / no_operation. Lets the agent start a slow run, do other work, then check
    back without blocking (replaces watching the run_finished event). On a
    finished run with a figure, the reply includes figure_path."""
    tab_id = str(arguments["tab_id"])
    result = _poll_operation_by_key(f"tab:{tab_id}", f"Run on tab {tab_id!r}")
    return _with_figure(tab_id, result)


def tool_gui_analyze(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the tab's run result — synchronous to the agent.

    Analyze runs on a worker thread, but this tool awaits its completion before
    returning, so to the agent it is one synchronous call: on return the result
    and figure are ready. Folds in figure_path (a temp PNG) so the agent need
    not call gui_tab_figure_screenshot. Read the scalar fit summary with
    gui_tab_get_analyze_result. 'updates' optionally overrides analyze params.
    """
    tab_id = str(arguments["tab_id"])
    params: Dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    # Start (captures operation_id under analyze:<tab_id>, strips it from reply)
    # then block until the worker settles — figure is in State by then, so the
    # subsequent figure render sees has_figure=true.
    send_gui_rpc("analyze.start", params)
    result = _await_operation_by_key(
        f"analyze:{tab_id}", f"Analyze on tab {tab_id!r}", 600.0
    )
    return _with_figure(tab_id, result)


def _connect_soc_summary() -> Dict[str, Any]:
    """The SoC summary folded into a settled connect reply: only the human-
    readable ``description`` (print(soccfg)) + ``is_mock``. The structured ``cfg``
    (DAC/ADC channels, sample rates, freq ranges — ~2 KB) is NOT folded here; it
    duplicates the description and is rarely needed at connect time. Fetch it on
    demand with gui_soc_info."""
    info = send_gui_rpc("soc.info", {})
    return {
        "description": info.get("description"),
        "is_mock": info.get("is_mock"),
    }


def tool_gui_connect_start(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Connect the SoC, waiting briefly for the (usually fast) connect to land.

    Degrades like device ops: settles -> {status:'finished', view:<view.snapshot>};
    still running -> {status:'pending'} (await with gui_connect_wait). kind='mock'
    or kind='remote' with ip+port. The settled reply folds in only the SoC
    *description* + is_mock; call gui_soc_info for the structured cfg.
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
            "soc": _connect_soc_summary(),
        },
        "await it with gui_connect_wait() or poll gui_connect_poll().",
    )


def tool_gui_connect_wait(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Block until the SoC connect completes (semantic wait). Raises on failure.

    On success also returns the SoC hardware summary (same as gui_soc_info)."""
    timeout = float(arguments.get("timeout", 120.0))
    result = _await_operation_by_key("soc", "SoC connect", timeout)
    if result.get("status") == "finished":
        result["soc"] = _connect_soc_summary()
    return result


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
        f"await it with gui_device_wait_operation(name={name!r}) or gui_device_poll(name={name!r}).",
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
        f"await it with gui_device_wait_operation(name={name!r}) or gui_device_poll(name={name!r}).",
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
        # mcp<->RPC bookkeeping only; never an agent-facing tool (version numbers
        # must not surface to the agent — used internally by _refresh_versions).
        "resources.versions",
        # operation handle await: agent drives it via semantic wait tools (e.g.
        # gui_device_wait_operation), which translate name -> operation_id; the raw
        # by-id RPC is never an agent tool.
        "operation.await",
        # operation progress by id: internal — the poll tools fold its bars into
        # their reply, so the agent never calls it directly.
        "operation.progress",
        # hand-written short-wait degrade (like device ops): a fast run / connect
        # returns its product, a slow one degrades to a handle (gui_run_wait /
        # gui_connect_wait).
        "run.start",
        "connect.start",
        # hand-written to fold figure_path into the synchronous reply (analysis
        # almost always produces a figure; agent skips gui_tab_figure_screenshot).
        "analyze.start",
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
                "clean": {
                    "type": "boolean",
                    "default": False,
                    "description": "Start without restoring the previous persisted session (gui_state_v1.json is left untouched at startup; a normal close still flushes over it). Default false.",
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
    "gui_editor_set_fields": {
        "handler": tool_gui_editor_set_fields,
        "description": (
            "Batch-apply several field edits to ONE cfg-editor session in order. "
            "Convenience fan-out over gui_editor_set_field — NOT atomic: it stops "
            "at the first failure (fail-fast) and edits applied before it are NOT "
            "rolled back; the error names the failing path and how many already "
            "applied. On success returns {applied, valid} — the count applied "
            "and whether the resulting draft is valid. It does NOT echo cfg "
            "content (reading it would force a lowering pass that eagerly "
            "evaluates EvalValue); read the cfg with gui_tab_list_paths if "
            "needed. Each edit's 'path' is dotted (see gui_tab_list_paths) and "
            "'value' is a JSON scalar or an md-ref {__kind:eval, expr}, exactly "
            "as gui_editor_set_field."
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
            "/ setup — whichever was started last) completes or 'timeout' s elapse. "
            "Returns {status, waited_seconds}: 'finished' / 'timed_out' (still "
            "running — re-wait or gui_device_poll) / 'no_operation'. Raises only on "
            "a genuine failure/cancellation. Use after a gui_device_* tool returned "
            "status='pending'."
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
    "gui_device_poll": {
        "handler": tool_gui_device_poll,
        "description": (
            "Non-blocking status of the named device's latest operation (connect "
            "/ disconnect / setup): 'finished' / 'running' / 'cancelled' / "
            "'failed' / 'no_operation'. Returns immediately (cf. "
            "gui_device_wait_operation). While 'running' the reply carries the "
            "live progress bars (active, bars) — e.g. a setup ramp."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Device name"}},
            "required": ["name"],
        },
    },
    "gui_run_start": {
        "handler": tool_gui_run_start,
        "description": (
            "Start a run. Waits up to wait_seconds (default 1.0): a fast run "
            "(small reps/rounds) finishes in time -> {status:'finished', tab:{...}} "
            "(tab snapshot, has_run_result set); a slow run -> {status:'pending'} "
            "(await with gui_run_wait or gui_run_poll — a 'running' poll reply "
            "carries the live progress bars)."
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
            "Block until the run on tab_id completes or 'timeout' s elapse. "
            "Returns {status, waited_seconds}: status='finished' (done; figure_path "
            "folded in if any), 'timed_out' (still running after the wait — not an "
            "error, re-wait or gui_run_poll), or 'no_operation' (none in flight). "
            "Raises only on a genuine run failure/cancellation. Use after "
            "gui_run_start returned status='pending'."
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
    "gui_run_poll": {
        "handler": tool_gui_run_poll,
        "description": (
            "Non-blocking status of the run on tab_id: 'finished' / 'running' / "
            "'cancelled' / 'failed' / 'no_operation' ('cancelled' = a user/agent "
            "cancel, distinct from 'failed'). Unlike gui_run_wait this returns "
            "immediately — start a slow run, do other work, then poll back. While "
            "'running' the reply carries the live progress bars (active, bars); "
            "on a finished run with a figure it includes figure_path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_analyze": {
        "handler": tool_gui_analyze,
        "description": (
            "Analyze the tab's run result — SYNCHRONOUS: on return the analysis "
            "is complete. Read the scalar fit summary with "
            "gui_tab_get_analyze_result. The reply includes figure_path (a temp "
            "PNG of the fitted plot) when a figure was produced — open it (Read) "
            "instead of calling gui_tab_figure_screenshot. 'updates' optionally "
            "overrides analyze params (see gui_adapter_analyze_spec)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": "Analyze param updates",
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
            "gui_connect_wait or gui_connect_poll)."
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
            "Block until the SoC connect completes or 'timeout' s elapse. Returns "
            "{status, waited_seconds}: 'finished' (also returns the SoC summary) / "
            "'timed_out' (still connecting — re-wait or gui_connect_poll) / "
            "'no_operation'. Raises only on a genuine connect failure. Use after "
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
    "gui_connect_poll": {
        "handler": tool_gui_connect_poll,
        "description": (
            "Non-blocking status of the SoC connect: 'finished' / 'running' / "
            "'cancelled' / 'failed' / 'no_operation'. On finished, also returns "
            "the SoC hardware summary (same as gui_soc_info). Returns immediately."
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
            "gui_device_wait_operation or gui_device_poll. type_name is the "
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
        "gui_editor_set_fields",
        "gui_context_set_md_attrs",
        "gui_device_wait_operation",
        "gui_run_start",
        "gui_run_wait",
        "gui_run_poll",
        "gui_analyze",
        "gui_connect_start",
        "gui_connect_wait",
        "gui_connect_poll",
        "gui_device_poll",
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
                        # Piggyback (ADR-0013): drain GUI diagnostics buffered
                        # since the last tool call onto this result, so the agent
                        # gets the GUI's error/info feedback ("Data saved to …",
                        # a run-failure reason) without a dedicated poll. Only
                        # diagnostics ride here now — resource-change events are
                        # not exposed to the agent (Phase 120c-2).
                        pending = _drain_pending()
                        if pending["diagnostics"]:
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
                            # Surface the machine-readable reason tag (e.g.
                            # no_run_result / no_project) when the wire carried
                            # one, so the agent can branch on it without parsing
                            # the message prose. GuiRpcError.reason is set from
                            # the wire error envelope (Phase 129).
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
            sys.stderr.write(f"MCP Loop Exception: {e}\n{traceback.format_exc()}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
