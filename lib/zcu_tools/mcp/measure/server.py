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
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from tempfile import gettempdir
from typing import Any

logger = logging.getLogger(__name__)

# This bridge is launched standalone (``python .../mcp_server.py``), so the repo
# ``lib`` dir is not on sys.path by default. Add it so the wire-contract modules
# import cleanly. Importing under ``zcu_tools.gui.app.main`` runs that package's
# ``__init__`` which eagerly loads Qt; the bridge tolerates this (it never builds
# a QApplication), trading a heavier import for a single MethodSpec source of
# truth shared with the dispatcher.
# lib/zcu_tools/mcp/measure -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
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
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    _port_is_open,
    assemble_tools,
    generate_tools,
    resolve_connect_port,
    run_stdio_loop,
)
from zcu_tools.mcp.core.call_log import wrap_handler  # noqa: E402

# ``MCP_VERSION`` is this MCP bridge's own code revision (the mcp_server / tool
# layer, NOT the wire contract). It is REPORTED — never compared — in the version
# banner so an agent can confirm a reconnect picked up the bridge-side edits. Bump
# it on a meaningful bridge change you want to be able to spot a reload of,
# including pure mcp-side convenience changes (new override tools, fold tweaks,
# tool renames) that leave the wire contract untouched. A wire-contract change is
# tracked separately by WIRE_VERSION (see ``wire_version.py``); the two are
# independent. (Git history holds the per-version evolution.)
MCP_VERSION = 62  # Read-reveal guard refresh: a pure read only refreshes the version keys it reveals (not the whole table), so reading an unrelated resource no longer absorbs another resource's staleness; writes still whole-table refresh (self-block safe). MCP-side bookkeeping change, WIRE unchanged.

# ---------------------------------------------------------------------------
# Server usage instructions (returned in the MCP `initialize` result)
# ---------------------------------------------------------------------------

_SERVER_INSTRUCTIONS = """\
Drive a live qubit-measure GUI over a TCP control socket. This is the machine
contract (tool semantics + call rules); the operating manual (how to choose
wait/poll/background, hardware-safety, when to act vs ask the user) lives in the
run-measure-gui SKILL — follow it for workflow.

First, orient: call gui_overview once to read the live picture (project /
context / soc / open tabs / what is running), and re-read it whenever you need
the current state (gui_overview is the single orientation read — its 'state' field
carries the four readiness flags; there is no separate state-check tool). Do NOT
assume any state — the user may be driving the same GUI alongside you.

Tools are tiered: prefer RECOMMENDED; reach for ON-DEMAND when the bundles don't
fit; DEV tools are for debugging the GUI/MCP itself, not for measuring.

RECOMMENDED — the primary flow:
  - The 4-phase bundle (breadcrumb open -> run -> analyze_review -> commit):
    gui_tab_open (new tab + adapter guide) -> gui_tab_run (configure + run) ->
    gui_tab_analyze_review (analyze + writeback preview) -> gui_tab_commit
    (writeback + optional save). Each folds the cross-tool reads you would
    otherwise chain by hand.
  - Lifecycle / startup the bundles depend on: gui_overview (orient — its 'state'
    field has the four readiness flags has_project / has_context /
    has_active_context / has_soc), gui_launch / gui_bridge_connect, gui_soc_connect
    (kind='mock'|'remote'), gui_project_apply, gui_context_create /
    gui_context_switch.

ON-DEMAND — the fine-grained base tools, when a bundle doesn't fit:
  - Tabs + cfg: gui_adapter_list / gui_tab_new / gui_tab_snapshot;
    gui_tab_get_cfg (read tree) / gui_tab_set_cfg (batch write);
    gui_editor_open / gui_editor_get_cfg / gui_editor_set (batch) for non-tab
    editors (addressed by editor_id).
  - Run + analyze: gui_tab_run_start, gui_tab_analyze_start, gui_tab_post_analyze_start
    (each waits briefly then degrades to a handle). A FINISHED run/analyze reply
    (settled in the short wait) already carries 'figure' — the plot rendered to a
    temp PNG. After a pending->finished op, read the figure with
    gui_tab_get_current_figure and the fit summary with gui_tab_get_analyze_result /
    gui_tab_get_post_analyze_result (the generic wait/poll report only status).
    gui_tab_save (artifact + figure selectors) persists data and/or the figure and
    returns the resolved destinations; gui_tab_writeback_apply commits the draft.
  - Async handles: every degrading op returns a 'handle' in its START reply; drive
    it with the generic gui_op_poll(handle) / gui_op_wait(handle).
  - Devices / context / predictor / adapters: gui_device_*, gui_context_*,
    gui_predictor_*, gui_adapter_*.
ON-DEMAND — screenshot (a window/dialog grab; useful to show the user what the GUI
looks like, not part of the measurement loop):
  - gui_screenshot(target): 'window' grabs the whole main window, a dialog name
    (setup/device/predictor/inspect/startup) grabs that dialog; always writes a PNG
    file (never inline base64).
DEV — debugging the GUI/MCP itself (the version table + in-flight handles are
normally hidden from the operator; do NOT use these for measurement):
  - gui_debug_resource_versions (the per-resource optimistic-concurrency version
    table, for debugging stale-guard rejections; wire/gui/mcp *code* versions are
    in the gui_launch / gui_bridge_connect 'note' field, not here),
    gui_debug_operations (the in-flight operation handles, semantic key -> id).

Startup precondition: gui_overview's 'state' field must report all four flags true
before running experiments. Run/save require an active file-backed context;
save/analyze require an existing run result. A precondition violation returns
precondition_failed; editing cfg while a tab is running likewise.

gui_soc_connect is SYNCHRONOUS — NOT part of the async-handle family: it blocks
until the SoC is connected and returns {status:'finished', soc:{...}} in one call
(no handle). A remote board that is unreachable fails fast (~1s).

The async ops (run / analyze / post_analyze / device) share ONE contract — the
per-tool descriptions only name what each waits on; the mechanics live here.
Completion is detected by wait/poll on a handle, NOT events (nothing is pushed):
  - A short-wait START (gui_tab_run_start, gui_tab_analyze_start,
    gui_tab_post_analyze_start, gui_device_*) waits up to wait_seconds (default
    1.0): settles in time -> {status:'finished', handle, <product>}
    (gui_tab_run_start -> {tab, figure}; gui_tab_analyze_start -> {summary, figure};
    gui_tab_post_analyze_start -> {summary}; gui_device_* -> {snapshot}); a slow one
    degrades to {status:'pending', handle}. EITHER way the reply carries 'handle' —
    an opaque token you feed to the two generic drains below.
  - gui_op_poll(handle) returns immediately, NEVER raises: 'finished' | 'running' |
    'cancelled' | 'failed' | 'no_operation'. 'cancelled' (user/agent cancel) is
    distinct from 'failed'. While 'running' the reply folds the live progress bars
    (active, bars[token/format/percent]) — no separate progress tool.
  - gui_op_wait(handle, timeout=120) BLOCKS your whole turn until the op ends and
    RAISES only on genuine failure; returns {status, waited_seconds[, ...]}:
    'finished' (success), 'cancelled' (user/agent cancel — NOT a raise; read
    optional 'feedback' for the Stop reason), 'timed_out' (still running — NOT a
    failure — re-wait or switch to gui_op_poll). Because a wait holds the turn and
    nothing pushes a completion, reserve inline wait for ops you expect to finish
    quickly; for a long op (a big sweep, a slow ramp) either gui_op_poll
    (non-blocking — you check back) or call gui_op_wait from a BACKGROUND agent so
    your main loop stays free and the harness re-invokes you with the result.
  - gui_op_poll / gui_op_wait report ONLY status (+progress / feedback / cancel
    reason): they do NOT fold the figure / summary / snapshot. After a
    pending->finished op, read the product via its typed getter
    (gui_tab_get_current_figure, gui_tab_get_analyze_result,
    gui_tab_get_post_analyze_result, gui_device_snapshot).
  - CANCEL stays op-specific (no generic cancel): gui_tab_run_cancel (the running
    run), gui_tab_analyze_cancel(tab_id) (an interactive analyze), gui_device_cancel
    (a device op). Post-analysis has NO cancel (pure CPU recompute).
  - USER FEEDBACK WAKEUP (ADR-0025): gui_op_wait can return early with
    status='user_feedback' and a 'feedback' string while the op is STILL running.
    Treat the feedback as a HIGH-PRIORITY instruction and re-plan; you still hold
    the handle, so you may cancel (the op-specific cancel tool) or re-wait.
  - INTERACTIVE analyze (see gui_adapter_guide): the user marks the plot and
    clicks Done, so it never settles in the short wait — a 'pending' is EXPECTED;
    prompt the user, then gui_op_poll (do not block on gui_op_wait).

Diagnostic push: every tool reply piggybacks any {severity:'error'|'info', title,
message} the GUI surfaced since your last call, under "notifications since last
call" — UNSOLICITED, including failures not tied to the call you just made.
Resource-change events are NOT exposed.

Stale model (optimistic concurrency): a guarded op (run / save / editor_save) rejects
with precondition_failed when a dependency a GUI user changed under you moved
since you last observed it; the error names which resources to re-read. Re-read
then retry.

Call contract — read before issuing defensive/duplicate calls:
  - A failed call always raises; it never returns stale or partial data. One call
    is enough — never fire a backup copy of the same tool in the same turn.
  - Query tools (gui_*_list / _get* / _snapshot / _read / _inspect / _poll) are
    read-only and side-effect-free: safe to retry across turns, wasteful to
    duplicate within one.
  - Mutating tools have side effects and must be sent exactly once: gui_tab_run_start
    (a duplicate starts a SECOND run), gui_editor_set, gui_tab_new /
    gui_tab_close, gui_tab_save, gui_device_connect / _disconnect / _apply,
    gui_context_md_write / _md_delete / _ml_delete_* / _ml_rename_*, gui_editor_save.

Agent-to-user prompting: gui_prompt_user(message, timeout=600) opens a prompt
dialog for the user and BLOCKS your entire turn until the user replies, dismisses,
or the dialog times out. Returns {reason:'reply'|'dismiss'|'timeout', reply?}:
  - 'reply': user answered; read the reply string and act on it.
  - 'dismiss': user explicitly closed the prompt — do NOT ask again immediately.
  - 'timeout': no one was watching the GUI — do NOT wait again, continue or poll.
"""

# ---------------------------------------------------------------------------
# App-specific state (socket I/O lives in the shared McpBridge; what stays here
# is measure-gui's policy: the diagnostic queue, the optimistic-concurrency
# bookkeeping, and the async-operation handle map)
# ---------------------------------------------------------------------------

# Diagnostic push queue (FIFO, drop-oldest when full). The GUI still emits its
# full EventBus stream + diagnostics on the wire, but the agent is exposed only
# to diagnostics — user-facing error/info feedback ("Data saved to …", a run
# failure reason) that no version-guard or poll could surface. Resource-change
# events are dropped in _deliver_event. Diagnostics piggyback on the next tool
# reply; _DIAGNOSTIC_COND guards the queue (notified on append).
_DIAGNOSTIC_QUEUE_MAX = 1024
_DIAGNOSTIC_QUEUE: deque[dict[str, Any]] = deque(maxlen=_DIAGNOSTIC_QUEUE_MAX)
_DIAGNOSTIC_COND = threading.Condition()

# --- Optimistic-concurrency bookkeeping (policy lives here, mcp side) --------
#
# The agent never sees version numbers; they are bookkeeping between this mcp
# layer and the RPC server. ``_LAST_SEEN`` is the guard BASELINE: the per-key
# version each guarded op compares against (it attaches the subset it depends on
# as ``expected_versions``; the server compares atomically and rejects with
# PRECONDITION_FAILED if any moved — a concurrent, possibly human, edit).
#
# How the baseline is refreshed after a successful RPC depends on the method
# kind (the central safety property of this guard):
#   - WRITE / mutation (and any read we are not 100% sure about) -> whole-table
#     ``_refresh_versions()``. A write self-bumps the keys it touched, so
#     re-reading the whole table absorbs the op's own bump and the agent is not
#     later blocked by its own write (self-block safe). Whole-table is ALWAYS
#     safe — it can only fail to *improve* on the read-absorption issue below,
#     never produce a false reject.
#   - A pure read we are sure about -> ``_refresh_revealed_versions()``: refresh
#     ONLY the keys that read actually revealed (``_READ_REVEALS``), leaving
#     every other key's baseline untouched. This stops an unrelated read from
#     silently advancing resource X's baseline to current (which would let a
#     concurrent edit to X slip past the next write to X).
#
# Two intentional residual gaps (honestly narrower than the old "any RPC absorbs
# everything" model, but not zero):
#   (a) An UNRELATED WRITE between two operations still whole-table refreshes, so
#       it absorbs (masks) a concurrent edit to a resource it did not touch. Pure
#       reads no longer do this (unless they fall back to whole-table); writes
#       still do (whole-table is the price of self-block safety). Narrower than
#       before, not eliminated.
#   (b) ``_LAST_SEEN`` is PROCESS-GLOBAL and shared across every agent driving
#       this one bridge — there is NO per-agent isolation (the platform carries
#       no agent identity; per-agent staleness cursors are Not Planned upstream).
#       So two agents concurrently writing the same GUI do NOT get per-agent
#       staleness guarantees. True per-agent guarding would need each agent to
#       carry an opaque baseline cursor (not implemented).
#
# On a stale rejection we re-read the whole table so the next attempt carries
# fresh baselines (connect and stale-reject paths are always whole-table).
_LAST_SEEN: dict[str, int] = {}

# Dependency map (the single place that knows what each guarded op depends on).
# Patterns use {tab_id}/{editor_id} placeholders and a literal ``device:*`` that
# expands to every current device:* key. save.* does NOT depend on cfg — the
# saved content comes from the run result's own cfg_snapshot. tab.writeback_apply
# depends on the run+analyze results it recomputes from, plus context (it writes
# md/ml). Note: md/ml content edits bump the ``context`` version, so any op
# depending on ``context`` (tab.run_start / editor.commit / tab.writeback_apply) detects
# a concurrent md/ml change.
_GUARD_DEPS: dict[str, tuple[str, ...]] = {
    # ``device:*`` guards mutations of *existing* devices; ``devices:__set__``
    # guards *set membership* (a device added/removed since the agent last read
    # versions) which the per-member glob cannot reveal.
    "tab.run_start": (
        "tab:{tab_id}:cfg",
        "tab:{tab_id}",
        "soc",
        "context",
        "device:*",
        "devices:__set__",
    ),
    "tab.save_data": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "tab.save_image": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    # tab.save_post_image renders the post-analysis figure, so it depends on the
    # post-analysis result (not the primary run result) + the save path.
    "tab.save_post_image": ("tab:{tab_id}:post_analyze", "tab:{tab_id}:save_path"),
    "tab.save_result": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    # tab.save_set_paths writes the save_path override; guard on that very
    # resource so a concurrent save-path edit (GUI user / another agent) is
    # detected (read-modify-write optimistic concurrency).
    "tab.save_set_paths": ("tab:{tab_id}:save_path",),
    # tab.writeback_set / tab.writeback_apply edit + apply the persistent draft
    # (computed from run+analyze results, write md/ml). A concurrent rerun/
    # reanalyze or context edit must invalidate them.
    "tab.writeback_set": ("tab:{tab_id}:result", "tab:{tab_id}:analyze", "context"),
    "tab.writeback_apply": (
        "tab:{tab_id}:result",
        "tab:{tab_id}:analyze",
        "context",
    ),
    "editor.commit": ("editor:{editor_id}", "context"),
}

# Read-reveal map (the mirror of ``_GUARD_DEPS`` for the read side). Maps a PURE
# READ method to the version-key patterns it reveals — the keys whose current
# value the agent genuinely observed by making that read. After such a read we
# refresh ONLY these keys in ``_LAST_SEEN`` (``_refresh_revealed_versions``),
# leaving every other key's baseline untouched, so reading resource Y does not
# advance resource X's baseline and mask a concurrent edit to X.
#
# SAFETY — the only way a wrong entry here causes trouble is MASKING / false-ACCEPT:
# mapping a read to a key it does NOT fully reveal advances that key's baseline to
# current → expected==current → the server lets through a concurrent edit the agent
# never actually saw. This is a security hole (the guard is bypassed), NOT a
# false-reject (a too-high baseline never triggers a reject; the server only rejects
# when expected LAGS current). So the bar is: add an entry ONLY when you are 100%
# sure the read observes the COMPLETE state tracked by that key. When unsure, leave
# it OUT — an unmapped read falls back to the whole-table refresh (safe, no masking).
#
# Patterns use the same {tab_id}/{editor_id}/{name} placeholders and literal
# ``device:*`` glob as ``_GUARD_DEPS``; ``_expand_pattern_keys`` shares the
# expansion logic. ``devices:__set__`` is a literal key (set cardinality).
#
# Deliberately LEFT OUT (whole-table) and why:
#   - ALL context.* reads — the ``context`` key is a COARSE key tracking the entire
#     active md/ml CONTENT. No single content-read reveals the whole of it:
#     context.md_get returns only key names, md_get_attr returns one value,
#     ml_get returns only entry names/kinds, ml_list_roles returns the STATIC role
#     catalog (not content at all). Mapping any of these to ``context`` would advance
#     the content baseline on a read that observed only a projection, masking
#     concurrent edits to any md/ml entry the read did not touch. This is confirmed
#     over-broad, not merely uncertain — same root cause as context.labels/active.
#   - context.labels / context.active — neither is tracked by any version key
#     (a context-switch does not bump ``context``; there is no set-membership key
#     for the context list). No valid mapping exists.
#   - tab.snapshot — reveals many facets (editor_id, save paths, interaction,
#     existence flag); mapping it correctly would require intersecting multiple keys,
#     and a wrong mapping masks concurrent edits. Left whole-table.
#   - analyze/post-analyze reads (tab.get_analyze_result/params,
#     tab.get_post_analyze_result/params) — the analyze trio is server-gated on
#     has_run_result, NOT on a version key; the facet↔key mapping is uncertain.
#   - device.list — each entry returns only {name, type_name, status}; the
#     ``device:{name}`` key also bumps on info/value changes (set_device_info),
#     which device.list does NOT observe. Mapping to ``device:*`` would mask
#     concurrent info/value changes. Only set membership (devices:__set__) is
#     truly fully revealed by listing the set.
_READ_REVEALS: dict[str, tuple[str, ...]] = {
    # Reads the chosen tab's settable cfg tree: the whole cfg state for that tab.
    "tab.get_cfg": ("tab:{tab_id}:cfg",),
    # Reads one editing session's full draft: the whole editor state for that id.
    "editor.get": ("editor:{editor_id}",),
    # Reads one device's full cached snapshot: name/address/status/error/info —
    # the full state tracked by device:{name}. Slightly broader than just status
    # (it includes info/value), but there is no guarded op that consumes only
    # device:{name} without also consuming devices:__set__, so the masking risk
    # is bounded. Left in as the safe narrowing over whole-table.
    "device.snapshot": ("device:{name}",),
    # Lists the set of registered devices (membership only). device.list reveals
    # set membership (devices:__set__) but NOT device info/value (which also bumps
    # device:{name}), so mapping to device:* would mask concurrent info changes.
    # Only devices:__set__ (set cardinality) is fully revealed.
    "device.list": ("devices:__set__",),
}

# --- Async-operation handle bookkeeping (debug-only projection) ---------------
#
# Start ops (device.setup / tab.run_start) return an ``operation_id``; the agent
# now holds it directly as ``handle`` (exposed in the START reply, ADR-0026 §8)
# and drives gui_op_poll / gui_op_wait with it. So this map is NO LONGER on the
# wait/poll path. It survives only as the "latest handle per resource" projection
# that gui_debug_operations reads: semantic key -> latest operation_id for that
# resource. "Latest wins": starting overwrites the key.
#
# Lifecycle: entries are written on each START RPC (latest-wins) and are NEVER
# removed — they persist for the entire MCP server process lifetime. A stale key
# for a completed operation is expected and does not indicate an active operation.
_OP_BY_KEY: dict[str, int] = {}

# Which semantic key a start RPC's operation_id belongs to (param -> key).
# Device connect/disconnect/setup all key on the device name: "latest wins" means
# the most recent operation for that device is the one a wait tool awaits.
_OP_KEY_OF: dict[str, Callable[[dict[str, Any]], str]] = {
    "device.connect": lambda p: f"device:{p.get('name', '')}",
    "device.reconnect": lambda p: f"device:{p.get('name', '')}",
    "device.disconnect": lambda p: f"device:{p.get('name', '')}",
    "device.setup": lambda p: f"device:{p.get('name', '')}",
    "tab.run_start": lambda p: f"tab:{p.get('tab_id', '')}",
    "tab.analyze": lambda p: f"analyze:{p.get('tab_id', '')}",
    "tab.post_analyze": lambda p: f"post_analyze:{p.get('tab_id', '')}",
}


def _deliver_event(msg: dict[str, Any]) -> None:
    # The GUI still emits its full EventBus stream over the wire (RPC-side
    # registration unchanged), but the agent is NOT exposed to resource-change
    # events: stale detection is the version-guard's job, and async completion is
    # polled via gui_op_poll / gui_op_wait on the op handle. Only diagnostics (the GUI's own
    # error/info push — "Data saved to …", run-failure reason) reach the agent,
    # piggybacked on the next tool reply. Everything else is dropped here.
    if msg.get("event") != "diagnostic":
        return
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.append(msg)
        _DIAGNOSTIC_COND.notify_all()


def _drain_pending() -> dict[str, list]:
    """Take the diagnostics buffered since the last drain — for piggyback on any
    tool result. The agent gets GUI error/info feedback without a dedicated poll
    call (resource-change events are not exposed; see _deliver_event)."""
    with _DIAGNOSTIC_COND:
        diagnostics = [m for m in _DIAGNOSTIC_QUEUE]
        _DIAGNOSTIC_QUEUE.clear()
    return {"diagnostics": diagnostics}


# The shared transport bridge for this process. ``on_event=_deliver_event`` routes
# the GUI's event-push stream through measure-gui's diagnostic filter (only
# diagnostics are queued for piggyback; resource-change events are dropped). All
# socket I/O, the reader thread, RID routing, the launched subprocess + pid/log
# files live in the bridge; this module composes its ``send_rpc_raw`` with the
# version guard + operation tracking below.
_CONFIG = MCPBridgeConfig(
    app_name="gui",
    app_slug="measure",
    tool_prefix="gui_",
    default_port=8765,
    mcp_version=MCP_VERSION,
    wire_version=MCP_WIRE_VERSION,
    server_display_name="qubit-measure-control",
    server_instructions=_SERVER_INSTRUCTIONS,
    pid_file=Path(gettempdir()) / "zcu_tools_gui.pid",
    log_file=Path(gettempdir()) / "zcu_tools_gui_debug.log",
    run_script_name="run_measure_gui.py",
)

_BRIDGE = McpBridge(_CONFIG, on_event=_deliver_event)


def _read_version_table() -> dict[str, int] | None:
    """Read the full resource version table verbatim (no side effects).

    Pure read via ``resources.versions`` (the single read entry point), through
    ``send_rpc_raw`` to avoid recursing back into the guard. Returns the table or
    None on any failure — a missing table just means no guard this round.
    """
    try:
        resp = _BRIDGE.send_rpc_raw("resources.versions", {}, 5.0)
    except Exception:  # pragma: no cover — best-effort resync
        return None
    if not resp.get("ok", False):
        return None
    versions = resp.get("result", {}).get("versions")
    return versions if isinstance(versions, dict) else None


def _refresh_versions() -> None:
    """Replace the whole ``_LAST_SEEN`` baseline with the current version table.

    Called on connect, after a stale rejection, and after every successful WRITE
    (or unclassified read) — whole-table is always safe (self-block safe for a
    write, since it absorbs the op's own bump). Failures are swallowed.
    """
    versions = _read_version_table()
    if versions is not None:
        _LAST_SEEN.clear()
        _LAST_SEEN.update(versions)


def _expand_pattern_keys(
    patterns: tuple[str, ...], params: dict[str, Any], source_table: dict[str, int]
) -> dict[str, int]:
    """Expand dependency/reveal patterns into concrete {key: version} entries.

    Shared by ``_build_expected_versions`` (source = ``_LAST_SEEN`` baseline) and
    ``_refresh_revealed_versions`` (source = the just-read current table). The
    literal ``device:*`` glob expands to every ``device:`` key in ``source_table``
    at its value there; ``{tab_id}``/``{editor_id}``/``{name}`` placeholders fill
    from ``params``; a key absent from ``source_table`` reads as version 0.
    """
    out: dict[str, int] = {}
    for pattern in patterns:
        if pattern == "device:*":
            for key, version in source_table.items():
                if key.startswith("device:"):
                    out[key] = version
            continue
        key = pattern.format(
            tab_id=params.get("tab_id", ""),
            editor_id=params.get("editor_id", ""),
            name=params.get("name", ""),
        )
        out[key] = source_table.get(key, 0)
    return out


def _build_expected_versions(method: str, params: dict[str, Any]) -> dict[str, int]:
    """Resolve a guarded method's dependency patterns into expected versions.

    Policy lives here: expand the ``_GUARD_DEPS`` patterns against the current
    ``_LAST_SEEN`` baseline. Returns the subset of versions the op depends on;
    the server compares only these.
    """
    deps = _GUARD_DEPS.get(method)
    if not deps:
        return {}
    return _expand_pattern_keys(deps, params, _LAST_SEEN)


def _refresh_revealed_versions(method: str, params: dict[str, Any]) -> None:
    """Refresh ONLY the keys a pure read revealed into ``_LAST_SEEN``.

    The narrow counterpart to ``_refresh_versions``: after a read listed in
    ``_READ_REVEALS`` we re-read the whole table but copy back only the keys that
    read genuinely observed (its reveal patterns expanded against the just-read
    table), leaving every other key's baseline untouched. That is what stops an
    unrelated read from advancing a sibling resource's baseline (the read-
    absorption bug). The caller guarantees ``method in _READ_REVEALS``.
    """
    versions = _read_version_table()
    if versions is None:
        return
    revealed = _expand_pattern_keys(_READ_REVEALS[method], params, versions)
    _LAST_SEEN.update(revealed)


def _describe_stale_keys(keys: list) -> list[str]:
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
        self, message: str, *, reason: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.code = code


def _ensure_connected() -> None:
    """Lazily attach to the running GUI before the first guarded RPC.

    An agent should not have to call gui_bridge_connect by hand: the first time
    any gui_* tool reaches send_gui_rpc with no live socket, we resolve the running
    GUI's port via session discovery (the SAME path gui_bridge_connect takes) and
    attach. This is a measure-specific attach policy (session-discovery slug +
    resolve_connect_port), so it stays here rather than in the shared bridge,
    whose send_rpc_raw also serves the read-only apps.

    Only the control channel is attached — never soc.connect: choosing the SoC
    (mock vs remote) is the user's decision, not an auto-connect side effect.

    Fail-fast when no GUI is discoverable, with a message that no longer tells
    the agent to "call gui_bridge_connect first" (it is automatic now).
    """
    if _BRIDGE.is_connected:
        return
    port = resolve_connect_port(_CONFIG, None)
    # Fast-fail: probe the port before attempting a full TCP connect so a cold
    # start (no GUI listening) returns an actionable error immediately instead
    # of hanging ~30s until the socket timeout fires.  _port_is_open uses a
    # 0.5s timeout and is the same probe used by the bridge's launch path.
    if not _port_is_open(port):
        raise RuntimeError(
            "no running measure-gui found to attach to "
            f"(tried 127.0.0.1:{port}); start one with gui_launch."
        )
    try:
        _BRIDGE.connect(port)
    except RuntimeError as exc:
        raise RuntimeError(
            "no running measure-gui found to attach to "
            f"(tried 127.0.0.1:{port}); start one with gui_launch."
        ) from exc


def send_gui_rpc(
    method: str,
    params: dict[str, Any],
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Issue one RPC against the GUI; raises on error or timeout.

    If no socket is live yet, lazily attaches to the running GUI first
    (:func:`_ensure_connected`) so an agent never has to call gui_bridge_connect by
    hand. For guarded methods (run/save/commit) attaches ``expected_versions``
    from the mcp-side bookkeeping so the server can reject stale operations. On
    a stale rejection the version table is re-read so the agent's retry is fresh.

    After a successful round-trip the guard baseline is refreshed by method kind:
    a pure read listed in ``_READ_REVEALS`` refreshes only the keys it revealed
    (narrow); every other method — every write, plus any read we are not sure
    about — refreshes the whole table (self-block safe). See the ``_LAST_SEEN``
    comment for the safety rationale and the two residual gaps.
    """
    _ensure_connected()

    send_params = params
    if method in _GUARD_DEPS:
        send_params = dict(params)
        send_params["expected_versions"] = _build_expected_versions(method, params)

    resp = _BRIDGE.send_rpc_raw(method, send_params, timeout_seconds)
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
    # refresh the guard baseline so its own writes are not later seen as stale by
    # its own guarded ops. Split by method kind (see the _LAST_SEEN comment):
    #   - a pure read we are SURE about (in _READ_REVEALS) refreshes ONLY the keys
    #     it revealed, so reading resource Y leaves resource X's baseline alone —
    #     a concurrent edit to X is then still caught by the next write to X.
    #   - everything else (writes; reads we are not sure about) whole-table
    #     refreshes: a write self-bumps the keys it touched, so re-reading the
    #     whole table absorbs its own bump (self-block safe). Whole-table is
    #     always safe; it only fails to *narrow* the read-absorption window.
    if method in _READ_REVEALS:
        _refresh_revealed_versions(method, params)
    else:
        _refresh_versions()
    result = dict(resp.get("result", {}))
    # A start op returns an ``operation_id``. The agent now holds it directly: it
    # is renamed to ``handle`` (an opaque token the agent feeds to gui_op_poll /
    # gui_op_wait) and KEPT in the reply, so the START tool can fold it in
    # (ADR-0026 §8). _OP_BY_KEY is still written under the semantic key (latest
    # wins) but only as the debug-only latest-handle-per-resource projection that
    # gui_debug_operations reads — it is NO LONGER on the wait/poll main path.
    key_of = _OP_KEY_OF.get(method)
    if key_of is not None and "operation_id" in result:
        handle = int(result.pop("operation_id"))
        _OP_BY_KEY[key_of(params)] = handle
        result["handle"] = handle
    return result


# ---------------------------------------------------------------------------
# Connection lifecycle tools
# ---------------------------------------------------------------------------


def tool_gui_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    # connect attaches to a GUI that is ALREADY running (launch starts a new one),
    # so a missing GUI is the error case here. Omitting 'port' auto-discovers the
    # running GUI via its session file (covers the ephemeral-fallback case where
    # it is not on 8765), then falls back to the agreed-upon 8765.
    requested = arguments.get("port")
    if requested is not None and not isinstance(requested, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    port = resolve_connect_port(_CONFIG, requested)
    note = _BRIDGE.connect(port, arguments.get("token"))
    # Fold the situational overview into the connect reply so attaching alone
    # gives the agent the current picture (the same data gui_overview returns),
    # saving a follow-up probe. The socket is live by here, so the fan-out reads
    # resolve against the just-attached GUI.
    return {"note": note, "overview": _assemble_overview()}


def tool_gui_disconnect(arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    note = _BRIDGE.disconnect()
    # App-specific housekeeping: drop any buffered diagnostics — they belong to
    # the connection that just closed.
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.clear()
    return {"note": note}


def tool_gui_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    port = int(arguments.get("port", _CONFIG.default_port))
    token: str | None = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    clean = bool(arguments.get("clean", False))
    # lib/zcu_tools/mcp/measure -> repo root
    repo_root = Path(__file__).parents[4]
    # clean → run_measure_gui --clean (skip restoring the persisted session).
    extra_args = ["--clean"] if clean else None
    note = _BRIDGE.launch(repo_root, port, token, auto_connect, extra_args=extra_args)
    # Fold the situational overview only when auto_connect actually attached the
    # bridge — the fan-out reads need a live socket. With auto_connect=false the
    # GUI is up but not yet attached, so there is no live state to read.
    if _BRIDGE.is_connected:
        return {"note": note, "overview": _assemble_overview()}
    return {"note": note}


def tool_gui_stop(arguments: dict[str, Any]) -> dict[str, Any]:
    # Graceful close over the existing RPC channel (app.shutdown runs the GUI's
    # normal window-close path on its main thread, no OS signal), then await /
    # optionally force-kill. timeout_kill defaults False here (measure-gui prefers
    # leaving a slow-closing GUI alone for a retry rather than killing it).
    timeout = float(arguments.get("timeout", 10.0))
    timeout_kill = bool(arguments.get("timeout_kill", False))
    result = _BRIDGE.stop(
        timeout=timeout, timeout_kill=timeout_kill, shutdown_rpc="app.shutdown"
    )
    # The bridge's disconnect does not clear measure-gui's diagnostic queue; do it
    # here so a later session does not see the previous one's buffered messages.
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.clear()
    # Branch on the bridge's machine-readable outcome (no prose string-matching).
    return {"stopped": result["exited"], "note": result["note"]}


# ---------------------------------------------------------------------------
# Workflow tools (thin pass-through wrappers)
# ---------------------------------------------------------------------------


def _assemble_overview() -> dict[str, Any]:
    """One-shot situational overview of the live GUI, fanned out over existing
    read RPCs (no new wire method).

    Packs the readiness flags, the project identity, the active context label, the
    SoC summary, the open tabs (each with its adapter + running flag), the running
    tab and the user's currently-focused tab. ``active_tab`` is where the USER's
    eye is (a collaboration cue) — NOT the agent's operation target, which is
    always the explicit tab_id the agent passes.

    This overview is the single orientation SSOT: it folds in the project paths
    (result_dir/database_path) and the readiness flags, so the retired
    gui_state_check / gui_project_info tools have no separate surface.

    ``project`` is read from project.info only while a project is applied
    (project.info fast-fails no_project otherwise); it carries the full wire
    shape {chip_name, qub_name, res_name, result_dir, database_path}; ``null``
    when no project. ``is_mock`` is likewise read from soc.info only
    while connected (soc.info fast-fails without a SoC), so a not-yet-set-up GUI
    still yields a well-formed overview.
    """
    has_proj = send_gui_rpc("state.has_project", {}).get("value", False)
    has_ctx = send_gui_rpc("state.has_context", {}).get("value", False)
    has_act = send_gui_rpc("state.has_active_context", {}).get("value", False)
    has_soc = send_gui_rpc("state.has_soc", {}).get("value", False)

    project: dict[str, Any] | None = None
    if has_proj:
        info = send_gui_rpc("project.info", {})
        # Mirror the full project.info wire shape (long keys also match the other
        # tool-GUIs: fluxdep/dispersive/autofluxdep). Folding result_dir +
        # database_path here makes the overview the single orientation SSOT,
        # superseding the retired gui_project_info tool.
        project = {
            "chip_name": info.get("chip_name"),
            "qub_name": info.get("qub_name"),
            "res_name": info.get("res_name"),
            "result_dir": info.get("result_dir"),
            "database_path": info.get("database_path"),
        }

    soc: dict[str, Any] = {"connected": has_soc, "is_mock": None}
    if has_soc:
        soc["is_mock"] = send_gui_rpc("soc.info", {}).get("is_mock")

    tab_snaps = send_gui_rpc("tab.snapshot", {}).get("tabs", [])
    tabs = [
        {
            "tab_id": snap.get("tab_id"),
            "adapter": snap.get("adapter_name"),
            "is_running": bool(snap.get("interaction", {}).get("is_running", False)),
        }
        for snap in tab_snaps
    ]

    return {
        "state": {
            "has_project": has_proj,
            "has_context": has_ctx,
            "has_active_context": has_act,
            "has_soc": has_soc,
        },
        "project": project,
        "context": send_gui_rpc("context.active", {}).get("label"),
        "soc": soc,
        "tabs": tabs,
        "running_tab": send_gui_rpc("run.running_tab", {}).get("tab_id"),
        "active_tab": send_gui_rpc("view.snapshot", {}).get("active_tab_id"),
    }


def tool_gui_overview(arguments: dict[str, Any]) -> dict[str, Any]:
    """Situational overview of the live GUI (see _assemble_overview)."""
    del arguments
    return _assemble_overview()


# ---------------------------------------------------------------------------
# Batch / dialog / view tools
# ---------------------------------------------------------------------------


def _coerce_pairs(
    value: object, *, field: str, keys: tuple[str, str]
) -> list[dict[str, Any]]:
    """Validate a batch list of {k0, k1} dicts, fail-fast on shape errors.

    Validation happens up front (before any RPC) so a malformed item never lets
    a partial batch fire — keeping the failure boundary at 'nothing applied'
    rather than 'some applied'.
    """
    k0, k1 = keys
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field!r} must be a non-empty list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or k0 not in item or k1 not in item:
            raise ValueError(f"{field}[{i}] must be an object with {k0!r} and {k1!r}")
        out.append(item)
    return out


def _resolve_editor_id(arguments: dict[str, Any]) -> str:
    """Return the ``editor_id`` from arguments (required, fail-fast).

    Tab cfg editing goes through gui_tab_set_cfg / gui_tab_get_cfg. The editor
    tools (gui_editor_get_cfg / gui_editor_set) operate on non-tab editors
    (e.g. gui_editor_open on an ml entry) and require an explicit editor_id —
    tab_id is not accepted here.
    """
    editor_id = arguments.get("editor_id")
    if editor_id is None:
        raise ValueError(
            "supply 'editor_id' (tab cfg editing uses gui_tab_set_cfg / "
            "gui_tab_get_cfg; editor tools require an explicit editor_id)"
        )
    return str(editor_id)


def _fold_tab_editing_context(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold a fresh tab's editing context into ``reply``, in place.

    After tab.new the agent always reads tab.snapshot (for the editor_id) and
    tab.get_cfg (the settable cfg tree) before it can edit cfg. Folding those
    reads collapses the calls into one. Pure mcp-side fan-out over EXISTING wire
    reads. Reused by gui_tab_open (Phase ①). Adds {editor_id, tree}; the
    caller owns ``tab_id`` and ``adapter`` in ``reply``. tab.get_cfg returns a
    nested current-value tree, so the settable paths and their current values
    arrive in one ``tree``.
    """
    # tab.snapshot always returns {tabs: [...]}; a single tab_id yields a
    # one-element list (no shape-switch).
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})["tabs"][0]
    reply["editor_id"] = snap.get("editor_id")
    reply["tree"] = send_gui_rpc("tab.get_cfg", {"tab_id": tab_id}).get("tree")
    return reply


def tool_gui_editor_open(arguments: dict[str, Any]) -> dict[str, Any]:
    """Open a stateful editing session over an EXISTING ml entry, addressed later
    by ``editor_id``.

    Thin override over the editor.new RPC: folds the wire ``tree`` key to ``cfg``
    so every cfg view the agent reads (gui_tab_get_cfg, gui_editor_get_cfg, this
    open reply) uses the same ``cfg`` key. Returns ``{editor_id, cfg}`` — cfg is
    the nested current-value tree of the freshly-opened draft.
    """
    opened = send_gui_rpc(
        "editor.new",
        {
            "item_kind": str(arguments["item_kind"]),
            "from_name": str(arguments["from_name"]),
        },
    )
    return {"editor_id": opened["editor_id"], "cfg": opened.get("tree")}


def tool_gui_editor_get_cfg(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read an editing session's settable cfg as a nested current-value tree.

    Thin override over the editor.get RPC: folds the wire ``tree`` key to ``cfg``
    (the same key gui_tab_get_cfg uses). Returns ``{cfg}``.
    """
    editor_id = _resolve_editor_id(arguments)
    params: dict[str, Any] = {"editor_id": editor_id}
    prefix = arguments.get("prefix")
    if prefix is not None:
        params["prefix"] = str(prefix)
    got = send_gui_rpc("editor.get", params)
    return {"cfg": got.get("tree")}


def tool_gui_editor_set(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-set fields on ONE cfg-editor session, fail-fast in order.

    The editor is addressed by ``editor_id`` (from gui_editor_open). For tab cfg
    editing use gui_tab_set_cfg instead. Batch-only fan-out (a for-loop over the
    single-field editor.set_field RPC) — there is no atomicity: edits before the
    failing one stay applied and are NOT rolled back. On the first error this
    raises, reporting how many succeeded and which path failed so the agent can
    reconcile. On success returns ``{applied, valid}`` — the count applied and
    whether the resulting draft is valid. It does NOT echo cfg content (that would
    force a lowering pass which eagerly evaluates EvalValue); read the cfg with
    gui_editor_get_cfg if needed.
    """
    editor_id = _resolve_editor_id(arguments)
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
                f"batch set failed at edits[{i}] (path={edit['path']!r}); "
                f"{i} edit(s) already applied and NOT rolled back: {exc}"
            ) from exc
        valid = bool(res.get("valid", True))
    return {"applied": len(edits), "valid": valid}


def tool_gui_context_md_write(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-write MetaDict attributes, fail-fast in order (E5).

    Batch-only fan-out over context.md_set_attr — there is no atomicity: attrs
    before the failing one stay set and are NOT rolled back. On the first error
    this raises a message carrying ``applied_count`` (how many succeeded) and
    ``failed_index`` (the 0-based position of the failing attr) so the agent can
    reconcile (this surface never returns a structured partial result — across
    bridges, a failed write always raises). Complex md scalars round-trip via the
    {"__complex__": [re, im]} tag on the value. On success returns ``{applied}``.
    """
    attrs = _coerce_pairs(arguments.get("attrs"), field="attrs", keys=("key", "value"))
    for i, attr in enumerate(attrs):
        try:
            send_gui_rpc(
                "context.md_set_attr",
                {"key": str(attr["key"]), "value": attr["value"]},
            )
        except Exception as exc:
            raise RuntimeError(
                f"batch md write failed at attrs[{i}] (key={attr['key']!r}); "
                f"applied_count={i}, failed_index={i} — attrs before it stay set "
                f"and are NOT rolled back: {exc}"
            ) from exc
    return {"applied": len(attrs)}


def tool_gui_context_md_read(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read MetaDict attributes — the whole tree, or a named subset.

    Omit ``keys`` to read every attribute (fans out context.md_get for the key
    list, then context.md_get_attr for each); pass ``keys`` to read only that
    subset. Returns ``{values: {key: value}}`` — a map keyed by attribute name the
    agent indexes straight into. Reads are side-effect-free, so there is no
    partial-state concern; an unknown key in an explicit ``keys`` list fails fast
    (the underlying RPC raises invalid_params), never silently skipped. Complex md
    scalars arrive as the {"__complex__": [re, im]} tag (symmetric with write).
    """
    raw_keys = arguments.get("keys")
    if raw_keys is None:
        keys = [str(k) for k in send_gui_rpc("context.md_get", {}).get("keys", [])]
    elif isinstance(raw_keys, list):
        keys = [str(k) for k in raw_keys]
    else:
        raise ValueError("'keys' must be a list (or omitted to read the whole tree)")
    values: dict[str, Any] = {}
    for key in keys:
        res = send_gui_rpc("context.md_get_attr", {"key": key})
        values[key] = res.get("value")
    return {"values": values}


def tool_gui_context_md_delete(arguments: dict[str, Any]) -> dict[str, Any]:
    """Batch-delete MetaDict attributes, fail-fast in order.

    Batch-only fan-out over context.md_del_attr. Idempotent per key: deleting a
    key that does not exist is a no-op (not an error), matching delete semantics.
    No atomicity — keys before a (non-idempotent) failure stay deleted and are NOT
    rolled back; on the first such error this raises a message carrying
    ``applied_count`` and ``failed_index``. On success returns ``{deleted: [key]}``.
    """
    raw_keys = arguments.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("'keys' must be a non-empty list")
    keys = [str(k) for k in raw_keys]
    for i, key in enumerate(keys):
        try:
            send_gui_rpc("context.md_del_attr", {"key": key})
        except Exception as exc:
            raise RuntimeError(
                f"batch md delete failed at keys[{i}] (key={key!r}); "
                f"applied_count={i}, failed_index={i} — keys before it stay deleted "
                f"and are NOT rolled back: {exc}"
            ) from exc
    return {"deleted": keys}


def tool_gui_context_list(arguments: dict[str, Any]) -> dict[str, Any]:
    """List context labels plus the active one (the orientation read for contexts).

    Folds context.active + context.labels into one reply
    ``{active, has_active_context, labels: [str]}``. ``labels`` are plain strings:
    per-label unit/value are NOT available — unit/value are transient creation
    metadata (consumed by the device + the auto-label), never persisted (FC2). Only
    the active context's unit could be inferred, and even that is out of scope here.
    """
    del arguments
    active = send_gui_rpc("context.active", {}).get("label")
    labels = list(send_gui_rpc("context.labels", {}).get("labels", []))
    return {
        "active": active,
        "has_active_context": active is not None,
        "labels": labels,
    }


def tool_gui_context_ml_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read one ModuleLibrary entry's full cfg WITHOUT opening a tab.

    Opens a headless, gc-reclaimable cfg-editor draft on the existing ml entry
    (editor.new), reads its settable tree, then discards the draft (editor.discard).
    This is a pure read: opening/discarding a draft bumps no agent-visible resource
    version (only an *edit* would bump the editor version, and only editor.commit
    bumps context) — so it never disturbs concurrency guards. Returns ``{cfg}`` (the
    nested current-value tree, same shape as gui_tab_get_cfg). The draft is always
    discarded, even if the read raises.
    """
    item_kind = str(arguments["item_kind"])
    name = str(arguments["name"])
    opened = send_gui_rpc("editor.new", {"item_kind": item_kind, "from_name": name})
    editor_id = opened["editor_id"]
    try:
        return {"cfg": opened.get("tree")}
    finally:
        send_gui_rpc("editor.discard", {"editor_id": editor_id})


def _await_operation_by_handle(
    operation_id: int | None, what: str, timeout: float
) -> dict[str, Any]:
    """Block on a wire ``operation_id`` until it settles, or ``timeout`` s elapse;
    semantic result. The op-agnostic core of the generic gui_op_wait (ADR-0026 §8).

    Returns ``{status, waited_seconds[, message[, feedback]]}``:
    - 'finished': settled OK.
    - 'cancelled': user/agent cancelled the op. ``feedback`` carries the Stop
      reason when "Send & Stop" was used; absent on a plain cancel. NOT a raise
      (ADR-0025 §cancelled-wire — cancelled is a normal terminal outcome, not a
      crash; the agent reads feedback and re-plans).
    - 'user_feedback': a user-feedback string arrived before the op settled
      (ADR-0025). ``feedback`` carries the text; ``reason`` is 'user_feedback'.
      The operation is still running; the agent holds the handle and can re-await
      or cancel via the op-specific cancel tool.
    - 'timed_out': still running after the bounded wait — NOT a crash, no raise.
    - 'no_operation': no handle supplied / nothing tracked.
    ``waited_seconds`` is how long the wait actually blocked. A genuine
    ``failed`` outcome still raises (the agent must see it as an error).
    """
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
        raise  # genuine failure — surfaces to the agent as an error
    # Unwrap the structured reason from the wire payload (ADR-0025).
    reason = res.get("reason", "completed")
    waited = round(time.monotonic() - start, 3)
    if reason == "user_feedback":
        feedback = res.get("feedback") or ""
        return {
            "status": "user_feedback",
            "reason": "user_feedback",
            "feedback": feedback,
            "waited_seconds": waited,
            "message": (
                f"User sent feedback while {what} was running. "
                "Treat this as a high-priority instruction and re-plan. "
                "The operation is still running — you may gui_tab_run_cancel or re-await."
            ),
        }
    status = res.get("status", "finished")
    if status == "cancelled":
        # Structured cancellation: return status + optional Stop reason. Not a
        # raise — cancelled is a normal terminal outcome (ADR-0025 §cancelled-wire).
        out: dict[str, Any] = {
            "status": "cancelled",
            "waited_seconds": waited,
            "message": f"{what} was cancelled.",
        }
        feedback = res.get("feedback")
        if feedback:
            out["feedback"] = feedback
        return out
    return {
        "status": status,
        "waited_seconds": waited,
        "message": f"{what} completed.",
    }


def _slim_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Project the wire progress payload down to the fields an agent acts on.

    The wire ``operation.progress`` carries Qt-scaled counters (maximum / value /
    n / total) that the GUI's progress widget needs but the agent does not — it
    reasons over the human-readable ``format`` line and the ``percent`` only.
    Keep ``{token, format, percent}`` per bar; the precision/wire layer is left
    untouched (this folding is mcp-side policy).
    """
    bars = progress.get("bars", [])
    return {
        "active": progress.get("active", False),
        "bars": [
            {
                "token": b.get("token"),
                "format": b.get("format"),
                "percent": b.get("percent"),
            }
            for b in bars
        ],
    }


def _poll_operation_by_handle(operation_id: int | None, what: str) -> dict[str, Any]:
    """Non-blocking status of a wire ``operation_id`` (no event needed). The
    op-agnostic core of the generic gui_op_poll (ADR-0026 §8).

    Maps a zero-timeout await onto a plain status: 'finished' (settled OK),
    'running' (still in flight), 'failed'/'cancelled' (terminal error — does NOT
    raise here, unlike the blocking wait; poll reports it as a status), or
    'no_operation' (no handle supplied / already reaped). Lets an agent that
    started a slow op go do other work and check back without blocking.
    """
    if operation_id is None:
        return {"status": "no_operation", "message": f"No operation for {what}."}
    try:
        send_gui_rpc("operation.await", {"operation_id": operation_id, "timeout": 0.0})
    except RuntimeError as exc:
        if _is_timeout_error(exc):
            # Still running — fold the live progress bars into the poll reply so
            # the agent watches progress without a separate tool call, slimmed to
            # {token, format, percent} (Qt-scaled counters are dropped here).
            progress = send_gui_rpc(
                "operation.progress", {"operation_id": operation_id}
            )
            return {
                "status": "running",
                "message": f"{what} still in progress.",
                **_slim_progress(progress),
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


def tool_gui_op_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of any in-flight operation, by ``handle`` (ADR-0026 §8).

    ``handle`` is the opaque token a START tool (gui_tab_run_start /
    gui_tab_analyze_start / gui_tab_post_analyze_start / gui_device_*) returned in
    its reply. Maps a zero-timeout await onto a plain status — NEVER raises:
    finished | running | cancelled | failed | no_operation. While 'running' the
    reply folds the live progress bars (active, bars[token/format/percent]).
    Returns only the status (+progress): the op's product (figure/summary/snapshot)
    is read from the START finished reply or the matching typed getter.
    """
    handle = int(arguments["handle"])
    return _poll_operation_by_handle(handle, "operation")


def tool_gui_op_wait(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until any in-flight operation settles, by ``handle`` (ADR-0026 §8).

    ``handle`` is the opaque token a START tool returned. Blocks up to ``timeout``
    seconds. Returns {status, waited_seconds[, feedback]}: finished | cancelled
    (read optional 'feedback' for the Stop reason — NOT a raise) | user_feedback
    (op still running, agent re-plans then re-waits or cancels) | timed_out (still
    running — re-wait or gui_op_poll) | no_operation. RAISES only on a genuine
    failure. Returns only the status: the op's product (figure/summary/snapshot)
    is read from the START finished reply or the matching typed getter.
    """
    handle = int(arguments["handle"])
    timeout = float(arguments.get("timeout", 120.0))
    return _await_operation_by_handle(handle, "operation", timeout)


def _start_op_with_short_wait(
    key: str,
    what: str,
    wait_seconds: float,
    product: Callable[[], dict[str, Any]],
    pending_hint: str,
) -> dict[str, Any]:
    """Wait briefly for a just-started async op, degrading to a handle on timeout.

    The start RPC must already have run (its operation_id captured under ``key`` by
    send_gui_rpc, and also kept in the START reply as ``handle``). Awaits up to
    ``wait_seconds``:
    - settles in time -> ``{status:'finished', handle, **product()}`` so the caller
      sees the op's resulting state immediately (device snapshot / tab snapshot);
    - still running -> ``{status:'pending', handle, message:<hint>}`` so the caller
      can poll/wait the handle via gui_op_poll / gui_op_wait (ADR-0026 §8).
      operation.await still raises on failure/cancel.

    The reply always carries ``handle`` (pending AND finished) when a handle was
    captured, so the agent has one consistent token to drive gui_op_poll /
    gui_op_wait. Shared by device connect/disconnect/setup and tab.run_start.
    (soc.connect is excluded: it is synchronous and returns its product directly.)
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
                "handle": operation_id,
                "message": f"{what} still in progress after {wait_seconds}s; {pending_hint}",
            }
        raise  # genuine failure/cancellation surfaces as an error
    return {"status": "finished", "handle": operation_id, **product()}


def _device_snapshot(name: str) -> Any:
    """Fetch one device's snapshot (now including its live ``info`` params)."""
    return send_gui_rpc("device.snapshot", {"name": name}).get("snapshot")


def _run_tab_summary(tab_id: str) -> dict[str, Any]:
    """A run-finished tab summary: only {tab_id, interaction}. The full
    tab.snapshot also carries adapter_name / editor_id / save_paths, none of
    which change across a run — re-sending them every run is wasted tokens
    (the agent already has them from gui_tab_snapshot). To see the plot, call
    gui_tab_get_current_figure(tab_id)."""
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})["tabs"][0]
    interaction = snap.get("interaction", {}) if isinstance(snap, dict) else {}
    return {"tab_id": tab_id, "interaction": interaction}


def tool_gui_tab_run_start(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start a run, waiting briefly for a fast (small reps/rounds) run to finish.

    A run has both modes — a tiny sweep finishes in well under a second, a big
    one takes minutes — so it degrades like device ops: settles in time ->
    {status:'finished', tab:<tab.snapshot>, figure:<path>} (has_run_result set;
    figure is the run plot rendered to a temp PNG, the op's OWN visual result);
    still running -> {status:'pending', handle} (no figure yet; poll/wait the
    handle with gui_op_poll(handle=<handle>) / gui_op_wait(handle=<handle>)). The
    reply always carries 'handle'; send_gui_rpc attaches the version guard.
    NOTE: a generic gui_op_wait/poll only reports status — to see the plot after a
    pending->finished run, call gui_tab_get_current_figure(tab_id).
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("tab.run_start", {"tab_id": tab_id})
    reply = _start_op_with_short_wait(
        f"tab:{tab_id}",
        f"Run on tab {tab_id!r}",
        wait_seconds,
        lambda: {"tab": _run_tab_summary(tab_id)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait; "
        "see the plot when finished with gui_tab_get_current_figure"
        f"(tab_id={tab_id!r}).",
    )
    # The figure is this op's OWN visual result — fold it on FINISHED only
    # (a pending run has no settled plot yet). Failure is swallowed so a
    # plotting hiccup never masks an otherwise-good run reply.
    return _fold_finished_figure(tab_id, reply)


# Run wait/poll retired (ADR-0026 §8): the generic gui_op_wait / gui_op_poll drive
# the handle from gui_tab_run_start's reply. The figure that the old wait/poll
# folded on finished-after-degrade is now read via gui_tab_get_current_figure.


def _fold_analyze_params(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold the tab's analyze-params spec into a FINISHED run reply, in place.

    After a run the agent's next decision is analyze, whose knobs come from
    tab.get_analyze_params; surfacing them next to the run figure saves a round-trip.
    Only acts on ``reply['status'] == 'finished'`` (a pending run has no settled
    result to analyze). A fetch failure is swallowed (recorded as
    ``analyze_params: None``) so it never masks an otherwise-good run reply — the
    agent can still call gui_tab_get_analyze_params explicitly. The wire reply is
    {analyze_params: ...}; we surface that value under 'analyze_params'.
    """
    if reply.get("status") != "finished":
        return reply
    try:
        reply["analyze_params"] = send_gui_rpc(
            "tab.get_analyze_params", {"tab_id": tab_id}
        ).get("analyze_params")
    except Exception:
        reply["analyze_params"] = None
    return reply


def _analyze_summary_product(result_method: str, tab_id: str) -> dict[str, Any]:
    """Fold the analyze (or post-analyze) summary into a finished short-wait reply.

    When the short wait settles in time, the agent's next move is always to read
    the fit summary; folding it back here saves that extra round-trip. ``summary``
    mirrors the shape of the dedicated getter (gui_tab_get_analyze_result /
    gui_tab_get_post_analyze_result) — a dict for a FIT result, or None when the
    settled op produced no scalar summary (e.g. an INTERACTIVE pick that the user
    has not committed). The getters stay for re-fetch and for the wait/poll path
    (which does not run this product).
    """
    return {"summary": send_gui_rpc(result_method, {"tab_id": tab_id}).get("summary")}


def tool_gui_tab_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start analyze, waiting briefly (degrades like a run).

    Analyze has both modes — a FIT computes on a worker (usually finishes in well
    under a second), an INTERACTIVE pick waits for the USER to mark the plot and
    click Done (never settles in the short wait). So it degrades like gui_tab_run_start:
    settles -> {status:'finished', handle, summary, figure}; still running ->
    {status:'pending', handle} (poll/wait the handle with gui_op_poll(handle=<handle>)
    / gui_op_wait(handle=<handle>)). For an INTERACTIVE adapter (see
    gui_adapter_guide) a 'pending' is expected — prompt the user to do the pick,
    then poll. 'updates' optionally overrides analyze params. A finished FIT reply
    carries the fit 'summary' (same shape as gui_tab_get_analyze_result — analyze's
    OWN result, the *_err fields included) AND 'figure' — the fit plot rendered to
    a temp PNG (analyze's OWN visual result). Review the proposed writeback with
    gui_tab_writeback_list (not folded here; that fold lives in gui_tab_analyze_review).
    The reply always carries 'handle'; 'summary'/'figure' appear only on a finished
    FIT. After a pending->finished analyze read gui_tab_get_analyze_result and the
    plot with gui_tab_get_current_figure (a generic gui_op_wait/poll only reports
    status).
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    # Start (captures operation_id under analyze:<tab_id>, keeps it in the reply as
    # 'handle'), then wait briefly. A FIT usually finishes here; an INTERACTIVE pick
    # degrades to a handle the user/agent then drives to completion.
    send_gui_rpc("tab.analyze", params)
    reply = _start_op_with_short_wait(
        f"analyze:{tab_id}",
        f"Analyze on tab {tab_id!r}",
        wait_seconds,
        lambda: _analyze_summary_product("tab.get_analyze_result", tab_id),
        "poll/wait the returned handle with gui_op_poll / gui_op_wait; for an "
        "INTERACTIVE pick, prompt the user to mark the lines + click Done first, "
        f"then read gui_tab_get_analyze_result(tab_id={tab_id!r}).",
    )
    # The figure is analyze's OWN visual result — fold it on a FINISHED FIT reply.
    # An INTERACTIVE 'pending' has no settled plot yet (_fold_finished_figure is a
    # no-op on any non-finished status). writeback_preview stays in gui_tab_analyze_review.
    return _fold_finished_figure(tab_id, reply)


# Analyze wait/poll retired (ADR-0026 §8): the generic gui_op_wait / gui_op_poll
# drive the handle from gui_tab_analyze_start's reply. After a pending->finished
# analyze the agent reads gui_tab_get_analyze_result + gui_tab_get_current_figure.


def tool_gui_tab_post_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start the second-layer (post) analysis, waiting briefly (degrades like a run).

    Post-analysis runs on top of the tab's PRIMARY analyze result (e.g.
    single-shot multi-backend ge discrimination) and is FIT-only — it computes on
    a worker, so it usually settles in the short wait -> {status:'finished',
    handle, summary:{...}} (the fit summary is folded in, same shape as
    gui_tab_get_post_analyze_result, so the common read happens in one call). A
    slow one degrades to {status:'pending', handle} (poll/wait the handle with
    gui_op_poll(handle=<handle>) / gui_op_wait(handle=<handle>)). Fast-fails with
    precondition_failed when the tab has no primary analyze result yet — run
    gui_tab_analyze_start first. There is NO cancel for post-analysis: it is a pure
    CPU recompute with no stop point. 'updates' optionally overrides post params
    (see gui_tab_get_post_analyze_params). The reply always carries 'handle';
    'summary' appears only on finished. After a pending->finished post-analysis
    read gui_tab_get_post_analyze_result (a generic gui_op_wait/poll only reports
    status).
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    # Start (captures operation_id under post_analyze:<tab_id>, keeps it in the
    # reply as 'handle'), then wait briefly. A FIT-only worker usually finishes here.
    send_gui_rpc("tab.post_analyze", params)
    return _start_op_with_short_wait(
        f"post_analyze:{tab_id}",
        f"Post-analysis on tab {tab_id!r}",
        wait_seconds,
        lambda: _analyze_summary_product("tab.get_post_analyze_result", tab_id),
        "poll/wait the returned handle with gui_op_poll / gui_op_wait, then read "
        f"gui_tab_get_post_analyze_result(tab_id={tab_id!r}).",
    )


# Post-analyze wait/poll retired (ADR-0026 §8): the generic gui_op_wait /
# gui_op_poll drive the handle from gui_tab_post_analyze_start's reply. After a
# pending->finished post-analysis the agent reads gui_tab_get_post_analyze_result.


_SAVE_ARTIFACTS = frozenset({"data", "image", "both"})
_SAVE_FIGURES = frozenset({"primary", "post"})


def tool_gui_tab_save(arguments: dict[str, Any]) -> dict[str, Any]:
    """Save a tab's result data and/or figure; return the resolved destinations.

    Two orthogonal selectors:
      - artifact='data'|'image'|'both' (default 'both'): which artifacts to save.
      - figure='primary'|'post' (default 'primary'): which figure the 'image'
        artifact targets — the primary run/analyze plot or the post-analysis plot.

    Collision policy (NOT uniform across artifacts, by design — it mirrors the
    underlying savers):
      - DATA is written async with a uniqueness suffix: the resolved path is
        ``<stem>.hdf5`` with ``_N`` appended on collision, so a data save NEVER
        overwrites an existing file. The write itself runs async (data_async=true)
        — only the resolved path is known synchronously here.
      - IMAGE is written synchronously and OVERWRITES the destination if it exists
        (no uniqueness suffix).

    Returns {data_path, image_path, data_async, image_error} (all four keys always
    present, null when not applicable):
      - data_path: resolved data path when artifact included 'data', else null.
      - image_path: written image path when the image save succeeded, else null.
      - data_async: true when a data save was started (it completes off-turn).
      - image_error: the image-save error message when the image save FAILED, else
        null.

    Error boundary (the image step's policy depends on whether data was saved too):
      - artifact='both': a data save ran first, so EVERY image failure — a
        wire-level GuiRpcError AND a stale-version RuntimeError — is folded into
        image_error and NOT raised, so the already-resolved data_path is never
        lost (a completed sub-result is never discarded).
      - artifact='image': nothing to protect, so an image failure RAISES (Fast-Fail).
      - A precondition failure on the DATA save always raises (there is no
        data_error field — the data save is the agent's primary intent).
    """
    tab_id = str(arguments["tab_id"])
    artifact = str(arguments.get("artifact", "both"))
    figure = str(arguments.get("figure", "primary"))
    if artifact not in _SAVE_ARTIFACTS:
        raise ValueError(
            f"artifact must be one of {sorted(_SAVE_ARTIFACTS)}, got {artifact!r}"
        )
    if figure not in _SAVE_FIGURES:
        raise ValueError(
            f"figure must be one of {sorted(_SAVE_FIGURES)}, got {figure!r}"
        )

    comment = arguments.get("comment")
    data_path_arg = arguments.get("data_path")
    image_path_arg = arguments.get("image_path")

    out: dict[str, Any] = {
        "data_path": None,
        "image_path": None,
        "data_async": False,
        "image_error": None,
    }

    # DATA (async): a precondition failure here is a real Fast-Fail — let it raise
    # (there is no data_error field; the data save is the agent's primary intent).
    if artifact in ("data", "both"):
        data_params: dict[str, Any] = {"tab_id": tab_id}
        if data_path_arg is not None:
            data_params["data_path"] = str(data_path_arg)
        if comment is not None:
            data_params["comment"] = str(comment)
        out["data_path"] = send_gui_rpc("tab.save_data", data_params).get("data_path")
        out["data_async"] = True

    # IMAGE (sync, overwrites). The error policy depends on whether a data save
    # already succeeded in THIS call:
    #   - artifact in ('data','both'): a data save ran first, so ANY image failure
    #     (a wire-level GuiRpcError AND a stale-version RuntimeError) folds into
    #     image_error rather than raising — re-raising would discard the already
    #     resolved data_path, violating the "a completed sub-result is never lost"
    #     bundle contract.
    #   - artifact == 'image': nothing to protect, so a failure RAISES (Fast-Fail).
    if artifact in ("image", "both"):
        method = "tab.save_post_image" if figure == "post" else "tab.save_image"
        image_params: dict[str, Any] = {"tab_id": tab_id}
        if image_path_arg is not None:
            image_params["image_path"] = str(image_path_arg)
        data_was_saved = artifact == "both"
        try:
            out["image_path"] = send_gui_rpc(method, image_params).get("image_path")
        except Exception as exc:
            if not data_was_saved:
                raise
            out["image_error"] = str(exc)

    return out


def tool_gui_soc_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    """Connect the SoC SYNCHRONOUSLY and return its hardware summary.

    Unlike run / analyze / device ops, connect is no longer a degrading async
    handle: the soc.connect RPC runs the connect on the GUI's main thread and
    returns once the board is connected and all post-connect side effects are
    applied, so this is one blocking call with no _wait / _poll follow-up.
    kind='mock' (offline) or kind='remote' with ip+port. Returns
    {soc:{description, is_mock}}; call gui_soc_info for the structured cfg. A
    remote connect to an unreachable board fails fast (~1s).

    The SoC has no teardown (Pyro4-backed): there is no disconnect / reconnect /
    health-check tool — those are deferred (E3).
    """
    params: dict[str, Any] = {"kind": str(arguments["kind"])}
    if "ip" in arguments:
        params["ip"] = str(arguments["ip"])
    if "port" in arguments:
        params["port"] = int(arguments["port"])
    # Explicit per-call timeout ~2.0s: a small margin above make_soc_proxy's 1s
    # COMMTIMEOUT so the board-side 1s cap fires first with a clean error, rather
    # than this socket round-trip being cut. Do NOT rely on the 30s default here.
    result = send_gui_rpc("soc.connect", params, 2.0)
    return {"soc": result.get("soc")}


_SCREENSHOT_DIALOGS = ("setup", "device", "predictor", "inspect", "startup")
_SCREENSHOT_TARGETS = frozenset({"window", *_SCREENSHOT_DIALOGS})


def tool_gui_screenshot(arguments: dict[str, Any]) -> dict[str, Any]:
    """Capture the main window OR a named dialog as a PNG FILE; return its path.

    ``target`` switches what is grabbed:
      - target='window' → the WHOLE main window (client area + non-dialog floating
        widgets) via the view.screenshot wire method.
      - target=<dialog name> → that named dialog via dialog.screenshot.

    Mirrors gui_tab_get_current_figure / the old dialog grab: the convenience
    layer never returns inline base64 (a full-window grab would blow the
    tool-output token budget — the footgun this override removes). Both wire
    methods return base64 for raw consumers; we decode + write here. When out_path
    is omitted we synthesise a per-target temp path under gettempdir() (a single
    measure_window.png for the window — there is only one — and a per-dialog
    measure_dialog_<name>.png for a dialog), overwriting the previous grab.
    """
    import base64

    target = str(arguments["target"])
    # Client-side validation: reject an unknown target fast with the allowed set
    # rather than letting an invalid dialog name reach (and fail at) the wire.
    if target not in _SCREENSHOT_TARGETS:
        raise ValueError(
            f"target must be one of {sorted(_SCREENSHOT_TARGETS)}, got {target!r}"
        )
    out_path_arg = arguments.get("out_path")
    if target == "window":
        method, params = "view.screenshot", {}
        default_name = "measure_window.png"
    else:
        method, params = "dialog.screenshot", {"name": target}
        default_name = f"measure_dialog_{target}.png"
    out_path = (
        str(out_path_arg)
        if out_path_arg is not None
        else str(Path(gettempdir()) / default_name)
    )
    res = send_gui_rpc(method, params)
    png = base64.b64decode(res["png_b64"])
    Path(out_path).write_bytes(png)
    return {"bytes": res.get("bytes", len(png)), "saved_to": out_path}


# ---------------------------------------------------------------------------
# Dev / debug tools — for debugging the GUI/MCP itself, NOT the measurement
# workflow. They surface internals the operator face deliberately hides (the
# optimistic-concurrency version table, the in-flight operation handles). Pure
# reads, no mutation.
# ---------------------------------------------------------------------------


def tool_gui_debug_resource_versions(arguments: dict[str, Any]) -> dict[str, Any]:
    """Dump the full per-resource version table (DEV — debugging stale-guard).

    Reads resources.versions verbatim — the same table _refresh_versions consumes
    for the optimistic-concurrency guard, but here the version numbers are
    returned as-is instead of being kept as mcp<->RPC bookkeeping hidden from the
    operator. The only window into why a guarded op rejected (or did not reject)
    on a stale dependency. Returns the flat table {resource_key: version_int} —
    the GUI's live authoritative versions (a fresh read), NOT the mcp-side
    ``_LAST_SEEN`` cache. Side effect: like any ``send_gui_rpc`` round-trip the
    call resyncs ``_LAST_SEEN`` to the current table, but it never bumps a
    resource version (those move only on a real edit/run/writeback).

    Note: wire/gui/mcp *code* version握手 (WIRE_VERSION/GUI_VERSION/MCP_VERSION)
    lives in gui_launch / gui_bridge_connect's 'note' field — not in this table.
    """
    del arguments
    res = send_gui_rpc("resources.versions", {})
    return res.get("versions", {})


def tool_gui_debug_operations(arguments: dict[str, Any]) -> dict[str, Any]:
    """Dump the mcp-side per-key operation-handle cache (DEV).

    The ONLY source is the _OP_BY_KEY map (semantic key -> latest operation_id for
    that resource) — the debug-only "latest handle per resource" projection
    (ADR-0026 §8). It is NO LONGER on the wait/poll path (the agent drives
    gui_op_poll / gui_op_wait with the handle a START reply gave it); surfacing it
    here answers "what handle did the last run/analyze/setup on this tab/device
    get". 'latest wins' and entries are NEVER removed — a stale key for a completed
    op is normal, so the cache cannot itself certify liveness (the optional 'live'
    field is therefore omitted; the cache has no authoritative source for it).

    The live device enumeration is NOT duplicated here — gui_device_list_operations
    is the dedicated wire enumerator for that.

    Returns {handles: {key: {operation_id: int}}}.
    """
    del arguments
    # latest-wins, never-removed cache (ADR-0026 §8). No 'live' flag: the cache
    # has no liveness source — the wire enumerator lives in gui_device_list_operations.
    handles = {key: {"operation_id": op_id} for key, op_id in _OP_BY_KEY.items()}
    return {"handles": handles}


def _render_tab_figure(tab_id: str, out_path: str | None = None) -> dict[str, Any]:
    """Render ``tab_id``'s current figure to a PNG FILE (never inline base64).

    Drives the wire in out_path mode; synthesises a per-tab temp path under
    gettempdir() (overwriting the previous render of the same tab) when no path
    is given. Returns the wire reply ({saved_to, bytes}).
    """
    resolved = out_path or str(Path(gettempdir()) / f"measure_fig_{tab_id}.png")
    return send_gui_rpc(
        "tab.get_current_figure", {"tab_id": tab_id, "out_path": resolved}
    )


def _fold_finished_figure(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold the tab's current figure into a FINISHED run/analyze reply, in place.

    The operator looks at the plot after nearly every run/analyze, so saving the
    figure here collapses the separate gui_tab_get_current_figure call. Only acts
    when ``reply['status'] == 'finished'`` (a pending/cancelled/timed_out op has no
    settled figure to render). Renders to the per-tab temp PNG and adds
    ``figure: <saved_to>``. A render failure is swallowed (recorded as
    ``figure: None``) so a plotting hiccup never masks an otherwise-good result —
    the agent can still re-request the figure explicitly.
    """
    if reply.get("status") != "finished":
        return reply
    try:
        reply["figure"] = _render_tab_figure(tab_id).get("saved_to")
    except Exception:
        reply["figure"] = None
    return reply


def _fold_writeback_preview(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold the tab's writeback preview into a FINISHED analyze reply, in place.

    A FIT analyze recomputes the persistent writeback draft (the proposed md/ml/wf
    values + apply targets); surfacing it next to the fit summary lets the agent
    review the fit AND the proposed writeback in one call before
    gui_tab_writeback_apply (Phase ③). Only acts on ``reply['status'] ==
    'finished'`` (an INTERACTIVE 'pending' has not produced a draft yet). The wire
    tab.writeback_preview reply is {has_draft, items}; we surface that object
    verbatim under 'writeback_preview' (has_draft is false when no draft exists
    yet). Mirrors the figure/guide folds: a fetch failure is swallowed (omitted)
    so a preview hiccup never breaks the analyze reply — the agent can still call
    gui_tab_writeback_list.
    """
    if reply.get("status") != "finished":
        return reply
    try:
        reply["writeback_preview"] = send_gui_rpc(
            "tab.writeback_preview", {"tab_id": tab_id}
        )
    except Exception:
        pass
    return reply


# ---------------------------------------------------------------------------
# Bundle tools — the four-phase recommended flow
# (gui_tab_open -> run -> analyze_review -> commit)
#
# Each bundle composes several BASE operations into the agent's natural decision
# points and folds in the OTHER operations' outputs (the cross-tool folds the
# base tools deliberately do NOT carry). The base tools stay pure (each returns
# only its own result, least-surprise); the folding lives only here. The four
# names trace the operator's decision path: open (new tab + guide) -> run
# (configure + run) -> analyze_review (analyze + preview the writeback) ->
# commit (apply the writeback + optionally save).
# ---------------------------------------------------------------------------


def tool_gui_tab_open(arguments: dict[str, Any]) -> dict[str, Any]:
    """open (Phase ①): create a tab for ``adapter_name`` and fold its editing
    context + the adapter guide into one reply.

    Composes tab.new with the fan-out reads the agent always makes before editing
    cfg (tab.snapshot for editor_id, tab.get_cfg for the settable cfg tree)
    plus the adapter's orientation guide (adapter.guide).

    The guide is included BY DEFAULT so that any fresh agent context, sub-agent,
    or context-reset session receives the orientation text it needs without having
    to remember to pass a flag. The server no longer tracks whether the guide was
    previously sent — that decision belongs to the caller (the agent), which is the
    only one who knows whether its context already contains the guide.

    Pass ``skip_guide=true`` to suppress the guide fetch when you know the guide is
    already in your context (e.g. you already opened a tab for the same adapter
    earlier in this session) — the reply will carry ``guide_omitted: True`` to
    confirm the intentional omission. Callers who are unsure should NOT pass
    skip_guide=true; getting a duplicate guide wastes fewer tokens than missing it.

    Returns {tab_id, adapter, editor_id, tree, guide} by default;
    {tab_id, adapter, editor_id, tree, guide_omitted: True} when skip_guide=true.
    Configure + run with gui_tab_run(tab_id, edits).
    """
    adapter_name = str(arguments["adapter_name"])
    # skip_guide lets a caller that already has the guide in context suppress the
    # fetch; the default (False) ensures fresh/sub-agent contexts always get it.
    skip_guide = bool(arguments.get("skip_guide", False))
    tab_id = str(send_gui_rpc("tab.new", {"adapter_name": adapter_name})["tab_id"])
    reply: dict[str, Any] = {"tab_id": tab_id, "adapter": adapter_name}
    _fold_tab_editing_context(tab_id, reply)
    if not skip_guide:
        # Default path: fetch and include the full guide so the caller always has it.
        reply["guide"] = send_gui_rpc(
            "adapter.guide", {"adapter_name": adapter_name}
        ).get("guide")
    else:
        # Caller explicitly opted out; signal the intentional omission.
        reply["guide_omitted"] = True
    return reply


def tool_gui_tab_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """run (Phase ②): apply ``edits`` then run the existing ``tab_id``, STOPPING
    before analyze.

    Applies ``edits`` via gui_tab_set_cfg (single wire call carrying the whole
    batch) when given, then gui_tab_run_start. ``edits`` is an OPTIONAL ORDERED
    list of {path, value} (omit/empty runs the tab's current cfg); the order is
    preserved because a $ref switch must be applied before the paths it unlocks
    (a {path: value} map would lose that ordering). A finished reply is
    gui_tab_run_start's reply ({status, handle, tab}) with {figure, analyze_params}
    folded in — the run plot rendered to a temp PNG and the analyze knobs for this
    tab — so the agent sees them together. A slow run degrades to
    {status:'pending', handle, owed} where ``owed`` names what is not yet
    available: drive the handle with gui_op_wait(handle) / gui_op_poll(handle),
    then read the plot with gui_tab_get_current_figure (a generic wait/poll only
    reports status, it does not fold the figure). ``wait_seconds`` (default 1.0)
    bounds the short wait, same as gui_tab_run_start.

    It deliberately STOPS before analyze: a successful run is NOT a successful
    analyze — look at the figure, then gui_tab_analyze_review.
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    edits = arguments.get("edits") or []
    if not isinstance(edits, list):
        raise ValueError("'edits' must be an ordered list of {path, value} objects")
    # Apply edits via tab.set_cfg (tab-keyed batch; editor tools are now
    # editor_id-only). The ordered list is forwarded verbatim — a single wire call
    # carries the whole batch, preserving the agent's ref-switch-before-children order.
    if edits:
        send_gui_rpc(
            "tab.set_cfg",
            {
                "tab_id": tab_id,
                "edits": [{"path": str(e["path"]), "value": e["value"]} for e in edits],
            },
        )
    reply = tool_gui_tab_run_start({"tab_id": tab_id, "wait_seconds": wait_seconds})
    # A pending run owes the figure (read it via the getter after the handle
    # settles) — name it so the agent does not wait for a fold that never comes.
    if reply.get("status") != "finished":
        reply["owed"] = "figure (gui_tab_get_current_figure after the handle finishes)"
        return reply
    # The figure is already folded by tool_gui_tab_run_start; do NOT double-fold it
    # here. Only add analyze_params (the stage-specific fold).
    return _fold_analyze_params(tab_id, reply)


def tool_gui_tab_analyze_review(arguments: dict[str, Any]) -> dict[str, Any]:
    """analyze_review (Phase ③): analyze ``tab_id`` and fold the fit review into
    one reply.

    Composes gui_tab_analyze_start; a finished FIT reply ({status, handle, summary})
    gains {figure, writeback_preview} — the fit plot rendered to a temp PNG and the
    proposed writeback {has_draft, items} the fit produced — so the agent reviews
    the fit AND the proposed writeback in one call before gui_tab_commit. ``updates``
    optionally overrides the analyze params; ``wait_seconds`` (default 1.0) bounds
    the short wait. An INTERACTIVE analysis degrades to {status:'pending', handle,
    owed} (no folds — nothing settled to render/preview); prompt the user, then
    drive the handle with gui_op_wait(handle) / gui_op_poll(handle), then read
    gui_tab_get_analyze_result.
    """
    tab_id = str(arguments["tab_id"])
    analyze_args: dict[str, Any] = {
        "tab_id": tab_id,
        "wait_seconds": float(arguments.get("wait_seconds", 1.0)),
    }
    if arguments.get("updates") is not None:
        analyze_args["updates"] = arguments["updates"]
    reply = tool_gui_tab_analyze(analyze_args)
    # A pending (INTERACTIVE) analyze owes both folds — read them via the getters
    # after the handle settles. Name them so the agent does not wait in vain.
    if reply.get("status") != "finished":
        reply["owed"] = (
            "summary (gui_tab_get_analyze_result), figure "
            "(gui_tab_get_current_figure), writeback_preview "
            "(gui_tab_writeback_list) after the handle finishes"
        )
        return reply
    # The figure is already folded by tool_gui_tab_analyze; do NOT double-fold it
    # here. Only add writeback_preview (the stage-specific fold).
    return _fold_writeback_preview(tab_id, reply)


def tool_gui_tab_commit(arguments: dict[str, Any]) -> dict[str, Any]:
    """commit (Phase ④): apply the tab's writeback draft, optionally saving.

    Composes tab.writeback_apply (applies the currently-selected draft items;
    returns {applied_ids, written, context_version}); ``save`` selects an optional
    follow-up save of the same artifacts as gui_tab_save: 'none' (default,
    apply-only), 'data', 'image', or 'both'.

    fail-soft across the two steps (ADR-0026 §5.2): the apply runs first and is
    committed; if the follow-up save then fails, the applied writeback is NOT lost
    — the reply carries {status, applied_ids, written, context_version, saved,
    save_error?} where ``saved`` is the gui_tab_save result (or null) and
    ``save_error`` is the save failure message. ``status`` is 'committed' when both
    steps succeed (or save='none'), 'partial' when the apply succeeded but the save
    failed. This partial-status surface is ONLY for this cross-step bundle — a
    single wire/tool call (the apply itself, or gui_tab_save invoked on its own)
    still Fast-Fails by raising.
    """
    tab_id = str(arguments["tab_id"])
    save = str(arguments.get("save", "none"))
    if save not in ("none", "data", "image", "both"):
        raise ValueError(
            f"save must be one of ['none', 'data', 'image', 'both'], got {save!r}"
        )
    # The apply is the primary intent — a precondition failure here is a real
    # Fast-Fail (single wire call), so it raises before any save is attempted.
    apply_reply = dict(send_gui_rpc("tab.writeback_apply", {"tab_id": tab_id}))
    out: dict[str, Any] = {"status": "committed", **apply_reply, "saved": None}
    if save == "none":
        return out
    # The apply is already committed; a save failure must NOT discard applied_ids,
    # so it folds into save_error (status='partial') rather than raising — the
    # cross-step bundle contract. gui_tab_save itself still surfaces a per-artifact
    # image failure in-band (image_error) and raises only on a DATA precondition.
    try:
        out["saved"] = tool_gui_tab_save({"tab_id": tab_id, "artifact": save})
    except Exception as exc:
        out["status"] = "partial"
        out["save_error"] = str(exc)
    return out


def tool_gui_tab_get_current_figure(arguments: dict[str, Any]) -> dict[str, Any]:
    """Render the tab's current figure to a PNG FILE and return its path.

    The convenience layer always drives the wire in out_path mode so the agent
    never receives inline base64 (a large figure would blow the token budget —
    the footgun this override removes). When out_path is omitted we synthesise a
    per-tab temp path under gettempdir(), overwriting the previous render of the
    same tab. The raw wire method still supports base64 for non-MCP consumers.
    """
    tab_id = str(arguments["tab_id"])
    out_path_arg = arguments.get("out_path")
    return _render_tab_figure(
        tab_id, str(out_path_arg) if out_path_arg is not None else None
    )


# ---------------------------------------------------------------------------
# Context-query / device-query tools
# ---------------------------------------------------------------------------


def tool_gui_device_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    # type_name/address omitted => reconnect a remembered (memory-only) device,
    # reusing its stored type/address (E4: reconnect folded into connect). Both
    # wire methods key on device:{name} in _OP_BY_KEY, so the short-wait/handle
    # path is identical regardless of which one ran.
    type_name = arguments.get("type_name")
    address = arguments.get("address")
    if type_name is None and address is None:
        send_gui_rpc("device.reconnect", {"name": name})
    else:
        params: dict[str, Any] = {
            "type_name": str(type_name),
            "name": name,
            "address": str(address),
        }
        if "remember" in arguments:
            params["remember"] = bool(arguments["remember"])
        send_gui_rpc("device.connect", params)  # operation_id captured into _OP_BY_KEY
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} connect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


def tool_gui_device_disconnect(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    params: dict[str, Any] = {"name": name}
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.disconnect", params)
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} disconnect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


def tool_gui_device_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    updates = arguments.get("updates", {})
    if not isinstance(updates, dict):
        raise ValueError("'updates' must be an object")
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.setup", {"name": name, "updates": dict(updates)})
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} apply",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


# Seconds the off-main consumer backstop outlasts the dialog's QTimer so the
# dialog is the timeout SSOT (ADR-0025 §dialog-timeout): the QTimer fires at
# `timeout`, enqueues a Timeout event, and the consumer (waiting `timeout+slack`)
# reads it instead of pre-empting — closing the lost-reply gap at the boundary.
_NOTIFY_CONSUMER_SLACK: float = 10.0


def tool_gui_prompt_user(arguments: dict[str, Any]) -> dict[str, Any]:
    """Blocking request-reply prompt to the user. BLOCKS the turn until they respond.

    Serially composes notify.open (main thread: mint token + open dialog) and
    notify.await (off-main: block until Reply/Dismiss/QTimer). Neither method
    uses _OP_KEY_OF — not a start-op, no operation_id captured. There is NO poll
    or cancel: this is a single blocking request-reply, not an async handle.

    Returns {reason: 'reply'|'dismiss'|'timeout', reply?}: 'reply' carries the
    user's answer (a possibly-empty string); 'dismiss'/'timeout' carry no reply.
    Never raises on timeout or dismiss — those are expected outcomes (ADR-0025 §6).
    """
    message = str(arguments["message"])
    # Clamp to a sane minimum so the dialog always arms its QTimer (timeout<=0
    # would leave a never-auto-closing dialog with a fast consumer timeout — a
    # lost-reply window).
    timeout = max(float(arguments.get("timeout", 600.0)), 1.0)
    # Step 1: main-thread open — mints token + opens dialog (QTimer fires at
    # `timeout`; the dialog is the timeout SSOT, ADR-0025).
    open_result = send_gui_rpc(
        "notify.open", {"message": message, "timeout": timeout}, 30.0
    )
    token = int(open_result["token"])
    # Step 2: off-main await — blocks the IO worker until the dialog settles.
    # The consumer backstop MUST outlast the dialog's QTimer (timeout + slack) so
    # the dialog fires first and enqueues Timeout; a reply landing in the gap
    # would otherwise be lost.
    await_timeout = timeout + _NOTIFY_CONSUMER_SLACK
    await_result = send_gui_rpc(
        "notify.await",
        {"token": token, "timeout": await_timeout},
        await_timeout + 5.0,
    )
    # Forward the structured reason (reply/dismiss/timeout) verbatim; never raise.
    out: dict[str, Any] = {"reason": await_result.get("reason", "timeout")}
    reply = await_result.get("reply")
    if reply is not None:
        out["reply"] = reply
    return out


# ---------------------------------------------------------------------------
# Generated tools — derived from dispatch.METHOD_REGISTRY (the wire SSOT)
# ---------------------------------------------------------------------------

# Methods that must NOT be auto-generated: they need extra client-side work
# (file writes, fan-out, MCP-side queues) or multi-field coercion, and are
# hand-written in _OVERRIDE_TOOLS below. Lifecycle tools (gui_bridge_connect/
# launch/stop/gui_bridge_detach) have no RPC method and are hand-written too.
_NON_GENERATED_METHODS = frozenset(
    {
        # coerce_* → frozen request (multi-field) + mcp-side short-wait degrade
        # (await the returned operation_id briefly, then return snapshot or handle).
        "device.connect",
        "device.disconnect",
        "device.setup",
        # device.reconnect is wire-only: gui_device_connect routes here when
        # type_name/address are omitted (E4 — reconnect folded into connect). No
        # standalone gui_device_reconnect tool is generated.
        "device.reconnect",
        # client-side file write of base64 PNG — both raw-base64 wire methods are
        # excluded: dialog.screenshot and view.screenshot are only reachable via
        # the gui_screenshot override, which decodes + writes the PNG file
        # (never returning inline base64 to the agent).
        "dialog.screenshot",
        "view.screenshot",
        "tab.get_current_figure",
        # editor.new / editor.get are hand-written overrides so the agent sees
        # the {cfg} reply key (the wire handler returns {tree}; the override folds
        # the key) and the renamed tools gui_editor_open / gui_editor_get_cfg.
        "editor.new",
        "editor.get",
        # editor.set_field is wire-only now: the agent edits in batches via
        # gui_editor_set (the override fans out over it); there is no single-field
        # agent tool. The edits param is also array-of-objects (not a plain ARRAY),
        # which the generator cannot express.
        "editor.set_field",
        # same reason as editor.set_field: edits is array-of-{path,value} objects.
        "tab.set_cfg",
        # fan-out / MCP-side queue (handled at the service, not the registry)
        "state.has_project",
        "state.has_context",
        "state.has_active_context",
        "state.has_soc",
        # mcp<->RPC bookkeeping only; never an agent-facing tool (version numbers
        # must not surface to the agent — used internally by _refresh_versions).
        "resources.versions",
        # operation handle await: the generic gui_op_wait override drives it by the
        # handle a START reply gave the agent (ADR-0026 §8); the raw by-id RPC is
        # never an agent tool. Device ops use the same generic handle path now.
        "operation.await",
        # operation progress by id: internal — gui_op_poll folds its bars into the
        # poll reply, so the agent never calls it directly.
        "operation.progress",
        # hand-written short-wait degrade (like device ops): a fast run returns its
        # product, a slow one degrades to a handle (poll/wait via gui_op_poll /
        # gui_op_wait).
        "tab.run_start",
        # hand-written synchronous wrapper (passes an explicit ~2s timeout so the
        # board-side 1s COMMTIMEOUT fires first); not auto-generated so the override
        # gui_soc_connect tool is the only soc-connect surface.
        "soc.connect",
        # hand-written short-wait degrade (analyze is an async worker / interactive
        # pick; mirrors tab.run_start). To see the fit plot the agent calls
        # gui_tab_get_current_figure.
        "tab.analyze",
        # hand-written short-wait degrade (FIT-only worker, mirrors tab.analyze).
        "tab.post_analyze",
        # internal-only wire method: _assemble_overview calls it directly to fetch
        # the running tab id; no agent-facing MCP tool is generated (Phase 170b).
        "run.running_tab",
        # internal-only wire method: _assemble_overview folds the project identity
        # + paths into its `project` sub-object (the single orientation SSOT), so
        # the standalone gui_project_info tool is retired (keep the wire method).
        "project.info",
        # notify.open / notify.await are the two-RPC internals of gui_prompt_user;
        # the agent never calls them directly — only the hand-written tool is exposed.
        "notify.open",
        "notify.await",
        # internal-only wire methods (no agent tool): the spec + handler stay (the
        # dispatch key-match needs the spec) but no agent tool is generated.
        #   app.shutdown — gui_stop drives it via _BRIDGE.stop(shutdown_rpc=...).
        #   view.snapshot — _assemble_overview reads its active_tab_id for overview.
        "app.shutdown",
        "view.snapshot",
        # context.active + context.labels are folded into gui_context_list (one
        # orientation read); the individual tools are retired (keep the wire
        # methods — gui_context_list fans out over both).
        "context.active",
        "context.labels",
        # context.md_get / md_get_attr feed gui_context_md_read (whole-tree or
        # subset); md_set_attr feeds gui_context_md_write (batch); md_del_attr feeds
        # gui_context_md_delete (batch). The single-attr/list-keys tools are retired
        # — only the merged batch tools are agent-facing (keep the wire methods).
        "context.md_get",
        "context.md_get_attr",
        "context.md_set_attr",
        "context.md_del_attr",
        # The four save wire methods are folded into the single gui_tab_save tool
        # (artifact + figure selectors); the override fans out over them, so no
        # per-method agent tool is generated. The wire methods stay — gui_tab_save
        # and the stage4 bundle both call them directly (each keeps its own
        # optimistic-concurrency guard).
        "tab.save_data",
        "tab.save_image",
        "tab.save_post_image",
        "tab.save_result",
    }
)


# Tool generation (coerce / forward / per-spec schema) is the shared
# ``generate_tools`` helper; measure-gui's guarded ``send_gui_rpc`` is injected as
# the send_fn so generated forwarders carry the version guard + operation capture.


# ---------------------------------------------------------------------------
# Hand-written tools — lifecycle + overrides that the generator cannot express
# ---------------------------------------------------------------------------


_OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_bridge_connect": {
        "handler": tool_gui_connect,
        "description": (
            "Attach the MCP control BRIDGE to an already-running GUI — NOT the "
            "SoC (gui_soc_connect) nor an instrument (gui_device_connect). "
            "Attaches to an ALREADY-RUNNING GUI's TCP control port. "
            "OPTIONAL: the first gui_* call already auto-attaches to the running "
            "GUI via session discovery — call this only to pin a specific 'port' or "
            "pass a 'token'. Omit 'port' to auto-discover the running GUI via its "
            "session file (covers the case where it fell back off port 8765), "
            "falling back to 8765 if none is found. Errors if no GUI is listening — "
            "use gui_launch to start one. Returns {note, overview}: 'overview' is "
            "the same situational picture gui_overview returns, so attaching gives "
            "you the current state in one call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": (
                        "TCP port of a running GUI control service. Omit to "
                        "auto-discover (then fall back to 8765)."
                    ),
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
        },
    },
    "gui_bridge_detach": {
        "handler": tool_gui_disconnect,
        "description": (
            "Detach the MCP control bridge (closes socket only; does NOT stop "
            "the GUI — use gui_stop)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_launch": {
        "handler": tool_gui_launch,
        "description": (
            "Launch the qubit-measure GUI as a NEW subprocess on TCP control "
            "port 'port' (default 8765), wait until it is ready, and optionally "
            "connect. Use this as the first step to start a session. Errors if "
            "the port is already in use (a stale GUI still running) — stop it "
            "first (gui_stop) or pass a different port; this avoids silently "
            "attaching to old code. By default auto_connect=true so the bridge "
            "is attached automatically (gui_bridge_connect). Returns {note} — "
            "plus 'overview' (the same picture gui_overview returns) when "
            "auto_connect attached the bridge."
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
                    "description": "Optional shared auth token (also passed to gui_bridge_connect if auto_connect=true)",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Attach the bridge automatically (gui_bridge_connect) once port is ready (default true)",
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
            "Stops ONLY a GUI this server launched; a connect-only session has "
            "nothing to stop. Stops the GUI started by gui_launch, then "
            "disconnects the MCP socket. Closes gracefully via the app.shutdown "
            "RPC (the GUI's normal window-close: persist session, disconnect "
            "devices, cleanup) — no OS kill, cross-platform. Waits up to "
            "'timeout' s for it to exit. Returns {stopped, note}: 'stopped' is "
            "true once the process is gone (graceful exit or force-kill), false "
            "when a graceful close timed out and was left running (re-run to "
            "retry); timeout_kill=true force-kills on timeout."
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
    "gui_overview": {
        "handler": tool_gui_overview,
        "description": (
            "One-shot SITUATIONAL OVERVIEW of the live GUI and the single "
            "orientation SSOT — call any time to re-orient (it folds in the "
            "readiness flags and the project paths, so there is no separate "
            "state-check or project-info tool). Packs (from existing read RPCs): "
            "state (the four readiness flags has_project / has_context / "
            "has_active_context / has_soc), project ({chip_name, qub_name, "
            "res_name, result_dir, database_path} or null when none applied), "
            "context (active context label), soc ({connected, is_mock}), "
            "tabs ([{tab_id, adapter, is_running}]), running_tab, and active_tab. "
            "active_tab is where the USER is currently focused — a collaboration "
            "cue, NOT your operation target (you always act on an explicit tab_id). "
            "The same overview is folded into gui_bridge_connect's reply, "
            "so right after attaching you already have it."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_editor_open": {
        "handler": tool_gui_editor_open,
        "description": (
            "Open a stateful editing session over an EXISTING ModuleLibrary "
            "module/waveform (by 'from_name'). To create a new blank/shaped entry, "
            "use gui_context_ml_create_from_role then gui_editor_open(from_name=...) "
            "to edit it. item_kind is 'module' or 'waveform'. Returns "
            "{editor_id, cfg} — cfg is the nested current-value tree (same shape "
            "as gui_editor_get_cfg / gui_tab_get_cfg). Address later edits with the "
            "returned editor_id via gui_editor_set; persist with gui_editor_save."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_kind": {
                    "type": "string",
                    "description": "'module' or 'waveform'",
                },
                "from_name": {
                    "type": "string",
                    "description": "Existing ml entry name to load for editing",
                },
            },
            "required": ["item_kind", "from_name"],
        },
    },
    "gui_editor_get_cfg": {
        "handler": tool_gui_editor_get_cfg,
        "description": (
            "Read an editing session's settable cfg as a NESTED tree of current "
            "values, addressed by 'editor_id' (from gui_editor_open). Returns "
            "{cfg} — the same tree shape and '$'-prefixed leaf encoding as "
            "gui_tab_get_cfg (SCALAR / ENUM '$value'+'$choices' / SWEEP edges / "
            "REF '$ref'). Edit a leaf with gui_editor_set using its dotted path. "
            "'prefix' (optional, dotted) returns just the sub-tree rooted there; "
            "a prefix matching nothing returns {}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_editor_open)",
                },
                "prefix": {
                    "type": "string",
                    "description": (
                        "Return only the sub-tree rooted at this dotted path "
                        "(e.g. 'modules.readout'); omit for the whole draft. "
                        "No match → {}"
                    ),
                },
            },
            "required": ["editor_id"],
        },
    },
    "gui_editor_set": {
        "handler": tool_gui_editor_set,
        "description": (
            "Batch-set fields on ONE cfg-editor session in order (fail-fast, "
            "non-atomic), addressed by 'editor_id' (from gui_editor_open). For tab "
            "cfg editing use gui_tab_set_cfg instead. 'edits' is an ORDERED list of "
            "{path, value}: 'path' is dotted (see gui_editor_get_cfg); 'value' is a "
            "JSON scalar or an md-ref {__kind:eval, expr} (the eval form is accepted "
            "only on a scalar leaf, never a sweep edge). Apply ref-switch edits "
            "before dependent inner-path edits (a ref switch removes child paths). "
            "Stops at the first failure and edits applied before it are NOT rolled "
            "back; the error names the failing path and how many already applied. On "
            "success returns {applied, valid} — the count applied and whether the "
            "resulting draft is valid. It does NOT echo cfg content (reading it "
            "would force a lowering pass that eagerly evaluates EvalValue); read the "
            "cfg with gui_editor_get_cfg if needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": "Editor session id (from gui_editor_open)",
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
                                # Untyped schema = "any JSON value", matching the
                                # generator's JsonType.JSON rendering. A 'type' union
                                # listing "string" would let the client coerce a
                                # number (0.2) to "0.2" and fail the float check.
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
    "gui_tab_set_cfg": {
        "handler": lambda arguments: send_gui_rpc(
            "tab.set_cfg",
            {
                "tab_id": str(arguments["tab_id"]),
                "edits": _coerce_pairs(
                    arguments.get("edits"), field="edits", keys=("path", "value")
                ),
            },
        ),
        "description": (
            "Batch-set cfg fields on a tab in order (non-atomic batch). Apply "
            "ref-switch edits BEFORE dependent inner-path edits — a ref switch "
            "removes child paths and a stale inner-path edit after it will fail. "
            "On the first failing edit the call RAISES (same contract as "
            "gui_context_md_write): edits applied before it stay applied and are "
            "NOT rolled back. On success returns {valid, removed, added} "
            "aggregated across the batch. A running tab is rejected (cancel the "
            "run first). Read the current tree with gui_tab_get_cfg."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to edit",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered list of edits; each {path, value}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Dotted field path (from gui_tab_get_cfg)",
                            },
                            "value": {
                                # Untyped: numbers must not be coerced to strings.
                                "description": "JSON scalar or {__kind:eval, expr}",
                            },
                        },
                        "required": ["path", "value"],
                    },
                },
            },
            "required": ["tab_id", "edits"],
        },
    },
    "gui_context_list": {
        "handler": tool_gui_context_list,
        "description": (
            "List the context labels plus which one is active — the orientation "
            "read for contexts. Returns {active: str|null, has_active_context: bool, "
            "labels: [str]}. 'labels' are plain strings: per-label unit/value are "
            "NOT available (unit/value are transient creation metadata, never "
            "persisted). Switch with gui_context_switch; create with "
            "gui_context_create."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_context_md_read": {
        "handler": tool_gui_context_md_read,
        "description": (
            "Read MetaDict attributes — omit 'keys' to read the WHOLE tree, or pass "
            "'keys' to read only that subset. Returns {values: {key: value}} keyed "
            "by attribute name. An unknown key in an explicit 'keys' list fails fast "
            "(invalid_params) — keys are never silently skipped. Reads are "
            'side-effect-free. Complex scalars arrive as {"__complex__": [re, im]} '
            "(symmetric with gui_context_md_write)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "description": "MetaDict keys to read; omit to read every attribute",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "gui_context_md_write": {
        "handler": tool_gui_context_md_write,
        "description": (
            "Batch-write MetaDict attributes in order. NOT atomic: stops at the "
            "first failure (fail-fast), attrs set before it are NOT rolled back, and "
            "the error message carries applied_count + failed_index. Returns "
            "{applied} on success. Complex scalars round-trip via "
            '{"__complex__": [re, im]} on the value.'
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
    "gui_context_md_delete": {
        "handler": tool_gui_context_md_delete,
        "description": (
            "Batch-delete MetaDict attributes by key. Idempotent per key: deleting a "
            "missing key is a no-op (not an error). NOT atomic across keys: on a "
            "(non-idempotent) failure the error carries applied_count + failed_index "
            "and keys before it stay deleted. Returns {deleted: [key]} on success."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "description": "MetaDict keys to delete",
                    "items": {"type": "string"},
                },
            },
            "required": ["keys"],
        },
    },
    "gui_context_ml_inspect": {
        "handler": tool_gui_context_ml_inspect,
        "description": (
            "Read one ModuleLibrary entry's full cfg WITHOUT opening a tab or a "
            "lasting editing session. Returns {cfg} — the nested current-value tree "
            "(same shape as gui_tab_get_cfg). A pure read: it opens a headless draft "
            "on the entry and discards it, bumping no agent-visible resource "
            "version. Use gui_context_ml_list first to find names + kinds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_kind": {
                    "type": "string",
                    "description": "'module' or 'waveform'",
                },
                "name": {"type": "string", "description": "ml entry name"},
            },
            "required": ["item_kind", "name"],
        },
    },
    "gui_tab_run_start": {
        "handler": tool_gui_tab_run_start,
        "description": (
            "Start a run on tab_id (shared short-wait START contract — see server "
            "instructions). A fast run settles -> {status:'finished', handle, "
            "tab:{...}, figure:<path>} — the tab snapshot (has_run_result set) AND "
            "the run plot rendered to a temp PNG (the run's OWN visual result). A "
            "slow run degrades to {status:'pending', handle} (no figure yet; "
            "poll/wait the handle with gui_op_poll / gui_op_wait, then read the plot "
            "with gui_tab_get_current_figure). The reply always carries 'handle'."
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
    "gui_op_poll": {
        "handler": tool_gui_op_poll,
        "description": (
            "Non-blocking status of ANY in-flight operation, by 'handle' (the "
            "opaque token a START tool — gui_tab_run_start / gui_tab_analyze_start "
            "/ gui_tab_post_analyze_start / gui_device_* — returned in its reply). "
            "NEVER raises; returns {status, ...} with status in "
            "finished|running|cancelled|failed|no_operation. While 'running' the "
            "reply folds the live progress bars (active, bars[token/format/percent]) "
            "— no separate progress tool. Reports only the status: the op's product "
            "(figure/summary/snapshot) comes from the START finished reply or the "
            "matching typed getter (e.g. gui_tab_get_current_figure, "
            "gui_tab_get_analyze_result, gui_device_snapshot)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "integer",
                    "description": "Operation handle from a START tool's reply",
                }
            },
            "required": ["handle"],
        },
    },
    "gui_op_wait": {
        "handler": tool_gui_op_wait,
        "description": (
            "Block until ANY in-flight operation settles, by 'handle' (the opaque "
            "token a START tool returned). Blocks up to 'timeout' seconds and holds "
            "your turn; for a long op prefer gui_op_poll (non-blocking) or run this "
            "from a background agent. Returns {status, waited_seconds, ...} with "
            "status in finished|cancelled|user_feedback|timed_out|no_operation: "
            "'cancelled' (read optional 'feedback' for the Stop reason — NOT a "
            "raise), 'user_feedback' (op STILL running — treat 'feedback' as a "
            "high-priority instruction, re-plan, then re-wait or cancel via the "
            "op-specific cancel tool), 'timed_out' (still running — re-wait or "
            "gui_op_poll). RAISES only on a genuine failure. Reports only the "
            "status: read the op's product via the matching typed getter."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "integer",
                    "description": "Operation handle from a START tool's reply",
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 120)",
                },
            },
            "required": ["handle"],
        },
    },
    "gui_tab_open": {
        "handler": tool_gui_tab_open,
        "description": (
            "Phase ① of the recommended flow (open -> run -> analyze_review -> "
            "commit) — open. = tab.new + tab.snapshot + tab.get_cfg + adapter.guide. "
            "Create a tab for 'adapter_name' (see gui_adapter_list) and fold its "
            "editing context (tab.snapshot for editor_id, tab.get_cfg for the "
            "settable cfg tree) + the adapter guide into ONE reply. The guide is "
            "INCLUDED BY DEFAULT — this ensures any fresh context, sub-agent, or "
            "context-reset session that opens a tab always receives the orientation "
            "text without having to remember a flag. Returns "
            "{tab_id, adapter, editor_id, tree, guide}. "
            "Pass skip_guide=true only if you already have the guide in your context "
            "(e.g. you opened a tab for this same adapter earlier in this session and "
            "your context still contains it) — the reply will carry "
            "'guide_omitted: True' to confirm the omission. When in doubt, do NOT "
            "pass skip_guide=true; a duplicate guide wastes fewer tokens than a "
            "missing one (sub-agents sharing no context would be starved otherwise). "
            "'tree' is the nested current-value cfg tree (the gui_tab_set_cfg path "
            "source AND the read-only values view, in one — see gui_tab_get_cfg for "
            "the node shape with $value/$choices/$ref). Then configure + run with "
            "gui_tab_run(tab_id, edits)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Adapter to instantiate (see gui_adapter_list)",
                },
                "skip_guide": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Suppress the adapter guide fetch (reply carries "
                        "guide_omitted: true). Only pass true when you are certain "
                        "the guide is already in your context — e.g. you opened a "
                        "tab for this same adapter earlier this session. Default "
                        "false (guide always sent) so sub-agents / new contexts are "
                        "never starved."
                    ),
                },
            },
            "required": ["adapter_name"],
        },
    },
    "gui_tab_run": {
        "handler": tool_gui_tab_run,
        "description": (
            "Phase ② of the recommended flow (open -> run -> analyze_review -> "
            "commit) — run. = gui_tab_set_cfg + gui_tab_run_start. Apply 'edits' "
            "then run the already-created 'tab_id' (from gui_tab_open), then STOP "
            "before analyze. 'edits' is an OPTIONAL ORDERED list of {path, value} "
            "(dotted paths, see gui_tab_get_cfg; numbers stay numbers). The order is "
            "preserved — apply a $ref switch BEFORE the paths it unlocks. Omit/empty "
            "'edits' to run the tab's current cfg. A fast run returns "
            "{status:'finished', handle, tab, figure, analyze_params} — 'figure' "
            "comes from gui_tab_run_start's own FINISHED reply (the run plot "
            "rendered to a temp PNG); 'analyze_params' is the stage-specific fold "
            "(the analyze knobs for this tab). A slow run degrades to "
            "{status:'pending', handle, owed} — 'owed' names what is not yet "
            "available; drive the handle with gui_op_wait(handle) / "
            "gui_op_poll(handle), then read the plot with gui_tab_get_current_figure. "
            "STOPS before analyze on purpose: a successful run is NOT a successful "
            "analyze — look at the figure, then gui_tab_analyze_review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to configure + run (from gui_tab_open)",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "value": {},
                        },
                        "required": ["path", "value"],
                    },
                    "description": (
                        "Optional ORDERED list of {path, value} cfg edits applied "
                        "before the run (dotted paths, see gui_tab_get_cfg). Order "
                        "is preserved (ref-switch before its children). Numbers stay "
                        "numbers. Omit/empty to run with the tab's current cfg."
                    ),
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Short-wait bound for the run (default 1.0).",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_analyze_review": {
        "handler": tool_gui_tab_analyze_review,
        "description": (
            "Phase ③ of the recommended flow (open -> run -> analyze_review -> "
            "commit) — analyze_review. = gui_tab_analyze_start + "
            "gui_tab_writeback_list. Analyze 'tab_id' and fold the writeback "
            "review into ONE reply. A finished FIT returns {status:'finished', "
            "handle, summary, figure, writeback_preview} — 'summary' is the fit "
            "result (same shape as gui_tab_get_analyze_result), 'figure' comes from "
            "gui_tab_analyze_start's own FINISHED reply (the fit plot rendered to a "
            "temp PNG), and 'writeback_preview' is the stage-specific fold "
            "({has_draft, items} — the proposed writeback values/targets) — so you "
            "review the fit + the proposed writeback in one call before "
            "gui_tab_commit. 'updates' optionally overrides the analyze params; "
            "'wait_seconds' (default 1.0) bounds the short wait. An INTERACTIVE "
            "analysis (e.g. flux_dep) degrades to {status:'pending', handle, owed} "
            "(no folds; 'owed' names the pending reads) — prompt the user, then "
            "drive the handle with gui_op_wait(handle) / gui_op_poll(handle), then "
            "read gui_tab_get_analyze_result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to analyze (from gui_tab_run)",
                },
                "updates": {
                    "type": "object",
                    "description": "Optional overrides for the analyze params.",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Short-wait bound for the analyze (default 1.0).",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_commit": {
        "handler": tool_gui_tab_commit,
        "description": (
            "Phase ④ of the recommended flow (open -> run -> analyze_review -> "
            "commit) — commit. = gui_tab_writeback_apply + (optionally) gui_tab_save. "
            "Apply the tab's writeback draft (edit it first via "
            "gui_tab_writeback_set_item / gui_editor_*), optionally saving afterwards. "
            "Applies the items currently selected; returns {status, applied_ids, "
            "written, context_version, saved, save_error?}. 'save' selects the "
            "follow-up save artifacts (same vocabulary as gui_tab_save): 'none' "
            "(default, apply-only), 'data', 'image', or 'both'.\n"
            "fail-soft across the two steps: the apply runs first and is committed; "
            "if the follow-up save then fails, the applied writeback is NOT lost — "
            "'saved' is the gui_tab_save result (or null) and 'save_error' is the "
            "save failure message. 'status' is 'committed' when both steps succeed "
            "(or save='none'), 'partial' when the apply succeeded but the save "
            "failed. This partial-status surface is ONLY for this cross-step bundle "
            "— a single wire/tool call (the apply, or gui_tab_save on its own) still "
            "Fast-Fails by raising."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab whose writeback draft to apply (from gui_tab_analyze_review)",
                },
                "save": {
                    "type": "string",
                    "enum": ["none", "data", "image", "both"],
                    "default": "none",
                    "description": (
                        "Optional follow-up save after applying (same artifacts as "
                        "gui_tab_save): 'none' (default, apply only), 'data', "
                        "'image', or 'both'."
                    ),
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_analyze_start": {
        "handler": tool_gui_tab_analyze,
        "description": (
            "Start analyze on tab_id (shared short-wait START contract — see server "
            "instructions, incl. the INTERACTIVE-analyze note). A FIT settles -> "
            "{status:'finished', handle, summary, figure} — the fit summary (same "
            "shape as gui_tab_get_analyze_result, the *_err fields included) AND the "
            "fit plot rendered to a temp PNG (analyze's OWN visual result). Review "
            "the proposed writeback with gui_tab_writeback_list (not folded here; "
            "that fold lives in gui_tab_analyze_review). An INTERACTIVE analysis (e.g. "
            "flux_dep) degrades to {status:'pending', handle, summary:None} — no "
            "figure (nothing settled yet); prompt the user to mark the plot + click "
            "Done, then poll/wait the handle with gui_op_poll / gui_op_wait, then "
            "read gui_tab_get_analyze_result + gui_tab_get_current_figure. The reply "
            "always carries 'handle'. 'updates' optionally overrides analyze params "
            "(see gui_tab_get_analyze_params for the current params)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": "Analyze param updates",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_post_analyze_start": {
        "handler": tool_gui_tab_post_analyze,
        "description": (
            "Start the second-layer (post) analysis on tab_id (shared short-wait "
            "START contract — see server instructions). Runs on top of the tab's "
            "PRIMARY analyze result (e.g. single-shot multi-backend ge "
            "discrimination) and is FIT-only (no INTERACTIVE mode), so it usually "
            "settles -> {status:'finished', handle, summary:{...}} (folded in, same "
            "shape as gui_tab_get_post_analyze_result). A slow one degrades to "
            "{status:'pending', handle}; poll/wait the handle with gui_op_poll / "
            "gui_op_wait, then read gui_tab_get_post_analyze_result. The reply always "
            "carries 'handle'. There is NO cancel for post-analysis: it is a pure "
            "CPU recompute with no stop point. Fast-fails with precondition_failed "
            "when the tab has no primary analyze result yet — run "
            "gui_tab_analyze_start first. 'updates' optionally overrides post params "
            "(see gui_tab_get_post_analyze_params). The post figure shares the tab's "
            "plot container; see it with gui_tab_get_current_figure and persist it "
            "with gui_tab_save."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "updates": {
                    "type": "object",
                    "description": "Post-analysis param updates",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_tab_save": {
        "handler": tool_gui_tab_save,
        "description": (
            "Save a tab's result data and/or figure; return the resolved "
            "destinations. Two orthogonal selectors:\n"
            "  - artifact='data'|'image'|'both' (default 'both'): which artifacts.\n"
            "  - figure='primary'|'post' (default 'primary'): which figure the "
            "'image' artifact targets (the primary run/analyze plot, or the "
            "post-analysis plot).\n"
            "Collision policy (NOT uniform — it mirrors the savers): DATA is "
            "written ASYNC with a uniqueness suffix (<stem>.hdf5, '_N' on "
            "collision) so it NEVER overwrites; IMAGE is written SYNC and "
            "OVERWRITES an existing destination.\n"
            "Returns {data_path, image_path, data_async, image_error} (all keys "
            "always present, null when N/A): data_path is the resolved data path; "
            "image_path is the written image path (null if the image save failed); "
            "data_async is true when a data save was started (it finishes off-turn); "
            "image_error carries the image-save error message when it FAILED.\n"
            "Error boundary: with artifact='both' a data save ran first, so EVERY "
            "image failure (wire GuiRpcError AND stale-version RuntimeError) folds "
            "into image_error and is NOT raised — the resolved data_path is never "
            "lost. With artifact='image' (nothing to protect) an image failure "
            "RAISES (Fast-Fail). A precondition failure on the DATA save always "
            "raises (no data_error field). Optional data_path / image_path override "
            "the tab's configured destinations; comment annotates the data file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string", "description": "Tab id"},
                "artifact": {
                    "type": "string",
                    "enum": ["data", "image", "both"],
                    "default": "both",
                    "description": "Which artifacts to save (default 'both')",
                },
                "figure": {
                    "type": "string",
                    "enum": ["primary", "post"],
                    "default": "primary",
                    "description": (
                        "Which figure the 'image' artifact targets (default 'primary')"
                    ),
                },
                "data_path": {
                    "type": "string",
                    "description": "Override the data destination path",
                },
                "image_path": {
                    "type": "string",
                    "description": "Override the image destination path",
                },
                "comment": {
                    "type": "string",
                    "description": "Optional comment annotating the data file",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_soc_connect": {
        "handler": tool_gui_soc_connect,
        "description": (
            "Connect the SoC SYNCHRONOUSLY (NOT a degrading async handle — there is "
            "no gui_soc_connect_wait / _poll). One blocking call returns "
            "{soc:{description, is_mock}} once the board is connected (call "
            "gui_soc_info for the structured cfg). kind='mock' (offline) or "
            "kind='remote' with ip+port. A remote connect to an unreachable board "
            "fails fast (~1s). The SoC has no teardown (Pyro4-backed): there is no "
            "disconnect / reconnect / health-check tool — those are deferred (E3)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "'mock' or 'remote'"},
                "ip": {"type": "string", "description": "Board IP (remote)"},
                "port": {"type": "integer", "description": "Board port (remote)"},
            },
            "required": ["kind"],
        },
    },
    "gui_screenshot": {
        "handler": tool_gui_screenshot,
        "description": (
            "Capture the GUI to a PNG FILE and return its path. 'target' selects "
            "what to grab:\n"
            "  - target='window': the WHOLE main window — its client area AND the "
            "non-dialog floating widgets (the feedback widget, the left-edge "
            "handle) that a per-dialog grab cannot see.\n"
            "  - target=<dialog name> (one of: setup, device, predictor, inspect, "
            "startup): that dialog; fails PRECONDITION_FAILED if it is not "
            "currently open.\n"
            "The PNG is ALWAYS written to disk and the reply is {saved_to, bytes} — "
            "Read the saved_to path to view it (never inline base64, so it cannot "
            "blow the token budget). Omit out_path to write a per-target file under "
            "the temp dir (overwritten each call); pass out_path (absolute) to "
            "choose the location.\n"
            "Timing note: a floating widget repositions via QTimer.singleShot, so a "
            "screenshot taken in the same turn as a UI change may catch a "
            "pre-reposition frame — do a wire round-trip (any read tool) first, or "
            "re-grab, if a widget looks mislaid."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": [
                        "window",
                        "setup",
                        "device",
                        "predictor",
                        "inspect",
                        "startup",
                    ],
                    "description": (
                        "'window' for the whole main window, or a dialog name "
                        "(setup, device, predictor, inspect, startup)"
                    ),
                },
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional absolute path to write the PNG; omit to use a "
                        "per-target file under the temp dir"
                    ),
                },
            },
            "required": ["target"],
        },
    },
    "gui_debug_resource_versions": {
        "handler": tool_gui_debug_resource_versions,
        "description": (
            "DEV TOOL (debugging the GUI/MCP itself, not the measurement): dump the "
            "per-resource optimistic-concurrency version table as a flat "
            "{resource_key: int} map, for debugging stale-guard rejections. "
            "These version numbers are normally hidden from the operator "
            "(mcp<->RPC bookkeeping); read them only when debugging why a "
            "guarded op rejected (or failed to reject) on a stale dependency. "
            "Returns the GUI's live (authoritative) table — not the mcp-side "
            "cache. SIDE EFFECT: like any RPC round-trip this resyncs the mcp "
            "last-seen baseline to the current table; it does NOT bump any "
            "resource version (the numbers only move on a real edit/run/writeback). "
            "NOTE: wire/gui/mcp *code* version握手 (WIRE_VERSION / GUI_VERSION / "
            "MCP_VERSION) lives in the gui_launch / gui_bridge_connect 'note' field "
            "— it is not in this table."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_debug_operations": {
        "handler": tool_gui_debug_operations,
        "description": (
            "DEV TOOL (debugging the GUI/MCP itself, not the measurement): dump the "
            "mcp-side per-key operation-handle cache, normally hidden from the "
            "operator. Returns {handles: {key: {operation_id}}} keyed by semantic "
            "key (e.g. 'tab:<id>', 'analyze:<id>', 'post_analyze:<id>', "
            "'device:<name>') -> the latest operation_id captured for that resource "
            "— the only view of run/analyze/post_analyze handles. Use when debugging "
            "no_operation / a stuck wait (e.g. which handle did the last run on this "
            "tab get). The live DEVICE enumeration is NOT duplicated here — "
            "gui_device_list_operations is the dedicated wire enumerator. "
            "Lifecycle: a key is written when the matching start RPC fires "
            "(tab.run_start / tab.analyze / tab.post_analyze / "
            "device.connect/disconnect/setup) with 'latest wins', and is NEVER "
            "removed — entries persist for the entire MCP server process lifetime. "
            "A stale key for a completed operation is normal, so the cache cannot "
            "certify liveness (there is no 'live' field)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_device_connect": {
        "handler": tool_gui_device_connect,
        "description": (
            "Connect a hardware device. Two modes by which params you pass:\n"
            "  - FIRST connect / re-register: pass type_name (driver class, e.g. "
            "'YOKOGS200', 'SGS100A') AND address (VISA/GPIB/IP). remember defaults "
            "to true (device persists across sessions); set remember=false for a "
            "memory-only device.\n"
            "  - RECONNECT a remembered device: pass ONLY name (omit type_name and "
            "address) — the stored type/address are reused (this also covers a "
            "memory-only device that was disconnected with remember=true).\n"
            "Waits up to wait_seconds (default 1.0): if it lands in time returns "
            "{status:'finished', handle, snapshot:{...}} (snapshot includes the "
            "device's live info params); otherwise {status:'pending', handle} — "
            "poll/wait the handle with gui_op_poll / gui_op_wait. The reply always "
            "carries 'handle'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Friendly name for this device",
                },
                "type_name": {
                    "type": "string",
                    "description": (
                        "Driver class name, e.g. 'YOKOGS200'. Omit (with address) "
                        "to reconnect a remembered device by name."
                    ),
                },
                "address": {
                    "type": "string",
                    "description": (
                        "VISA or IP address. Omit (with type_name) to reconnect a "
                        "remembered device by name."
                    ),
                },
                "remember": {
                    "type": "boolean",
                    "description": "Persist device across sessions (default true)",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_device_disconnect": {
        "handler": tool_gui_device_disconnect,
        "description": (
            "Disconnect a device. Waits up to wait_seconds (default 1.0): if it "
            "lands in time returns {status:'finished', handle, snapshot:{...}}; "
            "otherwise {status:'pending', handle} — poll/wait the handle with "
            "gui_op_poll / gui_op_wait. The reply always carries 'handle'. Two "
            "terminal states by 'remember': remember=true (default) keeps the "
            "device in persistent storage as memory-only (reconnect later via "
            "gui_device_connect with name only); remember=false also removes it "
            "from persistent storage."
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
    "gui_device_apply": {
        "handler": tool_gui_device_setup,
        "description": (
            "Apply device-field updates: patch the device's info fields via "
            "'updates' (e.g. {'value': 0.5} to ramp a source's output value — this "
            "is the way to set an output value, ramped/cancellable, no separate "
            "set_value). Waits up to wait_seconds (default 1.0): if it lands in "
            "time returns {status:'finished', handle, snapshot:{...}}; otherwise "
            "{status:'pending', handle} — poll/wait the handle with gui_op_poll / "
            "gui_op_wait (a 'running' poll reply carries the live progress bars, "
            "e.g. a ramp). The reply always carries 'handle'. The device must "
            "already be connected. Read the settable fields with gui_device_fields."
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
    "gui_tab_get_current_figure": {
        "handler": tool_gui_tab_get_current_figure,
        "description": (
            "A run/analyze START reply that FINISHED in the short wait ALREADY "
            "folds the figure (gui_tab_run_start / gui_tab_analyze_start, incl. 2D "
            "scans) — read its 'figure' field then. Call THIS when the figure was "
            "NOT folded: after a pending->finished op (the generic gui_op_wait / "
            "gui_op_poll report only status, NOT the figure), a re-render, a "
            "mid-flight (non-finished) plot you must inspect, or to choose out_path. "
            "Renders the tab's CURRENT figure to a PNG FILE — whatever is on top "
            "of the tab's plot stack: the run's 2D map while/after a run, the "
            "analysis fit once analyzed, or a post-analysis figure. The PNG is "
            "rendered at a fixed small geometry (~640x480), independent of the GUI "
            "window size. The reply is {saved_to, bytes} — Read the saved_to path "
            "to view the plot (never inline base64). Omit out_path to write a "
            "per-tab file under the temp dir (overwritten each call); pass out_path "
            "to choose the location. Fails with PRECONDITION_FAILED if the tab has "
            "no figure yet (run has not completed)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional absolute path to write the PNG; omit to use a "
                        "per-tab file under the temp dir"
                    ),
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_prompt_user": {
        "handler": tool_gui_prompt_user,
        "description": (
            "A BLOCKING request-reply prompt: ask the user a question and BLOCK "
            "your entire turn until they respond (or the timeout expires). Opens a "
            "non-modal prompt dialog in the GUI showing 'message'. There is NO poll "
            "or cancel — it is one blocking request-reply, not an async handle. "
            "Returns {reason, reply?} — switch on 'reason':\n"
            "  - reason='reply': user answered; 'reply' is present (a string, "
            "possibly empty) — read and act on it.\n"
            "  - reason='dismiss': user explicitly closed the prompt; NO 'reply' "
            "key — do NOT ask again immediately; respect the user's choice.\n"
            "  - reason='timeout': no one responded within 'timeout' seconds; NO "
            "'reply' key — the user is probably not watching; do NOT keep blocking, "
            "continue or poll.\n"
            "timeout (default 600s) is the dialog auto-close timer; a value <= 0 is "
            "clamped to 1.0s so the dialog always arms its auto-close. The RPC call "
            "blocks the whole turn for up to timeout+15s; use only when you genuinely "
            "need a human decision before continuing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Question or message to display to the user",
                },
                "timeout": {
                    "type": "number",
                    "default": 600,
                    "description": (
                        "Seconds before the dialog auto-closes with reason='timeout' "
                        "(default 600; a value <= 0 is clamped to 1.0). Set shorter "
                        "for time-sensitive prompts."
                    ),
                },
            },
            "required": ["message"],
        },
    },
}


# Tool names served by the hand-written overrides rather than the generator:
# lifecycle tools (no RPC method) + the convenience/coercion/file-write tools.
_OVERRIDE_NAMES = frozenset(
    {
        "gui_bridge_connect",
        "gui_bridge_detach",
        "gui_launch",
        "gui_stop",
        "gui_device_connect",
        "gui_device_disconnect",
        "gui_device_apply",
        "gui_screenshot",
        "gui_debug_resource_versions",
        "gui_debug_operations",
        "gui_tab_get_current_figure",
        "gui_overview",
        "gui_editor_open",
        "gui_editor_get_cfg",
        "gui_editor_set",
        "gui_tab_set_cfg",
        "gui_tab_save",
        "gui_context_list",
        "gui_context_md_read",
        "gui_context_md_write",
        "gui_context_md_delete",
        "gui_context_ml_inspect",
        "gui_op_poll",
        "gui_op_wait",
        "gui_tab_run_start",
        "gui_tab_open",
        "gui_tab_run",
        "gui_tab_analyze_review",
        "gui_tab_commit",
        "gui_tab_analyze_start",
        "gui_tab_post_analyze_start",
        "gui_soc_connect",
        "gui_prompt_user",
    }
)


# Generated tools (schema from the ParamSpec SSOT, forwarding through the guarded
# send_gui_rpc) overlaid with the hand-written override subset (lifecycle /
# fan-out / file-write / coercion). assemble_tools fails fast on a name collision.
TOOLS: dict[str, dict[str, Any]] = assemble_tools(
    generate_tools(_CONFIG, METHOD_SPECS, _NON_GENERATED_METHODS, send_gui_rpc),
    _OVERRIDE_TOOLS,
    _OVERRIDE_NAMES,
)

# Wrap every top-level handler for call logging (Phase 166).  Transparent
# side-effect only: same args in, same result out (or re-raised exception);
# only writes one JSONL line per invocation.  Applied after assemble_tools so
# generated + override + bundle tools are all covered in a single pass.
for _tool_name, _tool_entry in TOOLS.items():
    _tool_entry["handler"] = wrap_handler(_tool_name, _tool_entry["handler"])


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def _cleanup_on_exit() -> None:
    """Stop the GUI when the MCP host disconnects (stdin EOF) — only if WE launched it.

    An attach-only server (lazy auto-connect, e.g. the external-terminal agent's
    loopback MCP server) must NOT shut down a GUI it merely connected to: that GUI
    belongs to whoever launched it. Without this guard, closing the agent window
    would terminate the user's GUI via the shared pid-file fallback in stop().
    """
    if not _BRIDGE.launched_gui:
        return
    try:
        # Best-effort graceful close on host disconnect; force-kill on timeout so
        # we don't leak a GUI process when the bridge goes away.
        tool_gui_stop({"timeout_kill": True})
    except Exception:
        logger.debug("gui_stop on exit failed", exc_info=True)


def _setup_logging() -> None:
    """Attach the MCP server process's per-session file logging.

    stdout is the JSON-RPC transport, so logging must never touch it — the shared
    helper only adds a stderr (WARNING) handler plus a DEBUG file handler. Attach
    at ``zcu_tools.mcp`` so this module + tool error logs reach the file.
    parents[4]: server.py -> measure -> mcp -> zcu_tools -> lib -> repo root.
    """
    from zcu_tools.gui.logging_setup import setup_gui_logging

    setup_gui_logging(
        app_name="measure",
        log_root=Path(__file__).resolve().parents[4],
        group="mcp",
        extra_namespaces=("zcu_tools.mcp",),
    )


def _format_diagnostic(msg: dict[str, Any]) -> str:
    """One diagnostic as a single compact line ``severity: title — message``.

    The wire diagnostic is ``{event:'diagnostic', payload:{severity, title,
    message}}``. We render the payload fields the agent reads rather than the full
    JSON envelope (indent=2 was pure token noise). An unknown shape (no payload)
    falls back to its compact JSON so nothing is silently dropped.
    """
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return json.dumps(msg, separators=(",", ":"))
    severity = payload.get("severity", "info")
    title = payload.get("title") or ""
    message = payload.get("message") or ""
    head = f"{severity}: {title}" if title else str(severity)
    return f"{head} — {message}" if message else head


def _piggyback_blocks() -> list[dict[str, Any]]:
    """Diagnostics buffered since the last tool call, as extra content blocks.

    Piggyback (ADR-0013): drain GUI diagnostics onto every successful tool reply
    so the agent gets the GUI's error/info feedback ("Data saved to …", a
    run-failure reason) without a dedicated poll. Only diagnostics ride here —
    resource-change events are not exposed to the agent. Rendered as a compact
    one-line-per-diagnostic list (no indented JSON) to keep the piggyback
    token-light.
    """
    pending = _drain_pending()
    diagnostics = pending["diagnostics"]
    if not diagnostics:
        return []
    lines = "\n".join(_format_diagnostic(m) for m in diagnostics)
    return [{"type": "text", "text": "notifications since last call:\n" + lines}]


def main() -> None:
    run_stdio_loop(
        _CONFIG,
        TOOLS,
        on_cleanup=_cleanup_on_exit,
        on_each_reply=_piggyback_blocks,
        on_start=_setup_logging,
        on_error=logger.exception,
        server_version="1.1.0",
    )


if __name__ == "__main__":
    main()
