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
#      separate gui_tab_get_current_figure call. gui_analyze becomes a hand-written
#      override (was a generated forwarder) to attach it.
# v14: gui_analyze awaits the analyze operation (WIRE 15, Phase 120c). analyze is
#      an async worker; gui_analyze starts it (analyze.start now returns an
#      operation_id, captured under analyze:<tab_id>) then awaits via
#      _await_operation_by_key, so it stays synchronous to the agent AND the
#      figure is rendered (has_figure true) by the time figure_path is attached.
# v15: non-blocking per-domain poll (Phase 120c-1) — gui_run_poll / gui_device_poll
#      / gui_connect_poll map a zero-timeout operation.await onto finished /
#      running / failed / no_operation, keyed on the semantic name (tab_id /
#      device name / soc), no operation_id exposed (ADR-0002 kept). Lets an agent
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
# MCP 24: wait-tool guidance only — gui_run_wait / gui_connect_wait /
#      gui_device_wait_operation descriptions + _SERVER_INSTRUCTIONS now spell out
#      that a wait BLOCKS the whole turn and that a long op should either poll or
#      be run from a background agent (the only way to free the main loop and be
#      re-invoked on completion — there is no server push). Text only, no behavior
#      or wire change.
# MCP 25: gui_tab_figure_screenshot renamed to gui_tab_get_current_figure
#      (mirrors WIRE 21 tab.figure_screenshot -> tab.get_current_figure) — same
#      tool, clearer name: "the figure currently shown" (run 2D map or analysis
#      fit). gui_save_data / gui_save_image / gui_save_result now return the
#      resolved written path(s) ({data_path[, image_path]}) instead of {ok}/{}
#      (WIRE 21).
# MCP 26: gui_analyze degrades like a run instead of blocking — short-wait then
#      {status:'finished'|'pending'}; added gui_analyze_wait / gui_analyze_poll
#      (mirror gui_run_wait/poll). A FIT usually settles in the wait; an
#      INTERACTIVE analysis (AnalysisMode.INTERACTIVE — the user picks on the plot
#      and clicks Done) never settles, so it degrades to pending and the agent
#      prompts + polls. No wire change (analyze.start + operation.await/poll
#      already exist; this is mcp-side degrade + two new poll/wait tools).
# MCP 27: post-analysis tools (WIRE 22). gui_post_analyze degrades like gui_analyze
#      (short-wait then {status:'finished'|'pending'}; FIT-only, no INTERACTIVE
#      mode) + gui_post_analyze_wait / gui_post_analyze_poll; gui_tab_get_post_
#      analyze_params / gui_tab_get_post_analyze_result auto-generate from the new
#      method_specs. Unlike gui_analyze, gui_post_analyze folds NO figure_path (the
#      post figure lives in the tab's separate post container, which the render
#      view does not screenshot).
# MCP 28: complex writeback scalars (WIRE 23). gui_writeback_preview /
#      gui_writeback_set descriptions now document the {"__complex__": [re, im]}
#      encoding for a complex metadict proposed_value (the GE adapter proposes
#      g_center / e_center as complex). Tool descriptions auto-generate from the
#      updated method_specs; the mcp forwards the tagged value verbatim.
# MCP 29: figure/screenshot consolidation (WIRE 24). Removed gui_view_screenshot
#      (whole-window grab) and the figure_path folding into run/analyze replies
#      (_figure_path_if_any / _with_figure gone) — looking at a plot is now ONLY
#      gui_tab_get_current_figure, which renders a small fixed-size PNG (token-light)
#      and also covers the post-analysis figure. Added gui_save_post_image
#      (auto-generated from the new save.post_image method_spec; mirrors
#      gui_save_image). gui_save_image keeps full save quality.
# MCP 30: device active-op enumeration (WIRE 25). gui_device_active_operations
#      returns the full list of in-flight device ops (sorted by device name); each
#      entry carries 'kind' + 'device_name'. Concurrent setups are observable in
#      one call.
# MCP 31: convenience-layer batch (no wire change). gui_analyze / gui_post_analyze
#      fold the fit summary into a finished short-wait reply (one fewer round-trip;
#      the getters stay for re-fetch + the wait/poll path). New
#      gui_context_get_md_attrs (batch read, fan-out over context.get_md_attr) —
#      the symmetric counterpart of gui_context_set_md_attrs. gui_editor_set_field
#      / _set_fields accept 'tab_id' and resolve the editor_id from the tab
#      snapshot (editor_id still works for non-tab editors). The 'running' poll
#      reply slims each progress bar to {token, format, percent}. gui_editor_commit
#      renamed to gui_editor_save_as_module (wire key stays editor.commit). Removed
#      gui_device_active_setups (the device.active_setups wire method was dropped;
#      gui_device_active_operations covers it).
# MCP 32: gui_tab_get_current_figure always returns a written file path
#      ({saved_to, bytes}), never inline base64 — out_path when given, else a
#      per-tab temp file (gettempdir()/measure_fig_<tab_id>.png). Removes the
#      base64 reply that overran the tool-output token limit. Wire unchanged (the
#      wire tab.get_current_figure still supports png_b64 for raw consumers).
# MCP 33: convenience-layer slimming (no wire change). gui_dialog_screenshot now
#      always writes a file ({saved_to, bytes}) like gui_tab_get_current_figure —
#      out_path when given, else gettempdir()/measure_dialog_<dialog_name>.png; the
#      base64 reply branch is gone (the wire dialog.screenshot still returns base64
#      for raw consumers). Piggyback diagnostics render as a compact one-line-per-
#      diagnostic list ('severity: title — message') instead of indented JSON. The
#      shared async contract (short-wait degrade / blocking-vs-background / timed_out
#      / user-feedback wakeup / INTERACTIVE analyze) is consolidated into
#      _SERVER_INSTRUCTIONS once; the 15 async tool descriptions (run / connect /
#      analyze / post_analyze _start/_wait/_poll + device wait/poll) keep only their
#      per-op difference. New gui_overview: a one-shot situational picture (state /
#      context / soc / tabs / running_tab / active_tab) fanned out over existing read
#      RPCs — also folded into gui_connect's reply ({note, overview}). gui_connect_start
#      no longer folds the view snapshot (kept the soc fold).
# v34: gui_project_info generated tool rides the WIRE 27 bump (read the project
#      identity chip/qub/res + result_dir/database_path). gui_overview now folds a
#      ``project`` field ({chip, qub, res} via project.info, guarded by
#      has_project, else null). _SERVER_INSTRUCTIONS gains an orient paragraph
#      (call gui_overview first, do not assume state).
# MCP 35: (a) gui_dialog_screenshot override tracks the WIRE 28 param rename
#      (dialog_name -> name): the tool's inputSchema, the forwarded wire param, and
#      the temp-file name all use 'name', matching gui_dialog_open / gui_dialog_close.
#      (b) gui_overview.project now uses long keys {chip_name, qub_name, res_name},
#      matching the project.info wire shape and all other tool-GUIs (fluxdep /
#      dispersive / autofluxdep); the short-key remap ({chip,qub,res}) is removed.
#      (c) gui_connect reply is {note, overview} — no wire change (already correct).
# MCP 36: _ensure_connected fast-fail on cold start (no WIRE change). A port probe
#      (_port_is_open, 0.5s) is run before the full TCP connect so that calling any
#      gui_* tool when no GUI is listening returns the actionable
#      "no running measure-gui found…" error immediately instead of hanging ~30s
#      until the socket timeout fires.
# MCP 37: gui_notify_user — agent-initiated user prompt (WIRE 30, Stage 4b).
#      Serially composes notify.open (main thread: mint token + open non-modal
#      dialog) and notify.await (off-main: block until Reply/Dismiss/QTimer).
#      Returns {reason:'reply'|'dismiss'|'timeout', reply?}. BLOCKS the turn;
#      never raises on timeout or dismiss. notify.open / notify.await are
#      excluded from auto-generation via _NON_GENERATED_METHODS.
# MCP 38: cancelled _wait returns structured data instead of raising (WIRE 31).
#      gui_run_wait / gui_analyze_wait / gui_post_analyze_wait /
#      gui_device_wait_operation / gui_connect_wait now return
#      {status:'cancelled', feedback?} on a cancelled operation instead of
#      raising precondition_failed. 'feedback' is present only when the user
#      clicked "Send & Stop" (carries the Stop reason); absent on a plain cancel.
#      'failed' still raises. _SERVER_INSTRUCTIONS and tool descriptions updated
#      to reflect the new contract.
# MCP 39: predictor.set_model_params wire method (WIRE 32) yields the new
#      auto-generated gui_predictor_set_model_params MCP tool — builds and
#      installs a FluxoniumPredictor directly from typed EJ/EC/EL + flux params,
#      bypassing params.json. Observable MCP surface change after MCP reconnect.
# MCP 40: SoC-connect tool rename (no wire change). gui_connect_start /
#      gui_connect_wait / gui_connect_poll renamed to gui_soc_connect /
#      gui_soc_connect_wait / gui_soc_connect_poll — disambiguates the SoC-board
#      connect family from gui_connect (the MCP bridge attach). The underlying
#      wire RPCs (connect.start / operation.await / operation.poll keyed by "soc")
#      are unchanged; only the agent-facing MCP tool names changed.
# MCP 41: SoC connect is now synchronous (WIRE 33). The async connect.start +
#      "soc" operation-handle is replaced by a synchronous soc.connect RPC;
#      gui_soc_connect makes one blocking call (explicit ~2s timeout so the
#      board-side 1s COMMTIMEOUT fires first) and returns {status:'finished',
#      soc:{description, is_mock}} directly. Removed gui_soc_connect_wait /
#      gui_soc_connect_poll and the "soc" key from _OP_KEY_OF; soc.connect is
#      hand-written (in _NON_GENERATED_METHODS). run / analyze / device keep their
#      async-handle contract. _SERVER_INSTRUCTIONS no longer lists connect among
#      the degrading async ops.
# MCP 42: convenience-layer bug fix + two fan-outs (no wire change). (1) JsonType
#      .JSON now renders an UNTYPED MCP schema (no "type" key) instead of a union
#      that listed "string"; gui_editor_set_field's hand-written 'value' schema
#      dropped its "type" union to match. The string member let the MCP client
#      coerce a number (0.2) to "0.2", failing a downstream float-field check —
#      an untyped schema passes numbers through. Same fix covers set_field value /
#      writeback.set proposed_value / context.set_md_attr value. (2) gui_tab_new is
#      now a fan-out override: its reply folds editor_id + settable paths +
#      cfg_summary (tab.snapshot + tab.list_paths + tab.get_cfg_summary), ~4 calls
#      collapse to 1. (3) gui_run_start / gui_run_wait / gui_analyze fold the
#      tab's current figure (rendered to a temp PNG, path in 'figure') into a
#      FINISHED reply (never on a pending/interactive analyze). All three are
#      mcp-side only — wire validation still accepts any JSON, so WIRE/GUI do not
#      bump.
# MCP 43: convenience-layer "run a stage" macro + first-use guide fold (no wire
#      change). (1) gui_run_stage(adapter_name, edits?) folds create-tab +
#      configure + run into one call: the gui_tab_new fan-out, then edits via the
#      plural set_fields path (numbers stay numbers — the singular-coerce bug is
#      not on this path), then gui_run_start. It STOPS before analyze (run success
#      != analyze success); the reply is the run's finished reply (with the folded
#      'figure') plus tab_id/editor_id, or a 'pending' handle if the run is slow.
#      (2) First-use adapter-guide fold: a per-process set tracks which
#      adapter_names had their guide returned this session; the FIRST gui_tab_new /
#      gui_run_stage for an adapter folds the adapter.guide reply in as 'guide' and
#      marks it seen, later calls omit it. gui_adapter_guide stays for an explicit
#      re-read. Both are mcp-side only.
# MCP 44: four-phase convenience redesign aligned to the agent's decision points +
#      a run_poll figure fix (no wire change). Phase ① (gui_tab_new) unchanged — it
#      already folds guide+editor_id+paths+cfg_summary. Phase ② gui_run_stage now
#      takes tab_id (NOT adapter_name): it operates on an ALREADY-CREATED tab (from
#      gui_tab_new, so the agent has seen the cfg), does set_fields + gui_run_start,
#      and a FINISHED reply folds the analyze-params spec (tab.get_analyze_params)
#      as 'analyze_params' next to the figure; it no longer creates a tab or folds
#      the guide (that rides gui_tab_new). Phase ③ gui_analyze's FINISHED reply ALSO
#      folds the writeback preview (writeback.preview) as 'writeback_preview' — the
#      fit + the proposed writeback values/targets in one call (swallowed on
#      failure, omitted on INTERACTIVE pending). Phase ④ gui_writeback_apply becomes
#      a hand-written override gaining save_data:bool=false — when true it chains
#      save.data after the apply and folds the resolved 'data_path' (apply runs
#      first, committed even if the save fails); default false = apply-only,
#      behavior unchanged. Consistency fix: gui_run_poll's FINISHED branch now folds
#      the figure (_fold_finished_figure) like run_start/_wait/_stage, so the
#      pending->poll path also gets the plot.
# MCP 45: base-tools-pure + gui_run_stage1..4 restructure (no wire change). The
#      base tools now return ONLY their own operation's result (least-surprise —
#      the name describes the function); cross-tool folding moves into four
#      explicit bundle tools that the docs recommend as the primary flow:
#        - gui_tab_new reverts to the auto-generated tab.new forwarder (returns
#          just {tab_id}); the fan-out + guide fold moves to gui_run_stage1.
#        - gui_run_start / _wait / _poll drop the figure fold (keep the short-wait
#          degrade contract — that is the tool's own async contract, not a fold);
#          look at the plot via gui_tab_get_current_figure.
#        - gui_analyze drops the figure + writeback-preview folds (keeps its own
#          inline 'summary' incl. *_err + the degrade contract).
#        - gui_writeback_apply reverts to the auto-generated writeback.apply
#          forwarder (returns just {applied_ids}); the save_data chaining moves to
#          gui_run_stage4.
#        - removed gui_run_stage and the now-unused _GUIDE_RETURNED seen-set.
#      The four bundle tools reuse the existing fold helpers (_fold_finished_figure
#      / _fold_analyze_params / _fold_writeback_preview / the tab_new fan-out reads,
#      now _fold_tab_editing_context): stage1 = new+guide (always returns guide);
#      stage2 = configure+run (set_fields + run_start, folds figure+analyze_params,
#      stops before analyze); stage3 = analyze (folds summary+figure+writeback_
#      preview, pending on INTERACTIVE); stage4 = writeback+save (apply, optional
#      save_data folds data_path).
# MCP 46: figure fold re-added to the four plot-producing base ops (no wire change).
#      The FIGURE is each op's OWN visual result, so folding it into a FINISHED
#      reply is "pure" by the same logic as the inline summary. The NON-plot folds
#      (writeback_preview, save, cfg reads, guide) remain ONLY in the stageN bundles.
#        - gui_run_start: FINISHED reply includes 'figure' (same path as before
#          MCP 45). Pending remains pure (no figure).
#        - gui_run_wait: FINISHED reply includes 'figure'. Non-finished unchanged.
#        - gui_run_poll: FINISHED branch includes 'figure'. Running/cancelled unchanged.
#        - gui_analyze: FINISHED (FIT) reply includes 'figure'. INTERACTIVE 'pending'
#          (no settled plot) does NOT. writeback_preview stays in gui_run_stage3 only.
#      gui_run_stage2 and gui_run_stage3 no longer double-fold the figure — they
#      inherit it from the underlying run/analyze reply they wrap.
#      gui_tab_get_current_figure description updated: marked rarely-needed (normally
#      read the folded figure from the run/analyze finished reply instead).
MCP_VERSION = 46

# ---------------------------------------------------------------------------
# Server usage instructions (returned in the MCP `initialize` result)
# ---------------------------------------------------------------------------

_SERVER_INSTRUCTIONS = """\
Drive a live qubit-measure GUI over a TCP control socket. This is the machine
contract (tool semantics + call rules); the operating manual (how to choose
wait/poll/background, hardware-safety, when to act vs ask the user) lives in the
run-measure-gui SKILL — follow it for workflow.

First, orient: call gui_overview once to read the live picture (project /
context / soc / open tabs / what is running), and re-read it (or gui_state_check)
whenever you need the current state. Do NOT assume any state — the user may be
driving the same GUI alongside you.

Tool families:
  - Lifecycle: gui_launch / gui_connect / gui_stop / gui_app_shutdown.
  - Startup: gui_soc_connect (kind='mock'|'remote'), gui_startup_apply,
    gui_context_use / gui_context_new, gui_state_check (the four readiness flags
    has_project / has_context / has_active_context / has_soc).
  - Tabs + cfg: gui_adapter_list / gui_tab_new / gui_tab_snapshot;
    gui_tab_get_cfg / gui_tab_list_paths; gui_editor_set_field / _set_fields
    (a tab's cfg form is the editor session keyed by its tab_id — pass tab_id or
    an explicit editor_id).
  - Run + analyze: gui_run_start, gui_analyze, gui_post_analyze (each waits briefly
    then degrades to a handle). A FINISHED run/analyze reply already carries 'figure'
    — the plot rendered to a temp PNG — so a separate gui_tab_get_current_figure call
    is rarely needed (use it only for a re-render, mid-flight plot, or custom
    out_path). gui_tab_get_analyze_result / gui_tab_get_post_analyze_result re-read
    the fit summary; gui_save_data / _image / _result / _post_image persist and
    return the resolved written path.
  - Async handles: every degrading op exposes a _wait (blocking) and a _poll
    (non-blocking) tool keyed by name/tab_id.

Startup precondition: gui_state_check must report all four flags true before
running experiments. Run/save require an active file-backed context; save/analyze
require an existing run result. A precondition violation returns
precondition_failed; editing cfg while a tab is running likewise.

gui_soc_connect is SYNCHRONOUS — NOT part of the async-handle family: it blocks
until the SoC is connected and returns {status:'finished', soc:{...}} in one call
(no _wait / _poll). A remote board that is unreachable fails fast (~1s).

The other async ops (run / analyze / post_analyze / device) share ONE contract —
the per-tool descriptions only name what each waits on; the mechanics live here.
Completion is detected by wait/poll on a handle, NOT events (nothing is pushed):
  - A short-wait START (gui_run_start, gui_analyze, gui_post_analyze, gui_device_*)
    waits up to wait_seconds (default 1.0): settles in time -> {status:'finished',
    <product>} (gui_run_start -> {tab, figure}; gui_analyze -> {summary, figure};
    gui_post_analyze -> {summary}; gui_device_* -> {snapshot}); a slow one degrades
    to {status:'pending'}, which you then drive with the matching _wait / _poll.
    gui_run_wait and gui_run_poll also fold 'figure' on 'finished'.
  - _poll returns immediately: 'finished' | 'running' | 'cancelled' | 'failed' |
    'no_operation'. 'cancelled' (user/agent cancel) is distinct from 'failed'.
    While 'running' the reply folds the live progress bars
    (active, bars[token/format/percent]) — no separate progress tool.
  - _wait BLOCKS your whole turn until the op ends and RAISES only on genuine
    failure; returns {status, waited_seconds[, ...]}:
    'finished' (success), 'cancelled' (user/agent cancel — NOT a raise; read
    optional 'feedback' for the Stop reason), 'timed_out' (still running — NOT
    a failure — re-wait or switch to _poll). Because a _wait holds the turn and
    nothing pushes a completion, reserve inline _wait for ops you expect to finish
    quickly; for a long op (a big sweep, a slow ramp) either _poll (non-blocking
    — you check back) or call the _wait from a BACKGROUND agent so your main loop
    stays free and the harness re-invokes you with the result.
  - USER FEEDBACK WAKEUP (ADR-0023): any _wait can return early with
    status='user_feedback' and a 'feedback' string while the op is STILL running.
    Treat the feedback as a HIGH-PRIORITY instruction and re-plan; you still hold
    the handle, so you may cancel (gui_run_cancel) or re-await.
  - INTERACTIVE analyze (see gui_adapter_guide): the user marks the plot and
    clicks Done, so it never settles in the short wait — a 'pending' is EXPECTED;
    prompt the user, then _poll (do not block on _wait).

Diagnostic push: every tool reply piggybacks any {severity:'error'|'info', title,
message} the GUI surfaced since your last call, under "notifications since last
call" — UNSOLICITED, including failures not tied to the call you just made.
Resource-change events are NOT exposed.

Stale model (optimistic concurrency): a guarded op (run / save / save_as_module) rejects
with precondition_failed when a dependency a GUI user changed under you moved
since you last observed it; the error names which resources to re-read. Re-read
then retry.

Call contract — read before issuing defensive/duplicate calls:
  - A failed call always raises; it never returns stale or partial data. One call
    is enough — never fire a backup copy of the same tool in the same turn.
  - Query tools (gui_*_list / _get* / _snapshot / _check / _active* / _poll) are
    read-only and side-effect-free: safe to retry across turns, wasteful to
    duplicate within one.
  - Mutating tools have side effects and must be sent exactly once: gui_run_start
    (a duplicate starts a SECOND run), gui_editor_set_field, gui_tab_new /
    gui_tab_close, gui_save_*, gui_device_connect / _disconnect / _setup,
    gui_context_set_* / _del_* / _rename_*, gui_editor_save_as_module.

Agent-to-user prompting: gui_notify_user(message, timeout=600) opens a prompt
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
# events are dropped in _deliver_event (Phase 120c-2). Diagnostics piggyback on
# the next tool reply; _DIAGNOSTIC_COND guards the queue (notified on append).
_DIAGNOSTIC_QUEUE_MAX = 1024
_DIAGNOSTIC_QUEUE: deque[dict[str, Any]] = deque(maxlen=_DIAGNOSTIC_QUEUE_MAX)
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
_LAST_SEEN: dict[str, int] = {}

# Dependency map (the single place that knows what each guarded op depends on).
# Patterns use {tab_id}/{editor_id} placeholders and a literal ``device:*`` that
# expands to every current device:* key. save.* does NOT depend on cfg — the
# saved content comes from the run result's own cfg_snapshot. writeback.apply
# depends on the run+analyze results it recomputes from, plus context (it writes
# md/ml). Note: md/ml content edits bump the ``context`` version, so any op
# depending on ``context`` (run.start / editor.commit / writeback.apply) detects
# a concurrent md/ml change.
_GUARD_DEPS: dict[str, tuple[str, ...]] = {
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
    # save.post_image renders the post-analysis figure, so it depends on the
    # post-analysis result (not the primary run result) + the save path.
    "save.post_image": ("tab:{tab_id}:post_analyze", "tab:{tab_id}:save_path"),
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
# Start ops (device.setup / run.start) return an ``operation_id``
# the agent never sees (mcp/RPC bookkeeping, like version numbers). The agent
# refers to an in-flight operation by a name it understands; mcp maps that
# semantic key to the latest operation_id for it. ``operation.await`` then
# blocks on that id. "Latest wins": starting overwrites the key, since the agent
# semantically means "the current operation for this resource".
_OP_BY_KEY: dict[str, int] = {}

# Which semantic key a start RPC's operation_id belongs to (param -> key).
# Device connect/disconnect/setup all key on the device name: "latest wins" means
# the most recent operation for that device is the one a wait tool awaits.
_OP_KEY_OF: dict[str, Callable[[dict[str, Any]], str]] = {
    "device.connect": lambda p: f"device:{p.get('name', '')}",
    "device.disconnect": lambda p: f"device:{p.get('name', '')}",
    "device.setup": lambda p: f"device:{p.get('name', '')}",
    "run.start": lambda p: f"tab:{p.get('tab_id', '')}",
    "analyze.start": lambda p: f"analyze:{p.get('tab_id', '')}",
    "post_analyze.start": lambda p: f"post_analyze:{p.get('tab_id', '')}",
}


def _deliver_event(msg: dict[str, Any]) -> None:
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


def _refresh_versions() -> None:
    """Re-read the full resource version table into ``_LAST_SEEN``.

    Pure read via ``resources.versions`` (the single read entry point). Called
    on connect and after a stale rejection so the next guarded op carries fresh
    baselines. Failures are swallowed — a missing table just means no guard.
    """
    try:
        resp = _BRIDGE.send_rpc_raw("resources.versions", {}, 5.0)
    except Exception:  # pragma: no cover — best-effort resync
        return
    if not resp.get("ok", False):
        return
    versions = resp.get("result", {}).get("versions")
    if isinstance(versions, dict):
        _LAST_SEEN.clear()
        _LAST_SEEN.update(versions)


def _build_expected_versions(method: str, params: dict[str, Any]) -> dict[str, int]:
    """Resolve a guarded method's dependency patterns into expected versions.

    Policy lives here: expand {tab_id}/{editor_id} placeholders and the literal
    ``device:*`` (every current device:* key) against ``_LAST_SEEN``. Returns the
    subset of versions the op depends on; the server compares only these.
    """
    deps = _GUARD_DEPS.get(method)
    if not deps:
        return {}
    expected: dict[str, int] = {}
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

    An agent should not have to call gui_connect by hand: the first time any
    gui_* tool reaches send_gui_rpc with no live socket, we resolve the running
    GUI's port via session discovery (the SAME path gui_connect takes) and
    attach. This is a measure-specific attach policy (session-discovery slug +
    resolve_connect_port), so it stays here rather than in the shared bridge,
    whose send_rpc_raw also serves the read-only apps.

    Only the control channel is attached — never soc.connect: choosing the SoC
    (mock vs remote) is the user's decision, not an auto-connect side effect.

    Fail-fast when no GUI is discoverable, with a message that no longer tells
    the agent to "call gui_connect first" (it is automatic now).
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
    (:func:`_ensure_connected`) so an agent never has to call gui_connect by
    hand. For guarded methods (run/save/commit) attaches ``expected_versions``
    from the mcp-side bookkeeping so the server can reject stale operations. On
    a stale rejection the version table is re-read so the agent's retry is fresh.
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


def tool_gui_disconnect(arguments: dict[str, Any]) -> str:
    del arguments
    note = _BRIDGE.disconnect()
    # App-specific housekeeping: drop any buffered diagnostics — they belong to
    # the connection that just closed.
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.clear()
    return note


def tool_gui_launch(arguments: dict[str, Any]) -> str:
    port = int(arguments.get("port", _CONFIG.default_port))
    token: str | None = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    clean = bool(arguments.get("clean", False))
    # lib/zcu_tools/mcp/measure -> repo root
    repo_root = Path(__file__).parents[4]
    # clean → run_measure_gui --clean (skip restoring the persisted session).
    extra_args = ["--clean"] if clean else None
    return _BRIDGE.launch(repo_root, port, token, auto_connect, extra_args=extra_args)


def tool_gui_stop(arguments: dict[str, Any]) -> str:
    # Graceful close over the existing RPC channel (app.shutdown runs the GUI's
    # normal window-close path on its main thread, no OS signal), then await /
    # optionally force-kill. timeout_kill defaults False here (measure-gui prefers
    # leaving a slow-closing GUI alone for a retry rather than killing it).
    timeout = float(arguments.get("timeout", 10.0))
    timeout_kill = bool(arguments.get("timeout_kill", False))
    note = _BRIDGE.stop(
        timeout=timeout, timeout_kill=timeout_kill, shutdown_rpc="app.shutdown"
    )
    # The bridge's disconnect does not clear measure-gui's diagnostic queue; do it
    # here so a later session does not see the previous one's buffered messages.
    with _DIAGNOSTIC_COND:
        _DIAGNOSTIC_QUEUE.clear()
    return note


# ---------------------------------------------------------------------------
# Workflow tools (thin pass-through wrappers)
# ---------------------------------------------------------------------------


def tool_gui_state_check(arguments: dict[str, Any]) -> dict[str, Any]:
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


def _assemble_overview() -> dict[str, Any]:
    """One-shot situational overview of the live GUI, fanned out over existing
    read RPCs (no new wire method).

    Packs the readiness flags, the project identity, the active context label, the
    SoC summary, the open tabs (each with its adapter + running flag), the running
    tab and the user's currently-focused tab. ``active_tab`` is where the USER's
    eye is (a collaboration cue) — NOT the agent's operation target, which is
    always the explicit tab_id the agent passes.

    ``project`` is read from project.info only while a project is applied
    (project.info fast-fails no_project otherwise); long keys {chip_name,
    qub_name, res_name} matching the wire shape (result_dir/database_path
    omitted for conciseness); ``null`` when no project. ``is_mock`` is
    likewise read from soc.info only
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
        # Use long keys to match project.info wire shape and other tool-GUIs
        # (fluxdep/dispersive/autofluxdep all use chip_name/qub_name/res_name).
        # result_dir/database_path are omitted here — overview stays concise.
        project = {
            "chip_name": info.get("chip_name"),
            "qub_name": info.get("qub_name"),
            "res_name": info.get("res_name"),
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
    """Resolve the editor session id from either an explicit ``editor_id`` or a
    ``tab_id`` (mutually exclusive, fail-fast).

    A tab's cfg form is a CfgEditorService session keyed by its tab_id; the agent
    can address it by tab_id without first running gui_tab_snapshot to pluck out
    the editor_id. The explicit editor_id path stays for non-tab editors (e.g.
    gui_editor_open on an ml entry). Fails fast on: neither / both supplied, or a
    tab whose form has no live model yet (editor_id is None on the snapshot).
    """
    editor_id = arguments.get("editor_id")
    tab_id = arguments.get("tab_id")
    if (editor_id is None) == (tab_id is None):
        raise ValueError("supply exactly one of 'editor_id' or 'tab_id'")
    if editor_id is not None:
        return str(editor_id)
    snap = send_gui_rpc("tab.snapshot", {"tab_id": str(tab_id)})
    resolved = snap.get("editor_id")
    if resolved is None:
        raise ValueError(
            f"tab {tab_id!r} cfg form has no live editor session yet "
            "(no editor_id on its snapshot)"
        )
    return str(resolved)


def _fold_tab_editing_context(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold a fresh tab's editing context into ``reply``, in place.

    After tab.new the agent always reads tab.snapshot (for the editor_id),
    tab.list_paths (the settable dotted paths) and tab.get_cfg_summary (the
    current values) before it can edit cfg. Folding those reads collapses ~4
    calls into one. Pure mcp-side fan-out over EXISTING wire reads — no wire
    change. Reused by gui_run_stage1 (Phase ①). Adds {editor_id, paths,
    cfg_summary}; the caller owns ``tab_id`` and ``adapter`` in ``reply``.
    """
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})
    reply["editor_id"] = snap.get("editor_id")
    reply["paths"] = send_gui_rpc("tab.list_paths", {"tab_id": tab_id}).get("paths")
    reply["cfg_summary"] = send_gui_rpc("tab.get_cfg_summary", {"tab_id": tab_id}).get(
        "summary"
    )
    return reply


def tool_gui_editor_set_field(arguments: dict[str, Any]) -> dict[str, Any]:
    """Set one cfg field, addressing the editor by ``editor_id`` OR ``tab_id``.

    Thin override over the editor.set_field RPC that adds tab_id resolution: a
    tab's cfg form is the editor session keyed by its tab_id, so the agent can
    edit it without first running gui_tab_snapshot to find the editor_id. Returns
    the RPC reply unchanged ({valid, removed, added}). See gui_editor_set_fields
    for the batch form and the value/path semantics.
    """
    editor_id = _resolve_editor_id(arguments)
    return send_gui_rpc(
        "editor.set_field",
        {
            "editor_id": editor_id,
            "path": str(arguments["path"]),
            "value": arguments["value"],
        },
    )


def tool_gui_editor_set_fields(arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply several editor.set_field edits to ONE editor, fail-fast in order.

    The editor is addressed by ``editor_id`` OR ``tab_id`` (a tab's cfg form is
    the editor session keyed by its tab_id). Convenience fan-out (a for-loop over
    the single-field RPC) — there is no atomicity: edits before the failing one
    stay applied and are NOT rolled back. On the first error this raises, reporting
    how many succeeded and which path failed so the agent can reconcile. On success
    returns ``{applied, valid}`` — the count applied and whether the resulting
    draft is valid. It does NOT echo cfg content (that would force a lowering pass
    which eagerly evaluates EvalValue); read the cfg with gui_tab_list_paths if
    needed.
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
                f"batch set_field failed at edits[{i}] (path={edit['path']!r}); "
                f"{i} edit(s) already applied and NOT rolled back: {exc}"
            ) from exc
        valid = bool(res.get("valid", True))
    return {"applied": len(edits), "valid": valid}


def tool_gui_context_set_md_attrs(arguments: dict[str, Any]) -> dict[str, Any]:
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


def tool_gui_context_get_md_attrs(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read several MetaDict attributes at once, fail-fast on any missing key.

    Convenience fan-out over context.get_md_attr (the read counterpart of
    gui_context_set_md_attrs). Returns ``{values: {key: value}}`` — a map keyed by
    the requested key (the natural shape for a batch read; the agent indexes
    straight in). Reads are side-effect-free, so there is no partial-state concern:
    an unknown key fails fast (the underlying RPC raises invalid_params), never
    silently skipped.
    """
    keys = arguments.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("'keys' must be a non-empty list")
    values: dict[str, Any] = {}
    for key in keys:
        key_str = str(key)
        res = send_gui_rpc("context.get_md_attr", {"key": key_str})
        values[key_str] = res.get("value")
    return {"values": values}


def _await_operation_by_key(key: str, what: str, timeout: float) -> dict[str, Any]:
    """Block until the latest operation for ``key`` settles, or ``timeout`` s
    elapse; semantic result.

    Returns ``{status, waited_seconds[, message[, feedback]]}``:
    - 'finished': settled OK.
    - 'cancelled': user/agent cancelled the op. ``feedback`` carries the Stop
      reason when "Send & Stop" was used; absent on a plain cancel. NOT a raise
      (ADR-0025 §cancelled-wire — cancelled is a normal terminal outcome, not a
      crash; the agent reads feedback and re-plans).
    - 'user_feedback': a user-feedback string arrived before the op settled
      (ADR-0025). ``feedback`` carries the text; ``reason`` is 'user_feedback'.
      The operation is still running; the agent holds the key and can re-await
      or cancel via gui_run_cancel.
    - 'timed_out': still running after the bounded wait — NOT a crash, no raise.
    - 'no_operation': nothing tracked.
    ``waited_seconds`` is how long the wait actually blocked. A genuine
    ``failed`` outcome still raises (the agent must see it as an error).
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
                "The operation is still running — you may gui_run_cancel or re-await."
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


def _poll_operation_by_key(key: str, what: str) -> dict[str, Any]:
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


def _start_op_with_short_wait(
    key: str,
    what: str,
    wait_seconds: float,
    product: Callable[[], dict[str, Any]],
    pending_hint: str,
) -> dict[str, Any]:
    """Wait briefly for a just-started async op, degrading to a handle on timeout.

    The start RPC must already have run (its operation_id captured into _OP_BY_KEY
    under ``key`` by send_gui_rpc). Awaits up to ``wait_seconds``:
    - settles in time -> ``{status:'finished', **product()}`` so the caller sees
      the op's resulting state immediately (device snapshot / tab snapshot / soc);
    - still running -> ``{status:'pending', message:<hint>}`` so the caller can
      use the matching wait tool. operation.await still raises on failure/cancel.

    Shared by device connect/disconnect/setup and run.start — every op that has
    both a fast and a slow mode gets the same degrade. (soc.connect is excluded:
    it is synchronous now and returns its product directly.)
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


def _run_tab_summary(tab_id: str) -> dict[str, Any]:
    """A run-finished tab summary: only {tab_id, interaction}. The full
    tab.snapshot also carries adapter_name / editor_id / save_paths, none of
    which change across a run — re-sending them every run is wasted tokens
    (the agent already has them from gui_tab_snapshot). To see the plot, call
    gui_tab_get_current_figure(tab_id)."""
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})
    interaction = snap.get("interaction", {}) if isinstance(snap, dict) else {}
    return {"tab_id": tab_id, "interaction": interaction}


def tool_gui_device_wait_operation(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until the named device's current operation completes (semantic).

    Covers connect / disconnect / setup — whichever is the latest operation for
    the device. Returns status='finished' on success; status='cancelled' (with
    optional 'feedback') on cancellation (NOT a raise); raises only on genuine
    failure; status='no_operation' if nothing is in flight for that device.
    """
    name = str(arguments["name"])
    timeout = float(arguments.get("timeout", 120.0))
    return _await_operation_by_key(
        f"device:{name}", f"Device {name!r} operation", timeout
    )


def tool_gui_device_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of the named device's latest operation (connect /
    disconnect / setup): finished / running / failed / no_operation."""
    name = str(arguments["name"])
    return _poll_operation_by_key(f"device:{name}", f"Device {name!r} operation")


def tool_gui_run_start(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start a run, waiting briefly for a fast (small reps/rounds) run to finish.

    A run has both modes — a tiny sweep finishes in well under a second, a big
    one takes minutes — so it degrades like device ops: settles in time ->
    {status:'finished', tab:<tab.snapshot>, figure:<path>} (has_run_result set;
    figure is the run plot rendered to a temp PNG, the op's OWN visual result);
    still running -> {status:'pending'} (no figure yet; await with gui_run_wait or
    poll with gui_run_poll). send_gui_rpc attaches the version guard + captures
    the operation_id under tab:<tab_id>.
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("run.start", {"tab_id": tab_id})
    reply = _start_op_with_short_wait(
        f"tab:{tab_id}",
        f"Run on tab {tab_id!r}",
        wait_seconds,
        lambda: {"tab": _run_tab_summary(tab_id)},
        f"await it with gui_run_wait(tab_id={tab_id!r}).",
    )
    # The figure is this op's OWN visual result — fold it on FINISHED only
    # (a pending run has no settled plot yet). Failure is swallowed so a
    # plotting hiccup never masks an otherwise-good run reply.
    return _fold_finished_figure(tab_id, reply)


def tool_gui_run_wait(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until the run on ``tab_id`` completes (semantic wait, mirrors
    gui_device_wait_operation). Raises only on genuine failure. A cancelled run
    returns {status:'cancelled', feedback?} — read 'feedback' for the Stop reason
    (present when the user used "Send & Stop", absent on a plain cancel). On
    status='finished' the reply carries 'figure' — the run plot rendered to a
    temp PNG (the op's OWN visual result)."""
    tab_id = str(arguments["tab_id"])
    timeout = float(arguments.get("timeout", 600.0))
    reply = _await_operation_by_key(f"tab:{tab_id}", f"Run on tab {tab_id!r}", timeout)
    return _fold_finished_figure(tab_id, reply)


def tool_gui_run_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of the run on ``tab_id``: finished / running / failed
    / no_operation. Lets the agent start a slow run, do other work, then check
    back without blocking. On 'finished' the reply carries 'figure' — the run
    plot rendered to a temp PNG (the op's OWN visual result). Running / cancelled
    / failed / no_operation replies are unchanged."""
    tab_id = str(arguments["tab_id"])
    reply = _poll_operation_by_key(f"tab:{tab_id}", f"Run on tab {tab_id!r}")
    return _fold_finished_figure(tab_id, reply)


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


def tool_gui_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start analyze, waiting briefly (degrades like a run).

    Analyze has both modes — a FIT computes on a worker (usually finishes in well
    under a second), an INTERACTIVE pick waits for the USER to mark the plot and
    click Done (never settles in the short wait). So it degrades like gui_run_start:
    settles -> {status:'finished', summary, figure}; still running ->
    {status:'pending'} (await with gui_analyze_wait or gui_analyze_poll). For an
    INTERACTIVE adapter (see gui_adapter_guide) a 'pending' is expected — prompt
    the user to do the pick, then poll. 'updates' optionally overrides analyze
    params. A finished FIT reply carries the fit 'summary' (same shape as
    gui_tab_get_analyze_result — analyze's OWN result, the *_err fields included)
    AND 'figure' — the fit plot rendered to a temp PNG (analyze's OWN visual
    result). Review the proposed writeback with gui_writeback_preview (not folded
    here; that fold lives in gui_run_stage3). 'summary' is None and no 'figure'
    on an INTERACTIVE 'pending' (nothing settled yet).
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    # Start (captures operation_id under analyze:<tab_id>, strips it from reply),
    # then wait briefly. A FIT usually finishes here; an INTERACTIVE pick degrades
    # to a handle the user/agent then drives to completion.
    send_gui_rpc("analyze.start", params)
    reply = _start_op_with_short_wait(
        f"analyze:{tab_id}",
        f"Analyze on tab {tab_id!r}",
        wait_seconds,
        lambda: _analyze_summary_product("tab.get_analyze_result", tab_id),
        f"poll with gui_analyze_poll(tab_id={tab_id!r}); for an INTERACTIVE pick, "
        "prompt the user to mark the lines + click Done first.",
    )
    # The figure is analyze's OWN visual result — fold it on a FINISHED FIT reply.
    # An INTERACTIVE 'pending' has no settled plot yet (_fold_finished_figure is a
    # no-op on any non-finished status). writeback_preview stays in gui_run_stage3.
    return _fold_finished_figure(tab_id, reply)


def tool_gui_analyze_wait(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until the analyze on ``tab_id`` completes (mirrors gui_run_wait).
    Returns {status, waited_seconds}: 'finished' / 'cancelled' (read optional
    'feedback' for the Stop reason) / 'timed_out' (re-wait or gui_analyze_poll)
    / 'no_operation'. Raises only on a genuine failure. Use after gui_analyze
    returned status='pending'. NOTE: for an INTERACTIVE pick this blocks until
    the USER clicks Done — prefer gui_analyze_poll (non-blocking) so you can
    prompt and check back, or run this from a background agent. See the fit plot
    with gui_tab_get_current_figure(tab_id)."""
    tab_id = str(arguments["tab_id"])
    timeout = float(arguments.get("timeout", 600.0))
    return _await_operation_by_key(
        f"analyze:{tab_id}", f"Analyze on tab {tab_id!r}", timeout
    )


def tool_gui_analyze_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of the analyze on ``tab_id``: finished / running /
    failed / no_operation. For an INTERACTIVE pick, 'running' means the user has
    not clicked Done yet — keep checking back. On a finished analyze, read the
    scalar result with gui_tab_get_analyze_result and the fit plot with
    gui_tab_get_current_figure(tab_id)."""
    tab_id = str(arguments["tab_id"])
    return _poll_operation_by_key(f"analyze:{tab_id}", f"Analyze on tab {tab_id!r}")


def tool_gui_post_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start the second-layer (post) analysis, waiting briefly (degrades like a run).

    Post-analysis runs on top of the tab's PRIMARY analyze result (e.g.
    single-shot multi-backend ge discrimination) and is FIT-only — it computes on
    a worker, so it usually settles in the short wait -> {status:'finished',
    summary:{...}} (the fit summary is folded in, same shape as
    gui_tab_get_post_analyze_result, so the common read happens in one call). A
    slow one degrades to {status:'pending'} (await with gui_post_analyze_wait or
    gui_post_analyze_poll). Fast-fails with precondition_failed when the tab has no
    primary analyze result yet — run gui_analyze first. 'updates' optionally
    overrides post params (see gui_tab_get_post_analyze_params). The post figure is
    kept in the tab's separate post container; gui_tab_get_post_analyze_result
    stays for re-fetch and for the wait/poll path.
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    # Start (captures operation_id under post_analyze:<tab_id>, strips it from the
    # reply), then wait briefly. A FIT-only worker usually finishes in the wait.
    send_gui_rpc("post_analyze.start", params)
    return _start_op_with_short_wait(
        f"post_analyze:{tab_id}",
        f"Post-analysis on tab {tab_id!r}",
        wait_seconds,
        lambda: _analyze_summary_product("tab.get_post_analyze_result", tab_id),
        f"poll with gui_post_analyze_poll(tab_id={tab_id!r}).",
    )


def tool_gui_post_analyze_wait(arguments: dict[str, Any]) -> dict[str, Any]:
    """Block until the post-analysis on ``tab_id`` completes (mirrors
    gui_analyze_wait). Returns {status, waited_seconds}: 'finished' / 'cancelled'
    (read optional 'feedback' for the Stop reason) / 'timed_out' (re-wait or
    gui_post_analyze_poll) / 'no_operation'. Raises only on a genuine failure.
    Use after gui_post_analyze returned status='pending'. Read the scalar result
    with gui_tab_get_post_analyze_result."""
    tab_id = str(arguments["tab_id"])
    timeout = float(arguments.get("timeout", 600.0))
    return _await_operation_by_key(
        f"post_analyze:{tab_id}", f"Post-analysis on tab {tab_id!r}", timeout
    )


def tool_gui_post_analyze_poll(arguments: dict[str, Any]) -> dict[str, Any]:
    """Non-blocking status of the post-analysis on ``tab_id``: finished / running /
    failed / no_operation. On a finished post-analysis read the scalar result with
    gui_tab_get_post_analyze_result."""
    tab_id = str(arguments["tab_id"])
    return _poll_operation_by_key(
        f"post_analyze:{tab_id}", f"Post-analysis on tab {tab_id!r}"
    )


def tool_gui_soc_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    """Connect the SoC SYNCHRONOUSLY and return its hardware summary.

    Unlike run / analyze / device ops, connect is no longer a degrading async
    handle: the soc.connect RPC runs the connect on the GUI's main thread and
    returns once the board is connected and all post-connect side effects are
    applied, so this is one blocking call with no _wait / _poll follow-up.
    kind='mock' (offline) or kind='remote' with ip+port. Returns
    {soc:{description, is_mock}}; call gui_soc_info for the structured cfg. A
    remote connect to an unreachable board fails fast (~1s).
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
    return {"status": "finished", "soc": result.get("soc")}


def tool_gui_dialog_screenshot(arguments: dict[str, Any]) -> dict[str, Any]:
    """Capture a currently-open dialog as a PNG FILE and return its path.

    Mirrors gui_tab_get_current_figure: the convenience layer never returns
    inline base64 (a dialog grab would blow the tool-output token budget — the
    footgun this override removes). The wire dialog.screenshot has no out_path
    mode, so we decode the base64 here and write it; when out_path is omitted we
    synthesise a per-dialog temp path under gettempdir() (overwriting the previous
    grab of the same dialog). The raw wire method still returns base64 for non-MCP
    consumers.
    """
    import base64

    name = str(arguments["name"])
    out_path_arg = arguments.get("out_path")
    out_path = (
        str(out_path_arg)
        if out_path_arg is not None
        else str(Path(gettempdir()) / f"measure_dialog_{name}.png")
    )
    res = send_gui_rpc("dialog.screenshot", {"name": name})
    png = base64.b64decode(res["png_b64"])
    Path(out_path).write_bytes(png)
    return {"bytes": res.get("bytes", len(png)), "saved_to": out_path}


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
    review the fit AND the proposed writeback in one call before gui_writeback_apply
    (Phase ③). Only acts on ``reply['status'] == 'finished'`` (an INTERACTIVE
    'pending' has not produced a draft yet). The wire writeback.preview reply is
    {items: [...]}; we surface that list under 'writeback_preview'. Mirrors the
    figure/guide folds: a fetch failure is swallowed (omitted) so a preview hiccup
    never breaks the analyze reply — the agent can still call gui_writeback_preview.
    """
    if reply.get("status") != "finished":
        return reply
    try:
        reply["writeback_preview"] = send_gui_rpc(
            "writeback.preview", {"tab_id": tab_id}
        ).get("items")
    except Exception:
        pass
    return reply


# ---------------------------------------------------------------------------
# Bundle tools — the four-phase recommended flow (gui_run_stage1..4)
#
# Each bundle composes several BASE operations into the agent's natural decision
# points and folds in the OTHER operations' outputs (the cross-tool folds the
# base tools deliberately do NOT carry). The base tools stay pure (each returns
# only its own result, least-surprise); the folding lives only here.
# ---------------------------------------------------------------------------


def tool_gui_run_stage1(arguments: dict[str, Any]) -> dict[str, Any]:
    """Phase ① new+guide: create a tab for ``adapter_name`` and fold its editing
    context + the adapter guide into one reply.

    Composes tab.new with the fan-out reads the agent always makes before editing
    cfg (tab.snapshot for editor_id, tab.list_paths, tab.get_cfg_summary) plus the
    adapter's orientation guide (adapter.guide). The guide is ALWAYS returned (you
    call stage1 precisely to get a tab + its guide — no first-use gating). Returns
    {tab_id, adapter, editor_id, paths, cfg_summary, guide}; edit cfg + run with
    gui_run_stage2(tab_id, edits).
    """
    adapter_name = str(arguments["adapter_name"])
    tab_id = str(send_gui_rpc("tab.new", {"adapter_name": adapter_name})["tab_id"])
    reply: dict[str, Any] = {"tab_id": tab_id, "adapter": adapter_name}
    _fold_tab_editing_context(tab_id, reply)
    reply["guide"] = send_gui_rpc("adapter.guide", {"adapter_name": adapter_name}).get(
        "guide"
    )
    return reply


def tool_gui_run_stage2(arguments: dict[str, Any]) -> dict[str, Any]:
    """Phase ② configure+run: apply ``edits`` then run the existing ``tab_id``,
    STOPPING before analyze.

    Composes gui_editor_set_fields (the PLURAL path, so numeric values stay
    numbers) when ``edits`` is given, then gui_run_start. ``edits`` is an OPTIONAL
    {path: value} map (omit/empty runs the tab's current cfg). A finished reply is
    gui_run_start's reply ({status, tab}) with {figure, analyze_params} folded in —
    the run plot rendered to a temp PNG and the analyze knobs for this tab — so the
    agent sees them together. A slow run degrades to {status:'pending'} (no folds;
    drive it with gui_run_wait / gui_run_poll).

    It deliberately STOPS before analyze: a successful run is NOT a successful
    analyze — look at the figure, then gui_run_stage3.
    """
    tab_id = str(arguments["tab_id"])
    edits = arguments.get("edits") or {}
    if not isinstance(edits, dict):
        raise ValueError("'edits' must be an object mapping path -> value")
    # Apply edits through the plural set_fields handler (addressed by tab_id) so
    # the numbers-stay-numbers guarantee is inherited rather than re-implemented.
    if edits:
        tool_gui_editor_set_fields(
            {
                "tab_id": tab_id,
                "edits": [{"path": str(p), "value": v} for p, v in edits.items()],
            }
        )
    reply = tool_gui_run_start({"tab_id": tab_id})
    # The figure is already folded by tool_gui_run_start (MCP 46); do NOT
    # double-fold it here. Only add analyze_params (the stage-specific fold).
    # _fold_analyze_params is a no-op on a non-finished reply, so the pending
    # path is safe.
    return _fold_analyze_params(tab_id, reply)


def tool_gui_run_stage3(arguments: dict[str, Any]) -> dict[str, Any]:
    """Phase ③ analyze: analyze ``tab_id`` and fold the fit review into one reply.

    Composes gui_analyze; a finished FIT reply ({status, summary}) gains
    {figure, writeback_preview} — the fit plot rendered to a temp PNG and the
    proposed writeback values/targets the fit produced — so the agent reviews the
    fit AND the proposed writeback in one call before gui_run_stage4. An
    INTERACTIVE analysis degrades to {status:'pending'} (no folds — nothing
    settled to render/preview); prompt the user, then gui_analyze_poll.
    """
    tab_id = str(arguments["tab_id"])
    reply = tool_gui_analyze({"tab_id": tab_id})
    # The figure is already folded by tool_gui_analyze (MCP 46); do NOT
    # double-fold it here. Only add writeback_preview (the stage-specific fold).
    return _fold_writeback_preview(tab_id, reply)


def tool_gui_run_stage4(arguments: dict[str, Any]) -> dict[str, Any]:
    """Phase ④ writeback+save: apply the tab's writeback draft, optionally saving.

    Composes writeback.apply (applies the currently-selected draft items; returns
    {applied_ids}); when ``save_data`` is true it ALSO chains save.data for the
    same tab and folds the resolved {data_path}. ``save_data`` defaults false
    (apply only). The two steps are deliberately NOT atomic — apply runs first and
    is committed even if the follow-up save fails (the save error propagates so the
    agent sees it, but the writeback is already applied).
    """
    tab_id = str(arguments["tab_id"])
    reply = dict(send_gui_rpc("writeback.apply", {"tab_id": tab_id}))
    if bool(arguments.get("save_data", False)):
        # save.data resolves the destination from the tab's save_path; data_path
        # omitted here so the configured path is used (same as gui_save_data).
        reply["data_path"] = send_gui_rpc("save.data", {"tab_id": tab_id}).get(
            "data_path"
        )
    return reply


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
# Phase 81b tools — context queries / device queries
# ---------------------------------------------------------------------------


def tool_gui_device_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    params: dict[str, Any] = {
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
        f"await it with gui_device_wait_operation(name={name!r}) or gui_device_poll(name={name!r}).",
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
        f"Device {name!r} setup",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        f"await it with gui_device_wait_operation(name={name!r}) or watch 'device_setup_finished'.",
    )


# Seconds the off-main consumer backstop outlasts the dialog's QTimer so the
# dialog is the timeout SSOT (ADR-0025 §dialog-timeout): the QTimer fires at
# `timeout`, enqueues a Timeout event, and the consumer (waiting `timeout+slack`)
# reads it instead of pre-empting — closing the lost-reply gap at the boundary.
_NOTIFY_CONSUMER_SLACK: float = 10.0


def tool_gui_notify_user(arguments: dict[str, Any]) -> dict[str, Any]:
    """Agent-initiated user prompt. BLOCKS the turn until user responds.

    Serially composes notify.open (main thread: mint token + open dialog) and
    notify.await (off-main: block until Reply/Dismiss/QTimer). Neither method
    uses _OP_KEY_OF — not a start-op, no operation_id captured.

    Returns {reason: 'reply'|'dismiss'|'timeout', reply?}. Never raises on
    timeout or dismiss — those are expected outcomes (ADR-0025 §6).
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
        "dialog.screenshot",
        "tab.get_current_figure",
        # hand-written override: adds tab_id -> editor_id resolution on top of the
        # editor.set_field RPC (a tab's cfg form is the editor session keyed by its
        # tab_id), so the generator must not also emit gui_editor_set_field.
        "editor.set_field",
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
        # hand-written short-wait degrade (like device ops): a fast run returns its
        # product, a slow one degrades to a handle (gui_run_wait).
        "run.start",
        # hand-written synchronous wrapper (passes an explicit ~2s timeout so the
        # board-side 1s COMMTIMEOUT fires first); not auto-generated so the override
        # gui_soc_connect tool is the only soc-connect surface.
        "soc.connect",
        # hand-written short-wait degrade (analyze is an async worker / interactive
        # pick; mirrors run.start). To see the fit plot the agent calls
        # gui_tab_get_current_figure.
        "analyze.start",
        # hand-written short-wait degrade (FIT-only worker, mirrors analyze).
        "post_analyze.start",
        # notify.open / notify.await are the two-RPC internals of gui_notify_user;
        # the agent never calls them directly — only the hand-written tool is exposed.
        "notify.open",
        "notify.await",
    }
)


# Tool generation (coerce / forward / per-spec schema) is the shared
# ``generate_tools`` helper; measure-gui's guarded ``send_gui_rpc`` is injected as
# the send_fn so generated forwarders carry the version guard + operation capture.


# ---------------------------------------------------------------------------
# Hand-written tools — lifecycle + overrides that the generator cannot express
# ---------------------------------------------------------------------------


_OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_connect": {
        "handler": tool_gui_connect,
        "description": (
            "Connect the MCP bridge to an ALREADY-RUNNING GUI's TCP control port. "
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
    "gui_overview": {
        "handler": tool_gui_overview,
        "description": (
            "One-shot SITUATIONAL OVERVIEW of the live GUI — call any time to "
            "re-orient. Packs (from existing read RPCs): state (the four readiness "
            "flags), project ({chip_name,qub_name,res_name} or null when none applied), "
            "context (active context label), soc ({connected, is_mock}), "
            "tabs ([{tab_id, adapter, is_running}]), running_tab, and active_tab. "
            "active_tab is where the USER is currently focused — a collaboration "
            "cue, NOT your operation target (you always act on an explicit tab_id). "
            "The same overview is folded into gui_connect's reply, "
            "so right after connecting you already have it."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_editor_set_field": {
        "handler": tool_gui_editor_set_field,
        "description": (
            "Set ONE field in a cfg-editor session, addressing it by 'editor_id' "
            "OR 'tab_id' (supply exactly one). A tab's cfg form is the editor "
            "session keyed by its tab_id, so passing 'tab_id' edits the tab "
            "directly without first running gui_tab_snapshot to find the "
            "editor_id; 'editor_id' still works for non-tab editors (gui_editor_open "
            "on an ml entry). 'path' is dotted (see gui_tab_list_paths); 'value' is "
            "a JSON scalar or an md-ref {__kind:eval, expr} (the eval form is "
            "accepted only on a scalar leaf, never a sweep edge). Returns "
            "{valid, removed, added} — does NOT echo cfg content (read it with "
            "gui_tab_list_paths). A tab whose form has no live model yet fails fast."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "editor_id": {
                    "type": "string",
                    "description": (
                        "Editor session id (from gui_tab_snapshot or "
                        "gui_editor_open). Supply this OR 'tab_id'."
                    ),
                },
                "tab_id": {
                    "type": "string",
                    "description": (
                        "Edit the tab's cfg form directly (resolves its editor_id). "
                        "Supply this OR 'editor_id'."
                    ),
                },
                "path": {"type": "string", "description": "Dotted field path"},
                "value": {
                    # Untyped schema = "any JSON value", matching the generator's
                    # JsonType.JSON rendering (schema_property). A 'type' union that
                    # listed "string" let the MCP client coerce a number (0.2) to
                    # the string "0.2", which then failed the float-field check; an
                    # untyped schema passes a number through unchanged.
                    "description": "JSON scalar or {__kind:eval, expr}",
                },
            },
            "required": ["path", "value"],
        },
    },
    "gui_editor_set_fields": {
        "handler": tool_gui_editor_set_fields,
        "description": (
            "Batch-apply several field edits to ONE cfg-editor session in order, "
            "addressing it by 'editor_id' OR 'tab_id' (supply exactly one; a tab's "
            "cfg form is the editor session keyed by its tab_id). Convenience "
            "fan-out over gui_editor_set_field — NOT atomic: it stops "
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
                    "description": (
                        "Editor session id (from gui_tab_snapshot or "
                        "gui_editor_open). Supply this OR 'tab_id'."
                    ),
                },
                "tab_id": {
                    "type": "string",
                    "description": (
                        "Edit the tab's cfg form directly (resolves its editor_id). "
                        "Supply this OR 'editor_id'."
                    ),
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
            "required": ["edits"],
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
    "gui_context_get_md_attrs": {
        "handler": tool_gui_context_get_md_attrs,
        "description": (
            "Batch-read several MetaDict attributes at once — the read counterpart "
            "of gui_context_set_md_attrs. Convenience fan-out over "
            "gui_context_get_md_attr. Returns {values: {key: value}} keyed by the "
            "requested key. An unknown key fails fast (invalid_params) — keys are "
            "never silently skipped. Reads are side-effect-free."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "description": "MetaDict keys to read",
                    "items": {"type": "string"},
                },
            },
            "required": ["keys"],
        },
    },
    "gui_device_wait_operation": {
        "handler": tool_gui_device_wait_operation,
        "description": (
            "Block until the named device's current operation — connect / "
            "disconnect / setup, whichever was started last — completes (shared "
            "async _wait contract; see server instructions, esp. blocking / "
            "background for a slow ramp). Use after a gui_device_* tool returned "
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
            "Non-blocking status of the named device's latest operation — connect "
            "/ disconnect / setup (shared async _poll contract; see server "
            "instructions). While 'running' the folded progress bars cover e.g. a "
            "setup ramp."
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
            "Start a run on tab_id (shared short-wait START contract — see server "
            "instructions). A fast run settles -> {status:'finished', tab:{...}, "
            "figure:<path>} — the tab snapshot (has_run_result set) AND the run "
            "plot rendered to a temp PNG (the run's OWN visual result). A slow run "
            "degrades to {status:'pending'} (no figure yet; drive with gui_run_wait "
            "/ gui_run_poll)."
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
            "Block until the run on tab_id completes (the shared async _wait "
            "contract — see the server instructions for blocking / background / "
            "user-feedback / timed_out semantics). Use after gui_run_start returned "
            "status='pending'. On 'finished', the reply carries 'figure' — the run "
            "plot rendered to a temp PNG (the run's OWN visual result)."
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
            "Non-blocking status of the run on tab_id (shared async _poll contract "
            "— see server instructions for the status enum + folded progress bars). "
            "On 'finished' the reply carries 'figure' — the run plot rendered to a "
            "temp PNG (the run's OWN visual result). Running/cancelled/failed replies "
            "are unchanged (no figure)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_run_stage1": {
        "handler": tool_gui_run_stage1,
        "description": (
            "Phase ① of the recommended flow — new+guide. Create a tab for "
            "'adapter_name' (see gui_adapter_list) and fold its editing context + "
            "the adapter guide into ONE reply. Bundles tab.new + the fan-out reads "
            "the agent always makes before editing cfg (tab.snapshot for editor_id, "
            "tab.list_paths, tab.get_cfg_summary) + adapter.guide. Returns "
            "{tab_id, adapter, editor_id, paths, cfg_summary, guide}: 'guide' is the "
            "adapter's orientation guide (ALWAYS returned here); 'paths' are the "
            "settable dotted paths (the gui_editor_set_field path source); "
            "'cfg_summary' is the read-only nested values view. Then configure + run "
            "with gui_run_stage2(tab_id, edits)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_name": {
                    "type": "string",
                    "description": "Adapter to instantiate (see gui_adapter_list)",
                },
            },
            "required": ["adapter_name"],
        },
    },
    "gui_run_stage2": {
        "handler": tool_gui_run_stage2,
        "description": (
            "Phase ② of the recommended flow — configure+run. Apply 'edits' then "
            "run the already-created 'tab_id' (from gui_run_stage1), then STOP "
            "before analyze. Bundles gui_editor_set_fields + gui_run_start. 'edits' "
            "is an OPTIONAL {path: value} map (dotted paths, see gui_tab_list_paths); "
            "it goes through the plural set_fields path so numbers stay numbers (not "
            "stringified). Omit/empty 'edits' to run the tab's current cfg. A fast "
            "run returns {status:'finished', tab, figure, analyze_params} — 'figure' "
            "comes from gui_run_start's own FINISHED reply (the run plot rendered to "
            "a temp PNG); 'analyze_params' is the stage-specific fold (the analyze "
            "knobs for this tab). A slow run degrades to {status:'pending'} (drive "
            "it with gui_run_wait / gui_run_poll; 'figure' arrives in the wait/poll "
            "finished reply). STOPS before analyze on purpose: a successful run is "
            "NOT a successful analyze — look at the figure, then gui_run_stage3."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to configure + run (from gui_run_stage1)",
                },
                "edits": {
                    "type": "object",
                    "description": (
                        "Optional {path: value} cfg edits applied before the run "
                        "(dotted paths, see gui_tab_list_paths). Numbers stay "
                        "numbers. Omit/empty to run with the tab's current cfg."
                    ),
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_run_stage3": {
        "handler": tool_gui_run_stage3,
        "description": (
            "Phase ③ of the recommended flow — analyze. Analyze 'tab_id' and fold "
            "the writeback review into ONE reply. Bundles gui_analyze + the writeback "
            "preview. A finished FIT returns {status:'finished', summary, figure, "
            "writeback_preview} — 'summary' is the fit result (same shape as "
            "gui_tab_get_analyze_result), 'figure' comes from gui_analyze's own "
            "FINISHED reply (the fit plot rendered to a temp PNG), and "
            "'writeback_preview' is the stage-specific fold (the proposed writeback "
            "values/targets) — so you review the fit + the proposed writeback in one "
            "call before gui_run_stage4. An INTERACTIVE analysis (e.g. flux_dep) "
            "degrades to {status:'pending'} (no folds) — prompt the user, then "
            "gui_analyze_poll."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab to analyze (from gui_run_stage2)",
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_run_stage4": {
        "handler": tool_gui_run_stage4,
        "description": (
            "Phase ④ of the recommended flow — writeback+save. Apply the tab's "
            "writeback draft (edit it first via gui_writeback_set / gui_editor_*), "
            "optionally saving the data afterwards. Bundles writeback.apply + "
            "(optionally) save.data. Applies the items currently selected; returns "
            "{applied_ids}. OPTIONAL save_data (default false): when true, ALSO saves "
            "the run data after applying and folds the resolved {data_path} into the "
            "reply (apply + save in one call). The apply runs first and is committed "
            "even if the follow-up save fails; with save_data=false the behavior is "
            "apply-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "string",
                    "description": "Tab whose writeback draft to apply (from gui_run_stage3)",
                },
                "save_data": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, save the run data after applying the writeback "
                        "and fold the resolved data_path into the reply. Default "
                        "false (apply only)."
                    ),
                },
            },
            "required": ["tab_id"],
        },
    },
    "gui_analyze": {
        "handler": tool_gui_analyze,
        "description": (
            "Start analyze on tab_id (shared short-wait START contract — see server "
            "instructions, incl. the INTERACTIVE-analyze note). A FIT settles -> "
            "{status:'finished', summary, figure} — the fit summary (same shape as "
            "gui_tab_get_analyze_result, the *_err fields included) AND the fit plot "
            "rendered to a temp PNG (analyze's OWN visual result). Review the "
            "proposed writeback with gui_writeback_preview (not folded here; that "
            "fold lives in gui_run_stage3). An INTERACTIVE analysis (e.g. flux_dep) "
            "degrades to {status:'pending', summary:None} — no figure (nothing "
            "settled yet); prompt the user to mark the plot + click Done, then "
            "gui_analyze_poll. 'updates' optionally overrides analyze params (see "
            "gui_adapter_analyze_spec)."
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
    "gui_analyze_wait": {
        "handler": tool_gui_analyze_wait,
        "description": (
            "Block until the analyze on tab_id completes (shared async _wait "
            "contract — see server instructions). Use after gui_analyze returned "
            "status='pending'. On 'finished', see the fit plot with "
            "gui_tab_get_current_figure. For an INTERACTIVE pick this blocks until "
            "the USER clicks Done — prefer gui_analyze_poll."
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
    "gui_analyze_poll": {
        "handler": tool_gui_analyze_poll,
        "description": (
            "Non-blocking status of the analyze on tab_id (shared async _poll "
            "contract — see server instructions). For an INTERACTIVE pick, "
            "'running' means the user has not clicked Done yet — keep checking back. "
            "On 'finished', read gui_tab_get_analyze_result + the plot with "
            "gui_tab_get_current_figure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_post_analyze": {
        "handler": tool_gui_post_analyze,
        "description": (
            "Start the second-layer (post) analysis on tab_id (shared short-wait "
            "START contract — see server instructions). Runs on top of the tab's "
            "PRIMARY analyze result (e.g. single-shot multi-backend ge "
            "discrimination) and is FIT-only (no INTERACTIVE mode), so it usually "
            "settles -> {status:'finished', summary:{...}} (folded in, same shape as "
            "gui_tab_get_post_analyze_result). Fast-fails with precondition_failed "
            "when the tab has no primary analyze result yet — run gui_analyze first. "
            "'updates' optionally overrides post params (see "
            "gui_tab_get_post_analyze_params). The post figure shares the tab's plot "
            "container; see it with gui_tab_get_current_figure and persist it with "
            "gui_save_post_image."
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
    "gui_post_analyze_wait": {
        "handler": tool_gui_post_analyze_wait,
        "description": (
            "Block until the post-analysis on tab_id completes (shared async _wait "
            "contract — see server instructions). Use after gui_post_analyze "
            "returned status='pending'. On 'finished', read the scalar result with "
            "gui_tab_get_post_analyze_result."
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
    "gui_post_analyze_poll": {
        "handler": tool_gui_post_analyze_poll,
        "description": (
            "Non-blocking status of the post-analysis on tab_id (shared async _poll "
            "contract — see server instructions). On 'finished', read the scalar "
            "result with gui_tab_get_post_analyze_result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tab_id": {"type": "string"}},
            "required": ["tab_id"],
        },
    },
    "gui_soc_connect": {
        "handler": tool_gui_soc_connect,
        "description": (
            "Connect the SoC SYNCHRONOUSLY (NOT a degrading async handle — there is "
            "no gui_soc_connect_wait / _poll). One blocking call returns "
            "{status:'finished', soc:{description, is_mock}} once the board is "
            "connected (call gui_soc_info for the structured cfg). kind='mock' "
            "(offline) or kind='remote' with ip+port. A remote connect to an "
            "unreachable board fails fast (~1s)."
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
    "gui_dialog_screenshot": {
        "handler": tool_gui_dialog_screenshot,
        "description": (
            "Capture a currently-open dialog to a PNG FILE and return its path. "
            "name must be one of: setup, device, predictor, inspect, startup "
            "(same dialog names as gui_dialog_open / gui_dialog_close). "
            "Fails with PRECONDITION_FAILED if the named dialog is not currently open. "
            "The PNG is ALWAYS written to disk and the reply is {saved_to, bytes} — "
            "Read the saved_to path to view it (never inline base64, so it cannot "
            "blow the token budget). Omit out_path to write a per-dialog file under "
            "the temp dir (overwritten each call); pass out_path (absolute) to choose "
            "the location."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "One of: setup, device, predictor, inspect, startup",
                },
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional absolute path to write the PNG; omit to use a "
                        "per-dialog file under the temp dir"
                    ),
                },
            },
            "required": ["name"],
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
            "address. remember defaults to true (device persists across "
            "sessions); set remember=false for a memory-only device."
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
                    "description": "Persist device across sessions (default true)",
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
            "with gui_device_wait_operation or gui_device_poll — a 'running' poll "
            "reply carries the live progress bars, e.g. a setup ramp). The device "
            "must already be connected."
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
            "RARELY NEEDED: run/analyze FINISHED replies ALREADY fold the figure "
            "(including 2D scans via gui_run_start/_wait/_poll and gui_analyze). "
            "Call this only when you genuinely need it — a re-render, a mid-flight "
            "(non-finished) plot you must inspect, or you want to choose out_path. "
            "Normally read the 'figure' field from the finished run/analyze reply. "
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
    "gui_notify_user": {
        "handler": tool_gui_notify_user,
        "description": (
            "Ask the user a question and BLOCK your entire turn until they respond "
            "(or the timeout expires). Opens a non-modal prompt dialog in the GUI "
            "showing 'message'. Returns {reason, reply?}:\n"
            "  - reason='reply': user answered; read 'reply' (a string, possibly "
            "empty) and act on it.\n"
            "  - reason='dismiss': user explicitly closed the prompt — do NOT ask "
            "again immediately; respect the user's choice.\n"
            "  - reason='timeout': no one responded within 'timeout' seconds — the "
            "user is probably not watching; do NOT keep blocking, continue or poll.\n"
            "timeout (default 600s) is the dialog auto-close timer. The RPC call "
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
                        "(default 600). Set shorter for time-sensitive prompts."
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
        "gui_connect",
        "gui_disconnect",
        "gui_launch",
        "gui_stop",
        "gui_device_connect",
        "gui_device_disconnect",
        "gui_device_setup",
        "gui_dialog_screenshot",
        "gui_tab_get_current_figure",
        "gui_state_check",
        "gui_overview",
        "gui_editor_set_field",
        "gui_editor_set_fields",
        "gui_context_set_md_attrs",
        "gui_context_get_md_attrs",
        "gui_device_wait_operation",
        "gui_run_start",
        "gui_run_wait",
        "gui_run_poll",
        "gui_run_stage1",
        "gui_run_stage2",
        "gui_run_stage3",
        "gui_run_stage4",
        "gui_analyze",
        "gui_analyze_wait",
        "gui_analyze_poll",
        "gui_post_analyze",
        "gui_post_analyze_wait",
        "gui_post_analyze_poll",
        "gui_soc_connect",
        "gui_device_poll",
        "gui_notify_user",
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
    """Attach the MCP server process's per-session file logging (Phase 157).

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
    run-failure reason) without a dedicated poll. Only diagnostics ride here now —
    resource-change events are not exposed to the agent (Phase 120c-2). Rendered
    as a compact one-line-per-diagnostic list (no indented JSON) to keep the
    piggyback token-light.
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
