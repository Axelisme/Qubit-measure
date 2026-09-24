#!/usr/bin/env python
"""MCP server bridge for ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live GUI's ``RemoteControlAdapter`` over a
single persistent TCP socket. Event push from the GUI is received by a
dedicated reader thread, parked in an internal queue and exposed to the LLM
piggybacked on successful tool replies. Low-frequency resource events are
best-effort context; the agent still waits/polls operation handles for completion.

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
import logging
import runpy
from pathlib import Path
from tempfile import gettempdir
from typing import Any

logger = logging.getLogger(__name__)

# Standalone bootstrap: insert repo ``lib`` into sys.path and fast-fail if GUI
# extras are missing before importing the heavy GUI method-spec modules.
_BOOTSTRAP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_standalone.py"))
_BOOTSTRAP["bootstrap_standalone_server"](
    __file__,
    required_modules=(
        (
            "qtpy",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
        (
            "PyQt6",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

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
    resolve_connect_port,
    run_stdio_loop,
)
from zcu_tools.mcp.measure.assembly import build_measure_tools  # noqa: E402
from zcu_tools.mcp.measure.session import MeasureMcpSession  # noqa: E402
from zcu_tools.mcp.measure.tool_context import MeasureToolContext  # noqa: E402

# ``MCP_VERSION`` is this MCP bridge's own code revision (the mcp_server / tool
# layer, NOT the wire contract). It is REPORTED — never compared — in the version
# banner so an agent can confirm a reconnect picked up the bridge-side edits. Bump
# it on a meaningful bridge change you want to be able to spot a reload of,
# including pure mcp-side convenience changes (new override tools, fold tweaks,
# tool renames) that leave the wire contract untouched. A wire-contract change is
# tracked separately by WIRE_VERSION (see ``wire_version.py``); the two are
# independent. (Git history holds the per-version evolution.)
# MeasureMcpSession owns measure-only MCP policy state.
MCP_VERSION = 74

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
  - The recommended bundle flow (breadcrumb open -> run -> analyze_review -> writeback apply -> save):
    gui_tab_open (new tab + adapter guide) -> gui_tab_run (configure + run) ->
    gui_tab_analyze_review (analyze + writeback preview) -> gui_tab_writeback_apply
    (apply pane draft with destination_context) -> gui_tab_save_data / gui_tab_save_image when wanted. Each folds the cross-tool reads you would
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
  - Run / load / analyze: gui_tab_run_start, gui_tab_load_data,
    gui_tab_analyze_start, gui_tab_post_analyze_start. gui_tab_load_data is
    synchronous; run/analyze/post-analyze each waits briefly then degrades to a
    handle. A FINISHED run/analyze reply (settled in the short wait) already
    carries 'figure' — the plot rendered to a temp PNG via explicit pane (run pane live, analysis/post canonical). After a
    pending->finished op, read the figure with
    gui_tab_get_figure(tab_id, subtab_id) (run|analysis|post_analysis) and the fit summary with gui_tab_get_analyze_result /
    gui_tab_get_post_analyze_result (the generic wait/poll report only status).
    gui_tab_save_data persists data (tab-only); gui_tab_save_image(subtab_id) persists the pane's canonical image (analysis|post_analysis).
    gui_tab_writeback_apply(subtab_id) commits the pane's draft with destination_context.
  - Async handles: every degrading op returns a 'handle' in its START reply; drive
    it with the generic gui_op_poll(handle) / gui_op_wait(handle).
  - Devices / context / predictor / adapters: gui_device_*, gui_context_*,
    gui_predictor_*, gui_adapter_*.
  - Arbitrary waveforms: list_arb_waveform, get_arb_waveform_preview,
    set_arb_waveform. These manage qubit-scoped .npz assets and render preview PNGs.
ON-DEMAND — screenshot (a window/dialog grab; useful to show the user what the GUI
looks like, not part of the measurement loop):
  - gui_screenshot(target): 'window' grabs the whole main window, a dialog name
    (setup/device/predictor/inspect/arb_waveform/startup) grabs that dialog; always
    writes a PNG file (never inline base64).
DEV — debugging the GUI/MCP itself (the version table + in-flight handles are
normally hidden from the operator; do NOT use these for measurement):
  - gui_debug_resource_versions (the per-resource optimistic-concurrency version
    table, for debugging stale-guard rejections; wire/gui/mcp *code* versions are
    in the gui_launch / gui_bridge_connect 'note' field, not here),
    gui_debug_operations (the in-flight operation handles, semantic key -> id).

Startup precondition: gui_overview's 'state' field must report all four flags true
before running experiments. Run/save require an active file-backed context; load
requires an existing experiment context but no SoC; save/analyze require an
existing run result. A precondition violation returns precondition_failed;
editing cfg while a tab is running likewise.

gui_soc_connect is SYNCHRONOUS — NOT part of the async-handle family: it blocks
until the SoC is connected and returns {status:'finished', soc:{...}} in one call
(no handle). A remote board that is unreachable fails fast (~1s).

The async ops (run / analyze / post_analyze / device) share ONE contract — the
per-tool descriptions only name what each waits on; the mechanics live here.
Completion is detected by wait/poll on a handle, not by best-effort events:
  - A short-wait START (gui_tab_run_start, gui_tab_analyze_start,
    gui_tab_post_analyze_start, gui_device_*) waits up to wait_seconds (default
    1.0): settles in time -> {status:'finished', handle, <product>}
    (gui_tab_run_start -> {tab, figure}; gui_tab_analyze_start -> {summary, figure};
    gui_tab_post_analyze_start -> {summary, figure}; gui_device_* -> {snapshot}); a slow one
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
    (gui_tab_get_figure, gui_tab_get_analyze_result,
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

Push context: every successful tool reply piggybacks diagnostics and subscribed
low-frequency GUI events accumulated since the prior successful reply. Event
envelopes retain payload, process sequence, and origin attribution. This stream is
bounded and best-effort across disconnects; operation wait/poll and fresh snapshots
remain the completion/state authorities.

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
    gui_tab_close, gui_tab_save_data, gui_tab_save_image, gui_device_connect / _disconnect / _apply,
    gui_context_md_write / _md_delete / _ml_delete_* / _ml_rename_*,
    gui_editor_save, set_arb_waveform.

Agent-to-user prompting: gui_prompt_user(message, timeout=600) opens a prompt
dialog for the user and BLOCKS your entire turn until the user replies, dismisses,
or the dialog times out. Returns {reason:'reply'|'dismiss'|'timeout', reply?}:
  - 'reply': user answered; read the reply string and act on it.
  - 'dismiss': user explicitly closed the prompt — do NOT ask again immediately.
  - 'timeout': no one was watching the GUI — do NOT wait again, continue or poll.
"""

# ---------------------------------------------------------------------------
# App session + transport bridge
# ---------------------------------------------------------------------------

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


def main() -> None:
    session = MeasureMcpSession(
        _CONFIG,
        resolve_connect_port=resolve_connect_port,
        port_is_open=_port_is_open,
    )
    bridge = McpBridge(_CONFIG, on_event=session.deliver_event)
    session.attach_bridge(bridge)
    context = MeasureToolContext(
        config=_CONFIG,
        session=session,
        method_specs=METHOD_SPECS,
        resolve_connect_port=resolve_connect_port,
    )
    tools = build_measure_tools(context)

    def cleanup_on_exit() -> None:
        # An attach-only server must not stop somebody else's GUI.
        if not bridge.launched_gui:
            return
        try:
            tools["gui_stop"]["handler"]({"timeout_kill": True})
        except Exception:
            logger.debug("gui_stop on exit failed", exc_info=True)

    def piggyback_blocks() -> list[dict[str, Any]]:
        """Drain diagnostics and events on each successful tool reply."""
        pending = session.drain_pending()
        diagnostics = pending["diagnostics"]
        events = pending["events"]
        blocks: list[dict[str, Any]] = []
        if diagnostics:
            lines = "\n".join(_format_diagnostic(m) for m in diagnostics)
            blocks.append(
                {"type": "text", "text": "notifications since last call:\n" + lines}
            )
        if events:
            blocks.append(
                {
                    "type": "text",
                    "text": "events since last call:\n"
                    + json.dumps(events, separators=(",", ":")),
                }
            )
        return blocks

    run_stdio_loop(
        _CONFIG,
        tools,
        on_cleanup=cleanup_on_exit,
        on_each_reply=piggyback_blocks,
        on_start=_setup_logging,
        on_error=logger.exception,
        server_version="1.1.0",
    )


if __name__ == "__main__":
    main()
