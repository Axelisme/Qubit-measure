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
from collections.abc import Mapping, MutableMapping
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
    generated_rpc_timeout_seconds,
    resolve_connect_port,
    run_stdio_loop,
)
from zcu_tools.mcp.core.call_log import wrap_handler  # noqa: E402
from zcu_tools.mcp.measure import (  # noqa: E402
    tool_context,
    tools_cfg,
    tools_context,
    tools_debug,
    tools_device,
    tools_lifecycle,
    tools_notify,
    tools_operation,
    tools_overview,
    tools_screenshot,
    tools_soc,
    tools_tab,
)
from zcu_tools.mcp.measure.session import (  # noqa: E402
    GuiRpcError,
    MeasureMcpSession,
)
from zcu_tools.mcp.measure.session_policy import (  # noqa: E402
    describe_stale_keys,
    expand_pattern_keys,
)
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
MCP_VERSION = 70

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
  - The recommended bundle flow (breadcrumb open -> run -> analyze_review -> commit):
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
  - Run / load / analyze: gui_tab_run_start, gui_tab_load_data,
    gui_tab_analyze_start, gui_tab_post_analyze_start. gui_tab_load_data is
    synchronous; run/analyze/post-analyze each waits briefly then degrades to a
    handle. A FINISHED run/analyze reply (settled in the short wait) already
    carries 'figure' — the plot rendered to a temp PNG. After a
    pending->finished op, read the figure with
    gui_tab_get_current_figure and the fit summary with gui_tab_get_analyze_result /
    gui_tab_get_post_analyze_result (the generic wait/poll report only status).
    gui_tab_save (artifact + figure selectors) persists data and/or the figure and
    returns the resolved destinations; gui_tab_writeback_apply commits the draft.
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


def _session_resolve_connect_port(
    config: MCPBridgeConfig, requested: int | None
) -> int:
    return resolve_connect_port(config, requested)


def _session_port_is_open(port: int) -> bool:
    return _port_is_open(port)


_SESSION = MeasureMcpSession(
    _CONFIG,
    resolve_connect_port=_session_resolve_connect_port,
    port_is_open=_session_port_is_open,
)
_BRIDGE = McpBridge(_CONFIG, on_event=_SESSION.deliver_event)
_SESSION.attach_bridge(_BRIDGE)

# Compatibility aliases for internal tests and debugging helpers.  Ownership is
# still in MeasureMcpSession; these names reference its live mutable maps.
_LAST_SEEN: MutableMapping[str, int] = _SESSION.last_seen_versions
_GUARD_DEPS: Mapping[str, tuple[str, ...]] = _SESSION.policy.guard_deps
_READ_REVEALS: Mapping[str, tuple[str, ...]] = _SESSION.policy.read_reveals
_OP_BY_KEY: MutableMapping[str, int] = _SESSION.operation_handles


def _deliver_event(msg: dict[str, Any]) -> None:
    _SESSION.deliver_event(msg)


def _drain_pending() -> dict[str, list[dict[str, Any]]]:
    return _SESSION.drain_pending()


def _read_version_table() -> dict[str, int] | None:
    return _SESSION.read_version_table()


def _refresh_versions() -> None:
    _SESSION.refresh_versions()


def _expand_pattern_keys(
    patterns: tuple[str, ...], params: dict[str, Any], source_table: dict[str, int]
) -> dict[str, int]:
    return expand_pattern_keys(patterns, params, source_table)


def _build_expected_versions(method: str, params: dict[str, Any]) -> dict[str, int]:
    return _SESSION.build_expected_versions(method, params)


def _refresh_revealed_versions(method: str, params: dict[str, Any]) -> None:
    _SESSION.refresh_revealed_versions(method, params)


def _describe_stale_keys(keys: list) -> list[str]:
    return describe_stale_keys(keys)


def _ensure_connected() -> None:
    _SESSION.ensure_connected()


_EXPLICIT_TIMEOUT_METHODS = frozenset({"operation.await", "notify.await"})


def _default_rpc_timeout_seconds(method: str) -> float:
    if method in _EXPLICIT_TIMEOUT_METHODS:
        raise ValueError(f"{method!r} requires explicit timeout_seconds")
    return generated_rpc_timeout_seconds(METHOD_SPECS[method])


def send_gui_rpc(
    method: str,
    params: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    timeout = (
        _default_rpc_timeout_seconds(method)
        if timeout_seconds is None
        else float(timeout_seconds)
    )
    return _SESSION.send_gui_rpc(method, params, timeout)


# ---------------------------------------------------------------------------
# Tool context + compatibility exports
# ---------------------------------------------------------------------------


def _send_gui_rpc_late(
    method: str,
    params: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    # Override handlers intentionally resolve through the current server-level
    # symbol so tests and debuggers can monkeypatch ``server.send_gui_rpc``.
    if timeout_seconds is None:
        return send_gui_rpc(method, params)
    return send_gui_rpc(method, params, timeout_seconds)


def _overview_late() -> dict[str, Any]:
    # Lifecycle tools call this provider so monkeypatching server._assemble_overview
    # still affects gui_bridge_connect / gui_launch after the handler move.
    return _assemble_overview()


def _resolve_connect_port_late(config: MCPBridgeConfig, requested: int | None) -> int:
    # Preserve the old module-global lookup path for tests/debuggers that patch
    # server.resolve_connect_port.
    return resolve_connect_port(config, requested)


_TOOL_CTX = MeasureToolContext(
    config=_CONFIG,
    session=_SESSION,
    bridge=_BRIDGE,
    method_specs=METHOD_SPECS,
    send_gui_rpc=_send_gui_rpc_late,
    overview=_overview_late,
    resolve_connect_port=_resolve_connect_port_late,
)
tool_context.bind_context(_TOOL_CTX)

# Shared helper compatibility aliases. Ownership is in tool_context / domain
# modules; server.py keeps these names as the stable import facade.
_WAIT_TRANSPORT_SLACK_SECONDS = tool_context._WAIT_TRANSPORT_SLACK_SECONDS
_coerce_pairs = tool_context._coerce_pairs
_is_timeout_error = tool_context._is_timeout_error
_start_op_with_short_wait = tool_context._start_op_with_short_wait
_render_tab_figure = tool_context._render_tab_figure
_fold_finished_figure = tool_context._fold_finished_figure

# Domain helper / handler compatibility aliases.
tool_gui_connect = tools_lifecycle.tool_gui_connect
tool_gui_disconnect = tools_lifecycle.tool_gui_disconnect
tool_gui_launch = tools_lifecycle.tool_gui_launch
tool_gui_stop = tools_lifecycle.tool_gui_stop

_assemble_overview = tools_overview._assemble_overview
tool_gui_overview = tools_overview.tool_gui_overview

_resolve_editor_id = tools_cfg._resolve_editor_id
_fold_tab_editing_context = tools_cfg._fold_tab_editing_context
tool_gui_editor_open = tools_cfg.tool_gui_editor_open
tool_gui_editor_get_cfg = tools_cfg.tool_gui_editor_get_cfg
tool_gui_editor_set = tools_cfg.tool_gui_editor_set
tool_gui_tab_set_cfg = tools_cfg.tool_gui_tab_set_cfg

tool_gui_context_md_write = tools_context.tool_gui_context_md_write
tool_gui_context_md_read = tools_context.tool_gui_context_md_read
tool_gui_context_md_delete = tools_context.tool_gui_context_md_delete
tool_gui_context_list = tools_context.tool_gui_context_list
tool_gui_context_ml_inspect = tools_context.tool_gui_context_ml_inspect

_await_operation_by_handle = tools_operation._await_operation_by_handle
_poll_operation_by_handle = tools_operation._poll_operation_by_handle
tool_gui_op_poll = tools_operation.tool_gui_op_poll
tool_gui_op_wait = tools_operation.tool_gui_op_wait

_device_snapshot = tools_device._device_snapshot
tool_gui_device_connect = tools_device.tool_gui_device_connect
tool_gui_device_disconnect = tools_device.tool_gui_device_disconnect
tool_gui_device_setup = tools_device.tool_gui_device_setup

_run_tab_summary = tools_tab._run_tab_summary
_fold_analyze_params = tools_tab._fold_analyze_params
_analyze_summary_product = tools_tab._analyze_summary_product
_SAVE_ARTIFACTS = tools_tab._SAVE_ARTIFACTS
_SAVE_FIGURES = tools_tab._SAVE_FIGURES
_fold_writeback_preview = tools_tab._fold_writeback_preview
tool_gui_tab_run_start = tools_tab.tool_gui_tab_run_start
tool_gui_tab_analyze = tools_tab.tool_gui_tab_analyze
tool_gui_tab_post_analyze = tools_tab.tool_gui_tab_post_analyze
tool_gui_tab_save = tools_tab.tool_gui_tab_save
tool_gui_tab_open = tools_tab.tool_gui_tab_open
tool_gui_tab_run = tools_tab.tool_gui_tab_run
tool_gui_tab_analyze_review = tools_tab.tool_gui_tab_analyze_review
tool_gui_tab_commit = tools_tab.tool_gui_tab_commit
tool_gui_tab_get_current_figure = tools_tab.tool_gui_tab_get_current_figure

_SOC_CONNECT_TIMEOUT_SLACK = tools_soc._SOC_CONNECT_TIMEOUT_SLACK
_SOC_CONNECT_RECONCILE_TIMEOUT = tools_soc._SOC_CONNECT_RECONCILE_TIMEOUT
_soc_connect_rpc_timeout = tools_soc._soc_connect_rpc_timeout
tool_gui_soc_connect = tools_soc.tool_gui_soc_connect

_SCREENSHOT_DIALOGS = tools_screenshot._SCREENSHOT_DIALOGS
_SCREENSHOT_TARGETS = tools_screenshot._SCREENSHOT_TARGETS
tool_gui_screenshot = tools_screenshot.tool_gui_screenshot

tool_gui_debug_resource_versions = tools_debug.tool_gui_debug_resource_versions
tool_gui_debug_operations = tools_debug.tool_gui_debug_operations

_NOTIFY_CONSUMER_SLACK = tools_notify._NOTIFY_CONSUMER_SLACK
tool_gui_prompt_user = tools_notify.tool_gui_prompt_user


def _fold_tab_editing_context_late(
    tab_id: str, reply: dict[str, Any]
) -> dict[str, Any]:
    # gui_tab_open historically called the server-level symbol, and tests patch
    # that symbol directly. Keep the moved tab handler on the same late-bound path.
    return _fold_tab_editing_context(tab_id, reply)


tools_tab._fold_tab_editing_context = _fold_tab_editing_context_late


# ---------------------------------------------------------------------------
# Generated tools — derived from dispatch.METHOD_REGISTRY (the wire SSOT)
# ---------------------------------------------------------------------------

_TOOL_MODULES = (
    tools_lifecycle,
    tools_overview,
    tools_cfg,
    tools_context,
    tools_operation,
    tools_device,
    tools_tab,
    tools_soc,
    tools_screenshot,
    tools_debug,
    tools_notify,
)


def _merge_override_tools() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for module in _TOOL_MODULES:
        for name, entry in module.build_override_tools(_TOOL_CTX).items():
            if name in merged:
                raise RuntimeError(f"duplicate MCP override tool {name!r}")
            merged[name] = entry
    return merged


_NON_GENERATED_METHODS = frozenset().union(
    *(module.NON_GENERATED_METHODS for module in _TOOL_MODULES)
)
_OVERRIDE_TOOLS: dict[str, dict[str, Any]] = _merge_override_tools()
_OVERRIDE_NAMES = frozenset(_OVERRIDE_TOOLS)


# Generated tools (schema from the ParamSpec SSOT, forwarding through the guarded
# send_gui_rpc) overlaid with the hand-written override subset (lifecycle /
# fan-out / file-write / coercion). assemble_tools fails fast on a name collision.
TOOLS: dict[str, dict[str, Any]] = assemble_tools(
    generate_tools(_CONFIG, METHOD_SPECS, _NON_GENERATED_METHODS, send_gui_rpc),
    _OVERRIDE_TOOLS,
    _OVERRIDE_NAMES,
)

# Wrap every top-level handler for call logging. Transparent
# side-effect only: same args in, same result out (or re-raised exception);
# only writes one JSONL line per invocation.  Applied after assemble_tools so
# generated + override + bundle tools are all covered in a single pass.
for _tool_name, _tool_entry in TOOLS.items():
    _tool_entry["handler"] = wrap_handler(_tool_name, _tool_entry["handler"])


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
