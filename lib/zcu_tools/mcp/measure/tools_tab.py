"""Measure MCP tools-tab override tools — subtab-qualified (ticket 04)."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    _fold_finished_figure,
    _render_tab_figure,
    _start_op_with_short_wait,
    bind_context,
    send_gui_rpc,
)
from zcu_tools.mcp.measure.tools_cfg import _fold_tab_editing_context


def _run_tab_summary(tab_id: str) -> dict[str, Any]:
    """A run-finished tab summary: only {tab_id, interaction}. The full
    tab.snapshot also carries adapter_name / editor_id / save_paths, none of
    which change across a run — re-sending them every run is wasted tokens
    (the agent already has them from gui_tab_snapshot). To see the plot, call
    gui_tab_get_figure(tab_id, subtab_id='run')."""
    snap = send_gui_rpc("tab.snapshot", {"tab_id": tab_id})["tabs"][0]
    interaction = snap.get("interaction", {}) if isinstance(snap, dict) else {}
    return {"tab_id": tab_id, "interaction": interaction}


def tool_gui_tab_run_start(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start a run, waiting briefly for a fast (small reps/rounds) run to finish.

    A run has both modes — a tiny sweep finishes in well under a second, a big
    one takes minutes — so it degrades like device ops: settles in time ->
    {status:'finished', tab:<tab.snapshot>, figure:<path>} (has_run_result set;
    figure is the run pane's live plot rendered to a temp PNG, the op's OWN visual result);
    still running -> {status:'pending', handle} (no figure yet; poll/wait the
    handle with gui_op_poll(handle=<handle>) / gui_op_wait(handle=<handle>)). The
    reply always carries 'handle'; send_gui_rpc attaches the version guard.
    NOTE: a generic gui_op_wait/poll only reports status — to see the plot after a
    pending->finished run, call gui_tab_get_figure(tab_id, subtab_id='run').
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
        "see the plot when finished with gui_tab_get_figure"
        f"(tab_id={tab_id!r}, subtab_id='run').",
    )
    # The figure is this op's OWN visual result — fold it on FINISHED only
    # (a pending run has no settled plot yet). Failure is swallowed so a
    # plotting hiccup never masks an otherwise-good run reply.
    return _fold_finished_figure(tab_id, reply, subtab_id="run")


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
    """Fold the analyze (or post-analyze) summary into a finished short-wait reply."""
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
    a temp PNG via analysis pane (analyze's OWN visual result). Review the proposed writeback with
    gui_tab_writeback_list(subtab_id='analysis') (not folded here; that fold lives in gui_tab_analyze_review).
    The reply always carries 'handle'; 'summary'/'figure' appear only on a finished
    FIT. After a pending->finished analyze read gui_tab_get_analyze_result and the
    plot with gui_tab_get_figure(subtab_id='analysis').
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
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
    return _fold_finished_figure(tab_id, reply, subtab_id="analysis")


def tool_gui_tab_post_analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    """Start the second-layer (post) analysis, waiting briefly (degrades like a run).

    Post-analysis runs on top of the tab's PRIMARY analyze result (e.g.
    single-shot multi-backend ge discrimination) and is FIT-only — it computes on
    a worker, so it usually settles in the short wait -> {status:'finished',
    handle, summary:{...}, figure:<path>} (the fit summary is folded in, same shape as
    gui_tab_get_post_analyze_result, plus the post pane's figure). A
    slow one degrades to {status:'pending', handle} (poll/wait the handle with
    gui_op_poll(handle=<handle>) / gui_op_wait(handle=<handle>)). Fast-fails with
    precondition_failed when the tab has no primary analyze result yet — run
    gui_tab_analyze_start first. There is NO cancel for post-analysis: it is a pure
    CPU recompute with no stop point. 'updates' optionally overrides post params
    (see gui_tab_get_post_analyze_params). The reply always carries 'handle';
    'summary'/'figure' appear only on finished. After a pending->finished post-analysis
    read gui_tab_get_post_analyze_result (a generic gui_op_wait/poll only reports
    status).
    """
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    params: dict[str, Any] = {"tab_id": tab_id}
    if "updates" in arguments and arguments["updates"] is not None:
        params["updates"] = arguments["updates"]
    send_gui_rpc("tab.post_analyze", params)
    reply = _start_op_with_short_wait(
        f"post_analyze:{tab_id}",
        f"Post-analysis on tab {tab_id!r}",
        wait_seconds,
        lambda: _analyze_summary_product("tab.get_post_analyze_result", tab_id),
        "poll/wait the returned handle with gui_op_poll / gui_op_wait, then read "
        f"gui_tab_get_post_analyze_result(tab_id={tab_id!r}).",
    )
    # Fold post figure on finished (pane-specific)
    return _fold_finished_figure(tab_id, reply, subtab_id="post_analysis")


def _fold_writeback_preview(
    tab_id: str, reply: dict[str, Any], *, subtab_id: str
) -> dict[str, Any]:
    """Fold the pane's writeback preview into a FINISHED analyze reply, in place."""
    if reply.get("status") != "finished":
        return reply
    try:
        reply["writeback_preview"] = send_gui_rpc(
            "tab.writeback_preview", {"tab_id": tab_id, "subtab_id": subtab_id}
        )
    except Exception:
        pass
    return reply


def tool_gui_tab_open(arguments: dict[str, Any]) -> dict[str, Any]:
    """open (step 1): create a tab for ``adapter_name`` and fold its editing
    context + the adapter guide into one reply."""
    adapter_name = str(arguments["adapter_name"])
    skip_guide = bool(arguments.get("skip_guide", False))
    tab_id = str(send_gui_rpc("tab.new", {"adapter_name": adapter_name})["tab_id"])
    reply: dict[str, Any] = {"tab_id": tab_id, "adapter": adapter_name}
    _fold_tab_editing_context(tab_id, reply)
    if not skip_guide:
        reply["guide"] = send_gui_rpc(
            "adapter.guide", {"adapter_name": adapter_name}
        ).get("guide")
    else:
        reply["guide_omitted"] = True
    return reply


def tool_gui_tab_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """run (step 2): apply ``edits`` then run the existing ``tab_id``, STOPPING
    before analyze."""
    tab_id = str(arguments["tab_id"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    edits = arguments.get("edits") or []
    if not isinstance(edits, list):
        raise ValueError("'edits' must be an ordered list of {path, value} objects")
    if edits:
        send_gui_rpc(
            "tab.set_cfg",
            {
                "tab_id": tab_id,
                "edits": [{"path": str(e["path"]), "value": e["value"]} for e in edits],
            },
        )
    reply = tool_gui_tab_run_start({"tab_id": tab_id, "wait_seconds": wait_seconds})
    if reply.get("status") != "finished":
        reply["owed"] = (
            "figure (gui_tab_get_figure subtab_id='run' after the handle finishes)"
        )
        return reply
    return _fold_analyze_params(tab_id, reply)


def tool_gui_tab_analyze_review(arguments: dict[str, Any]) -> dict[str, Any]:
    """analyze_review (step 3): analyze ``tab_id`` and fold the fit review into
    one reply (pane-specific analysis)."""
    tab_id = str(arguments["tab_id"])
    analyze_args: dict[str, Any] = {
        "tab_id": tab_id,
        "wait_seconds": float(arguments.get("wait_seconds", 1.0)),
    }
    if arguments.get("updates") is not None:
        analyze_args["updates"] = arguments["updates"]
    reply = tool_gui_tab_analyze(analyze_args)
    if reply.get("status") != "finished":
        reply["owed"] = (
            "summary (gui_tab_get_analyze_result), figure "
            "(gui_tab_get_figure subtab_id='analysis'), writeback_preview "
            "(gui_tab_writeback_list subtab_id='analysis') after the handle finishes"
        )
        return reply
    return _fold_writeback_preview(tab_id, reply, subtab_id="analysis")


def tool_gui_tab_get_figure(arguments: dict[str, Any]) -> dict[str, Any]:
    """Render a pane's figure to a PNG FILE and return its path.

    Requires (tab_id, subtab_id) with closed values run|analysis|post_analysis.
    The convenience layer always drives the wire in out_path mode so the agent
    never receives inline base64. When out_path is omitted we synthesise a
    per-pane temp path under gettempdir(), overwriting the previous render of the
    same tab+subtab.
    """
    tab_id = str(arguments["tab_id"])
    subtab_id = str(arguments["subtab_id"])
    if subtab_id not in ("run", "analysis", "post_analysis"):
        raise ValueError(
            f"subtab_id must be one of ['run','analysis','post_analysis'], got {subtab_id!r}"
        )
    out_path_arg = arguments.get("out_path")
    return _render_tab_figure(
        tab_id, subtab_id, str(out_path_arg) if out_path_arg is not None else None
    )


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_tab_run_start": {
        "handler": tool_gui_tab_run_start,
        "description": (
            "Start a run on tab_id (shared short-wait START contract — see server "
            "instructions). A fast run settles -> {status:'finished', handle, "
            "tab:{...}, figure:<path>} — the tab snapshot (has_run_result set) AND "
            "the run pane's live plot rendered to a temp PNG (the run's OWN visual result). A "
            "slow run degrades to {status:'pending', handle} (no figure yet; "
            "poll/wait the handle with gui_op_poll / gui_op_wait, then read the plot "
            "with gui_tab_get_figure subtab_id='run'). The reply always carries 'handle'."
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
    "gui_tab_open": {
        "handler": tool_gui_tab_open,
        "description": (
            "Step 1 of the recommended flow (open -> run -> analyze_review -> "
            "writeback_apply -> save) — open. = tab.new + tab.snapshot + tab.get_cfg + adapter.guide. "
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
            "Step 2 of the recommended flow (open -> run -> analyze_review -> "
            "writeback_apply -> save) — run. = gui_tab_set_cfg + gui_tab_run_start. Apply 'edits' "
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
            "gui_op_poll(handle), then read the plot with gui_tab_get_figure subtab_id='run'. "
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
            "Step 3 of the recommended flow (open -> run -> analyze_review -> "
            "writeback_apply -> save) — analyze_review. = gui_tab_analyze_start + "
            "gui_tab_writeback_list. Analyze 'tab_id' and fold the writeback "
            "review into ONE reply. A finished FIT returns {status:'finished', "
            "handle, summary, figure, writeback_preview} — 'summary' is the fit "
            "result (same shape as gui_tab_get_analyze_result), 'figure' comes from "
            "gui_tab_analyze_start's own FINISHED reply (the fit plot rendered to a "
            "temp PNG via analysis pane), and 'writeback_preview' is the stage-specific fold "
            "({has_draft, items, destination_context} — the proposed writeback values/targets + active destination) — so you "
            "review the fit + the proposed writeback in one call before "
            "gui_tab_writeback_apply. 'updates' optionally overrides the analyze params; "
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
    "gui_tab_analyze_start": {
        "handler": tool_gui_tab_analyze,
        "description": (
            "Start analyze on tab_id (shared short-wait START contract — see server "
            "instructions, incl. the INTERACTIVE-analyze note). A FIT settles -> "
            "{status:'finished', handle, summary, figure} — the fit summary (same "
            "shape as gui_tab_get_analyze_result, the *_err fields included) AND the "
            "fit plot rendered to a temp PNG via analysis pane. Review "
            "the proposed writeback with gui_tab_writeback_list subtab_id='analysis' (not folded here; "
            "that fold lives in gui_tab_analyze_review). An INTERACTIVE analysis (e.g. "
            "flux_dep) degrades to {status:'pending', handle, summary:None} — no "
            "figure (nothing settled yet); prompt the user to mark the plot + click "
            "Done, then poll/wait the handle with gui_op_poll / gui_op_wait, then "
            "read gui_tab_get_analyze_result + gui_tab_get_figure. The reply "
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
            "settles -> {status:'finished', handle, summary:{...}, figure:<path>} (folded in, same "
            "shape as gui_tab_get_post_analyze_result, plus post pane's figure). A slow one degrades to "
            "{status:'pending', handle}; poll/wait the handle with gui_op_poll / "
            "gui_op_wait, then read gui_tab_get_post_analyze_result. The reply always "
            "carries 'handle'. There is NO cancel for post-analysis: it is a pure "
            "CPU recompute with no stop point. Fast-fails with precondition_failed "
            "when the tab has no primary analyze result yet — run "
            "gui_tab_analyze_start first. 'updates' optionally overrides post params "
            "(see gui_tab_get_post_analyze_params). The post figure is pane-specific; see it with gui_tab_get_figure subtab_id='post_analysis' and persist it "
            "with gui_tab_save_image (subtab_id='post_analysis') or gui_tab_save_data."
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
    "gui_tab_get_figure": {
        "handler": tool_gui_tab_get_figure,
        "description": (
            "Pane-qualified figure retrieval (run|analysis|post_analysis). Renders the "
            "specified pane's figure to a PNG FILE — Run is the live FigureContainer (view-only, not canonical), analysis/post are canonical State figures. "
            "The PNG is rendered at a fixed small geometry (~640x480), independent of the GUI "
            "window size. The reply is {saved_to, bytes} — Read the saved_to path "
            "to view the plot (never inline base64). Omit out_path to write a "
            "per-tab+subtab file under the temp dir (overwritten each call); pass out_path "
            "to choose the location. Fails with PRECONDITION_FAILED if the pane has "
            "no figure yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "string"},
                "subtab_id": {
                    "type": "string",
                    "enum": ["run", "analysis", "post_analysis"],
                    "description": "Pane: run|analysis|post_analysis",
                },
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional absolute path to write the PNG; omit to use a "
                        "per-tab+subtab file under the temp dir"
                    ),
                },
            },
            "required": ["tab_id", "subtab_id"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
