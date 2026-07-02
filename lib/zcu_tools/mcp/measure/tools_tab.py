"""Measure MCP tools-tab override tools."""

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


def _fold_writeback_preview(tab_id: str, reply: dict[str, Any]) -> dict[str, Any]:
    """Fold the tab's writeback preview into a FINISHED analyze reply, in place.

    A FIT analyze recomputes the persistent writeback draft (the proposed md/ml/wf
    values + apply targets); surfacing it next to the fit summary lets the agent
    review the fit AND the proposed writeback in one call before
    gui_tab_writeback_apply (step 3). Only acts on ``reply['status'] ==
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


def tool_gui_tab_open(arguments: dict[str, Any]) -> dict[str, Any]:
    """open (step 1): create a tab for ``adapter_name`` and fold its editing
    context + the adapter guide into one reply.

    Composes tab.new with the fan-out reads the agent always makes before editing
    cfg (tab.snapshot for editor_id, tab.get_cfg for the settable cfg tree)
    plus the adapter's orientation guide (adapter.guide).

    The guide is included BY DEFAULT so that any fresh agent context, sub-agent,
    or context-reset session receives the orientation text it needs without having
    to remember to pass a flag. The server does not track duplicate guide delivery;
    that decision belongs to the caller (the agent), which is the only one who knows
    whether its context already contains the guide.

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
    """run (step 2): apply ``edits`` then run the existing ``tab_id``, STOPPING
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
    """analyze_review (step 3): analyze ``tab_id`` and fold the fit review into
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
    """commit (step 4): apply the tab's writeback draft, optionally saving.

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


NON_GENERATED_METHODS = frozenset(
    {
        "tab.get_current_figure",
        "tab.run_start",
        "tab.analyze",
        "tab.post_analyze",
        "tab.save_data",
        "tab.save_image",
        "tab.save_post_image",
        "tab.save_result",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
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
    "gui_tab_open": {
        "handler": tool_gui_tab_open,
        "description": (
            "Step 1 of the recommended flow (open -> run -> analyze_review -> "
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
            "Step 2 of the recommended flow (open -> run -> analyze_review -> "
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
            "Step 3 of the recommended flow (open -> run -> analyze_review -> "
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
            "Step 4 of the recommended flow (open -> run -> analyze_review -> "
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
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
