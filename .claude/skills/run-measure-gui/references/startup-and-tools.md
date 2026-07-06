# Startup and MCP Tool Flow

Read this when launching, attaching to, configuring, or running `measure-gui` through MCP.

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.13.
- An X display. On a headless box: `sudo apt-get install -y xvfb` and prefix
  GUI launches with `xvfb-run -a`. On a desktop session `DISPLAY` is already
  set and no xvfb is needed.
- No hardware: the mock SoC + `FakeDevice` cover the whole flow offline.

## Run (agent path — MCP tools)

This is the path you use when asked to "drive measure-gui". The MCP server
auto-launches the GUI; you call tools.

### Overview and project-identity tools

```
gui_overview         # standalone "current state" summary — the SINGLE orientation read.
                     # Returns:
                     #   {state:{has_project,has_context,has_active_context,has_soc},
                     #    project:{chip_name,qub_name,res_name,result_dir,database_path} or null,
                     #    context, soc, tabs:[{tab_id,adapter,is_running}],
                     #    running_tab, active_tab}
                     # The 'state' field carries the four readiness flags (no separate
                     # state-check tool); 'project' carries the project identity + paths
                     # (no separate project-info tool). Auto-connects if not already
                     # connected (pure read, no side effects).
                     # active_tab is the USER's current focus (collaboration only —
                     # not an operation target; agent ops always take an explicit tab_id).
                     # FIRST CALL when attaching to a running session: call gui_overview
                     # to load current state before assuming anything — the user may be
                     # actively operating the same GUI.
                     # gui_bridge_connect also folds the same overview into its reply ({note, overview}).

gui_tab_set_active(tab_id)   # push a tab to the user's foreground (user↔agent collaboration).
                             # Agent operations all take an explicit tab_id and are independent
                             # of which tab is active.  Use gui_tab_set_active only to guide the
                             # user's attention — e.g. ask them to drag flux lines on a specific
                             # tab — or to read back which tab they are currently looking at via
                             # gui_overview's active_tab field.
```

### Startup (fresh session)

```
gui_launch                                      # spawns the GUI, connects; banner shows the
                                                # handshake: "wire vN (mcp==gui); gui code vX, mcp code vY"
# Same startup path a GUI user takes (no mock shortcut):
gui_soc_connect(kind="mock")                    # mock SoC (or kind="remote", ip, port). SYNCHRONOUS:
                                                # returns the SoC summary directly, no wait/poll; a
                                                # remote connect fail-fasts at ~1s if unreachable.
gui_project_apply(chip_name="Q1_Chip",          # apply the project; omit result_dir/database_path
                  qub_name="Q1", res_name="R1")  # to scope them under chip/qub (notebook layout)
gui_context_create(bind_device="fake_flux")     # create a context bound to a flux device (reads its
                                                # current value/unit; FakeDevice->none, YOKOGS200->A).
                                                # Mock sessions use "fake_flux" (FAKE_FLUX_DEVICE_NAME);
                                                # real hardware: use the device name you connected with.
                                                # Omit bind_device for an unbound context; clone_from=<label>
                                                # clones an existing one. Or gui_context_switch(label) for an existing one.
gui_overview                                    # global readiness: gui_overview.state has the four flags
                                                # (has_project/has_context/has_active_context/has_soc) — all must
                                                # be true before running; use gui_tab_snapshot for per-tab
                                                # progress (is_running/is_analyzing/has_run_result/…)
gui_soc_info                                    # the board: per-channel type, sample rate, port, max length.
                                                # cfg is opt-in: pass include_cfg=true for the full structured cfg.
```

Then the experiment loop (per tab):

```
gui_adapter_list                                  # available experiments
gui_adapter_guide(adapter_name="onetone/flux_dep")# READ FIRST: per-experiment behavior, expected md/ml, recommended
                                                  # ranges + gotchas — live here, not in this skill. (gui_tab_open
                                                  # folds this guide in, so the recommended flow rarely calls it directly.)
gui_tab_new(adapter_name="fake/freq")             # PURE: just creates a tab -> {tab_id}. (gui_tab_open creates + folds
                                                  # editor_id/tree/guide in one call.) id e.g. fake-freq-1a2b3c4d
gui_tab_snapshot(tab_id) -> editor_id             # per-tab progress + the cfg-editing session handle
gui_tab_get_cfg(tab_id)                           # returns a nested value tree (not a flat list). Leaf conventions:
                                                  #   scalar leaf    → bare value (null = unset)
                                                  #   enum leaf      → {"$value": v, "$choices": [...]}
                                                  #   sweep node     → {"start":..., "stop":..., "expts":..., "step":...}
                                                  #                    'step' is READ-ONLY (derived from start/stop/expts);
                                                  #                    set start/stop/expts — never set step directly.
                                                  #   ref node       → {"$ref":{"current":"key","options":[...]}, ...current-variant subtree...}
                                                  #                    options are bare names; a ref edit accepts a bare name
                                                  # $-prefixed keys mark leaf metadata; plain keys are subtree nodes.
                                                  # pass prefix="modules.readout" to return only that subtree (no match → {});
                                                  # pointing prefix at a sweep edge returns the whole sweep node.
                                                  # EDIT via dotted path (gui_tab_set_cfg) — path syntax unchanged.
gui_tab_set_cfg(tab_id, edits=[{path,value},…])   # batch-set tab cfg fields in order (non-atomic; first failure RAISES,
                                                  # edits before it stay applied). Apply a ref-switch edit BEFORE the inner
                                                  # paths it unlocks. Sweep edge fields (start/stop/expts) only accept
                                                  # plain numbers; eval/ref expressions are not accepted there (use scalar
                                                  # leaf fields for eval). If an adapter pre-wires an eval edge, override it
                                                  # by passing a numeric value directly. The 'step' field is READ-ONLY
                                                  # (derived from start/stop/expts) — do not set it; set start/stop/expts.
                                                  # (gui_editor_set edits a NON-tab editor draft instead, addressed by editor_id.)
# RECOMMENDED FLOW = the 4 bundle tools (breadcrumb open -> run -> analyze_review -> commit;
#   each folds the NEXT decision's input, stops at a decision point):
#   gui_tab_open(adapter_name)                  -> {tab_id, editor_id, tree, guide}   # ① open: new+guide
#                                                  tree = nested value tree (same shape as gui_tab_get_cfg — see above).
#                                                  guide = adapter guide (full prose) ALWAYS included by default so any
#                                                  fresh context / sub-agent / context-reset gets it without a flag.
#                                                  Pass skip_guide=true only if your context already has the guide
#                                                  (e.g. you opened a tab for the same adapter earlier this session);
#                                                  reply carries guide_omitted: true to confirm the intentional omission.
#   gui_tab_run(tab_id, edits=[{path,value},…]) -> {..run.., figure, analyze_params}   # ② run: configure+run; STOPS before analyze
#                                                  edits is an ORDERED list (ref-switch before the paths it unlocks).
#   gui_tab_analyze_review(tab_id)              -> {summary, figure, writeback_preview}   # ③ analyze_review: analyze + preview writeback
#   gui_tab_commit(tab_id, save_data=False)     -> {status, applied_ids, saved[, save_error]}   # ④ commit: writeback (+ optional save)
# The base tools below = ON-DEMAND (fine-grained control). DEV tools (gui_debug_resource_versions/_operations —
#   debugging the GUI/MCP itself, NOT the measurement flow) are a separate tier in the server instructions:
gui_tab_run_start(tab_id)                         # waits ~1s; finished -> {status:finished, handle, figure:<png>,...}, slow -> {status:pending, handle}
gui_op_wait(handle)                               # block until the op (by handle) ends (only after pending; blocks your turn —
                                                  # for a long run background it, see "Detecting completion"). Generic: drives ANY handle.
gui_op_poll(handle)                               # non-blocking TRUE status of the op (by handle); NEVER raises. DRAINS all buffered
                                                  # user feedback -> feedback:[...] (every queued nudge, any status). Generic: drives ANY handle.
gui_tab_get_current_figure(tab_id)                # RARELY NEEDED (run/analyze finished already fold the figure, incl 2D
                                                  # scans via run; use only for a re-render / mid-flight plot / chosen out_path).
                                                  # Writes the CURRENT plot (run's 2D map, analysis fit, or post-analysis
                                                  # figure — whatever is on the tab's plot stack) to a PNG FILE and
                                                  # replies {saved_to, bytes}. THE ONLY way to look at any plot, including
                                                  # non-analysis 2D scans (flux_dep / power_dep): Read the saved_to path.
                                                  # Always a file (fixed 640×480), never inline base64. Omit out_path to
                                                  # write a per-tab temp file; pass out_path="<abs path>" to choose where.
                                                  # gui_screenshot(target, out_path?) [ON-DEMAND] same contract: always
                                                  # writes a file, replies {saved_to, bytes}, never base64. target="window"
                                                  # grabs the main window; target=<dialog name> (setup/device/predictor/inspect/startup)
                                                  # grabs that dialog if open.
gui_tab_analyze_start(tab_id)                     # a FIT settles -> {status:finished, handle, summary:{...}, figure:<png>} (its own
                                                  # fit result + plot; same summary as gui_tab_get_analyze_result); an
                                                  # INTERACTIVE pick (flux_dep) -> {status:pending, handle} → see below
gui_tab_post_analyze_start(tab_id)                # second analysis layer on top of the primary fit (e.g. single-shot ge);
                                                  # FIT-only, settles -> {status:finished, handle, summary:{...}} inline;
                                                  # slow -> {pending, handle} then gui_op_wait/gui_op_poll; needs primary analyze first
gui_tab_get_post_analyze_result(tab_id)           # re-fetch post-analysis summary (params: gui_tab_get_post_analyze_params)
gui_tab_save(tab_id)                              # save data and/or figure -> {data_path, image_path, data_async, image_error}.
                                                  # artifact='data'|'image'|'both' (default both); figure='primary'|'post'
                                                  # (default primary) selects which plot the 'image' targets.
gui_tab_writeback_apply(tab_id)                   # PURE: apply the writeback -> {applied_ids}. (gui_tab_commit also folds an optional save.)
```

Detecting completion — completion is observed synchronously or by waiting/polling
a handle, never by subscribing to a push stream:

| situation | what to do |
|---|---|
| fast run / fast fit | the call returns `{status:finished}` (`gui_tab_run_start` / `gui_tab_analyze_start` when it settles within `wait_seconds`, default 1.0) |
| a START returned `{status:pending, handle}` | wait (`gui_op_wait(handle)`, blocks) or poll (`gui_op_poll(handle)`, non-blocking) — the SAME generic drains for run / analyze / post-analyze / device handles |
| want live progress bars | already in the `gui_op_poll(handle)` reply while `status:running` (active + bars); no separate progress tool |
| buffered user feedback during a long run | `gui_op_poll(handle)` DRAINS every queued nudge and returns them as `feedback:[...]` (any status — even `running`); it reports the TRUE status, never a false `finished`. Draining consumes them: a later `gui_op_wait` won't re-deliver those nudges (the sticky terminal outcome is still re-readable) |
| a poll or wait says `status:cancelled` | a user/agent cancel (distinct from `failed`); **not a raise, not an error** — read the Stop reason (`gui_op_wait`: `feedback`; `gui_op_poll`: `stop_reason`, present when user clicked "Send & Stop"), then re-plan |
| after a `pending`->`finished` run/analyze | `gui_op_wait`/`gui_op_poll` report ONLY status — read the figure with `gui_tab_get_current_figure` and the fit summary with `gui_tab_get_analyze_result` (they are NOT auto-folded after a degrade) |
