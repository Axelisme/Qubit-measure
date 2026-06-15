"""Per-app wire / GUI code version constants for measure-gui.

These are the two hand-maintained version integers reported by the no-auth
``wire.version`` handshake. They are intentionally NOT in the shared
``zcu_tools.gui.remote.wire`` module — each GUI app evolves its own wire contract
independently, so the constants (and their per-app changelog) live beside the
app's own RemoteControlAdapter.
"""

from __future__ import annotations

# Two independent hand-maintained versions reported by the no-auth
# ``wire.version`` handshake (which also surfaces the MCP server's own
# MCP_VERSION). Only WIRE_VERSION is *compared*; the code revisions are
# *reported* so an agent can eyeball whether a reload took effect:
#
#   WIRE_VERSION — the mcp<->RPC *interface contract* (RPC method set, their
#     params, event/serialization shape). The MCP server pins it; a mismatch
#     means the two sides speak different protocols → hard MISMATCH. Bump ONLY
#     on a contract change.
#   GUI_VERSION  — this GUI code's *revision*. Reported, never compared (the
#     MCP server does not pin it — a GUI revision is the GUI process's own
#     property). Bump on any meaningful GUI change you want to be able to spot a
#     reload of, INCLUDING pure-internal logic changes that DON'T touch the wire
#     (the whole point of the split: an internal change bumps GUI_VERSION, not
#     WIRE_VERSION, so the contract version stays put).
#
# (Replaces the old single-version scheme that conflated "is the contract
# compatible" with "is this process running the latest code".)
# v2: added ml.list_roles / ml.create_from_role (role catalog) and
#     context.rename_ml_module / context.rename_ml_waveform; editor.open dropped
#     its discriminator param (from_name-only).
# v3: removed device.set_value (set values via device.setup updates={"value":..});
#     device.connect / device.disconnect now return operation_id (operation handle
#     parity with device.setup); device.snapshot includes the device info payload.
# v4: removed context.set_ml_module / context.set_ml_waveform (raw-dict RPC); ml
#     entries are built/edited via the editor session (create_from_role + editor.*)
#     — ADR-0006, the single ml/md write authority is ContextService.
# v5: added device.setup_spec (discover the fields settable via device.setup's
#     updates — name/type/choices/current/settable — from the live info model).
# v6 (ADR-0013): (a) removed cfg.set_field — a tab's cfg is edited through its
#     CfgEditorService session (editor.set_field on the tab's editor_id), the
#     same draft the form attaches to, so agent + user share one model (F11);
#     tab.list_paths now reads that session too (wire shape unchanged). (b) new
#     unsolicited ``diagnostic`` event push ({severity: error|info, title,
#     message}) — the Controller fans diagnostics to the adapter (a diagnostic
#     View) out-of-band of the event subscription set; agents receive it via the
#     normal events poll without subscribing.
# v7: run_lock_changed split into run_started{tab_id} + run_finished{tab_id,
#     outcome, error_message} — one event name per real transition instead of a
#     single event whose meaning depended on which fields were present.
# v8: cfg path grammar drops the ModuleRef 'value' wrapper segment
#     (modules.qub_pulse.value.waveform.value.length -> ...qub_pulse.waveform.length;
#     a stale 'value' path is rejected); editor.set_field result adds
#     'removed'/'added' (paths a ref switch dropped/created).
# v9: run.progress bars add raw 'n'/'total' (alongside scaled maximum/value) —
#     progress derivation moved to the main-thread ProgressBarModel (SSOT), so
#     format/percent/timing are computed live at read.
# v10: device setup ↔ run alignment. device_setup_changed split into
#      device_setup_started{name} + device_setup_finished{name, outcome,
#      error_message} (mirrors run_started/run_finished); new device.setup_progress
#      (same shape as run.progress); device.active_setup now only {device_name}
#      and device.active_operation drops progress (live progress via
#      device.setup_progress).
# v11: added soc.info — read the connected SoC's QICK soccfg (human-readable
#      description + structured cfg: DAC/ADC channels, sample rates, freq ranges).
# v12: added adapter.guide — read an adapter's human-facing orientation guide
#      (behavior / expects_md / expects_ml / typical_writeback / recommended)
#      before running it. New method = contract change.
# v13: added app.shutdown — gracefully close the GUI via its normal window-close
#      path (no OS kill); new method = contract change.
# v17: removed session.persist / session.restore — persistence is now lifecycle-
#      driven (flush at close, restore at startup); agent no longer triggers it.
# v18: save.both → save.result (rename) + leaner replies (context.new returns the
#      new label, save.* returns {ok}, connect folds only the soc description not
#      the structured cfg, run-finished tab is {tab_id, interaction} only).
# v19: context.new params (value, unit, clone_from_current) → (bind_device,
#      clone_from). The flux value/unit are no longer agent-supplied; bind_device
#      names a connected flux device whose current value/unit (whitelist:
#      FakeDevice->none, YOKOGS200->A) name the context; clone_from is an existing
#      context label to clone. Aligns the agent path with the GUI setup dialog.
# v20: progress query unified by operation_id — added operation.progress
#      (operation_id) covering run + device setup; removed run.progress and
#      device.setup_progress (owner-keyed). context.new precondition (no project)
#      → PRECONDITION_FAILED reason=no_project instead of leaking controller_error.
# v21: (a) tab.figure_screenshot renamed tab.get_current_figure — same handler
#      (the figure currently on the tab's plot stack: a run's 2D map, or an
#      analysis fit once analyzed), clearer agent-facing name. (b) save.data /
#      save.image / save.result now return the resolved written path(s)
#      ({data_path[, image_path]}, .hdf5 + uniqueness suffix already applied)
#      instead of {ok}/{} — the path is known synchronously at start, so the
#      agent need not recover it from a later diagnostic.
# v22: post-analysis (second analysis layer) RPCs — post_analyze.start
#      (tab_id + updates -> {operation_id}; FIT-only worker, gates server-side on
#      the primary analyze result), tab.get_post_analyze_params and
#      tab.get_post_analyze_result (mirror the analyze read pair). New methods =
#      contract change.
# v23: complex writeback scalars round-trip losslessly. A metadict writeback
#      proposed_value that is a Python complex is serialized on the wire as
#      {"__complex__": [re, im]} (in writeback.preview) and accepted back in the
#      same shape (writeback.set), replacing the old one-way {"__repr__"} stringify.
#      Serialization-shape change to the writeback surface = contract change.
# v24: figure/screenshot consolidation — looking at a plot is now ONLY
#      tab.get_current_figure (the tab's current plot-stack container, which also
#      holds the post-analysis figure). Removed view.screenshot (whole-window /
#      tab Qt grab). Added save.post_image (mirrors save.image, targets the
#      post-analysis figure). Method-set change = contract change.
# v25: device active-op enumeration for Phase C concurrency. Renamed the singular
#      device.active_setup / device.active_operation to plural
#      device.active_setups / device.active_operations, each now returning a
#      list of ALL in-flight ops (sorted by device name) instead of an arbitrary
#      one; active_operations entries gain 'kind' + 'device_name'. Method-rename
#      + return-shape change = contract change.
# v26: removed device.active_setups — device.active_operations is a strict
#      superset (it lists every in-flight op, setups included, each tagged with
#      kind / device_name). The setup-only enumerator was redundant. Method-set
#      change = contract change. (The DeviceService domain getter
#      get_active_device_setups stays — the device dialog still uses it.)
# v27: added project.info — read the applied project identity (chip_name /
#      qub_name / res_name + resolved result_dir / database_path) from the
#      in-process ExpContext (fast-fails no_project when none is applied). It is
#      the sole wire query exposing the project identity; _assemble_overview now
#      folds {chip, qub, res} from it. New method = contract change.
# v28: dialog.screenshot's ParamSpec renamed dialog_name -> name, aligning it with
#      dialog.open / dialog.close (which already take 'name'); the handler reads
#      params["name"]. Wire param-name contract change.
# v29: added analyze.cancel(tab_id) — settle an in-flight (interactive) analyze as
#      cancelled and clear is_analyzing so the tab can be closed; the agent-side
#      counterpart of the GUI 'Done' button (interactive analyze is a separate op
#      from run, so run.cancel does not settle it). New method = contract change.
WIRE_VERSION = 29

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION.
# v2: tab_id is now '<adapter-slug>-<hash>' and an owner-keyed editor_id is
#     '<owner>-ed' (readability; ids stay opaque string keys — no wire change).
# v3: run_lock_changed split into run_started / run_finished (also a wire change
#     — WIRE_VERSION 7; bumped here too since it's a GUI code change).
# v4: cfg path grammar drops 'value' wrapper + set_field returns removed/added
#     (WIRE_VERSION 8).
# v5: progress refactor — device_progress.py -> pbar_host.py (beside plot_host),
#     mutable ProgressBarModel SSOT (worker forwards raw + throttles, main thread
#     computes format/timing live), run.progress adds raw n/total (WIRE 9).
# v6: device setup ↔ run alignment (WIRE 10) — split setup event, device.setup_progress.
# v7: progress big refactor — Qt-free ProgressService + ProgressContainer (owns
#     dict[operation_id, container]) behind a ProgressTransport port whose Qt
#     marshal (QtProgressTransport) is a driven adapter; run/device no longer
#     rebuild a ProgressModel; Views attach by owner_id. Wire shape unchanged.
# v8: soc.info RPC (WIRE 11) — expose the connected SoC's soccfg to the agent;
#     mcp folds the description into connect replies.
# v9: adapter.guide RPC (WIRE 12) — adapters carry a static AdapterGuide; GUI adds
#     a read-only "Guide" tab beside Config/Analysis.
# v10: app.shutdown RPC (WIRE 13) — MainWindow.request_shutdown drives the normal
#     close path (persist/teardown) without the device-setup modal; close logic
#     factored into _perform_close shared with closeEvent.
# v11: cfg introspection slimming (WIRE 14, Phase 120b) — list_paths gains
#     under/verbosity (compact default at mcp); set_field/set_fields stop echoing
#     cfg content (no lowering/eval side effect), set_field returns
#     {valid, removed, added}; adapter.cfg_spec lists ModuleRef '.ref' + choices
#     only (no variant inner-field descent); list_subtree_paths removed.
# v12: analyze takes an operation lease (WIRE 15, Phase 120c). analyze is an
#     async worker, not synchronous as v11 assumed — it now acquires an
#     OperationKind.ANALYZE lease (never conflicts; handle-only) and analyze.start
#     returns {operation_id}. The mcp gui_analyze tool awaits it, so analyze is
#     synchronous to the agent and its figure is ready when the reply lands.
# v13: stale-version guard error carries data.stale (WIRE 16, Phase 120c-3) — the
#     error envelope gains an optional ``data`` field; the version guard names the
#     resource identities that moved (no version numbers) so mcp translates them
#     into agent language ("the active context", "this tab's cfg", "device 'flux'").
# v16: context creation binds to a flux device (WIRE 19, Phase 128) —
#     Controller.new_context(bind_device, clone_from) resolves unit/value from
#     the device (read-only, strict whitelist) instead of taking raw value/unit;
#     the setup dialog's "New context" button drives the same path.
# v17: clean start (WIRE unchanged) — run_app(clean=) / run_measure_gui --clean /
#     gui_launch(clean=) skip restoring the persisted session at startup
#     (restore_all(load=False)); the file is left untouched and a normal close
#     still flushes over it.
# v18: run start clears prior run/analyze/writeback result (State.clear_tab_results
#     in RunService.start_run) so a run-in-flight tab honestly has no result;
#     progress query unified to operation.progress(operation_id), removing
#     get_run_progress / DeviceService.setup_progress (WIRE 20, Phase 129).
# v19: FakeFreqAdapter persists real (simulated) HDF5 on save by default
#     (persist_data=True) so "data saved" is truthful; SweepValue auto-derives
#     step from start/stop/expts at construction (auto_norm, SweepEditor opts
#     out) so default cfg step is consistent across views (Phase 130, WIRE 20).
# v20: tab.get_current_figure rename + save.* return resolved written path(s)
#     (WIRE 21); deviceref path '<path>.device' now resolves through set_field
#     (it was advertised by list_paths but the resolver rejected it — discovery
#     and setter were out of sync). Phase 144.
# v21: AnalysisMode.INTERACTIVE — onetone/twotone flux_dep let the USER drag two
#     lines on the 2D map (half/integer flux) and click Done; flx_half/flx_int/
#     flx_period write back. New InteractiveHost/InteractiveSession ports +
#     setup_interactive_analysis; analyze degrades like a run (gui_analyze ->
#     pending for an interactive pick) — see MCP 26. WIRE unchanged (no new RPC
#     method / param / event). Phase 145.
# v22: BackgroundService extraction (ADR-0019, internal). The 3 per-op QThread
#     workers + 3 runners (RunWorker/AnalyzeWorker/SaveDataWorker + runners)
#     collapse into one BackgroundService.submit(work, scopes, *, run_in_pool,
#     on_done, on_error) — the OffMain execution strategy with opt-in
#     OffMainScopes (figure routing+liveplot / pbar / cancel). run / analyze
#     (FIT) / save compose it; cancel interpretation (finished vs
#     cancelled+partial) moves into RunService (it owns the stop_event). The
#     interactive widget delegates run_background to a narrow InteractiveHostEnv
#     port (the Controller, backed by bg's pool) instead of owning a QThreadPool.
#     Pure-internal refactor; WIRE unchanged (no RPC/param/event change). Phase 146.
# v23: Handle / Exclusion split (ADR-0019 Phase B, internal). The old OperationGate
#     facade splits into OperationGate (pure hardware Exclusion: ensure_can_start /
#     register / release) + OperationHandles (the async Handle/Cancel facet: create
#     mints the token, settle / await_outcome / poll / cancel / cancel_all /
#     live_count). run / device / connect compose both under one token; analyze /
#     interactive take ONLY a handle (no fake exclusion lease). Terminal path
#     settles the handle then frees the exclusion. Shutdown + operation.await read
#     OperationHandles; active_operation_count = handles.live_count (now includes
#     analyze / interactive). WIRE unchanged (no RPC/param/event change). Phase 147.
# v24: device execution onto BackgroundService (ADR-0019 follow-up, internal). The
#     two device QThread workers (_DeviceCommandWorker / _DeviceSetupWorker) are
#     replaced by bg.submit: connect/disconnect carry no scopes; setup carries the
#     progress scope (the stop_event is captured by the work closure + polled by
#     the driver, not an ActiveTask scope) and its cancel interpretation moves into
#     DeviceService._on_setup_done. The "is a setup cancellable" check now reads
#     _active_kind (no _setup_worker field). WIRE unchanged. Phase 148.
# v25: optional analyze params. _resolve_field_info recognises Optional[T] (=
#      Union[T, None]) and flags it; the analyze form renders the existing
#      optional-scalar widget (a QLineEdit whose empty "(none)" state = None, with
#      a numeric validator — the same widget the cfg form uses for optional
#      scalars, ADR-0010); describe_analyze_params adds optional:true so an agent
#      knows it may pass null (dataclasses.replace already accepts None — no wire
#      change). twotone/ro_optimize/length now exposes its t0 length-penalty knob
#      (blank = raw SNR max). WIRE unchanged (the mcp forwards the spec / updates
#      verbatim, never interpreting optional). Phase 149.
# v26: onetone/twotone flux_dep no longer expose the force_magnitude ("Magnitude
#      only") analyze param — FluxPickParams is now an empty marker and the
#      magnitude-only projection is hardcoded per adapter (one-tone True — phase
#      uninformative; two-tone False) in setup_interactive_analysis. Observable as
#      an empty analyze_spec for those two adapters. WIRE unchanged (analyze_spec
#      content is per-adapter domain, not a wire-method contract).
# v27: device.connect / device.disconnect ParamSpec declared (type_name, name,
#      address, remember for connect; name, remember for disconnect). Semantics for
#      correct callers are unchanged — validate_params now enforces the contract
#      that the handlers already assumed. WIRE_VERSION unchanged (no new RPC method
#      or wire-shape change).
# v28: post-analysis dual-end RPCs (WIRE 22) — the existing PostAnalyzeService /
#      start_post_analyze / get_post_analyze_result façade is now reachable over
#      the wire (post_analyze.start + the two get_post_analyze read methods),
#      mirroring the analyze trio. Carrier layer + UI were already in place; this
#      only adds the dispatch handlers. Phase 4 (post-analysis dual-end).
# v29: complex writeback round-trip (WIRE 23). _json_safe encodes complex as
#      {"__complex__": [re, im]} + _coerce_wire_value decodes it on writeback.set;
#      the UI Edit dialog parses "re+imj" (complex branch in _coerce_scalar_input);
#      the GE adapter now proposes g_center / e_center (complex) alongside fid/ge_s.
# v30: figure/screenshot consolidation (WIRE 24). Removed MainWindow.take_screenshot
#      (and the RenderView protocol entry); the only screenshot path is
#      take_figure_screenshot, which now renders at a small fixed geometry
#      (SCREENSHOT_FIGSIZE/SCREENSHOT_DPI in figure_export) so the agent PNG stays
#      token-light while save_image keeps the full SAVE_FIGSIZE/SAVE_DPI quality.
#      Added save_post_image dispatch (Controller.save_post_image already existed).
# v31: device active-op enumeration (WIRE 25). DeviceService singular getters
#      (get_active_setup / get_active_device_operation) replaced by plural
#      enumerators (get_active_device_setups / get_active_device_operations); the
#      stale single-valued port declaration is dropped.
# v32: removed the device.active_setups wire method (WIRE 26) — superseded by
#      device.active_operations. The domain getter is untouched.
# v33: view.snapshot tab_ids / active_tab_id now project from State
#      (ctrl.list_tab_ids) instead of MainWindow._tab_widgets — a widget map that
#      diverged from the tab registry can no longer leak "ghost" tab ids that
#      gui_tab_list / gui_tab_list_paths reject. WIRE unchanged (no RPC/param/event
#      change; pure GUI-side projection fix).
# v34: (a) project.info dispatch handler (WIRE 27) — exposes the in-process
#      ExpContext project identity (chip/qub/res + result_dir/database_path) over
#      the wire. (b) Controller.build_agent_state_context renamed
#      build_agent_bootstrap_prompt and rewritten: it no longer bakes a GUI-state
#      snapshot (project/context/soc/tabs) into the launched agent's prompt — that
#      snapshot went stale the moment the user touched the GUI — but emits a
#      constant directive telling the agent to read live state via gui_overview
#      first. agent_launcher's state_context kwarg renamed bootstrap_prompt to
#      match. WIRE bumped for (a) only.
# v35: dialog.screenshot handler reads params["name"] (was params["dialog_name"]),
#      tracking the WIRE 28 param-name alignment with dialog.open / dialog.close.
# v36: (a) mathtext parser thread-safety fix — a process-wide lock wraps
#      matplotlib's MathTextParser.parse (+ main-thread prewarm) installed at each
#      GUI startup, so off-main analyze plotting ($-titles via tight_layout) can no
#      longer corrupt the shared singleton parser (the recurring ParseException).
#      (b) analyze.cancel wiring (WIRE 29): interactive-analyze teardown
#      (cancel_interactive + RenderHost.unmount_interactive_analysis).
GUI_VERSION = 36
