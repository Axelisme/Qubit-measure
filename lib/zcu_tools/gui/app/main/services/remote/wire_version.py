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
WIRE_VERSION = 21

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
GUI_VERSION = 20
