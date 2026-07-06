# Gotchas and Troubleshooting

Read this when a tool result, GUI state, fit, writeback, save, or device operation looks wrong.

## Gotchas

- **`gui_launch` banner is the reload check.** It prints `wire vN (mcp==gui);
  gui code vX, mcp code vY`. After editing GUI code you must `/mcp reconnect
  measure-gui` **then** `gui_launch` — the MCP server is a separate process and
  caches the old code until reconnected. `wire` mismatch = hard error; `gui
  code` / `mcp code` are reported, you eyeball them to confirm the reload.
- **`/mcp reconnect` does NOT stop the running GUI — it keeps its port.** A
  reconnect drops the bridge's socket but leaves the old `run_measure_gui.py` listening
  on 8765. So always `gui_stop` (or kill it) before relaunching. `gui_launch`
  fails fast if the port is occupied ("Port 8765 is already in use …")
  rather than silently attaching to the stale process — but if you ignore that
  error you stay on old code. **The handshake alone won't save you:** a stale
  process reports whatever version it was built at, so if it happens to match,
  `gui code vN` looks fine while you're on old behaviour. Confirm a relaunch by
  an *observable effect of your change*, not just the version banner.
- **`gui_launch` vs `gui_bridge_connect` (both default port 8765, opposite
  expectations).** launch starts a NEW GUI and needs the port FREE; connect
  attaches to an EXISTING GUI and needs the port LISTENING (errors "No GUI is
  listening on 127.0.0.1:8765" otherwise). Use launch to start, connect only to
  re-attach to one already up.
- **A run starts by clearing its tab's prior run/analyze/writeback result.** So
  while a run is in flight — and after it fails — the tab has no result:
  `gui_tab_analyze_start` / `gui_tab_save` fail-fast with `no_run_result` (the true reason:
  this run hasn't produced one yet), not a "busy" message. `gui_editor_set`
  while running is the one that returns `precondition_failed: ... is currently running`.
  Wait for the run to settle (`tab.snapshot.interaction.is_running` false, or drive
  the run handle with `gui_op_wait` / `gui_op_poll`) before analyzing/saving. **Cancelled runs are
  the exception:** if the worker produced a partial result before observing the
  stop signal, the tab intentionally keeps that partial result (`has_run_result`
  true) and analysis/save may proceed; if no partial result exists, analyze/save
  still fail with `no_run_result`. (The smoke harness waits on `is_analyzing`
  before saving.)
- **`run` success is not `analyze` success.** A completed acquisition can still
  produce a bad or misleading fit. On real data, open the figure and verify the
  model visually before you trust `gui_tab_get_analyze_result`, especially for
  lookback timing, narrow resonator windows, and overlapping dips. **A small fit
  error bar does not prove the fit is right** — a noisy Rabi/freq fit can converge
  to a confident-looking value with a tiny uncertainty; only the figure tells you
  whether the model actually matches the data.
- **Minimum writeback bar: inspect the analysis figure first.** Do not write fit
  results into the context or module library unless the plotted fit matches the
  feature you intended to measure.
- **Saved data is always `.hdf5`, with a uniqueness suffix.** `gui_tab_save` /
  `gui_tab_save` force the `.hdf5` extension and append `_N` (e.g. a
  `data_path` of `foo` or `foo.h5` lands as `foo_1.hdf5`). **The save reply
  returns the resolved path directly** (`{data_path}` / `{image_path}` /
  `{data_path, image_path}`) — read the file back by that, not by the path you
  passed in. (The tab's `save_paths` and the diagnostic also carry it.)
- **cfg paths have no `value` segment.** Module sub-fields are
  `modules.qub_pulse.freq`, not `...qub_pulse.value.freq`; an unknown path
  fails `invalid_params` rather than silently no-op'ing. Get editable paths from
  `gui_tab_get_cfg` (the nested tree — `$`-prefixed keys mark leaf metadata;
  plain keys are subtree nodes). `stage1`'s `tree` field is the same tree as
  `get_cfg` — use it directly.
- **`gui_editor_set` / `gui_editor_set` accept either a `tab_id`
  (convenience — the server resolves that tab's cfg-editor automatically) or an
  explicit `editor_id` from `gui_tab_snapshot`.** Both edit the same live draft
  the form shows (WYSIWYG — no separate commit step is needed to run). Switching
  a ModuleRef key (`<path>.ref`) returns `removed`/`added` settable paths so you
  needn't re-list.
- **Tab cfg edits are live — no commit needed before `gui_tab_run_start`.** Changes
  made via `gui_editor_set` / `gui_editor_set` take effect on the
  tab immediately (WYSIWYG). `gui_editor_save(name=...)` is a separate
  operation that saves the current draft as a *named ModuleLibrary module/waveform*
  — it has nothing to do with applying the tab's cfg edits, and you never need it
  just to run the experiment.
- **Real VISA devices (YOKOGS200/SGS100A) need a VISA backend.** Without
  `pyvisa-py` or the IVI binary, `gui_device_connect` fails
  `Could not locate a VISA implementation`. Use `FakeDevice` for offline device
  flows.
- **`tab.run_start` is fire-and-forget.** A duplicate call starts a *second* run.
  Issue mutating tools exactly once and read the reply.
- **If a writeback item is only partially trustworthy, apply only the safe
  subset.** Frequency-only writeback is often safer than writing a dubious
  linewidth or derived readout module after a marginal fit.
- **Mirror/image peaks in spectra (ZCU DAC artifact).** The Xilinx ZCU board's
  DAC has a finite sampling rate and side-band leakage, so a real transition can
  show up *mirrored* around `sample_f/2` in a twotone spectrum. A peak at the
  "wrong" frequency may be an alias, not a real transition — sanity-check
  against the current measured qubit/resonator context and the passband before
  trusting it, and ask the user if unsure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gui_launch` → `Port 8765 is already in use` | A previous GUI is still running on the port; `gui_stop` it (or kill the stale `run_measure_gui.py`), then relaunch — or launch on another port. |
| `gui_bridge_connect` → `No GUI is listening on 127.0.0.1:8765` | Nothing running there; `gui_launch` first (connect only re-attaches to a running GUI). |
| `precondition_failed: ... is currently running` on `gui_editor_set` | The tab is running; wait for it to finish (run clears prior results, so editing mid-run is blocked). |
| `no_run_result` on `gui_tab_analyze_start` / `gui_tab_save` | No result for *this* run yet — the run is still in flight, failed, or was cancelled before producing a partial result (a run clears the previous result on start). Wait for it to finish, or re-run. |
| `precondition_failed: no_project` on `gui_context_create` | No project applied; `gui_project_apply` first. |
| `precondition_failed` on run/save with no busy tab | Missing active file-backed context — check the readiness flags in `gui_overview`'s `state` field, then `gui_project_apply` + `gui_context_create`/`gui_context_switch` if a context is missing. |
| `invalid_params` on `gui_editor_set` | Path wrong (often a stray `value` segment); re-check `gui_tab_get_cfg`. |
| `Could not locate a VISA implementation` | Real device driver with no VISA backend; use `FakeDevice` or install `pyvisa-py`. |
| GUI never renders / launch times out | No X display; set `DISPLAY` or run under `xvfb-run -a`. |
| Stale GUI behaviour after a code change | `gui_stop`, `/mcp reconnect measure-gui`, then `gui_launch`; confirm the change via an observable effect (the version banner can match a stale process). |
