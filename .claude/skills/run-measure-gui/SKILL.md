---
name: run-measure-gui
description: Run, drive, screenshot, and smoke-test the measure-gui qubit-measurement GUI over its MCP control socket. Use when asked to launch/start/test the measure-gui app, drive a single-qubit measurement (lookback, onetone/twotone spectroscopy, Rabi, T1/T2, readout optimization) via the measure-gui MCP tools, take a GUI screenshot, or follow the recommended experiment flow.
skill_version: 13
---

# run-measure-gui

`measure-gui` is a Qt desktop GUI for superconducting-qubit (Fluxonium)
measurement on a ZCU216 FPGA. It is driven headlessly through an MCP server
(`measure-gui`, configured in `.mcp.json`) that launches the GUI subprocess and
relays a newline-delimited JSON RPC over a TCP control socket. **You drive it
with the `mcp__measure-gui__gui_*` tools** — there is no separate driver to
write; the tool sequence *is* the harness.

A standalone smoke driver (`smoke.py`, next to this file) talks the same socket
directly (wire method names, no MCP client) and runs the whole experiment loop
against a **mock SoC** — use it to verify the GUI launches and the loop works on
a fresh checkout.

Paths below are relative to the repo root (`<repo>` = the directory with
`.mcp.json`; the GUI launcher lives at `script/run_measure_gui.py`).

## Before touching real hardware — READ THIS

> **Mock SoC / `FakeDevice` flow? Skip this section.** It is safe and offline —
> none of the hardware-safety rules below apply. They kick in only when a real
> `YOKOGS200` / `SGS100A` is involved.

You drive **software only**. You cannot see the cabling, the sample, or the
fridge. So on a real (non-mock) session you **must get hardware facts from the
user first**, and you **must respect device safety limits**.

- **⚠️ YOKOGS200 in current mode: keep the value within ±7e-3 A.** Exceeding it
  can **physically destroy the instrument**. `gui_device_setup(name,
  updates={"value": ...})` and any flux-sweep edge must stay in range. When in
  doubt, ask — do not guess a flux value.
- **Read the board first with `gui_soc_info`** (after connect). It returns the
  QICK soccfg: a compact human-readable `description` table (per-channel
  **generator/readout type, converter port, sample rate (`fs`), max pulse/buffer
  length**) plus a structured `cfg` carrying the full detail (DDS frequency
  ranges, tile layout, …). `gui_connect_start` / `gui_connect_wait` also fold
  this description into their reply. This is the hardware you *can* see in software.
- **Ask the user for what soccfg does NOT tell you** (the wiring/physics):
  - Which DAC/ADC **channels** are wired to the readout transmission line, and
    which to the qubit drive line (soccfg lists the channels and their rates,
    but not what each is cabled to).
  - Qubit type: **Transmon or Fluxonium** (changes the expected spectrum and the
    flux model).
  - Estimated **readout-resonator frequency range**.
  - The **filter / balun passband** on each channel — signal outside it is
    heavily attenuated, so sweeps must stay inside. (The notebook encodes this
    in channel names, e.g. `qub_1_4_ch` = a 1–4 GHz balun line — mirror that
    convention in your md attrs to remind yourself.)
- **`nqz` (Nyquist zone) rule of thumb:** use `nqz=2` for tones **> 2000 MHz**;
  only use `nqz=1` for tones **< 1500 MHz** (there `nqz=1` gives stronger signal
  than `nqz=2`).
- **There are many more details than fit here. When anything looks wrong or
  ambiguous, stop and ask the user** — a wrong device value or out-of-band sweep
  is expensive (instrument damage or hours of bad data), not just a retry.

None of this applies to the mock SoC / `FakeDevice` flow (safe, offline) — but
the moment a real `YOKOGS200`/`SGS100A` is involved, it does.

## Prerequisites

- The project venv (`.venv`) with deps installed; Python is pinned to 3.9.
- An X display. On a headless box: `sudo apt-get install -y xvfb` and prefix
  GUI launches with `xvfb-run -a`. On a desktop session `DISPLAY` is already
  set and no xvfb is needed.
- No hardware: the mock SoC + `FakeDevice` cover the whole flow offline.

## Run (agent path — MCP tools)

This is the path you use when asked to "drive measure-gui". The MCP server
auto-launches the GUI; you call tools.

```
gui_launch                                      # spawns the GUI, connects; banner shows the
                                                # handshake: "wire vN (mcp==gui); gui code vX, mcp code vY"
# Same startup path a GUI user takes (no mock shortcut):
gui_connect_start(kind="mock")                  # mock SoC (or kind="remote", ip, port for hardware)
gui_startup_apply(chip_name="Q1_Chip",          # apply the project; omit result_dir/database_path
                  qub_name="Q1", res_name="R1")  # to scope them under chip/qub (notebook layout)
gui_context_new(bind_device="flux")             # create a context bound to a flux device (reads its
                                                # current value/unit; FakeDevice->none, YOKOGS200->A).
                                                # Omit bind_device for an unbound context; clone_from=<label>
                                                # clones an existing one. Or gui_context_use(label) for an existing one.
gui_state_check                                 # all four flags must be true before running
gui_soc_info                                    # the board: per-channel type, sample rate, port, max length (+ full cfg)
```

Then the experiment loop (per tab):

```
gui_adapter_list                                  # available experiments
gui_tab_new(adapter_name="fake/freq") -> tab_id   # readable id, e.g. fake-freq-1a2b3c4d
gui_tab_snapshot(tab_id) -> editor_id             # the cfg-editing session handle
gui_tab_list_paths(tab_id)                        # dotted cfg paths (compact: path+kind+choices)
gui_tab_get_cfg_summary(tab_id)                   # current values/expressions, nested (ref nodes wrap {chosen,value} → not a path source; see list_paths)
gui_editor_set_field(editor_id, "rounds", 30)     # WYSIWYG edit of the form's draft
gui_run_start(tab_id)                             # waits ~1s; finished -> {status:finished,...}, slow -> {status:pending}
gui_run_wait(tab_id)                              # block until done (only after pending)
gui_analyze(tab_id)                               # SYNCHRONOUS fit; reply folds in figure_path (Read it)
gui_save_data(tab_id) / gui_save_image / gui_save_result
gui_view_screenshot(tab_id)                       # base64 PNG of the window/tab
```

Detecting completion — completion is observed synchronously or by waiting/polling
a handle, never by subscribing to a push stream:

| situation | what to do |
|---|---|
| fast run / any analyze | **synchronous** — the call returns when done (`gui_analyze` always; `gui_run_start` when it finishes within `wait_seconds`, default 1.0) |
| `gui_run_start` returned `{status:pending}` | `gui_run_wait(tab_id)` (blocks) or `gui_run_poll(tab_id)` (non-blocking) |
| want live progress bars | already in the `gui_run_poll` reply while `status:running` (active + bars); no separate progress tool |
| a poll says `status:cancelled` | a user/agent cancel (distinct from `failed`); not an error to recover from |

A `diagnostic{severity}` push (errors / info the GUI would show in a dialog) rides
along in the *next* tool reply's notifications — you get it without asking. Don't
busy-poll `gui_run_running_tab` in a sleep loop.

The full, authoritative tool reference is the **MCP server instructions block**
(shown by the client when the server connects, defined in
`lib/zcu_tools/mcp/measure/server.py`). Read it for the call
contract (failed calls raise — never fire duplicates), preconditions, and the
diagnostic-push model.

## Run (smoke harness — verify the loop without an MCP client)

`smoke.py` launches the GUI itself and drives the mock pipeline over the raw
socket. Use it to prove a fresh checkout works:

```bash
# desktop session (DISPLAY set):
.venv/bin/python .claude/skills/run-measure-gui/smoke.py
# headless:
xvfb-run -a .venv/bin/python .claude/skills/run-measure-gui/smoke.py
```

Expected tail (≈30–60s):

```
[smoke] ready: {'project': True, 'context': True, 'active_context': True, 'soc': True}
[smoke] tab: fake-freq-09dbdc70
[smoke] edited reps/rounds
[smoke] progress: rounds %v/%m [0:00<0:04] (3.3%)
[smoke] run finished (saw_live_progress=True)
[smoke] analyzed + saved data
[smoke] screenshot -> /tmp/measure_gui_smoke.png (137535 bytes)
[smoke] tab closed
[smoke] SMOKE OK
```

It writes the screenshot to the OS temp dir (`$TMPDIR/measure_gui_smoke.png` —
the Analysis tab with the fitted resonator dip + writeback panel). It uses
control port **8799** so a live MCP session on the default 8765 is undisturbed.

## Recommended single-qubit flow (notebook → adapters)

`notebook_md/single_qubit.md` is the canonical bring-up procedure. Each notebook
section maps to a GUI **adapter** (`gui_tab_new(adapter_name=...)`); run →
analyze → write results back (the Writeback panel, or just feed the next tab).
The flow threads calibrated values through the MetaDict context (`r_f`, `q_f`,
`pi_gain`, …) — the GUI persists these per flux-context. **`magic_names.md`
(next to this file) is the glossary for these names**: meaning, unit, which
experiment produces each, and the cross-experiment relations (e.g.
`reset_f = r_f − q_f`, `post_delay = 5/(2π·rf_w)`). Per-experiment recommended
sweep ranges stay in `gui_adapter_guide`, not the glossary.

The **standard order** to characterize one bias point (→ = "then"; `(opt)` =
optional). Each stage's fitted result is written back into the context and
consumed by the next:

1. `lookback` → `timeFly` (readout time-of-flight)
2. `onetone/freq` → `r_f`, `rf_w` (resonator freq + linewidth)
3. `onetone/power_dep` **(opt)** → readout saturation power
4. `onetone/flux_dep` → flux period / sweet spot — **then move the flux bias to
   the integer (half-flux) point** before continuing
5. `twotone/freq` → `q_f`, `qf_w` (qubit freq)
6. `twotone/flux_dep` **(opt)** → fit qubit model parameters
7. `twotone/rabi/len_rabi` → `pi_len`, `pi2_len`, `rabi_f`
8. `twotone/rabi/amp_rabi` → `pi_gain`, `pi2_gain`
9. optimize readout **(opt)** (`twotone/power_dep` etc. → readout freq/gain/length)
10. `twotone/t2ramsey` → `t2*` — **use it to calibrate `q_f`** (Ramsey detuning)
11. **re-run `twotone/rabi/amp_rabi`** with the corrected `q_f`
12. `twotone/t1` → `t1`

Steps 7–12 (Rabi/T2/recalibrate/Rabi/T1) are the per-point qubit
characterization. Reset characterization (single/dual/bath, in the notebook) is
a separate sub-procedure layered on top once a π pulse exists.

**Readout bring-up rules that generalize beyond one notebook run:**

- **Do not start `onetone/freq` on real hardware until `lookback` has set a
  believable `timeFly`.** A guessed trigger offset can shift the readout window
  off the arriving signal and distort every downstream resonator fit.
- **Run `lookback` off-resonance when calibrating `timeFly` for an `hm`
  resonator.** On resonance the resonator can absorb the probe and hide the
  clean square-step edge you actually need for timing.
- **Tune `lookback` to the rising edge, not the saturated plateau.** Leave
  enough pre-trigger baseline to see the step start cleanly, keep the pulse /
  readout window short enough that the plot focuses on the edge, and only then
  trust the predicted offset.
- **If `lookback` is using a trigger expression derived from an old `timeFly`,
  break that feedback loop before re-measuring.** Use a direct fallback
  `trig_offset` (or remove the old md value) when you need an unbiased timing
  re-calibration.
- **Probe gain is part of the timing calibration.** If the user tells you the
  line only works at a stronger gain, re-run `lookback` at that gain before
  trusting the earlier `timeFly`.
- **Do not confuse GUI pulse gain with external device power/current values.**
  The readout-pulse `gain` fields in these adapters are normalized waveform
  amplitudes, typically safe in the range `-1` to `+1`, and do **not** mean the
  same thing as a source's physical output setting.
- **For power-insensitive readout experiments such as `lookback` and
  `onetone/freq`, `gain=1.0` is often the right default on a weak line.** If the
  user tells you the line is not power-sensitive for the task at hand, prefer
  the stronger readout drive so the signal-to-noise ratio is good enough to
  judge timing and resonator features cleanly.
- **After every important `run`, inspect the analysis figure before trusting the
  scalar summary.** A plausible-looking number can still come from a visibly bad
  fit.
- **Before choosing a `lookback` frequency, ask whether the readout resonator is
  hanger-like (`hm`) or transmission-like (`t`).** For a hanger, the resonant
  point absorbs and `lookback` should be done off-resonance; for a transmission
  resonator the resonant point transmits more strongly and `lookback` should be
  done on-resonance.
- **When the user does not specify the resonator family, ask instead of
  guessing.** A common lab convention is that multi-qubit chip-integrated
  readout resonators are hanger-type, while single-qubit sample-box readout
  modes are often transmission-type, but this is only a heuristic.
- **For narrow windows and overlapping peaks, treat `fit_bg_slope` as a
  potential overfit knob.** If the background line is doing obvious work the
  physics should be doing, re-run with `fit_bg_slope=false` and compare the
  figure, not just the summary.
- **Do not force single-peak `hm` fits onto obviously overlapped or truncated
  features.** In those cases it is better to save the trace, report a manually
  verified dip frequency, and leave linewidth / derived module writeback unset
  than to write back bad fit parameters.
- **When characterizing multiple resonators, rename writeback targets per peak
  before applying them.** Avoid clobbering shared keys like `r_f`, `rf_w`,
  `readout_rf`, or `ro_waveform` unless the user explicitly wants the latest
  result to become the canonical default.

Flux/RF sources (YOKOGS200, SGS100A) are driven as **devices**:
`gui_device_connect(type_name, name, address)` → `gui_device_setup(name,
updates={"value": ...})` ramps an output (cancellable; a slow setup degrades to
a handle, and its progress bars ride the `gui_device_poll` reply while running).
`gui_device_setup_spec(name)` lists the settable fields. Sweeping a device across an experiment is done in the adapter cfg's
`dev` / sweep section, not by manual per-point setup.

**Stash reusable constants in the context (md/ml), then reference them by name
in cfg.** Channel numbers, `res_probe_len`, probe-pulse lengths etc. go into the
MetaDict (`gui_context_set_md_attr`); named waveforms/modules go into the
ModuleLibrary (`gui_ml_create_from_role`/role tools). A cfg field can then reference
`md.<attr>` (e.g. a pulse `freq: r_f`) or a module/waveform by its library key,
instead of hard-coding — the notebook does exactly this (`md.res_ch`,
`ro_waveform`, `readout_rf`, `pi_amp`).

For mock/offline practice use `fake/freq` (resonator-spectroscopy fake, no
hardware) — the smoke harness uses it.

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
- **`gui_launch` vs `gui_connect` (both default port 8765, opposite
  expectations).** launch starts a NEW GUI and needs the port FREE; connect
  attaches to an EXISTING GUI and needs the port LISTENING (errors "No GUI is
  listening on 127.0.0.1:8765" otherwise). Use launch to start, connect only to
  re-attach to one already up.
- **A run starts by clearing its tab's prior run/analyze/writeback result.** So
  while a run is in flight — and after it fails — the tab has no result:
  `gui_analyze` / `gui_save_*` fail-fast with `no_run_result` (the true reason:
  this run hasn't produced one yet), not a "busy" message. `gui_editor_set_field`
  while running is the one that returns `precondition_failed: ... is currently running`.
  Wait for the run to settle (`tab.snapshot.interaction.is_running` false, or
  `gui_run_wait` / `gui_run_poll`) before analyzing/saving. **Cancelled runs are
  the exception:** if the worker produced a partial result before observing the
  stop signal, the tab intentionally keeps that partial result (`has_run_result`
  true) and analysis/save may proceed; if no partial result exists, analyze/save
  still fail with `no_run_result`. (The smoke harness waits on `is_analyzing`
  before `save.data`.)
- **`run` success is not `analyze` success.** A completed acquisition can still
  produce a bad or misleading fit. On real data, open the figure and verify the
  model visually before you trust `gui_tab_get_analyze_result`, especially for
  lookback timing, narrow resonator windows, and overlapping dips.
- **Minimum writeback bar: inspect the analysis figure first.** Do not write fit
  results into the context or module library unless the plotted fit matches the
  feature you intended to measure.
- **Saved data is always `.hdf5`, with a uniqueness suffix.** `gui_save_data` /
  `gui_save_result` force the `.hdf5` extension and append `_N` (e.g. a
  `data_path` of `foo` or `foo.h5` lands as `foo_1.hdf5`). The reply's diagnostic
  and the tab's `save_paths` report the resolved path — read the file back by
  that, not by the path you passed in.
- **cfg paths have no `value` segment.** Module sub-fields are
  `modules.qub_pulse.freq`, not `...qub_pulse.value.freq`; an unknown path
  fails `invalid_params` rather than silently no-op'ing. Always confirm against
  `gui_tab_list_paths`. **Do NOT lift paths out of `gui_tab_get_cfg_summary`:**
  that view deliberately wraps each module/waveform ref as `{chosen, value:{...}}`
  (so it can keep EvalValue expressions and the chosen variant, which lowering
  would drop) — so its keys carry the extra `.value.` segment. It is a values/
  expressions view, not a path source; get editable paths from `list_paths`.
- **Edit through the tab's `editor_id`, not the tab directly.** Take `editor_id`
  from `gui_tab_snapshot`; `gui_editor_set_field` edits the same draft the form
  shows. Switching a ModuleRef key (`<path>.ref`) returns `removed`/`added`
  settable paths so you needn't re-list.
- **Real VISA devices (YOKOGS200/SGS100A) need a VISA backend.** Without
  `pyvisa-py` or the IVI binary, `gui_device_connect` fails
  `Could not locate a VISA implementation`. Use `FakeDevice` for offline device
  flows.
- **`run.start` is fire-and-forget.** A duplicate call starts a *second* run.
  Issue mutating tools exactly once and read the reply.
- **If a writeback item is only partially trustworthy, apply only the safe
  subset.** Frequency-only writeback is often safer than writing a dubious
  linewidth or derived readout module after a marginal fit.
- **Mirror/image peaks in spectra (ZCU DAC artifact).** The Xilinx ZCU board's
  DAC has a finite sampling rate and side-band leakage, so a real transition can
  show up *mirrored* around `sample_f/2` in a twotone spectrum. A peak at the
  "wrong" frequency may be an alias, not a real transition — sanity-check
  against the predicted qubit/resonator frequency before trusting it, and ask
  the user if unsure.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gui_launch` → `Port 8765 is already in use` | A previous GUI is still running on the port; `gui_stop` it (or kill the stale `run_measure_gui.py`), then relaunch — or launch on another port. |
| `gui_connect` → `No GUI is listening on 127.0.0.1:8765` | Nothing running there; `gui_launch` first (connect only re-attaches to a running GUI). |
| `precondition_failed: ... is currently running` on `gui_editor_set_field` | The tab is running; wait for it to finish (run clears prior results, so editing mid-run is blocked). |
| `no_run_result` on `gui_analyze` / `gui_save_*` | No result for *this* run yet — the run is still in flight, failed, or was cancelled before producing a partial result (a run clears the previous result on start). Wait for it to finish, or re-run. |
| `precondition_failed: no_project` on `gui_context_new` | No project applied; `gui_startup_apply` first. |
| `precondition_failed` on run/save with no busy tab | Missing active file-backed context — `gui_state_check`, then `gui_startup_apply` + `gui_context_new`/`gui_context_use` if a context is missing. |
| `invalid_params` on `gui_editor_set_field` | Path wrong (often a stray `value` segment); re-check `gui_tab_list_paths`. |
| `Could not locate a VISA implementation` | Real device driver with no VISA backend; use `FakeDevice` or install `pyvisa-py`. |
| GUI never renders / launch times out | No X display; set `DISPLAY` or run under `xvfb-run -a`. |
| Stale GUI behaviour after a code change | `gui_stop`, `/mcp reconnect measure-gui`, then `gui_launch`; confirm the change via an observable effect (the version banner can match a stale process). |
