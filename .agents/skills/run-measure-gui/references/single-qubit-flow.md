# Single-Qubit Bring-Up Flow

Read this before running real or mock single-qubit bring-up measurements.

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

The **standard order** to bring up a flux point (→ = "then"). Each writeback is
reviewed by the agent/human before it becomes context for the next stage:

1. `lookback` → `timeFly` (readout time-of-flight)
2. `onetone/freq` → preliminary `r_f`, `rf_w` (resonator freq + linewidth)
3. `onetone/flux_dep` → user/agent-reviewed `flx_int`, `flx_half`, and
   `flx_period` when the resonator map supports those picks
4. Move the flux device to `flx_int` (the integer-flux sweet spot)
5. Re-run `onetone/freq` at `flx_int` → final `r_f`, `rf_w` for that flux point
6. `twotone/freq` **wide survey** at `flx_int` → find a real qubit feature
7. `twotone/freq` **narrow scan** around the observed feature → `q_f`, `qf_w`
8. Agent/user review of the figure and writeback preview → apply the safe subset
9. Continue with `twotone/rabi/len_rabi`, `twotone/rabi/amp_rabi`, optional
   readout optimization, `twotone/t2ramsey` to refine `q_f`, re-run amp Rabi with
   the corrected `q_f`, then `twotone/t1`

Optional readout power checks (`onetone/power_dep`, `twotone/power_dep`, etc.)
belong where the figure shows they are needed. Reset characterization
(single/dual/bath, in the notebook) is a separate sub-procedure layered on top
once a π pulse exists.

**Do not assume the flux position or qubit frequency in either mock or real
flows.** Treat the system as a black-box measurement: do not use simulator truth,
hidden fake-device parameters, or predictor output to choose `flx_int`, `flx_half`,
or `q_f`. Predictor output is non-authoritative context only after measured
features exist; it is not measurement evidence, does not choose writeback values,
and does not replace the wide `twotone/freq` survey.

**Do not use `twotone/flux_dep` early to find flux.** Before readout and qubit
parameters are credible, a two-tone flux map commonly shows nothing useful: the
readout tone can be wrong, the qubit-drive window can miss the transition, and the
gain may be inappropriate. Find `flx_int` / period with `onetone/flux_dep`, move
there, re-measure the resonator, and only then do wide → narrow `twotone/freq`.
Use `twotone/flux_dep` later for qubit-model mapping after `r_f`, `q_f`, and the
drive/readout settings are already believable.

At a fresh `flx_int`, `q_f` is unknown. Start `twotone/freq` with a **wide,
hardware-safe survey window** chosen from user/lab constraints and the relevant
passband, then narrow only after the figure shows a real feature. A narrow or
mis-centered twotone window returns pure noise that *looks* like "no qubit"; widen
and re-centre before giving up.

## Decision boundaries — act vs. ask

Separate what you may infer automatically from what needs the user. The
dividing line is **physical interpretation**: safe *procedure* is yours, *which
feature is real / is the target* is the user's.

**You may decide automatically** (safe procedure): which adapter implements a
notebook step; sweep ranges within the guide's recommended bounds; narrowing or
widening a window; raising readout gain to improve SNR within safe limits;
reversing a flux sweep direction; re-running after a bad fit.

**Stop and ask the user** (physical interpretation):

- A coarse one-tone scan shows **multiple plausible dips/peaks** — ask which
  feature is the target resonator **before** narrow-fitting or writeback. Do not
  pick the strongest/first by default.
- A **weak or unusual spectral structure** might be the intended mode or an
  unrelated one (or a DAC mirror image) — confirm it is the target before
  building on it.
- Data stays **ambiguous or poor after a few reasonable parameter changes** —
  stop and consult rather than keep guessing; the next choice likely depends on
  physics you cannot see.

When you stop, say what you see (with a `gui_tab_get_current_figure` PNG), name
the options, and let the user choose.

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
- **After every important `run`, look at the figure before trusting any number.**
  A finished `gui_tab_run_start` / `gui_tab_analyze_start` reply (settled in the short
  wait) FOLDS a `figure` (PNG path; `None` if the render failed) — Read that. After a
  `pending`->`finished` op the generic `gui_op_wait`/`gui_op_poll` report only status, so
  fetch the plot with `gui_tab_get_current_figure`. `gui_tab_get_current_figure`
  is rarely needed (a re-render, a mid-flight plot, or a chosen `out_path`). It returns the current plot
  whether or not the adapter does analysis, so for a **2D scan with no fit**
  (`onetone/twotone flux_dep`, `power_dep`) it is the 2D map itself.
  Judge: is the feature clean, the window right (too wide / too narrow), the SNR
  acceptable, the dispersive shift actually small? See the figure first, trust the
  number second. A plausible-looking fit value can still come from a visibly bad
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
- **Treat `fit_bg_amp_slope` as a multiplicative amplitude-background knob.**
  It fits a real log-amplitude slope across the whole resonator response and
  does not add phase slope. For narrow windows or overlapping peaks, re-run
  with `fit_bg_amp_slope=false` and compare the fitted envelope, circle, and
  linewidth—not just the summary.
- **Do not force single-peak `hm` fits onto obviously overlapped or truncated
  features.** In those cases it is better to save the trace, report a manually
  verified dip frequency, and leave linewidth / derived module writeback unset
  than to write back bad fit parameters.
- **When characterizing multiple resonators, rename writeback targets per peak
  before applying them.** Avoid clobbering shared keys like `r_f`, `rf_w`,
  `readout_rf`, or `ro_waveform` unless the user explicitly wants the latest
  result to become the canonical default.

Flux/RF sources (YOKOGS200, SGS100A) are driven as **devices**:
`gui_device_connect(type_name, name, address)` → `gui_device_apply(name,
updates={"value": ...})` ramps an output (a slow apply degrades to a handle; drive
it with `gui_op_wait(handle)` / `gui_op_poll(handle)`, and its progress bars ride the
`gui_op_poll` reply while running). Cancel a device op with `gui_device_cancel(name)`
(only an apply is cancellable). Omit type_name/address to reconnect a known device
by name. `gui_device_fields(name)` lists the settable fields; `gui_device_list`
shows each device's status (memory_only / connecting / connected / disconnecting /
setting_up). Sweeping a device across an experiment is done in the adapter cfg's
`dev` / sweep section, not by manual per-point setup. Different devices set up
**concurrently**: `gui_device_list_operations` lists every in-flight device op
in one call (each entry has its `handle` + `kind`, where kind is
device_connect / device_disconnect / device_apply) — then drive each handle with
`gui_op_wait` / `gui_op_poll`.

**Stash reusable constants in the context (md/ml), then reference them by name
in cfg.** Channel numbers, `res_probe_len`, probe-pulse lengths etc. go into the
MetaDict (`gui_context_md_write` — a batch `attrs=[{key,value},…]`); to read keys
use `gui_context_md_read(keys=[…])` (omit `keys` for the whole tree), which returns
`{values: {key: value}}`; delete with `gui_context_md_delete(keys=[…])`. Named
waveforms/modules go into the ModuleLibrary (`gui_context_ml_create_from_role` /
role tools; list with `gui_context_ml_list`, inspect one with
`gui_context_ml_inspect`). A cfg field can then reference `md.<attr>` (e.g. a pulse
`freq: r_f`) or a module/waveform by its library key, instead of hard-coding — the
notebook does exactly this (`md.res_ch`, `ro_waveform`, `readout_rf`, `pi_amp`).

For mock/offline practice use `fake/freq` (resonator-spectroscopy fake, no
hardware) — the smoke harness uses it.
