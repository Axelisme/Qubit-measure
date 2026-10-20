# Mux singleshot GE

## GUI setup

Open `mux/singleshot_ge` from **New Tab** after binding `$mux` to a MuxGroup.
The capture must contain every qubit and its matching resonator in readout order.
Readout references `$mux.module.readout` when present, preserving the library's tones and scalar settings.
The stored module must use the mux family with the full group tone count.
Without a slot, Readout starts inline from the blank mux template with group calibration tones; complete its required fields before running.

The Tones picker chooses which element to edit, not which qubit runs.
Name is read-only. Ordinary edits cannot add, remove, rename or reorder tones.
Set Participants to a JSON name list, for example `["Q3", "Q1"]`.
That order becomes execution/data rows for original positions 2/0.
The picker can show Q2 while Q3/Q1 run.
Empty, duplicate and unknown participants remain editable invalid drafts and block Run.

Every tone needs a valid drive, including nonparticipants.
Errors in another picker element remain visible on the Tones row.
Select that element to fix them; hiding it or removing its name from Participants
does not bypass validation. Reps, rounds and recovery delay are common controls.
Recovery initially uses five times the largest captured T1, with the existing
missing-T1 default. A stored invalid T1 leaves recovery blank for correction.

Reps and rounds are fixed at one in the Form.
Shots controls the number of raw acquisitions in each probe-off/on preparation.
Init pulse is optional and runs in both preparations.

## GUI results and saved sources

Run draws a combined `measurement` figure.
Each original tone has probe off/on lines on the shot-index axis.
The legend uses original index and source cfg name in participants order.
Per-tone native `measurement_i` figures remain available.
Analyze reads the selected Run/Load source, not the current Form.
Use the Analysis figure picker to inspect `fit_i` and the Post figure picker for `post_i`.
Here `i` is an original tone position, never a participants row number.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or a saved run.
Loading replaces the analysis source without changing current Form inputs.
A later Form edit does not alter the historical pulse, binding or destination.
Failed or partial rows do not produce calibration candidates.
Read each Writeback preview destination, value and fit diagnostics.
Deselect unwanted items before **Apply Selected**.
Preview and Analyze do not write; applying one item does not apply other candidates.

## Form drafts

Form persistence contains exactly `data` and `selections`.
Selections use typed paths with integer list positions.
Invalid leaf text and participants drafts survive save/restore.
Restoration reports dropped fields and does not run converters or refresh capture.
If the saved input cannot be read, the host reports it and uses fresh defaults.
Reset explicitly captures current parameters and rebuilds initial values.

A raw `FormState.write` replaces the complete value at its path and clears selections
at or below it. To change one leaf, use its complete path, for example
`("tones", 0, "drive", "module", "gain")` for an inline drive.
Do not send a partial tone or pulse dictionary expecting a merge.
Writing does not run a converter; selecting another module kind does.
Recipe/MCP logical field paths omit the inline wrapper, for example
`("tones", 0, "drive", "gain")`. The integer is the full original tone position,
not a row in Participants and not a name key.

## Core reference

## Acquisition

The complete `tones` list retains capture-generated `name` values and original
readout positions. `participants` is a nonempty, unique list of existing names.
Q3/Q1 selects original indices `[2, 0]`, retaining that execution/data order.
Every element requires valid drives, including nonparticipants.

`MuxGEExp.run(config, *, context)` acquires probe-off and probe-on raw IQ for
all active qubits in the same two programs. It does not simulate each qubit
separately or assemble unrelated shots. This core has no Form, catalog entry,
recipe or calibration proposals.

`MuxGECfg.modules` contains the common optional reset and the complete ordered
mux readout. `tones: list[MuxGEToneCfg]` has one element per full readout tone.
Each element contains fixed `name`, required probe `drive`, and optional `init_pulse`.
All drives must be valid. All generators must be independent and
must differ from the readout generator. An initial pulse uses its tone's drive
generator. Nonparticipant settings stay validated in cfg but do not run.

Each preparation runs common reset, then an existing `Join` of the active
branches, then one mux readout. A branch contains its optional initial pulse
followed by the probe in the probe-on preparation only. Branches start together;
shorter branches finish early and wait. Readout follows the longest branch.
There is no end-alignment padding or custom timing module. Native program
modules own MHz/us/degree/gain conversion and hardware quantization. The initial
pulse runs in both preparations; it is not an analysis initial-state option.

`shots` is the number of raw shots in each preparation. As in single-qubit GE,
run copies cfg and forces `rounds=1` and `reps=shots`. It warns when requested
rounds differs from one or requested reps differs from one. Caller cfg is not
changed. `RunRecord.cfg` retains the original requested controls, including those
warnings' inputs. The acquired result always represents one round per
preparation, with its actual shot count on `shot_indices`. It does not claim to
retain multiple requested rounds. The underlying raw getter retains only its
last round; forcing one round here avoids silently dropping earlier rounds.

The program-owned `MuxReadout.collect_raw_iq(prog, read_index=0)` reads the
initialized module of that same acquire. It selects the same read on every tone,
preserves all reps and divides each ADC sum by its rounded integration length.
It does not subtract offsets, rotate IQ, classify shots or average them. This
experiment declares one read. No lab code parses QICK raw buffers.

## Result and cancellation

`MuxGEResult` contains:

- `signals`: complex128 `(K, 2, shots)`, in ADC units per integration sample.
- `tone_indices`: int64 `(K,)`, the active original full-readout indices in order.
- `prepared_states`: int64 `[0, 1]`, meaning probe off/on, not pure g/e states.
- `shot_indices`: int64 `arange(shots)`.

At a given preparation and shot index, all tones belong to the same acquisition
shot. ADC channel numbers and duplicate labels do not reorder these axes. The
two preparations are separate acquires. They are not four-state joint assignment
calibration and do not identify a correlated two-qubit confusion matrix.

`context` supplies compatible hardware, a fresh `StopSignal`, devices and
caller-owned `Plots`. Run publishes `measurement_i` magnitude traces with both
preparations, where `i` is the original tone index. Caller finishes/releases the
plots and closes owned devices.

Completed preparations survive cancellation or acquisition error. A missing
preparation remains NaN across every tone and shot. An incomplete first round is
not accepted as a complete preparation. After run, call
`context.cancel_signal.raise_if_error()` before reporting success. Then inspect
`is_set()` to distinguish cancellation from completion. The signal retains an
acquisition error's cause even when run returns useful partial data. Validation
and build errors propagate. The caller records completion separately; a saved
partial array does not mean the operation succeeded.

## Analysis

`analyze(source, options, *, plots)` reads only the explicit `RunRecord`.
`MuxGEExp.Options()` applies the existing GE defaults to every active tone.
Use original-index `overrides` for different initial states or fit options:

```python
from zcu_lab.v2.mux.singleshot_ge import MuxGEExp
from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions

core = MuxGEExp()
options = core.Options(
    overrides={2: GEAnalyzeOptions(initial_state="excited", backend="pca")}
)
primary = core.analyze(source, options, plots=analysis_plots)
post = core.post_analyze(source, primary, core.PostOptions(), plots=analysis_plots)
```

`source` is the retained run/load record. `analysis_plots` is a fresh
caller-owned `Plots`. `default` and each override use `GEAnalyzeOptions`:
initial state, PCA/center backend, optional angle in degrees, log histogram,
T1 alignment and optional dimensionless length ratio. Initial state describes
the predominant state after reset/init and before probe. It only maps the
analysis rows to physical g/e; stored probe-off/on data never moves.

Each original index returns `MuxGEAnalysis(index, primary)` or
`MuxFitFailure(index, reason)`, with a named `fit_i` figure. The single-qubit GE
owner computes centers, width, fidelity, populations and fit quality. This
caller does not duplicate its fitter, layout or quality policy. Missing pairs
cannot fit, and numerical failures do not become zero-valued calibrations.
Inspect the returned quality before treating finite values as useful calibration.

This experiment isolates missing preparations and the GE owner's
`GENumericalError` per tone. The owner wraps numerical fitting or classification
failures and retains the original exception in `__cause__`. Mux reports its
reason as `MuxFitFailure` and continues with the remaining tones. Plotting and
other errors propagate unchanged, so a broken diagnostic plot is not reported
as a failed calibration fit. This is this experiment's error policy; it does
not require other experiments to isolate the same failures.

Post-analysis uses the adopted same-source primary without refitting. Caller
must retain that source association; passing another record's primary with the
same indices is not detectable. `MuxGEExp.PostOptions` has a common `default`
and original-index `overrides` of `GEPostAnalyzeOptions`. Radius is None for
automatic selection or a finite nonnegative value in normalized ADC units.
`consider_other` affects automatic-radius selection only. Post returns indexed
`MuxGEPostAnalysis` or retained/numerical failures and publishes `post_i`. The
existing 3x3 g/e/other diagnostic retains its isolated-other row. A returned
diagnostic is not a calibration approval: an infinite condition number or
unresolved clusters makes it unsuitable for calibration.

Both analysis operations allow `source.cfg=None`. Invalid source shapes,
shot/preparation axes, cfg associations and option indices raise before fitting.
Neither operation reads current settings or hardware. Use fresh plots for a
second analysis operation; duplicate `fit_i` or `post_i` names are rejected.
Caller finishes/releases the plots.

## Calling and saving

```python
from zcu_tools.experiment import RunRecord
from zcu_lab.v2.mux.singleshot_ge import MuxGEExp

core = MuxGEExp()
result = core.run(config, context=context)
source = RunRecord(config, result)
context.cancel_signal.raise_if_error()
# Preserve stopped/partial status in caller metadata; do not infer it from IQ.
core.save_run(source, native_path, metadata=metadata)
restored, device_snapshot = core.load_run(native_path)
primary = core.analyze(restored, core.Options(), plots=analysis_plots)
post = core.post_analyze(restored, primary, core.PostOptions(), plots=analysis_plots)
```

`config` is a validated `MuxGECfg`; `context` owns this operation's hardware,
plots and stop signal. `native_path` is a caller-selected local Path, and
`metadata` contains the caller's run identity, completion and provenance.
Default native persistence requires a cfg. It stores full tones/readout,
original indices, probe preparation and shot coordinates, and complex raw IQ.
Loading restores the historical source; subsequent edits to a current cfg do
not alter it. No current Form or library is consulted. This source contains the
inputs needed for later primary/post proposals, but the core produces no GUI
writeback.

## Physical smoke limits

Use the public `ftf(settings, workers=1)` environment with typed YAML
`readout.drive.kind: mux`, an ADC receiver list and independent drive lines.
The backend is `ftf-segment-v7`; older replay archives are rejected, not adapted.
All receiver IQ at a shot index must come from the same joint-state draw.
Small truncations and short smoke programs do not establish model convergence,
calibrated probe pulses or reliable readout fits. A poor or failed fit remains
visible as quality evidence or a per-tone failure, not a reason to change the
fitter or splice separately simulated shots.

## Adapter source and proposals

`MuxGEAdapter` in `gui.py` uses explicit Run/Load sources. Its primary and post
operations publish `fit_i` and `post_i` through the existing GE plot owner,
where i is the original tone index. Its native generated Form is registered as
`mux/singleshot_ge`. The registered MCP recipe is `mux_singleshot_ge`. See the
[mux recipe guide](../../../recipes/mux-guide.md) for typed overrides, origins,
source-bound reporting and explicit writeback decisions.

Both summaries retain original `position`, source `name` and per-tone results or
failure reasons. Q3/Q1 gives indices 2/0 in that row order. Quality diagnostics
identify the corresponding summary row. Missing source cfg leaves the name
unavailable; the current Form does not supply it. Post-analysis must use the
primary from the same Run/Load source, not a later Run's primary.

A successful, fully finite primary row passes the GE calibration validity check
before proposing `ext.readout_cal.fidelity`, `sigma`, `g_center` and
`e_center` on the captured qubit. Post proposals contain `radius` and the
existing 3x3 `confusion_matrix`, including the isolated-other diagnostic.
These are per-qubit GE calibrations, not a four-state joint assignment matrix.

Destinations come only from the same Run/Load capture's full binding. Missing
capture yields no proposals; inconsistent captured bindings raise. Failed or
partial rows yield none. Review cluster separation, quality, confusion diagnostics
and destination before explicitly applying individual items. Analysis, post and
proposal creation do not write or apply candidates.
