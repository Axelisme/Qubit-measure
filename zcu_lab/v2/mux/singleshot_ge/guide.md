# Mux singleshot GE

## Acquisition

`MuxGEExp.run(config, *, context)` acquires probe-off and probe-on raw IQ for
all active qubits in the same two programs. It does not simulate each qubit
separately or assemble unrelated shots. This core has no Form, catalog entry,
recipe or calibration proposals.

`MuxGECfg.modules` contains the common optional reset and the complete ordered
mux readout. `tones: list[MuxGEToneCfg]` has one element per full readout tone.
Each element contains `active`, the probe `drive`, and optional `init_pulse`.
Only active elements require a drive. Active generators must be independent and
must differ from the readout generator. An initial pulse uses its tone's drive
generator. Inactive settings stay in cfg but do not run.

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
